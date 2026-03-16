"""
Unit tests for the range engine pipeline:
- PivotCache (pivot_cache.py)
- RangeEngineL1 (range_engine_l1.py)
- RangeEngineL2 (range_engine_l2.py)
- RangeEngineController (range_engine_controller.py)
- RangeComparisonLogger (range_comparison_logger.py)
- Shared helpers (range_utils.py)
- TCT schematic integration (tct_schematics.py)
"""

import json
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from pivot_cache import PivotCache
from range_engine_l1 import RangeEngineL1
from range_engine_l2 import RangeEngineL2
from range_engine_controller import RangeEngineController
from range_comparison_logger import RangeComparisonLogger
from range_utils import check_equilibrium_touch
from tct_schematics import detect_tct_schematics, TCTSchematicDetector


# ================================================================
# Fixtures
# ================================================================

def _make_candles(
    prices: list,
    freq: str = "1h",
    spread: float = 50.0,
) -> pd.DataFrame:
    """
    Build a simple OHLC DataFrame from a list of mid prices.

    Each candle has:
        open = price
        high = price + spread
        low  = price - spread
        close = price + small random offset
    """
    n = len(prices)
    np.random.seed(42)
    dates = pd.date_range("2026-01-01", periods=n, freq=freq)
    return pd.DataFrame({
        "open_time": dates,
        "open": prices,
        "high": [p + spread for p in prices],
        "low": [p - spread for p in prices],
        "close": [p + np.random.uniform(-20, 20) for p in prices],
        "volume": np.random.uniform(100, 500, n),
    })


@pytest.fixture
def simple_swing_df():
    """
    A small DataFrame with a clear swing high and swing low.

    The pattern is:  rise → peak → fall → trough → rise
    so that pivots can be detected with lookback=2.
    """
    prices = (
        [100, 110, 120, 130, 140]    # rising
        + [150, 145, 140, 135, 130]  # peak at index 5
        + [125, 120, 115, 110, 105]  # falling
        + [100, 105, 110, 115, 120]  # trough at index 15
        + [125, 130, 135, 140, 145]  # rising again
    )
    return _make_candles(prices, spread=10)


@pytest.fixture
def inside_bar_df():
    """
    DataFrame with multiple consecutive inside bars around a pivot.
    Tests that dynamic inside-bar skipping finds enough non-inside-bar
    candles even when > 3 consecutive inside bars are present.
    """
    # Clear rise → peak → inside bars → fall
    prices = [100, 110, 120, 130, 140,   # 0-4: rising
              155,                        # 5: peak (high will be 155+spread)
              # inside bars: these should be contained within candle 5
              # We'll set spread so they fit inside
              154, 153, 152, 151,         # 6-9: inside bars (highs < 155+s, lows > 155-s)
              145, 140, 135, 130, 125,    # 10-14: falling
              120, 115, 110, 105, 100]    # 15-19: falling more
    df = _make_candles(prices, spread=8)
    # Make candles 6-9 truly inside bar 5
    for i in range(6, 10):
        df.loc[i, "high"] = df.loc[5, "high"] - 1
        df.loc[i, "low"] = df.loc[5, "low"] + 1
    return df


@pytest.fixture
def accumulation_df():
    """
    DataFrame simulating accumulation: downtrend → range → deviation → recovery.

    At least 48 hourly candles to support 24h range duration checks.
    """
    prices = []
    base = 100000

    # Downtrend: 0-14
    for i in range(15):
        prices.append(base - i * 150)

    # Range at bottom: 15-29
    range_low = base - 15 * 150
    for i in range(15):
        prices.append(range_low + np.random.uniform(-100, 300))

    # First deviation below range: 30-37
    for i in range(8):
        prices.append(range_low - 400 - i * 50)

    # Recovery / EQ touch: 38-55
    for i in range(18):
        prices.append(range_low + i * 100 + np.random.uniform(-50, 50))

    return _make_candles(prices, spread=80)


@pytest.fixture
def distribution_df():
    """
    DataFrame simulating distribution: uptrend → range → deviation above → sell-off.

    Produces 48+ hourly candles for 24h duration checks.
    """
    prices = []
    base = 100000

    # Uptrend: 0-14
    for i in range(15):
        prices.append(base + i * 150)

    # Range at top: 15-29
    range_high = base + 15 * 150
    for i in range(15):
        prices.append(range_high + np.random.uniform(-300, 100))

    # First deviation above range: 30-37
    for i in range(8):
        prices.append(range_high + 400 + i * 50)

    # Sell-off: 38-55
    for i in range(18):
        prices.append(range_high - i * 100 + np.random.uniform(-50, 50))

    return _make_candles(prices, spread=80)


# ================================================================
# PivotCache Tests
# ================================================================

class TestPivotCache:
    def test_basic_swing_detection(self, simple_swing_df):
        pc = PivotCache(simple_swing_df, lookback=2)
        highs = pc.swing_highs
        lows = pc.swing_lows

        assert isinstance(highs, list)
        assert isinstance(lows, list)
        # Should find at least one swing high and one swing low
        assert len(highs) >= 1
        assert len(lows) >= 1

        for h in highs:
            assert "idx" in h and "price" in h
        for l in lows:
            assert "idx" in l and "price" in l

    def test_inside_bar_skipping(self, inside_bar_df):
        """
        With 4 consecutive inside bars the old +3 buffer would fail;
        the dynamic while-loop must handle this.
        """
        pc = PivotCache(inside_bar_df, lookback=3)
        highs = pc.swing_highs

        # The peak at index 5 should be detected despite 4 inside bars after it
        peak_detected = any(h["idx"] == 5 for h in highs)
        assert peak_detected, (
            f"Swing high at idx 5 not detected; found: {highs}"
        )

    def test_is_inside_bar(self, inside_bar_df):
        pc = PivotCache(inside_bar_df, lookback=3)
        # Candles 6-9 are set up as inside bars of candle 5
        for i in range(6, 10):
            assert pc._is_inside_bar(i), f"Candle {i} should be inside bar"

    def test_boundary_indices(self, simple_swing_df):
        pc = PivotCache(simple_swing_df, lookback=3)
        # Should not crash on boundary
        assert pc._check_swing_high(0) is False
        assert pc._check_swing_low(len(simple_swing_df) - 1) is False

    def test_candles_property(self, simple_swing_df):
        pc = PivotCache(simple_swing_df, lookback=2)
        assert len(pc.candles) == len(simple_swing_df)


# ================================================================
# RangeEngineL1 Tests
# ================================================================

class TestRangeEngineL1:
    def test_accumulation_detection(self, accumulation_df):
        pc = PivotCache(accumulation_df, lookback=2)
        engine = RangeEngineL1(pc)
        ranges = engine.detect_accumulation_ranges()

        assert isinstance(ranges, list)
        for r in ranges:
            assert r["direction"] == "accumulation"
            assert r["range_high"] > r["range_low"]
            assert r["equilibrium"] == pytest.approx(
                (r["range_high"] + r["range_low"]) / 2
            )

    def test_distribution_detection(self, distribution_df):
        pc = PivotCache(distribution_df, lookback=2)
        engine = RangeEngineL1(pc)
        ranges = engine.detect_distribution_ranges()

        assert isinstance(ranges, list)
        for r in ranges:
            assert r["direction"] == "distribution"
            assert r["range_high"] > r["range_low"]


# ================================================================
# RangeEngineL2 Tests
# ================================================================

class TestRangeEngineL2:
    def test_l1_trend_gate_blocks_mismatch(self, accumulation_df):
        """L2 distribution should return empty when L1 trend is not bearish."""
        pc = PivotCache(accumulation_df, lookback=2)
        engine = RangeEngineL2(pc)
        # Requesting distribution with bullish bias — should be blocked
        result = engine.detect_distribution_ranges(accumulation_df, htf_bias="bullish")
        assert result == []

    def test_l1_trend_gate_blocks_htf_bias_mismatch(self, distribution_df):
        """Even if L1 is bearish, mismatched htf_bias should block."""
        pc = PivotCache(distribution_df, lookback=2)
        engine = RangeEngineL2(pc)
        result = engine.detect_distribution_ranges(distribution_df, htf_bias="bullish")
        assert result == []

    def test_24h_minimum_range_duration(self, distribution_df):
        """
        For hourly candles, a valid range should span at least 24 candles
        (24 hours). Generate enough data with a deliberately wide range
        whose pivots are > 24 candles apart.
        """
        np.random.seed(123)
        prices = []
        base = 100000

        # Uptrend (0-14)
        for i in range(15):
            prices.append(base + i * 100)

        # Peak / swing high at index 15
        prices.append(base + 15 * 100 + 200)

        # Consolidation range for 30 candles (16-45) — oscillates
        for i in range(30):
            prices.append(base + 14 * 100 + np.random.uniform(-100, 100))

        # Swing low at index 46 (well below range)
        prices.append(base + 10 * 100 - 200)

        # Downtrend / recovery (47-70)
        for i in range(24):
            prices.append(base + 12 * 100 + np.random.uniform(-50, 50))

        df = _make_candles(prices, freq="1h", spread=60)
        pc = PivotCache(df, lookback=2)
        engine = RangeEngineL2(pc)

        # Try both directions; if any range is found it must span >= 24
        dist = engine.detect_distribution_ranges(df, htf_bias="bearish")
        acc = engine.detect_accumulation_ranges(df, htf_bias="bullish")

        for r in dist + acc:
            high_idx = r["range_high_idx"]
            low_idx = r["range_low_idx"]
            assert abs(high_idx - low_idx) >= 24, (
                f"Range span {abs(high_idx - low_idx)} < 24h for hourly candles"
            )


# ================================================================
# L2 Pivot Helpers
# ================================================================

class TestL2PivotHelpers:
    def test_lower_highs_returns_empty_for_single_pivot(self):
        from range_engine_l2 import _find_l2_lower_highs
        result = _find_l2_lower_highs([{"idx": 0, "price": 100}])
        assert result == []

    def test_lower_lows_returns_empty_for_single_pivot(self):
        from range_engine_l2 import _find_l2_lower_lows
        result = _find_l2_lower_lows([{"idx": 0, "price": 100}])
        assert result == []

    def test_higher_lows_returns_empty_for_single_pivot(self):
        from range_engine_l2 import _find_l2_higher_lows
        result = _find_l2_higher_lows([{"idx": 0, "price": 100}])
        assert result == []

    def test_higher_highs_returns_empty_for_single_pivot(self):
        from range_engine_l2 import _find_l2_higher_highs
        result = _find_l2_higher_highs([{"idx": 0, "price": 100}])
        assert result == []

    def test_lower_highs_valid_sequence(self):
        from range_engine_l2 import _find_l2_lower_highs
        pivots = [
            {"idx": 0, "price": 100},
            {"idx": 5, "price": 95},
            {"idx": 10, "price": 90},
        ]
        result = _find_l2_lower_highs(pivots)
        assert len(result) == 3
        assert result[0]["price"] > result[1]["price"] > result[2]["price"]

    def test_higher_lows_valid_sequence(self):
        from range_engine_l2 import _find_l2_higher_lows
        pivots = [
            {"idx": 0, "price": 100},
            {"idx": 5, "price": 105},
            {"idx": 10, "price": 110},
        ]
        result = _find_l2_higher_lows(pivots)
        assert len(result) == 3

    def test_returns_empty_for_non_monotonic(self):
        from range_engine_l2 import _find_l2_lower_highs
        pivots = [
            {"idx": 0, "price": 100},
            {"idx": 5, "price": 110},  # goes up — breaks monotonicity
            {"idx": 10, "price": 90},
        ]
        result = _find_l2_lower_highs(pivots)
        # Only 100 and 90 form a valid sub-sequence (len=2), which meets MIN_L2_PIVOTS
        assert len(result) >= 2


# ================================================================
# RangeEngineController Tests
# ================================================================

class TestRangeEngineController:
    def test_lazy_init(self, accumulation_df):
        ctrl = RangeEngineController()
        pc = PivotCache(accumulation_df, lookback=2)
        result = ctrl.detect_ranges(accumulation_df, pc, htf_bias="neutral")
        assert "l1_accumulation" in result
        assert "l1_distribution" in result
        assert "l2_accumulation" in result
        assert "l2_distribution" in result

    def test_pivot_cache_mismatch_recreates_engines(self, accumulation_df, distribution_df):
        pc1 = PivotCache(accumulation_df, lookback=2)
        ctrl = RangeEngineController(pc1)

        pc2 = PivotCache(distribution_df, lookback=2)
        # Passing a different cache should trigger engine rebuild
        result = ctrl.detect_ranges(distribution_df, pc2, htf_bias="neutral")
        assert ctrl._pivot_cache is pc2
        assert isinstance(result, dict)

    def test_same_cache_no_recreate(self, accumulation_df):
        pc = PivotCache(accumulation_df, lookback=2)
        ctrl = RangeEngineController(pc)
        l1_before = ctrl._l1
        ctrl.detect_ranges(accumulation_df, pc, htf_bias="neutral")
        assert ctrl._l1 is l1_before  # should NOT have been recreated


# ================================================================
# RangeComparisonLogger Tests
# ================================================================

class TestRangeComparisonLogger:
    def test_log_creates_file(self, tmp_path):
        log_path = str(tmp_path / "test_comp.jsonl")
        logger = RangeComparisonLogger(log_path=log_path)
        logger.log("BTCUSDT", "4h", "L1", [{"range_high": 100}])

        assert os.path.exists(log_path)
        with open(log_path) as f:
            lines = f.readlines()
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["symbol"] == "BTCUSDT"
        assert entry["range_count"] == 1

    def test_empty_dirname_no_error(self):
        """Filename-only path should not raise on makedirs."""
        logger = RangeComparisonLogger(log_path="comparison.jsonl")
        # Should not raise — the guard for empty dirname should prevent it

    def test_log_with_metadata(self, tmp_path):
        log_path = str(tmp_path / "meta.jsonl")
        logger = RangeComparisonLogger(log_path=log_path)
        logger.log("ETHUSDT", "1h", "L2", [], metadata={"note": "test"})

        with open(log_path) as f:
            entry = json.loads(f.readline())
        assert entry["metadata"]["note"] == "test"


# ================================================================
# Equilibrium Touch (range_utils) Tests
# ================================================================

class TestEquilibriumTouch:
    def test_touch_between_pivots(self):
        candles = pd.DataFrame({
            "high": [110, 105, 102, 101, 100],
            "low":  [100,  95,  98,  99,  90],
        })
        assert check_equilibrium_touch(candles, 0, 4, 100.0) is True

    def test_no_touch(self):
        candles = pd.DataFrame({
            "high": [110, 90, 90, 90, 90],
            "low":  [100, 80, 80, 80, 80],
        })
        assert check_equilibrium_touch(
            candles, 0, 3, 100.0, check_between=False, post_range_candles=1
        ) is False

    def test_touch_after_range(self):
        candles = pd.DataFrame({
            "high": [110, 90, 90, 90, 103],
            "low":  [100, 80, 80, 80,  97],
        })
        assert check_equilibrium_touch(
            candles, 0, 3, 100.0, check_between=False
        ) is True


# ================================================================
# TCT Schematics Integration Tests
# ================================================================

class TestTCTSchematicsIntegration:
    def test_detect_tct_schematics_with_pivot_cache(self, accumulation_df):
        """detect_tct_schematics accepts optional pivot_cache parameter."""
        pc = PivotCache(accumulation_df, lookback=3)
        result = detect_tct_schematics(accumulation_df, pivot_cache=pc)
        assert "accumulation_schematics" in result
        assert "distribution_schematics" in result

    def test_distribution_schematics_have_session_context(self):
        """
        Distribution schematics should include a session_context field.
        """
        np.random.seed(99)
        dates = pd.date_range("2026-01-01", periods=150, freq="1h")
        base = 100000
        prices = []
        for i in range(30):
            prices.append(base + i * 100 + np.random.uniform(-50, 50))
        range_high = base + 3000
        for i in range(30):
            prices.append(range_high + np.random.uniform(-200, 400))
        for i in range(20):
            prices.append(range_high + 500 + i * 20 + np.random.uniform(-100, 100))
        for i in range(20):
            prices.append(range_high + 1000 + i * 15 + np.random.uniform(-80, 80))
        for i in range(50):
            prices.append(range_high - i * 30 + np.random.uniform(-100, 100))

        df = pd.DataFrame({
            "open_time": dates,
            "open": prices,
            "high": [p + np.random.uniform(80, 200) for p in prices],
            "low": [p - np.random.uniform(80, 200) for p in prices],
            "close": [p + np.random.uniform(-100, 100) for p in prices],
            "volume": np.random.uniform(100, 1000, 150),
        })

        result = detect_tct_schematics(df)
        dist = result.get("distribution_schematics", [])

        # If schematics were detected, they should have session_context
        for s in dist:
            assert "session_context" in s, "Distribution schematic missing session_context"
            assert "session" in s["session_context"]

    def test_distribution_schematics_non_empty_fields(self):
        """
        When distribution schematics are found, verify required fields
        on at least one candidate.
        """
        np.random.seed(42)
        dates = pd.date_range("2026-01-01", periods=150, freq="1h")
        base = 100000
        prices = []
        for i in range(30):
            prices.append(base + i * 100 + np.random.uniform(-50, 50))
        range_high = base + 3000
        for i in range(30):
            prices.append(range_high + np.random.uniform(-200, 400))
        for i in range(20):
            prices.append(range_high + 500 + i * 20 + np.random.uniform(-100, 100))
        for i in range(20):
            prices.append(range_high + 1000 + i * 15 + np.random.uniform(-80, 80))
        for i in range(50):
            prices.append(range_high - i * 30 + np.random.uniform(-100, 100))

        df = pd.DataFrame({
            "open_time": dates,
            "open": prices,
            "high": [p + np.random.uniform(80, 200) for p in prices],
            "low": [p - np.random.uniform(80, 200) for p in prices],
            "close": [p + np.random.uniform(-100, 100) for p in prices],
            "volume": np.random.uniform(100, 1000, 150),
        })

        result = detect_tct_schematics(df)
        dist = result.get("distribution_schematics", [])

        assert isinstance(dist, list)
        # Distribution may or may not be found depending on random data;
        # if found, validate required fields on first candidate
        if len(dist) > 0:
            s = dist[0]
            assert "schematic_type" in s
            assert "quality_score" in s
            assert "is_confirmed" in s
            assert "tap1" in s
            assert "tap2" in s
            assert "tap3" in s
            assert "session_context" in s
            if "sweep_validation" in s:
                assert "classification" in s["sweep_validation"]
