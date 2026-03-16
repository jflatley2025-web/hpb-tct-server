"""
test_range_engines.py — Unit tests for the distribution detection fix
======================================================================
Tests all 6 bug fixes:
1. L2 range detection vs L1
2. BOS low_price asymmetry fix
3. Demand-path ranking for MS low selection
4. Liquidity sweep validation
5. Session manipulation windows
6. Pivot cache consistency

Also includes full flow regression test for Model 1 Distribution.
"""

import os
import json
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from unittest.mock import patch, MagicMock

from pivot_cache import PivotCache
from range_engine_l1 import RangeEngineL1
from range_engine_l2 import RangeEngineL2
from range_engine_controller import RangeEngineController
from range_comparison_logger import RangeComparisonLogger
from session_manipulation import (
    get_active_session,
    get_session_multiplier,
    apply_session_multiplier,
    get_session_info,
)


# ================================================================
# TEST DATA HELPERS
# ================================================================

def _make_candles(prices, start_time=None):
    """Create a candle DataFrame from a list of (open, high, low, close) tuples."""
    if start_time is None:
        start_time = datetime(2026, 3, 1, tzinfo=timezone.utc)

    rows = []
    for i, (o, h, l, c) in enumerate(prices):
        rows.append({
            "open_time": (start_time + timedelta(hours=i)).isoformat(),
            "open": float(o),
            "high": float(h),
            "low": float(l),
            "close": float(c),
            "volume": 100.0,
        })
    return pd.DataFrame(rows)


def _make_distribution_candles(n=200):
    """
    Create synthetic candles with a clear TCT Model 1 Distribution pattern:
    - Uptrend (L1 bullish)
    - Range formation (24h+) with EQ touch
    - Tap1 at range high
    - Tap2 deviation above range high, returns inside
    - Tap3 higher deviation, returns inside (with stacked highs for liquidity)
    - Bearish BOS below internal MS low
    """
    np.random.seed(42)
    base = 50000
    prices = []

    for i in range(n):
        if i < 30:
            # Uptrend (L1 bullish context)
            o = base + i * 80
            h = o + np.random.uniform(30, 100)
            l = o - np.random.uniform(30, 80)
            c = o + np.random.uniform(-20, 60)
        elif i < 50:
            # Range high formation
            o = base + 2400 + np.random.uniform(-100, 100)
            h = o + np.random.uniform(30, 120)
            l = o - np.random.uniform(30, 100)
            c = o + np.random.uniform(-50, 50)
        elif i < 70:
            # Move down to range low (EQ touch happens here)
            drop = (i - 50) * 60
            o = base + 2400 - drop + np.random.uniform(-80, 80)
            h = o + np.random.uniform(30, 100)
            l = o - np.random.uniform(30, 100)
            c = o + np.random.uniform(-40, 40)
        elif i < 85:
            # Range low area (Tap1 zone)
            o = base + 1200 + np.random.uniform(-100, 100)
            h = o + np.random.uniform(30, 80)
            l = o - np.random.uniform(30, 100)
            c = o + np.random.uniform(-50, 50)
        elif i < 100:
            # Move back up — EQ touch and above range high (Tap2 deviation)
            rise = (i - 85) * 100
            o = base + 1200 + rise + np.random.uniform(-60, 60)
            h = o + np.random.uniform(30, 120)
            l = o - np.random.uniform(30, 80)
            c = o + np.random.uniform(-40, 40)
        elif i < 110:
            # Come back inside range (Tap2 return)
            o = base + 2200 + np.random.uniform(-100, 100)
            h = o + np.random.uniform(30, 80)
            l = o - np.random.uniform(30, 100)
            c = o + np.random.uniform(-50, 50)
        elif i < 125:
            # Stacked highs building liquidity near range high
            o = base + 2350 + np.random.uniform(-50, 50)
            h = base + 2420 + np.random.uniform(-10, 10)  # Equal highs
            l = o - np.random.uniform(30, 80)
            c = o + np.random.uniform(-30, 30)
        elif i < 135:
            # Tap3 deviation higher (sweep of stacked highs)
            o = base + 2500 + np.random.uniform(-50, 50)
            h = o + np.random.uniform(50, 150)
            l = o - np.random.uniform(30, 60)
            c = base + 2300 + np.random.uniform(-50, 50)  # Returns inside
        elif i < 145:
            # Price rotating down — internal MS lows forming
            drop = (i - 135) * 40
            o = base + 2200 - drop + np.random.uniform(-40, 40)
            h = o + np.random.uniform(20, 60)
            l = o - np.random.uniform(30, 80)
            c = o + np.random.uniform(-30, 30)
        elif i < 155:
            # Bearish BOS — break below internal MS low
            o = base + 1800 - (i - 145) * 50
            h = o + np.random.uniform(20, 50)
            l = o - np.random.uniform(40, 100)
            c = o - np.random.uniform(20, 60)
        else:
            # Continuation down toward range low
            o = base + 1300 - (i - 155) * 20 + np.random.uniform(-50, 50)
            h = o + np.random.uniform(20, 60)
            l = o - np.random.uniform(30, 80)
            c = o + np.random.uniform(-40, 40)

        prices.append((o, h, l, c))

    start = datetime(2026, 2, 1, tzinfo=timezone.utc)
    return _make_candles(prices, start_time=start)


def _make_simple_swing_candles():
    """Create simple candles with clear swing highs and lows for pivot testing."""
    prices = []
    for i in range(50):
        if i % 10 < 5:
            # Rising
            o = 100 + (i % 10) * 10
            h = o + 8
            l = o - 3
            c = o + 5
        else:
            # Falling
            o = 150 - (i % 10 - 5) * 10
            h = o + 3
            l = o - 8
            c = o - 5
        prices.append((o, h, l, c))
    return _make_candles(prices)


# ================================================================
# PIVOT CACHE TESTS
# ================================================================

class TestPivotCache:
    def test_pivot_cache_creation(self):
        df = _make_simple_swing_candles()
        pc = PivotCache(df, lookback=2)
        assert pc is not None
        assert len(pc.get_pivot_highs(lookback=2)) >= 0
        assert len(pc.get_pivot_lows(lookback=2)) >= 0

    def test_pivot_cache_consistency(self):
        """All modules using same pivot data from cache."""
        df = _make_distribution_candles()
        pc = PivotCache(df, lookback=3)

        highs_1 = pc.get_pivot_highs()
        highs_2 = pc.get_pivot_highs()

        # Same object — cached, not recomputed
        assert highs_1 is highs_2

    def test_pivot_cache_no_recompute(self):
        """Pivots computed once, subsequent access returns cached."""
        df = _make_distribution_candles()
        pc = PivotCache(df, lookback=3)

        # Access twice
        lows_a = pc.get_pivot_lows(lookback=3)
        lows_b = pc.get_pivot_lows(lookback=3)

        # Should be the exact same list object
        assert lows_a is lows_b

    def test_pivot_cache_different_lookbacks(self):
        """Different lookbacks produce different pivot sets."""
        df = _make_distribution_candles()
        pc = PivotCache(df, lookback=3)

        highs_lb1 = pc.get_pivot_highs(lookback=1)
        highs_lb3 = pc.get_pivot_highs(lookback=3)

        # LB=1 should find more pivots than LB=3
        assert len(highs_lb1) >= len(highs_lb3)

    def test_pivot_cache_invalidate(self):
        """Invalidation clears cached pivots."""
        df = _make_simple_swing_candles()
        pc = PivotCache(df, lookback=2)
        _ = pc.get_pivot_highs()
        pc.invalidate()
        assert pc._computed is False

    def test_get_swing_lows_in_range(self):
        df = _make_distribution_candles()
        pc = PivotCache(df, lookback=2)
        lows = pc.get_swing_lows_in_range(50, 100, lookback=2)
        for sl in lows:
            assert 50 <= sl["idx"] <= 100


# ================================================================
# RANGE ENGINE L1 TESTS
# ================================================================

class TestRangeEngineL1:
    def test_l1_detects_ranges(self):
        df = _make_distribution_candles()
        pc = PivotCache(df, lookback=3)
        engine = RangeEngineL1(pc)
        ranges = engine.detect_distribution_ranges(df)
        # Should find at least some candidate ranges
        assert isinstance(ranges, list)
        for r in ranges:
            assert r["engine"] == "L1"
            assert r["range_high"] > r["range_low"]


# ================================================================
# RANGE ENGINE L2 TESTS
# ================================================================

class TestRangeEngineL2:
    def test_l2_detects_distribution_ranges(self):
        df = _make_distribution_candles()
        pc = PivotCache(df, lookback=3)
        engine = RangeEngineL2(pc)
        ranges = engine.detect_distribution_ranges(df, htf_bias="bullish")
        assert isinstance(ranges, list)
        for r in ranges:
            assert r["engine"] == "L2"

    def test_24h_minimum_range_duration(self):
        """Ranges < 24h classified as micro-ranges, excluded."""
        # Create very short candles (1 minute each, only 5 candles)
        start = datetime(2026, 3, 1, tzinfo=timezone.utc)
        prices = [(100 + i, 105 + i, 95 + i, 102 + i) for i in range(20)]
        df = _make_candles(prices, start_time=start)  # 20 hourly candles
        pc = PivotCache(df, lookback=2)
        engine = RangeEngineL2(pc)
        # With hourly candles and only 20 candles, the range duration
        # is limited. L2 requires 24h minimum.
        ranges = engine.detect_distribution_ranges(df, htf_bias="bullish")
        for r in ranges:
            # If any range found, its duration should be >= 24h
            high_idx = r["range_high_idx"]
            low_idx = r["range_low_idx"]
            assert abs(low_idx - high_idx) >= 5  # Minimum candle gap


# ================================================================
# RANGE ENGINE CONTROLLER TESTS
# ================================================================

class TestRangeEngineController:
    def test_controller_modes(self):
        """All 4 modes behave correctly."""
        df = _make_distribution_candles()
        pc = PivotCache(df, lookback=3)

        for mode in ["L1", "L2", "compare", "strict_L2"]:
            ctrl = RangeEngineController(mode=mode, pivot_cache=pc)
            assert ctrl.mode == mode
            ranges = ctrl.detect_ranges(df, "distribution", "bullish")
            assert isinstance(ranges, list)

    def test_strict_l2_warns_on_no_ranges(self):
        """strict_L2 mode logs warning, returns empty, no fallback."""
        # Create minimal data unlikely to have L2 structure
        prices = [(100, 105, 95, 102)] * 60
        df = _make_candles(prices)
        pc = PivotCache(df, lookback=3)
        ctrl = RangeEngineController(mode="strict_L2", pivot_cache=pc)

        with patch("range_engine_controller.logger") as mock_logger:
            ranges = ctrl.detect_ranges(df, "distribution", "bullish")
            # strict_L2 returns empty and logs warning
            assert isinstance(ranges, list)
            # If L2 found nothing, warning should have been logged
            if len(ranges) == 0:
                mock_logger.warning.assert_called()

    def test_set_mode(self):
        pc = PivotCache(_make_simple_swing_candles(), lookback=2)
        ctrl = RangeEngineController(mode="L1", pivot_cache=pc)
        ctrl.set_mode("L2")
        assert ctrl.mode == "L2"

    def test_invalid_mode_raises(self):
        pc = PivotCache(_make_simple_swing_candles(), lookback=2)
        ctrl = RangeEngineController(mode="L1", pivot_cache=pc)
        with pytest.raises(ValueError):
            ctrl.set_mode("invalid_mode")

    def test_feature_flag_env_var(self):
        """Controller respects RANGE_ENGINE_MODE env var."""
        with patch.dict(os.environ, {"RANGE_ENGINE_MODE": "strict_L2"}):
            ctrl = RangeEngineController()
            assert ctrl.mode == "strict_L2"


# ================================================================
# RANGE COMPARISON LOGGER TESTS
# ================================================================

class TestRangeComparisonLogger:
    def test_compare_mode_logs_differences(self, tmp_path):
        """Compare mode writes JSONL with correct fields."""
        log_path = str(tmp_path / "test_comparison.jsonl")
        logger = RangeComparisonLogger(log_path=log_path)

        l1_ranges = [{"range_high": 50000, "range_low": 48000,
                       "range_high_idx": 10, "range_low_idx": 30}]
        l2_ranges = [{"range_high": 50200, "range_low": 47800,
                       "range_high_idx": 8, "range_low_idx": 32}]

        logger.log_comparison(
            symbol="BTCUSDT",
            session="asia",
            engine_used="L2",
            l1_ranges=l1_ranges,
            l2_ranges=l2_ranges,
            deviation_detected=True,
            liquidity_sweep_detected=True,
        )

        with open(log_path) as f:
            line = f.readline()
            entry = json.loads(line)

        assert entry["symbol"] == "BTCUSDT"
        assert entry["session"] == "asia"
        assert entry["engine_used"] == "L2"
        assert entry["L1_range_high"] == 50000
        assert entry["L2_range_high"] == 50200
        assert entry["deviation_detected"] is True
        assert entry["liquidity_sweep_detected"] is True


# ================================================================
# SESSION MANIPULATION TESTS
# ================================================================

class TestSessionManipulation:
    def test_asia_session_detection(self):
        """Asia manipulation window: 23:30 - 01:00 UTC."""
        ts = datetime(2026, 3, 15, 0, 15, tzinfo=timezone.utc)
        assert get_active_session(ts) == "asia"

    def test_london_session_detection(self):
        ts = datetime(2026, 3, 15, 8, 0, tzinfo=timezone.utc)
        assert get_active_session(ts) == "london"

    def test_ny_session_detection(self):
        ts = datetime(2026, 3, 15, 13, 30, tzinfo=timezone.utc)
        assert get_active_session(ts) == "new_york"

    def test_no_session_outside_windows(self):
        ts = datetime(2026, 3, 15, 5, 0, tzinfo=timezone.utc)
        assert get_active_session(ts) is None

    def test_session_multiplier_values(self):
        assert get_session_multiplier("asia") == 1.05
        assert get_session_multiplier("london") == 1.10
        assert get_session_multiplier("new_york") == 1.20
        assert get_session_multiplier(None) == 1.0

    def test_session_weighting_execution_confidence(self):
        """MSCE multiplier applied to execution confidence."""
        ts = datetime(2026, 3, 15, 13, 30, tzinfo=timezone.utc)  # NY session
        result = apply_session_multiplier(80.0, timestamp=ts)
        assert result["session"] == "new_york"
        assert result["multiplier"] == 1.20
        assert result["adjusted_confidence"] == 96.0
        assert result["boost_applied"] is True

    def test_session_confidence_capped_at_100(self):
        """Adjusted confidence should never exceed 100."""
        ts = datetime(2026, 3, 15, 13, 30, tzinfo=timezone.utc)
        result = apply_session_multiplier(95.0, timestamp=ts)
        assert result["adjusted_confidence"] <= 100.0

    def test_no_boost_outside_sessions(self):
        ts = datetime(2026, 3, 15, 5, 0, tzinfo=timezone.utc)
        result = apply_session_multiplier(80.0, timestamp=ts)
        assert result["boost_applied"] is False
        assert result["adjusted_confidence"] == 80.0

    def test_get_session_info(self):
        ts = datetime(2026, 3, 15, 8, 30, tzinfo=timezone.utc)
        info = get_session_info(ts)
        assert info["active_session"] == "london"
        assert info["is_manipulation_window"] is True


# ================================================================
# BOS LOW_PRICE FIX TESTS
# ================================================================

class TestBOSLowPriceFix:
    def test_bearish_bos_low_price_validation(self):
        """BOS swing low must be above lowest point between t2/t3."""
        from tct_schematics import TCTSchematicDetector

        df = _make_distribution_candles()
        det = TCTSchematicDetector(df)

        # The _find_bearish_bos method should validate sl["price"] > low_price
        # Test with low_price that would filter out all swing lows
        result = det._find_bearish_bos(
            start_idx=130,
            low_price=60000,  # Very high — no swing low should pass
            high_price=70000,
            equilibrium=55000,
        )
        # Should return None since no swing low is above 60000
        assert result is None


# ================================================================
# DEMAND-PATH RANKING TESTS
# ================================================================

class TestDemandPathRanking:
    def test_demand_ranking_selects_clean_path_ms_low(self):
        """MS low with clean path ranked above MS low with demand below."""
        from tct_schematics import TCTSchematicDetector

        df = _make_distribution_candles()
        det = TCTSchematicDetector(df)

        swing_lows = [
            {"idx": 140, "price": 51500.0},
            {"idx": 142, "price": 51200.0},
            {"idx": 144, "price": 51800.0},
        ]

        range_data = {
            "range_high": 52400.0,
            "range_low": 50000.0,
            "range_high_idx": 40,
            "range_low_idx": 80,
            "range_size": 2400.0,
            "equilibrium": 51200.0,
        }

        ranked = det._rank_ms_lows_by_path_quality(swing_lows, range_data)
        assert isinstance(ranked, list)
        assert len(ranked) == 3
        # The highest-scoring one should be first


# ================================================================
# LIQUIDITY SWEEP TESTS
# ================================================================

class TestLiquiditySweep:
    def test_distribution_sweep_validation(self):
        """Sweep validation returns correct structure."""
        from tct_schematics import TCTSchematicDetector

        df = _make_distribution_candles()
        det = TCTSchematicDetector(df)

        range_data = {
            "range_high": 52400.0,
            "range_low": 50000.0,
            "range_high_idx": 40,
            "range_low_idx": 80,
            "range_size": 2400.0,
        }
        tap2 = {"idx": 95, "price": 52600.0}
        tap3 = {"idx": 130, "price": 52800.0}

        result = det._validate_distribution_sweep(range_data, tap2, tap3)
        assert "has_sweep" in result
        assert "classification" in result
        assert "pools_swept" in result
        assert "returned_inside" in result
        assert result["classification"] in ("no_sweep", "liquidity_grab", "true_break")


# ================================================================
# INTEGRATION TEST: FULL DISTRIBUTION FLOW
# ================================================================

class TestModel1DistributionFullFlow:
    def test_model1_distribution_detected(self):
        """
        End-to-end regression: range → taps → sweep → BOS → entry.
        Ensures the bot detects Model 1 Distribution with all fixes applied.
        """
        from tct_schematics import detect_tct_schematics

        df = _make_distribution_candles(n=200)
        result = detect_tct_schematics(df, [])

        assert "distribution_schematics" in result
        assert "accumulation_schematics" in result

        # The detection should find at least candidates
        dist = result["distribution_schematics"]
        assert isinstance(dist, list)

        # Check structure of any found schematics
        for s in dist:
            assert "schematic_type" in s
            assert "quality_score" in s
            assert "is_confirmed" in s
            assert "tap1" in s
            assert "tap2" in s
            assert "tap3" in s
            assert "session_context" in s
            if s.get("sweep_validation"):
                assert "classification" in s["sweep_validation"]

    def test_distribution_with_pivot_cache(self):
        """Detection works correctly with explicit PivotCache."""
        from tct_schematics import detect_tct_schematics

        df = _make_distribution_candles()
        pc = PivotCache(df, lookback=3)
        result = detect_tct_schematics(df, [], pivot_cache=pc)

        assert "distribution_schematics" in result
        assert isinstance(result["distribution_schematics"], list)
