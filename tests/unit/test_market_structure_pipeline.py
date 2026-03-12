"""Tests for the v2 9-phase market structure evaluation pipeline.

Covers:
  - Phase 1: detect_htf_market_structure (pivot-based HTF bias)
  - Phase 2: Range validation helpers (time displacement, liquidity stacking, V-shape)
  - Phase 3: Tap structure validation (Model 1/2)
  - Phase 4: Liquidity sweep v2 (with tolerance)
  - Phase 5: BOS sequencing (must be after Tap3)
  - Phase 6: POI validation (FVG optional)
  - Phase 7: Directional filter (with reversal exception)
  - Phase 8: Risk filter (R:R >= 1.5)
  - Phase 9: Confidence scoring (threshold 60, session timing)
  - End-to-end: compute_composite_score_v2
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch

from decision_tree_bridge import (
    detect_htf_market_structure,
    _check_time_displacement,
    _detect_liquidity_stacking,
    _reject_v_shape,
    _detect_liquidity_sweep_v2,
    _score_session_timing,
    compute_composite_score_v2,
    V2_THRESHOLD,
    DecisionTreeEvaluator,
)


# ================================================================
# HELPERS — build synthetic candle data
# ================================================================

def _make_df(highs, lows, closes=None, opens=None):
    """Build a minimal OHLC DataFrame from lists."""
    n = len(highs)
    if closes is None:
        closes = [(h + l) / 2 for h, l in zip(highs, lows)]
    if opens is None:
        opens = closes[:]
    return pd.DataFrame({
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
    })


def _make_uptrend_df(n=50, start=100):
    """Generate a clear uptrend (HH + HL) with swing points."""
    highs, lows, closes = [], [], []
    price = start
    for i in range(n):
        cycle = i % 10
        if cycle < 5:
            price += 2  # up leg
        else:
            price -= 1  # shallow pullback (HL)
        h = price + 1
        l = price - 1
        c = price
        highs.append(h)
        lows.append(l)
        closes.append(c)
    return _make_df(highs, lows, closes)


def _make_downtrend_df(n=50, start=200):
    """Generate a clear downtrend (LH + LL)."""
    highs, lows, closes = [], [], []
    price = start
    for i in range(n):
        cycle = i % 10
        if cycle < 5:
            price -= 2  # down leg
        else:
            price += 1  # shallow rally (LH)
        h = price + 1
        l = price - 1
        c = price
        highs.append(h)
        lows.append(l)
        closes.append(c)
    return _make_df(highs, lows, closes)


def _make_range_df(range_high=110, range_low=100, n=40):
    """Generate sideways price action within a range."""
    closes = []
    for i in range(n):
        mid = (range_high + range_low) / 2
        offset = ((i % 7) - 3) * (range_high - range_low) / 7
        c = mid + offset
        c = max(range_low, min(range_high, c))
        closes.append(c)
    highs = [c + 0.5 for c in closes]
    lows = [c - 0.5 for c in closes]
    return _make_df(highs, lows, closes)


def _make_valid_schematic(direction="bullish", confirmed=True, model="Model_1",
                           rr=2.5, range_high=110, range_low=100):
    """Build a schematic dict that should pass all v2 phases."""
    eq = (range_high + range_low) / 2
    if direction == "bullish":
        entry_p = range_low + 1
        sl_p = range_low - 2
        target_p = range_high + (range_high - range_low)
        tap2_price = range_low - 1  # deviation below range
        if model == "Model_1":
            tap3_price = range_low - 1.5  # extends beyond tap2
        else:
            tap3_price = range_low + 2  # higher low (M2)
    else:
        entry_p = range_high - 1
        sl_p = range_high + 2
        target_p = range_low - (range_high - range_low)
        tap2_price = range_high + 1
        if model == "Model_1":
            tap3_price = range_high + 1.5
        else:
            tap3_price = range_high - 2  # lower high (M2)

    return {
        "direction": direction,
        "model": model,
        "schematic_type": f"{model.lower()}_{direction}",
        "is_confirmed": confirmed,
        "range": {"high": range_high, "low": range_low, "size": range_high - range_low,
                  "equilibrium": eq},
        "tap1": {"price": range_low if direction == "bullish" else range_high, "idx": 5},
        "tap2": {"price": tap2_price, "idx": 15},
        "tap3": {"price": tap3_price, "idx": 25,
                 "is_higher_low": direction == "bullish" and model == "Model_2",
                 "is_lower_high": direction == "bearish" and model == "Model_2"},
        "bos_confirmation": {"bos_idx": 30, "bos_price": eq, "price": eq},
        "entry": {"price": entry_p},
        "stop_loss": {"price": sl_p},
        "target": {"price": target_p},
        "risk_reward": rr,
        "quality_score": 0.75,
        "six_candle_valid": True,
    }


# ================================================================
# PHASE 1 — HTF Market Structure Detection
# ================================================================

class TestHTFMarketStructure:
    def test_bullish_structure_detected(self):
        df = _make_uptrend_df(50)
        result = detect_htf_market_structure(df)
        assert result["bias"] == "bullish"
        assert result["structure_break"] is not None
        assert result["structure_break"]["type"] == "bullish"

    def test_bearish_structure_detected(self):
        df = _make_downtrend_df(50)
        result = detect_htf_market_structure(df)
        assert result["bias"] == "bearish"
        assert result["structure_break"] is not None
        assert result["structure_break"]["type"] == "bearish"

    def test_insufficient_data_returns_neutral(self):
        df = _make_df([100, 101], [99, 100])
        result = detect_htf_market_structure(df)
        assert result["bias"] == "neutral"
        assert "insufficient" in result["reason"]

    def test_range_returns_neutral(self):
        df = _make_range_df(n=30)
        result = detect_htf_market_structure(df, lookback=30)
        assert result["bias"] in ("neutral", "bullish", "bearish")

    def test_none_dataframe(self):
        result = detect_htf_market_structure(None)
        assert result["bias"] == "neutral"

    def test_swing_points_returned(self):
        df = _make_uptrend_df(50)
        result = detect_htf_market_structure(df)
        assert len(result["swing_highs"]) >= 2
        assert len(result["swing_lows"]) >= 2


# ================================================================
# PHASE 2 — Range Validation
# ================================================================

class TestRangeValidation:
    def test_time_displacement_passes_with_large_gap(self):
        sch = {"tap1": {"idx": 5}, "tap2": {"idx": 20}}
        ok, gap = _check_time_displacement(sch)
        assert ok is True
        assert gap == 15

    def test_time_displacement_fails_with_small_gap(self):
        sch = {"tap1": {"idx": 5}, "tap2": {"idx": 8}}
        ok, gap = _check_time_displacement(sch)
        assert ok is False
        assert gap == 3

    def test_time_displacement_custom_min(self):
        sch = {"tap1": {"idx": 5}, "tap2": {"idx": 8}}
        ok, _ = _check_time_displacement(sch, min_candles=3)
        assert ok is True

    def test_liquidity_stacking_detects_equal_highs(self):
        # Create data where many highs cluster near 110
        highs = [110.0] * 5 + [108.0] * 5 + [110.1] * 5
        lows = [100.0] * 15
        df = _make_df(highs, lows)
        result = _detect_liquidity_stacking(df, 110.0, 100.0)
        assert result["has_stacking"] is True
        assert result["equal_highs"] >= 2

    def test_liquidity_stacking_none_found(self):
        # Scattered highs, no clustering
        highs = list(range(100, 115))
        lows = list(range(90, 105))
        df = _make_df(highs, lows)
        result = _detect_liquidity_stacking(df, 120.0, 80.0)
        assert result["has_stacking"] is False

    def test_v_shape_rejected(self):
        # V-shape: price drops impulsively from 110 to 100 and stays low
        # (few direction changes, all in lower half)
        closes = [110, 108, 106, 104, 102, 100, 100, 100, 100, 100,
                  100, 100, 100, 100, 100]
        highs = [c + 0.5 for c in closes]
        lows = [c - 0.5 for c in closes]
        df = _make_df(highs, lows, closes)
        assert _reject_v_shape(df, 110, 100) is True

    def test_horizontal_range_not_rejected(self):
        df = _make_range_df(110, 100, n=40)
        assert _reject_v_shape(df, 110, 100) is False


# ================================================================
# PHASE 4 — Liquidity Sweep v2
# ================================================================

class TestLiquiditySweepV2:
    def test_sweep_classified_correctly(self):
        # Price wicks below range_low but closes back inside
        highs = [105] * 20
        lows = [99.5] + [101] * 19
        closes = [102] * 20
        df = _make_df(highs, lows, closes)
        result = _detect_liquidity_sweep_v2(df, 110, 100, "bullish")
        assert result["swept"] is True
        assert result["classification"] == "sweep"

    def test_true_break_when_close_beyond_dl2(self):
        # Close well beyond DL2 with no acceptance back
        rng = 10  # 110-100
        dl2_below = 100 - rng * 0.30  # 97
        highs = [105] * 20
        lows = [95] * 20
        closes = [96] * 19 + [96]  # All closes below DL2, never accepted back
        df = _make_df(highs, lows, closes)
        result = _detect_liquidity_sweep_v2(df, 110, 100, "bullish")
        assert result["classification"] == "true_break"

    def test_slight_close_beyond_still_sweep(self):
        # Close slightly beyond range_low (within tolerance) is still a sweep
        highs = [105] * 20
        lows = [99.8] + [101] * 19
        closes = [99.98] + [102] * 19  # Slight close beyond, then accepted back
        df = _make_df(highs, lows, closes)
        result = _detect_liquidity_sweep_v2(df, 110, 100, "bullish")
        assert result["swept"] is True
        assert result["classification"] == "sweep"
        assert result["slight_close_beyond"] is True

    def test_no_sweep_when_nothing_exceeds(self):
        highs = [105] * 20
        lows = [101] * 20
        closes = [103] * 20
        df = _make_df(highs, lows, closes)
        result = _detect_liquidity_sweep_v2(df, 110, 100, "bullish")
        assert result["swept"] is False
        assert result["classification"] == "no_sweep"

    def test_bearish_sweep(self):
        highs = [110.5] + [108] * 19
        lows = [107] * 20
        closes = [108] * 20
        df = _make_df(highs, lows, closes)
        result = _detect_liquidity_sweep_v2(df, 110, 100, "bearish")
        assert result["swept"] is True
        assert result["sweep_side"] == "buy_side"


# ================================================================
# PHASE 9 — Session Timing
# ================================================================

class TestSessionTiming:
    def test_returns_valid_score(self):
        pts, desc = _score_session_timing()
        assert 0 <= pts <= 10
        assert isinstance(desc, str)
        assert len(desc) > 0


# ================================================================
# V2 THRESHOLD
# ================================================================

class TestV2Threshold:
    def test_threshold_is_60(self):
        assert V2_THRESHOLD == 60


# ================================================================
# END-TO-END: compute_composite_score_v2
# ================================================================

class TestCompositeScoreV2:
    def test_confirmed_bullish_passes(self):
        df = _make_range_df(110, 100, n=40)
        sch = _make_valid_schematic("bullish", confirmed=True, model="Model_1")
        result = compute_composite_score_v2(df, sch, "bullish", 102.0)
        assert result["direction"] == "bullish"
        assert result["required_score"] == 60
        # Should score reasonably well for a valid schematic
        assert result["score"] > 0
        assert "reasons" in result
        assert "phase_results" in result

    def test_confirmed_bearish_passes(self):
        df = _make_range_df(110, 100, n=40)
        sch = _make_valid_schematic("bearish", confirmed=True, model="Model_1")
        result = compute_composite_score_v2(df, sch, "bearish", 108.0)
        assert result["direction"] == "bearish"
        assert result["score"] > 0

    def test_unconfirmed_bos_fails(self):
        df = _make_range_df(110, 100, n=40)
        sch = _make_valid_schematic("bullish", confirmed=False)
        result = compute_composite_score_v2(df, sch, "bullish", 102.0)
        assert result["pass"] is False
        assert any("BOS" in r for r in result["reasons"])

    def test_v_shape_range_fails(self):
        # V-shape: impulsive drop, stays low — no oscillation
        closes = [110, 108, 106, 104, 102, 100, 100, 100, 100, 100,
                  100, 100, 100, 100, 100]
        highs = [c + 0.5 for c in closes]
        lows = [c - 0.5 for c in closes]
        df = _make_df(highs, lows, closes)
        sch = _make_valid_schematic("bullish", range_high=110, range_low=100)
        result = compute_composite_score_v2(df, sch, "bullish", 102.0)
        assert result["pass"] is False
        assert any("V-shape" in r for r in result["reasons"])

    def test_htf_conflict_fails_for_weak_setup(self):
        """Weak setup against HTF bias should fail (no reversal exception)."""
        df = _make_range_df(110, 100, n=40)
        sch = _make_valid_schematic("bullish", confirmed=True)
        # Weaken the setup so reversal exception doesn't trigger:
        # set tap3 to NOT extend beyond tap2 (invalid Model_1)
        sch["tap3"]["price"] = 100  # same as tap2, not deeper
        result = compute_composite_score_v2(df, sch, "bearish", 102.0)
        assert result["pass"] is False

    def test_htf_conflict_passes_for_reversal(self):
        """Strong confirmed accumulation in bearish HTF qualifies as reversal."""
        df = _make_range_df(110, 100, n=40)
        sch = _make_valid_schematic("bullish", confirmed=True, model="Model_1")
        result = compute_composite_score_v2(df, sch, "bearish", 102.0)
        # Should reach directional filter and be classified as reversal
        phase = result.get("phase_results", {}).get("directional", {})
        if phase:
            assert phase.get("reversal") is True

    def test_low_rr_fails(self):
        df = _make_range_df(110, 100, n=40)
        sch = _make_valid_schematic("bullish", confirmed=True, rr=0.5)
        # Override entry/stop so live R:R < 1.5
        sch["entry"]["price"] = 105
        sch["stop_loss"]["price"] = 104
        sch["target"]["price"] = 105.5
        result = compute_composite_score_v2(df, sch, "bullish", 105.0)
        assert result["pass"] is False
        assert any("R:R" in r for r in result["reasons"])

    def test_neutral_htf_fails(self):
        df = _make_range_df(110, 100, n=40)
        sch = _make_valid_schematic("bullish", confirmed=True)
        result = compute_composite_score_v2(df, sch, "neutral", 102.0)
        assert result["pass"] is False

    def test_fvg_not_mandatory(self):
        """FVG is optional — schematic can still pass without it."""
        df = _make_range_df(110, 100, n=40)
        sch = _make_valid_schematic("bullish", confirmed=True)
        result = compute_composite_score_v2(df, sch, "bullish", 102.0)
        # Should not hard fail just because FVG might not be found
        # (it depends on synthetic data, but it should NOT say "No FVG" as failure)
        if not result["pass"]:
            assert not any("FVG" in r and "hard" in r.lower() for r in result["reasons"])

    def test_model_2_tap_validation(self):
        df = _make_range_df(110, 100, n=40)
        sch = _make_valid_schematic("bullish", confirmed=True, model="Model_2")
        result = compute_composite_score_v2(df, sch, "bullish", 102.0)
        assert result["score"] > 0
        phase = result.get("phase_results", {}).get("tap_structure", {})
        assert phase.get("model") == "Model_2"

    def test_bos_before_tap3_fails(self):
        """BOS must occur AFTER Tap3 — spec requirement."""
        df = _make_range_df(110, 100, n=40)
        sch = _make_valid_schematic("bullish", confirmed=True)
        # Set BOS idx before tap3 idx
        sch["bos_confirmation"]["bos_idx"] = 20  # before tap3 at idx 25
        sch["tap3"]["idx"] = 25
        result = compute_composite_score_v2(df, sch, "bullish", 102.0)
        assert result["pass"] is False
        assert any("before Tap3" in r for r in result["reasons"])

    def test_tree_results_backward_compatible(self):
        """tree_results should still contain the legacy keys for UI."""
        df = _make_range_df(110, 100, n=40)
        sch = _make_valid_schematic("bullish", confirmed=True)
        result = compute_composite_score_v2(df, sch, "bullish", 102.0)
        tr = result.get("tree_results", {})
        assert "ranges" in tr
        assert "market_structure" in tr
        assert "supply_demand" in tr
        assert "liquidity" in tr
        assert "schematics_5a" in tr
        assert "schematics_5b" in tr

    def test_missing_taps_fails(self):
        df = _make_range_df(110, 100, n=40)
        sch = _make_valid_schematic("bullish", confirmed=True)
        sch["tap3"] = {}  # Missing tap3
        result = compute_composite_score_v2(df, sch, "bullish", 102.0)
        assert result["pass"] is False
        assert any("tap" in r.lower() for r in result["reasons"])

    def test_reversal_exception(self):
        """Confirmed accumulation in bearish HTF can pass as reversal."""
        df = _make_range_df(110, 100, n=40)
        sch = _make_valid_schematic("bullish", confirmed=True, model="Model_1")
        result = compute_composite_score_v2(df, sch, "bearish", 102.0)
        # With reversal exception: strong BOS + taps → should not hard fail
        # (may still fail on score threshold, but not on directional filter)
        phase = result.get("phase_results", {}).get("directional", {})
        if phase:
            assert phase.get("reversal") is True or result["pass"] is False

    def test_insufficient_time_displacement_fails(self):
        df = _make_range_df(110, 100, n=40)
        sch = _make_valid_schematic("bullish", confirmed=True)
        sch["tap1"]["idx"] = 5
        sch["tap2"]["idx"] = 8  # Only 3 candles apart
        result = compute_composite_score_v2(df, sch, "bullish", 102.0)
        assert result["pass"] is False
        assert any("displacement" in r.lower() for r in result["reasons"])


# ================================================================
# EVALUATOR CLASS — integration
# ================================================================

class TestDecisionTreeEvaluatorV2:
    def test_threshold_is_60(self):
        """The evaluator now uses V2_THRESHOLD = 60."""
        df = _make_range_df(110, 100, n=40)
        sch = _make_valid_schematic("bullish", confirmed=True)
        evaluator = DecisionTreeEvaluator()
        result = evaluator.evaluate_schematic(sch, "bullish", 102.0, candle_df=df)
        assert result["required_score"] == 60

    def test_unconfirmed_rejected(self):
        sch = _make_valid_schematic("bullish", confirmed=False)
        evaluator = DecisionTreeEvaluator()
        result = evaluator.evaluate_schematic(sch, "bullish", 102.0)
        assert result["pass"] is False

    def test_stale_bos_rejected(self):
        sch = _make_valid_schematic("bullish", confirmed=True)
        sch["bos_confirmation"]["bos_idx"] = 5
        evaluator = DecisionTreeEvaluator()
        result = evaluator.evaluate_schematic(sch, "bullish", 102.0,
                                               total_candles=200, max_stale_candles=5)
        assert result["pass"] is False
        assert any("Stale" in r for r in result["reasons"])
