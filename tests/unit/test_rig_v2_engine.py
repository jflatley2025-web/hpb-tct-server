"""
Tests for rig_v2_engine — multi-range hierarchical RIG evaluation.
"""

import pytest
from rig_v2_engine import evaluate_rig_v2


# --- Helpers ---

def _ctx(htf_bias="bullish", session="London", session_bias="bearish"):
    """Build a minimal gate context."""
    return {
        "gates": {
            "1A": {"bias": htf_bias},
            "MSCE": {"session": session, "session_bias": session_bias},
        }
    }


def _range(high, low, duration=48, direction="bullish", liquidity=0.5):
    """Build a range dict."""
    return {
        "range_high": high,
        "range_low": low,
        "range_duration_hours": duration,
        "direction": direction,
        "liquidity_score": liquidity,
    }


# --- Edge cases ---

class TestEdgeCases:

    def test_no_ranges_not_evaluated(self):
        result = evaluate_rig_v2(_ctx(), [], 100.0)
        assert result["status"] == "NOT_EVALUATED"
        assert result["evaluated"] is True
        assert "No ranges" in result["reason"]

    def test_none_ranges_not_evaluated(self):
        """None range list treated same as empty."""
        result = evaluate_rig_v2(_ctx(), None, 100.0)
        assert result["status"] == "NOT_EVALUATED"

    def test_no_htf_ranges_not_evaluated(self):
        """All ranges below 24h → no HTF → NOT_EVALUATED."""
        ranges = [_range(110, 100, duration=12)]
        result = evaluate_rig_v2(_ctx(), ranges, 105.0)
        assert result["status"] == "NOT_EVALUATED"
        assert "HTF" in result["reason"]

    def test_invalid_range_skipped(self):
        """Range with high <= low is silently skipped."""
        ranges = [_range(100, 110, duration=48)]  # inverted
        result = evaluate_rig_v2(_ctx(), ranges, 105.0)
        assert result["status"] == "NOT_EVALUATED"

    def test_missing_range_fields_skipped(self):
        """Range missing range_high/range_low is skipped."""
        ranges = [{"range_duration_hours": 48}]
        result = evaluate_rig_v2(_ctx(), ranges, 105.0)
        assert result["status"] == "NOT_EVALUATED"

    def test_empty_context_safe(self):
        """Empty context doesn't crash, returns NOT_EVALUATED."""
        ranges = [_range(110, 100, duration=48)]
        result = evaluate_rig_v2({}, ranges, 105.0)
        assert result["evaluated"] is True
        # neutral bias + no session_bias → NOT_EVALUATED from v1
        assert result["status"] == "NOT_EVALUATED"


# --- HTF dominance ---

class TestHTFDominance:

    def test_selects_longest_duration(self):
        """Dominant HTF range = longest duration."""
        ranges = [
            _range(120, 100, duration=48),
            _range(115, 105, duration=72),
        ]
        result = evaluate_rig_v2(
            _ctx(session_bias="bullish"), ranges, 110.0
        )
        assert result["htf_range"]["range_duration_hours"] == 72

    def test_breaks_tie_by_liquidity(self):
        """Same duration → higher liquidity_score wins."""
        ranges = [
            _range(120, 100, duration=48, liquidity=0.3),
            _range(115, 105, duration=48, liquidity=0.9),
        ]
        result = evaluate_rig_v2(
            _ctx(session_bias="bullish"), ranges, 110.0
        )
        assert result["htf_range"]["liquidity_score"] == 0.9

    def test_missing_liquidity_defaults_to_zero(self):
        """Range without liquidity_score treated as 0."""
        ranges = [
            {"range_high": 120, "range_low": 100, "range_duration_hours": 48},
            _range(115, 105, duration=48, liquidity=0.5),
        ]
        result = evaluate_rig_v2(
            _ctx(session_bias="bullish"), ranges, 110.0
        )
        assert result["htf_range"]["liquidity_score"] == 0.5

    def test_exactly_24h_is_htf(self):
        """Duration == 24h is HTF (boundary inclusive)."""
        ranges = [_range(110, 100, duration=24)]
        result = evaluate_rig_v2(
            _ctx(session_bias="bullish"), ranges, 105.0
        )
        assert result["htf_range"] is not None
        assert result["status"] != "NOT_EVALUATED"


# --- Displacement ---

class TestDisplacement:

    def test_htf_displacement_computed(self):
        """Displacement is from price within dominant HTF range."""
        ranges = [_range(110, 100, duration=48)]
        result = evaluate_rig_v2(
            _ctx(session_bias="bullish"), ranges, 105.0
        )
        assert result["displacement"] == pytest.approx(0.5)

    def test_price_at_range_low(self):
        ranges = [_range(110, 100, duration=48)]
        result = evaluate_rig_v2(
            _ctx(session_bias="bullish"), ranges, 100.0
        )
        assert result["displacement"] == pytest.approx(0.0)

    def test_price_at_range_high(self):
        ranges = [_range(110, 100, duration=48)]
        result = evaluate_rig_v2(
            _ctx(session_bias="bullish"), ranges, 110.0
        )
        assert result["displacement"] == pytest.approx(1.0)


# --- RIG v1 passthrough ---

class TestRIGv1Passthrough:

    def test_aligned_bias_valid(self):
        """HTF bullish + session bullish + high displacement → VALID."""
        ranges = [_range(110, 100, duration=48)]
        result = evaluate_rig_v2(
            _ctx(htf_bias="bullish", session_bias="bullish"),
            ranges, 108.0
        )
        assert result["status"] == "VALID"

    def test_counter_bias_low_disp_blocks(self):
        """HTF bullish + session bearish + low displacement → BLOCK (from v1)."""
        ranges = [_range(110, 100, duration=48)]
        result = evaluate_rig_v2(
            _ctx(htf_bias="bullish", session_bias="bearish"),
            ranges, 101.0  # displacement ~0.1 (below 0.25 threshold)
        )
        assert result["status"] == "BLOCK"
        assert result["confidence"] == 0.0


# --- MTF conflict detection ---

class TestMTFConflict:

    def test_mtf_conflict_blocks_valid(self):
        """HTF mid-range + MTF extreme displacement → BLOCK override."""
        htf = _range(110, 100, duration=48)
        mtf = _range(106, 104, duration=6)  # tight MTF range
        # price=105 → HTF disp=0.5 (mid) | MTF disp=0.5 → no conflict

        # MTF range where price is at extreme
        mtf_extreme = _range(105.5, 104.5, duration=6)
        # price=105 → MTF disp = (105-104.5)/(105.5-104.5) = 0.5 → not extreme

        # Need: HTF mid (0.25-0.75) AND MTF extreme (<0.2 or >0.8)
        # price=105, HTF 100-110 → 0.5 (mid) ✓
        # price=105, MTF 105-106 → disp=0.0 (<0.2) ✓
        mtf_trap = _range(106, 105, duration=6)

        result = evaluate_rig_v2(
            _ctx(htf_bias="bullish", session_bias="bullish"),
            [htf, mtf_trap], 105.0
        )
        assert result["status"] == "BLOCK"
        assert result["mtf_conflict"] is True
        assert result["confidence"] == 0.0
        assert "MTF conflict" in result["reason"]

    def test_no_mtf_conflict_when_htf_not_mid(self):
        """HTF displacement outside mid-range → no conflict check."""
        htf = _range(110, 100, duration=48)
        mtf = _range(102, 101, duration=6)  # would be extreme if checked
        # price=101 → HTF disp=0.1 (not mid-range)
        result = evaluate_rig_v2(
            _ctx(htf_bias="bullish", session_bias="bearish"),
            [htf, mtf], 101.0
        )
        # BLOCK comes from v1 (counter-bias + low disp), not from MTF
        assert result["mtf_conflict"] is False

    def test_no_conflict_when_mtf_not_extreme(self):
        """HTF mid + MTF also mid → no conflict."""
        htf = _range(110, 100, duration=48)
        mtf = _range(106, 104, duration=6)
        # price=105 → HTF 0.5 (mid), MTF 0.5 (mid) → no conflict
        result = evaluate_rig_v2(
            _ctx(htf_bias="bullish", session_bias="bullish"),
            [htf, mtf], 105.0
        )
        assert result["status"] == "VALID"
        assert result["mtf_conflict"] is False

    def test_mtf_extreme_high_triggers_conflict(self):
        """MTF displacement > 0.8 also triggers conflict."""
        htf = _range(110, 100, duration=48)
        mtf = _range(104, 103, duration=4)
        # price=105 → HTF=0.5 (mid), MTF=(105-103)/(104-103)=2.0 clamped to 1.0 (>0.8) ✓
        result = evaluate_rig_v2(
            _ctx(htf_bias="bullish", session_bias="bullish"),
            [htf, mtf], 105.0
        )
        assert result["mtf_conflict"] is True
        assert result["status"] == "BLOCK"

    def test_mtf_conflict_does_not_override_v1_block(self):
        """If v1 already BLOCKs, MTF conflict doesn't change outcome."""
        htf = _range(110, 100, duration=48)
        mtf = _range(106, 105, duration=6)
        # price=101 → HTF=0.1 (low), so v1 blocks on counter-bias
        # HTF not in mid-range → mtf_conflict = False
        result = evaluate_rig_v2(
            _ctx(htf_bias="bullish", session_bias="bearish"),
            [htf, mtf], 101.0
        )
        assert result["status"] == "BLOCK"
        assert result["confidence"] == 0.0

    def test_no_mtf_ranges_no_conflict(self):
        """Only HTF ranges, no MTF → no conflict possible."""
        ranges = [
            _range(110, 100, duration=48),
            _range(115, 105, duration=36),
        ]
        result = evaluate_rig_v2(
            _ctx(htf_bias="bullish", session_bias="bullish"),
            ranges, 105.0
        )
        assert result["mtf_conflict"] is False


# --- Output structure ---

class TestOutputStructure:

    def test_all_required_fields(self):
        """Output contains all required fields."""
        ranges = [_range(110, 100, duration=48)]
        result = evaluate_rig_v2(
            _ctx(session_bias="bullish"), ranges, 105.0
        )
        required = [
            "status", "reason", "confidence", "evaluated",
            "displacement", "htf_bias", "session_bias",
            "htf_range", "mtf_conflict",
        ]
        for field in required:
            assert field in result, f"Missing field: {field}"

    def test_gate_label_is_rig_v2(self):
        ranges = [_range(110, 100, duration=48)]
        result = evaluate_rig_v2(
            _ctx(session_bias="bullish"), ranges, 105.0
        )
        assert result["Gate"] == "RIG_v2"

    def test_evaluated_always_true(self):
        """Evaluated is True even on NOT_EVALUATED status."""
        result = evaluate_rig_v2(_ctx(), [], 100.0)
        assert result["evaluated"] is True


# --- Malformed input robustness ---

class TestMalformedInputs:

    def test_non_dict_ranges_skipped(self):
        """Non-dict items in ranges list are silently skipped."""
        ranges = [None, 123, "bad", _range(110, 100, duration=48)]
        result = evaluate_rig_v2(
            _ctx(session_bias="bullish"), ranges, 105.0
        )
        # The one valid range should be found
        assert result["htf_range"] is not None
        assert result["status"] != "NOT_EVALUATED"

    def test_all_non_dict_ranges_not_evaluated(self):
        """All non-dict items → no valid ranges → NOT_EVALUATED."""
        ranges = [None, 123, "bad", True]
        result = evaluate_rig_v2(_ctx(), ranges, 105.0)
        assert result["status"] == "NOT_EVALUATED"
        assert result["evaluated"] is True

    def test_bad_numeric_duration(self):
        """Non-numeric range_duration_hours treated as 0 (MTF)."""
        ranges = [
            {"range_high": 110, "range_low": 100, "range_duration_hours": "abc"},
        ]
        result = evaluate_rig_v2(_ctx(), ranges, 105.0)
        # "abc" → 0.0 → MTF, no HTF → NOT_EVALUATED
        assert result["status"] == "NOT_EVALUATED"

    def test_bad_numeric_range_high(self):
        """Non-numeric range_high → range skipped."""
        ranges = [
            {"range_high": "bad", "range_low": 100, "range_duration_hours": 48},
        ]
        result = evaluate_rig_v2(_ctx(), ranges, 105.0)
        assert result["status"] == "NOT_EVALUATED"

    def test_bad_numeric_range_low(self):
        """Non-numeric range_low → range skipped."""
        ranges = [
            {"range_high": 110, "range_low": None, "range_duration_hours": 48},
        ]
        result = evaluate_rig_v2(_ctx(), ranges, 105.0)
        assert result["status"] == "NOT_EVALUATED"

    def test_mixed_valid_and_invalid(self):
        """Mix of valid and invalid ranges — valid ones used, invalid skipped."""
        ranges = [
            None,                                    # non-dict
            {"range_high": "x", "range_low": 100},   # bad high
            _range(100, 110, duration=48),            # inverted
            _range(110, 100, duration=48),            # valid HTF
            {"range_duration_hours": 48},             # missing high/low
        ]
        result = evaluate_rig_v2(
            _ctx(session_bias="bullish"), ranges, 105.0
        )
        assert result["htf_range"]["range_high"] == 110
        assert result["status"] != "NOT_EVALUATED"

    def test_bad_liquidity_score_does_not_crash(self):
        """Non-numeric liquidity_score → safe fallback in ranking."""
        ranges = [
            {"range_high": 110, "range_low": 100, "range_duration_hours": 48,
             "liquidity_score": "bad"},
            _range(115, 105, duration=48, liquidity=0.5),
        ]
        result = evaluate_rig_v2(
            _ctx(session_bias="bullish"), ranges, 110.0
        )
        # "bad" → 0.0, so the 0.5 range wins the tiebreak
        assert result["htf_range"]["liquidity_score"] == 0.5

    def test_non_dict_context_safe(self):
        """Non-dict context doesn't crash."""
        ranges = [_range(110, 100, duration=48)]
        result = evaluate_rig_v2("not_a_dict", ranges, 105.0)
        assert result["evaluated"] is True

    def test_none_context_safe(self):
        """None context doesn't crash."""
        ranges = [_range(110, 100, duration=48)]
        result = evaluate_rig_v2(None, ranges, 105.0)
        assert result["evaluated"] is True


# --- Liquidity integration ---

def _ctx_with_liq(liq_data, htf_bias="bullish", session_bias="bullish"):
    """Build context with liquidity data."""
    ctx = _ctx(htf_bias=htf_bias, session_bias=session_bias)
    ctx["liquidity"] = liq_data
    return ctx


class TestLiquidityTrueBreakBlock:

    def test_true_break_returns_invalid(self):
        """TRUE_BREAK + liquidity_valid=False → INVALID, confidence=0."""
        liq = {"liquidity_valid": False, "sweep_class": "true_break",
               "path_score": 0.0, "trade_bias": "WAIT", "entry_ready": False}
        ranges = [_range(110, 100, duration=48)]
        result = evaluate_rig_v2(_ctx_with_liq(liq), ranges, 108.0)
        assert result["status"] == "INVALID"
        assert result["confidence"] == 0.0
        assert "TRUE_BREAK" in result["reason"]
        assert result["liquidity"]["valid"] is False

    def test_true_break_with_sweep_classification_key(self):
        """sweep_classification (alternate key) also triggers block."""
        liq = {"liquidity_valid": False, "sweep_classification": "true_break",
               "path_score": 0.0, "trade_bias": "WAIT", "entry_ready": False}
        ranges = [_range(110, 100, duration=48)]
        result = evaluate_rig_v2(_ctx_with_liq(liq), ranges, 108.0)
        assert result["status"] == "INVALID"

    def test_true_break_valid_liq_does_not_block(self):
        """liquidity_valid=True + true_break → no block (contradictory but safe)."""
        liq = {"liquidity_valid": True, "sweep_class": "true_break",
               "path_score": 1.0, "trade_bias": "LONG", "entry_ready": True}
        ranges = [_range(110, 100, duration=48)]
        result = evaluate_rig_v2(
            _ctx_with_liq(liq, htf_bias="bullish"), ranges, 108.0
        )
        assert result["status"] != "INVALID"


class TestLiquidityBiasAlignment:

    def test_aligned_bias_boosts_confidence(self):
        """Aligned liquidity bias adds +0.10."""
        liq = {"liquidity_valid": True, "trade_bias": "BULLISH",
               "path_score": 1.0, "entry_ready": False}
        ranges = [_range(110, 100, duration=48)]
        result_with = evaluate_rig_v2(
            _ctx_with_liq(liq, htf_bias="bullish"), ranges, 108.0
        )
        result_without = evaluate_rig_v2(
            _ctx(htf_bias="bullish", session_bias="bullish"), ranges, 108.0
        )
        # With liquidity: confidence * 1.0 + 0.10
        assert result_with["confidence"] > result_without["confidence"]

    def test_opposing_bias_penalizes_confidence(self):
        """Opposing liquidity bias subtracts -0.15 vs aligned +0.10."""
        liq_opposing = {"liquidity_valid": True, "trade_bias": "BEARISH",
                        "path_score": 1.0, "entry_ready": False}
        liq_aligned = {"liquidity_valid": True, "trade_bias": "BULLISH",
                       "path_score": 1.0, "entry_ready": False}
        ranges = [_range(110, 100, duration=48)]
        result_opposing = evaluate_rig_v2(
            _ctx_with_liq(liq_opposing, htf_bias="bullish"), ranges, 108.0
        )
        result_aligned = evaluate_rig_v2(
            _ctx_with_liq(liq_aligned, htf_bias="bullish"), ranges, 108.0
        )
        # Opposing must be strictly less than aligned
        assert result_opposing["confidence"] < result_aligned["confidence"]

    def test_wait_bias_no_change(self):
        """trade_bias=WAIT leaves confidence unchanged (before path_score)."""
        liq = {"liquidity_valid": True, "trade_bias": "WAIT",
               "path_score": 1.0, "entry_ready": False}
        ranges = [_range(110, 100, duration=48)]
        result = evaluate_rig_v2(
            _ctx_with_liq(liq, htf_bias="bullish"), ranges, 108.0
        )
        result_without = evaluate_rig_v2(
            _ctx(htf_bias="bullish", session_bias="bullish"), ranges, 108.0
        )
        # path_score=1.0 → no multiplicative change, no bias change
        assert result["confidence"] == pytest.approx(result_without["confidence"])


class TestLiquidityPathScore:

    def test_partial_path_reduces_confidence(self):
        """path_score=0.65 multiplies confidence down."""
        liq = {"liquidity_valid": True, "trade_bias": "WAIT",
               "path_score": 0.65, "entry_ready": False}
        ranges = [_range(110, 100, duration=48)]
        result = evaluate_rig_v2(
            _ctx_with_liq(liq, htf_bias="bullish"), ranges, 108.0
        )
        result_without = evaluate_rig_v2(
            _ctx(htf_bias="bullish", session_bias="bullish"), ranges, 108.0
        )
        # 0.65x multiplier
        expected = result_without["confidence"] * 0.65
        assert result["confidence"] == pytest.approx(expected, abs=0.01)

    def test_zero_path_score_no_multiplication(self):
        """path_score=0 skips multiplication (guard: if path_score > 0)."""
        liq = {"liquidity_valid": True, "trade_bias": "WAIT",
               "path_score": 0.0, "entry_ready": False}
        ranges = [_range(110, 100, duration=48)]
        result = evaluate_rig_v2(
            _ctx_with_liq(liq, htf_bias="bullish"), ranges, 108.0
        )
        result_without = evaluate_rig_v2(
            _ctx(htf_bias="bullish", session_bias="bullish"), ranges, 108.0
        )
        assert result["confidence"] == pytest.approx(result_without["confidence"])


class TestLiquidityEntryReady:

    def test_entry_ready_adds_boost(self):
        """entry_ready=True adds +0.05."""
        liq_ready = {"liquidity_valid": True, "trade_bias": "WAIT",
                     "path_score": 1.0, "entry_ready": True}
        liq_not = {"liquidity_valid": True, "trade_bias": "WAIT",
                   "path_score": 1.0, "entry_ready": False}
        ranges = [_range(110, 100, duration=48)]
        result_ready = evaluate_rig_v2(
            _ctx_with_liq(liq_ready, htf_bias="bullish"), ranges, 108.0
        )
        result_not = evaluate_rig_v2(
            _ctx_with_liq(liq_not, htf_bias="bullish"), ranges, 108.0
        )
        assert result_ready["confidence"] == pytest.approx(
            result_not["confidence"] + 0.05, abs=0.001
        )


class TestLiquidityMissing:

    def test_no_liquidity_data_safe(self):
        """Missing liquidity → no crash, confidence unchanged, reason appended."""
        ranges = [_range(110, 100, duration=48)]
        result = evaluate_rig_v2(
            _ctx(htf_bias="bullish", session_bias="bullish"), ranges, 108.0
        )
        assert "liquidity" in result
        assert result["liquidity"]["valid"] is None
        assert "unavailable" in result["reason"]

    def test_none_liquidity_safe(self):
        """Explicit None liquidity → safe fallback."""
        ctx = _ctx(htf_bias="bullish", session_bias="bullish")
        ctx["liquidity"] = None
        ranges = [_range(110, 100, duration=48)]
        result = evaluate_rig_v2(ctx, ranges, 108.0)
        assert result["liquidity"]["valid"] is None

    def test_empty_liquidity_dict_safe(self):
        """Empty liquidity dict → treated as missing."""
        ctx = _ctx(htf_bias="bullish", session_bias="bullish")
        ctx["liquidity"] = {}
        ranges = [_range(110, 100, duration=48)]
        result = evaluate_rig_v2(ctx, ranges, 108.0)
        assert result["liquidity"]["valid"] is None


class TestLiquidityConfidenceClamp:

    def test_confidence_clamped_to_one(self):
        """Multiple boosts can't exceed 1.0."""
        liq = {"liquidity_valid": True, "trade_bias": "BULLISH",
               "path_score": 1.0, "entry_ready": True}
        ranges = [_range(110, 100, duration=48)]
        result = evaluate_rig_v2(
            _ctx_with_liq(liq, htf_bias="bullish"), ranges, 108.0
        )
        assert result["confidence"] <= 1.0

    def test_confidence_clamped_to_zero(self):
        """Heavy penalty can't go below 0.0."""
        liq = {"liquidity_valid": True, "trade_bias": "BEARISH",
               "path_score": 0.3, "entry_ready": False}
        ranges = [_range(110, 100, duration=48)]
        # Use counter-bias session to get low base confidence
        result = evaluate_rig_v2(
            _ctx_with_liq(liq, htf_bias="bullish", session_bias="bearish"),
            ranges, 103.0
        )
        assert result["confidence"] >= 0.0


class TestLiquidityDebugOutput:

    def test_liquidity_debug_fields(self):
        """Output includes liquidity debug with all expected fields."""
        liq = {"liquidity_valid": True, "trade_bias": "LONG",
               "path_score": 0.65, "entry_ready": True}
        ranges = [_range(110, 100, duration=48)]
        result = evaluate_rig_v2(
            _ctx_with_liq(liq, htf_bias="bullish"), ranges, 108.0
        )
        liq_out = result["liquidity"]
        assert liq_out["valid"] is True
        assert liq_out["bias"] == "LONG"
        assert liq_out["path_score"] == pytest.approx(0.65)
        assert liq_out["entry_ready"] is True
