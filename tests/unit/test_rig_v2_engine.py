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
