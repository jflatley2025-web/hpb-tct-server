"""
Unit tests for hpb_rig_validator.py
Tests the TCT-aware Range Integrity Gate (RIG) validation logic.

Decision tree:
  - Trend-aligned → VALID
  - Counter-bias mid-range → BLOCK
  - Counter-bias extreme with displacement → CONDITIONAL
  - Counter-bias extreme without displacement → BLOCK
"""
import pytest
from datetime import datetime
from hpb_rig_validator import range_integrity_validator


def _ctx(*, htf_bias="bullish", session_bias="bearish", disp=0.5,
         range_valid=True, duration=48, score=0.80, session="London"):
    """Build a minimal RIG context for testing."""
    return {
        "gates": {
            "1A": {"bias": htf_bias},
            "RCM": {"valid": range_valid, "range_duration_hours": duration},
            "MSCE": {"session_bias": session_bias, "session": session},
            "1D": {"score": score},
        },
        "local_range_displacement": disp,
    }


# ═══════════════════════════════════════════════════════════════════
# Case A: Trend-aligned trades → always VALID
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestTrendAligned:
    """Trend-aligned trades pass unconditionally regardless of position."""

    def test_aligned_at_low(self):
        result = range_integrity_validator(_ctx(session_bias="bullish", disp=0.10))
        assert result["status"] == "VALID"
        assert result["counter_bias"] is False
        assert result["confidence_modifier"] == 1.0

    def test_aligned_at_mid(self):
        result = range_integrity_validator(_ctx(session_bias="bullish", disp=0.50))
        assert result["status"] == "VALID"

    def test_aligned_at_high(self):
        result = range_integrity_validator(_ctx(session_bias="bullish", disp=0.90))
        assert result["status"] == "VALID"

    def test_aligned_preserves_confidence(self):
        result = range_integrity_validator(_ctx(session_bias="bullish", disp=0.50, score=0.85))
        assert result["confidence"] == pytest.approx(0.85)
        assert result["confidence_modifier"] == 1.0


# ═══════════════════════════════════════════════════════════════════
# Case B: Counter-bias mid-range → BLOCK
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestCounterBiasMidRange:
    """Counter-bias trades in mid-range (0.25 < pos < 0.75) are blocked."""

    def test_mid_range_center(self):
        """Displacement 0.50 → mid zone → BLOCK"""
        result = range_integrity_validator(_ctx(disp=0.50))
        assert result["status"] == "BLOCK"
        assert result["position"] == "mid"
        assert result["counter_bias"] is True
        assert result["confidence"] == 0.0
        assert "mid-range" in result["reason"]

    def test_mid_range_lower_edge(self):
        """Displacement 0.26 → still mid zone → BLOCK"""
        result = range_integrity_validator(_ctx(disp=0.26))
        assert result["status"] == "BLOCK"
        assert result["position"] == "mid"

    def test_mid_range_upper_edge(self):
        """Displacement 0.74 → still mid zone → BLOCK"""
        result = range_integrity_validator(_ctx(disp=0.74))
        assert result["status"] == "BLOCK"
        assert result["position"] == "mid"

    def test_mid_range_zeroes_confidence(self):
        result = range_integrity_validator(_ctx(disp=0.50, score=0.90))
        assert result["confidence"] == 0.0
        assert result["confidence_modifier"] == 0.0


# ═══════════════════════════════════════════════════════════════════
# Case C: Counter-bias at range extreme WITH displacement → CONDITIONAL
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestCounterBiasExtremeWithDisplacement:
    """Counter-bias at LOW/HIGH zone with sufficient displacement → CONDITIONAL."""

    def test_low_zone_with_displacement(self):
        """Disp 0.20 → LOW zone, disp >= 0.15 → CONDITIONAL"""
        result = range_integrity_validator(_ctx(disp=0.20))
        assert result["status"] == "CONDITIONAL"
        assert result["position"] == "low"
        assert result["counter_bias"] is True
        assert 0.0 < result["confidence_modifier"] <= 0.7

    def test_high_zone_with_displacement(self):
        """Disp 0.80 → HIGH zone, (1-0.80)=0.20 >= 0.15 → CONDITIONAL"""
        result = range_integrity_validator(_ctx(disp=0.80))
        assert result["status"] == "CONDITIONAL"
        assert result["position"] == "high"
        assert 0.0 < result["confidence_modifier"] <= 0.7

    def test_low_zone_at_threshold(self):
        """Disp exactly 0.15 → LOW zone, meets threshold → CONDITIONAL"""
        result = range_integrity_validator(_ctx(disp=0.15))
        assert result["status"] == "CONDITIONAL"
        assert result["position"] == "low"

    def test_high_zone_at_threshold(self):
        """Disp 0.85 → HIGH zone, (1-0.85)=0.15 meets threshold → CONDITIONAL"""
        result = range_integrity_validator(_ctx(disp=0.85))
        assert result["status"] == "CONDITIONAL"
        assert result["position"] == "high"

    def test_confidence_penalized(self):
        """CONDITIONAL reduces confidence from base score."""
        result = range_integrity_validator(_ctx(disp=0.20, score=0.80))
        assert result["confidence"] < 0.80
        assert result["confidence"] > 0.0

    def test_short_range_further_penalizes(self):
        """Range < 24h applies additional penalty."""
        r_long = range_integrity_validator(_ctx(disp=0.20, duration=48))
        r_short = range_integrity_validator(_ctx(disp=0.20, duration=12))
        assert r_short["confidence_modifier"] < r_long["confidence_modifier"]

    def test_low_zone_boundary(self):
        """Disp exactly 0.25 → LOW zone boundary → CONDITIONAL"""
        result = range_integrity_validator(_ctx(disp=0.25))
        assert result["status"] == "CONDITIONAL"
        assert result["position"] == "low"

    def test_high_zone_boundary(self):
        """Disp exactly 0.75 → HIGH zone boundary → CONDITIONAL"""
        result = range_integrity_validator(_ctx(disp=0.75))
        assert result["status"] == "CONDITIONAL"
        assert result["position"] == "high"


# ═══════════════════════════════════════════════════════════════════
# Case D: Counter-bias at extreme WITHOUT displacement → BLOCK
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestCounterBiasExtremeNoDisplacement:
    """Counter-bias at extreme but displacement < 0.15 → BLOCK."""

    def test_low_zone_no_displacement(self):
        """Disp 0.05 → LOW zone, disp < 0.15 → BLOCK"""
        result = range_integrity_validator(_ctx(disp=0.05))
        assert result["status"] == "BLOCK"
        assert result["position"] == "low"
        assert result["confidence"] == 0.0

    def test_high_zone_strong_displacement(self):
        """Disp 0.95 → HIGH zone, disp 0.95 >= 0.15 → CONDITIONAL (strong displacement)"""
        result = range_integrity_validator(_ctx(disp=0.95))
        assert result["status"] == "CONDITIONAL"
        assert result["position"] == "high"

    def test_low_zone_just_below_threshold(self):
        """Disp 0.14 → below 0.15 threshold → BLOCK"""
        result = range_integrity_validator(_ctx(disp=0.14))
        assert result["status"] == "BLOCK"

    def test_low_zone_very_low_displacement(self):
        """Disp 0.02 → LOW zone, disp < 0.15 → BLOCK"""
        result = range_integrity_validator(_ctx(disp=0.02))
        assert result["status"] == "BLOCK"


# ═══════════════════════════════════════════════════════════════════
# Range validity edge cases
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestRangeValidity:
    """Range validity and missing data handling."""

    def test_invalid_range_passes(self):
        """Invalid range → defers to RCM, returns VALID."""
        result = range_integrity_validator(_ctx(range_valid=False, disp=0.50))
        assert result["status"] == "VALID"

    def test_missing_displacement_defaults_neutral(self):
        """Missing displacement defaults to 0.5 (mid) → counter-bias mid → BLOCK."""
        ctx = _ctx(disp=0.50)
        del ctx["local_range_displacement"]
        result = range_integrity_validator(ctx)
        assert result["status"] == "BLOCK"

    def test_missing_gates(self):
        """Empty gates → no crash, returns VALID."""
        result = range_integrity_validator({"gates": {}, "local_range_displacement": 0.50})
        assert result["status"] == "VALID"
        assert result["Gate"] == "RIG"

    def test_partial_gates(self):
        """Partial gates → no crash."""
        ctx = {"gates": {"1A": {"bias": "bullish"}}, "local_range_displacement": 0.50}
        result = range_integrity_validator(ctx)
        assert "status" in result
        assert result["Gate"] == "RIG"

    def test_neutral_htf_bias_counter(self):
        """Neutral HTF + bullish session → counter-bias detected."""
        result = range_integrity_validator(
            _ctx(htf_bias="neutral", session_bias="bullish", disp=0.50)
        )
        assert result["counter_bias"] is True
        assert result["status"] == "BLOCK"  # mid-range counter-bias


# ═══════════════════════════════════════════════════════════════════
# Output structure
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestOutputStructure:
    """Verify output dict has all required fields."""

    def test_all_fields_present(self):
        result = range_integrity_validator(_ctx(disp=0.50))
        for key in ("timestamp", "status", "Gate", "reason", "confidence",
                     "htf_bias", "session_bias", "evaluated",
                     "position", "counter_bias", "confidence_modifier"):
            assert key in result, f"Missing key: {key}"

    def test_timestamp_format(self):
        result = range_integrity_validator(_ctx(disp=0.50))
        datetime.strptime(result["timestamp"], "%Y-%m-%d %H:%M:%S")

    def test_evaluated_always_true(self):
        result = range_integrity_validator(_ctx(disp=0.50))
        assert result["evaluated"] is True


# ═══════════════════════════════════════════════════════════════════
# Session name parametrized
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestSessionNames:
    @pytest.mark.parametrize("session_name", [
        "London", "New York", "Tokyo", "Sydney", "Unknown"
    ])
    def test_sessions_at_mid_range(self, session_name):
        """Counter-bias mid-range blocks regardless of session name."""
        result = range_integrity_validator(_ctx(disp=0.50, session=session_name))
        assert result["status"] == "BLOCK"

    @pytest.mark.parametrize("session_name", [
        "London", "New York", "Tokyo", "Sydney", "Unknown"
    ])
    def test_sessions_at_extreme(self, session_name):
        """Counter-bias at extreme with displacement → CONDITIONAL."""
        result = range_integrity_validator(_ctx(disp=0.20, session=session_name))
        assert result["status"] == "CONDITIONAL"


# ═══════════════════════════════════════════════════════════════════
# evaluate_rig standalone function tests
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestEvaluateRigStandalone:
    """Tests for the new evaluate_rig() function (spec-compliant)."""

    def test_case_a_trend_aligned(self):
        from hpb_rig_validator import evaluate_rig
        result = evaluate_rig("bullish", "bullish", 100, 110, 105, 0.5, 48)
        assert result["rig_status"] == "VALID"
        assert result["confidence_modifier"] == 1.0
        assert result["counter_bias"] is False

    def test_case_b_counter_bias_mid(self):
        from hpb_rig_validator import evaluate_rig
        result = evaluate_rig("bullish", "bearish", 100, 110, 105, 0.5, 48)
        assert result["rig_status"] == "BLOCK"
        assert result["confidence_modifier"] == 0.0
        assert "mid-range" in result["reason"]

    def test_case_c_extreme_with_displacement(self):
        from hpb_rig_validator import evaluate_rig
        result = evaluate_rig("bullish", "bearish", 100, 110, 102, 0.2, 48)
        assert result["rig_status"] == "CONDITIONAL"
        assert 0.5 <= result["confidence_modifier"] <= 0.7

    def test_case_d_extreme_no_displacement(self):
        from hpb_rig_validator import evaluate_rig
        result = evaluate_rig("bullish", "bearish", 100, 110, 101, 0.1, 48)
        assert result["rig_status"] == "BLOCK"
        assert result["confidence_modifier"] == 0.0
        assert "No displacement" in result["reason"]

    def test_displacement_bonus(self):
        """Displacement > 0.25 adds +0.1 to confidence."""
        from hpb_rig_validator import evaluate_rig
        r_low = evaluate_rig("bullish", "bearish", 100, 110, 102, 0.20, 48)
        r_high = evaluate_rig("bullish", "bearish", 100, 110, 102, 0.30, 48)
        assert r_high["confidence_modifier"] > r_low["confidence_modifier"]

    def test_session_expansion_bonus(self):
        """Bullish reversal at LOW + expansion session → +0.1."""
        from hpb_rig_validator import evaluate_rig
        r_neutral = evaluate_rig("bearish", "bullish", 100, 110, 102, 0.2, 48, "neutral")
        r_expansion = evaluate_rig("bearish", "bullish", 100, 110, 102, 0.2, 48, "expansion")
        assert r_expansion["confidence_modifier"] > r_neutral["confidence_modifier"]

    def test_session_distribution_bonus(self):
        """Bearish reversal at HIGH + distribution session → +0.1."""
        from hpb_rig_validator import evaluate_rig
        r_neutral = evaluate_rig("bullish", "bearish", 100, 110, 108, 0.8, 48, "neutral")
        r_distrib = evaluate_rig("bullish", "bearish", 100, 110, 108, 0.8, 48, "distribution")
        assert r_distrib["confidence_modifier"] > r_neutral["confidence_modifier"]

    def test_short_duration_penalty(self):
        """Duration < 24h applies conf *= 0.8."""
        from hpb_rig_validator import evaluate_rig
        r_long = evaluate_rig("bullish", "bearish", 100, 110, 102, 0.2, 48)
        r_short = evaluate_rig("bullish", "bearish", 100, 110, 102, 0.2, 12)
        assert r_short["confidence_modifier"] < r_long["confidence_modifier"]

    def test_displacement_zero_fix(self):
        """local_displacement == 0.0 is recomputed from range midpoint."""
        from hpb_rig_validator import evaluate_rig
        # price=108, range=[100,110], mid=105, abs(108-105)/10 = 0.3
        # So recomputed disp = 0.3 >= 0.15 → CONDITIONAL
        result = evaluate_rig("bullish", "bearish", 100, 110, 108, 0.0, 48)
        assert result["rig_status"] == "CONDITIONAL"

    def test_output_format(self):
        from hpb_rig_validator import evaluate_rig
        result = evaluate_rig("bullish", "bearish", 100, 110, 105, 0.5, 48)
        for key in ("rig_status", "confidence_modifier", "position",
                     "counter_bias", "reason"):
            assert key in result


# ═══════════════════════════════════════════════════════════════════
# Edge cases: zone threshold boundaries
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestZoneBoundaryEdgeCases:
    """Test behavior at exact zone boundaries (0.25, 0.75) and epsilon away."""

    def test_position_exactly_0_25_is_low(self):
        """Position 0.25 → LOW zone (boundary inclusive)."""
        from hpb_rig_validator import evaluate_rig
        result = evaluate_rig("bullish", "bearish", 100, 110, 102.5, 0.25, 48)
        assert result["position"] == pytest.approx(0.25)
        # LOW zone, counter-bias, disp=0.25 >= 0.15 → CONDITIONAL
        assert result["rig_status"] == "CONDITIONAL"

    def test_position_0_2501_is_mid(self):
        """Position just above 0.25 → MID zone → BLOCK."""
        from hpb_rig_validator import evaluate_rig
        result = evaluate_rig("bullish", "bearish", 100, 1000, 325.09, 0.26, 48)
        # 325.09 is at ~0.2501 position → MID zone
        assert result["rig_status"] == "BLOCK"

    def test_position_0_2499_is_low(self):
        """Position just below 0.25 → LOW zone → CONDITIONAL (not mid-range BLOCK)."""
        from hpb_rig_validator import evaluate_rig
        # position = (324.9 - 100) / (1000 - 100) = 0.2499 → LOW zone
        result = evaluate_rig("bullish", "bearish", 100, 1000, 324.9, 0.24, 48)
        assert result["rig_status"] == "CONDITIONAL"

    def test_position_exactly_0_75_is_high(self):
        """Position 0.75 → HIGH zone (boundary inclusive)."""
        from hpb_rig_validator import evaluate_rig
        result = evaluate_rig("bullish", "bearish", 100, 110, 107.5, 0.75, 48)
        assert result["position"] == pytest.approx(0.75)
        assert result["rig_status"] == "CONDITIONAL"

    def test_position_0_7499_is_mid(self):
        """Position just below 0.75 → MID zone → BLOCK."""
        from hpb_rig_validator import evaluate_rig
        result = evaluate_rig("bullish", "bearish", 100, 1000, 774.9, 0.74, 48)
        assert result["rig_status"] == "BLOCK"

    def test_displacement_exactly_0_15_allows(self):
        """Displacement exactly 0.15 → meets threshold → CONDITIONAL."""
        from hpb_rig_validator import evaluate_rig
        result = evaluate_rig("bullish", "bearish", 100, 110, 101.5, 0.15, 48)
        assert result["rig_status"] == "CONDITIONAL"

    def test_displacement_0_1499_blocks(self):
        """Displacement just below 0.15 → BLOCK."""
        from hpb_rig_validator import evaluate_rig
        result = evaluate_rig("bullish", "bearish", 100, 110, 101.4, 0.149, 48)
        assert result["rig_status"] == "BLOCK"
        assert result["confidence_modifier"] == 0.0

    def test_position_at_0_extremes(self):
        """Price at exact range low → position 0.0, disp determines outcome."""
        from hpb_rig_validator import evaluate_rig
        r = evaluate_rig("bullish", "bearish", 100, 110, 100, 0.05, 48)
        assert r["position"] == pytest.approx(0.0)
        assert r["rig_status"] == "BLOCK"  # disp < 0.15

    def test_position_at_1_extremes(self):
        """Price at exact range high → position 1.0."""
        from hpb_rig_validator import evaluate_rig
        r = evaluate_rig("bullish", "bearish", 100, 110, 110, 0.20, 48)
        assert r["position"] == pytest.approx(1.0)
        assert r["rig_status"] == "CONDITIONAL"


# ═══════════════════════════════════════════════════════════════════
# Confidence modifier scaling tests
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestConfidenceModifierScaling:
    """Validate confidence_modifier ranges and scaling behavior."""

    def test_conf_mod_range(self):
        """CONDITIONAL confidence_modifier is always in [0.0, 0.7]."""
        from hpb_rig_validator import evaluate_rig
        for disp in [0.15, 0.20, 0.25, 0.30, 0.50, 0.80, 1.0]:
            r = evaluate_rig("bullish", "bearish", 100, 110, 102, disp, 48)
            if r["rig_status"] == "CONDITIONAL":
                assert 0.0 <= r["confidence_modifier"] <= 0.7, \
                    f"conf_mod {r['confidence_modifier']} out of [0, 0.7] at disp={disp}"

    def test_conf_mod_base_is_0_5(self):
        """Base CONDITIONAL conf_mod starts at 0.5 (no bonuses)."""
        from hpb_rig_validator import evaluate_rig
        # disp=0.20 (< 0.25 so no bonus), neutral session, duration >= 24
        r = evaluate_rig("bullish", "bearish", 100, 110, 102, 0.20, 48, "neutral")
        assert r["rig_status"] == "CONDITIONAL"
        assert r["confidence_modifier"] == pytest.approx(0.5)

    def test_conf_mod_displacement_bonus_adds_0_1(self):
        """Displacement > 0.25 adds +0.1."""
        from hpb_rig_validator import evaluate_rig
        r_no_bonus = evaluate_rig("bullish", "bearish", 100, 110, 102, 0.20, 48, "neutral")
        r_with_bonus = evaluate_rig("bullish", "bearish", 100, 110, 102, 0.30, 48, "neutral")
        assert r_with_bonus["confidence_modifier"] == pytest.approx(
            r_no_bonus["confidence_modifier"] + 0.1
        )

    def test_conf_mod_session_bonus_adds_0_1(self):
        """Session alignment adds +0.1 for bullish reversal + expansion."""
        from hpb_rig_validator import evaluate_rig
        r_neutral = evaluate_rig("bearish", "bullish", 100, 110, 102, 0.20, 48, "neutral")
        r_expansion = evaluate_rig("bearish", "bullish", 100, 110, 102, 0.20, 48, "expansion")
        assert r_expansion["confidence_modifier"] == pytest.approx(
            r_neutral["confidence_modifier"] + 0.1
        )

    def test_conf_mod_both_bonuses_stack(self):
        """Displacement + session bonuses stack to 0.7."""
        from hpb_rig_validator import evaluate_rig
        r = evaluate_rig("bearish", "bullish", 100, 110, 102, 0.30, 48, "expansion")
        assert r["confidence_modifier"] == pytest.approx(0.7)

    def test_conf_mod_capped_at_0_7(self):
        """Even with all bonuses, conf_mod cannot exceed 0.7."""
        from hpb_rig_validator import evaluate_rig
        r = evaluate_rig("bearish", "bullish", 100, 110, 102, 0.99, 48, "expansion")
        assert r["confidence_modifier"] <= 0.7

    def test_conf_mod_short_duration_penalty(self):
        """Duration < 24h multiplies conf_mod by 0.8."""
        from hpb_rig_validator import evaluate_rig
        r_long = evaluate_rig("bullish", "bearish", 100, 110, 102, 0.20, 48, "neutral")
        r_short = evaluate_rig("bullish", "bearish", 100, 110, 102, 0.20, 12, "neutral")
        assert r_short["confidence_modifier"] == pytest.approx(
            r_long["confidence_modifier"] * 0.8
        )

    def test_block_always_zero(self):
        """BLOCK status always has confidence_modifier == 0.0."""
        from hpb_rig_validator import evaluate_rig
        # Mid-range block
        r1 = evaluate_rig("bullish", "bearish", 100, 110, 105, 0.5, 48)
        assert r1["confidence_modifier"] == 0.0
        # No-displacement block
        r2 = evaluate_rig("bullish", "bearish", 100, 110, 101, 0.1, 48)
        assert r2["confidence_modifier"] == 0.0

    def test_valid_always_one(self):
        """VALID status (trend-aligned) always has confidence_modifier == 1.0."""
        from hpb_rig_validator import evaluate_rig
        for pos in [101, 105, 109]:
            r = evaluate_rig("bullish", "bullish", 100, 110, pos, 0.5, 48)
            assert r["confidence_modifier"] == 1.0


# ═══════════════════════════════════════════════════════════════════
# Displacement 0.0 fix validation
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestDisplacementZeroFix:
    """Validate displacement 0.0 recomputation from range midpoint."""

    def test_zero_disp_at_high_recomputes(self):
        """disp=0.0, price near high → recomputed disp is distance from mid."""
        from hpb_rig_validator import evaluate_rig
        # range=[100,110], mid=105, price=109 → recomputed = |109-105|/10 = 0.4
        r = evaluate_rig("bullish", "bearish", 100, 110, 109, 0.0, 48)
        assert r["rig_status"] == "CONDITIONAL"  # 0.4 >= 0.15

    def test_zero_disp_at_mid_recomputes(self):
        """disp=0.0, price at mid → recomputed disp = 0.0 → BLOCK (no displacement)."""
        from hpb_rig_validator import evaluate_rig
        # range=[100,110], mid=105, price=105 → recomputed = 0.0
        r = evaluate_rig("bullish", "bearish", 100, 110, 105, 0.0, 48)
        # position=0.5 → MID zone → BLOCK
        assert r["rig_status"] == "BLOCK"

    def test_zero_disp_at_low_recomputes(self):
        """disp=0.0, price near low → recomputed disp is distance from mid."""
        from hpb_rig_validator import evaluate_rig
        # range=[100,110], mid=105, price=101 → recomputed = |101-105|/10 = 0.4
        r = evaluate_rig("bullish", "bearish", 100, 110, 101, 0.0, 48)
        # position=0.1 → LOW zone, disp 0.4 >= 0.15 → CONDITIONAL
        assert r["rig_status"] == "CONDITIONAL"

    def test_nonzero_disp_not_recomputed(self):
        """disp=0.10 (nonzero) is used as-is, not recomputed."""
        from hpb_rig_validator import evaluate_rig
        r = evaluate_rig("bullish", "bearish", 100, 110, 101, 0.10, 48)
        # disp 0.10 < 0.15 → BLOCK (would be CONDITIONAL if recomputed)
        assert r["rig_status"] == "BLOCK"

    def test_degenerate_range_zero_disp(self):
        """Degenerate range (high==low) + disp=0.0 → defaults to neutral."""
        from hpb_rig_validator import evaluate_rig
        r = evaluate_rig("bullish", "bearish", 100, 100, 100, 0.0, 48)
        # Degenerate range → position 0.5, disp stays 0.0 (range_size=0)
        # MID zone, counter-bias → BLOCK
        assert r["rig_status"] == "BLOCK"


# ═══════════════════════════════════════════════════════════════════
# compute_displacement tests (unchanged)
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestComputeDisplacement:
    """Tests for the compute_displacement helper function"""

    def test_mid_range(self):
        from hpb_rig_validator import compute_displacement
        assert compute_displacement(105.0, 110.0, 100.0) == 0.5

    def test_at_range_low(self):
        from hpb_rig_validator import compute_displacement
        assert compute_displacement(100.0, 110.0, 100.0) == 0.0

    def test_at_range_high(self):
        from hpb_rig_validator import compute_displacement
        assert compute_displacement(110.0, 110.0, 100.0) == 1.0

    def test_below_range_clamps_to_zero(self):
        from hpb_rig_validator import compute_displacement
        assert compute_displacement(90.0, 110.0, 100.0) == 0.0

    def test_above_range_clamps_to_one(self):
        from hpb_rig_validator import compute_displacement
        assert compute_displacement(120.0, 110.0, 100.0) == 1.0

    def test_degenerate_range_returns_none(self):
        from hpb_rig_validator import compute_displacement
        assert compute_displacement(100.0, 100.0, 100.0) is None

    def test_missing_price_returns_none(self):
        from hpb_rig_validator import compute_displacement
        assert compute_displacement(None, 110.0, 100.0) is None

    def test_missing_range_high_returns_none(self):
        from hpb_rig_validator import compute_displacement
        assert compute_displacement(105.0, None, 100.0) is None

    def test_missing_range_low_returns_none(self):
        from hpb_rig_validator import compute_displacement
        assert compute_displacement(105.0, 110.0, None) is None

    def test_quarter_displacement(self):
        from hpb_rig_validator import compute_displacement
        result = compute_displacement(102.5, 110.0, 100.0)
        assert abs(result - 0.25) < 1e-10

    def test_three_quarter_displacement(self):
        from hpb_rig_validator import compute_displacement
        result = compute_displacement(107.5, 110.0, 100.0)
        assert abs(result - 0.75) < 1e-10


@pytest.mark.unit
class TestComputeDisplacementEdgeCases:

    def test_inverted_range_returns_none(self):
        from hpb_rig_validator import compute_displacement
        assert compute_displacement(105.0, 100.0, 110.0) is None

    def test_slightly_inverted_range_returns_none(self):
        from hpb_rig_validator import compute_displacement
        assert compute_displacement(100.0, 99.99, 100.0) is None
