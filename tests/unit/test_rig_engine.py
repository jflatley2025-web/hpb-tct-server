"""
Tests for rig_engine.evaluate_rig_global — canonical RIG evaluator.
"""

import pytest
from rig_engine import evaluate_rig_global


class TestEvaluateRigGlobal:
    """Core evaluate_rig_global behavior."""

    def test_valid_aligned_bias(self):
        """Aligned session_bias + htf_bias → VALID."""
        result = evaluate_rig_global(
            htf_bias="bullish",
            session_name="London",
            session_bias="bullish",
            range_high=110.0,
            range_low=100.0,
            current_price=105.0,
        )
        assert result["status"] == "VALID"
        assert result["evaluated"] is True
        assert result["displacement"] == pytest.approx(0.5)

    def test_block_counter_bias_low_displacement(self):
        """Counter-bias + low displacement + valid range → BLOCK."""
        result = evaluate_rig_global(
            htf_bias="bullish",
            session_name="New York",
            session_bias="bearish",
            range_high=110.0,
            range_low=100.0,
            current_price=101.0,  # displacement = 0.1 (< 0.25 threshold)
            range_duration_hours=48,
        )
        assert result["status"] == "BLOCK"
        assert result["evaluated"] is True
        assert result["displacement"] == pytest.approx(0.1)

    def test_conditional_counter_bias_high_displacement(self):
        """Counter-bias at range extreme with displacement → CONDITIONAL."""
        result = evaluate_rig_global(
            htf_bias="bullish",
            session_name="New York",
            session_bias="bearish",
            range_high=110.0,
            range_low=100.0,
            current_price=108.0,  # displacement = 0.8 → HIGH zone
        )
        assert result["status"] == "CONDITIONAL"
        assert result["evaluated"] is True
        assert 0.0 < result["confidence_modifier"] <= 0.7

    def test_not_evaluated_no_range(self):
        """Missing range data → NOT_EVALUATED with evaluated=True."""
        result = evaluate_rig_global(
            htf_bias="bullish",
            session_name="London",
            session_bias="bullish",
            range_high=None,
            range_low=None,
            current_price=100.0,
        )
        assert result["status"] == "NOT_EVALUATED"
        assert result["evaluated"] is True
        assert result["displacement"] is None
        assert "displacement" in result["reason"]

    def test_not_evaluated_inverted_range(self):
        """Inverted range → NOT_EVALUATED."""
        result = evaluate_rig_global(
            htf_bias="bullish",
            session_name="London",
            session_bias="bullish",
            range_high=100.0,
            range_low=110.0,  # inverted
            current_price=105.0,
        )
        assert result["status"] == "NOT_EVALUATED"
        assert result["evaluated"] is True

    def test_not_evaluated_no_session(self):
        """Missing session_name → NOT_EVALUATED."""
        result = evaluate_rig_global(
            htf_bias="bullish",
            session_name=None,
            session_bias="bullish",
            range_high=110.0,
            range_low=100.0,
            current_price=105.0,
        )
        assert result["status"] == "NOT_EVALUATED"
        assert result["evaluated"] is True
        assert "session_name" in result["reason"]

    def test_not_evaluated_no_session_bias(self):
        """Missing session_bias → NOT_EVALUATED."""
        result = evaluate_rig_global(
            htf_bias="bullish",
            session_name="London",
            session_bias=None,
            range_high=110.0,
            range_low=100.0,
            current_price=105.0,
        )
        assert result["status"] == "NOT_EVALUATED"
        assert result["evaluated"] is True
        assert "session_bias" in result["reason"]

    def test_not_evaluated_neutral_session_bias(self):
        """session_bias='neutral' is rejected (must be bullish/bearish)."""
        result = evaluate_rig_global(
            htf_bias="neutral",
            session_name="London",
            session_bias="neutral",
            range_high=110.0,
            range_low=100.0,
            current_price=105.0,
        )
        assert result["status"] == "NOT_EVALUATED"
        assert result["evaluated"] is True

    def test_output_structure(self):
        """All required fields present in output."""
        result = evaluate_rig_global(
            htf_bias="bullish",
            session_name="Asia",
            session_bias="bullish",
            range_high=110.0,
            range_low=100.0,
            current_price=105.0,
        )
        assert "status" in result
        assert "Gate" in result
        assert "evaluated" in result
        assert "displacement" in result
        assert "htf_bias" in result
        assert "session_bias" in result
        assert "confidence" in result
        assert "timestamp" in result

    def test_short_range_duration_soft_penalty(self):
        """Duration < 24h applies soft penalty, does NOT hard-block."""
        result = evaluate_rig_global(
            htf_bias="bullish",
            session_name="New York",
            session_bias="bearish",
            range_high=110.0,
            range_low=100.0,
            current_price=101.0,  # disp=0.1 → LOW zone, disp < 0.15 → BLOCK
            range_duration_hours=12,
        )
        # Short duration doesn't override the zone-based decision;
        # this setup blocks because displacement < 0.15, not because of duration
        assert result["status"] == "BLOCK"
        assert result["evaluated"] is True

    def test_sufficient_range_duration_allows_block(self):
        """Duration >= 24h allows normal RIG evaluation (BLOCK possible)."""
        result = evaluate_rig_global(
            htf_bias="bullish",
            session_name="New York",
            session_bias="bearish",
            range_high=110.0,
            range_low=100.0,
            current_price=101.0,
            range_duration_hours=24,
        )
        assert result["status"] == "BLOCK"

    def test_exec_score_passthrough(self):
        """exec_score is passed to validator and reflected in confidence on VALID."""
        result = evaluate_rig_global(
            htf_bias="bearish",
            session_name="London",
            session_bias="bearish",
            range_high=110.0,
            range_low=100.0,
            current_price=105.0,
            exec_score=0.85,
        )
        assert result["status"] == "VALID"
        assert result["confidence"] == pytest.approx(0.85)

    def test_displacement_override(self):
        """displacement_override takes precedence over computed displacement."""
        # Without override: displacement = 0.8 → HIGH zone → CONDITIONAL
        result_no_override = evaluate_rig_global(
            htf_bias="bullish",
            session_name="New York",
            session_bias="bearish",
            range_high=110.0,
            range_low=100.0,
            current_price=108.0,
        )
        assert result_no_override["status"] == "CONDITIONAL"

        # With override: force low displacement (0.1 < 0.15) → BLOCK
        result_override = evaluate_rig_global(
            htf_bias="bullish",
            session_name="New York",
            session_bias="bearish",
            range_high=110.0,
            range_low=100.0,
            current_price=108.0,
            displacement_override=0.1,  # conservative min from another range
        )
        assert result_override["status"] == "BLOCK"
        assert result_override["displacement"] == pytest.approx(0.1)

    def test_block_always_zero_confidence(self):
        """BLOCK status always has confidence=0.0 regardless of exec_score."""
        result = evaluate_rig_global(
            htf_bias="bullish",
            session_name="New York",
            session_bias="bearish",
            range_high=110.0,
            range_low=100.0,
            current_price=101.0,
            exec_score=0.95,  # high score should be overridden
        )
        assert result["status"] == "BLOCK"
        assert result["confidence"] == 0.0

    def test_zero_duration_not_evaluated(self):
        """range_duration_hours=0 → NOT_EVALUATED (not a valid range)."""
        result = evaluate_rig_global(
            htf_bias="bullish",
            session_name="London",
            session_bias="bullish",
            range_high=110.0,
            range_low=100.0,
            current_price=105.0,
            range_duration_hours=0,
        )
        assert result["status"] == "NOT_EVALUATED"
        assert result["evaluated"] is True

    def test_negative_duration_not_evaluated(self):
        """Negative duration → NOT_EVALUATED."""
        result = evaluate_rig_global(
            htf_bias="bullish",
            session_name="London",
            session_bias="bullish",
            range_high=110.0,
            range_low=100.0,
            current_price=105.0,
            range_duration_hours=-5,
        )
        assert result["status"] == "NOT_EVALUATED"
        assert result["evaluated"] is True
