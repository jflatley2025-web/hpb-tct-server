"""
Unit tests for hpb_rig_validator.py
Tests the Range Integrity Gate (RIG) validation logic
"""
import pytest
from datetime import datetime
from hpb_rig_validator import range_integrity_validator


@pytest.mark.unit
class TestRangeIntegrityValidator:
    """Tests for RIG validator function"""

    def test_rig_allows_aligned_bias(self, mock_context_basic):
        """Test RIG allows trade when session bias aligns with HTF bias"""
        result = range_integrity_validator(mock_context_basic)

        assert result["status"] == "VALID"
        assert result["Gate"] == "RIG"
        assert result["reason"] is None
        assert result["confidence"] > 0.0
        assert result["htf_bias"] == "bullish"
        assert result["session_bias"] == "bullish"

    def test_rig_blocks_counter_bias(self, mock_context_counter_bias):
        """Test RIG blocks trade when session bias counters HTF bias"""
        result = range_integrity_validator(mock_context_counter_bias)

        assert result["status"] == "BLOCK"
        assert result["Gate"] == "RIG"
        assert result["reason"] is not None
        assert "Counter-bias" in result["reason"]
        assert result["confidence"] == 0.0
        assert result["htf_bias"] == "bullish"
        assert result["session_bias"] == "bearish"

    def test_rig_allows_invalid_range(self, mock_context_invalid_range):
        """Test RIG allows trade when range is already broken"""
        result = range_integrity_validator(mock_context_invalid_range)

        # Should NOT block because range is invalid (not intact)
        assert result["status"] == "VALID"
        assert result["Gate"] == "RIG"

    def test_rig_allows_short_duration_range(self):
        """Test RIG allows trade when range duration is too short"""
        context = {
            "gates": {
                "1A": {"bias": "bullish", "score": 0.85},
                "RCM": {"valid": True, "range_duration_hours": 12},  # Less than MIN_DURATION (24)
                "MSCE": {"session_bias": "bearish", "session": "London"},
                "1D": {"score": 0.80}
            },
            "local_range_displacement": 0.15,
            "ExecutionConfidence_Total": 0.75
        }

        result = range_integrity_validator(context)

        # Should NOT block because range duration < 24 hours
        assert result["status"] == "VALID"

    def test_rig_allows_high_displacement(self):
        """Test RIG allows trade when displacement is high (range breaking)"""
        context = {
            "gates": {
                "1A": {"bias": "bullish", "score": 0.85},
                "RCM": {"valid": True, "range_duration_hours": 48},
                "MSCE": {"session_bias": "bearish", "session": "London"},
                "1D": {"score": 0.80}
            },
            "local_range_displacement": 0.35,  # Above DISP_THRESHOLD (0.25)
            "ExecutionConfidence_Total": 0.75
        }

        result = range_integrity_validator(context)

        # Should NOT block because displacement > 25% (range is breaking)
        assert result["status"] == "VALID"

    def test_rig_blocks_only_when_all_conditions_met(self):
        """Test RIG blocks only when ALL blocking conditions are met"""
        # ALL conditions for blocking:
        # 1. range_valid = True
        # 2. range_duration >= 24 hours
        # 3. local_disp < 0.25
        # 4. session_bias != htf_bias

        context = {
            "gates": {
                "1A": {"bias": "bullish", "score": 0.85},
                "RCM": {"valid": True, "range_duration_hours": 48},
                "MSCE": {"session_bias": "bearish", "session": "New York"},
                "1D": {"score": 0.75}
            },
            "local_range_displacement": 0.10,  # < 0.25
            "ExecutionConfidence_Total": 0.70
        }

        result = range_integrity_validator(context)

        assert result["status"] == "BLOCK"
        assert result["confidence"] == 0.0

    def test_rig_result_structure(self, mock_context_basic):
        """Test RIG result has correct structure"""
        result = range_integrity_validator(mock_context_basic)

        # Check all required fields exist
        assert "timestamp" in result
        assert "status" in result
        assert "Gate" in result
        assert "reason" in result
        assert "confidence" in result
        assert "htf_bias" in result
        assert "session_bias" in result

        # Validate timestamp format
        try:
            datetime.strptime(result["timestamp"], "%Y-%m-%d %H:%M:%S")
        except ValueError:
            pytest.fail("Timestamp format is invalid")

    def test_rig_handles_missing_gates(self):
        """Test RIG handles missing gate data gracefully"""
        context = {
            "gates": {},
            "local_range_displacement": 0.15,
            "ExecutionConfidence_Total": 0.75
        }

        result = range_integrity_validator(context)

        # Should not crash, should return VALID with defaults
        assert result["status"] == "VALID"
        assert result["Gate"] == "RIG"

    def test_rig_handles_partial_gates(self):
        """Test RIG handles partial gate data"""
        context = {
            "gates": {
                "1A": {"bias": "bullish"},
                # Missing RCM, MSCE, 1D
            },
            "local_range_displacement": 0.15,
        }

        result = range_integrity_validator(context)

        # Should not crash
        assert "status" in result
        assert result["Gate"] == "RIG"

    def test_rig_neutral_htf_bias(self):
        """Test RIG behavior with neutral HTF bias"""
        context = {
            "gates": {
                "1A": {"bias": "neutral", "score": 0.70},
                "RCM": {"valid": True, "range_duration_hours": 48},
                "MSCE": {"session_bias": "bullish", "session": "London"},
                "1D": {"score": 0.75}
            },
            "local_range_displacement": 0.15,
            "ExecutionConfidence_Total": 0.70
        }

        result = range_integrity_validator(context)

        # Neutral bias should still check session_bias != htf_bias
        assert result["status"] == "BLOCK"
        assert result["htf_bias"] == "neutral"

    def test_rig_case_sensitivity(self):
        """Test RIG handles bias string case variations"""
        context = {
            "gates": {
                "1A": {"bias": "Bullish", "score": 0.85},  # Capitalized
                "RCM": {"valid": True, "range_duration_hours": 48},
                "MSCE": {"session_bias": "bullish", "session": "London"},
                "1D": {"score": 0.80}
            },
            "local_range_displacement": 0.15,
            "ExecutionConfidence_Total": 0.75
        }

        result = range_integrity_validator(context)

        # Should handle different cases
        assert "status" in result
        assert result["htf_bias"] == "Bullish"


@pytest.mark.unit
class TestRIGThresholds:
    """Test RIG threshold constants"""

    def test_min_duration_threshold(self):
        """Test MIN_DURATION threshold at boundary"""
        # Exactly at threshold
        context = {
            "gates": {
                "1A": {"bias": "bullish"},
                "RCM": {"valid": True, "range_duration_hours": 24},  # Exactly MIN_DURATION
                "MSCE": {"session_bias": "bearish", "session": "London"},
                "1D": {"score": 0.80}
            },
            "local_range_displacement": 0.15,
            "ExecutionConfidence_Total": 0.75
        }

        result = range_integrity_validator(context)

        # At 24 hours, should block (>= 24)
        assert result["status"] == "BLOCK"

    def test_displacement_threshold(self):
        """Test DISP_THRESHOLD at boundary"""
        # Exactly at threshold
        context = {
            "gates": {
                "1A": {"bias": "bullish"},
                "RCM": {"valid": True, "range_duration_hours": 48},
                "MSCE": {"session_bias": "bearish", "session": "London"},
                "1D": {"score": 0.80}
            },
            "local_range_displacement": 0.25,  # Exactly at DISP_THRESHOLD
            "ExecutionConfidence_Total": 0.75
        }

        result = range_integrity_validator(context)

        # At 0.25, should NOT block (< 0.25 required for blocking)
        assert result["status"] == "VALID"


@pytest.mark.unit
class TestRIGSessionNames:
    """Test RIG with different session names"""

    @pytest.mark.parametrize("session_name", [
        "London", "New York", "Tokyo", "Sydney", "Unknown"
    ])
    def test_rig_different_sessions(self, session_name):
        """Test RIG works with different session names"""
        context = {
            "gates": {
                "1A": {"bias": "bullish"},
                "RCM": {"valid": True, "range_duration_hours": 48},
                "MSCE": {"session_bias": "bearish", "session": session_name},
                "1D": {"score": 0.80}
            },
            "local_range_displacement": 0.15,
            "ExecutionConfidence_Total": 0.75
        }

        result = range_integrity_validator(context)

        # Should block regardless of session name if conditions met
        assert result["status"] == "BLOCK"
        assert session_name in result["reason"]


@pytest.mark.unit
class TestComputeDisplacement:
    """Tests for the compute_displacement helper function"""

    def test_mid_range(self):
        """Price at midpoint → displacement = 0.5"""
        from hpb_rig_validator import compute_displacement
        assert compute_displacement(105.0, 110.0, 100.0) == 0.5

    def test_at_range_low(self):
        """Price at range low → displacement = 0.0"""
        from hpb_rig_validator import compute_displacement
        assert compute_displacement(100.0, 110.0, 100.0) == 0.0

    def test_at_range_high(self):
        """Price at range high → displacement = 1.0"""
        from hpb_rig_validator import compute_displacement
        assert compute_displacement(110.0, 110.0, 100.0) == 1.0

    def test_below_range_clamps_to_zero(self):
        """Price below range → clamps to 0.0"""
        from hpb_rig_validator import compute_displacement
        assert compute_displacement(90.0, 110.0, 100.0) == 0.0

    def test_above_range_clamps_to_one(self):
        """Price above range → clamps to 1.0"""
        from hpb_rig_validator import compute_displacement
        assert compute_displacement(120.0, 110.0, 100.0) == 1.0

    def test_degenerate_range_returns_none(self):
        """range_high == range_low → None"""
        from hpb_rig_validator import compute_displacement
        assert compute_displacement(100.0, 100.0, 100.0) is None

    def test_missing_price_returns_none(self):
        """Missing current_price → None"""
        from hpb_rig_validator import compute_displacement
        assert compute_displacement(None, 110.0, 100.0) is None

    def test_missing_range_high_returns_none(self):
        """Missing range_high → None"""
        from hpb_rig_validator import compute_displacement
        assert compute_displacement(105.0, None, 100.0) is None

    def test_missing_range_low_returns_none(self):
        """Missing range_low → None"""
        from hpb_rig_validator import compute_displacement
        assert compute_displacement(105.0, 110.0, None) is None

    def test_quarter_displacement(self):
        """Price at 25% of range → displacement = 0.25"""
        from hpb_rig_validator import compute_displacement
        result = compute_displacement(102.5, 110.0, 100.0)
        assert abs(result - 0.25) < 1e-10

    def test_three_quarter_displacement(self):
        """Price at 75% of range → displacement = 0.75"""
        from hpb_rig_validator import compute_displacement
        result = compute_displacement(107.5, 110.0, 100.0)
        assert abs(result - 0.75) < 1e-10


@pytest.mark.unit
class TestRIGWithDisplacement:
    """Integration tests: RIG evaluation with displacement scenarios"""

    def test_mid_range_blocks_counter_bias(self):
        """Mid-range (0.4-0.6) with counter-bias → BLOCK (displacement < 0.25)"""
        context = {
            "gates": {
                "1A": {"bias": "bullish"},
                "RCM": {"valid": True, "range_duration_hours": 48},
                "MSCE": {"session_bias": "bearish", "session": "London"},
                "1D": {"score": 0.80},
            },
            "local_range_displacement": 0.15,  # Below 0.25 threshold → blocks
        }
        result = range_integrity_validator(context)
        assert result["status"] == "BLOCK"

    def test_range_high_allows_short(self):
        """Price at range high (>0.75) → VALID (displacement above threshold)"""
        context = {
            "gates": {
                "1A": {"bias": "bullish"},
                "RCM": {"valid": True, "range_duration_hours": 48},
                "MSCE": {"session_bias": "bearish", "session": "London"},
                "1D": {"score": 0.80},
            },
            "local_range_displacement": 0.80,  # Above 0.25 threshold
        }
        result = range_integrity_validator(context)
        assert result["status"] == "VALID"

    def test_range_low_allows_long(self):
        """Price at range low (<0.25) but aligned bias → VALID"""
        context = {
            "gates": {
                "1A": {"bias": "bullish"},
                "RCM": {"valid": True, "range_duration_hours": 48},
                "MSCE": {"session_bias": "bullish", "session": "London"},
                "1D": {"score": 0.80},
            },
            "local_range_displacement": 0.10,  # Low displacement but aligned
        }
        result = range_integrity_validator(context)
        assert result["status"] == "VALID"

    def test_true_mid_range_allows_when_above_threshold(self):
        """True mid-range (0.5) is above DISP_THRESHOLD → VALID (not blocked)"""
        context = {
            "gates": {
                "1A": {"bias": "bullish"},
                "RCM": {"valid": True, "range_duration_hours": 48},
                "MSCE": {"session_bias": "bearish", "session": "London"},
                "1D": {"score": 0.80},
            },
            "local_range_displacement": 0.50,  # Mid-range but above 0.25 threshold
        }
        result = range_integrity_validator(context)
        # 0.50 > 0.25 threshold so RIG does NOT block
        assert result["status"] == "VALID"


@pytest.mark.unit
class TestComputeDisplacementEdgeCases:
    """Additional edge case tests for compute_displacement"""

    def test_inverted_range_returns_none(self):
        """range_high < range_low (inverted) → None"""
        from hpb_rig_validator import compute_displacement
        assert compute_displacement(105.0, 100.0, 110.0) is None

    def test_slightly_inverted_range_returns_none(self):
        """range_high barely below range_low → None"""
        from hpb_rig_validator import compute_displacement
        assert compute_displacement(100.0, 99.99, 100.0) is None
