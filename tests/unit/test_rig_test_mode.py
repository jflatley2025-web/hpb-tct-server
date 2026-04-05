"""
Unit tests for rig_test_mode.py
Tests the temporary RIG test mode for displacement validation.
"""
import os
import pytest
from unittest.mock import patch
from rig_test_mode import (
    compute_extremity,
    evaluate_rig_test_status,
    build_test_context,
    run_single_scenario,
    run_all_scenarios,
    TEST_SCENARIOS,
    TEST_RANGE_HIGH,
    TEST_RANGE_LOW,
    TEST_RANGE_DURATION,
)


@pytest.mark.unit
class TestComputeExtremity:
    """Tests for the extremity metric (distance from mid-range)."""

    def test_mid_range_is_zero(self):
        """Displacement 0.5 (dead center) → extremity 0.0"""
        assert compute_extremity(0.5) == 0.0

    def test_range_high_is_one(self):
        """Displacement 1.0 (at range high) → extremity 1.0"""
        assert compute_extremity(1.0) == 1.0

    def test_range_low_is_one(self):
        """Displacement 0.0 (at range low) → extremity 1.0"""
        assert compute_extremity(0.0) == 1.0

    def test_quarter_displacement(self):
        """Displacement 0.25 → extremity 0.5"""
        assert abs(compute_extremity(0.25) - 0.5) < 1e-10

    def test_three_quarter_displacement(self):
        """Displacement 0.75 → extremity 0.5"""
        assert abs(compute_extremity(0.75) - 0.5) < 1e-10

    def test_none_returns_none(self):
        """None displacement → None"""
        assert compute_extremity(None) is None


@pytest.mark.unit
class TestEvaluateRigTestStatus:
    """Tests: displacement alone never blocks — always VALID."""

    def test_mid_range_valid(self):
        """Displacement ~0.5 (mid-range) → VALID (no longer blocks)"""
        assert evaluate_rig_test_status(0.5) == "VALID"

    def test_high_extreme_valid(self):
        """Displacement 0.875 (near range high) → VALID"""
        assert evaluate_rig_test_status(0.875) == "VALID"

    def test_low_extreme_valid(self):
        """Displacement 0.125 (near range low) → VALID"""
        assert evaluate_rig_test_status(0.125) == "VALID"

    def test_at_upper_boundary_valid(self):
        """Displacement 0.76 → VALID"""
        assert evaluate_rig_test_status(0.76) == "VALID"

    def test_at_lower_boundary_valid(self):
        """Displacement 0.24 → VALID"""
        assert evaluate_rig_test_status(0.24) == "VALID"

    def test_at_mid_range_upper_edge_valid(self):
        """Displacement 0.74 → VALID (no displacement-based blocking)"""
        assert evaluate_rig_test_status(0.74) == "VALID"

    def test_at_mid_range_lower_edge_valid(self):
        """Displacement 0.26 → VALID (no displacement-based blocking)"""
        assert evaluate_rig_test_status(0.26) == "VALID"

    def test_none_not_evaluated(self):
        """None displacement → NOT_EVALUATED"""
        assert evaluate_rig_test_status(None) == "NOT_EVALUATED"

    def test_exact_boundary_0_25(self):
        """Displacement exactly 0.25 → VALID"""
        assert evaluate_rig_test_status(0.25) == "VALID"

    def test_exact_boundary_0_75(self):
        """Displacement exactly 0.75 → VALID"""
        assert evaluate_rig_test_status(0.75) == "VALID"


@pytest.mark.unit
class TestBuildTestContext:
    """Tests for context building with injected values."""

    def test_default_range_params(self):
        """Uses TEST_RANGE_* defaults when no overrides provided"""
        context, _ = build_test_context(70000)
        rcm = context["gates"]["RCM"]
        assert rcm["range_high"] == TEST_RANGE_HIGH
        assert rcm["range_low"] == TEST_RANGE_LOW
        assert rcm["range_duration_hours"] == TEST_RANGE_DURATION
        assert rcm["valid"] is True

    def test_custom_range_params(self):
        """Uses custom range params when provided"""
        context, _ = build_test_context(
            50000, range_high=55000, range_low=45000, range_duration=72
        )
        rcm = context["gates"]["RCM"]
        assert rcm["range_high"] == 55000
        assert rcm["range_low"] == 45000
        assert rcm["range_duration_hours"] == 72

    def test_displacement_calculation(self):
        """Displacement is correctly computed from injected values"""
        context, displacement = build_test_context(70000)
        # (70000 - 68000) / (72000 - 68000) = 0.5
        assert abs(displacement - 0.5) < 1e-10

    def test_displacement_set_in_context(self):
        """Displacement is injected into context for RIG validator"""
        context, displacement = build_test_context(70000)
        assert context["local_range_displacement"] == displacement

    def test_displacement_is_not_none(self):
        """Valid inputs must produce non-None displacement in context"""
        context, _ = build_test_context(70000)
        assert context["local_range_displacement"] is not None

    def test_counter_bias_default(self):
        """Default setup has counter-bias (htf=bullish, session=bearish)"""
        context, _ = build_test_context(70000)
        assert context["gates"]["1A"]["bias"] == "bullish"
        assert context["gates"]["MSCE"]["session_bias"] == "bearish"

    def test_custom_bias(self):
        """Custom bias values are applied"""
        context, _ = build_test_context(
            70000, htf_bias="bearish", session_bias="bullish"
        )
        assert context["gates"]["1A"]["bias"] == "bearish"
        assert context["gates"]["MSCE"]["session_bias"] == "bullish"

    def test_invalid_range_raises(self):
        """Degenerate range (high == low) raises ValueError"""
        with pytest.raises(ValueError):
            build_test_context(70000, range_high=68000, range_low=68000)

    def test_invalid_duration_raises(self):
        """Zero duration raises ValueError"""
        with pytest.raises(ValueError):
            build_test_context(70000, range_duration=0)


@pytest.mark.unit
class TestRunSingleScenario:
    """Tests for running individual test scenarios."""

    def test_mid_range_scenario(self):
        """Mid-range scenario produces correct displacement and VALID status"""
        scenario = {"name": "mid_range", "current_price": 70000, "expected_displacement": 0.5}
        result = run_single_scenario(scenario)

        assert result["scenario"] == "mid_range"
        assert result["price"] == 70000
        assert abs(result["displacement"] - 0.5) < 1e-10
        assert result["rig_status"] == "VALID"

    def test_high_extreme_scenario(self):
        """High extreme scenario produces correct displacement and VALID status"""
        scenario = {"name": "high_extreme", "current_price": 71500, "expected_displacement": 0.875}
        result = run_single_scenario(scenario)

        assert result["scenario"] == "high_extreme"
        assert result["price"] == 71500
        assert abs(result["displacement"] - 0.875) < 1e-10
        assert result["rig_status"] == "VALID"

    def test_low_extreme_scenario(self):
        """Low extreme scenario produces correct displacement and VALID status"""
        scenario = {"name": "low_extreme", "current_price": 68500, "expected_displacement": 0.125}
        result = run_single_scenario(scenario)

        assert result["scenario"] == "low_extreme"
        assert result["price"] == 68500
        assert abs(result["displacement"] - 0.125) < 1e-10
        assert result["rig_status"] == "VALID"

    def test_result_includes_production_status(self):
        """Result includes the production RIG validator output for comparison"""
        scenario = {"name": "test", "current_price": 70000, "expected_displacement": 0.5}
        result = run_single_scenario(scenario)

        assert "production_rig_status" in result
        assert result["production_rig_status"] in ("VALID", "BLOCK")

    def test_result_includes_extremity(self):
        """Result includes the extremity metric"""
        scenario = {"name": "test", "current_price": 70000, "expected_displacement": 0.5}
        result = run_single_scenario(scenario)

        assert "extremity" in result
        assert abs(result["extremity"] - 0.0) < 1e-10  # mid-range → extremity 0

    def test_not_evaluable_displacement(self):
        """Degenerate range produces NOT_EVALUABLE without running RIG"""
        scenario = {"name": "invalid", "current_price": 70000}

        result = run_single_scenario(
            scenario,
            range_high=68000,
            range_low=68000,
        )

        assert result["displacement"] is None
        assert result["rig_status"] == "NOT_EVALUABLE"


@pytest.mark.unit
class TestRunAllScenarios:
    """Tests for the full scenario runner."""

    def test_returns_empty_when_disabled(self):
        """Returns empty list when RIG_TEST_MODE is not enabled"""
        with patch("rig_test_mode.RIG_TEST_MODE", False):
            results = run_all_scenarios()
            assert results == []

    def test_runs_all_scenarios_when_enabled(self):
        """Runs all predefined scenarios when enabled"""
        with patch("rig_test_mode.RIG_TEST_MODE", True):
            results = run_all_scenarios()
            assert len(results) == len(TEST_SCENARIOS)

    def test_scenario_names_match(self):
        """All predefined scenario names appear in results"""
        with patch("rig_test_mode.RIG_TEST_MODE", True):
            results = run_all_scenarios()
            names = {r["scenario"] for r in results}
            expected = {s["name"] for s in TEST_SCENARIOS}
            assert names == expected

    def test_all_displacements_valid(self):
        """All scenarios produce non-None displacement values"""
        with patch("rig_test_mode.RIG_TEST_MODE", True):
            results = run_all_scenarios()
            for r in results:
                assert r["displacement"] is not None

    def test_expected_statuses(self):
        """All displacement scenarios return VALID (no displacement-based blocking)"""
        with patch("rig_test_mode.RIG_TEST_MODE", True):
            results = run_all_scenarios()
            by_name = {r["scenario"]: r for r in results}

            assert by_name["mid_range"]["rig_status"] == "VALID"
            assert by_name["high_extreme"]["rig_status"] == "VALID"
            assert by_name["low_extreme"]["rig_status"] == "VALID"

    def test_custom_range_override(self):
        """Custom range parameters are applied to all scenarios"""
        with patch("rig_test_mode.RIG_TEST_MODE", True):
            results = run_all_scenarios(
                range_high=80000, range_low=60000, range_duration=72
            )
            assert len(results) == len(TEST_SCENARIOS)
            # Mid-range: (70000-60000)/(80000-60000) = 0.5 → VALID (confidence penalized, not blocked)
            mid = next(r for r in results if r["scenario"] == "mid_range")
            assert abs(mid["displacement"] - 0.5) < 1e-10


@pytest.mark.unit
class TestTestModeFlag:
    """Tests for the RIG_TEST_MODE environment variable flag."""

    def test_default_is_disabled(self):
        """Test mode is OFF by default"""
        with patch.dict(os.environ, {}, clear=True):
            # Re-import to pick up env var
            import importlib
            import rig_test_mode
            importlib.reload(rig_test_mode)
            assert rig_test_mode.RIG_TEST_MODE is False

    def test_enabled_when_true(self):
        """Test mode is ON when RIG_TEST_MODE=true"""
        with patch.dict(os.environ, {"RIG_TEST_MODE": "true"}):
            import importlib
            import rig_test_mode
            importlib.reload(rig_test_mode)
            assert rig_test_mode.RIG_TEST_MODE is True

    def test_enabled_case_insensitive(self):
        """Test mode accepts TRUE, True, etc."""
        with patch.dict(os.environ, {"RIG_TEST_MODE": "TRUE"}):
            import importlib
            import rig_test_mode
            importlib.reload(rig_test_mode)
            assert rig_test_mode.RIG_TEST_MODE is True

    def test_disabled_with_false(self):
        """Test mode is OFF when RIG_TEST_MODE=false"""
        with patch.dict(os.environ, {"RIG_TEST_MODE": "false"}):
            import importlib
            import rig_test_mode
            importlib.reload(rig_test_mode)
            assert rig_test_mode.RIG_TEST_MODE is False
