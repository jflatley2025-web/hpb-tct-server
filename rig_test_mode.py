# rig_test_mode.py
"""
Temporary RIG Test Mode for Displacement Validation

Allows controlled injection of range parameters and price to simulate
displacement scenarios and validate mid-range BLOCK vs extreme VALID behavior.

IMPORTANT:
- ONLY active when RIG_TEST_MODE=true (env var or flag)
- DO NOT deploy to production enabled
- DO NOT modify real execution logic
"""
import os
import logging
from hpb_rig_validator import compute_displacement, range_integrity_validator

logger = logging.getLogger(__name__)

# Test mode flag — read from environment, default OFF
RIG_TEST_MODE = os.environ.get("RIG_TEST_MODE", "false").lower() == "true"

# --- Injected test range parameters (only used when RIG_TEST_MODE=True) ---
TEST_RANGE_HIGH = 72000
TEST_RANGE_LOW = 68000
TEST_RANGE_DURATION = 48  # hours

# Extremity threshold: how far from center price must be to count as "extreme"
# displacement > 0.75 or displacement < 0.25 → extreme → VALID
# 0.25 <= displacement <= 0.75 → mid-range → BLOCK
EXTREME_THRESHOLD = 0.25  # distance from center (0.5 ± 0.25)

# --- Test scenarios ---
TEST_SCENARIOS = [
    {"name": "mid_range", "current_price": 70000, "expected_displacement": 0.5},
    {"name": "high_extreme", "current_price": 71500, "expected_displacement": 0.875},
    {"name": "low_extreme", "current_price": 68500, "expected_displacement": 0.125},
]


def is_test_mode_enabled():
    """Check if RIG test mode is active."""
    return RIG_TEST_MODE


def compute_extremity(displacement):
    """Compute how far displacement is from mid-range (0.5).

    Returns a value 0.0–1.0 where:
    - 0.0 = dead center (mid-range)
    - 1.0 = at range boundary (extreme)
    """
    if displacement is None:
        return None
    return abs(displacement - 0.5) * 2


def evaluate_rig_test_status(displacement):
    """Determine RIG status using mid-range BLOCK / extreme VALID rule.

    Mid-range (0.25 <= disp <= 0.75): price hasn't escaped range → BLOCK
    Extreme  (disp > 0.75 or disp < 0.25): price at range edge → VALID
    """
    if displacement is None:
        return "NOT_EVALUATED"
    extremity = compute_extremity(displacement)
    if extremity < (EXTREME_THRESHOLD * 2):  # within ±0.25 of center
        return "BLOCK"
    return "VALID"


def build_test_context(current_price, range_high=None, range_low=None,
                       range_duration=None, htf_bias="bullish",
                       session_bias="bearish"):
    """Build an HPB context with injected test values for RIG evaluation.

    Uses test defaults if range parameters are not provided.
    Sets counter-bias by default (session_bias != htf_bias) to isolate
    displacement as the deciding factor.
    """
    rh = range_high if range_high is not None else TEST_RANGE_HIGH
    rl = range_low if range_low is not None else TEST_RANGE_LOW
    rd = range_duration if range_duration is not None else TEST_RANGE_DURATION

    displacement = compute_displacement(current_price, rh, rl)

    context = {
        "gates": {
            "1A": {"bias": htf_bias, "score": 0.85},
            "RCM": {
                "valid": True,
                "range_duration_hours": rd,
                "range_high": rh,
                "range_low": rl,
            },
            "MSCE": {"session_bias": session_bias, "session": "London"},
            "1D": {"score": 0.80},
        },
        "local_range_displacement": displacement if displacement is not None else 0.0,
    }
    return context, displacement


def run_single_scenario(scenario, range_high=None, range_low=None,
                        range_duration=None):
    """Run a single test scenario and return results."""
    current_price = scenario["current_price"]
    name = scenario["name"]

    context, displacement = build_test_context(
        current_price,
        range_high=range_high,
        range_low=range_low,
        range_duration=range_duration,
    )

    # Run through the production RIG validator for comparison
    production_result = range_integrity_validator(context)

    # Evaluate using the mid-range BLOCK / extreme VALID rule
    test_status = evaluate_rig_test_status(displacement)
    extremity = compute_extremity(displacement)

    result = {
        "scenario": name,
        "price": current_price,
        "displacement": displacement,
        "extremity": extremity,
        "rig_status": test_status,
        "production_rig_status": production_result.get("status"),
        "production_reason": production_result.get("reason"),
    }

    logger.info("RIG TEST: %s", {
        "price": current_price,
        "displacement": displacement,
        "rig_status": test_status,
    })

    return result


def run_all_scenarios(range_high=None, range_low=None, range_duration=None):
    """Run all predefined test scenarios.

    Returns a list of result dicts with displacement and status info.
    Only executes when RIG_TEST_MODE is enabled.
    """
    if not is_test_mode_enabled():
        logger.warning("RIG_TEST_MODE is not enabled. Set RIG_TEST_MODE=true to run.")
        return []

    rh = range_high if range_high is not None else TEST_RANGE_HIGH
    rl = range_low if range_low is not None else TEST_RANGE_LOW
    rd = range_duration if range_duration is not None else TEST_RANGE_DURATION

    logger.info("RIG TEST MODE — Running scenarios with range=[%s, %s], duration=%sh",
                rl, rh, rd)

    results = []
    for scenario in TEST_SCENARIOS:
        result = run_single_scenario(
            scenario, range_high=rh, range_low=rl, range_duration=rd
        )
        results.append(result)

    logger.info("RIG TEST MODE — Complete. %d scenarios evaluated.", len(results))
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    # Allow running directly for quick validation
    os.environ["RIG_TEST_MODE"] = "true"
    RIG_TEST_MODE = True  # override module-level flag for direct execution
    results = run_all_scenarios()
    for r in results:
        print(f"  {r['scenario']:15s}  price={r['price']}  disp={r['displacement']:.3f}  "
              f"extremity={r['extremity']:.3f}  status={r['rig_status']}  "
              f"(production={r['production_rig_status']})")
