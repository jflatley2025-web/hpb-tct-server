# rig_test_mode.py
"""
Temporary RIG Test Mode for Displacement Validation

Allows controlled injection of range parameters and price to simulate
displacement scenarios.  Mid-range displacement applies a confidence
penalty rather than blocking — only structural counter-bias blocks.

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

    Returns a value 0.0-1.0 where:
    - 0.0 = dead center (mid-range)
    - 1.0 = at range boundary (extreme)
    """
    if displacement is None:
        return None
    return abs(displacement - 0.5) * 2


def evaluate_rig_test_status(displacement):
    """Determine RIG test status — mid-range applies confidence penalty, not block.

    Mid-range (inclusive): 0.25 <= disp <= 0.75 -> VALID (with confidence penalty)
    Extreme: disp > 0.75 or disp < 0.25 -> VALID (full confidence)

    RIG no longer blocks based on displacement alone.  Blocking requires
    all four structural conditions (see range_integrity_validator).
    """
    if displacement is None:
        return "NOT_EVALUATED"
    # All displacement values return VALID — blocking is only for
    # structural counter-bias conditions in range_integrity_validator.
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

    if rh <= rl:
        raise ValueError(f"range_high ({rh}) must be greater than range_low ({rl})")
    if rd <= 0:
        raise ValueError(f"range_duration ({rd}) must be positive")

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
        "local_range_displacement": displacement,
    }
    return context, displacement


def run_single_scenario(scenario, range_high=None, range_low=None,
                        range_duration=None):
    """Run a single test scenario and return results."""
    current_price = scenario["current_price"]
    name = scenario["name"]

    try:
        context, displacement = build_test_context(
            current_price,
            range_high=range_high,
            range_low=range_low,
            range_duration=range_duration,
        )
    except ValueError as e:
        return {
            "scenario": name,
            "price": current_price,
            "displacement": None,
            "extremity": None,
            "rig_status": "NOT_EVALUABLE",
            "production_rig_status": "NOT_EVALUATED",
            "production_reason": str(e),
        }

    if displacement is None:
        return {
            "scenario": name,
            "price": current_price,
            "displacement": None,
            "extremity": None,
            "rig_status": "NOT_EVALUABLE",
            "production_rig_status": "NOT_EVALUATED",
            "production_reason": "Invalid displacement (degenerate range or inputs)",
        }

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


def _is_truthy(value):
    return str(value).lower() in {"1", "true", "yes"}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    env_flag = os.getenv("RIG_TEST_MODE")

    if _is_truthy(env_flag) or RIG_TEST_MODE:
        results = run_all_scenarios()
        for r in results:
            print(
                f"  {r['scenario']:15s}  price={r['price']}  "
                f"disp={r['displacement']:.3f if r['displacement'] is not None else 'None'}  "
                f"extremity={r['extremity']:.3f if r['extremity'] is not None else 'None'}  "
                f"status={r['rig_status']}  "
                f"(production={r['production_rig_status']})"
            )
    else:
        print("RIG test mode not enabled. Set RIG_TEST_MODE=1 to run.")
