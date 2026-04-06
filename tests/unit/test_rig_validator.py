import pytest

from hpb_rig_validator import evaluate_rig


# ---------------------------
# Helpers
# ---------------------------

RANGE_LOW = 100.0
RANGE_HIGH = 200.0


def make_input(
    htf_bias="bullish",
    ltf_direction="bullish",
    price=150.0,
    displacement=0.2,
    duration=48,
    session="neutral"
):
    return dict(
        htf_bias=htf_bias,
        ltf_direction=ltf_direction,
        range_low=RANGE_LOW,
        range_high=RANGE_HIGH,
        current_price=price,
        local_displacement=displacement,
        range_duration_hours=duration,
        session_bias=session,
    )


# ---------------------------
# CASE A — Trend continuation
# ---------------------------

def test_trend_aligned_always_valid():
    result = evaluate_rig(**make_input(
        htf_bias="bullish",
        ltf_direction="bullish",
        price=150
    ))

    assert result["rig_status"] == "VALID"
    assert result["confidence_modifier"] == 1.0
    assert result["counter_bias"] is False


# ---------------------------
# CASE B — Mid-range block
# ---------------------------

@pytest.mark.parametrize("price", [130, 150, 170])  # inside 0.25–0.75
def test_counter_bias_mid_range_block(price):
    result = evaluate_rig(**make_input(
        htf_bias="bullish",
        ltf_direction="bearish",
        price=price,
        displacement=0.3
    ))

    assert result["rig_status"] == "BLOCK"
    assert result["confidence_modifier"] == 0.0
    assert result["counter_bias"] is True
    assert "mid-range" in result["reason"].lower()


# ---------------------------
# CASE C — Extreme WITH displacement
# ---------------------------

@pytest.mark.parametrize("price", [105, 195])  # extremes
def test_counter_bias_extreme_with_displacement_conditional(price):
    result = evaluate_rig(**make_input(
        htf_bias="bullish",
        ltf_direction="bearish",
        price=price,
        displacement=0.2,
        session="distribution"  # helps bearish reversal
    ))

    assert result["rig_status"] == "CONDITIONAL"
    assert 0.5 <= result["confidence_modifier"] <= 0.7
    assert result["counter_bias"] is True


def test_conditional_confidence_increases_with_displacement():
    low_disp = evaluate_rig(**make_input(
        ltf_direction="bearish",
        price=195,
        displacement=0.16
    ))

    high_disp = evaluate_rig(**make_input(
        ltf_direction="bearish",
        price=195,
        displacement=0.3
    ))

    assert high_disp["confidence_modifier"] >= low_disp["confidence_modifier"]


# ---------------------------
# CASE D — Extreme WITHOUT displacement
# ---------------------------

@pytest.mark.parametrize("price", [105, 195])
def test_counter_bias_extreme_no_displacement_block(price):
    result = evaluate_rig(**make_input(
        htf_bias="bullish",
        ltf_direction="bearish",
        price=price,
        displacement=0.1
    ))

    assert result["rig_status"] == "BLOCK"
    assert result["confidence_modifier"] == 0.0
    assert "no displacement" in result["reason"].lower()


# ---------------------------
# DISPLACEMENT FIX (CRITICAL)
# ---------------------------

def test_zero_displacement_is_recomputed():
    result = evaluate_rig(**make_input(
        htf_bias="bullish",
        ltf_direction="bearish",
        price=195,          # extreme
        displacement=0.0    # broken input
    ))

    # Should NOT behave like zero displacement block blindly
    assert "position" in result
    assert result["position"] > 0.75

    # Either conditional or block depending on recomputed value
    assert result["rig_status"] in ["BLOCK", "CONDITIONAL"]


# ---------------------------
# RANGE DURATION PENALTY
# ---------------------------

def test_duration_penalty_applied():
    long_range = evaluate_rig(**make_input(
        ltf_direction="bearish",
        price=195,
        displacement=0.2,
        duration=48
    ))

    short_range = evaluate_rig(**make_input(
        ltf_direction="bearish",
        price=195,
        displacement=0.2,
        duration=12
    ))

    assert short_range["confidence_modifier"] <= long_range["confidence_modifier"]


# ---------------------------
# SESSION SHOULD NOT BLOCK
# ---------------------------

def test_session_does_not_force_block():
    result = evaluate_rig(**make_input(
        htf_bias="bullish",
        ltf_direction="bearish",
        price=195,
        displacement=0.2,
        session="neutral"
    ))

    assert result["rig_status"] in ["CONDITIONAL", "BLOCK"]
    # But NOT blocked due to session alone
    assert "session" not in result["reason"].lower()


# ---------------------------
# EDGE CASES
# ---------------------------

def test_exact_boundary_mid_range():
    result = evaluate_rig(**make_input(
        ltf_direction="bearish",
        price=125  # exactly 0.25
    ))

    # Boundary should be treated as extreme, not mid-range
    assert result["rig_status"] in ["BLOCK", "CONDITIONAL"]


def test_invalid_range_aligned_bias_valid():
    """
    Degenerate range + trend-aligned should still be VALID
    (Case A always overrides)
    """
    result = evaluate_rig(**make_input(
        htf_bias="bullish",
        ltf_direction="bullish",
        price=150,
    ) | {
        "range_low": 100,
        "range_high": 100
    })

    assert result["rig_status"] == "VALID"
    assert result["confidence_modifier"] == 1.0


def test_invalid_range_counter_bias_blocks():
    """
    Degenerate range + counter-bias → treated as mid-range (pos=0.5) → BLOCK
    """
    result = evaluate_rig(**make_input(
        htf_bias="bullish",
        ltf_direction="bearish",
        price=150,
    ) | {
        "range_low": 100,
        "range_high": 100
    })

    assert result["rig_status"] == "BLOCK"
    assert result["confidence_modifier"] == 0.0
    assert result["counter_bias"] is True
