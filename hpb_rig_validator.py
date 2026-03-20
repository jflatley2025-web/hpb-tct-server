# hpb_rig_validator.py
from datetime import datetime


def compute_displacement(current_price, range_high, range_low):
    """Compute local range displacement: where price sits within the HTF range.

    Returns a float between 0.0 and 1.0, or None if the range is degenerate
    (range_high == range_low) or inputs are missing.
    """
    if current_price is None or range_high is None or range_low is None:
        return None
    if range_high == range_low:
        return None
    displacement = (current_price - range_low) / (range_high - range_low)
    return max(0.0, min(1.0, displacement))


def range_integrity_validator(context):
    """
    HPB Range Integrity Validator (RIG)
    Blocks invalid counter-bias trades when HTF range remains valid.
    Standalone version – no TensorTrade runtime required.
    """

    # Extract info from HPB context
    gates = context.get("gates", {})
    rcm = gates.get("RCM", {})
    msce = gates.get("MSCE", {})
    one_a = gates.get("1A", {})
    one_d = gates.get("1D", {})

    htf_bias = one_a.get("bias", "neutral")
    session_bias = msce.get("session_bias", htf_bias)
    session_name = msce.get("session", "Unknown")

    local_disp = context.get("local_range_displacement", 0.0)
    range_valid = rcm.get("valid", False)
    range_duration = rcm.get("range_duration_hours", 0)
    exec_conf = one_d.get("score", 0.0)

    # Thresholds (from HP-TCT variables)
    MIN_DURATION = 24        # hours
    DISP_THRESHOLD = 0.25    # <25% of range = weak displacement

    if range_valid and range_duration >= MIN_DURATION and local_disp < DISP_THRESHOLD:
        if session_bias != htf_bias:
            return {
                "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                "status": "BLOCK",
                "Gate": "RIG",
                "reason": f"Counter-bias {session_name} session during intact HTF range.",
                "confidence": 0.0,
                "htf_bias": htf_bias,
                "session_bias": session_bias
            }

    # Otherwise, range structure is safe to continue
    return {
        "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "status": "VALID",
        "Gate": "RIG",
        "reason": None,
        "confidence": exec_conf,
        "htf_bias": htf_bias,
        "session_bias": session_bias
    }
