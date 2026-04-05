# hpb_rig_validator.py
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


def compute_displacement(current_price, range_high, range_low):
    """Compute local range displacement: where price sits within the HTF range.

    Returns a float between 0.0 and 1.0, or None if the range is degenerate
    (range_high == range_low) or inputs are missing.
    """
    if current_price is None or range_high is None or range_low is None:
        return None
    if range_high <= range_low:
        return None
    displacement = (current_price - range_low) / (range_high - range_low)
    return max(0.0, min(1.0, displacement))


def range_integrity_validator(context):
    """
    HPB Range Integrity Validator (RIG) — structural integrity gate.

    RIG may ONLY block when ALL four conditions are true:
      1. HTF range is valid
      2. Range duration >= 24 hours
      3. Local displacement < 25% of HTF range
      4. Session bias is opposite HTF bias

    If NOT all four conditions are met → VALID (no block).

    These do NOT block:
      - Equilibrium proximity
      - Mid-range positioning
      - Weak displacement alone (without counter-bias)
      - Missing minor inputs
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

    # Default to 0.5 (mid-range / neutral) when displacement is missing.
    # 0.0 would falsely satisfy the "weak displacement" block condition.
    local_disp = context.get("local_range_displacement")
    if local_disp is None:
        local_disp = 0.5
        logger.warning(
            "INVALID_DISPLACEMENT_INPUT: local_range_displacement missing from context, defaulting to 0.5"
        )
    range_valid = rcm.get("valid", False)
    range_duration = rcm.get("range_duration_hours", 0)
    exec_conf = one_d.get("score", 0.0)

    # Strict block thresholds
    MIN_DURATION = 24        # hours
    DISP_THRESHOLD = 0.25    # <25% of range = weak displacement

    # --- Evaluate the 4 strict block conditions ---
    cond_range_valid = range_valid
    cond_duration = range_duration >= MIN_DURATION
    cond_weak_disp = local_disp < DISP_THRESHOLD
    cond_counter_bias = (session_bias != htf_bias)

    all_block_conditions = (
        cond_range_valid
        and cond_duration
        and cond_weak_disp
        and cond_counter_bias
    )

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    if all_block_conditions:
        reason = f"Counter-bias {session_name} session during intact HTF range."
        logger.info(
            "RIG BLOCK: %s",
            {
                "rig_status": "BLOCK",
                "rig_reason": reason,
                "local_displacement": local_disp,
                "range_duration": range_duration,
                "htf_bias": htf_bias,
                "session_bias": session_bias,
            },
        )
        return {
            "timestamp": ts,
            "status": "BLOCK",
            "Gate": "RIG",
            "reason": reason,
            "confidence": 0.0,
            "htf_bias": htf_bias,
            "session_bias": session_bias,
            "evaluated": True,
        }

    # Not all conditions met → VALID
    reason = None
    logger.debug(
        "RIG VALID: %s",
        {
            "rig_status": "VALID",
            "rig_reason": reason,
            "local_displacement": local_disp,
            "range_duration": range_duration,
            "htf_bias": htf_bias,
            "session_bias": session_bias,
            "range_valid": range_valid,
            "counter_bias": cond_counter_bias,
        },
    )
    return {
        "timestamp": ts,
        "status": "VALID",
        "Gate": "RIG",
        "reason": reason,
        "confidence": exec_conf,
        "htf_bias": htf_bias,
        "session_bias": session_bias,
        "evaluated": True,
    }
