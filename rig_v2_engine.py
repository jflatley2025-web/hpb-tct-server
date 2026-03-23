"""
rig_v2_engine.py — Multi-Range Hierarchical RIG (Range Integrity Gate v2)
=========================================================================
Extends RIG v1 (single-range) to evaluate HTF + MTF range interactions.

Key addition: detects MTF conflicts (liquidity traps / fake breakouts)
where price appears displaced on HTF but is trapped within an MTF range.

Reuses:
  - evaluate_rig_global()   from rig_engine.py   (RIG v1 logic)
  - compute_displacement()  from hpb_rig_validator.py
"""

from hpb_rig_validator import compute_displacement
from rig_engine import evaluate_rig_global

# HTF threshold: ranges shorter than this are MTF
HTF_MIN_DURATION_HOURS = 24

# Displacement bands for MTF conflict detection
HTF_MID_RANGE_LOW = 0.25
HTF_MID_RANGE_HIGH = 0.75
MTF_EXTREME_LOW = 0.2
MTF_EXTREME_HIGH = 0.8


def _safe_number(value, default: float = 0.0) -> float:
    """Coerce value to float, returning default on failure."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _range_sort_key(r: dict) -> tuple:
    """Deterministic sort key for HTF range selection.

    Uses only intrinsic structural data so the result is stable
    regardless of input ordering.  Priority: duration > liquidity >
    range_size > midpoint.
    """
    duration = _safe_number(r.get("range_duration_hours"))
    liquidity = _safe_number(r.get("liquidity_score"))
    high = _safe_number(r.get("range_high"))
    low = _safe_number(r.get("range_low"))
    range_size = high - low
    midpoint = (high + low) / 2 if high > low else 0.0
    return (duration, liquidity, range_size, midpoint)


def _not_evaluated_v2(base: dict, reason: str, displacement=None,
                      confidence: float = 0.0, htf_range=None) -> dict:
    """Build a canonical NOT_EVALUATED response for RIG v2."""
    return {
        **base,
        "status": "NOT_EVALUATED",
        "Gate": "RIG_v2",
        "reason": reason,
        "confidence": confidence,
        "displacement": displacement,
        "htf_range": htf_range,
    }


def evaluate_rig_v2(context: dict, ranges: list, current_price: float) -> dict:
    """
    Multi-range RIG evaluator — hierarchical HTF + MTF conflict detection.

    Args:
        context:       Pipeline context dict containing gates (1A, RCM, MSCE).
        ranges:        List of range dicts from range_engine_l2 / forming pool.
                       Each must have range_high, range_low, range_duration_hours.
        current_price: Latest price.

    Returns:
        Dict with status, reason, confidence, displacement, htf_range,
        mtf_conflict, htf_bias, session_bias, evaluated.
    """
    # --- Normalize inputs defensively ---
    if not isinstance(context, dict):
        context = {}
    gates = context.get("gates", context) or {}
    one_a = gates.get("1A", {}) or {}
    msce = gates.get("MSCE", {}) or {}

    htf_bias = one_a.get("bias", "neutral")
    session_name = msce.get("session")
    session_bias = msce.get("session_bias")

    base = {
        "evaluated": True,
        "htf_bias": htf_bias,
        "session_bias": session_bias,
        "mtf_conflict": False,
    }

    # --- Guard: no ranges at all or wrong type ---
    if not isinstance(ranges, list) or not ranges:
        return _not_evaluated_v2(base, "No ranges provided")

    # --- Partition into HTF and MTF ---
    htf_ranges = []
    mtf_ranges = []
    for r in ranges:
        if not isinstance(r, dict):
            continue
        rh = r.get("range_high")
        rl = r.get("range_low")
        if not isinstance(rh, (int, float)) or not isinstance(rl, (int, float)):
            continue
        if rh <= rl:
            continue

        dur = _safe_number(r.get("range_duration_hours"))
        if dur >= HTF_MIN_DURATION_HOURS:
            htf_ranges.append(r)
        else:
            mtf_ranges.append(r)

    if not htf_ranges:
        return _not_evaluated_v2(base, "No valid HTF ranges (duration >= 24h)")

    # --- Select dominant HTF range (deterministic, order-independent) ---
    dominant = max(htf_ranges, key=_range_sort_key)

    dom_high = dominant["range_high"]
    dom_low = dominant["range_low"]
    dom_duration = _safe_number(dominant.get("range_duration_hours"), default=48)

    # --- HTF displacement ---
    htf_disp = compute_displacement(current_price, dom_high, dom_low)
    if htf_disp is None:
        return _not_evaluated_v2(
            base,
            "Invalid HTF displacement (degenerate range or missing price)",
            htf_range=dominant,
        )

    # --- Run RIG v1 on dominant HTF range ---
    v1_result = evaluate_rig_global(
        htf_bias=htf_bias,
        session_name=session_name,
        session_bias=session_bias,
        range_high=dom_high,
        range_low=dom_low,
        current_price=current_price,
        range_duration_hours=dom_duration,
    )

    # --- MTF conflict detection ---
    # A conflict exists when HTF displacement is mid-range (indecisive)
    # but an MTF range shows extreme displacement (price trapped at edge).
    # This signals a liquidity trap / fake breakout.
    mtf_conflict = False
    htf_in_mid = HTF_MID_RANGE_LOW < htf_disp < HTF_MID_RANGE_HIGH

    if htf_in_mid and mtf_ranges:
        for mr in mtf_ranges:
            mtf_disp = compute_displacement(
                current_price, mr["range_high"], mr["range_low"]
            )
            if mtf_disp is not None and (
                mtf_disp < MTF_EXTREME_LOW or mtf_disp > MTF_EXTREME_HIGH
            ):
                mtf_conflict = True
                break

    # --- Override: MTF conflict blocks a VALID v1 result ---
    status = v1_result.get("status", "NOT_EVALUATED")
    reason = v1_result.get("reason")
    confidence = v1_result.get("confidence", 0.0)

    if status == "VALID" and mtf_conflict:
        status = "BLOCK"
        reason = "MTF conflict inside HTF range"
        confidence = 0.0

    # BLOCK always zeroes confidence (defensive)
    if status == "BLOCK":
        confidence = 0.0

    return {
        "status": status,
        "Gate": "RIG_v2",
        "reason": reason,
        "confidence": confidence,
        "evaluated": True,
        "displacement": htf_disp,
        "htf_bias": htf_bias,
        "session_bias": session_bias,
        "htf_range": dominant,
        "mtf_conflict": mtf_conflict,
    }
