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

from typing import List, Optional

from hpb_rig_validator import compute_displacement
from rig_engine import evaluate_rig_global

# HTF threshold: ranges shorter than this are MTF
HTF_MIN_DURATION_HOURS = 24

# Displacement bands for MTF conflict detection
HTF_MID_RANGE_LOW = 0.25
HTF_MID_RANGE_HIGH = 0.75
MTF_EXTREME_LOW = 0.2
MTF_EXTREME_HIGH = 0.8


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
    # --- Extract gate context safely ---
    gates = context.get("gates", context)
    one_a = gates.get("1A", {})
    msce = gates.get("MSCE", {})

    htf_bias = one_a.get("bias", "neutral")
    session_name = msce.get("session")
    session_bias = msce.get("session_bias")

    base = {
        "evaluated": True,
        "htf_bias": htf_bias,
        "session_bias": session_bias,
        "htf_range": None,
        "mtf_conflict": False,
    }

    # --- Guard: no ranges at all ---
    if not ranges:
        return {
            **base,
            "status": "NOT_EVALUATED",
            "Gate": "RIG_v2",
            "reason": "No ranges provided",
            "confidence": 0.0,
            "displacement": None,
        }

    # --- Partition into HTF and MTF ---
    htf_ranges = []
    mtf_ranges = []
    for r in ranges:
        rh = r.get("range_high")
        rl = r.get("range_low")
        dur = r.get("range_duration_hours", 0)
        if rh is None or rl is None or rh <= rl:
            continue
        if dur >= HTF_MIN_DURATION_HOURS:
            htf_ranges.append(r)
        else:
            mtf_ranges.append(r)

    if not htf_ranges:
        return {
            **base,
            "status": "NOT_EVALUATED",
            "Gate": "RIG_v2",
            "reason": "No valid HTF ranges (duration >= 24h)",
            "confidence": 0.0,
            "displacement": None,
        }

    # --- Select dominant HTF range: longest duration, then highest liquidity ---
    dominant = max(
        htf_ranges,
        key=lambda r: (
            r.get("range_duration_hours", 0),
            r.get("liquidity_score", 0),
        ),
    )

    dom_high = dominant["range_high"]
    dom_low = dominant["range_low"]
    dom_duration = dominant.get("range_duration_hours", 48)

    # --- HTF displacement ---
    htf_disp = compute_displacement(current_price, dom_high, dom_low)
    if htf_disp is None:
        return {
            **base,
            "status": "NOT_EVALUATED",
            "Gate": "RIG_v2",
            "reason": "Invalid HTF displacement (degenerate range or missing price)",
            "confidence": 0.0,
            "displacement": None,
            "htf_range": dominant,
        }

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
