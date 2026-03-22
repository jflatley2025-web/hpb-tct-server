"""
rig_engine.py — Canonical RIG (Range Integrity Gate) Evaluator
================================================================
Single entry point for RIG evaluation across all pipelines (5A, 5B, Phemex).

Replaces:
  - _evaluate_rig_from_forming()  (schematics_5b_trader.py)
  - _evaluate_rig_safe()          (server_mexc.py)
  - _evaluate_rig()               (schematics_5b_trader.py, phemex_tct_trader.py)

Requires real gate context — never fabricates RCM/MSCE data.
Fail-closed: missing inputs → NOT_EVALUATED (never silent VALID).
"""

from datetime import datetime, timezone
from typing import Optional

from hpb_rig_validator import range_integrity_validator, compute_displacement

# Minimum range age (hours) for RIG to consider blocking.
# Ranges younger than this are too immature to enforce.
MIN_RANGE_DURATION = 24


def evaluate_rig_global(
    htf_bias: str,
    session_name: str,
    session_bias: str,
    range_high: Optional[float],
    range_low: Optional[float],
    current_price: Optional[float],
    range_duration_hours: float = 48,
    exec_score: float = 0.0,
    displacement_override: Optional[float] = None,
) -> dict:
    """
    Canonical RIG evaluator — one function for all pipelines.

    Args:
        htf_bias:               "bullish" / "bearish" / "neutral" from Gate 1A
        session_name:           Real session from MSCE ("Asia", "London", "New York", or None)
        session_bias:           Time-based directional bias from MSCE
        range_high:             Top of dominant range (from RCM / forming schematics)
        range_low:              Bottom of dominant range
        current_price:          Latest price
        range_duration_hours:   Estimated range age (default 48 for forming)
        exec_score:             1D execution score (0.0-1.0)
        displacement_override:  Pre-computed conservative displacement (e.g. min across
                                multiple ranges). When provided, used instead of computing
                                from range_high/range_low.

    Returns:
        Dict with status, Gate, reason, confidence, displacement,
        htf_bias, session_bias, evaluated, timestamp.
    """
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    # --- Compute displacement ---
    # Use override (conservative min across ranges) if provided,
    # otherwise compute from the primary range.
    if displacement_override is not None:
        displacement = displacement_override
    else:
        displacement = compute_displacement(current_price, range_high, range_low)

    if displacement is None:
        return {
            "status": "NOT_EVALUATED",
            "Gate": "RIG",
            "reason": _missing_reason(
                displacement=False,
                session=session_name is not None,
                bias=session_bias is not None,
                duration=range_duration_hours is not None,
            ),
            "confidence": 0.0,
            "evaluated": True,
            "displacement": None,
            "htf_bias": htf_bias,
            "session_bias": session_bias,
            "timestamp": ts,
        }

    # --- Validate required gate inputs ---
    has_session = session_name is not None
    has_bias = session_bias is not None and session_bias in ("bullish", "bearish")
    has_duration = range_duration_hours is not None and range_duration_hours > 0

    if not (has_session and has_bias and has_duration):
        return {
            "status": "NOT_EVALUATED",
            "Gate": "RIG",
            "reason": _missing_reason(
                displacement=True,
                session=has_session,
                bias=has_bias,
                duration=has_duration,
            ),
            "confidence": 0.0,
            "evaluated": True,
            "displacement": displacement,
            "htf_bias": htf_bias,
            "session_bias": session_bias,
            "timestamp": ts,
        }

    # --- Guard: range must meet minimum duration ---
    if range_duration_hours < MIN_RANGE_DURATION:
        return {
            "status": "NOT_EVALUATED",
            "Gate": "RIG",
            "reason": f"Range duration {range_duration_hours}h below minimum {MIN_RANGE_DURATION}h",
            "confidence": 0.0,
            "evaluated": True,
            "displacement": displacement,
            "htf_bias": htf_bias,
            "session_bias": session_bias,
            "timestamp": ts,
        }

    # --- Build validator context (NO FAKE DATA) ---
    rig_context = {
        "gates": {
            "1A": {"bias": htf_bias or "neutral"},
            "RCM": {
                "valid": True,
                "range_duration_hours": range_duration_hours,
            },
            "MSCE": {
                "session_bias": session_bias,
                "session": session_name,
            },
            "1D": {"score": exec_score or 0.0},
        },
        "local_range_displacement": displacement,
    }

    result = range_integrity_validator(rig_context)

    # --- Normalize output ---
    result["evaluated"] = True
    result["displacement"] = displacement

    # Enforce: BLOCK always zeroes confidence (prevent downstream override)
    if result.get("status") == "BLOCK":
        result["confidence"] = 0.0

    return result


def _missing_reason(displacement: bool, session: bool,
                    bias: bool, duration: bool) -> str:
    """Build a diagnostic reason string for NOT_EVALUATED."""
    parts = []
    if not displacement:
        parts.append("displacement")
    if not session:
        parts.append("session_name")
    if not bias:
        parts.append("session_bias")
    if not duration:
        parts.append("range_duration")
    return f"Missing: {', '.join(parts)}" if parts else "Unknown"
