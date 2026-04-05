# hpb_rig_validator.py
"""
Range Integrity Gate (RIG) — TCT-aware structural filter.

Self-contained, modular, pluggable.  No external dependencies beyond stdlib.

Decision tree:
  A. Trend-aligned (ltf_direction == htf_bias) → VALID, conf 1.0
  B. Counter-bias, mid-range (0.25 < pos < 0.75) → BLOCK
  C. Counter-bias, extreme, displacement >= 0.15 → CONDITIONAL (0.5–0.7)
  D. Counter-bias, extreme, displacement <  0.15 → BLOCK
"""
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

def compute_displacement(current_price, range_high, range_low):
    """Compute where price sits within the HTF range (0.0–1.0).

    Returns None if inputs are missing or range is degenerate.
    """
    if current_price is None or range_high is None or range_low is None:
        return None
    if range_high <= range_low:
        return None
    displacement = (current_price - range_low) / (range_high - range_low)
    return max(0.0, min(1.0, displacement))


def safe_confidence_modifier(rig_result: dict, default: float = 0.6) -> float:
    """Extract and clamp confidence_modifier from a RIG result dict.

    Coerces to float, falls back to *default* on None/invalid,
    and clamps to [0.0, 1.0].
    """
    try:
        mod = float(rig_result.get("confidence_modifier", default))
    except (TypeError, ValueError):
        mod = default
    return max(0.0, min(1.0, mod))


# ---------------------------------------------------------------------------
# Core RIG evaluator — spec-compliant standalone function
# ---------------------------------------------------------------------------

def evaluate_rig(
    htf_bias: str,
    ltf_direction: str,
    range_low: float,
    range_high: float,
    current_price: float,
    local_displacement: float,
    range_duration_hours: float,
    session_bias: str = "neutral",
) -> dict:
    """
    Evaluate Range Integrity Gate with TCT counter-bias logic.

    Args:
        htf_bias:             "bullish" | "bearish" — HTF directional context
        ltf_direction:        "bullish" | "bearish" — trade direction on LTF
        range_low:            Bottom of dominant HTF range
        range_high:           Top of dominant HTF range
        current_price:        Latest price
        local_displacement:   0–1 normalized position within range
        range_duration_hours: Age of the range in hours
        session_bias:         "expansion" | "distribution" | "neutral" | "trending"
                              (modifies confidence in Case C only, never blocks)

    Returns:
        {
            "rig_status":          "VALID" | "BLOCK" | "CONDITIONAL",
            "confidence_modifier": float (0.0–1.0),
            "position":            float (0.0–1.0),
            "counter_bias":        bool,
            "reason":              str,
        }
    """
    # --- Compute position ---
    if range_high > range_low and current_price is not None:
        position = (current_price - range_low) / (range_high - range_low)
        position = max(0.0, min(1.0, position))
    else:
        position = 0.5  # neutral fallback for degenerate range

    # --- Fix displacement 0.0 ---
    # If displacement is exactly 0.0 (likely a pipeline default, not real),
    # recompute from distance to range midpoint.
    if local_displacement == 0.0 and range_high > range_low:
        range_mid = (range_high + range_low) / 2.0
        local_displacement = abs(current_price - range_mid) / (range_high - range_low)

    # --- Counter-bias detection ---
    counter_bias = (ltf_direction != htf_bias)

    # === Case A: Trend-aligned → VALID ===
    if not counter_bias:
        return {
            "rig_status": "VALID",
            "confidence_modifier": 1.0,
            "position": round(position, 4),
            "counter_bias": False,
            "reason": "Trend-aligned — no restriction",
        }

    # === Counter-bias below ===
    in_low = position <= 0.25
    in_high = position >= 0.75
    in_extreme = in_low or in_high

    # === Case B: Counter-bias, mid-range → BLOCK ===
    if not in_extreme:
        return {
            "rig_status": "BLOCK",
            "confidence_modifier": 0.0,
            "position": round(position, 4),
            "counter_bias": True,
            "reason": "Counter-bias in mid-range (no edge)",
        }

    # === Cases C & D: Counter-bias at extreme ===

    # === Case D: No displacement at extreme → BLOCK ===
    if local_displacement < 0.15:
        return {
            "rig_status": "BLOCK",
            "confidence_modifier": 0.0,
            "position": round(position, 4),
            "counter_bias": True,
            "reason": "No displacement at range extreme",
        }

    # === Case C: Extreme WITH displacement → CONDITIONAL ===
    conf_mod = 0.5

    # +0.1 for strong displacement
    if local_displacement > 0.25:
        conf_mod += 0.1

    # +0.1 for session alignment with reversal logic
    if in_low and ltf_direction == "bullish" and session_bias == "expansion":
        conf_mod += 0.1
    elif in_high and ltf_direction == "bearish" and session_bias == "distribution":
        conf_mod += 0.1

    # Range duration penalty (soft, never blocks)
    if range_duration_hours < 24:
        conf_mod *= 0.8

    conf_mod = round(max(0.0, min(0.7, conf_mod)), 3)
    zone = "low" if in_low else "high"

    return {
        "rig_status": "CONDITIONAL",
        "confidence_modifier": conf_mod,
        "position": round(position, 4),
        "counter_bias": True,
        "reason": (
            f"Counter-bias at range {zone} — allowed with displacement "
            f"(conf x{conf_mod})"
        ),
    }


# ---------------------------------------------------------------------------
# Legacy adapter — wraps evaluate_rig for existing context-based callers
# (rig_engine.py, decision_engine_v2.py)
# ---------------------------------------------------------------------------

def range_integrity_validator(context: dict) -> dict:
    """
    Adapter: translates legacy context dict into evaluate_rig() call.

    Maps the existing gates-based context to the new standalone signature,
    then converts the output back to the legacy format expected by callers.
    """
    gates = context.get("gates", {})
    one_a = gates.get("1A", {})
    msce = gates.get("MSCE", {})
    rcm = gates.get("RCM", {})
    one_d = gates.get("1D", {})

    htf_bias = one_a.get("bias", "neutral")

    # Legacy uses session_bias as the trade direction proxy.
    # In the new model, session_bias IS ltf_direction for the adapter path.
    session_bias_raw = msce.get("session_bias", htf_bias)
    session_name = msce.get("session", "Unknown")

    local_disp = context.get("local_range_displacement")
    if local_disp is None:
        local_disp = 0.5
        logger.warning(
            "INVALID_DISPLACEMENT_INPUT: local_range_displacement missing, defaulting to 0.5"
        )

    range_valid = rcm.get("valid", False)
    range_duration = rcm.get("range_duration_hours", 0)
    exec_conf = one_d.get("score", 0.0)

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    # If range is not valid, defer to RCM (fail-open)
    if not range_valid:
        return {
            "timestamp": ts,
            "Gate": "RIG",
            "status": "VALID",
            "reason": "Range not valid — RIG defers to RCM",
            "confidence": exec_conf,
            "confidence_modifier": 1.0,
            "htf_bias": htf_bias,
            "session_bias": session_bias_raw,
            "evaluated": True,
            "position": None,
            "counter_bias": False,
            "local_displacement": local_disp,
            "range_duration": range_duration,
        }

    # Use real range data when available (passed by rig_engine.evaluate_rig_global),
    # otherwise fall back to unit range [0,1] where position = displacement.
    ctx_rh = context.get("range_high")
    ctx_rl = context.get("range_low")
    ctx_price = context.get("current_price")

    if ctx_rh is not None and ctx_rl is not None and ctx_rh > ctx_rl and ctx_price is not None:
        rig_range_low = ctx_rl
        rig_range_high = ctx_rh
        rig_price = ctx_price
    else:
        rig_range_low = 0.0
        rig_range_high = 1.0
        rig_price = local_disp

    result = evaluate_rig(
        htf_bias=htf_bias,
        ltf_direction=session_bias_raw,
        range_low=rig_range_low,
        range_high=rig_range_high,
        current_price=rig_price,
        local_displacement=local_disp,
        range_duration_hours=range_duration,
        session_bias="neutral",  # session type not available in legacy context
    )

    # Map to legacy output format
    status = result["rig_status"]
    conf_mod = result["confidence_modifier"]
    position_label = (
        "low" if result["position"] <= 0.25
        else "high" if result["position"] >= 0.75
        else "mid"
    )

    if status == "BLOCK":
        confidence = 0.0
    elif status == "CONDITIONAL":
        confidence = exec_conf * conf_mod
    else:
        confidence = exec_conf

    logger.info(
        "RIG %s: %s",
        status,
        {"rig_status": status, "rig_reason": result["reason"],
         "local_displacement": local_disp, "range_duration": range_duration,
         "confidence_modifier": conf_mod,
         "htf_bias": htf_bias, "session_bias": session_bias_raw},
    )

    return {
        "timestamp": ts,
        "Gate": "RIG",
        "status": status,
        "reason": result["reason"],
        "confidence": confidence,
        "confidence_modifier": conf_mod,
        "htf_bias": htf_bias,
        "session_bias": session_bias_raw,
        "evaluated": True,
        "position": position_label,
        "counter_bias": result["counter_bias"],
        "local_displacement": local_disp,
        "range_duration": range_duration,
    }
