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


# ---------------------------------------------------------------------------
# Range position zone thresholds
# ---------------------------------------------------------------------------
_LOW_ZONE_UPPER = 0.25   # position <= 0.25 → LOW zone (near range low)
_HIGH_ZONE_LOWER = 0.75  # position >= 0.75 → HIGH zone (near range high)
_EXTREME_DISP_MIN = 0.15  # minimum displacement at extreme for conditional pass
_SHORT_RANGE_HOURS = 24   # ranges younger than this weaken confidence


def _classify_position(position: float) -> str:
    """Classify price position within range as 'low', 'mid', or 'high'."""
    if position <= _LOW_ZONE_UPPER:
        return "low"
    if position >= _HIGH_ZONE_LOWER:
        return "high"
    return "mid"


def range_integrity_validator(context):
    """
    HPB Range Integrity Validator (RIG) — TCT-aware structural gate.

    Decision tree:
      1. Trend-aligned trades → PASS (no restriction)
      2. Counter-bias in mid-range → BLOCK (no edge, no setup)
      3. Counter-bias at range extreme with displacement → CONDITIONAL
         (allowed with confidence penalty — this is TCT reversal behavior)
      4. Counter-bias at range extreme WITHOUT displacement → BLOCK
         (no confirmation of reversal)

    Returns dict with:
      status:              "VALID" | "BLOCK" | "CONDITIONAL"
      confidence:          float (execution confidence)
      confidence_modifier: float (1.0 = no penalty, 0.5-0.7 = penalized)
      position:            "low" | "mid" | "high"
      counter_bias:        bool
    """
    gates = context.get("gates", {})
    rcm = gates.get("RCM", {})
    msce = gates.get("MSCE", {})
    one_a = gates.get("1A", {})
    one_d = gates.get("1D", {})

    htf_bias = one_a.get("bias", "neutral")
    session_bias = msce.get("session_bias", htf_bias)
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

    # --- Classify position and counter-bias ---
    position = _classify_position(local_disp)
    counter_bias = (session_bias != htf_bias)

    base_result = {
        "timestamp": ts,
        "Gate": "RIG",
        "htf_bias": htf_bias,
        "session_bias": session_bias,
        "evaluated": True,
        "position": position,
        "counter_bias": counter_bias,
        "local_displacement": local_disp,
        "range_duration": range_duration,
    }

    # --- Case A: Trend-aligned → PASS unconditionally ---
    if not counter_bias:
        logger.debug(
            "RIG PASS (trend-aligned): %s",
            {"position": position, "htf_bias": htf_bias,
             "session_bias": session_bias, "displacement": local_disp},
        )
        return {
            **base_result,
            "status": "VALID",
            "reason": None,
            "confidence": exec_conf,
            "confidence_modifier": 1.0,
        }

    # --- Counter-bias trades below: range context required ---
    if not range_valid:
        # No valid range → cannot assess, pass through (fail-open for
        # broken range context — RCM gate handles range quality upstream).
        return {
            **base_result,
            "status": "VALID",
            "reason": "Range not valid — RIG defers to RCM",
            "confidence": exec_conf,
            "confidence_modifier": 1.0,
        }

    # --- Case B: Counter-bias in MID range → HARD BLOCK ---
    if position == "mid":
        reason = "Counter-bias in mid-range — no structural edge"
        logger.info(
            "RIG BLOCK: %s",
            {"rig_status": "BLOCK", "rig_reason": reason,
             "local_displacement": local_disp, "range_duration": range_duration,
             "htf_bias": htf_bias, "session_bias": session_bias},
        )
        return {
            **base_result,
            "status": "BLOCK",
            "reason": reason,
            "confidence": 0.0,
            "confidence_modifier": 0.0,
        }

    # --- Case C: Counter-bias at EXTREME (low or high) ---

    # C1: No displacement at extreme → BLOCK (no reversal confirmation)
    if local_disp < _EXTREME_DISP_MIN or (1.0 - local_disp) < _EXTREME_DISP_MIN:
        # Price is at extreme but displacement is negligible — could be
        # a slow drift, not a displacement-driven reversal.
        # Re-check: for LOW zone, displacement IS the position value;
        # for HIGH zone, displacement from high is (1 - position).
        # If either is < 0.15, there's no meaningful push into the zone.
        pass  # fall through to displacement check below

    # Determine displacement strength at the relevant extreme
    if position == "low":
        extreme_disp = local_disp  # how deep into the low zone
    else:  # position == "high"
        extreme_disp = 1.0 - local_disp  # how deep into the high zone

    # Note: extreme_disp here measures how far INTO the extreme zone
    # we are.  But the spec says "local_displacement < 0.15 → BLOCK".
    # local_disp IS the range position (0=low, 1=high), so:
    #   - At LOW zone: local_disp is small (e.g. 0.10).  The "displacement"
    #     the spec refers to is whether price has MOVED into this zone,
    #     which we approximate as: being in the zone IS the displacement.
    #   - We use local_disp for LOW, (1-local_disp) for HIGH.

    # C1: Displacement too weak at extreme → BLOCK
    if position == "low" and local_disp < _EXTREME_DISP_MIN:
        reason = "Counter-bias at range low — insufficient displacement"
        logger.info(
            "RIG BLOCK: %s",
            {"rig_status": "BLOCK", "rig_reason": reason,
             "local_displacement": local_disp, "range_duration": range_duration,
             "htf_bias": htf_bias, "session_bias": session_bias},
        )
        return {
            **base_result,
            "status": "BLOCK",
            "reason": reason,
            "confidence": 0.0,
            "confidence_modifier": 0.0,
        }

    if position == "high" and (1.0 - local_disp) < _EXTREME_DISP_MIN:
        reason = "Counter-bias at range high — insufficient displacement"
        logger.info(
            "RIG BLOCK: %s",
            {"rig_status": "BLOCK", "rig_reason": reason,
             "local_displacement": local_disp, "range_duration": range_duration,
             "htf_bias": htf_bias, "session_bias": session_bias},
        )
        return {
            **base_result,
            "status": "BLOCK",
            "reason": reason,
            "confidence": 0.0,
            "confidence_modifier": 0.0,
        }

    # C2: Sufficient displacement at extreme → CONDITIONAL (allow with penalty)
    # Confidence modifier: 0.5 base, scaled up toward 0.7 by displacement strength
    conf_mod = 0.5 + min(0.2, extreme_disp * 0.5)

    # Range duration penalty: weak range context reduces confidence further
    if range_duration < _SHORT_RANGE_HOURS:
        conf_mod *= 0.8

    conf_mod = round(max(0.0, min(1.0, conf_mod)), 3)

    reason = (
        f"Counter-bias at range {position} — allowed with displacement "
        f"(conf x{conf_mod})"
    )
    logger.info(
        "RIG CONDITIONAL: %s",
        {"rig_status": "CONDITIONAL", "rig_reason": reason,
         "local_displacement": local_disp, "range_duration": range_duration,
         "confidence_modifier": conf_mod,
         "htf_bias": htf_bias, "session_bias": session_bias},
    )
    return {
        **base_result,
        "status": "CONDITIONAL",
        "reason": reason,
        "confidence": exec_conf * conf_mod,
        "confidence_modifier": conf_mod,
    }
