"""
msce_engine.py — MSCE (Multi-Session Context Engine) for RIG
=============================================================
Provides real time-based session context for RIG evaluation.

Uses session_manipulation.py for session detection, then derives
directional session_bias from session timing + HTF structure.

Session model (TCT spec):
  - Asia (accumulation):    bias aligns with HTF (range-building)
  - London (expansion):     bias aligns with HTF (breakout phase)
  - New York (distribution): bias opposes HTF (reversal/distribution)
  - Off-session:            bias aligns with HTF (default)
"""

import logging
from typing import Optional, Tuple

logger = logging.getLogger("MSCEEngine")


# Map session_manipulation names → TCT session types
_SESSION_TYPE_MAP = {
    "asia": ("Asia", "accumulation"),
    "london": ("London", "expansion"),
    "new_york": ("New York", "distribution"),
}

# Broader session windows for naming (used when NOT in manipulation window)
_BROAD_SESSION_HOURS = [
    (0, 8, "Asia"),
    (8, 13, "London"),
    (13, 24, "New York"),
]


def get_msce_context(htf_bias: str) -> dict:
    """
    Build full MSCE context for the current moment.

    Returns dict with:
        session:      display name ("Asia", "London", "New York")
        session_bias: directional bias ("bullish", "bearish", or None)
        session_type: TCT phase ("accumulation", "expansion", "distribution")
        is_manipulation_window: bool
    """
    session_name, session_type, in_window = _detect_session()
    session_bias = _derive_session_bias(session_type, htf_bias)

    return {
        "session": session_name,
        "session_bias": session_bias,
        "session_type": session_type,
        "is_manipulation_window": in_window,
    }


def _detect_session() -> Tuple[str, str, bool]:
    """
    Detect current session name, type, and whether we're in a
    manipulation window.

    Returns:
        (session_name, session_type, is_manipulation_window)
    """
    try:
        from session_manipulation import get_active_session
        active = get_active_session()  # returns "asia", "london", "new_york", or None
    except ImportError:
        active = None

    if active and active in _SESSION_TYPE_MAP:
        name, stype = _SESSION_TYPE_MAP[active]
        return name, stype, True

    # Not in a manipulation window — use broad session hours
    from datetime import datetime, timezone
    hour = datetime.now(timezone.utc).hour

    for start, end, name in _BROAD_SESSION_HOURS:
        if start <= hour < end:
            # Derive type from broad session
            if name == "Asia":
                return name, "accumulation", False
            elif name == "London":
                return name, "expansion", False
            else:
                return name, "distribution", False

    # Fallback (shouldn't happen given 0-24 coverage)
    return "Unknown", "accumulation", False


def _derive_session_bias(session_type: str, htf_bias: str) -> Optional[str]:
    """
    Derive directional session bias from session type + HTF structure.

    TCT model:
      - accumulation (Asia):     aligns with HTF → same direction
      - expansion (London):      aligns with HTF → same direction
      - distribution (New York): opposes HTF → reversal/counter
      - unknown:                 aligns with HTF (conservative default)

    Returns "bullish", "bearish", or None (when HTF is neutral/unknown).
    """
    if htf_bias not in ("bullish", "bearish"):
        return None

    if session_type == "distribution":
        # NY distribution opposes HTF structure
        return "bearish" if htf_bias == "bullish" else "bullish"

    # accumulation + expansion align with HTF
    return htf_bias
