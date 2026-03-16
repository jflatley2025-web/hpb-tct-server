"""
session_manipulation.py — Session Manipulation Detection
=========================================================
Detects active trading sessions and manipulation windows per TCT methodology.
Integrates with MSCE (Multi-Session Context Engine) to apply session weight
multipliers to execution confidence.

Session windows (UTC):
- Asia:     23:30 - 01:00 (manipulation window)
- London:   07:30 - 09:00 (manipulation window)
- New York: 13:00 - 14:30 (manipulation window)

Session timing modifies execution confidence, NOT schematic quality scores.
"""

import logging
from datetime import datetime, timezone, time as dt_time
from typing import Optional

logger = logging.getLogger("SessionManipulation")

# Session manipulation windows (UTC)
SESSION_WINDOWS = {
    "asia": {
        "start": dt_time(23, 30),
        "end": dt_time(1, 0),
        "multiplier": 1.05,
        "crosses_midnight": True,
    },
    "london": {
        "start": dt_time(7, 30),
        "end": dt_time(9, 0),
        "multiplier": 1.10,
        "crosses_midnight": False,
    },
    "new_york": {
        "start": dt_time(13, 0),
        "end": dt_time(14, 30),
        "multiplier": 1.20,
        "crosses_midnight": False,
    },
}


def get_active_session(timestamp: Optional[datetime] = None) -> Optional[str]:
    """
    Determine the active manipulation session for a given timestamp.

    Args:
        timestamp: UTC datetime. Defaults to current UTC time.

    Returns:
        Session name ("asia", "london", "new_york") or None.
    """
    if timestamp is None:
        timestamp = datetime.now(timezone.utc)

    # Ensure timezone-aware — warn on naive datetimes
    if timestamp.tzinfo is None:
        logger.warning("Naive datetime passed to get_active_session; assuming UTC")
        timestamp = timestamp.replace(tzinfo=timezone.utc)

    current_time = timestamp.time()

    for session_name, window in SESSION_WINDOWS.items():
        if window["crosses_midnight"]:
            # Asia window spans midnight: 23:30 -> 01:00
            if current_time >= window["start"] or current_time <= window["end"]:
                return session_name
        elif window["start"] <= current_time <= window["end"]:
            return session_name

    return None


def get_session_multiplier(session_name: Optional[str]) -> float:
    """
    Get the execution confidence multiplier for a session.

    Returns 1.0 (no boost) if no active manipulation session.
    """
    if session_name is None:
        return 1.0
    window = SESSION_WINDOWS.get(session_name)
    if window is None:
        return 1.0
    return window["multiplier"]


def apply_session_multiplier(
    execution_confidence: float,
    timestamp: Optional[datetime] = None,
) -> dict:
    """
    Apply session manipulation multiplier to execution confidence.

    This is the MSCE integration point. Session timing modifies
    execution confidence, not schematic quality scores.

    Args:
        execution_confidence: Base confidence value (0-100).
        timestamp: UTC datetime for session detection.

    Returns:
        Dict with adjusted confidence and session metadata.
    """
    session = get_active_session(timestamp)
    multiplier = get_session_multiplier(session)
    adjusted = min(execution_confidence * multiplier, 100.0)

    result = {
        "original_confidence": execution_confidence,
        "adjusted_confidence": round(adjusted, 2),
        "session": session,
        "multiplier": multiplier,
        "boost_applied": session is not None,
    }

    if session:
        logger.debug(
            f"Session boost: {session} x{multiplier} — "
            f"{execution_confidence:.1f} → {adjusted:.1f}"
        )

    return result


def get_session_info(timestamp: Optional[datetime] = None) -> dict:
    """Return full session context for logging/display."""
    if timestamp is None:
        timestamp = datetime.now(timezone.utc)

    session = get_active_session(timestamp)
    return {
        "timestamp": timestamp.isoformat(),
        "active_session": session,
        "multiplier": get_session_multiplier(session),
        "is_manipulation_window": session is not None,
    }
