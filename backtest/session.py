"""
backtest/session.py — MSCE Session Engine for Backtesting
==========================================================
Determines trading session and bias for any UTC timestamp.
Broader session windows than manipulation-only windows in
session_manipulation.py — these cover full session activity periods.

Session windows (UTC):
- Asia:    00:00–08:00  (low volatility, manipulation-prone)
- London:  08:00–13:00  (high volatility, trend-setting)
- NY:      13:00–22:00  (highest volatility, continuation/reversal)
- Off:     22:00–00:00  (dead zone)
"""

from datetime import datetime, timezone, time as dt_time
from typing import Dict, Optional


# Full session windows for backtest classification
SESSION_WINDOWS = {
    "asia": {
        "start": dt_time(0, 0),
        "end": dt_time(8, 0),
        "bias": "neutral",
        "confidence_multiplier": 0.9,
        "crosses_midnight": False,
    },
    "london": {
        "start": dt_time(8, 0),
        "end": dt_time(13, 0),
        "bias": "trending",
        "confidence_multiplier": 1.1,
        "crosses_midnight": False,
    },
    "new_york": {
        "start": dt_time(13, 0),
        "end": dt_time(22, 0),
        "bias": "trending",
        "confidence_multiplier": 1.2,
        "crosses_midnight": False,
    },
    "off": {
        "start": dt_time(22, 0),
        "end": dt_time(0, 0),
        "bias": "neutral",
        "confidence_multiplier": 0.8,
        "crosses_midnight": True,
    },
}


def get_session(timestamp_utc: datetime) -> Dict:
    """
    Classify a UTC timestamp into a trading session.

    Returns:
        Dict with keys: session, bias, confidence_multiplier
    """
    if timestamp_utc.tzinfo is None:
        timestamp_utc = timestamp_utc.replace(tzinfo=timezone.utc)
    else:
        # Normalize aware timestamps to UTC so session windows match correctly
        timestamp_utc = timestamp_utc.astimezone(timezone.utc)

    t = timestamp_utc.time()

    for name, window in SESSION_WINDOWS.items():
        if window["crosses_midnight"]:
            if t >= window["start"] or t < window["end"]:
                return {
                    "session": name,
                    "bias": window["bias"],
                    "confidence_multiplier": window["confidence_multiplier"],
                }
        else:
            if window["start"] <= t < window["end"]:
                return {
                    "session": name,
                    "bias": window["bias"],
                    "confidence_multiplier": window["confidence_multiplier"],
                }

    # Fallback (should not happen with complete windows)
    return {
        "session": "off",
        "bias": "neutral",
        "confidence_multiplier": 0.8,
    }


def get_session_name(timestamp_utc: datetime) -> str:
    """Return just the session name for a UTC timestamp."""
    return get_session(timestamp_utc)["session"]


def get_confidence_multiplier(timestamp_utc: datetime) -> float:
    """Return the confidence multiplier for a UTC timestamp."""
    return get_session(timestamp_utc)["confidence_multiplier"]
