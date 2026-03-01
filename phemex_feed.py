"""
phemex_feed.py — OHLCV Feed with TTL Cache

Fetches BTC/USDT candle data from MEXC's public REST API for three timeframes:
  HTF = 4h  (100 candles)
  MTF = 1h  (150 candles)
  LTF = 15m (200 candles)

Switched from ccxt+Phemex to httpx+MEXC because Phemex's exchange API
returns code:30000 for OHLCV calls regardless of symbol format or ccxt
version. MEXC provides identical BTC/USDT price data and is already used
by all other bots in this codebase. Trade execution on Phemex is separate.

Design decisions:
  - TTL cache per timeframe: HTF candles change every 4h, so fetching every 15s
    wastes 479 out of 480 calls and risks rate limits.
  - Read-only DataFrames: the feed marks returned DataFrames as not writeable.
    Gate functions that need to annotate data take an explicit .copy().
  - Explicit candle limits: more candles for LTF (needs tap history),
    fewer for HTF (each candle covers 4 hours of context).
  - Fail loudly on unsupported timeframe; return None on per-fetch errors so
    the trading loop can skip gracefully rather than crashing.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional

import httpx
import pandas as pd

logger = logging.getLogger("TCT-PhemexFeed")

# ---------------------------------------------------------------------------
# Timeframe constants
# ---------------------------------------------------------------------------

# MEXC spot symbol — BTCUSDT format (no slash or colon suffix)
SYMBOL = os.getenv("MEXC_FEED_SYMBOL", "BTCUSDT")

LTF_TF = "15m"
MTF_TF = "1h"
HTF_TF = "4h"

LTF_LIMIT = 200
MTF_LIMIT = 150
HTF_LIMIT = 100

# TTL in seconds: slightly under each candle close interval so the cache
# refreshes just after a new candle closes.
_TTL: dict[str, int] = {
    LTF_TF: 14 * 60,        # 14 min — refresh just before the 15m close
    MTF_TF: 59 * 60,        # 59 min — refresh just before the 1h close
    HTF_TF: 239 * 60,       # 239 min — refresh just before the 4h close
}

_LIMITS: dict[str, int] = {
    LTF_TF: LTF_LIMIT,
    MTF_TF: MTF_LIMIT,
    HTF_TF: HTF_LIMIT,
}

# MEXC uses "60m" for 1-hour intervals; "15m" and "4h" match directly.
_MEXC_INTERVAL: dict[str, str] = {
    LTF_TF: "15m",
    MTF_TF: "60m",
    HTF_TF: "4h",
}

MEXC_KLINES_URL = "https://api.mexc.com/api/v3/klines"

# DataFrame column names — consistent with the rest of this codebase.
OHLCV_COLUMNS = ["open_time", "open", "high", "low", "close", "volume"]

# ---------------------------------------------------------------------------
# Module-level cache
# ---------------------------------------------------------------------------

# Structure: {timeframe: (DataFrame, expires_at)}
_cache: dict[str, tuple[pd.DataFrame, float]] = {}


def clear_cache() -> None:
    """Clear the TTL cache. Useful in tests to force a fresh fetch."""
    _cache.clear()


# ---------------------------------------------------------------------------
# Core fetch
# ---------------------------------------------------------------------------

def _ohlcv_to_dataframe(raw: list) -> pd.DataFrame:
    """
    Convert MEXC raw klines list to a typed DataFrame.

    MEXC format: [[timestamp_ms, open, high, low, close, volume, ...], ...]
    Variable column count (8 or 12); only the first 6 fields are used.
    Numeric fields may be strings — astype(float) handles both.
    """
    rows = [[row[0], row[1], row[2], row[3], row[4], row[5]] for row in raw]
    df = pd.DataFrame(rows, columns=OHLCV_COLUMNS)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = df[col].astype(float)
    df.reset_index(drop=True, inplace=True)
    return df


def _make_readonly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mark all numpy arrays backing the DataFrame as non-writeable.

    Any gate function that tries to mutate the DataFrame in-place will raise
    a ValueError immediately instead of silently corrupting downstream gates.
    Gates that need to annotate data must call df.copy() explicitly.
    """
    for col in df.columns:
        arr = df[col].values
        arr.flags.writeable = False
    return df


def fetch_candles(timeframe: str) -> Optional[pd.DataFrame]:
    """
    Return a read-only OHLCV DataFrame for the given timeframe.

    Uses a TTL cache — only calls the MEXC API when the cached data has
    expired. Returns None if the API call fails, so the caller can skip
    the tick gracefully.

    Args:
        timeframe: one of "15m", "1h", "4h"

    Returns:
        Read-only DataFrame with columns [open_time, open, high, low, close,
        volume], or None on error.

    Raises:
        ValueError: if timeframe is not one of the supported values.
    """
    if timeframe not in _TTL:
        raise ValueError(
            f"Unsupported timeframe '{timeframe}'. "
            f"Must be one of: {list(_TTL)}"
        )

    # Return cached data if still fresh
    if timeframe in _cache:
        cached_df, expires_at = _cache[timeframe]
        if time.time() < expires_at:
            logger.debug("Cache hit for %s (expires in %.0fs)", timeframe,
                         expires_at - time.time())
            return cached_df

    # Fetch from MEXC
    limit = _LIMITS[timeframe]
    interval = _MEXC_INTERVAL[timeframe]
    try:
        response = httpx.get(
            MEXC_KLINES_URL,
            params={"symbol": SYMBOL, "interval": interval, "limit": limit},
            timeout=10.0,
        )
        response.raise_for_status()
        raw = response.json()
    except httpx.HTTPStatusError as exc:
        logger.error("HTTP error fetching %s candles: %s", timeframe, exc)
        return None
    except httpx.RequestError as exc:
        logger.error("Network error fetching %s candles: %s", timeframe, exc)
        return None
    except Exception as exc:  # noqa: BLE001
        logger.error("Unexpected error fetching %s candles: %s", timeframe, exc)
        return None

    if not raw:
        logger.warning("Empty OHLCV response for %s", timeframe)
        return None

    df = _ohlcv_to_dataframe(raw)
    df = _make_readonly(df)

    ttl = _TTL[timeframe]
    _cache[timeframe] = (df, time.time() + ttl)
    logger.debug("Fetched %d candles for %s (TTL=%ds)", len(df), timeframe, ttl)

    return df


def fetch_all() -> dict[str, Optional[pd.DataFrame]]:
    """
    Fetch all three timeframes and return them as a dict.

    Returns:
        {"15m": DataFrame|None, "1h": DataFrame|None, "4h": DataFrame|None}
    """
    return {
        LTF_TF: fetch_candles(LTF_TF),
        MTF_TF: fetch_candles(MTF_TF),
        HTF_TF: fetch_candles(HTF_TF),
    }
