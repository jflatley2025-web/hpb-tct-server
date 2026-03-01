"""
phemex_feed.py — Phemex OHLCV Feed with TTL Cache

Fetches candle data from Phemex via ccxt for three timeframes:
  HTF = 4h  (100 candles)
  MTF = 1h  (150 candles)
  LTF = 15m (200 candles)

Design decisions:
  - TTL cache per timeframe: HTF candles change every 4h, so fetching every 15s
    wastes 479 out of 480 calls and risks Phemex rate limits.
  - Read-only DataFrames: the feed marks returned DataFrames as not writeable.
    Gate functions that need to annotate data take an explicit .copy() at that
    point — not at every gate entry.
  - Explicit candle limits: more candles for LTF (needs tap history),
    fewer for HTF (each candle covers 4 hours of context).
  - Fail loudly on exchange init errors; return None on per-fetch errors so
    the trading loop can skip gracefully rather than crashing.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional

import ccxt
import pandas as pd

logger = logging.getLogger("TCT-PhemexFeed")

# ---------------------------------------------------------------------------
# Timeframe constants
# ---------------------------------------------------------------------------

SYMBOL = os.getenv("PHEMEX_SYMBOL", "BTC/USDT")

# Candle counts per timeframe. LTF needs more history for tap spacing detection;
# HTF needs fewer because each 4h candle covers 4 hours of context.
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

# DataFrame column names — consistent with the rest of this codebase.
OHLCV_COLUMNS = ["open_time", "open", "high", "low", "close", "volume"]

# ---------------------------------------------------------------------------
# Module-level cache and exchange singleton
# ---------------------------------------------------------------------------

# Structure: {timeframe: (DataFrame, expires_at)}
_cache: dict[str, tuple[pd.DataFrame, float]] = {}

# Exchange is initialized once on first use (lazy) so tests can inject mocks.
_exchange: Optional[ccxt.phemex] = None


def _get_exchange() -> ccxt.phemex:
    """Return the ccxt.phemex singleton, creating it on first call.

    load_markets() is called once at init time so ccxt can resolve symbol
    IDs correctly (e.g. 'BTC/USDT' → 'sBTCUSDT' for spot). Without it,
    Phemex returns code:30000 "Please double check input arguments".
    """
    global _exchange
    if _exchange is None:
        api_key = os.getenv("PHEMEX_API_KEY", "")
        api_secret = os.getenv("PHEMEX_API_SECRET", "")
        _exchange = ccxt.phemex(
            {
                "apiKey": api_key,
                "secret": api_secret,
                "enableRateLimit": True,
            }
        )
        try:
            _exchange.load_markets()
            logger.info("Phemex exchange initialised (symbol=%s)", SYMBOL)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Phemex market load failed — candle fetches may error: %s", exc)
    return _exchange


def set_exchange(exchange: ccxt.phemex) -> None:
    """
    Inject a mock or pre-configured exchange instance.

    Intended for testing — call this before fetch_candles() to replace the
    ccxt.phemex singleton without touching environment variables.
    """
    global _exchange
    _exchange = exchange


def clear_cache() -> None:
    """Clear the TTL cache. Useful in tests to force a fresh fetch."""
    _cache.clear()


# ---------------------------------------------------------------------------
# Core fetch
# ---------------------------------------------------------------------------

def _ohlcv_to_dataframe(raw: list) -> pd.DataFrame:
    """
    Convert ccxt raw OHLCV list to a typed DataFrame.

    ccxt format: [[timestamp_ms, open, high, low, close, volume], ...]
    """
    df = pd.DataFrame(raw, columns=OHLCV_COLUMNS)
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

    Uses a TTL cache — only calls the exchange API when the cached data has
    expired. Returns None if the exchange call fails, so the caller can skip
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

    # Fetch from exchange
    limit = _LIMITS[timeframe]
    try:
        exchange = _get_exchange()
        raw = exchange.fetch_ohlcv(SYMBOL, timeframe=timeframe, limit=limit)
    except ccxt.NetworkError as exc:
        logger.error("Network error fetching %s candles: %s", timeframe, exc)
        return None
    except ccxt.ExchangeError as exc:
        logger.error("Exchange error fetching %s candles: %s", timeframe, exc)
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
