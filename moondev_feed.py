"""
moondev_feed.py — MoonDev API data feed for HPB–TCT paper trading
=================================================================
Provides OHLCV candle data and live prices from the MoonDev API
(https://api.moondev.com) as a drop-in replacement for the MEXC
feed used by schematics_5b_trader.py.

Enabled when MOONDEV_PAPER_TRADING=true is set in the environment.
Falls back to None on any fetch failure so callers can fall back to MEXC.

Rate limits: 60 req/s sustained, 200 req/s burst.
We add a small delay between requests to stay well under the limit.
"""

import os
import time
import threading
import logging
import requests
import pandas as pd
from typing import Optional

logger = logging.getLogger("MoonDevFeed")

# ================================================================
# CONFIGURATION
# ================================================================
MOONDEV_API_KEY = os.getenv("MOONDEV_API_KEY")
if not MOONDEV_API_KEY:
    raise RuntimeError(
        "MOONDEV_API_KEY environment variable is not set. "
        "Export it before starting the process."
    )
MOONDEV_BASE_URL = os.getenv("MOONDEV_BASE_URL", "https://api.moondev.com")

# Minimum delay between consecutive requests (seconds).
# 60 req/s sustained → ≥ 1/60 s each. We use 0.1 s to be a good
# citizen and stay well under the limit for a shared key.
_REQUEST_DELAY_S = 0.1
_last_request_time: float = 0.0
_rate_limit_lock = threading.Lock()

# MoonDev uses its own coin naming; keep a map for the symbols we care about.
# The API uses short uppercase tickers (e.g. "BTC") for /api/prices but
# the /api/candles/{coin} endpoint accepts full symbols like "BTCUSDT".
_SYMBOL_MAP = {
    "BTCUSDT": "BTCUSDT",
    "ETHUSDT": "ETHUSDT",
    "SOLUSDT": "SOLUSDT",
    "BCHUSDT": "BCHUSDT",
    "WIFUSDT": "WIFUSDT",
    "DOGEUSDT": "DOGEUSDT",
    "HBARUSDT": "HBARUSDT",
    "FETUSDT": "FETUSDT",
    "XMRUSDT": "XMRUSDT",
    "FARTCOINUSDT": "FARTCOINUSDT",
    "PEPEUSDT": "PEPEUSDT",
    "XRPUSDT": "XRPUSDT",
}

# Map TCT / MEXC timeframe strings → MoonDev interval strings.
# MoonDev uses standard interval notation.
_INTERVAL_MAP: dict = {
    "1m":  "1m",
    "5m":  "5m",
    "15m": "15m",
    "30m": "30m",
    "1h":  "1h",
    "2h":  "2h",
    "4h":  "4h",
    "6h":  "6h",
    "12h": "12h",
    "1d":  "1d",
    "1w":  "1w",
    # MEXC aliases used in schematics_5b_trader
    "60m": "1h",
    "240m": "4h",
    "1440m": "1d",
}

# MoonDev short ticker for /api/prices lookup (strip USDT suffix)
_PRICE_TICKER_MAP = {
    "BTCUSDT": "BTC",
    "ETHUSDT": "ETH",
    "SOLUSDT": "SOL",
    "BCHUSDT": "BCH",
    "WIFUSDT": "WIF",
    "DOGEUSDT": "DOGE",
    "HBARUSDT": "HBAR",
    "FETUSDT": "FET",
    "XMRUSDT": "XMR",
    "FARTCOINUSDT": "FARTCOIN",
    "PEPEUSDT": "PEPE",
    "XRPUSDT": "XRP",
}


# ================================================================
# INTERNAL HELPERS
# ================================================================

def _rate_limit():
    """Block until the minimum inter-request delay has elapsed (thread-safe)."""
    global _last_request_time
    with _rate_limit_lock:
        elapsed = time.monotonic() - _last_request_time
        if elapsed < _REQUEST_DELAY_S:
            time.sleep(_REQUEST_DELAY_S - elapsed)
        _last_request_time = time.monotonic()


def _get(path: str, params: Optional[dict] = None, timeout: int = 20) -> requests.Response:
    """Authenticated GET to the MoonDev API with rate limiting."""
    _rate_limit()
    url = f"{MOONDEV_BASE_URL}{path}"
    headers = {
        "X-API-Key": MOONDEV_API_KEY,
        "User-Agent": "HPB-TCT-Server/1.0",
    }
    qparams = params or {}
    resp = requests.get(url, params=qparams, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return resp


def _parse_candle_response(raw) -> Optional[pd.DataFrame]:
    """
    Parse MoonDev candle response into the standard DataFrame format used
    by the rest of the system: columns [open_time, open, high, low, close, volume].

    MoonDev may return either:
    - A list of OHLCV arrays: [[timestamp_ms, open, high, low, close, volume], ...]
    - A list of dicts: [{"time": ..., "open": ..., "high": ..., "low": ...,
                          "close": ..., "volume": ...}, ...]
    - A dict with a "data" key wrapping either of the above

    We handle all three formats gracefully.
    """
    if isinstance(raw, dict):
        # Unwrap common envelope keys
        for key in ("data", "candles", "ohlcv", "result"):
            if key in raw and isinstance(raw[key], list):
                raw = raw[key]
                break
        else:
            logger.warning("[MoonDev] Unexpected dict response without known data key: %s", list(raw.keys()))
            return None

    if not isinstance(raw, list) or len(raw) == 0:
        logger.warning("[MoonDev] Empty or non-list candle response")
        return None

    rows = []
    first = raw[0]

    if isinstance(first, (list, tuple)):
        # Format: [[timestamp_ms, open, high, low, close, volume], ...]
        for r in raw:
            if len(r) >= 6:
                rows.append([r[0], r[1], r[2], r[3], r[4], r[5]])
            elif len(r) >= 5:
                # No volume field — pad with 0
                rows.append([r[0], r[1], r[2], r[3], r[4], 0.0])

    elif isinstance(first, dict):
        # Format: [{"time": ..., "open": ..., "high": ..., "low": ..., "close": ..., "volume": ...}]
        # Also handles "t", "o", "h", "l", "c", "v" short keys
        _key_aliases = {
            "timestamp": ("time", "t", "timestamp", "open_time", "ts"),
            "open":      ("open", "o"),
            "high":      ("high", "h"),
            "low":       ("low", "l"),
            "close":     ("close", "c"),
            "volume":    ("volume", "v", "vol"),
        }

        def _pick(d, keys):
            for k in keys:
                if k in d:
                    return d[k]
            return None

        for r in raw:
            ts   = _pick(r, _key_aliases["timestamp"])
            o    = _pick(r, _key_aliases["open"])
            h    = _pick(r, _key_aliases["high"])
            lo   = _pick(r, _key_aliases["low"])
            c    = _pick(r, _key_aliases["close"])
            vol  = _pick(r, _key_aliases["volume"]) or 0.0
            if ts is not None and o is not None and h is not None and lo is not None and c is not None:
                rows.append([ts, o, h, lo, c, vol])

    if not rows:
        logger.warning("[MoonDev] Could not parse any candle rows from response")
        return None

    df = pd.DataFrame(rows, columns=["open_time", "open", "high", "low", "close", "volume"])

    # Normalise open_time — could be unix seconds or milliseconds or ISO string
    if pd.api.types.is_numeric_dtype(df["open_time"]):
        # Heuristic: values > 1e12 are milliseconds
        if df["open_time"].iloc[0] > 1e12:
            df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        else:
            df["open_time"] = pd.to_datetime(df["open_time"], unit="s", utc=True)
    else:
        df["open_time"] = pd.to_datetime(df["open_time"], utc=True, errors="coerce")

    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.dropna(subset=["open_time", "open", "high", "low", "close"], inplace=True)
    df.sort_values("open_time", inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


# ================================================================
# PUBLIC API
# ================================================================

def fetch_candles(symbol: str, tf: str, limit: int = 300) -> Optional[pd.DataFrame]:
    """
    Fetch OHLCV candles from MoonDev API.

    Args:
        symbol: Trading pair, e.g. "BTCUSDT"
        tf:     Timeframe string used by the TCT engine, e.g. "4h", "1h", "30m"
        limit:  Number of candles to return (max 1000)

    Returns:
        DataFrame with columns [open_time, open, high, low, close, volume]
        in the same format produced by fetch_candles_sync (MEXC feed), or
        None if the fetch fails so the caller can fall back to MEXC.
    """
    coin = _SYMBOL_MAP.get(symbol, symbol)
    interval = _INTERVAL_MAP.get(tf, tf)

    try:
        resp = _get(
            f"/api/candles/{coin}",
            params={"interval": interval, "limit": limit},
        )
        raw = resp.json()
        df = _parse_candle_response(raw)
        if df is not None and len(df) > 0:
            # Trim to requested limit (some APIs ignore limit param)
            if len(df) > limit:
                df = df.iloc[-limit:].reset_index(drop=True)
            logger.debug(
                "[MoonDev] Fetched %d candles for %s/%s", len(df), symbol, tf
            )
        return df
    except Exception as exc:
        logger.warning("[MoonDev] Candle fetch failed %s/%s: %s", symbol, tf, exc)
        return None


def fetch_live_price(symbol: str = "BTCUSDT") -> Optional[float]:
    """
    Fetch the current price from MoonDev /api/prices.

    Returns:
        Price as float, or None on error (caller should fall back to MEXC).
    """
    try:
        resp = _get("/api/prices")
        data = resp.json()

        # /api/prices returns data for 228 coins. Format could be:
        # - dict keyed by ticker: {"BTC": {"price": 65000, ...}, ...}
        # - list of dicts:        [{"coin": "BTC", "price": 65000, ...}, ...]
        short_ticker = _PRICE_TICKER_MAP.get(symbol, symbol.replace("USDT", ""))

        if isinstance(data, dict):
            # Try full symbol first, then short ticker
            for key in (symbol, short_ticker, symbol.lower(), short_ticker.lower()):
                entry = data.get(key)
                if entry is not None:
                    if isinstance(entry, dict):
                        for price_key in ("price", "last", "mark_price", "markPrice", "close"):
                            if price_key in entry:
                                return float(entry[price_key])
                    elif isinstance(entry, (int, float)):
                        return float(entry)

        elif isinstance(data, list):
            for item in data:
                if not isinstance(item, dict):
                    continue
                coin_name = (
                    item.get("coin") or item.get("symbol") or
                    item.get("ticker") or item.get("name") or ""
                ).upper()
                if coin_name in (symbol.upper(), short_ticker.upper()):
                    for price_key in ("price", "last", "mark_price", "markPrice", "close"):
                        if price_key in item:
                            return float(item[price_key])

        logger.warning("[MoonDev] Could not find price for %s in /api/prices response", symbol)
        return None

    except Exception as exc:
        logger.warning("[MoonDev] Live price fetch failed %s: %s", symbol, exc)
        return None


def is_enabled() -> bool:
    """Return True if MoonDev paper trading is enabled via env var."""
    return os.getenv("MOONDEV_PAPER_TRADING", "false").lower() in ("true", "1", "yes")
