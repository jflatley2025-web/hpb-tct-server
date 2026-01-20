# ================================================================
# tensortrade_env_mexc.py — HPB–TCT v21.2 (MEXC Data Environment)
# ================================================================
# Purpose:
# Provides live MEXC OHLCV data feeds for TensorTrade simulations.
# Automatically detects MEXC credentials, uses the same intervals
# as server_mexc.py, and returns normalized observation arrays.
# ================================================================

import os
import time
import httpx
import numpy as np
import pandas as pd
from datetime import datetime, timezone

# ================================================================
# CONFIGURATION
# ================================================================
MEXC_URL_BASE = os.getenv("MEXC_URL_BASE", "https://api.mexc.com")
SYMBOL = os.getenv("SYMBOL", "BTCUSDT")
MEXC_KEY = os.getenv("MEXC_KEY")
MEXC_SECRET = os.getenv("MEXC_SECRET")

AUTH_MODE = bool(MEXC_KEY and MEXC_SECRET)
print(f"[TT-ENV] MEXC Auth Mode: {'🔒 PRIVATE' if AUTH_MODE else '🌐 PUBLIC'}")
print(f"[TT-ENV] Using symbol: {SYMBOL}")

# Timeframes supported by TensorTrade training loops
VALID_INTERVALS = {
    "1m","3m","5m","15m","30m","1h","2h","4h","6h",
    "8h","12h","1d","3d","1w","1M"
}

DEFAULT_INTERVAL = os.getenv("TRAIN_INTERVAL", "1h")

# ================================================================
# UTILITY FUNCTIONS
# ================================================================
async def fetch_mexc_ohlcv(symbol=SYMBOL, interval=DEFAULT_INTERVAL, limit=1000):
    """Fetch OHLCV candles from MEXC API."""
    if interval not in VALID_INTERVALS:
        print(f"[TT-ENV] Unsupported interval: {interval}")
        return None
    url = f"{MEXC_URL_BASE}/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    try:
        async with httpx.AsyncClient(timeout=15) as c:
            r = await c.get(url, params=params)
            if r.status_code != 200:
                print(f"[TT-ENV] HTTP {r.status_code} for {symbol} {interval}")
                return None
            data = r.json()
            if not data:
                return None
            df = pd.DataFrame(data, columns=[
                "open_time","open","high","low","close","volume",
                "close_time","quote_vol","trades","taker_base",
                "taker_quote","ignore"
            ])
            df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
            df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
            for col in ["open","high","low","close","volume"]:
                df[col] = df[col].astype(float)
            df = df[["open_time","open","high","low","close","volume"]]
            df = df.sort_values("open_time")
            return df
    except Exception as e:
        print(f"[TT-ENV] Error fetching MEXC data: {e}")
        return None


def normalize(df: pd.DataFrame):
    """Normalize OHLCV features for stable training."""
    if df is None or df.empty:
        return np.zeros((1, 5))
    arr = df[["open","high","low","close","volume"]].values
    min_v = np.min(arr, axis=0)
    max_v = np.max(arr, axis=0)
    norm = (arr - min_v) / (max_v - min_v + 1e-9)
    return norm


async def get_tensortrade_ready(symbol=SYMBOL, interval=DEFAULT_INTERVAL, limit=512):
    """Return a ready-to-train numpy tensor of normalized OHLCV data."""
    df = await fetch_mexc_ohlcv(symbol, interval, limit)
    if df is None:
        return np.zeros((limit, 5))
    norm = normalize(df)
    print(f"[TT-ENV] {symbol} {interval} shape={norm.shape}")
    return norm


# ================================================================
# SYNCHRONOUS WRAPPER (for TensorTrade training loops)
# ================================================================
def fetch_training_batch(symbol=SYMBOL, interval=DEFAULT_INTERVAL, limit=512):
    """Blocking wrapper for environments that are not async."""
    import asyncio
    return asyncio.run(get_tensortrade_ready(symbol, interval, limit))


# ================================================================
# ENTRY TEST
# ================================================================
if __name__ == "__main__":
    print("[TT-ENV] Testing TensorTrade MEXC Environment...")
    data = fetch_training_batch(SYMBOL, "1h", 256)
    print(f"[TT-ENV] Loaded {data.shape[0]} candles for {SYMBOL}")
