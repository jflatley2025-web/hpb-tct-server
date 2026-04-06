"""
mexc_data.py — Shared MEXC data utilities and trading constants
================================================================
Provides candle fetching, live price, and configuration constants
used by the active 5B engine and server_mexc.

Extracted from the retired 5A_tct_trader.py so that shared utilities
survive the removal of the dead 5A trading engine.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ================================================================
# CONFIGURATION CONSTANTS
# ================================================================

STARTING_BALANCE: float = float(os.getenv("TENSOR_TCT_STARTING_BALANCE", "5000"))
RISK_PER_TRADE_PCT: float = 1.0          # % of balance risked per trade
AUTO_SCAN_INTERVAL: int = int(os.getenv("TENSOR_TCT_SCAN_INTERVAL", "300"))
DUPLICATE_COOLDOWN_SECONDS: int = 3600   # 1 hour
DUPLICATE_PRICE_TOLERANCE: float = 0.001  # 0.1%
MIN_RR: float = 1.5
MIN_QUALITY_SCORE: float = 0.70
ENTRY_THRESHOLD: int = 50  # Fixed — never adapts

# ================================================================
# MEXC API ENDPOINTS
# ================================================================

MEXC_KLINES_URL = "https://api.mexc.com/api/v3/klines"
MEXC_TICKER_URL = "https://api.mexc.com/api/v3/ticker/price"

# ================================================================
# TRADE LOG PATHS
# ================================================================

_DEFAULT_LOG_DIR = "/opt/render/project/chroma_db"
_LOG_DIR = (os.getenv("TENSOR_TCT_LOG_DIR", _DEFAULT_LOG_DIR) or "").strip() or _DEFAULT_LOG_DIR
try:
    os.makedirs(_LOG_DIR, exist_ok=True)
except OSError:
    _LOG_DIR = os.path.dirname(os.path.abspath(__file__))

TRADE_LOG_PATH = os.path.join(_LOG_DIR, "tensor_trade_log.json")
TRADE_LOG_BACKUP_PATH = os.path.join(_LOG_DIR, "tensor_trade_log_backup.json")

# ================================================================
# MEXC DATA FETCHING
# ================================================================

_MEXC_INTERVAL_MAP: Dict[str, str] = {"1h": "60m", "2h": "4h"}


def fetch_candles_sync(symbol: str, tf: str, limit: int = 300) -> Optional[pd.DataFrame]:
    """
    Fetch OHLCV candles from MEXC synchronously.

    Args:
        symbol: e.g. "BTCUSDT"
        tf: timeframe e.g. "4h" (or "1h" — normalized to MEXC's "60m")
        limit: number of candles (max 1000)

    Returns:
        DataFrame with columns [open_time, open, high, low, close, volume], or None on error.
    """
    mexc_interval = _MEXC_INTERVAL_MAP.get(tf, tf)
    try:
        resp = requests.get(
            MEXC_KLINES_URL,
            params={"symbol": symbol, "interval": mexc_interval, "limit": limit},
            timeout=20,
            headers={"User-Agent": "HPB-TCT-Server/1.0"},
        )
        resp.raise_for_status()
        raw = resp.json()
        if not isinstance(raw, list) or not raw:
            logger.warning("[MEXC-FETCH] Empty response for %s/%s", symbol, tf)
            return None

        rows = [[r[0], r[1], r[2], r[3], r[4], r[5]] for r in raw]
        df = pd.DataFrame(rows, columns=["open_time", "open", "high", "low", "close", "volume"])
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        for col in ("open", "high", "low", "close", "volume"):
            df[col] = df[col].astype(float)
        df.reset_index(drop=True, inplace=True)
        return df
    except Exception as exc:
        logger.warning("[MEXC-FETCH] Candle fetch failed %s/%s: %s", symbol, tf, exc)
        return None


def fetch_live_price(symbol: str = "BTCUSDT") -> Optional[float]:
    """
    Fetch the current mid price from MEXC ticker.

    Returns:
        Price as float, or None on error.
    """
    try:
        resp = requests.get(
            MEXC_TICKER_URL,
            params={"symbol": symbol},
            timeout=10,
            headers={"User-Agent": "HPB-TCT-Server/1.0"},
        )
        resp.raise_for_status()
        data = resp.json()
        return float(data["price"])
    except Exception as exc:
        logger.warning("[MEXC-FETCH] Live price fetch failed %s: %s", symbol, exc)
        return None
