"""OHLCV window loader — fetches candle data around a trade's time range.

Uses MEXC public API (no auth required). Completely standalone.
Can also load from local CSV cache to avoid repeated API calls.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import requests


# MEXC interval mapping (MEXC uses "60m" for 1h)
_MEXC_INTERVAL = {
    "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
    "1h": "60m", "2h": "120m", "4h": "4h", "6h": "6h",
    "8h": "8h", "12h": "12h", "1d": "1d", "1w": "1w",
}

# Timeframe duration in seconds
TF_SECONDS = {
    "1m": 60, "5m": 300, "15m": 900, "30m": 1800,
    "1h": 3600, "2h": 7200, "4h": 14400, "6h": 21600,
    "8h": 28800, "12h": 43200, "1d": 86400, "1w": 604800,
}

MEXC_API_URL = "https://api.mexc.com/api/v3/klines"
MAX_CANDLES_PER_REQUEST = 1000
REQUEST_DELAY = 0.5  # seconds between paginated requests


class OhlcvWindow:
    """Loads and caches an OHLCV candle window around a trade.

    The window extends before the trade entry (for range/tap detection)
    and after the trade exit (for context).
    """

    def __init__(self, cache_dir: Optional[str | Path] = None):
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load(
        self,
        symbol: str,
        timeframe: str,
        center_time: datetime,
        lookback_bars: int = 200,
        lookahead_bars: int = 50,
    ) -> pd.DataFrame:
        """Load OHLCV window centered around a time.

        Args:
            symbol: e.g. "BTCUSDT"
            timeframe: e.g. "1h", "4h"
            center_time: the trade entry time
            lookback_bars: bars before center_time (for range detection)
            lookahead_bars: bars after center_time (for exit context)

        Returns:
            DataFrame with columns: open_time, open, high, low, close, volume
            Sorted by open_time ascending.
        """
        if timeframe not in TF_SECONDS:
            raise ValueError(f"Unsupported timeframe: {timeframe}")

        tf_sec = TF_SECONDS[timeframe]
        start_time = center_time - timedelta(seconds=tf_sec * lookback_bars)
        end_time = center_time + timedelta(seconds=tf_sec * lookahead_bars)

        # Check cache first
        cached = self._load_cache(symbol, timeframe, start_time, end_time)
        if cached is not None:
            return cached

        # Fetch from MEXC
        df = self._fetch_range(symbol, timeframe, start_time, end_time)

        # Cache for reuse
        if self.cache_dir and not df.empty:
            self._save_cache(symbol, timeframe, start_time, end_time, df)

        return df

    def _fetch_range(
        self,
        symbol: str,
        timeframe: str,
        start_time: datetime,
        end_time: datetime,
    ) -> pd.DataFrame:
        """Fetch candles from MEXC with pagination."""
        interval = _MEXC_INTERVAL.get(timeframe, timeframe)
        tf_ms = TF_SECONDS[timeframe] * 1000

        start_ms = int(start_time.timestamp() * 1000)
        end_ms = int(end_time.timestamp() * 1000)

        all_rows = []
        since_ms = start_ms

        while since_ms < end_ms:
            try:
                resp = requests.get(
                    MEXC_API_URL,
                    params={
                        "symbol": symbol,
                        "interval": interval,
                        "startTime": since_ms,
                        "endTime": end_ms,
                        "limit": MAX_CANDLES_PER_REQUEST,
                    },
                    timeout=20,
                    headers={"User-Agent": "LocalChartOverlay/1.0"},
                )
                resp.raise_for_status()
                batch = resp.json()
            except (requests.RequestException, json.JSONDecodeError) as e:
                raise ConnectionError(f"MEXC API error for {symbol} {timeframe}: {e}")

            if not batch:
                break

            for row in batch:
                ts = int(row[0])
                if start_ms <= ts <= end_ms:
                    all_rows.append(row)

            last_ts = int(batch[-1][0])
            if last_ts <= since_ms:
                break
            since_ms = last_ts + tf_ms

            if since_ms < end_ms:
                time.sleep(REQUEST_DELAY)

        if not all_rows:
            return pd.DataFrame(
                columns=["open_time", "open", "high", "low", "close", "volume"]
            )

        df = pd.DataFrame(
            [[r[0], r[1], r[2], r[3], r[4], r[5]] for r in all_rows],
            columns=["open_time", "open", "high", "low", "close", "volume"],
        )
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        for col in ("open", "high", "low", "close", "volume"):
            df[col] = df[col].astype(float)

        df = df.drop_duplicates(subset=["open_time"]).sort_values("open_time")
        return df.reset_index(drop=True)

    # ── Local cache ───────────────────────────────────────────────────

    def _cache_key(
        self, symbol: str, timeframe: str,
        start: datetime, end: datetime,
    ) -> str:
        s = int(start.timestamp())
        e = int(end.timestamp())
        return f"{symbol}_{timeframe}_{s}_{e}"

    def _load_cache(
        self, symbol: str, timeframe: str,
        start: datetime, end: datetime,
    ) -> Optional[pd.DataFrame]:
        if not self.cache_dir:
            return None
        key = self._cache_key(symbol, timeframe, start, end)
        # Try parquet first, fall back to CSV
        parquet_path = self.cache_dir / f"{key}.parquet"
        csv_path = self.cache_dir / f"{key}.csv"
        try:
            if parquet_path.exists():
                return pd.read_parquet(parquet_path)
        except ImportError:
            pass
        if csv_path.exists():
            df = pd.read_csv(csv_path, parse_dates=["open_time"])
            df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
            return df
        return None

    def _save_cache(
        self, symbol: str, timeframe: str,
        start: datetime, end: datetime,
        df: pd.DataFrame,
    ):
        if not self.cache_dir:
            return
        key = self._cache_key(symbol, timeframe, start, end)
        try:
            df.to_parquet(self.cache_dir / f"{key}.parquet", index=False)
        except ImportError:
            # Fallback to CSV if pyarrow/fastparquet not installed
            df.to_csv(self.cache_dir / f"{key}.csv", index=False)

    @staticmethod
    def load_from_csv(path: str | Path) -> pd.DataFrame:
        """Load OHLCV from a CSV file (offline/testing use).

        Expects columns: open_time (or timestamp), open, high, low, close, volume
        """
        df = pd.read_csv(path)
        # Normalize column names
        renames = {}
        for col in df.columns:
            lc = col.lower().strip()
            if lc in ("timestamp", "time", "date", "datetime"):
                renames[col] = "open_time"
            elif lc == "open":
                renames[col] = "open"
            elif lc == "high":
                renames[col] = "high"
            elif lc == "low":
                renames[col] = "low"
            elif lc == "close":
                renames[col] = "close"
            elif lc == "volume":
                renames[col] = "volume"
        df = df.rename(columns=renames)

        if "open_time" not in df.columns:
            raise ValueError("CSV must have a timestamp/open_time column")

        df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
        for col in ("open", "high", "low", "close", "volume"):
            if col in df.columns:
                df[col] = df[col].astype(float)

        return df.sort_values("open_time").reset_index(drop=True)
