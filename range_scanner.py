"""
range_scanner.py — HPB–TCT v21.2 Range Finder (MEXC Self-Healing Edition)
Robust MEXC HTF/LTF scanner with automatic symbol sanitation,
interval validation, and adaptive retry logic.
"""

import asyncio
import math
import statistics
import httpx
import logging
import os
from datetime import datetime

# ================================================================
# CONFIGURATION
# ================================================================
MEXC_URL = "https://api.mexc.com/api/v3/klines"
SYMBOL = os.getenv("SYMBOL", "BTCUSDT").replace("/", "").replace("-", "").upper()

# Valid MEXC intervals (case-sensitive)
VALID_INTERVALS = {
    "1s","1m","3m","5m","15m","30m","1h","2h","4h","6h","8h",
    "12h","1d","3d","1w","1M"
}

# Preferred scanning sets
LTF_INTERVALS = ["1m", "3m", "5m", "15m", "30m", "1h"]
HTF_INTERVALS = ["2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w"]

logger = logging.getLogger("HPB-TCT-RangeScanner")


# ================================================================
# RANGE CANDIDATE CONTAINER
# ================================================================
class RangeCandidate:
    def __init__(self, tf, high, low, candles):
        self.timeframe = tf
        self.range_high = high
        self.range_low = low
        self.eq = (high + low) / 2
        self.candles = candles
        self.score = 0.0


# ================================================================
# MEXC RANGE SCANNER
# ================================================================
class MEXCRangeScanner:
    def __init__(self, symbol=SYMBOL, limit=300):
        self.symbol = symbol
        self.limit = limit
        self.results = {"LTF": [], "HTF": []}
        self.paused = False
        self.current_tf = None
        logger.info(f"[INIT] HPB–TCT v21.2 MEXC Scanner initialized ({symbol})")

    # ------------------------------------------------------------
    # Safe candle fetcher with retries + self-healing
    # ------------------------------------------------------------
    async def fetch_klines(self, tf):
        if tf not in VALID_INTERVALS:
            logger.warning(f"[INVALID_TF] Skipping unsupported interval: {tf}")
            return []

        params = {"symbol": self.symbol, "interval": tf, "limit": self.limit}
        headers = {"User-Agent": "HPB-TCT-v21.2/MEXC"}

        for attempt in range(3):  # up to 3 retries
            try:
                async with httpx.AsyncClient(timeout=20, headers=headers) as c:
                    r = await c.get(MEXC_URL, params=params)
                    if r.status_code != 200:
                        logger.error(f"[MEXC_FAIL] {tf} — HTTP {r.status_code}")
                        await asyncio.sleep(1.5 * (attempt + 1))
                        continue

                    data = r.json()
                    if not isinstance(data, list) or not data:
                        logger.warning(f"[MEXC_EMPTY] No kline data for {self.symbol} {tf}")
                        await asyncio.sleep(1.5 * (attempt + 1))
                        continue

                    candles = [
                        {"t": int(x[0]), "h": float(x[2]), "l": float(x[3]), "c": float(x[4])}
                        for x in data
                    ]
                    return candles[::-1]

            except Exception as e:
                logger.error(f"[FETCH_ERROR] {tf}: {e}")
                await asyncio.sleep(1.5 * (attempt + 1))

        logger.error(f"[MEXC_ABORT] Failed to fetch {tf} after retries.")
        return []

    # ------------------------------------------------------------
    # Range detection + scoring
    # ------------------------------------------------------------
    def detect_range(self, candles):
        """Identify basic range high/low from recent candles."""
        if not candles:
            return None
        highs = [c["h"] for c in candles]
        lows = [c["l"] for c in candles]
        return max(highs), min(lows)

    def score_range(self, candles, high, low):
        """Basic smoothness + displacement scoring."""
        if not candles:
            return 0.0
        eq = (high + low) / 2
        diffs = [abs(c["c"] - eq) for c in candles]
        disp = statistics.pstdev(diffs) / (high - low + 1e-9)
        smoothness = 1 - min(disp, 1)
        t_disp = min(len(candles) / 300, 1)
        score = 0.5 * smoothness + 0.5 * t_disp
        return round(score, 3)

    # ------------------------------------------------------------
    # Core scan loop
    # ------------------------------------------------------------
    async def scan_timeframes(self, group_name, tfs):
        for tf in tfs:
            if self.paused:
                self.current_tf = tf
                logger.info(f"[PAUSE] Scanner paused at {tf}")
                return
            candles = await self.fetch_klines(tf)
            if not candles:
                continue
            rng = self.detect_range(candles)
            if not rng:
                continue
            high, low = rng
            sc = self.score_range(candles, high, low)
            rc = RangeCandidate(tf, high, low, candles)
            rc.score = sc
            self.results[group_name].append(rc)
            logger.info(f"[SCAN] {group_name} {tf} | Score={sc}")
            await asyncio.sleep(1.0)  # mild delay to avoid rate limits

        self.results[group_name].sort(key=lambda x: x.score, reverse=True)
        self.results[group_name] = self.results[group_name][:3]

    async def run_scan(self):
        await asyncio.gather(
            self.scan_timeframes("LTF", LTF_INTERVALS),
            self.scan_timeframes("HTF", HTF_INTERVALS),
        )
        logger.info("[SCAN_COMPLETE] ✅ Range scan completed successfully.")
        return self.results

    # ------------------------------------------------------------
    # Pause/resume logic
    # ------------------------------------------------------------
    def pause(self):
        self.paused = True
        logger.info("[PAUSE] Scanner paused manually.")

    async def resume(self):
        self.paused = False
        if self.current_tf:
            idx = (LTF_INTERVALS + HTF_INTERVALS).index(self.current_tf)
            remaining = (LTF_INTERVALS + HTF_INTERVALS)[idx:]
            group = "LTF" if self.current_tf in LTF_INTERVALS else "HTF"
            logger.info(f"[RESUME] Resuming from {self.current_tf}")
            await self.scan_timeframes(group, remaining)
