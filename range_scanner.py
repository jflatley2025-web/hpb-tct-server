"""
range_scanner.py — HPB–TCT v19.4 Range Finder (Stable)
Robust Bybit HTF/LTF scanner with safe error handling + pause/resume support.
"""

import asyncio
import math
import statistics
import httpx
import logging
from datetime import datetime

BYBIT_URL = "https://api.bybit.com/v5/market/kline"

# Timeframes supported by Bybit v5
LTF_INTERVALS = ["1", "3", "5", "15", "30", "60"]
HTF_INTERVALS = ["120", "240", "360", "720", "D", "W"]  # Monthly dropped for stability

logger = logging.getLogger("HPB-TCT-RangeScanner")


class RangeCandidate:
    def __init__(self, tf, high, low, candles):
        self.timeframe = tf
        self.range_high = high
        self.range_low = low
        self.eq = (high + low) / 2
        self.candles = candles
        self.score = 0.0


class BybitRangeScanner:
    def __init__(self, symbol="BTCUSDT", category="linear", limit=200):
        self.symbol = symbol
        self.category = category
        self.limit = limit
        self.results = {"LTF": [], "HTF": []}
        self.paused = False
        self.current_tf = None

    # ------------------------------------------------------------
    # Safe candle fetcher with category fallbacks
    # ------------------------------------------------------------
    async def fetch_klines(self, tf):
        params = {"symbol": self.symbol, "interval": tf, "limit": self.limit}
        headers = {"User-Agent": "HPB-TCT-v19.4/Enhanced"}
        data = []
        try:
            async with httpx.AsyncClient(timeout=30, headers=headers) as c:
                # Try linear first
                params["category"] = "linear"
                r = await c.get(BYBIT_URL, params=params)
                data = r.json().get("result", {}).get("list", [])

                # Try inverse if no result
                if not data:
                    params["category"] = "inverse"
                    r2 = await c.get(BYBIT_URL, params=params)
                    data = r2.json().get("result", {}).get("list", [])

                # Try spot as final fallback
                if not data:
                    params["category"] = "spot"
                    r3 = await c.get(BYBIT_URL, params=params)
                    data = r3.json().get("result", {}).get("list", [])

            if not data:
                logger.warning(f"[BYBIT_EMPTY] No kline data for {self.symbol} {tf}")
                return []

            candles = [
                {"t": int(x[0]), "h": float(x[2]), "l": float(x[3]), "c": float(x[4])}
                for x in data
            ]
            return candles[::-1]
        except Exception as e:
            logger.error(f"[FETCH_ERROR] {tf}: {e}")
            return []

    # ------------------------------------------------------------
    # Range detection + scoring
    # ------------------------------------------------------------
    def detect_range(self, candles):
        """Identify basic range high/low from recent candles"""
        if not candles:
            return None
        highs = [c["h"] for c in candles]
        lows = [c["l"] for c in candles]
        return max(highs), min(lows)

    def score_range(self, candles, high, low):
        """Basic smoothness + displacement scoring"""
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
            await asyncio.sleep(1.5)  # avoid rate-limit issues

        self.results[group_name].sort(key=lambda x: x.score, reverse=True)
        self.results[group_name] = self.results[group_name][:3]

    async def run_scan(self):
        await asyncio.gather(
            self.scan_timeframes("LTF", LTF_INTERVALS),
            self.scan_timeframes("HTF", HTF_INTERVALS),
        )
        logger.info("[SCAN_COMPLETE] Range scan completed successfully.")
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
