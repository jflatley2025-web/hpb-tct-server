"""
range_scanner.py — HPB–TCT v19.3 Range Finder (Stable)
Robust Bybit HTF/LTF scanner with safe error handling.
"""

import asyncio, math, statistics, httpx
from datetime import datetime

BYBIT_URL = "https://api.bybit.com/v5/market/kline"

# Timeframes supported by Bybit v5 linear category
LTF_INTERVALS = ["1", "3", "5", "15", "30", "60"]
HTF_INTERVALS = ["120", "240", "360", "720", "D", "W"]  # dropped M for stability

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

    async def fetch_klines(self, tf):
        """Fetch OHLC data safely from Bybit API"""
        params = {
            "category": self.category,
            "symbol": self.symbol,
            "interval": tf,
            "limit": self.limit,
        }
        try:
            async with httpx.AsyncClient(timeout=30) as c:
                r = await c.get(BYBIT_URL, params=params)
                r.raise_for_status()
                res = r.json()
                data = res.get("result", {}).get("list")
                if not data:
                    print(f"[WARN] No data for tf {tf}")
                    return []
        except Exception as e:
            print(f"[ERROR] fetch_klines({tf}) failed → {e}")
            return []

        # Parse [open_time, open, high, low, close, volume]
        try:
            candles = [
                {
                    "t": int(x[0]),
                    "h": float(x[2]),
                    "l": float(x[3]),
                    "c": float(x[4]),
                }
                for x in data
            ]
            return candles[::-1]  # chronological order
        except Exception as e:
            print(f"[PARSE ERROR] {tf}: {e}")
            return []

    def detect_range(self, candles):
        """Identify basic range high/low from recent candles"""
        if not candles:
            return None
        highs = [c["h"] for c in candles]
        lows = [c["l"] for c in candles]
        return max(highs), min(lows)

    def score_range(self, candles, high, low):
        """Basic RCM-compatible smoothness + displacement scoring"""
        if not candles:
            return 0.0
        eq = (high + low) / 2
        diffs = [abs(c["c"] - eq) for c in candles]
        disp = statistics.pstdev(diffs) / (high - low + 1e-9)
        smoothness = 1 - min(disp, 1)
        t_disp = min(len(candles) / 300, 1)
        score = 0.5 * smoothness + 0.5 * t_disp
        return round(score, 3)

    async def scan_timeframes(self, group_name, tfs):
        for tf in tfs:
            if self.paused:
                self.current_tf = tf
                print(f"[PAUSE] Paused at {tf}")
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
            await asyncio.sleep(1.5)  # avoid rate limit

        self.results[group_name].sort(key=lambda x: x.score, reverse=True)
        self.results[group_name] = self.results[group_name][:3]

    async def run_scan(self):
        await asyncio.gather(
            self.scan_timeframes("LTF", LTF_INTERVALS),
            self.scan_timeframes("HTF", HTF_INTERVALS),
        )
        return self.results

    def pause(self):
        self.paused = True

    async def resume(self):
        self.paused = False
        if self.current_tf:
            idx = (LTF_INTERVALS + HTF_INTERVALS).index(self.current_tf)
            remaining = (LTF_INTERVALS + HTF_INTERVALS)[idx:]
            group = "LTF" if self.current_tf in LTF_INTERVALS else "HTF"
            await self.scan_timeframes(group, remaining)
