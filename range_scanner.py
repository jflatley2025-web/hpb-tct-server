"""
range_scanner.py – HPB–TCT v19.2 Range Finder Core
Scans Bybit HTF & LTF candles → detects 3 best ranges per category
"""

import asyncio, math, statistics, httpx
from datetime import datetime

BYBIT_URL = "https://api.bybit.com/v5/market/kline"

LTF_INTERVALS = ["1", "3", "5", "15", "30", "60"]
HTF_INTERVALS = ["120", "240", "360", "720", "D", "W", "M"]

class RangeCandidate:
    def __init__(self, tf, high, low, candles):
        self.timeframe = tf
        self.range_high = high
        self.range_low = low
        self.eq = (high + low) / 2
        self.candles = candles
        self.score = 0.0

class BybitRangeScanner:
    def __init__(self, symbol="BTCUSDT", category="linear", limit=300):
        self.symbol = symbol
        self.category = category
        self.limit = limit
        self.results = {"LTF": [], "HTF": []}
        self.paused = False
        self.current_tf = None

    async def fetch_klines(self, tf):
        params = {
            "category": self.category,
            "symbol": self.symbol,
            "interval": tf,
            "limit": self.limit,
        }
        async with httpx.AsyncClient(timeout=15) as c:
            r = await c.get(BYBIT_URL, params=params)
            data = r.json().get("result", {}).get("list", [])
        # candles: [open_time, open, high, low, close, volume]
        candles = [
            {
                "t": int(x[0]),
                "h": float(x[2]),
                "l": float(x[3]),
                "c": float(x[4])
            }
            for x in data
        ]
        return candles[::-1]  # chronological order

    def detect_range(self, candles):
        highs = [c["h"] for c in candles]
        lows = [c["l"] for c in candles]
        if not highs or not lows:
            return None
        high, low = max(highs), min(lows)
        return high, low

    def score_range(self, candles, high, low):
        """ crude v1 RCM-compatible range score """
        eq = (high + low) / 2
        diffs = [abs(c["c"] - eq) for c in candles]
        disp = statistics.pstdev(diffs) / (high - low + 1e-9)
        # smooth liquidity curve → low stdev = higher score
        smoothness = 1 - min(disp, 1)
        # time-displacement proxy
        t_disp = min(len(candles) / 300, 1)
        return round(0.5 * smoothness + 0.5 * t_disp, 3)

    async def scan_timeframes(self, group_name, tfs):
        for tf in tfs:
            if self.paused:
                self.current_tf = tf
                return
            candles = await self.fetch_klines(tf)
            rng = self.detect_range(candles)
            if rng:
                high, low = rng
                sc = self.score_range(candles, high, low)
                rc = RangeCandidate(tf, high, low, candles)
                rc.score = sc
                self.results[group_name].append(rc)
            await asyncio.sleep(1.2)  # avoid rate limit

        # sort and keep top 3
        self.results[group_name].sort(key=lambda x: x.score, reverse=True)
        self.results[group_name] = self.results[group_name][:3]

    async def run_scan(self):
        await asyncio.gather(
            self.scan_timeframes("LTF", LTF_INTERVALS),
            self.scan_timeframes("HTF", HTF_INTERVALS)
        )
        return self.results

    def pause(self):
        self.paused = True

    async def resume(self):
        self.paused = False
        if self.current_tf:
            idx = (LTF_INTERVALS + HTF_INTERVALS).index(self.current_tf)
            await self.scan_timeframes("LTF" if self.current_tf in LTF_INTERVALS else "HTF",
                                       (LTF_INTERVALS + HTF_INTERVALS)[idx:])
