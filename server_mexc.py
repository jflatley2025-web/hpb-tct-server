# ================================================================
# server_mexc.py — HPB–TCT v21.1 (MEXC Feed + Auto Auth Detection)
# ================================================================

import os
import json
import asyncio
import statistics
import httpx
import hmac
import hashlib
from datetime import datetime
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ================================================================
# CONFIGURATION
# ================================================================
MEXC_URL_BASE = os.getenv("MEXC_URL_BASE", "https://api.mexc.com")
SYMBOL = os.getenv("SYMBOL", "BTCUSDT")
MEXC_KEY = os.getenv("MEXC_KEY")
MEXC_SECRET = os.getenv("MEXC_SECRET")

AUTH_MODE = bool(MEXC_KEY and MEXC_SECRET)
print(f"[INIT] MEXC Auth Mode: {'🔒PRIVATE' if AUTH_MODE else '🌐PUBLIC'}")

LTF_INTERVALS = ["1m", "5m", "15m", "30m", "1h"]
HTF_INTERVALS = ["4h", "6h", "12h", "1d", "1w"]

# ================================================================
# FASTAPI APP
# ================================================================
app = FastAPI(title="HPB–TCT v21.1", version="2.1.1")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# ================================================================
# MEXC AUTH SIGNING (for private endpoints)
# ================================================================
def mexc_sign(params: dict, secret: str):
    """Return query string + signature for MEXC private requests."""
    query = "&".join([f"{k}={v}" for k, v in sorted(params.items())])
    signature = hmac.new(secret.encode(), query.encode(), hashlib.sha256).hexdigest()
    return query + "&signature=" + signature

async def get_account_info():
    """Fetch MEXC account info if keys available."""
    if not AUTH_MODE:
        return {"status": "PUBLIC_MODE", "note": "No API keys configured."}
    ts = int(datetime.utcnow().timestamp() * 1000)
    params = {"timestamp": ts}
    query = mexc_sign(params, MEXC_SECRET)
    url = f"{MEXC_URL_BASE}/api/v3/account?{query}"
    headers = {"X-MEXC-APIKEY": MEXC_KEY}
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.get(url, headers=headers)
            if r.status_code == 200:
                return {"status": "OK", "data": r.json()}
            return {"status": f"ERROR {r.status_code}", "detail": r.text}
    except Exception as e:
        return {"status": "EXCEPTION", "detail": str(e)}

# ================================================================
# RANGE SCANNER (Public)
# ================================================================
class RangeCandidate:
    def __init__(self, tf, high, low, candles):
        self.timeframe = tf
        self.range_high = high
        self.range_low = low
        self.eq = (high + low) / 2
        self.candles = candles
        self.score = 0.0


class MEXCRangeScanner:
    def __init__(self, symbol=SYMBOL, limit=500):
        self.symbol = symbol
        self.limit = limit
        self.results = {"LTF": [], "HTF": []}

    async def fetch_mexc(self, tf):
        url = f"{MEXC_URL_BASE}/api/v3/klines"
        params = {"symbol": self.symbol, "interval": tf, "limit": self.limit}
        try:
            async with httpx.AsyncClient(timeout=15) as c:
                r = await c.get(url, params=params)
                if r.status_code != 200:
                    print(f"[MEXC_FAIL] {tf} — HTTP {r.status_code}")
                    return []
                data = r.json()
                if not data:
                    return []
                candles = [
                    {"t": int(x[0]), "o": float(x[1]), "h": float(x[2]),
                     "l": float(x[3]), "c": float(x[4]), "v": float(x[5])}
                    for x in data
                ]
                return candles
        except Exception as e:
            print(f"[MEXC_ERROR] {tf}: {e}")
            return []

    def detect_range(self, candles):
        if not candles:
            return None
        highs = [c["h"] for c in candles]
        lows = [c["l"] for c in candles]
        return max(highs), min(lows)

    def score_range(self, candles, high, low):
        if not candles:
            return 0.0
        eq = (high + low) / 2
        diffs = [abs(c["c"] - eq) for c in candles]
        disp = statistics.pstdev(diffs) / (high - low + 1e-9)
        smoothness = 1 - min(disp, 1)
        t_disp = min(len(candles) / 300, 1)
        return round(0.5 * smoothness + 0.5 * t_disp, 3)

    async def scan(self, group, tfs):
        for tf in tfs:
            candles = await self.fetch_mexc(tf)
            if not candles:
                continue
            rng = self.detect_range(candles)
            if not rng:
                continue
            high, low = rng
            sc = self.score_range(candles, high, low)
            rc = RangeCandidate(tf, high, low, candles)
            rc.score = sc
            self.results[group].append(rc)
            print(f"[SCAN] {group} {tf} | Score={sc}")
            await asyncio.sleep(0.5)
        self.results[group].sort(key=lambda x: x.score, reverse=True)
        self.results[group] = self.results[group][:3]

    async def run_scan(self):
        await asyncio.gather(
            self.scan("LTF", LTF_INTERVALS),
            self.scan("HTF", HTF_INTERVALS),
        )
        print("[SCAN_COMPLETE] ✅ Range scan completed.")
        return self.results

# ================================================================
# LIVE PRICE
# ================================================================
async def get_live_price(symbol=SYMBOL):
    url = f"{MEXC_URL_BASE}/api/v3/ticker/price"
    params = {"symbol": symbol}
    try:
        async with httpx.AsyncClient(timeout=5) as c:
            r = await c.get(url, params=params)
            if r.status_code != 200:
                return None
            data = r.json()
            return float(data.get("price", 0))
    except Exception as e:
        print(f"[PRICE_ERROR] {e}")
        return None

# ================================================================
# ROUTES
# ================================================================
@app.get("/")
async def root():
    price = await get_live_price()
    return {
        "server": "HPB–TCT v21.1",
        "exchange": "MEXC",
        "auth_mode": "PRIVATE" if AUTH_MODE else "PUBLIC",
        "symbol": SYMBOL,
        "last_price": price,
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/status")
async def status():
    return {"status": "ok", "time": datetime.utcnow().isoformat()}

@app.get("/api/ranges")
async def get_ranges():
    scanner = MEXCRangeScanner()
    results = await scanner.run_scan()
    return JSONResponse(content={
        g: [{"timeframe": r.timeframe, "range_high": r.range_high,
             "range_low": r.range_low, "eq": r.eq, "score": r.score}
            for r in results[g]] for g in results
    })

@app.get("/api/account")
async def get_account():
    return await get_account_info()

@app.get("/debug/env")
async def debug_env():
    keys = ["MEXC_URL_BASE", "SYMBOL", "MEXC_KEY", "MEXC_SECRET"]
    return {"loaded": {k: bool(os.getenv(k)) for k in keys}}

# ================================================================
# ENTRY POINT
# ================================================================
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("server_mexc:app", host="0.0.0.0", port=port, reload=False)
