# ================================================================
# server.py — HPB–TCT v19.8-debug (OKX Env Check + Auth Validation)
# ================================================================

import os
import json
import hmac
import base64
import hashlib
import asyncio
import statistics
import httpx
from datetime import datetime
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse

# ================================================================
# CONFIGURATION
# ================================================================
OKX_API_KEY = os.getenv("OKX_API_KEY") or os.getenv("OKX_KEY") or ""
OKX_SECRET_KEY = os.getenv("OKX_SECRET_KEY") or os.getenv("OKX_SECRET") or ""
OKX_PASSPHRASE = os.getenv("OKX_PASSPHRASE") or ""
OKX_URL = "https://www.okx.com/api/v5/market/candles"
SYMBOL = os.getenv("SYMBOL", "BTC-USDT-SWAP")

LTF_INTERVALS = ["1m", "3m", "5m", "15m", "30m", "1H"]
HTF_INTERVALS = ["2H", "4H", "6H", "12H", "1D", "1W"]

# ================================================================
# FASTAPI APP
# ================================================================
app = FastAPI(title="HPB–TCT AutoLearn v19.8-debug", version="1.3.1")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# ================================================================
# OKX REQUEST SIGNING
# ================================================================
def okx_headers(path, method="GET"):
    ts = datetime.utcnow().isoformat("T", "milliseconds") + "Z"
    msg = f"{ts}{method}{path}"
    sign = base64.b64encode(
        hmac.new(OKX_SECRET_KEY.encode(), msg.encode(), hashlib.sha256).digest()
    ).decode()
    return {
        "OK-ACCESS-KEY": OKX_API_KEY,
        "OK-ACCESS-SIGN": sign,
        "OK-ACCESS-TIMESTAMP": ts,
        "OK-ACCESS-PASSPHRASE": OKX_PASSPHRASE,
        "Content-Type": "application/json",
        "User-Agent": "HPB-TCT-v19.8/Render",
    }

async def verify_okx_auth():
    """Check if OKX API credentials work by calling /api/v5/account/balance"""
    url = "https://www.okx.com/api/v5/account/balance"
    path = "/api/v5/account/balance"
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(url, headers=okx_headers(path))
            if r.status_code == 200 and "data" in r.json():
                return True
            print(f"[OKX_AUTH_FAIL] HTTP {r.status_code}: {r.text}")
            return False
    except Exception as e:
        print(f"[OKX_AUTH_EXCEPTION] {e}")
        return False

# ================================================================
# RANGE SCANNER
# ================================================================
class RangeCandidate:
    def __init__(self, tf, high, low, candles):
        self.timeframe = tf
        self.range_high = high
        self.range_low = low
        self.eq = (high + low) / 2
        self.candles = candles
        self.score = 0.0

class RangeScanner:
    def __init__(self, symbol=SYMBOL, limit=200):
        self.symbol = symbol
        self.limit = limit
        self.results = {"LTF": [], "HTF": []}

    async def fetch_okx(self, tf):
        params = {"instId": self.symbol, "bar": tf, "limit": str(self.limit)}
        path = f"/api/v5/market/candles?instId={self.symbol}&bar={tf}&limit={self.limit}"
        try:
            async with httpx.AsyncClient(timeout=20) as c:
                r = await c.get(OKX_URL, params=params, headers=okx_headers(path))
                if r.status_code != 200:
                    print(f"[OKX_FAIL] {tf} — HTTP {r.status_code}")
                    return []
                data = r.json().get("data", [])
                if not data:
                    print(f"[OKX_EMPTY] {tf} — No data returned.")
                    return []
                candles = [
                    {"t": int(x[0]), "h": float(x[2]), "l": float(x[3]), "c": float(x[4])}
                    for x in data
                ]
                return candles[::-1]
        except Exception as e:
            print(f"[OKX_ERROR] {tf}: {e}")
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
            candles = await self.fetch_okx(tf)
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
            await asyncio.sleep(1)
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
# ROUTES
# ================================================================
@app.get("/")
async def root():
    auth_ok = await verify_okx_auth()
    return {
        "server": "HPB–TCT v19.8-debug",
        "okx_auth": "✅ Verified" if auth_ok else "❌ Invalid",
        "symbol": SYMBOL,
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/api/ranges")
async def get_ranges():
    scanner = RangeScanner()
    results = await scanner.run_scan()
    data = {
        g: [
            {"timeframe": r.timeframe, "range_high": r.range_high, "range_low": r.range_low, "eq": r.eq, "score": r.score}
            for r in results[g]
        ]
        for g in results
    }
    return JSONResponse(content=data)

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard_page():
    html = """
    <!DOCTYPE html>
    <html><head>
    <title>Market Structure Range Dashboard (v19.8-debug)</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>body{font-family:Arial;background-color:#0d1117;color:#eee;margin:30px;}
    .chart{width:100%;height:400px;margin-bottom:40px;}
    button{background-color:#1f6feb;border:none;padding:8px 12px;color:white;border-radius:5px;margin:5px;cursor:pointer;}
    button:hover{background-color:#388bfd;}
    #status{margin-top:10px;font-size:14px;color:#ccc;}</style></head>
    <body><h1>📊 Market Structure Range Dashboard (v19.8-debug)</h1>
    <div id="status">Loading...</div><div id="ltf" class="chart"></div><div id="htf" class="chart"></div>
    <script>
    async function fetchRanges(){const r=await fetch('/api/ranges');return r.json();}
    function plotRange(div,data){if(!data.length){Plotly.newPlot(div,[],{title:'No range data found'});return;}
    const r=data[0];const trace={x:['Low','EQ','High'],y:[r.range_low,r.eq,r.range_high],
    type:'scatter',mode:'lines+markers',line:{color:'#00ccff'}};Plotly.newPlot(div,[trace],{title:`${div.toUpperCase()} | ${r.timeframe} | Score ${r.score}`});}
    async function refresh(){document.getElementById('status').innerText='⏳ Updating... '+new Date().toLocaleTimeString();
    const d=await fetchRanges();plotRange('ltf',d.LTF);plotRange('htf',d.HTF);
    document.getElementById('status').innerText='✅ Updated '+new Date().toLocaleTimeString();}
    refresh();setInterval(refresh,60000);
    </script></body></html>
    """
    return HTMLResponse(content=html)

@app.get("/debug/env")
async def debug_env():
    keys = ["OKX_API_KEY", "OKX_SECRET_KEY", "OKX_KEY", "OKX_SECRET", "OKX_PASSPHRASE"]
    loaded = {k: bool(os.getenv(k)) for k in keys}
    return {"loaded": loaded}

# ================================================================
# ENTRY POINT
# ================================================================
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)
