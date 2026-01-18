# ================================================================
# server.py — HPB–TCT v19.8b (Env Compatibility + OKX Auth Fix)
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

from tensortrade_env import HPB_TensorTrade_Env
from tensortrade_config_ext import AUTO_INIT, TENSORTRADE_CONFIG
from hpb_rig_validator import range_integrity_validator

# ================================================================
# CONFIGURATION
# ================================================================
# 🔧 Auto-detect both naming styles for Render or local environments
OKX_API_KEY = os.getenv("OKX_API_KEY") or os.getenv("OKX_KEY", "")
OKX_SECRET_KEY = os.getenv("OKX_SECRET_KEY") or os.getenv("OKX_SECRET", "")
OKX_PASSPHRASE = os.getenv("OKX_PASSPHRASE", "")

SYMBOL = "BTC-USDT-SWAP"
OKX_BASE = "https://www.okx.com"
OKX_PATH = "/api/v5/market/candles"

LTF_INTERVALS = ["1m", "3m", "5m", "15m", "30m", "1H"]
HTF_INTERVALS = ["2H", "4H", "6H", "12H", "1D", "1W"]

CACHE_FILE = "range_cache.json"

# ================================================================
# FASTAPI APP
# ================================================================
app = FastAPI(title="HPB–TCT v19.8b", version="1.5.1")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# ================================================================
# ENV INITIALIZATION
# ================================================================
print("🔧 Initializing HPB–TCT Environment (v19.8b)...")
try:
    env = AUTO_INIT()
    print(f"✅ Environment initialized successfully with config: {TENSORTRADE_CONFIG}")
except Exception as e:
    print(f"⚠️ Environment initialization failed: {e}")
    env = None

# ================================================================
# OKX SIGNING (millisecond precision)
# ================================================================
def okx_headers(method, path, query=""):
    ts = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    msg = f"{ts}{method}{path}{query}"
    sign = base64.b64encode(
        hmac.new(OKX_SECRET_KEY.encode(), msg.encode(), hashlib.sha256).digest()
    ).decode()
    return {
        "OK-ACCESS-KEY": OKX_API_KEY,
        "OK-ACCESS-SIGN": sign,
        "OK-ACCESS-TIMESTAMP": ts,
        "OK-ACCESS-PASSPHRASE": OKX_PASSPHRASE,
        "Content-Type": "application/json",
        "User-Agent": "HPB-TCT-v19.8b",
    }

async def verify_okx_auth():
    """Check if OKX API key works using /account/balance."""
    path = "/api/v5/account/balance"
    query = "?ccy=USDT"
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.get(f"{OKX_BASE}{path}{query}", headers=okx_headers("GET", path, query))
            if r.status_code == 200:
                return True
            else:
                print(f"[OKX_AUTH_FAIL] HTTP {r.status_code}: {r.text}")
                return False
    except Exception as e:
        print(f"[OKX_AUTH_ERROR] {e}")
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
        query = f"?instId={self.symbol}&bar={tf}&limit={self.limit}"
        try:
            async with httpx.AsyncClient(timeout=20) as c:
                r = await c.get(f"{OKX_BASE}{OKX_PATH}{query}", headers=okx_headers("GET", OKX_PATH, query))
                if r.status_code != 200:
                    print(f"[OKX_FAIL] {tf} — HTTP {r.status_code}: {r.text}")
                    return []
                j = r.json()
                data = j.get("data", [])
                if not data:
                    print(f"[OKX_EMPTY] {tf}")
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
            await asyncio.sleep(0.8)
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
# CACHE HANDLING
# ================================================================
def save_cache(data):
    try:
        with open(CACHE_FILE, "w") as f:
            json.dump(data, f)
    except Exception as e:
        print(f"[CACHE_WRITE_ERROR] {e}")

def load_cache():
    if not os.path.exists(CACHE_FILE):
        return {"LTF": [], "HTF": []}
    try:
        with open(CACHE_FILE, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"[CACHE_READ_ERROR] {e}")
        return {"LTF": [], "HTF": []}

# ================================================================
# ROUTES
# ================================================================
@app.get("/")
async def root():
    return {"status": "running", "version": "19.8b"}

@app.get("/status")
async def status():
    verified = await verify_okx_auth()
    return {
        "server": "HPB–TCT v19.8b",
        "okx_auth": "✅ Verified" if verified else "❌ Invalid",
        "symbol": SYMBOL,
        "timestamp": datetime.utcnow().isoformat(),
    }

@app.get("/api/ranges")
async def get_ranges():
    scanner = RangeScanner()
    try:
        results = await scanner.run_scan()
        data = {
            g: [
                {"timeframe": r.timeframe, "range_high": r.range_high, "range_low": r.range_low, "eq": r.eq, "score": r.score}
                for r in results[g]
            ]
            for g in results
        }
        save_cache(data)
        return JSONResponse(content=data)
    except Exception as e:
        print(f"[SCAN_ERROR] {e}")
        cached = load_cache()
        return JSONResponse(content=cached)

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard_page():
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>📊 HPB–TCT Dashboard v19.8b</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body { background-color: #0d1117; color: #eee; font-family: Arial; margin: 30px; }
            #status { background:#161b22; padding:10px; border-radius:10px; margin-bottom:20px; }
            .chart { width: 100%; height: 400px; margin-bottom: 40px; }
        </style>
    </head>
    <body>
        <h1>📈 Market Structure Range Dashboard (v19.8b)</h1>
        <div id="status">Checking OKX Auth...</div>
        <div id="ltf" class="chart"></div>
        <div id="htf" class="chart"></div>

        <script>
            async function fetchStatus(){ return fetch('/status').then(r=>r.json()); }
            async function fetchRanges(){ return fetch('/api/ranges').then(r=>r.json()); }
            function plotRange(div, data){
                if(!data.length){Plotly.newPlot(div,[],{title:'No range data found'});return;}
                const r=data[0];
                const trace={x:['Low','EQ','High'],y:[r.range_low,r.eq,r.range_high],
                             type:'scatter',mode:'lines+markers',line:{color:'#00ccff'}};
                Plotly.newPlot(div,[trace],{title:`${div.toUpperCase()} | ${r.timeframe} | Score ${r.score}`});
            }
            async function refresh(){
                const s=await fetchStatus();
                document.getElementById('status').innerHTML=
                    `<b>Server:</b> ${s.server}<br><b>Symbol:</b> ${s.symbol}<br><b>OKX Auth:</b> ${s.okx_auth}<br><b>Last Updated:</b> ${new Date().toLocaleTimeString()}`;
                const d=await fetchRanges();
                plotRange('ltf',d.LTF);
                plotRange('htf',d.HTF);
            }
            refresh();
            setInterval(refresh,60000);
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html)

# ================================================================
# ENTRY POINT
# ================================================================
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("server:app", host="0.0.0.0", port=port)
