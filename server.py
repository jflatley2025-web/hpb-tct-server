# ================================================================
# server.py — HPB–TCT v19.5 (Bybit + OKX Hybrid Range Dashboard)
# ================================================================

import os
import json
import logging
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
# LOGGING
# ================================================================
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("HPB-TCT-Server")

# ================================================================
# FASTAPI
# ================================================================
app = FastAPI(title="HPB–TCT AutoLearn v19.5", version="1.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# ================================================================
# STATE MANAGEMENT
# ================================================================
STATE_FILE = os.path.join(os.getcwd(), "hpb_autolearn_state.json")

def load_state():
    if not os.path.exists(STATE_FILE):
        return {"train_cycles_completed": 0, "bias_history": []}
    try:
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {}

def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)

state = load_state()

# ================================================================
# ENVIRONMENT INITIALIZATION
# ================================================================
logger.info("🔧 Initializing HPB–TCT Environment (v19.5 Enhanced Logic)...")
try:
    env = AUTO_INIT()
    logger.info(f"✅ Environment initialized successfully with config: {TENSORTRADE_CONFIG}")
except Exception as e:
    logger.error(f"❌ Failed to initialize HPB environment: {e}")
    env = None

if env is not None and not any(hasattr(env, fn) for fn in ["auto_train", "train", "simulate", "run"]):
    def dummy_train(episodes=5):
        logger.info(f"[DUMMY_TRAIN] Simulating {episodes} pseudo-episodes.")
        return {"episodes": episodes, "reward": 0.0}
    env.train = dummy_train

# ================================================================
# RANGE SCANNER — BYBIT + OKX HYBRID
# ================================================================
BYBIT_URL = "https://api.bybit.com/v5/market/kline"
OKX_URL = "https://www.okx.com/api/v5/market/candles"

LTF_INTERVALS = ["1", "3", "5", "15", "30", "60"]
HTF_INTERVALS = ["120", "240", "360", "720", "D", "W"]

class RangeCandidate:
    def __init__(self, tf, high, low, candles):
        self.timeframe = tf
        self.range_high = high
        self.range_low = low
        self.eq = (high + low) / 2
        self.candles = candles
        self.score = 0.0

class RangeScanner:
    def __init__(self, symbol="BTCUSDT", limit=200):
        self.symbol = symbol
        self.limit = limit
        self.results = {"LTF": [], "HTF": []}
        self.paused = False
        self.current_tf = None

    async def fetch_klines(self, tf):
        headers = {"User-Agent": "HPB-TCT-v19.5/Hybrid"}
        BYBIT_CANDIDATES = [
            ("linear", self.symbol),
            ("linear", self.symbol.replace("-", "")),
            ("inverse", self.symbol.replace("USDT", "USD")),
        ]
        data = []

        # --- Try Bybit first ---
        try:
            async with httpx.AsyncClient(timeout=20, headers=headers) as c:
                for category, sym in BYBIT_CANDIDATES:
                    params = {"category": category, "symbol": sym, "interval": tf, "limit": self.limit}
                    r = await c.get(BYBIT_URL, params=params)
                    if r.status_code == 403:
                        logger.warning(f"[BYBIT_BLOCKED] {sym} {tf} — 403 Forbidden")
                        data = []
                        break
                    j = r.json()
                    if j.get("retCode") == 0:
                        data = j.get("result", {}).get("list", [])
                        if data:
                            logger.info(f"[BYBIT_OK] ✅ {category.upper()} {sym} {tf} got {len(data)} candles.")
                            break
                    await asyncio.sleep(0.3)
        except Exception as e:
            logger.warning(f"[BYBIT_FAIL] {tf}: {e}")
            data = []

        # --- If Bybit failed, use OKX fallback ---
        if not data:
            try:
                okx_tf = {
                    "1": "1m", "3": "3m", "5": "5m", "15": "15m",
                    "30": "30m", "60": "1H", "120": "2H", "240": "4H",
                    "360": "6H", "720": "12H", "D": "1D", "W": "1W"
                }.get(tf, "1H")
                params = {"instId": f"{self.symbol.replace('-', '')}-SWAP", "bar": okx_tf, "limit": str(self.limit)}
                async with httpx.AsyncClient(timeout=20, headers=headers) as c:
                    r = await c.get(OKX_URL, params=params)
                    j = r.json()
                    data = j.get("data", [])
                    if data:
                        logger.info(f"[OKX_OK] ✅ Fallback success: {len(data)} candles from OKX ({okx_tf}).")
            except Exception as e:
                logger.error(f"[OKX_FAIL] {tf}: {e}")
                return []

        # --- Parse candles ---
        try:
            candles = [
                {"t": int(x[0]), "h": float(x[2]), "l": float(x[3]), "c": float(x[4])}
                for x in data
            ]
            return candles[::-1]
        except Exception as e:
            logger.error(f"[PARSE_ERROR] {tf}: {e}")
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

    async def scan_timeframes(self, group, tfs):
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
            self.results[group].append(rc)
            logger.info(f"[SCAN] {group} {tf} | Score={sc}")
            await asyncio.sleep(1)
        self.results[group].sort(key=lambda x: x.score, reverse=True)
        self.results[group] = self.results[group][:3]

    async def run_scan(self):
        await asyncio.gather(
            self.scan_timeframes("LTF", LTF_INTERVALS),
            self.scan_timeframes("HTF", HTF_INTERVALS),
        )
        logger.info("[SCAN_COMPLETE] ✅ Range scan completed successfully.")
        return self.results

# ================================================================
# ROUTES
# ================================================================
@app.get("/")
async def root():
    return {"status": "running", "environment": "HPB–TCT v19.5"}

@app.get("/status")
async def status():
    return {"server": "HPB–TCT v19.5", "heartbeat": datetime.utcnow().isoformat()}

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
    <html>
    <head>
        <title>Market Structure Range Dashboard</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body { font-family: Arial; background-color: #0d1117; color: #eee; margin: 30px; }
            .chart { width: 100%; height: 400px; margin-bottom: 40px; }
            button { background-color: #1f6feb; border: none; padding: 8px 12px; color: white; border-radius: 5px; margin: 5px; cursor: pointer; }
            button:hover { background-color: #388bfd; }
        </style>
    </head>
    <body>
        <h1>📊 Market Structure Range Dashboard (v19.5)</h1>
        <div id="ltf" class="chart"></div>
        <button onclick="loadExtra('LTF',2)">Show LTF #2</button>
        <button onclick="loadExtra('LTF',3)">Show LTF #3</button>
        <hr style="margin:40px 0;">
        <div id="htf" class="chart"></div>
        <button onclick="loadExtra('HTF',2)">Show HTF #2</button>
        <button onclick="loadExtra('HTF',3)">Show HTF #3</button>

        <script>
            async function fetchRanges(){
                const res = await fetch('/api/ranges');
                return res.json();
            }
            function plotRange(div, data){
                if(!data.length){Plotly.newPlot(div, [], {title:'No range data found'});return;}
                const r = data[0];
                const trace = {x:['Low','EQ','High'],y:[r.range_low,r.eq,r.range_high],
                               type:'scatter',mode:'lines+markers',line:{color:'#00ccff'}};
                Plotly.newPlot(div,[trace],{title:`${div.toUpperCase()} Best Range (${r.timeframe}) | Score ${r.score}`});
            }
            async function init(){
                const data = await fetchRanges();
                plotRange('ltf', data.LTF);
                plotRange('htf', data.HTF);
                window._ranges=data;
            }
            function loadExtra(group,n){
                const arr=window._ranges[group];
                if(!arr[n-1])return;
                const r=arr[n-1];
                const trace={x:['Low','EQ','High'],y:[r.range_low,r.eq,r.range_high],
                             type:'scatter',mode:'lines+markers',line:{color:'#ffcc00'}};
                Plotly.addTraces(group.toLowerCase(),trace);
            }
            init();
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
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)
