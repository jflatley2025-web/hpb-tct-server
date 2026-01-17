# ================================================================
# server.py — HPB–TCT v19.3 AutoLearn + Range Scanner Dashboard (Stable Build)
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


# ────────────────────────────────────────────────────────────────
# Logging setup
# ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("HPB-TCT-Server")

# ────────────────────────────────────────────────────────────────
# FastAPI setup
# ────────────────────────────────────────────────────────────────
app = FastAPI(title="HPB–TCT AutoLearn v19.3", version="1.0.6")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ────────────────────────────────────────────────────────────────
# Persistent AutoLearn state
# ────────────────────────────────────────────────────────────────
STATE_FILE = os.path.join(os.getcwd(), "hpb_autolearn_state.json")

def load_state():
    if not os.path.exists(STATE_FILE):
        return {
            "last_timestamp": None,
            "train_cycles_completed": 0,
            "last_RIG_status": None,
            "last_bias": None,
            "last_confidence": None,
            "bias_history": [],
        }
    try:
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {}

def save_state(state: dict):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)

state = load_state()

# ────────────────────────────────────────────────────────────────
# Environment Initialization
# ────────────────────────────────────────────────────────────────
logger.info("🔧 Initializing HPB–TCT Environment (v19.3)...")
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
# RANGE SCANNER IMPLEMENTATION (INTEGRATED)
# ================================================================
BYBIT_URL = "https://api.bybit.com/v5/market/kline"
LTF_INTERVALS = ["1", "3", "5", "15", "30", "60"]
HTF_INTERVALS = ["120", "240", "360", "720", "D", "W"]  # no 'M' for stability

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
        """Fetch OHLC data safely from Bybit API (with fallback + user-agent)"""
        params = {
            "category": self.category,
            "symbol": self.symbol,
            "interval": tf,
            "limit": self.limit,
        }
        headers = {"User-Agent": "HPB-TCT-v19.3/Render"}
        try:
            async with httpx.AsyncClient(timeout=30, headers=headers) as c:
                r = await c.get(BYBIT_URL, params=params)
                res = r.json()
                data = res.get("result", {}).get("list")

                # Fallback: retry with category=spot if no data
                if not data:
                    print(f"[WARN] No data for {tf} ({self.category}), trying spot fallback.")
                    params["category"] = "spot"
                    r2 = await c.get(BYBIT_URL, params=params)
                    res = r2.json()
                    data = res.get("result", {}).get("list", [])
                    if not data:
                        print(f"[FAIL] Still no data for {tf} in spot fallback.")
                        return []
        except Exception as e:
            print(f"[ERROR] fetch_klines({tf}) failed → {e}")
            return []

        try:
            candles = [
                {"t": int(x[0]), "h": float(x[2]), "l": float(x[3]), "c": float(x[4])}
                for x in data
            ]
            print(f"[OK] {tf} fetched {len(candles)} candles.")
            return candles[::-1]
        except Exception as e:
            print(f"[PARSE ERROR] {tf}: {e}")
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
            await asyncio.sleep(1.5)
        self.results[group_name].sort(key=lambda x: x.score, reverse=True)
        self.results[group_name] = self.results[group_name][:3]

    async def run_scan(self):
        await asyncio.gather(
            self.scan_timeframes("LTF", LTF_INTERVALS),
            self.scan_timeframes("HTF", HTF_INTERVALS),
        )
        return self.results


# ================================================================
# API ROUTES
# ================================================================
@app.get("/")
async def root():
    return {
        "status": "running",
        "environment": "HPB–TCT v19.3 AutoLearn",
        "initialized": env is not None,
        "train_cycles_completed": state.get("train_cycles_completed", 0),
        "last_bias": state.get("last_bias"),
        "last_confidence": state.get("last_confidence"),
    }

@app.get("/status")
async def status():
    return {
        "server": "HPB–TCT v19.3",
        "initialized": env is not None,
        "train_cycles_completed": state.get("train_cycles_completed", 0),
        "last_RIG_status": state.get("last_RIG_status"),
        "heartbeat": datetime.utcnow().isoformat(),
    }

@app.get("/api/ranges")
async def get_ranges():
    scanner = BybitRangeScanner()
    results = await scanner.run_scan()
    data = {
        "LTF": [
            {"timeframe": r.timeframe, "range_high": r.range_high,
             "range_low": r.range_low, "eq": r.eq, "score": r.score}
            for r in results["LTF"]
        ],
        "HTF": [
            {"timeframe": r.timeframe, "range_high": r.range_high,
             "range_low": r.range_low, "eq": r.eq, "score": r.score}
            for r in results["HTF"]
        ]
    }
    return JSONResponse(content=data)


# ================================================================
# DASHBOARD VISUALIZATION
# ================================================================
@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    html = """
    <html>
      <head>
        <title>📊 Market Structure Range Dashboard</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
          body {font-family:sans-serif;background:#0d1117;color:#eee;margin:2em;}
          .chart{width:100%;height:400px;margin-bottom:3em;}
          button{margin:4px;padding:6px 10px;background:#1f6feb;color:white;border:0;border-radius:6px;}
        </style>
      </head>
      <body>
        <h1>📊 Market Structure Range Dashboard</h1>
        <div id="ltf" class="chart"></div>
        <button onclick="loadExtra('LTF',2)">Show LTF #2</button>
        <button onclick="loadExtra('LTF',3)">Show LTF #3</button>
        <hr>
        <div id="htf" class="chart"></div>
        <button onclick="loadExtra('HTF',2)">Show HTF #2</button>
        <button onclick="loadExtra('HTF',3)">Show HTF #3</button>

        <script>
          async function fetchRanges(){
              const res = await fetch('/api/ranges');
              const data = await res.json();
              console.log("Fetched:", data);
              return data;
          }
          function plotRange(div, data){
              if (!data.length){Plotly.newPlot(div, [], {title:'No range data available'});return;}
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
# TRAINING AND VALIDATION ROUTES
# ================================================================
@app.get("/train")
async def train_agent(episodes: int = 5):
    if env is None:
        return {"error": "Environment not initialized."}
    try:
        logger.info(f"🚀 Starting AutoLearn training for {episodes} episodes...")
        if hasattr(env, "auto_train"):
            env.auto_train(episodes=episodes)
        elif hasattr(env, "train"):
            env.train(episodes)
        elif hasattr(env, "simulate"):
            env.simulate(episodes)
        elif hasattr(env, "run"):
            env.run(episodes)
        else:
            return {"warning": "No training function available."}
        state["train_cycles_completed"] = state.get("train_cycles_completed", 0) + episodes
        state["last_timestamp"] = datetime.utcnow().isoformat()
        save_state(state)
        return {"status": "completed", "episodes": episodes, "state": state}
    except Exception as e:
        logger.error(f"Training error: {e}")
        return {"error": str(e)}

@app.get("/validate_gates")
async def validate_gates():
    try:
        context = {
            "gates": {"1A": {"bias": "bearish"}, "RCM": {"valid": True}, "MSCE": {"session_bias": "bullish"}},
            "local_range_displacement": 0.12,
        }
        result = range_integrity_validator(context)
        state.update({
            "last_timestamp": datetime.utcnow().isoformat(),
            "last_RIG_status": result.get("status"),
            "last_bias": result.get("htf_bias"),
            "last_confidence": result.get("confidence"),
        })
        save_state(state)
        return {"RIG_Validation": result}
    except Exception as e:
        logger.error(f"Gate validation error: {e}")
        return {"error": str(e)}

# ================================================================
# ENTRY POINT
# ================================================================
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)
