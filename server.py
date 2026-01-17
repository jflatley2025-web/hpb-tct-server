# ================================================================
# server.py — HPB–TCT v19.3 AutoLearn + Range Scanner Dashboard
# ================================================================

import os
import json
import logging
from datetime import datetime
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
import asyncio

from tensortrade_env import HPB_TensorTrade_Env
from tensortrade_config_ext import AUTO_INIT, TENSORTRADE_CONFIG
from hpb_rig_validator import range_integrity_validator

# Range Scanner Integration
from range_scanner import BybitRangeScanner

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
app = FastAPI(title="HPB–TCT AutoLearn v19.3", version="1.0.5")

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
        logger.info("🧠 Creating new AutoLearn state file.")
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
    except Exception as e:
        logger.error(f"⚠️ Failed to load state: {e}")
        return {}

def save_state(state: dict):
    try:
        with open(STATE_FILE, "w") as f:
            json.dump(state, f, indent=2)
        logger.info("💾 AutoLearn state saved successfully.")
    except Exception as e:
        logger.error(f"❌ Failed to save state: {e}")

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

# Dummy fallback training if env has none
if env is not None and not any(hasattr(env, fn) for fn in ["auto_train", "train", "simulate", "run"]):
    def dummy_train(episodes=5):
        logger.info(f"[DUMMY_TRAIN] Simulating {episodes} pseudo-episodes.")
        return {"episodes": episodes, "reward": 0.0}
    env.train = dummy_train

# ────────────────────────────────────────────────────────────────
# Core Routes
# ────────────────────────────────────────────────────────────────
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

@app.head("/")
async def head_root():
    return {"status": "ok"}

@app.get("/status")
async def status():
    """Server + AutoLearn state overview"""
    return {
        "server": "HPB–TCT v19.3",
        "initialized": env is not None,
        "last_state_update": state.get("last_timestamp"),
        "train_cycles_completed": state.get("train_cycles_completed", 0),
        "bias": state.get("last_bias"),
        "confidence": state.get("last_confidence"),
        "RIG_status": state.get("last_RIG_status"),
        "heartbeat": datetime.utcnow().isoformat(),
    }

# ────────────────────────────────────────────────────────────────
# RANGE SCANNER API + DASHBOARD
# ────────────────────────────────────────────────────────────────
@app.get("/api/ranges")
async def get_ranges():
    """Returns top 3 LTF and HTF ranges from Bybit scan"""
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

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    html = """
    <html>
      <head>
        <title>HPB–TCT v19.3 Range Dashboard</title>
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
          async function fetchRanges(){return await fetch('/api/ranges').then(r=>r.json());}
          function plotRange(div, data){
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

# ────────────────────────────────────────────────────────────────
# TRAIN + GATE VALIDATION + SUMMARY (unchanged)
# ────────────────────────────────────────────────────────────────
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
            logger.warning("⚠️ No recognized training function found.")
            return {"warning": "No training function available."}

        state["train_cycles_completed"] = state.get("train_cycles_completed", 0) + episodes
        state["last_timestamp"] = datetime.utcnow().isoformat()
        bias = state.get("last_bias", "neutral")
        conf = state.get("last_confidence", 0.0)
        history = state.get("bias_history", [])
        history.append({"bias": bias, "confidence": conf, "ts": state["last_timestamp"]})
        state["bias_history"] = history[-10:]
        save_state(state)
        return {"status": "completed", "episodes": episodes, "state": state}
    except Exception as e:
        logger.error(f"Training error: {e}")
        return {"error": str(e)}

@app.get("/validate_gates")
async def validate_gates():
    try:
        context = {
            "gates": {
                "1A": {"bias": "bearish"},
                "RCM": {"valid": True, "range_duration_hours": 36},
                "MSCE": {"session_bias": "bullish", "session": "NY"},
                "1D": {"score": 0.85},
            },
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

# ────────────────────────────────────────────────────────────────
# ENTRY POINT
# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    logger.info(f"🚀 Starting HPB–TCT v19.3 server on port {port} ...")
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)
