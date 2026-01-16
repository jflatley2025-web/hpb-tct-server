# ================================================================
# server_v19.2.py — HPB–TCT AutoLearn Orchestrator + Telegram Sync
# ================================================================
# • TensorTrade / HPB extended runtime environment
# • Persistent state (AutoLearn cycles + RIG validation)
# • Telegram bot integration routes (/bot/status, /bot/train)
# • Render-compatible health + graceful startup
# ================================================================

import os
import json
import logging
from datetime import datetime
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional

# ────────────────────────────────────────────────────────────────
# Imports: HPB TensorTrade + Validators
# ────────────────────────────────────────────────────────────────
from tensortrade_env import HPB_TensorTrade_Env
from tensortrade_config_ext import AUTO_INIT, TENSORTRADE_CONFIG
from hpb_rig_validator import range_integrity_validator

TensorTradeEnv = HPB_TensorTrade_Env

# ────────────────────────────────────────────────────────────────
# Logging Configuration
# ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("HPB-TCT-Server")

# ────────────────────────────────────────────────────────────────
# FastAPI App Setup
# ────────────────────────────────────────────────────────────────
app = FastAPI(title="HPB–TCT AutoLearn v19.2", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ────────────────────────────────────────────────────────────────
# Persistent AutoLearn State
# ────────────────────────────────────────────────────────────────
STATE_FILE = os.path.join(os.getcwd(), "hpb_autolearn_state.json")

def load_state():
    if not os.path.exists(STATE_FILE):
        logger.info("🧠 Creating new AutoLearn state file.")
        return {
            "version": "v19.2",
            "last_timestamp": None,
            "train_cycles_completed": 0,
            "last_RIG_status": "UNKNOWN",
            "last_bias": "neutral",
            "last_confidence": 0.0,
            "heartbeat": None,
        }
    try:
        with open(STATE_FILE, "r") as f:
            state = json.load(f)
            return state
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
logger.info("🔧 Initializing HPB–TCT Environment (v19.2)...")
try:
    env = AUTO_INIT()
    logger.info(f"✅ Environment initialized successfully with config: {TENSORTRADE_CONFIG}")
except Exception as e:
    logger.error(f"❌ Environment initialization failed: {e}")
    env = None

# ────────────────────────────────────────────────────────────────
# Utility Functions
# ────────────────────────────────────────────────────────────────
def heartbeat_update():
    """Updates the heartbeat timestamp in the persistent state."""
    state["heartbeat"] = datetime.utcnow().isoformat()
    save_state(state)

def get_safe_state():
    """Return safe, Telegram-compatible summary."""
    return {
        "version": state.get("version", "v19.2"),
        "bias": state.get("last_bias", "neutral"),
        "confidence": state.get("last_confidence", 0.0),
        "RIG_status": state.get("last_RIG_status", "UNKNOWN"),
        "train_cycles": state.get("train_cycles_completed", 0),
        "timestamp": state.get("last_timestamp"),
        "heartbeat": state.get("heartbeat"),
    }

# ────────────────────────────────────────────────────────────────
# Root / Health Routes
# ────────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    """Basic health check."""
    heartbeat_update()
    return {
        "status": "running",
        "environment": "HPB–TCT AutoLearn v19.2",
        "initialized": env is not None,
        "heartbeat": state.get("heartbeat"),
    }

@app.head("/")
async def head_root():
    """Render HEAD health check."""
    return {"status": "ok"}

@app.get("/status")
async def status():
    heartbeat_update()
    return get_safe_state()

# ────────────────────────────────────────────────────────────────
# Dashboard Route
# ────────────────────────────────────────────────────────────────
@app.get("/dashboard")
async def dashboard():
    """Summary endpoint for front-end or Telegram bot display."""
    return get_safe_state()

# ────────────────────────────────────────────────────────────────
# FETCH CONTEXT
# ────────────────────────────────────────────────────────────────
@app.get("/fetch_context")
async def fetch_context():
    """Retrieve environment snapshot (safe for cold start)."""
    try:
        if not env:
            return {"error": "Environment not initialized."}

        context_data = {}
        if hasattr(env, "price_feed"):
            context_data = env.price_feed
        elif hasattr(env, "exchange"):
            context_data = env.exchange.__dict__
        elif hasattr(env, "current_step"):
            context_data = {"step": env.current_step}

        if not context_data:
            context_data = {
                "symbol": getattr(env, "symbol", "BTC-USDT"),
                "interval": getattr(env, "interval", "1H"),
                "window_size": getattr(env, "window_size", 100),
                "timestamp": datetime.utcnow().isoformat(),
            }

        heartbeat_update()
        return {"context": context_data}
    except Exception as e:
        logger.error(f"Context fetch error: {e}")
        return {"error": str(e)}

# ────────────────────────────────────────────────────────────────
# VALIDATE GATES
# ────────────────────────────────────────────────────────────────
@app.get("/validate_gates")
async def validate_gates():
    """Runs RIG validation and updates state."""
    try:
        context = {
            "gates": {
                "1A": {"bias": "bearish"},
                "RCM": {"valid": True, "range_duration_hours": 36},
                "MSCE": {"session_bias": "bullish", "session": "NY"},
            },
            "local_range_displacement": 0.12,
        }
        result = range_integrity_validator(context)

        state.update({
            "last_timestamp": datetime.utcnow().isoformat(),
            "last_RIG_status": result.get("status", "OK"),
            "last_bias": result.get("htf_bias", "neutral"),
            "last_confidence": result.get("confidence", 0.0),
        })
        save_state(state)
        heartbeat_update()
        return {"RIG_Validation": result}
    except Exception as e:
        logger.error(f"Gate validation error: {e}")
        return {"error": str(e)}

# ────────────────────────────────────────────────────────────────
# TRAINING ENDPOINT
# ────────────────────────────────────────────────────────────────
@app.get("/train")
async def train_agent(episodes: int = 5):
    """Executes AutoLearn cycles and updates persistent state."""
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

        state["train_cycles_completed"] = state.get("train_cycles_completed", 0) + episodes
        state["last_timestamp"] = datetime.utcnow().isoformat()
        save_state(state)
        heartbeat_update()

        return {"status": "completed", "episodes": episodes, "state": get_safe_state()}
    except Exception as e:
        logger.error(f"Training error: {e}")
        return {"error": str(e)}

# ────────────────────────────────────────────────────────────────
# TELEGRAM BOT SYNC ENDPOINTS
# ────────────────────────────────────────────────────────────────
@app.get("/bot/ping")
async def bot_ping():
    """Telegram bot heartbeat check."""
    heartbeat_update()
    return {"bot_status": "online", "server": "HPB–TCT v19.2", "heartbeat": state.get("heartbeat")}

@app.get("/bot/status")
async def bot_status():
    """Telegram-friendly status endpoint."""
    heartbeat_update()
    return get_safe_state()

@app.get("/bot/train")
async def bot_train(episodes: Optional[int] = 3):
    """Telegram-triggered training request."""
    response = await train_agent(episodes)
    heartbeat_update()
    return response

# ────────────────────────────────────────────────────────────────
# STATE MANAGEMENT
# ────────────────────────────────────────────────────────────────
@app.get("/state")
async def get_state():
    """Returns full AutoLearn state JSON."""
    return state

@app.post("/reset_state")
async def reset_state():
    """Resets AutoLearn state."""
    global state
    state = {
        "version": "v19.2",
        "last_timestamp": None,
        "train_cycles_completed": 0,
        "last_RIG_status": "UNKNOWN",
        "last_bias": "neutral",
        "last_confidence": 0.0,
        "heartbeat": datetime.utcnow().isoformat(),
    }
    save_state(state)
    return {"status": "reset", "state": state}

# ────────────────────────────────────────────────────────────────
# ENTRY POINT
# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"🚀 Starting HPB–TCT AutoLearn v19.2 server on port {port} ...")
    uvicorn.run("server_v19_2:app", host="0.0.0.0", port=port, reload=False)
