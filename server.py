# ================================================================
# server.py — HPB–TCT v19.1 AutoLearn Runtime (Render + Telegram Safe)
# ================================================================
# • Dynamic environment + RIG validation + AutoLearn persistence
# • Render-compatible port binding & /status health path
# • Telegram bot–safe responses (non-null /dashboard, /context)
# ================================================================

import os
import json
import logging
from datetime import datetime
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# ────────────────────────────────────────────────────────────────
# Imports (Environment + Validator)
# ────────────────────────────────────────────────────────────────
from tensortrade_env import HPB_TensorTrade_Env
from tensortrade_config_ext import AUTO_INIT, TENSORTRADE_CONFIG
from hpb_rig_validator import range_integrity_validator

# Backward compatibility alias
TensorTradeEnv = HPB_TensorTrade_Env

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
app = FastAPI(title="HPB–TCT AutoLearn v19.1", version="1.0.4")

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
        logger.info("🧠 No prior AutoLearn state found, creating new file.")
        return {
            "last_timestamp": None,
            "train_cycles_completed": 0,
            "last_RIG_status": None,
            "last_bias": "neutral",
            "last_confidence": 0.0,
        }
    try:
        with open(STATE_FILE, "r") as f:
            state = json.load(f)
            logger.info("✅ Loaded AutoLearn persistent state.")
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
logger.info("🔧 Initializing HPB–TCT TensorTrade Environment...")
try:
    env = AUTO_INIT()
    logger.info(f"✅ Environment initialized successfully with config: {TENSORTRADE_CONFIG}")
except Exception as e:
    logger.error(f"❌ Failed to initialize HPB environment: {e}")
    env = None

# ────────────────────────────────────────────────────────────────
# API ROUTES
# ────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    """Health check + environment confirmation"""
    return {
        "status": "running",
        "environment": "HPB–TCT v19.1 AutoLearn",
        "initialized": env is not None,
        "train_cycles_completed": state.get("train_cycles_completed", 0),
        "last_bias": state.get("last_bias", "neutral"),
        "last_confidence": state.get("last_confidence", 0.0),
    }

@app.head("/")
async def head_root():
    """Render health check compatibility."""
    return {"status": "ok"}

@app.get("/status")
async def status():
    """Operational server status summary."""
    return {
        "server": "HPB–TCT v19.1",
        "initialized": env is not None,
        "last_state_update": state.get("last_timestamp"),
        "train_cycles": state.get("train_cycles_completed", 0),
        "bias": state.get("last_bias", "neutral"),
        "confidence": state.get("last_confidence", 0.0),
        "RIG_status": state.get("last_RIG_status", "UNKNOWN"),
    }

@app.get("/dashboard")
async def dashboard():
    """Returns summarized AutoLearn + RIG overview (Telegram safe)."""
    bias = state.get("last_bias", "neutral")
    confidence = state.get("last_confidence", 0.0)
    rig_status = state.get("last_RIG_status", "UNKNOWN")
    train_cycles = state.get("train_cycles_completed", 0)
    return {
        "version": "HPB–TCT v19.1",
        "bias": bias,
        "confidence": confidence,
        "RIG_status": rig_status,
        "train_cycles_completed": train_cycles,
        "timestamp": state.get("last_timestamp"),
    }

@app.get("/help")
async def help_info():
    """Display available API routes (for Telegram)."""
    return {
        "routes": [
            "/status",
            "/dashboard",
            "/train?episodes=5",
            "/validate_gates",
            "/fetch_context",
            "/reset_state",
        ],
        "note": "HPB–TCT v19.1 AutoLearn Server (Render)",
    }

# ────────────────────────────────────────────────────────────────
# FETCH CONTEXT (Safe fallback)
# ────────────────────────────────────────────────────────────────
@app.get("/fetch_context")
async def fetch_context():
    """Fetch latest environment or synthetic context snapshot."""
    try:
        if not env:
            return {"error": "Environment not initialized."}

        context_data = {}
        if hasattr(env, "price_feed"):
            context_data = env.price_feed
        elif hasattr(env, "exchange"):
            context_data = env.exchange.__dict__
        elif hasattr(env, "observation_space"):
            context_data = str(env.observation_space)
        elif hasattr(env, "current_step"):
            context_data = {"step": env.current_step}

        # Always return a synthetic fallback
        if not context_data:
            context_data = {
                "symbol": getattr(env, "symbol", "BTC-USDT"),
                "interval": getattr(env, "interval", "1H"),
                "window_size": getattr(env, "window_size", 100),
                "timestamp": datetime.utcnow().isoformat(),
            }

        return {"context": context_data}
    except Exception as e:
        logger.error(f"Context fetch error: {e}")
        return {"error": str(e)}

# ────────────────────────────────────────────────────────────────
# VALIDATE GATES
# ────────────────────────────────────────────────────────────────
@app.get("/validate_gates")
async def validate_gates():
    """Run Range Integrity Gate validation and persist result."""
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

        # Save to persistent state
        state.update({
            "last_timestamp": datetime.utcnow().isoformat(),
            "last_RIG_status": result.get("status", "OK"),
            "last_bias": result.get("htf_bias", "neutral"),
            "last_confidence": result.get("confidence", 0.0),
        })
        save_state(state)

        return {"RIG_Validation": result}
    except Exception as e:
        logger.error(f"Gate validation error: {e}")
        return {"error": str(e)}

# ────────────────────────────────────────────────────────────────
# TRAIN (AutoLearn)
# ────────────────────────────────────────────────────────────────
@app.get("/train")
async def train_agent(episodes: int = 5):
    """Run AutoLearn training cycles and update persistent state."""
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
            logger.warning("⚠️ No recognized training function found in HPB_TensorTrade_Env.")
            return {"warning": "No training function found on environment."}

        state["train_cycles_completed"] = state.get("train_cycles_completed", 0) + episodes
        state["last_timestamp"] = datetime.utcnow().isoformat()
        save_state(state)

        return {"status": "completed", "episodes": episodes, "state": state}
    except Exception as e:
        logger.error(f"Training error: {e}")
        return {"error": str(e)}

# ────────────────────────────────────────────────────────────────
# STATE MANAGEMENT
# ────────────────────────────────────────────────────────────────
@app.get("/state")
async def get_state():
    """Return current AutoLearn state JSON."""
    return state

@app.post("/reset_state")
async def reset_state():
    """Reset persistent AutoLearn state."""
    global state
    state = {
        "last_timestamp": None,
        "train_cycles_completed": 0,
        "last_RIG_status": "UNKNOWN",
        "last_bias": "neutral",
        "last_confidence": 0.0,
    }
    save_state(state)
    return {"status": "reset", "state": state}

# ────────────────────────────────────────────────────────────────
# ENTRY POINT
# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"🚀 Starting HPB–TCT AutoLearn v19.1 server on port {port} ...")
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)
