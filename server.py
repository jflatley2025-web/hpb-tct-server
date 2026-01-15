# ================================================================
# server.py — HPB–TCT v19 AutoLearn Runtime (Fixed + Extended)
# ================================================================
# • Dynamic environment support (TensorTrade / HPB)
# • AutoLearn state persistence
# • Compatible with Render (port binding + health checks)
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

TensorTradeEnv = HPB_TensorTrade_Env  # backward compatibility alias

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
app = FastAPI(title="HPB–TCT AutoLearn v19", version="1.0.3")

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
            "last_bias": None,
            "last_confidence": None,
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
        "environment": "HPB–TCT v19 AutoLearn",
        "initialized": env is not None,
        "train_cycles_completed": state.get("train_cycles_completed", 0),
        "last_bias": state.get("last_bias"),
        "last_confidence": state.get("last_confidence"),
    }

@app.head("/")
async def head_root():
    """Handle Render HEAD request health checks."""
    return {"status": "ok"}

@app.get("/status")
async def status():
    """Operational server status summary."""
    return {
        "server": "HPB–TCT v19",
        "initialized": env is not None,
        "last_state_update": state.get("last_timestamp"),
        "train_cycles": state.get("train_cycles_completed", 0),
        "bias": state.get("last_bias"),
        "confidence": state.get("last_confidence"),
        "RIG_status": state.get("last_RIG_status"),
    }

@app.get("/dashboard")
async def dashboard():
    """Returns summarized AutoLearn + RIG overview."""
    return {
        "version": "HPB–TCT v19",
        "bias": state.get("last_bias"),
        "confidence": state.get("last_confidence"),
        "RIG_status": state.get("last_RIG_status"),
        "train_cycles_completed": state.get("train_cycles_completed", 0),
    }

# ────────────────────────────────────────────────────────────────
# FETCH CONTEXT (Fixed)
# ────────────────────────────────────────────────────────────────
@app.get("/fetch_context")
async def fetch_context():
    """Fetch latest available context or observation from HPB TensorTrade environment."""
    try:
        if not env:
            return {"error": "Environment not initialized."}

        # Fallback logic — use whatever structure exists
        context_data = {}
        if hasattr(env, "price_feed"):
            context_data = env.price_feed
        elif hasattr(env, "exchange"):
            context_data = env.exchange.__dict__
        elif hasattr(env, "observation_space"):
            context_data = str(env.observation_space)
        elif hasattr(env, "current_step"):
            context_data = {"step": env.current_step}
        else:
            context_data = {"warning": "No context or feed attribute found in environment."}

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
# TRAIN (Fixed)
# ────────────────────────────────────────────────────────────────
@app.get("/train")
async def train_agent(episodes: int = 5):
    """Run AutoLearn training cycles and update persistent state."""
    if env is None:
        return {"error": "Environment not initialized."}
    try:
        logger.info(f"🚀 Starting AutoLearn training for {episodes} episodes...")

        # Try known training methods
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
        "last_RIG_status": None,
        "last_bias": None,
        "last_confidence": None,
    }
    save_state(state)
    return {"status": "reset", "state": state}

# ────────────────────────────────────────────────────────────────
# ENTRY POINT
# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"🚀 Starting HPB–TCT AutoLearn v19 server on port {port} ...")
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)
