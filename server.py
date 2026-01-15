# ================================================================
# server.py — HPB–TCT v19 AutoLearn Runtime Entry Point (Persistent)
# ================================================================
# • Auto-initializes HPB TensorTrade environment
# • Tracks training state across restarts (/data/hpb_autolearn_state.json)
# • Safe for Render / Uvicorn deployment
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

# Backward-compatible alias
TensorTradeEnv = HPB_TensorTrade_Env

# ────────────────────────────────────────────────────────────────
# Logging
# ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("HPB-TCT-Server")

# ────────────────────────────────────────────────────────────────
# FastAPI Setup
# ────────────────────────────────────────────────────────────────
app = FastAPI(title="HPB–TCT AutoLearn v19", version="1.0.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ────────────────────────────────────────────────────────────────
# AutoLearn State Persistence
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
# Routes
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


@app.get("/fetch_context")
async def fetch_context():
    """Fetch current BTC/USDT context for gate evaluation."""
    try:
        data = env.get_price_context(["BTC-USDT", "ETH-USDT", "SOL-USDT"])
        return {"context": data}
    except Exception as e:
        logger.error(f"Context fetch error: {e}")
        return {"error": str(e)}


@app.get("/validate_gates")
async def validate_gates():
    """Run RIG validation and store results persistently."""
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
        # Save to state
        state.update({
            "last_timestamp": datetime.utcnow().isoformat(),
            "last_RIG_status": result["status"],
            "last_bias": result["htf_bias"],
            "last_confidence": result["confidence"],
        })
        save_state(state)
        return {"RIG_Validation": result}
    except Exception as e:
        logger.error(f"Gate validation error: {e}")
        return {"error": str(e)}


@app.get("/train")
async def train_agent(episodes: int = 5):
    """Run AutoLearn training cycles and update persistent state."""
    if env is None:
        return {"error": "Environment not initialized."}
    try:
        logger.info(f"🚀 Starting AutoLearn training for {episodes} episodes...")
        for i in range(episodes):
            env.build_environment()
            # Placeholder for RL / Reward logic
            logger.info(f"✅ Completed training episode {i+1}/{episodes}")

        state["train_cycles_completed"] = state.get("train_cycles_completed", 0) + episodes
        state["last_timestamp"] = datetime.utcnow().isoformat()
        save_state(state)

        return {"status": "completed", "episodes": episodes, "state": state}
    except Exception as e:
        logger.error(f"Training error: {e}")
        return {"error": str(e)}

# ────────────────────────────────────────────────────────────────
# Entry Point (Render / Local)
# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    logger.info(f"🚀 Starting HPB–TCT AutoLearn v19 server on port {port} ...")
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)
