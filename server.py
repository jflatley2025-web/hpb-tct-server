# ================================================================
# server.py — HPB–TCT v19 AutoLearn Runtime Entry Point
# ================================================================
# Purpose:
#  • Start HPB-TensorTrade live environment (v19 compatible)
#  • Auto-initialize HPB_TensorTrade_Env with AutoLearn extensions
#  • Maintain backward compatibility for older references (TensorTradeEnv)
# ================================================================

import os
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# ────────────────────────────────────────────────────────────────
# Imports (AutoLearn + Environment)
# ────────────────────────────────────────────────────────────────
from tensortrade_env import HPB_TensorTrade_Env as TensorTradeEnv
from tensortrade_config_ext import AUTO_INIT, TENSORTRADE_CONFIG
from hpb_rig_validator import range_integrity_validator

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
app = FastAPI(title="HPB–TCT AutoLearn v19", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ────────────────────────────────────────────────────────────────
# AutoLearn Environment Initialization
# ────────────────────────────────────────────────────────────────
logger.info("🔧 Initializing HPB–TCT TensorTrade Environment...")
try:
    env = AUTO_INIT()  # builds and returns a live environment
    logger.info(f"✅ Environment initialized successfully with config: {TENSORTRADE_CONFIG}")
except Exception as e:
    logger.error(f"❌ Failed to initialize HPB environment: {e}")
    env = None

# Optional backward alias for older code expecting TensorTradeEnv
TensorTradeEnv = HPB_TensorTrade_Env

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
    """Run RIG validation logic for live HPB context."""
    try:
        # Dummy example of current context (replace with live gate state)
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
        return {"RIG_Validation": result}
    except Exception as e:
        logger.error(f"Gate validation error: {e}")
        return {"error": str(e)}


@app.get("/train")
async def train_agent(episodes: int = 5):
    """Run AutoLearn training cycles for v19."""
    if env is None:
        return {"error": "Environment not initialized."}
    try:
        logger.info(f"🚀 Starting AutoLearn training for {episodes} episodes...")
        for i in range(episodes):
            env.build_environment()
            # Here you would trigger your RL agent training or reward updates
            logger.info(f"✅ Completed training episode {i+1}/{episodes}")
        return {"status": "completed", "episodes": episodes}
    except Exception as e:
        logger.error(f"Training error: {e}")
        return {"error": str(e)}

# ────────────────────────────────────────────────────────────────
# Local Run Entry Point (for Render/Uvicorn)
# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    logger.info(f"🚀 Starting HPB–TCT AutoLearn v19 server on port {port} ...")
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)
