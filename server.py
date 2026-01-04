"""
server.py
────────────────────────────────────────────
HPB–TCT v17.3 Server
Integrates TensorTrade v1.0.3 Environment + HPB Gate Logic
────────────────────────────────────────────
"""

import gym_fix  # must be first

import os
import json
import traceback
from datetime import datetime
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from tensortrade_config_ext import AUTO_INIT, TENSORTRADE_CONFIG, snapshot_environment
from High_Probability_Model_v17.validate_gates import validate_gates





# ───────────────────────────────────────────────
# Server setup
# ───────────────────────────────────────────────
app = FastAPI(title="HPB–TCT v17.3 Server", version="1.0")

# Auto-initialize environment
ENV = AUTO_INIT()


@app.get("/")
def home():
    return {
        "status": "OK",
        "message": "HPB–TCT v17.3 Server Running",
        "timestamp": datetime.utcnow().isoformat(),
        "symbol": TENSORTRADE_CONFIG["symbol"],
        "interval": TENSORTRADE_CONFIG["interval"],
    }

from pydantic import BaseModel

# Define the request body schema
class ValidateRequest(BaseModel):
    symbol: str = "BTC-USDT-SWAP"
    interval: str = "1H"

# ───────────────────────────────────────────────
# Validate endpoint (Gate 1A–1D chain)
# ───────────────────────────────────────────────
@app.post("/validate")
async def validate(request: ValidateRequest):
    try:
        symbol = request.symbol
        interval = request.interval

        context = {"symbol": symbol, "interval": interval}
        gates = validate_gates(context)

        response = {
            "timestamp": datetime.utcnow().isoformat(),
            "mode": "live",
            "symbol": symbol,
            "interval": interval,
            "gates": gates.get("gates", {}),
            "Session_Info": gates.get("Session_Info", {}),
            "ExecutionConfidence_Total": gates.get("ExecutionConfidence_Total", 0.0),
            "Reward_Summary": gates.get("Reward_Summary", "N/A"),
        }
        return JSONResponse(response)

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(
            {"error": str(e), "trace": traceback.format_exc()}, status_code=500
        )


# Define request model for /backtest
class BacktestRequest(BaseModel):
    episodes: int = 10  # default value for testing


# ───────────────────────────────────────────────
# Backtest endpoint (RL reward simulation)
# ───────────────────────────────────────────────
@app.post("/backtest")
async def backtest(request: BacktestRequest):
    try:
        episodes = request.episodes
        print(f"[BACKTEST] Running {episodes} episodes...")

        ENV.simulate_training(episodes=episodes)

        return JSONResponse(
            {
                "timestamp": datetime.utcnow().isoformat(),
                "mode": "backtest",
                "episodes": episodes,
                "message": "Backtest completed successfully.",
            }
        )
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(
            {"error": str(e), "trace": traceback.format_exc()}, status_code=500
        )


# ───────────────────────────────────────────────
# Status endpoint
# ───────────────────────────────────────────────
@app.get("/status")
def status():
    try:
        snapshot = json.loads(snapshot_environment(ENV))
        return {"status": "ready", "env": snapshot}
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ───────────────────────────────────────────────
# Run Uvicorn (for Replit)
# ───────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)
