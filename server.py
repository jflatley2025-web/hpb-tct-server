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
# HPB–TCT AUTO VISUAL DASHBOARD (HTF / MTF / LTF)
# ───────────────────────────────────────────────
from fastapi.responses import HTMLResponse
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import httpx
from apscheduler.schedulers.background import BackgroundScheduler
import asyncio

# store the latest rendered dashboard HTML
latest_dashboard_html = "<h3>Initializing dashboard...</h3>"

async def fetch_ohlc(symbol="BTC-USDT", interval="1H", limit=200):
    """Fetch OHLC data from OKX (handles variable column count)."""
    url = f"https://www.okx.com/api/v5/market/candles?instId={symbol}-SWAP&bar={interval}&limit={limit}"
    async with httpx.AsyncClient() as client:
        resp = await client.get(url)
    data = resp.json().get("data", [])
    if not data:
        return pd.DataFrame()

    # OKX returns 9 columns per candle (timestamp, o, h, l, c, vol, volCcy, volQuote, confirm)
    # Only the first 6 are needed for charting
    df = pd.DataFrame(data).iloc[:, :6]
    df.columns = ["ts", "o", "h", "l", "c", "v"]

    df = df.astype({"o": float, "h": float, "l": float, "c": float, "v": float})
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    return df.sort_values("ts")


def plot_range_chart(df: pd.DataFrame, title: str):
    if df.empty:
        return go.Figure().update_layout(title=f"{title} – No data", template="plotly_dark")
    high, low = df["h"].max(), df["l"].min()
    eq = (high + low) / 2
    x = np.arange(len(df))
    curve = np.poly1d(np.polyfit(x, df["l"].rolling(5).mean(), 2))
    curve_y = curve(x)

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df["ts"], open=df["o"], high=df["h"], low=df["l"], close=df["c"], name="Price"
    ))
    fig.add_hline(y=high, line=dict(color="red", dash="dash"), annotation_text="Range High")
    fig.add_hline(y=low, line=dict(color="green", dash="dash"), annotation_text="Range Low")
    fig.add_hline(y=eq, line=dict(color="orange", dash="dot"), annotation_text="EQ")
    fig.add_trace(go.Scatter(x=df["ts"], y=curve_y, mode="lines",
                             line=dict(color="blue", width=2), name="Liquidity Curve"))
    fig.update_layout(
        title=title,
        template="plotly_dark",
        height=600,
        xaxis_rangeslider_visible=False,
        margin=dict(l=30, r=30, t=60, b=30)
    )
    return fig

async def generate_dashboard():
    """Regenerate the multi-timeframe dashboard HTML."""
    global latest_dashboard_html
    try:
        htf_df = await fetch_ohlc(interval="4H")
        mtf_df = await fetch_ohlc(interval="1H")
        ltf_df = await fetch_ohlc(interval="15m")

        htf_fig = plot_range_chart(htf_df, "HTF Range (4H)")
        mtf_fig = plot_range_chart(mtf_df, "MTF Range (1H)")
        ltf_fig = plot_range_chart(ltf_df, "LTF Range (15m)")

        html = (
            f"<h2>HPB–TCT Range Dashboard</h2>"
            f"<p>Updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC</p>"
            + htf_fig.to_html(full_html=False, include_plotlyjs="cdn")
            + mtf_fig.to_html(full_html=False, include_plotlyjs=False)
            + ltf_fig.to_html(full_html=False, include_plotlyjs=False)
        )
        latest_dashboard_html = html
        print(f"[HPB] Dashboard updated @ {datetime.utcnow().isoformat()}")
    except Exception as e:
        latest_dashboard_html = f"<h3>Error generating dashboard:</h3><pre>{e}</pre>"
        print(f"[HPB-ERROR] Dashboard generation failed: {e}")

@app.get("/dashboard", response_class=HTMLResponse)
async def get_dashboard():
    """Return the latest dashboard HTML."""
    return latest_dashboard_html

# ───────────────────────────────────────────────
# Scheduler – auto-refresh dashboard every 24h
# ───────────────────────────────────────────────
scheduler = BackgroundScheduler()
scheduler.add_job(lambda: asyncio.run(generate_dashboard()), "interval", hours=24)
scheduler.start()

@app.on_event("startup")
async def startup_event():
    print("[HPB] Initializing dashboard...")
    await generate_dashboard()

# ───────────────────────────────────────────────
# Run Uvicorn (for Replit)
# ───────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)
