"""
server.py
────────────────────────────────────────────
HPM–TCT v19 RIG EXTENDED Server (Phase 8)
Integrates TensorTrade v1.0.3 Environment,
HPB Gate Logic + Risk Management + P03 Confluence Engine
────────────────────────────────────────────
"""

import gym_fix  # must be first
import os, json, traceback, asyncio, subprocess, time, threading
from datetime import datetime
from fastapi import FastAPI
from fastapi.responses import JSONResponse, HTMLResponse

# local imports
from tensortrade_config_ext import AUTO_INIT, TENSORTRADE_CONFIG, snapshot_environment
from High_Probability_Model_v17_RIG.validate_gates import validate_gates
from risk_model import compute_risk_profile
from p03_model import compute_p03_confluence

import pandas as pd
import numpy as np
import plotly.graph_objs as go
import httpx
from apscheduler.schedulers.background import BackgroundScheduler


# ───────────────────────────────────────────────
# Server setup
# ───────────────────────────────────────────────
app = FastAPI(title="HPM–TCT v19 RIG EXTENDED + P03", version="8.0")
ENV = AUTO_INIT()


@app.get("/")
def home():
    return {
        "status": "OK",
        "message": "HPM–TCT v19 RIG EXTENDED + P03 Server Running",
        "timestamp": datetime.utcnow().isoformat(),
        "symbol": TENSORTRADE_CONFIG["symbol"],
        "interval": TENSORTRADE_CONFIG["interval"],
    }


# ───────────────────────────────────────────────
# Validation / Backtest
# ───────────────────────────────────────────────
from pydantic import BaseModel

class ValidateRequest(BaseModel):
    symbol: str = "BTC-USDT-SWAP"
    interval: str = "1H"


@app.post("/validate")
async def validate(request: ValidateRequest):
    try:
        context = {"symbol": request.symbol, "interval": request.interval}
        gates = validate_gates(context)
        return JSONResponse({
            "timestamp": datetime.utcnow().isoformat(),
            "mode": "live",
            "symbol": request.symbol,
            "interval": request.interval,
            "gates": gates.get("gates", {}),
            "Session_Info": gates.get("Session_Info", {}),
            "ExecutionConfidence_Total": gates.get("ExecutionConfidence_Total", 0.0),
            "Reward_Summary": gates.get("Reward_Summary", "N/A"),
        })
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)


class BacktestRequest(BaseModel):
    episodes: int = 10


@app.post("/backtest")
async def backtest(request: BacktestRequest):
    try:
        ENV.simulate_training(episodes=request.episodes)
        return JSONResponse({
            "timestamp": datetime.utcnow().isoformat(),
            "mode": "backtest",
            "episodes": request.episodes,
            "message": "Backtest completed successfully.",
        })
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/status")
def status():
    try:
        snapshot = json.loads(snapshot_environment(ENV))
        return {"status": "ready", "env": snapshot}
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ───────────────────────────────────────────────
# Live Algo Context + Data
# ───────────────────────────────────────────────
async def fetch_algo_context(symbol="BTC-USDT-SWAP", interval="1H"):
    try:
        context = {"symbol": symbol, "interval": interval}
        gates = validate_gates(context)
        print(f"[ALGO] Gates fetched for {symbol} ({interval})")
        return gates
    except Exception as e:
        print(f"[ALGO ERROR] {e}")
        return {"error": str(e)}


async def fetch_ohlc(symbol="BTC-USDT", interval="1H", limit=200):
    url = f"https://www.okx.com/api/v5/market/candles?instId={symbol}-SWAP&bar={interval}&limit={limit}"
    async with httpx.AsyncClient() as client:
        resp = await client.get(url)
    data = resp.json().get("data", [])
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data).iloc[:, :6]
    df.columns = ["ts", "o", "h", "l", "c", "v"]
    df = df.astype({"o": float, "h": float, "l": float, "c": float, "v": float})
    df["ts"] = pd.to_datetime(pd.to_numeric(df["ts"]), unit="ms")
    return df.sort_values("ts")


# ───────────────────────────────────────────────
# Chart Renderer (simplified for brevity)
# ───────────────────────────────────────────────
def plot_range_chart(df: pd.DataFrame, title: str):
    if df.empty:
        return go.Figure().update_layout(title=f"{title} – No data", template="plotly_dark")
    fig = go.Figure(go.Candlestick(
        x=df["ts"], open=df["o"], high=df["h"], low=df["l"], close=df["c"], name="Price"
    ))
    fig.update_layout(title=title, template="plotly_dark", height=600, xaxis_rangeslider_visible=False)
    return fig


# ───────────────────────────────────────────────
# Dashboard Generator
# ───────────────────────────────────────────────
latest_dashboard_html = "<h3>Initializing dashboard...</h3>"

async def generate_dashboard():
    global latest_dashboard_html
    try:
        # fetch data + risk profile
        algo_context = await fetch_algo_context("BTC-USDT-SWAP", "1H")
        risk_profile = await compute_risk_profile(algo_context)
        p03_data = await compute_p03_confluence(risk_profile)

        htf_df = await fetch_ohlc(interval="4H")
        mtf_df = await fetch_ohlc(interval="1H")
        ltf_df = await fetch_ohlc(interval="15m")

        htf_fig = plot_range_chart(htf_df, "HTF Range (4H)")
        mtf_fig = plot_range_chart(mtf_df, "MTF Range (1H)")
        ltf_fig = plot_range_chart(ltf_df, "LTF Range (15m)")

        html = (
            f"<h2>HPM–TCT v19 RIG EXTENDED Dashboard</h2>"
            f"<p>Updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC</p>"
            f"<p><b>Signal:</b> {risk_profile.get('signal')} | "
            f"<b>Risk:</b> {risk_profile.get('risk_score')} | "
            f"<b>Vol:</b> {risk_profile.get('volatility_score')} | "
            f"<b>Price:</b> {risk_profile.get('current_price')}</p>"
            f"<h3>🧠 P03 Confluence Model</h3>"
            f"<pre>{json.dumps(p03_data.get('confidence_matrix', {}), indent=2)}</pre>"
            f"<p><b>Phase:</b> {p03_data.get('phase')} | "
            f"<b>Bias:</b> {p03_data.get('execution_bias')} | "
            f"<b>Expected RR:</b> {p03_data.get('expected_reward_ratio')} | "
            f"Vol Band: {p03_data.get('volatility_band')}</p>"
            + htf_fig.to_html(full_html=False, include_plotlyjs="cdn")
            + mtf_fig.to_html(full_html=False, include_plotlyjs=False)
            + ltf_fig.to_html(full_html=False, include_plotlyjs=False)
        )

        latest_dashboard_html = html
        print(f"[HPB] Dashboard updated @ {datetime.utcnow().isoformat()}")

    except Exception as e:
        latest_dashboard_html = f"<h3>Error generating dashboard:</h3><pre>{e}</pre>"
        print(f"[HPB ERROR] {e}")


@app.get("/dashboard", response_class=HTMLResponse)
async def get_dashboard():
    return latest_dashboard_html


# ───────────────────────────────────────────────
# New Endpoint – P03 Confluence
# ───────────────────────────────────────────────
@app.get("/p03")
async def get_p03_confluence_endpoint():
    try:
        algo_context = await fetch_algo_context("BTC-USDT-SWAP", "1H")
        risk_profile = await compute_risk_profile(algo_context)
        p03_data = await compute_p03_confluence(risk_profile)
        return JSONResponse(p03_data)
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)


# ───────────────────────────────────────────────
# Background Tasks
# ───────────────────────────────────────────────
def touch_keepalive():
    """Refresh temp file every 10 min to prevent Render idle."""
    while True:
        try:
            keepalive_file = "/tmp/render_keepalive.flag"
            subprocess.run(["touch", keepalive_file], check=True)
            print(f"[KEEPALIVE] Touched {keepalive_file} @ {datetime.utcnow().isoformat()}")
        except Exception as e:
            print(f"[KEEPALIVE ERROR] {e}")
        time.sleep(600)

threading.Thread(target=touch_keepalive, daemon=True).start()


@app.on_event("startup")
async def startup_event():
    print("[HPB] Initializing dashboard...")
    await generate_dashboard()
    scheduler = BackgroundScheduler()
    scheduler.add_job(lambda: asyncio.run(generate_dashboard()), "interval", hours=24)
    scheduler.start()


# ───────────────────────────────────────────────
# Run Server
# ───────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)
