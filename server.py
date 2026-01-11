"""
server.py
────────────────────────────────────────────
HPM–TCT v19 RIG EXTENDED Server
Phase 7: Adds Risk Management + Signals Module
────────────────────────────────────────────
"""

import gym_fix
import os, json, traceback, asyncio, threading, subprocess, time
from datetime import datetime
import pandas as pd, numpy as np, plotly.graph_objs as go, httpx
from fastapi import FastAPI
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel
from apscheduler.schedulers.background import BackgroundScheduler

from tensortrade_config_ext import AUTO_INIT, TENSORTRADE_CONFIG, snapshot_environment
from High_Probability_Model_v17_RIG.validate_gates import validate_gates
from risk_model import compute_risk_profile

# ───────────────────────────────────────────────
# Server Setup
# ───────────────────────────────────────────────
app = FastAPI(title="HPM–TCT v19 RIG EXTENDED + Risk", version="1.1")
ENV = AUTO_INIT()

# In-memory context cache
live_context_cache = {
    "last_update": None,
    "symbol": "BTC-USDT-SWAP",
    "interval": "1H",
    "gates": {},
    "ExecutionConfidence_Total": 0.0,
    "Reward_Summary": "INIT",
    "_conf_history": []
}

# ───────────────────────────────────────────────
# Core Endpoints
# ───────────────────────────────────────────────
@app.get("/")
def home():
    return {"status": "OK", "message": "HPB–TCT Server + Risk Management Layer Online"}

@app.get("/status")
def status():
    try:
        snap = json.loads(snapshot_environment(ENV))
        return {"status": "ready", "env": snap}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# ───────────────────────────────────────────────
# Risk + Signals Endpoint
# ───────────────────────────────────────────────
@app.get("/signals")
async def get_signals():
    """Generate a live risk-weighted signal."""
    try:
        profile = await compute_risk_profile(live_context_cache)

        # Append to local signals log
        os.makedirs("logs", exist_ok=True)
        path = "logs/signals_log.json"
        try:
            signals = json.load(open(path)) if os.path.exists(path) else []
        except Exception:
            signals = []

        signals.append(profile)
        if len(signals) > 100:
            signals = signals[-100:]
        json.dump(signals, open(path, "w"), indent=2)

        return JSONResponse(profile)

    except Exception as e:
        return JSONResponse({"error": str(e), "trace": traceback.format_exc()}, status_code=500)

# ───────────────────────────────────────────────
# Dashboard + Algo Context + Risk Overlay
# ───────────────────────────────────────────────
async def fetch_algo_context(symbol="BTC-USDT-SWAP", interval="1H"):
    try:
        context = {"symbol": symbol, "interval": interval}
        gates = validate_gates(context)
        print(f"[ALGO] Gates fetched successfully for {symbol} ({interval})")
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

def plot_with_risk_overlay(df, title, risk_profile):
    if df.empty:
        return go.Figure().update_layout(title=f"{title} – No Data", template="plotly_dark")

    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df["ts"], open=df["o"], high=df["h"], low=df["l"], close=df["c"], name="Price"))

    if "stop_loss" in risk_profile:
        fig.add_hline(y=risk_profile["stop_loss"], line=dict(color="red", dash="dot"),
                      annotation_text="Stop Loss", annotation_position="bottom right")
    if "take_profit" in risk_profile:
        fig.add_hline(y=risk_profile["take_profit"], line=dict(color="green", dash="dot"),
                      annotation_text="Take Profit", annotation_position="top right")

    fig.update_layout(title=f"{title} | {risk_profile.get('signal', 'No Signal')}",
                      template="plotly_dark", height=600, xaxis_rangeslider_visible=False)
    return fig

latest_dashboard_html = "<h3>Initializing dashboard...</h3>"

async def generate_dashboard():
    global latest_dashboard_html
    try:
        algo_context = await fetch_algo_context("BTC-USDT-SWAP", "1H")
        live_context_cache.update(algo_context)
        risk_profile = compute_risk_profile(live_context_cache)

        df = await fetch_ohlc(interval="1H")
        fig = plot_with_risk_overlay(df, "MTF Range (1H)", risk_profile)

        html = (
            f"<h2>HPM–TCT v19 RIG EXTENDED + Risk Dashboard</h2>"
            f"<p>Updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC</p>"
            f"<p><b>Signal:</b> {risk_profile['signal']} | "
            f"<b>Risk:</b> {risk_profile['risk_score']} | "
            f"<b>Confidence:</b> {risk_profile['confidence_smoothed']}</p>"
            + fig.to_html(full_html=False, include_plotlyjs="cdn")
        )
        latest_dashboard_html = html
        print(f"[HPB] Dashboard + Risk updated @ {datetime.utcnow().isoformat()}")
    except Exception as e:
        latest_dashboard_html = f"<h3>Error generating dashboard:</h3><pre>{e}</pre>"
        print(f"[HPB ERROR] {e}")

@app.get("/dashboard", response_class=HTMLResponse)
async def get_dashboard():
    return latest_dashboard_html

# ───────────────────────────────────────────────
# Live Context Updater
# ───────────────────────────────────────────────
async def update_live_context():
    global live_context_cache
    while True:
        try:
            gates = validate_gates({"symbol": "BTC-USDT-SWAP", "interval": "1H"})
            live_context_cache.update({
                "last_update": datetime.utcnow().isoformat(),
                "gates": gates,
                "ExecutionConfidence_Total": gates.get("ExecutionConfidence_Total", 0.0),
                "Reward_Summary": gates.get("Reward_Summary", "N/A"),
            })
            print(f"[LIVE UPDATE] Refreshed gate context @ {live_context_cache['last_update']}")
        except Exception as e:
            print(f"[LIVE UPDATE ERROR] {e}")
        await asyncio.sleep(60)

# ───────────────────────────────────────────────
# Startup & Scheduler
# ───────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    print("[HPB] Starting Risk Layer...")
    asyncio.create_task(update_live_context())
    await generate_dashboard()

scheduler = BackgroundScheduler()
scheduler.add_job(lambda: asyncio.run(generate_dashboard()), "interval", hours=1)
scheduler.start()

# ───────────────────────────────────────────────
# Keepalive (safe)
# ───────────────────────────────────────────────
def touch_keepalive():
    while True:
        try:
            subprocess.run(["touch", "/tmp/render_keepalive.flag"], check=True)
            print(f"[KEEPALIVE] Refreshed /tmp/render_keepalive.flag at {datetime.utcnow().isoformat()}")
        except Exception as e:
            print(f"[KEEPALIVE ERROR] {e}")
        time.sleep(600)
threading.Thread(target=touch_keepalive, daemon=True).start()

# ───────────────────────────────────────────────
# Run Server
# ───────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
