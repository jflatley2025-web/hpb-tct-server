"""
server.py
────────────────────────────────────────────
HPM–TCT v19 RIG EXTENDED Server
Phase 7.3 – Multi-Timeframe Risk Model Integrated
────────────────────────────────────────────
"""

import gym_fix  # must be first

import os, json, traceback, asyncio, time, threading, subprocess
from datetime import datetime
import pandas as pd, numpy as np, plotly.graph_objs as go, httpx
from fastapi import FastAPI
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel
from apscheduler.schedulers.background import BackgroundScheduler

# ───────────────────────────────────────────────
# Local imports
# ───────────────────────────────────────────────
from tensortrade_config_ext import AUTO_INIT, TENSORTRADE_CONFIG, snapshot_environment
from High_Probability_Model_v17_RIG.validate_gates import validate_gates
from risk_model import compute_risk_profile  # Phase 7.3 model

# ───────────────────────────────────────────────
# Server setup
# ───────────────────────────────────────────────
app = FastAPI(title="HPM–TCT v19 RIG EXTENDED Server", version="1.0")
ENV = AUTO_INIT()

@app.get("/")
def home():
    return {
        "status": "OK",
        "message": "HPM–TCT v19 RIG EXTENDED Server Running",
        "timestamp": datetime.utcnow().isoformat(),
        "symbol": TENSORTRADE_CONFIG["symbol"],
        "interval": TENSORTRADE_CONFIG["interval"],
    }

# ───────────────────────────────────────────────
# Validation / Backtest Endpoints
# ───────────────────────────────────────────────
class ValidateRequest(BaseModel):
    symbol: str = "BTC-USDT-SWAP"
    interval: str = "1H"

@app.post("/validate")
async def validate(request: ValidateRequest):
    try:
        ctx = {"symbol": request.symbol, "interval": request.interval}
        gates = validate_gates(ctx)
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
# Algo Context and OHLC Data
# ───────────────────────────────────────────────
async def fetch_algo_context(symbol="BTC-USDT-SWAP", interval="1H"):
    try:
        ctx = {"symbol": symbol, "interval": interval}
        gates = validate_gates(ctx)
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
# Chart Renderer (Algo-Aware)
# ───────────────────────────────────────────────
def plot_range_chart(df: pd.DataFrame, title: str):
    if df.empty:
        return go.Figure().update_layout(title=f"{title} – No data", template="plotly_dark")

    import scipy.signal as sig
    highs, lows = df["h"].to_numpy(), df["l"].to_numpy()
    x = np.arange(len(df))
    swing_high, _ = sig.find_peaks(highs, distance=5)
    swing_low, _ = sig.find_peaks(-lows, distance=5)

    slope = np.polyfit(x[-min(len(df), 40):], df["c"].values[-min(len(df), 40):], 1)[0]
    phase = "accumulation" if slope >= 0 else "distribution"

    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df["ts"], open=df["o"], high=df["h"], low=df["l"], close=df["c"]))
    fig.update_layout(title=f"{title} | {phase.title()} Structure", template="plotly_dark",
                      height=600, xaxis_rangeslider_visible=False)
    return fig

# ───────────────────────────────────────────────
# Dashboard Generator
# ───────────────────────────────────────────────
latest_dashboard_html = "<h3>Initializing dashboard...</h3>"

async def generate_dashboard():
    global latest_dashboard_html
    try:
        algo_context = await fetch_algo_context("BTC-USDT-SWAP", "1H")
        risk_profile = await compute_risk_profile(algo_context)   # ✅ Fixed await

        htf_df = await fetch_ohlc(interval="4H")
        mtf_df = await fetch_ohlc(interval="1H")
        ltf_df = await fetch_ohlc(interval="15m")

        htf_fig = plot_range_chart(htf_df, "HTF Range (4H)")
        mtf_fig = plot_range_chart(mtf_df, "MTF Range (1H)")
        ltf_fig = plot_range_chart(ltf_df, "LTF Range (15m)")

        html = (
            f"<h2>HPM–TCT v19 RIG EXTENDED Dashboard</h2>"
            f"<p>Updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC</p>"
            f"<p><b>Signal:</b> {risk_profile['signal']} | "
            f"<b>Risk:</b> {risk_profile['risk_score']} | "
            f"<b>Vol:</b> {risk_profile['volatility_score']} | "
            f"<b>Price:</b> {risk_profile['current_price']}</p>"
            f"<pre>{json.dumps(risk_profile['timeframes'], indent=2)}</pre>"
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
# Scheduler (24 h refresh)
# ───────────────────────────────────────────────
scheduler = BackgroundScheduler()
scheduler.add_job(lambda: asyncio.run(generate_dashboard()), "interval", hours=24)
scheduler.start()

@app.on_event("startup")
async def startup_event():
    print("[HPB] Starting Risk Layer...")
    await generate_dashboard()
    asyncio.create_task(update_live_context())
    print("[HPB] Live updater running (60 s interval).")

# ───────────────────────────────────────────────
# Live Context Updater
# ───────────────────────────────────────────────
live_context_cache = {
    "last_update": None, "symbol": "BTC-USDT-SWAP",
    "interval": "1H", "gates": {}, "ExecutionConfidence_Total": 0.0, "Reward_Summary": "INIT"
}

async def update_live_context():
    global live_context_cache
    while True:
        try:
            ctx = {"symbol": live_context_cache["symbol"], "interval": live_context_cache["interval"]}
            gates = validate_gates(ctx)
            live_context_cache.update({
                "last_update": datetime.utcnow().isoformat(),
                "gates": gates,
                "ExecutionConfidence_Total": gates.get("ExecutionConfidence_Total", 0.0),
                "Reward_Summary": gates.get("Reward_Summary", "N/A"),
            })
            print(f"[LIVE UPDATE] Refreshed @ {live_context_cache['last_update']}")
        except Exception as e:
            print(f"[LIVE UPDATE ERROR] {e}")
        await asyncio.sleep(60)

# ───────────────────────────────────────────────
# Keepalive (Threaded)
# ───────────────────────────────────────────────
def touch_keepalive():
    while True:
        try:
            path = "/tmp/render_keepalive.flag"
            subprocess.run(["touch", path], check=True)
            print(f"[KEEPALIVE] Touched {path} @ {datetime.utcnow().isoformat()}")
        except Exception as e:
            print(f"[KEEPALIVE ERROR] {e}")
        time.sleep(600)

threading.Thread(target=touch_keepalive, daemon=True).start()

# ───────────────────────────────────────────────
# Signals Endpoint – uses Phase 7.3 Risk Model
# ───────────────────────────────────────────────
@app.get("/signals")
async def get_signals():
    try:
        profile = await compute_risk_profile(live_context_cache)  # ✅ Fixed await
        return JSONResponse(profile)
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)

# ───────────────────────────────────────────────
# Run Server
# ───────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)
