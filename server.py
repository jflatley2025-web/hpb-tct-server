"""
server.py
────────────────────────────────────────────
HPM–TCT v19 RIG EXTENDED Server
Integrates TensorTrade v1.0.3 Environment + HPB Gate Logic + Algo Overlays
────────────────────────────────────────────
"""

import gym_fix  # must be first

import os
import json
import traceback
from datetime import datetime
from fastapi import FastAPI
from fastapi.responses import JSONResponse, HTMLResponse

from tensortrade_config_ext import AUTO_INIT, TENSORTRADE_CONFIG, snapshot_environment
from High_Probability_Model_v17_RIG.validate_gates import validate_gates

import pandas as pd
import numpy as np
import plotly.graph_objs as go
import httpx
from apscheduler.schedulers.background import BackgroundScheduler
import asyncio

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
# Validation & Backtest Endpoints
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
        return JSONResponse({"error": str(e), "trace": traceback.format_exc()}, status_code=500)

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
        return JSONResponse({"error": str(e), "trace": traceback.format_exc()}, status_code=500)

@app.get("/status")
def status():
    try:
        snapshot = json.loads(snapshot_environment(ENV))
        return {"status": "ready", "env": snapshot}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# ───────────────────────────────────────────────
# Fetch + Algo Link
# ───────────────────────────────────────────────
async def fetch_algo_context(symbol="BTC-USDT-SWAP", interval="1H"):
    """Fetch latest gate/validation context from algo core."""
    try:
        context = {"symbol": symbol, "interval": interval}
        gates = validate_gates(context)
        print(f"[ALGO] Gates fetched successfully for {symbol} ({interval})")
        return gates
    except Exception as e:
        print(f"[ALGO ERROR] {e}")
        return {"error": str(e)}

async def fetch_ohlc(symbol="BTC-USDT", interval="1H", limit=200):
    """Fetch OHLC data from OKX (handles variable column count)."""
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
# Chart Renderer (Algo-aware)
# ───────────────────────────────────────────────
def plot_range_chart(df: pd.DataFrame, title: str, algo_data=None):
    if df.empty:
        return go.Figure().update_layout(title=f"{title} – No data", template="plotly_dark")

    import scipy.signal as sig
    from scipy.interpolate import make_interp_spline

    high, low = df["h"].max(), df["l"].min()
    eq = (high + low) / 2
    x = np.arange(len(df))
    lows = df["l"].to_numpy()
    highs = df["h"].to_numpy()

    swing_low_idx, _ = sig.find_peaks(-lows, distance=5)
    swing_high_idx, _ = sig.find_peaks(highs, distance=5)

    def classify_pivots(values, idx, window=5):
        prim, internal = [], []
        for i in idx:
            left = values[max(0, i - window):i]
            right = values[i + 1:i + 1 + window]
            if len(left) == 0 or len(right) == 0:
                continue
            if values[i] == max(np.concatenate([left, [values[i]], right])):
                prim.append(i)
            else:
                internal.append(i)
        return prim, internal

    primary_highs, _ = classify_pivots(highs, swing_high_idx)
    primary_lows, _ = classify_pivots(-lows, swing_low_idx)

    slope = np.polyfit(x[-min(len(df), 40):], df["c"].values[-min(len(df), 40):], 1)[0]
    phase = "accumulation" if slope >= 0 else "distribution"

    def is_higher_low_seq(idx, values):
        return len(idx) >= 2 and sum(values[idx[i + 1]] > values[idx[i]] for i in range(len(idx) - 1)) / (len(idx) - 1) > 0.4

    def is_lower_high_seq(idx, values):
        return len(idx) >= 2 and sum(values[idx[i + 1]] < values[idx[i]] for i in range(len(idx) - 1)) / (len(idx) - 1) > 0.6

    valid_accum = is_higher_low_seq(primary_lows, -lows)
    valid_dist = is_lower_high_seq(primary_highs, highs)

    liquidity_segments = []
    if (phase == "accumulation" and valid_accum) or (phase == "distribution" and valid_dist):
        pivots = primary_lows if phase == "accumulation" else primary_highs
        values = lows if phase == "accumulation" else highs
        for i in range(len(pivots) - 1):
            p1, p2 = pivots[i], pivots[i + 1]
            x_seg = np.arange(p1, p2 + 1)
            y_seg = values[x_seg]
            if len(x_seg) < 3:
                continue
            coeffs = np.polyfit(x_seg, y_seg, 2)
            poly = np.poly1d(coeffs)
            smooth_x = np.linspace(p1, p2, 60)
            smooth_y = poly(smooth_x)
            time_seg = np.interp(smooth_x, x, df["ts"].astype("int64"))
            time_seg = pd.to_datetime(time_seg)
            liquidity_segments.append((time_seg, smooth_y))

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df["ts"], open=df["o"], high=df["h"], low=df["l"], close=df["c"], name="Price"
    ))
    fig.add_hline(y=high, line=dict(color="red", dash="dash"), annotation_text="Range High")
    fig.add_hline(y=low, line=dict(color="green", dash="dash"), annotation_text="Range Low")
    fig.add_hline(y=eq, line=dict(color="orange", dash="dot"), annotation_text="EQ")

    for t_seg, y_seg in liquidity_segments:
        fig.add_trace(go.Scatter(
            x=t_seg, y=y_seg, mode="lines", line=dict(color="blue", width=3),
            name="Liquidity Curve", showlegend=False
        ))

    # Overlay Algo Structural Data
    if algo_data:
        exec_conf = algo_data.get("ExecutionConfidence_Total", 0)
        rcm = algo_data.get("RCM", {})
        rig = algo_data.get("RIG", {})
        phase_label = algo_data.get("1A", {}).get("bias", "Neutral").title()
        fig.add_annotation(text=f"ExecutionConfidence: {exec_conf:.2f}",
                           xref="paper", yref="paper", x=0.98, y=0.98,
                           showarrow=False, font=dict(size=14, color="cyan"),
                           bgcolor="rgba(0,0,0,0.6)")
        if not rig.get("passed", True):
            fig.add_annotation(text=f"⚠️ RIG Blocked ({rig.get('reason','')})",
                               xref="paper", yref="paper", x=0.98, y=0.93,
                               showarrow=False, font=dict(size=12, color="red"),
                               bgcolor="rgba(0,0,0,0.6)")
        if rcm.get("valid", False):
            range_high, range_low = rcm.get("range_high"), rcm.get("range_low")
            if range_high and range_low:
                fig.add_hrect(y0=range_low, y1=range_high, line_width=0,
                              fillcolor="rgba(0,128,255,0.2)",
                              annotation_text=f"RCM Valid Range ({phase_label})",
                              annotation_position="top left")

    fig.update_layout(
        title=f"{title} | {phase.title()} Structure" if liquidity_segments else f"{title} | No TCT Curve",
        template="plotly_dark", height=600, xaxis_rangeslider_visible=False,
        margin=dict(l=30, r=30, t=60, b=30)
    )
    return fig

# ───────────────────────────────────────────────
# Dashboard Generator
# ───────────────────────────────────────────────
latest_dashboard_html = "<h3>Initializing dashboard...</h3>"

async def generate_dashboard():
    global latest_dashboard_html
    try:
        algo_context = await fetch_algo_context("BTC-USDT-SWAP", "1H")
        htf_df = await fetch_ohlc(interval="4H")
        mtf_df = await fetch_ohlc(interval="1H")
        ltf_df = await fetch_ohlc(interval="15m")

        htf_fig = plot_range_chart(htf_df, "HTF Range (4H)", algo_context)
        mtf_fig = plot_range_chart(mtf_df, "MTF Range (1H)", algo_context)
        ltf_fig = plot_range_chart(ltf_df, "LTF Range (15m)", algo_context)

        html = (
            f"<h2>HPM–TCT v19 RIG EXTENDED Dashboard</h2>"
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
    return latest_dashboard_html

scheduler = BackgroundScheduler()
scheduler.add_job(lambda: asyncio.run(generate_dashboard()), "interval", hours=24)
scheduler.start()

@app.on_event("startup")
async def startup_event():
    print("[HPB] Initializing dashboard...")
    await generate_dashboard()

# ───────────────────────────────────────────────
# Run Server
# ───────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)
