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
    """
    TCT-aware liquidity visualization.
    Only draws liquidity arcs when structure confirms valid accumulation/distribution ranges
    as defined in 2025 Liquidity-1_E_REVIEW & Liquidity-1_T_REVIEW.
    """

    if df.empty:
        return go.Figure().update_layout(title=f"{title} – No data", template="plotly_dark")

    import numpy as np
    import scipy.signal as sig
    from scipy.interpolate import make_interp_spline

    # ───────────────────────────────────────────────
    # Step 1. Basic range metrics
    # ───────────────────────────────────────────────
    high, low = df["h"].max(), df["l"].min()
    eq = (high + low) / 2
    x = np.arange(len(df))

    lows = df["l"].to_numpy()
    highs = df["h"].to_numpy()

    # Detect swing highs/lows
    swing_low_idx, _ = sig.find_peaks(-lows, distance=5)
    swing_high_idx, _ = sig.find_peaks(highs, distance=5)

    # ───────────────────────────────────────────────
    # Step 2. Classify pivots: primary vs internal
    # (based on local dominance strength)
    # ───────────────────────────────────────────────
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

    primary_highs, internal_highs = classify_pivots(highs, swing_high_idx)
    primary_lows, internal_lows = classify_pivots(-lows, swing_low_idx)  # invert for lows

    # ───────────────────────────────────────────────
    # Step 3. Determine structure phase (TCT logic)
    # ───────────────────────────────────────────────
    # Calculate short-term slope to estimate structural bias
    slope = np.polyfit(x[-min(len(df), 40):], df["c"].values[-min(len(df), 40):], 1)[0]
    phase = "accumulation" if slope >= 0 else "distribution"

    # ───────────────────────────────────────────────
    # Step 4. Validate TCT structural conditions
    # ───────────────────────────────────────────────
    def is_higher_low_seq(idx, values):
        if len(idx) < 2:
            return False
        ups = sum(values[idx[i + 1]] > values[idx[i]] for i in range(len(idx) - 1))
        return ups / (len(idx) - 1) > 0.4  # allow 40% of pivots to rise

    def is_lower_high_seq(idx, values):
        if len(idx) < 2:
            return False
        downs = sum(values[idx[i + 1]] < values[idx[i]] for i in range(len(idx) - 1))
        return downs / (len(idx) - 1) > 0.6

    valid_accum = is_higher_low_seq(primary_lows, -lows)
    valid_dist = is_lower_high_seq(primary_highs, highs)


    # ───────────────────────────────────────────────
    # Step 5. Price-near-liquidity filter (TCT condition)
    # ───────────────────────────────────────────────
    def near_liquidity(df, pivot_i, window=8, tol=0.007):
        price = df["c"].iloc[pivot_i]
        nearby = df["c"].iloc[max(0, pivot_i - window):pivot_i + window]
        return (abs(nearby - price) / price < tol).sum() > window / 2

    # ───────────────────────────────────────────────
    # Step 6. Generate TCT-valid liquidity arcs
    # ───────────────────────────────────────────────
    liquidity_segments = []

    if phase == "accumulation" and valid_accum:
        pivots = primary_lows
        values = lows
    elif phase == "distribution" and valid_dist:
        pivots = primary_highs
        values = highs
    else:
        pivots = []
        values = []

    if len(pivots) >= 2:
        for i in range(len(pivots) - 1):
            p1, p2 = pivots[i], pivots[i + 1]
            if not (near_liquidity(df, p1) and near_liquidity(df, p2)):
                continue

            # fit parabolic arc between pivots
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

    # ───────────────────────────────────────────────
    # Step 6.5 — Debug Diagnostics (Print to Render Logs)
    # ───────────────────────────────────────────────
    print(f"[TCT] ---- {title} ----")
    print(f"[TCT] phase: {phase}")
    print(f"[TCT] primary_lows: {primary_lows}")
    print(f"[TCT] primary_highs: {primary_highs}")
    print(f"[TCT] valid_accum: {valid_accum}, valid_dist: {valid_dist}")
    print(f"[TCT] pivots used: {len(pivots)}")
    print(f"[TCT] segments generated: {len(liquidity_segments)}")



    # ───────────────────────────────────────────────
    # Step 7. Plotly visualization
    # ───────────────────────────────────────────────
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df["ts"], open=df["o"], high=df["h"], low=df["l"], close=df["c"], name="Price"
    ))

    fig.add_hline(y=high, line=dict(color="red", dash="dash"), annotation_text="Range High")
    fig.add_hline(y=low, line=dict(color="green", dash="dash"), annotation_text="Range Low")
    fig.add_hline(y=eq, line=dict(color="orange", dash="dot"), annotation_text="EQ")

    for t_seg, y_seg in liquidity_segments:
        fig.add_trace(go.Scatter(
            x=t_seg,
            y=y_seg,
            mode="lines",
            line=dict(color="blue", width=3),
            name="Liquidity Curve",
            showlegend=False
        ))

    fig.update_layout(
        title=f"{title} | {phase.title()} Structure" if liquidity_segments else f"{title} | No TCT Curve",
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
