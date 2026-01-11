import os
import json
import time
import threading
import asyncio
from datetime import datetime
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
import ccxt
import pandas as pd
import plotly.graph_objects as go
from apscheduler.schedulers.background import BackgroundScheduler

app = FastAPI(title="HPB–TCT v19 RIG EXTENDED (Phase 9)")

# ────────────────────────────────
# ENVIRONMENT CONFIGURATION
# ────────────────────────────────
BINANCE_KEY = os.getenv("BINANCE_KEY")
BINANCE_SECRET = os.getenv("BINANCE_SECRET")
BINANCE_MODE = os.getenv("BINANCE_MODE", "testnet").lower()

exchange = None

def init_exchange():
    """Initialize Binance client based on environment (testnet/live)."""
    global exchange
    try:
        if BINANCE_MODE == "testnet":
            exchange = ccxt.binance({
                "apiKey": BINANCE_KEY,
                "secret": BINANCE_SECRET,
                "enableRateLimit": True,
            })
            exchange.set_sandbox_mode(True)
            print("[EXCHANGE] Connected to Binance Testnet")
        else:
            exchange = ccxt.binance({
                "apiKey": BINANCE_KEY,
                "secret": BINANCE_SECRET,
                "enableRateLimit": True,
            })
            print("[EXCHANGE] Connected to Binance Live Environment")

        print(f"[HPB] Environment: {BINANCE_MODE.upper()} active")
        return True
    except Exception as e:
        print(f"[EXCHANGE ERROR] {e}")
        return False

# ────────────────────────────────
# KEEPALIVE THREAD (Render Free Tier)
# ────────────────────────────────
def touch_keepalive():
    """Prevents Render free dyno from sleeping."""
    while True:
        try:
            keepalive_file = "/tmp/render_keepalive.flag"
            with open(keepalive_file, "a"):
                os.utime(keepalive_file, None)
            print(f"[KEEPALIVE] Touched {keepalive_file} @ {datetime.utcnow().isoformat()}")
        except Exception as e:
            print(f"[KEEPALIVE ERROR] {e}")
        time.sleep(600)  # every 10 minutes

threading.Thread(target=touch_keepalive, daemon=True).start()

# ────────────────────────────────
# DATA FETCHING + DASHBOARD
# ────────────────────────────────
def fetch_price_data(symbol="BTC/USDT", timeframe="1h", limit=200):
    try:
        candles = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df
    except Exception as e:
        print(f"[FETCH ERROR] {e}")
        return pd.DataFrame()

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    try:
        df = fetch_price_data()
        if df.empty:
            return HTMLResponse("<h3>No data available.</h3>")

        fig = go.Figure(
            data=[go.Candlestick(
                x=df["timestamp"],
                open=df["open"],
                high=df["high"],
                low=df["low"],
                close=df["close"]
            )]
        )
        fig.update_layout(
            title="HTF Range (4H) | Distribution Structure",
            xaxis_title="Time",
            yaxis_title="Price (USDT)",
            template="plotly_dark",
            height=600
        )
        graph_html = fig.to_html(include_plotlyjs="cdn")

        return HTMLResponse(f"""
        <h2>HPM–TCT v19 RIG EXTENDED Dashboard (Phase 9)</h2>
        <p>Updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC</p>
        <p><b>Mode:</b> {BINANCE_MODE.upper()}</p>
        {graph_html}
        """)
    except Exception as e:
        return HTMLResponse(f"<h3>Error generating dashboard: {e}</h3>")

# ────────────────────────────────
# STATUS + TRADE EXECUTION
# ────────────────────────────────
@app.get("/status")
async def status():
    """Check exchange connection."""
    try:
        ticker = exchange.fetch_ticker("BTC/USDT")
        return JSONResponse({
            "exchange": "binance-testnet" if BINANCE_MODE == "testnet" else "binance-live",
            "status": "connected",
            "mode": BINANCE_MODE,
            "symbol": ticker["symbol"],
            "price": ticker["last"],
            "time": datetime.utcnow().isoformat()
        })
    except Exception as e:
        return JSONResponse({"status": "error", "details": str(e)})

@app.get("/execute")
async def execute_trade(symbol: str = "BTC/USDT", side: str = "buy", size: float = 0.001):
    """Simulate or execute a trade depending on mode."""
    try:
        if BINANCE_MODE == "testnet":
            print(f"[TRADE TEST] Simulating {side.upper()} {size} {symbol}")
            return JSONResponse({
                "mode": "testnet",
                "status": "simulated",
                "action": side,
                "size": size,
                "symbol": symbol,
                "time": datetime.utcnow().isoformat()
            })
        else:
            order = exchange.create_market_order(symbol, side, size)
            print(f"[TRADE LIVE] {side.upper()} order executed: {order}")
            return JSONResponse({
                "mode": "live",
                "status": "executed",
                "order": order
            })
    except Exception as e:
        print(f"[TRADE ERROR] {e}")
        return JSONResponse({"status": "error", "details": str(e)})

# ────────────────────────────────
# BACKGROUND REFRESHER
# ────────────────────────────────
def refresh_loop():
    """Periodic data fetch to keep system warm."""
    while True:
        try:
            df = fetch_price_data()
            if not df.empty:
                print(f"[REFRESH] Updated market data at {datetime.utcnow().isoformat()}")
        except Exception as e:
            print(f"[REFRESH ERROR] {e}")
        time.sleep(60)

threading.Thread(target=refresh_loop, daemon=True).start()

# ────────────────────────────────
# STARTUP
# ────────────────────────────────
@app.on_event("startup")
async def startup_event():
    print("[INIT] Starting HPB–TCT Server (Phase 9)")
    init_exchange()
