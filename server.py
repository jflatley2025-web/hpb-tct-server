import os
import json
import time
import threading
from datetime import datetime
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
import ccxt
import pandas as pd
import plotly.graph_objects as go

app = FastAPI(title="HPB–TCT v19 RIG EXTENDED (Phase 9.2)")

# ───────────────────────────────
# ENVIRONMENT CONFIGURATION
# ───────────────────────────────
BINANCE_KEY = os.getenv("BINANCE_KEY")
BINANCE_SECRET = os.getenv("BINANCE_SECRET")
BINANCE_MODE = os.getenv("BINANCE_MODE", "testnet").lower()

exchange = None
trade_state = {
    "position": "FLAT",
    "entry_price": None,
    "pnl": 0.0,
    "trades": []
}

# ───────────────────────────────
# INITIALIZE BINANCE CONNECTION
# ───────────────────────────────
def init_exchange():
    """Initialize Binance (Testnet or Live)."""
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
            print("[EXCHANGE] Connected to Binance Live")
        print(f"[HPB] Environment: {BINANCE_MODE.upper()} active")
        return True
    except Exception as e:
        print(f"[EXCHANGE ERROR] {e}")
        return False

# ───────────────────────────────
# KEEPALIVE (Render Free Tier)
# ───────────────────────────────
def touch_keepalive():
    while True:
        try:
            f = "/tmp/render_keepalive.flag"
            with open(f, "a"):
                os.utime(f, None)
            print(f"[KEEPALIVE] Touched {f} @ {datetime.utcnow().isoformat()}")
        except Exception as e:
            print(f"[KEEPALIVE ERROR] {e}")
        time.sleep(600)

threading.Thread(target=touch_keepalive, daemon=True).start()

# ───────────────────────────────
# MARKET DATA
# ───────────────────────────────
def fetch_price_data(symbol="BTC/USDT", timeframe="1h", limit=200):
    try:
        candles = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df
    except Exception as e:
        print(f"[FETCH ERROR] {e}")
        return pd.DataFrame()

# ───────────────────────────────
# DASHBOARD
# ───────────────────────────────
@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    try:
        df = fetch_price_data()
        if df.empty:
            return HTMLResponse("<h3>No data available (Exchange uninitialized).</h3>")

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
            title="HTF 4H | Distribution Structure",
            template="plotly_dark",
            height=600
        )
        html = fig.to_html(include_plotlyjs="cdn")
        return HTMLResponse(f"""
        <h2>HPM–TCT v19 RIG EXTENDED Dashboard (Phase 9.2)</h2>
        <p>Updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC</p>
        <p><b>Mode:</b> {BINANCE_MODE.upper()}</p>
        <p><b>Position:</b> {trade_state['position']} | <b>PNL:</b> {trade_state['pnl']:.4f}</p>
        {html}
        """)
    except Exception as e:
        return HTMLResponse(f"<h3>Error: {e}</h3>")

# ───────────────────────────────
# STATUS
# ───────────────────────────────
@app.get("/status")
async def status():
    try:
        ticker = exchange.fetch_ticker("BTC/USDT")
        return JSONResponse({
            "exchange": "binance-testnet" if BINANCE_MODE == "testnet" else "binance-live",
            "mode": BINANCE_MODE,
            "symbol": ticker["symbol"],
            "price": ticker["last"],
            "connected": True,
            "timestamp": datetime.utcnow().isoformat()
        })
    except Exception as e:
        return JSONResponse({"connected": False, "error": str(e)})

# ───────────────────────────────
# TRADE EXECUTION / SIMULATION
# ───────────────────────────────
@app.get("/execute")
async def execute_trade(symbol: str = "BTC/USDT", side: str = "buy", size: float = 0.001):
    try:
        price = exchange.fetch_ticker(symbol)["last"] if exchange else 0.0
        trade = {
            "symbol": symbol,
            "side": side,
            "size": size,
            "price": price,
            "time": datetime.utcnow().isoformat()
        }

        # Simulated mode logic
        if BINANCE_MODE == "testnet":
            print(f"[TRADE TEST] Simulating {side.upper()} {size} {symbol} @ {price}")

            if side.lower() == "buy":
                trade_state["position"] = "LONG"
                trade_state["entry_price"] = price
            elif side.lower() == "sell":
                if trade_state["position"] == "LONG":
                    entry = trade_state["entry_price"]
                    trade_state["pnl"] = (price - entry) / entry
                    trade_state["position"] = "FLAT"
                    trade_state["entry_price"] = None

            trade_state["trades"].append(trade)
            return JSONResponse({"mode": "testnet", "trade": trade, "state": trade_state})

        # Live mode (actual order)
        order = exchange.create_market_order(symbol, side, size)
        trade_state["trades"].append(order)
        return JSONResponse({"mode": "live", "order": order})

    except Exception as e:
        print(f"[TRADE ERROR] {e}")
        return JSONResponse({"status": "error", "details": str(e)})

# ───────────────────────────────
# TRADE STATE ENDPOINT
# ───────────────────────────────
@app.get("/state")
async def get_state():
    """Return current simulated position and PnL."""
    return JSONResponse({
        "position": trade_state["position"],
        "entry_price": trade_state["entry_price"],
        "pnl": trade_state["pnl"],
        "trade_count": len(trade_state["trades"]),
        "trades": trade_state["trades"][-5:]  # recent 5
    })

# ───────────────────────────────
# REFRESH THREAD (after init)
# ───────────────────────────────
def refresh_loop():
    while True:
        try:
            if exchange is not None:
                df = fetch_price_data()
                if not df.empty:
                    print(f"[REFRESH] Updated market data @ {datetime.utcnow().isoformat()}")
            else:
                print("[REFRESH] Waiting for exchange init…")
        except Exception as e:
            print(f"[REFRESH ERROR] {e}")
        time.sleep(60)

# ───────────────────────────────
# STARTUP
# ───────────────────────────────
@app.on_event("startup")
async def startup_event():
    print("[INIT] Starting HPB–TCT Server (Phase 9.2)")
    ok = init_exchange()
    if ok:
        threading.Thread(target=refresh_loop, daemon=True).start()
        print("[SYSTEM] Market refresh thread started.")
    else:
        print("[SYSTEM] Exchange init failed; retrying background mode.")
        threading.Thread(target=init_exchange, daemon=True).start()
