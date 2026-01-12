import os
import json
import time
import threading
from datetime import datetime
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
import ccxt
import pandas as pd
import plotly.graph_objects as go

app = FastAPI(title="HPB–TCT v19 RIG EXTENDED (Phase 9.4 OKX)")

# ───────────────────────────────
# CONFIGURATION
# ───────────────────────────────
OKX_KEY = os.getenv("OKX_KEY", "")
OKX_SECRET = os.getenv("OKX_SECRET", "")
OKX_PASSPHRASE = os.getenv("OKX_PASSPHRASE", "")
OKX_MODE = os.getenv("OKX_MODE", "testnet").lower()

STATE_FILE = "/tmp/trade_state.json"
exchange = None

trade_state = {
    "position": "FLAT",
    "entry_price": None,
    "pnl": 0.0,
    "trades": []
}

# ───────────────────────────────
# STATE HANDLING
# ───────────────────────────────
def load_state():
    global trade_state
    try:
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, "r") as f:
                trade_state.update(json.load(f))
            print("[STATE] Loaded trade state.")
    except Exception as e:
        print(f"[STATE ERROR] {e}")

def save_state():
    try:
        with open(STATE_FILE, "w") as f:
            json.dump(trade_state, f)
    except Exception as e:
        print(f"[STATE SAVE ERROR] {e}")

# ───────────────────────────────
# EXCHANGE INITIALIZATION
# ───────────────────────────────
def init_exchange():
    """Initialize OKX connection."""
    global exchange
    try:
        exchange = ccxt.okx({
            "apiKey": OKX_KEY,
            "secret": OKX_SECRET,
            "password": OKX_PASSPHRASE,
            "enableRateLimit": True
        })
        if OKX_MODE == "testnet":
            exchange.set_sandbox_mode(True)
            print("[EXCHANGE] Connected to OKX Testnet")
        else:
            print("[EXCHANGE] Connected to OKX Live")
        print(f"[HPB] Environment: {OKX_MODE.upper()} active")
        return True
    except Exception as e:
        print(f"[EXCHANGE ERROR] {e}")
        return False

# ───────────────────────────────
# KEEPALIVE THREAD
# ───────────────────────────────
def keepalive():
    while True:
        try:
            flag_path = "/tmp/render_keepalive.flag"
            with open(flag_path, "a"):
                os.utime(flag_path, None)
            print(f"[KEEPALIVE] Updated flag @ {datetime.utcnow().isoformat()}")
        except Exception as e:
            print(f"[KEEPALIVE ERROR] {e}")
        time.sleep(600)

threading.Thread(target=keepalive, daemon=True).start()

# ───────────────────────────────
# DATA FETCHING (FIXED SYMBOL FORMAT)
# ───────────────────────────────
def fetch_price_data(symbol="BTC-USDT", timeframe="1h", limit=200):
    """
    OKX uses hyphen-separated symbols (e.g. BTC-USDT).
    """
    try:
        if exchange is None:
            raise Exception("Exchange not initialized.")
        markets = exchange.load_markets()
        if symbol not in markets:
            print(f"[WARN] {symbol} not found. Falling back to BTC-USDT.")
            symbol = "BTC-USDT"
        candles = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(
            candles,
            columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df
    except Exception as e:
        print(f"[FETCH ERROR] {e}")
        return pd.DataFrame()

# ───────────────────────────────
# DASHBOARD ENDPOINT
# ───────────────────────────────
@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    try:
        df = fetch_price_data()
        if df.empty:
            return HTMLResponse("<h3>No data available (Exchange uninitialized or restricted).</h3>")

        fig = go.Figure(data=[
            go.Candlestick(
                x=df["timestamp"],
                open=df["open"],
                high=df["high"],
                low=df["low"],
                close=df["close"]
            )
        ])
        fig.update_layout(
            title="HTF 4H | Distribution Structure (OKX)",
            template="plotly_dark",
            height=600
        )

        html = fig.to_html(include_plotlyjs="cdn")
        return HTMLResponse(f"""
        <h2>HPM–TCT v19 RIG EXTENDED Dashboard (Phase 9.4 OKX)</h2>
        <p>Updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC</p>
        <p><b>Mode:</b> {OKX_MODE.upper()}</p>
        <p><b>Position:</b> {trade_state['position']} | <b>PNL:</b> {trade_state['pnl']:.4f}</p>
        {html}
        """)
    except Exception as e:
        return HTMLResponse(f"<h3>Error: {e}</h3>")

# ───────────────────────────────
# STATUS ENDPOINT
# ───────────────────────────────
@app.get("/status")
async def status():
    try:
        ticker = exchange.fetch_ticker("BTC-USDT")
        return JSONResponse({
            "exchange": "okx-testnet" if OKX_MODE == "testnet" else "okx-live",
            "mode": OKX_MODE,
            "symbol": ticker["symbol"],
            "price": ticker["last"],
            "connected": True,
            "timestamp": datetime.utcnow().isoformat()
        })
    except Exception as e:
        return JSONResponse({"connected": False, "error": str(e)})

# ───────────────────────────────
# EXECUTION ENDPOINT
# ───────────────────────────────
@app.get("/execute")
async def execute_trade(symbol: str = "BTC-USDT", side: str = "buy", size: float = 0.001):
    try:
        price = exchange.fetch_ticker(symbol)["last"] if exchange else 0.0
        trade = {
            "symbol": symbol,
            "side": side,
            "size": size,
            "price": price,
            "time": datetime.utcnow().isoformat()
        }

        if OKX_MODE == "testnet":
            print(f"[TRADE TEST] Simulating {side.upper()} {size} {symbol} @ {price}")
            if side.lower() == "buy":
                trade_state["position"] = "LONG"
                trade_state["entry_price"] = price
            elif side.lower() == "sell" and trade_state["position"] == "LONG":
                entry = trade_state["entry_price"]
                trade_state["pnl"] = (price - entry) / entry
                trade_state["position"] = "FLAT"
                trade_state["entry_price"] = None

            trade_state["trades"].append(trade)
            save_state()
            return JSONResponse({"mode": "testnet", "trade": trade, "state": trade_state})

        # live mode
        order = exchange.create_market_order(symbol, side, size)
        trade_state["trades"].append(order)
        save_state()
        return JSONResponse({"mode": "live", "order": order})

    except Exception as e:
        print(f"[TRADE ERROR] {e}")
        return JSONResponse({"status": "error", "details": str(e)})

# ───────────────────────────────
# SIGNALS ENDPOINT (NEW)
# ───────────────────────────────
@app.post("/signals")
async def signal_handler(request: Request):
    """
    Receives JSON payloads for automated trade signals.
    Example:
    {
        "symbol": "BTC-USDT",
        "action": "buy",
        "size": 0.002
    }
    """
    try:
        data = await request.json()
        symbol = data.get("symbol", "BTC-USDT")
        action = data.get("action", "").lower()
        size = float(data.get("size", 0.001))

        if action not in ["buy", "sell"]:
            return JSONResponse({"status": "error", "message": "Invalid action"})

        print(f"[SIGNAL] Received: {action.upper()} {size} {symbol}")
        response = await execute_trade(symbol, action, size)
        return response
    except Exception as e:
        print(f"[SIGNAL ERROR] {e}")
        return JSONResponse({"status": "error", "details": str(e)})

# ───────────────────────────────
# STATE ENDPOINT
# ───────────────────────────────
@app.get("/state")
async def get_state():
    return JSONResponse({
        "position": trade_state["position"],
        "entry_price": trade_state["entry_price"],
        "pnl": trade_state["pnl"],
        "trade_count": len(trade_state["trades"]),
        "recent_trades": trade_state["trades"][-5:]
    })

# ───────────────────────────────
# BACKGROUND REFRESH LOOP
# ───────────────────────────────
def refresh_loop():
    while True:
        try:
            if exchange:
                df = fetch_price_data()
                if not df.empty:
                    print(f"[REFRESH] Market data updated @ {datetime.utcnow().isoformat()}")
            else:
                print("[REFRESH] Waiting for exchange init...")
        except Exception as e:
            print(f"[REFRESH ERROR] {e}")
        time.sleep(60)

# ───────────────────────────────
# STARTUP EVENT
# ───────────────────────────────
@app.on_event("startup")
async def startup_event():
    print("[INIT] Starting HPB–TCT Server (Phase 9.4 OKX)")
    load_state()
    ok = init_exchange()
    if ok:
        threading.Thread(target=refresh_loop, daemon=True).start()
        print("[SYSTEM] Market refresh thread started.")
    else:
        print("[SYSTEM] Exchange init failed; retry scheduled.")
        threading.Thread(target=init_exchange, daemon=True).start()
