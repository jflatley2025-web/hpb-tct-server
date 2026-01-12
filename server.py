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

app = FastAPI(title="HPB–TCT v19 RIG EXTENDED (Phase 9.4b OKX Stable)")

# ───────────────────────────────
# CONFIG
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
# STATE MANAGEMENT
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
    """Initialize OKX Exchange (Testnet or Live)"""
    global exchange
    try:
        exchange = ccxt.okx({
            "apiKey": OKX_KEY,
            "secret": OKX_SECRET,
            "password": OKX_PASSPHRASE,
            "enableRateLimit": True,
        })

        if OKX_MODE == "testnet":
            exchange.set_sandbox_mode(True)
            # ✅ full override with 'rest' key included
            exchange.urls["api"] = {
                "rest": "https://www.okx.com",
                "public": "https://www.okx.com/api/v5",
                "private": "https://www.okx.com/api/v5"
            }
            print("[EXCHANGE] Connected to OKX Testnet (URL override with REST endpoint)")
        else:
            print("[EXCHANGE] Connected to OKX Live")

        exchange.load_markets()
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
            path = "/tmp/render_keepalive.flag"
            with open(path, "a"):
                os.utime(path, None)
            print(f"[KEEPALIVE] Updated flag @ {datetime.utcnow().isoformat()}")
        except Exception as e:
            print(f"[KEEPALIVE ERROR] {e}")
        time.sleep(600)

threading.Thread(target=keepalive, daemon=True).start()

# ───────────────────────────────
# PRICE FETCHING (FIXED)
# ───────────────────────────────
def fetch_price_data(symbol="BTC-USDT", timeframe="1h", limit=200):
    """
    OKX uses hyphen-based symbols. We handle missing markets gracefully.
    """
    try:
        if not exchange:
            raise Exception("Exchange not initialized.")

        exchange.load_markets()

        # ✅ ensure valid symbol
        if symbol not in exchange.markets:
            print(f"[WARN] {symbol} not found, trying spot format BTC/USDT ...")
            if "BTC/USDT" in exchange.markets:
                symbol = "BTC/USDT"
            else:
                symbol = "BTC-USDT"

        market = exchange.market(symbol)
        if not market:
            raise Exception("Invalid market returned by OKX.")

        candles = exchange.fetch_ohlcv(market["symbol"], timeframe=timeframe, limit=limit)
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
    df = fetch_price_data()
    if df.empty:
        return HTMLResponse("<h3>No data available (exchange uninitialized or restricted).</h3>")
    fig = go.Figure(data=[
        go.Candlestick(
            x=df["timestamp"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"]
        )
    ])
    fig.update_layout(template="plotly_dark", title="OKX BTC/USDT 1h", height=600)
    html = fig.to_html(include_plotlyjs="cdn")
    return HTMLResponse(f"""
    <h2>HPB–TCT Phase 9.4b OKX Dashboard</h2>
    <p>Updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC</p>
    <p>Mode: {OKX_MODE.upper()} | Position: {trade_state['position']} | PNL: {trade_state['pnl']:.4f}</p>
    {html}
    """)

# ───────────────────────────────
# STATUS
# ───────────────────────────────
@app.get("/status")
async def status():
    try:
        ticker = exchange.fetch_ticker("BTC-USDT")
        return JSONResponse({
            "exchange": "okx",
            "mode": OKX_MODE,
            "price": ticker.get("last"),
            "timestamp": datetime.utcnow().isoformat(),
            "connected": True
        })
    except Exception as e:
        return JSONResponse({"connected": False, "error": str(e)})

# ───────────────────────────────
# EXECUTE TRADE
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
            print(f"[TRADE TEST] Simulated {side.upper()} {size} {symbol} @ {price}")
            if side == "buy":
                trade_state["position"] = "LONG"
                trade_state["entry_price"] = price
            elif side == "sell" and trade_state["position"] == "LONG":
                entry = trade_state["entry_price"] or price
                trade_state["pnl"] = (price - entry) / entry
                trade_state["position"] = "FLAT"
                trade_state["entry_price"] = None
            trade_state["trades"].append(trade)
            save_state()
            return JSONResponse({"mode": "testnet", "trade": trade, "state": trade_state})
        else:
            order = exchange.create_market_order(symbol, side, size)
            trade_state["trades"].append(order)
            save_state()
            return JSONResponse({"mode": "live", "order": order})
    except Exception as e:
        print(f"[TRADE ERROR] {e}")
        return JSONResponse({"error": str(e)})

# ───────────────────────────────
# SIGNALS ENDPOINT
# ───────────────────────────────
@app.post("/signals")
async def signal_handler(request: Request):
    try:
        data = await request.json()
        symbol = data.get("symbol", "BTC-USDT")
        action = data.get("action", "").lower()
        size = float(data.get("size", 0.001))
        print(f"[SIGNAL] {action.upper()} {size} {symbol}")
        return await execute_trade(symbol, action, size)
    except Exception as e:
        print(f"[SIGNAL ERROR] {e}")
        return JSONResponse({"error": str(e)})

# ───────────────────────────────
# STATE ENDPOINT
# ───────────────────────────────
@app.get("/state")
async def get_state():
    return JSONResponse(trade_state)

# ───────────────────────────────
# BACKGROUND REFRESH LOOP
# ───────────────────────────────
def refresh_loop():
    while True:
        try:
            df = fetch_price_data()
            if not df.empty:
                print(f"[REFRESH] Market data updated @ {datetime.utcnow().isoformat()}")
        except Exception as e:
            print(f"[REFRESH ERROR] {e}")
        time.sleep(60)

# ───────────────────────────────
# STARTUP EVENT
# ───────────────────────────────
@app.on_event("startup")
async def startup_event():
    print("[INIT] Starting HPB–TCT Server (Phase 9.4b OKX Stable)")
    load_state()
    if init_exchange():
        threading.Thread(target=refresh_loop, daemon=True).start()
        print("[SYSTEM] Market refresh thread started.")
