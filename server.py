import os
import ccxt
import threading
import time
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, HTMLResponse
from loguru import logger

# ───────────────────────────────
# CONFIGURATION
# ───────────────────────────────
EXCHANGE = os.getenv("EXCHANGE", "mexc").lower()
TRADE_MODE = os.getenv("TRADE_MODE", "paper").lower()  # 'paper' or 'live'

# OKX
OKX_KEY = os.getenv("OKX_KEY", "")
OKX_SECRET = os.getenv("OKX_SECRET", "")
OKX_PASSPHRASE = os.getenv("OKX_PASSPHRASE", "")
OKX_MODE = os.getenv("OKX_MODE", "live").lower()

# MEXC
MEXC_KEY = os.getenv("MEXC_KEY", "")
MEXC_SECRET = os.getenv("MEXC_SECRET", "")

# BINANCE
BINANCE_KEY = os.getenv("BINANCE_KEY", "")
BINANCE_SECRET = os.getenv("BINANCE_SECRET", "")

app = FastAPI(title="HPB–TCT Server", version="9.8 Unified Pro Hybrid")

# ───────────────────────────────
# GLOBAL STATE
# ───────────────────────────────
exchange = None
startup_log = []
event_log = []
MAX_LOG = 25
trade_state = {"mode": TRADE_MODE, "open": False, "entry": None, "side": None, "pnl": 0.0}

def log_event(msg):
    ts = datetime.utcnow().isoformat()
    entry = f"[{ts}] {msg}"
    event_log.append(entry)
    if len(event_log) > MAX_LOG:
        event_log.pop(0)
    logger.info(msg)

# ───────────────────────────────
# CONNECT + FAILOVER
# ───────────────────────────────
def connect_exchange():
    global startup_log
    startup_log = []
    connected = None

    def try_okx():
        if OKX_KEY and OKX_SECRET and OKX_PASSPHRASE:
            try:
                okx = ccxt.okx({
                    "apiKey": OKX_KEY,
                    "secret": OKX_SECRET,
                    "password": OKX_PASSPHRASE,
                    "enableRateLimit": True,
                })
                okx.fetch_ticker("BTC/USDT")
                startup_log.append("[OKX] ✅ Connected successfully")
                return okx
            except Exception as e:
                startup_log.append(f"[OKX] ❌ {e}")
        return None

    def try_mexc():
        if MEXC_KEY and MEXC_SECRET:
            try:
                mexc = ccxt.mexc({
                    "apiKey": MEXC_KEY,
                    "secret": MEXC_SECRET,
                    "enableRateLimit": True,
                })
                mexc.fetch_ticker("BTC/USDT")
                startup_log.append("[MEXC] ✅ Connected successfully")
                return mexc
            except Exception as e:
                startup_log.append(f"[MEXC] ❌ {e}")
        return None

    def try_binance():
        if BINANCE_KEY and BINANCE_SECRET:
            try:
                binance = ccxt.binance({
                    "apiKey": BINANCE_KEY,
                    "secret": BINANCE_SECRET,
                    "enableRateLimit": True,
                })
                binance.fetch_ticker("BTC/USDT")
                startup_log.append("[BINANCE] ✅ Connected successfully")
                return binance
            except Exception as e:
                startup_log.append(f"[BINANCE] ❌ {e}")
        return None

    connected = try_okx() or try_mexc() or try_binance()
    if connected:
        log_event(f"[EXCHANGE] Connected to {connected.id.upper()} (REST verified)")
    else:
        log_event("[EXCHANGE] ❌ No valid connection")
    return connected

def failover_loop():
    global exchange
    while True:
        try:
            if not exchange:
                exchange = connect_exchange()
            else:
                exchange.fetch_ticker("BTC/USDT")
        except Exception as e:
            log_event(f"[FAILOVER] Connection lost: {e}")
            exchange = connect_exchange()
        time.sleep(60)

# ───────────────────────────────
# BASIC ROUTES
# ───────────────────────────────
@app.get("/")
def root():
    return {
        "message": "HPB–TCT Unified Pro Hybrid running",
        "exchange": exchange.id if exchange else None,
        "trade_mode": TRADE_MODE,
        "timestamp": datetime.utcnow().isoformat(),
    }

@app.get("/status")
def status():
    if not exchange:
        return {"connected": False, "error": "No active exchange"}
    try:
        ticker = exchange.fetch_ticker("BTC/USDT")
        return {
            "exchange": exchange.id,
            "connected": True,
            "price": ticker.get("last"),
            "utc": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        return {"exchange": exchange.id, "connected": False, "error": str(e)}

# ───────────────────────────────
# DASHBOARD
# ───────────────────────────────
@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    if not exchange:
        return HTMLResponse("<h3>No active exchange connection.</h3>")
    try:
        candles = exchange.fetch_ohlcv("BTC/USDT", timeframe="1h", limit=100)
        df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

        fig = go.Figure(
            data=[go.Candlestick(
                x=df["timestamp"],
                open=df["open"],
                high=df["high"],
                low=df["low"],
                close=df["close"]
            )]
        )
        fig.update_layout(template="plotly_dark", title=f"{exchange.id.upper()} BTC/USDT (1h)", height=600)
        html = fig.to_html(include_plotlyjs="cdn")
        return HTMLResponse(f"""
        <h2>HPB–TCT Dashboard – {exchange.id.upper()}</h2>
        <p>Mode: <b>{TRADE_MODE.upper()}</b> | Updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC</p>
        {html}
        """)
    except Exception as e:
        return HTMLResponse(f"<h3>Error generating dashboard: {e}</h3>")

# ───────────────────────────────
# PAPER TRADING
# ───────────────────────────────
@app.post("/signals")
async def signals(req: Request):
    global trade_state
    data = await req.json()
    side = data.get("action", "").lower()
    size = float(data.get("size", 0.001))
    symbol = data.get("symbol", "BTC/USDT")
    price = 0.0
    try:
        price = exchange.fetch_ticker(symbol)["last"]
    except Exception:
        pass

    if side == "buy":
        trade_state.update({"open": True, "entry": price, "side": "LONG"})
        log_event(f"[PAPER] BUY {size} {symbol} @ {price}")
    elif side == "sell" and trade_state["open"]:
        pnl = (price - trade_state["entry"]) / trade_state["entry"]
        trade_state.update({"open": False, "pnl": pnl})
        log_event(f"[PAPER] SELL {size} {symbol} @ {price} | PNL {pnl:.4%}")
    else:
        log_event(f"[PAPER] Invalid action '{side}'")
    return JSONResponse({"trade_state": trade_state})

# ───────────────────────────────
# LIVE TRADE EXECUTION
# ───────────────────────────────
@app.post("/realtrade")
async def realtrade(req: Request):
    global trade_state
    if TRADE_MODE != "live":
        return JSONResponse({"error": "TRADE_MODE is set to paper — no live trades executed."})
    if not exchange:
        return JSONResponse({"error": "Exchange not connected."})

    data = await req.json()
    symbol = data.get("symbol", "BTC/USDT")
    side = data.get("action", "").lower()
    size = float(data.get("size", 0.001))

    try:
        order = exchange.create_market_order(symbol, side, size)
        trade_state.update({
            "last_order": order,
            "last_side": side,
            "symbol": symbol,
            "size": size,
            "timestamp": datetime.utcnow().isoformat(),
        })
        log_event(f"[LIVE] {side.upper()} {size} {symbol} – Executed")
        return JSONResponse({"status": "success", "order": order})
    except Exception as e:
        log_event(f"[LIVE ERROR] {e}")
        return JSONResponse({"status": "error", "error": str(e)})

# ───────────────────────────────
# DEBUG + DIAGNOSTICS
# ───────────────────────────────
@app.get("/debug/logs")
def debug_logs():
    return JSONResponse({"count": len(event_log), "entries": event_log})

@app.get("/diagnostics/full")
def diagnostics():
    diag = {
        "phase": "9.8 Unified Pro Hybrid",
        "exchange_connected": exchange.id if exchange else None,
        "trade_mode": TRADE_MODE,
        "okx_mode": OKX_MODE,
        "available_keys": {
            "okx": bool(OKX_KEY),
            "mexc": bool(MEXC_KEY),
            "binance": bool(BINANCE_KEY),
        },
        "startup_log": startup_log,
        "utc": datetime.utcnow().isoformat(),
    }
    return JSONResponse(diag)

# ───────────────────────────────
# STARTUP
# ───────────────────────────────
@app.on_event("startup")
async def startup_event():
    global exchange
    log_event(f"[INIT] Startup event triggered (Mode: {TRADE_MODE})")
    exchange = connect_exchange()
    threading.Thread(target=failover_loop, daemon=True).start()
    log_event("[SYSTEM] Failover monitor started")
