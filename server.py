import os
import ccxt
import threading
import time
from datetime import datetime
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from loguru import logger

# ───────────────────────────────
# CONFIG
# ───────────────────────────────
EXCHANGE = os.getenv("EXCHANGE", "mexc").lower()

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

app = FastAPI(title="HPB–TCT Server", version="9.6 Unified Pro")

# ───────────────────────────────
# GLOBALS
# ───────────────────────────────
exchange = None
startup_log = []
event_log = []        # rolling event buffer for /debug/logs
MAX_LOG = 25
trade_state = {"mode": "paper", "open": False, "entry": None, "pnl": 0.0}

def log_event(msg):
    ts = datetime.utcnow().isoformat()
    entry = f"[{ts}] {msg}"
    event_log.append(entry)
    if len(event_log) > MAX_LOG:
        event_log.pop(0)
    logger.info(msg)

# ───────────────────────────────
# EXCHANGE CONNECTION + FAILOVER
# ───────────────────────────────
def connect_exchange():
    """
    Attempt to connect in priority order: OKX → MEXC → Binance.
    Returns active exchange and connection log.
    """
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

    # priority order
    connected = try_okx() or try_mexc() or try_binance()
    if connected:
        log_event(f"[EXCHANGE] Connected to {connected.id.upper()} (REST verified)")
    else:
        log_event("[EXCHANGE] ❌ No valid connection")
    return connected

def failover_loop():
    """Continuously checks connection; auto-reconnects if lost."""
    global exchange
    while True:
        try:
            if not exchange:
                exchange = connect_exchange()
            else:
                # verify ticker call
                exchange.fetch_ticker("BTC/USDT")
        except Exception as e:
            log_event(f"[FAILOVER] Connection lost: {e}")
            exchange = connect_exchange()
        time.sleep(60)

# ───────────────────────────────
# API ENDPOINTS
# ───────────────────────────────
@app.get("/")
def root():
    return {
        "message": "HPB–TCT Unified Pro running",
        "exchange": exchange.id if exchange else None,
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
# PAPER-TRADE SIMULATOR
# ───────────────────────────────
@app.post("/signals")
async def signals(req: Request):
    global trade_state
    data = await req.json()
    side = data.get("action", "").lower()
    size = float(data.get("size", 0.001))
    symbol = data.get("symbol", "BTC/USDT")
    price = None
    try:
        price = exchange.fetch_ticker(symbol)["last"]
    except Exception:
        price = 0.0

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
# DEBUG LOGS ENDPOINT
# ───────────────────────────────
@app.get("/debug/logs")
def debug_logs():
    """Return the last 25 log entries."""
    return JSONResponse({"count": len(event_log), "entries": event_log})

# ───────────────────────────────
# DIAGNOSTICS
# ───────────────────────────────
@app.get("/diagnostics/full")
def diagnostics():
    diag = {
        "phase": "9.6 Unified Pro",
        "exchange_connected": exchange.id if exchange else None,
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
    log_event("[INIT] Startup event triggered")
    exchange = connect_exchange()
    threading.Thread(target=failover_loop, daemon=True).start()
    log_event("[SYSTEM] Failover monitor started")
