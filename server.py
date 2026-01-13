import os
import ccxt
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from loguru import logger
from datetime import datetime

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

app = FastAPI(title="HPB–TCT Server", version="9.5-unified")

# ───────────────────────────────
# EXCHANGE CONNECTOR
# ───────────────────────────────
def connect_exchange():
    """
    Try to connect to OKX, MEXC, then Binance — in that order.
    Returns an active ccxt exchange instance and a connection summary.
    """
    active_exchange = None
    connection_log = []

    # 1️⃣ Try OKX
    if OKX_KEY and OKX_SECRET and OKX_PASSPHRASE:
        try:
            okx = ccxt.okx({
                "apiKey": OKX_KEY,
                "secret": OKX_SECRET,
                "password": OKX_PASSPHRASE,
                "enableRateLimit": True,
            })
            # Force a simple API call to validate connection
            okx.fetch_ticker("BTC/USDT")
            connection_log.append("[OKX] ✅ Connected successfully")
            active_exchange = okx
        except Exception as e:
            msg = str(e)
            connection_log.append(f"[OKX] ❌ Failed: {msg}")

    # 2️⃣ Try MEXC
    if not active_exchange and MEXC_KEY and MEXC_SECRET:
        try:
            mexc = ccxt.mexc({
                "apiKey": MEXC_KEY,
                "secret": MEXC_SECRET,
                "enableRateLimit": True,
            })
            mexc.fetch_ticker("BTC/USDT")
            connection_log.append("[MEXC] ✅ Connected successfully")
            active_exchange = mexc
        except Exception as e:
            msg = str(e)
            connection_log.append(f"[MEXC] ❌ Failed: {msg}")

    # 3️⃣ Try Binance
    if not active_exchange and BINANCE_KEY and BINANCE_SECRET:
        try:
            binance = ccxt.binance({
                "apiKey": BINANCE_KEY,
                "secret": BINANCE_SECRET,
                "enableRateLimit": True,
            })
            binance.fetch_ticker("BTC/USDT")
            connection_log.append("[BINANCE] ✅ Connected successfully")
            active_exchange = binance
        except Exception as e:
            msg = str(e)
            connection_log.append(f"[BINANCE] ❌ Failed: {msg}")

    if active_exchange:
        logger.info(f"[EXCHANGE] Connected to {active_exchange.id.upper()} (REST verified)")
    else:
        logger.warning("[EXCHANGE] ❌ No valid exchange connection available")

    return active_exchange, connection_log

# initialize at startup
exchange, startup_log = connect_exchange()

# ───────────────────────────────
# API ENDPOINTS
# ───────────────────────────────

@app.get("/")
def root():
    return {
        "message": "HPB–TCT Server Active",
        "exchange": exchange.id if exchange else "none",
        "timestamp": datetime.utcnow().isoformat(),
    }

@app.get("/status")
def status():
    """
    Check current exchange connectivity and BTC/USDT ticker
    """
    if not exchange:
        return {"connected": False, "error": "No active exchange"}
    try:
        ticker = exchange.fetch_ticker("BTC/USDT")
        return {
            "exchange": exchange.id,
            "connected": True,
            "price": ticker.get("last"),
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        return {"exchange": exchange.id, "connected": False, "error": str(e)}

# ───────────────────────────────
# Optional Diagnostic Tool
# ───────────────────────────────
@app.get("/diagnostics/full")
def diagnostics():
    """
    Returns environment overview (safe, no secrets)
    """
    diag = {
        "phase": "9.5 Unified Exchange Diagnostics",
        "exchange_selected": EXCHANGE,
        "exchange_connected": exchange.id if exchange else None,
        "modes": {
            "okx_mode": OKX_MODE,
        },
        "available_exchanges": {
            "okx_key": bool(OKX_KEY),
            "mexc_key": bool(MEXC_KEY),
            "binance_key": bool(BINANCE_KEY),
        },
        "startup_log": startup_log,
        "utc": datetime.utcnow().isoformat(),
    }
    return JSONResponse(diag)

# ───────────────────────────────
# KEEPALIVE + STARTUP EVENTS
# ───────────────────────────────
@app.on_event("startup")
async def startup_event():
    logger.info("[INIT] Startup event triggered")
    logger.info(f"[KEEPALIVE] Updated flag @ {datetime.utcnow().isoformat()}")
