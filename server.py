# ───────────────────────────────
# server.py  |  HPB–TCT  Phase 9.4i OKX REST Stable + Diagnostics
# ───────────────────────────────
import os
import asyncio
from datetime import datetime
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import ccxt
from loguru import logger

# ───────────────────────────────
# CONFIG (values come from Render Environment Variables)
# ───────────────────────────────
OKX_KEY = os.getenv("OKX_KEY", "")
OKX_SECRET = os.getenv("OKX_SECRET", "")
OKX_PASSPHRASE = os.getenv("OKX_PASSPHRASE", "")
OKX_MODE = os.getenv("OKX_MODE", "testnet").lower()

# ───────────────────────────────
# FastAPI app setup
# ───────────────────────────────
app = FastAPI(title="HPB–TCT Server", version="9.4i")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ───────────────────────────────
# Initialize exchange
# ───────────────────────────────
logger.info("[INIT] Starting HPB–TCT Server (Phase 9.4i OKX REST Stable)")

try:
    exchange = ccxt.okx({
        "apiKey": OKX_KEY,
        "secret": OKX_SECRET,
        "password": OKX_PASSPHRASE,
        "enableRateLimit": True,
    })

    if OKX_MODE == "testnet":
        exchange.set_sandbox_mode(True)
        # override sandbox URLs to official okx.com testnet REST
        exchange.urls["api"]["rest"] = "https://www.okx.com"
        exchange.urls["api"]["public"] = "https://www.okx.com/api/v5"
        exchange.urls["api"]["private"] = "https://www.okx.com/api/v5"
        logger.info("[EXCHANGE] Connected to OKX Testnet (final REST override applied)")
    else:
        exchange.set_sandbox_mode(False)
        logger.info("[EXCHANGE] Connected to OKX Live")

except Exception as e:
    logger.error(f"[EXCHANGE INIT ERROR] {e}")
    exchange = None

# ───────────────────────────────
# Optional Diagnostic Tool
# ───────────────────────────────
@app.get("/debug/urls")
async def debug_urls():
    """
    Returns current mode, sandbox flag, and API URL configuration.
    Helps confirm whether app is using live or testnet endpoints.
    """
    try:
        return JSONResponse({
            "mode": OKX_MODE,
            "sandbox_mode": getattr(exchange, "sandbox", False),
            "urls": exchange.urls.get("api", {}) if exchange else None,
            "key_prefix": OKX_KEY[:6] + "..." if OKX_KEY else None
        })
    except Exception as e:
        return JSONResponse({"error": str(e)})

# ───────────────────────────────
# Status Endpoint
# ───────────────────────────────
@app.get("/status")
async def status():
    try:
        ticker = exchange.fetch_ticker("BTC/USDT")
        return JSONResponse({
            "exchange": "okx",
            "mode": OKX_MODE,
            "connected": True,
            "price": ticker["last"],
            "timestamp": datetime.utcnow().isoformat()
        })
    except Exception as e:
        logger.error(f"[STATUS ERROR] {e}")
        return JSONResponse({
            "exchange": "okx",
            "mode": OKX_MODE,
            "connected": False,
            "error": str(e)
        })

# ───────────────────────────────
# Keepalive Task
# ───────────────────────────────
async def keepalive():
    while True:
        logger.info(f"[KEEPALIVE] Updated flag @ {datetime.utcnow().isoformat()}")
        await asyncio.sleep(180)

# ───────────────────────────────
# Startup Event
# ───────────────────────────────
@app.on_event("startup")
async def startup_event():
    logger.info("[INIT] Startup event triggered")
    asyncio.create_task(keepalive())

# ───────────────────────────────
# Root Fallback
# ───────────────────────────────
@app.get("/")
async def root():
    return JSONResponse({
        "message": "HPB–TCT Server OK (Phase 9.4i)",
        "docs": "/docs",
        "status": "/status",
        "debug": "/debug/urls"
    })

# ───────────────────────────────
# Main entrypoint (Render uses this)
# ───────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=int(os.getenv("PORT", 10000)))
