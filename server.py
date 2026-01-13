# ───────────────────────────────
# server.py  |  HPB–TCT  Phase 9.4j OKX REST Live/Testnet Fix + Diagnostics
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
# CONFIG (Render environment variables)
# ───────────────────────────────
OKX_KEY = os.getenv("OKX_KEY", "")
OKX_SECRET = os.getenv("OKX_SECRET", "")
OKX_PASSPHRASE = os.getenv("OKX_PASSPHRASE", "")
OKX_MODE = os.getenv("OKX_MODE", "testnet").lower()

# ───────────────────────────────
# FastAPI app setup
# ───────────────────────────────
app = FastAPI(title="HPB–TCT Server", version="9.4j")

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
logger.info(f"[INIT] Starting HPB–TCT Server (Phase 9.4j OKX REST {OKX_MODE.upper()})")

try:
    exchange = ccxt.okx({
        "apiKey": OKX_KEY,
        "secret": OKX_SECRET,
        "password": OKX_PASSPHRASE,
        "enableRateLimit": True,
    })

    # Force-correct the REST URLs
    if OKX_MODE == "testnet":
        exchange.set_sandbox_mode(True)
        exchange.urls["api"]["rest"] = "https://www.okx.com"
        exchange.urls["api"]["public"] = "https://www.okx.com/api/v5"
        exchange.urls["api"]["private"] = "https://www.okx.com/api/v5"
        logger.info("[EXCHANGE] Connected to OKX Testnet (override applied)")
    else:
        exchange.set_sandbox_mode(False)
        exchange.urls["api"]["rest"] = "https://www.okx.com"
        exchange.urls["api"]["public"] = "https://www.okx.com/api/v5"
        exchange.urls["api"]["private"] = "https://www.okx.com/api/v5"
        logger.info("[EXCHANGE] Connected to OKX Live (REST override applied)")

except Exception as e:
    logger.error(f"[EXCHANGE INIT ERROR] {e}")
    exchange = None

# ───────────────────────────────
# Diagnostic endpoint
# ───────────────────────────────
@app.get("/debug/urls")
async def debug_urls():
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
# Status endpoint
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
# Keepalive
# ───────────────────────────────
async def keepalive():
    while True:
        logger.info(f"[KEEPALIVE] Updated flag @ {datetime.utcnow().isoformat()}")
        await asyncio.sleep(180)

@app.on_event("startup")
async def startup_event():
    logger.info("[INIT] Startup event triggered")
    asyncio.create_task(keepalive())

# ───────────────────────────────
# Root route
# ───────────────────────────────
@app.get("/")
async def root():
    return JSONResponse({
        "message": "HPB–TCT Server OK (Phase 9.4j)",
        "docs": "/docs",
        "status": "/status",
        "debug": "/debug/urls"
    })

# ───────────────────────────────
# Entrypoint (Render)
# ───────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=int(os.getenv("PORT", 10000)))
