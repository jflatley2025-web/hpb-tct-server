import os
import json
from datetime import datetime
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import httpx
import ccxt
import pandas as pd

# ───────────────────────────────
# CONFIG & ENVIRONMENT
# ───────────────────────────────
TRADE_MODE = os.getenv("TRADE_MODE", "paper").lower()
EXCHANGE = os.getenv("EXCHANGE", "mexc").lower()

MEXC_KEY = os.getenv("MEXC_KEY", "")
MEXC_SECRET = os.getenv("MEXC_SECRET", "")
OKX_KEY = os.getenv("OKX_KEY", "")
OKX_SECRET = os.getenv("OKX_SECRET", "")
OKX_PASSPHRASE = os.getenv("OKX_PASSPHRASE", "")
OKX_MODE = os.getenv("OKX_MODE", "live").lower()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# ───────────────────────────────
# INITIALIZE APP
# ───────────────────────────────
app = FastAPI(title="HPB-TCT Phase 10.0 AutoHedge Hybrid")

# ───────────────────────────────
# TELEGRAM UTILS
# ───────────────────────────────
async def send_telegram_message(text: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return {"status": "skipped", "reason": "Telegram not configured"}
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"}
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            await client.post(url, json=payload)
        return {"status": "sent"}
    except Exception as e:
        return {"status": "error", "detail": str(e)}

# ───────────────────────────────
# EXCHANGE INITIALIZATION
# ───────────────────────────────
def connect_exchange():
    try:
        if EXCHANGE == "mexc":
            ex = ccxt.mexc({"apiKey": MEXC_KEY, "secret": MEXC_SECRET})
        elif EXCHANGE == "okx":
            ex = ccxt.okx({
                "apiKey": OKX_KEY,
                "secret": OKX_SECRET,
                "password": OKX_PASSPHRASE,
                "options": {"defaultType": "spot" if OKX_MODE == "live" else "sandbox"}
            })
        else:
            raise ValueError("Unsupported exchange")
        ex.load_markets()
        return ex
    except Exception as e:
        return str(e)

exchange = connect_exchange()

# ───────────────────────────────
# DIAGNOSTICS & DEBUG
# ───────────────────────────────
@app.get("/")
async def root():
    return {"phase": "10.0 AutoHedge Hybrid", "status": "running", "mode": TRADE_MODE}

@app.get("/diagnostics/full")
async def diagnostics_full():
    return {
        "phase": "10.0 AutoHedge Hybrid",
        "mode": TRADE_MODE,
        "exchange": EXCHANGE,
        "connected": isinstance(exchange, ccxt.Exchange),
        "utc": datetime.utcnow().isoformat(),
    }

@app.get("/debug/logs")
async def debug_logs():
    log_path = "./logs/recent_trades.csv"
    if os.path.exists(log_path):
        df = pd.read_csv(log_path)
        return {"recent": df.tail(5).to_dict(orient="records")}
    else:
        return {"recent": []}

# ───────────────────────────────
# PAPER TRADE EXECUTION
# ───────────────────────────────
@app.post("/autotrade")
async def autotrade(request: Request):
    data = await request.json()
    symbol = data.get("symbol", "BTC/USDT")
    action = data.get("action", "buy").lower()
    size = float(data.get("size", 0.001))
    note = data.get("note", "")

    try:
        price = exchange.fetch_ticker(symbol)["last"] if isinstance(exchange, ccxt.Exchange) else 0
        entry = {
            "utc": datetime.utcnow().isoformat(),
            "symbol": symbol,
            "side": action,
            "size": size,
            "price": price,
            "note": note or ("live" if TRADE_MODE == "live" else "paper"),
        }

        os.makedirs("logs", exist_ok=True)
        log_path = "./logs/recent_trades.csv"
        header = not os.path.exists(log_path)
        pd.DataFrame([entry]).to_csv(log_path, mode="a", index=False, header=header)

        await send_telegram_message(
            f"📈 <b>{symbol}</b> | <b>{action.upper()}</b> {size}\n💰 Price: {price}\n🧠 Mode: {TRADE_MODE}"
        )

        return JSONResponse({"status": "ok", "trade": entry})
    except Exception as e:
        await send_telegram_message(f"⚠️ Trade error: {str(e)}")
        return JSONResponse({"status": "error", "detail": str(e)}, status_code=500)

# ───────────────────────────────
# TELEGRAM WEBHOOK + TEST ENDPOINT
# ───────────────────────────────
@app.post("/notify/test")
async def notify_test():
    msg = "✅ HPB-TCT Phase 10.0 is Online\nExchange: " + EXCHANGE + f"\nMode: {TRADE_MODE}"
    result = await send_telegram_message(msg)
    return {"telegram": result, "message": msg}

# ───────────────────────────────
# OPTIONAL HEALTH CHECK
# ───────────────────────────────
@app.get("/healthz")
async def healthz():
    return {"ok": True, "utc": datetime.utcnow().isoformat()}
