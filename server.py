import os, json, asyncio
from datetime import datetime, timedelta
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import httpx, ccxt, pandas as pd

# ───────────────────────────────
# CONFIG
# ───────────────────────────────
TRADE_MODE = os.getenv("TRADE_MODE", "paper").lower()
EXCHANGE = os.getenv("EXCHANGE", "mexc").lower()

MEXC_KEY = os.getenv("MEXC_KEY", "")
MEXC_SECRET = os.getenv("MEXC_SECRET", "")
OKX_KEY = os.getenv("OKX_KEY", "")
OKX_SECRET = os.getenv("OKX_SECRET", "")
OKX_PASSPHRASE = os.getenv("OKX_PASSPHRASE", "")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
RISK_LIMIT = float(os.getenv("RISK_LIMIT", "0.03"))   # per-trade %
DAILY_DRAWDOWN = float(os.getenv("DAILY_DRAWDOWN", "0.10"))  # 10% daily cap

# ───────────────────────────────
# INIT
# ───────────────────────────────
app = FastAPI(title="HPB-TCT Phase 10.5 AutoHedge Hybrid")
lock_trading = False
daily_loss = 0.0
reset_day = datetime.utcnow().date()

# ───────────────────────────────
# TELEGRAM UTILITIES
# ───────────────────────────────
async def tg_send(text: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"}
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            await c.post(url, json=payload)
    except Exception:
        pass

# ───────────────────────────────
# EXCHANGE CONNECT
# ───────────────────────────────
def connect_ex():
    try:
        if EXCHANGE == "mexc":
            ex = ccxt.mexc({"apiKey": MEXC_KEY, "secret": MEXC_SECRET})
        elif EXCHANGE == "okx":
            ex = ccxt.okx({
                "apiKey": OKX_KEY,
                "secret": OKX_SECRET,
                "password": OKX_PASSPHRASE,
            })
        else:
            raise ValueError("Unsupported exchange")
        ex.load_markets()
        return ex
    except Exception as e:
        return str(e)

exchange = connect_ex()

# ───────────────────────────────
# HELPERS
# ───────────────────────────────
def log_trade(entry: dict):
    os.makedirs("logs", exist_ok=True)
    path = "./logs/recent_trades.csv"
    pd.DataFrame([entry]).to_csv(path, mode="a", index=False, header=not os.path.exists(path))

def reset_daily_loss():
    global daily_loss, reset_day
    if datetime.utcnow().date() != reset_day:
        daily_loss = 0.0
        reset_day = datetime.utcnow().date()

# ───────────────────────────────
# ENDPOINTS
# ───────────────────────────────
@app.get("/")
async def root():
    return {"phase": "10.5", "mode": TRADE_MODE, "exchange": EXCHANGE, "locked": lock_trading}

@app.get("/diagnostics/full")
async def diag():
    return {
        "phase": "10.5 AutoHedge Hybrid",
        "mode": TRADE_MODE,
        "exchange": EXCHANGE,
        "locked": lock_trading,
        "utc": datetime.utcnow().isoformat()
    }

@app.get("/healthz")
async def healthz():
    return {"ok": True, "utc": datetime.utcnow().isoformat()}

# ───────────────────────────────
# PAPER / LIVE TRADE
# ───────────────────────────────
@app.post("/autotrade")
async def autotrade(request: Request):
    global daily_loss, lock_trading
    data = await request.json()
    symbol = data.get("symbol", "BTC/USDT")
    side = data.get("action", "buy").lower()
    size = float(data.get("size", 0.001))
    note = data.get("note", "")

    reset_daily_loss()
    if lock_trading:
        return JSONResponse({"status": "locked", "detail": "Daily drawdown limit hit"}, status_code=403)

    try:
        price = exchange.fetch_ticker(symbol)["last"]
        entry = {
            "utc": datetime.utcnow().isoformat(),
            "symbol": symbol,
            "side": side,
            "size": size,
            "price": price,
            "note": note or TRADE_MODE
        }
        log_trade(entry)

        # Simulated P&L
        pnl = price * size * (0.001 if side == "buy" else -0.001)
        daily_loss += abs(pnl)

        if daily_loss > DAILY_DRAWDOWN:
            lock_trading = True
            await tg_send(f"⚠️ Daily drawdown limit reached. Trading locked for the day.")

        await tg_send(f"📈 <b>{symbol}</b> {side.upper()} {size}\n💰 {price}\nMode: {TRADE_MODE}")
        return {"status": "ok", "trade": entry, "daily_loss": round(daily_loss, 4)}

    except Exception as e:
        await tg_send(f"⚠️ Trade error: {str(e)}")
        return JSONResponse({"error": str(e)}, status_code=500)

# ───────────────────────────────
# TELEGRAM COMMAND WEBHOOK
# ───────────────────────────────
@app.post(f"/webhook/{TELEGRAM_BOT_TOKEN}")
async def tg_webhook(request: Request):
    body = await request.json()
    message = body.get("message", {})
    text = message.get("text", "").strip().lower()
    chat_id = message.get("chat", {}).get("id")

    if not chat_id or str(chat_id) != TELEGRAM_CHAT_ID:
        return {"ignored": True}

    if text == "/status":
        msg = f"🧠 Phase 10.5 Status\nMode: {TRADE_MODE}\nLocked: {lock_trading}\nDaily Loss: {daily_loss:.3f}"
    elif text == "/balance":
        bal = exchange.fetch_balance()
        usd = bal['total'].get('USDT', 0)
        msg = f"💰 Balance: {usd} USDT"
    elif text == "/stop":
        global lock_trading
        lock_trading = True
        msg = "🛑 Trading manually stopped."
    elif text == "/mode":
        msg = f"Current mode: {TRADE_MODE}"
    else:
        msg = "Commands: /status /balance /stop /mode"

    await tg_send(msg)
    return {"ok": True}

# ───────────────────────────────
# TEST ENDPOINT
# ───────────────────────────────
@app.post("/notify/test")
async def notify_test():
    msg = f"✅ HPB-TCT Phase 10.5 Online\nExchange: {EXCHANGE}\nMode: {TRADE_MODE}"
    await tg_send(msg)
    return {"sent": True}
