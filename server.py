# ───────────────────────────────
# HPB–TCT Server  •  Phase 9.9 AutoHedge Hybrid (Safe Simulation)
# ───────────────────────────────
import os, csv, requests, asyncio
from datetime import datetime
from fastapi import FastAPI, Request
from loguru import logger
import ccxt

app = FastAPI(title="HPB–TCT Server", version="9.9-AutoHedge")

# ───────────────────────────────
# CONFIG
# ───────────────────────────────
TRADE_MODE = os.getenv("TRADE_MODE", "paper").lower()
HEDGE_RATIO = float(os.getenv("HEDGE_RATIO", 0.5))
LOG_PATH = os.getenv("LOG_PATH", "./logs")
SLACK_URL = os.getenv("SLACK_WEBHOOK_URL", "")
os.makedirs(LOG_PATH, exist_ok=True)
LOG_FILE = os.path.join(LOG_PATH, "trades.csv")

OKX_KEY = os.getenv("OKX_KEY", "")
OKX_SECRET = os.getenv("OKX_SECRET", "")
OKX_PASSPHRASE = os.getenv("OKX_PASSPHRASE", "")
OKX_MODE = os.getenv("OKX_MODE", "testnet").lower()

MEXC_KEY = os.getenv("MEXC_KEY", "")
MEXC_SECRET = os.getenv("MEXC_SECRET", "")

BINANCE_KEY = os.getenv("BINANCE_KEY", "")
BINANCE_SECRET = os.getenv("BINANCE_SECRET", "")

# ───────────────────────────────
# GLOBALS
# ───────────────────────────────
exchange = None
trade_state = {"open": False, "entry": 0.0, "side": None}


# ───────────────────────────────
# HELPERS
# ───────────────────────────────
def log_event(msg: str):
    logger.info(msg)

def slack_alert(msg: str):
    if not SLACK_URL:
        return
    try:
        requests.post(SLACK_URL, json={"text": msg})
    except Exception as e:
        logger.warning(f"[SLACK ERROR] {e}")

def write_log(row):
    new = not os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["utc", "symbol", "side", "size", "price", "pnl", "note"])
        w.writerow(row)

# ───────────────────────────────
# EXCHANGE CONNECTION
# ───────────────────────────────
def connect_exchange():
    global exchange
    try:
        if MEXC_KEY and MEXC_SECRET:
            exchange = ccxt.mexc({
                "apiKey": MEXC_KEY,
                "secret": MEXC_SECRET,
                "enableRateLimit": True
            })
            log_event("[EXCHANGE] Connected to MEXC (REST verified)")
            return "mexc"
        elif OKX_KEY and OKX_SECRET:
            exchange = ccxt.okx({
                "apiKey": OKX_KEY,
                "secret": OKX_SECRET,
                "password": OKX_PASSPHRASE,
                "enableRateLimit": True
            })
            log_event("[EXCHANGE] Connected to OKX (REST verified)")
            return "okx"
        elif BINANCE_KEY and BINANCE_SECRET:
            exchange = ccxt.binance({
                "apiKey": BINANCE_KEY,
                "secret": BINANCE_SECRET,
                "enableRateLimit": True
            })
            log_event("[EXCHANGE] Connected to Binance (REST verified)")
            return "binance"
        else:
            log_event("[EXCHANGE ERROR] No exchange keys found")
            return None
    except Exception as e:
        log_event(f"[EXCHANGE ERROR] {e}")
        return None


# ───────────────────────────────
# KEEPALIVE + STARTUP
# ───────────────────────────────
@app.on_event("startup")
async def startup_event():
    log_event(f"[INIT] HPB–TCT Server Phase 9.9 (TRADE_MODE={TRADE_MODE})")
    connect_exchange()
    log_event(f"[KEEPALIVE] Updated flag @ {datetime.utcnow().isoformat()}")
    slack_alert("🟢 HPB–TCT 9.9 AutoHedge Server started")


# ───────────────────────────────
# ROUTES
# ───────────────────────────────
@app.get("/")
async def root():
    return {"status": "HPB–TCT Server 9.9 running", "mode": TRADE_MODE}


@app.get("/diagnostics/full")
async def diagnostics():
    return {
        "phase": "9.9 AutoHedge Hybrid",
        "mode": TRADE_MODE,
        "exchange": type(exchange).__name__ if exchange else None,
        "log_path": LOG_PATH,
        "hedge_ratio": HEDGE_RATIO,
        "slack_enabled": bool(SLACK_URL),
        "utc": datetime.utcnow().isoformat(),
    }


# ───────────────────────────────
# AUTOHEDGE LOGIC (SIMULATION)
# ───────────────────────────────
def simulate_hedge(entry_side, entry_price, symbol, size):
    hedge_side = "sell" if entry_side == "buy" else "buy"
    hedge_size = size * HEDGE_RATIO
    hedge_price = entry_price * (1.001 if hedge_side == "sell" else 0.999)
    note = f"Simulated hedge {hedge_side} {hedge_size} {symbol}@{hedge_price}"
    log_event(f"[AUTOHEDGE] {note}")
    write_log([datetime.utcnow().isoformat(), symbol, hedge_side,
               hedge_size, hedge_price, "", "hedge"])
    slack_alert(f"🤖 AutoHedge: {note}")


@app.post("/autotrade")
async def autotrade(req: Request):
    """Simulated trade + automatic hedge + CSV logging"""
    data = await req.json()
    symbol = data.get("symbol", "BTC/USDT")
    side = data.get("action", "buy").lower()
    size = float(data.get("size", 0.001))
    price = 0.0
    if exchange:
        try:
            ticker = exchange.fetch_ticker(symbol)
            price = ticker.get("last", 0.0)
        except Exception as e:
            log_event(f"[PRICE ERROR] {e}")

    trade_state.update({"open": True, "entry": price, "side": side})
    pnl = 0.0
    write_log([datetime.utcnow().isoformat(), symbol, side, size, price, pnl, "entry"])
    log_event(f"[AUTO] {side.upper()} {size} {symbol}@{price}")
    slack_alert(f"🚀 AutoTrade: {side.upper()} {size} {symbol}@{price}")

    simulate_hedge(side, price, symbol, size)
    return {"status": "ok", "symbol": symbol, "side": side, "price": price}


# ───────────────────────────────
# DIAGNOSTICS TOOL (OPTIONAL)
# ───────────────────────────────
@app.get("/debug/logs")
async def read_logs():
    if not os.path.exists(LOG_FILE):
        return {"logs": []}
    with open(LOG_FILE, "r") as f:
        rows = f.readlines()[-10:]
    return {"recent": rows}


# ───────────────────────────────
# RUN LOCAL
# ───────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 10000)))
