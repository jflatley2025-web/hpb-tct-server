# ============================================================
# HPB–TCT Server | Phase 11 – Self-Learning Paper Trading
# ============================================================

import os
import json
import time
import asyncio
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from loguru import logger
from apscheduler.schedulers.asyncio import AsyncIOScheduler
import httpx
import ccxt

# ───────────────────────────────
# CONFIG & PATHS
# ───────────────────────────────
MODE = os.getenv("TRADE_MODE", "paper")
EXCHANGE = os.getenv("EXCHANGE", "mexc").lower()
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# Folders for logs and models
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)
TRADE_LOG = "data/trade_log.csv"
MODEL_PATH = "models/tct_model.pt"

# ───────────────────────────────
# SIMPLE NEURAL LEARNER
# ───────────────────────────────
class TCTModel(nn.Module):
    def __init__(self, input_size=3, hidden_size=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)

model = TCTModel()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH))
    logger.info("Loaded existing TCT model weights")

# ───────────────────────────────
# TELEGRAM UTILITIES
# ───────────────────────────────
async def telegram_send(msg: str):
    if not BOT_TOKEN or not CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": msg, "parse_mode": "HTML"}
    async with httpx.AsyncClient() as client:
        try:
            await client.post(url, data=payload)
        except Exception as e:
            logger.warning(f"Telegram send failed: {e}")

# ───────────────────────────────
# EXCHANGE CONNECTION
# ───────────────────────────────
def get_exchange():
    try:
        if EXCHANGE == "mexc":
            return ccxt.mexc({
                "apiKey": os.getenv("MEXC_KEY", ""),
                "secret": os.getenv("MEXC_SECRET", "")
            })
        elif EXCHANGE == "okx":
            return ccxt.okx({
                "apiKey": os.getenv("OKX_KEY", ""),
                "secret": os.getenv("OKX_SECRET", ""),
                "password": os.getenv("OKX_PASSPHRASE", "")
            })
        elif EXCHANGE == "binance":
            return ccxt.binance({
                "apiKey": os.getenv("BINANCE_KEY", ""),
                "secret": os.getenv("BINANCE_SECRET", "")
            })
        else:
            raise Exception("Unsupported exchange")
    except Exception as e:
        logger.error(f"Exchange init failed: {e}")
        return None

exchange = get_exchange()

# ───────────────────────────────
# CORE PAPER TRADING LOGIC
# ───────────────────────────────
async def get_price(symbol="BTC/USDT"):
    try:
        ticker = exchange.fetch_ticker(symbol)
        return float(ticker["last"])
    except Exception as e:
        logger.warning(f"Price fetch failed: {e}")
        return None

async def generate_signal(price):
    """Placeholder for real TCT signal logic"""
    if random.random() < 0.33:
        return "buy"
    elif random.random() < 0.66:
        return "sell"
    return "hold"

async def execute_trade(symbol="BTC/USDT"):
    price = await get_price(symbol)
    if not price:
        return None

    signal = await generate_signal(price)
    size = 0.001

    if signal == "hold":
        return None

    entry = {
        "utc": datetime.utcnow().isoformat(),
        "symbol": symbol,
        "side": signal,
        "size": size,
        "price": price,
        "pnl": 0.0,
        "note": "paper"
    }

    df = pd.DataFrame([entry])
    if os.path.exists(TRADE_LOG):
        df.to_csv(TRADE_LOG, mode="a", header=False, index=False)
    else:
        df.to_csv(TRADE_LOG, index=False)

    await telegram_send(f"📈 <b>Paper {signal.upper()}</b> {symbol} @ {price:.2f}")

    return entry

async def update_pnl():
    if not os.path.exists(TRADE_LOG):
        return
    df = pd.read_csv(TRADE_LOG)
    if len(df) < 2:
        return
    latest = df.tail(2)
    pnl = (latest.iloc[-1]["price"] - latest.iloc[-2]["price"]) * latest.iloc[-2]["size"]
    df.loc[df.index[-1], "pnl"] = pnl
    df.to_csv(TRADE_LOG, index=False)
    return pnl

async def train_model():
    if not os.path.exists(TRADE_LOG):
        return
    df = pd.read_csv(TRADE_LOG)
    if len(df) < 10:
        return

    x = torch.tensor(df[["price", "size", "pnl"]].values, dtype=torch.float32)
    y = torch.tensor(df["pnl"].values, dtype=torch.float32).unsqueeze(1)

    model.train()
    optimizer.zero_grad()
    out = model(x)
    loss = criterion(out, y)
    loss.backward()
    optimizer.step()

    torch.save(model.state_dict(), MODEL_PATH)
    logger.info(f"[LEARN] Model trained – loss={loss.item():.6f}")
    await telegram_send(f"🧠 Model updated | loss={loss.item():.6f}")

# ───────────────────────────────
# FASTAPI APP
# ───────────────────────────────
app = FastAPI(title="HPB–TCT Phase 11")

@app.get("/")
async def root():
    return {"phase": "11", "mode": MODE, "exchange": EXCHANGE, "status": "running"}

@app.get("/diagnostics")
async def diagnostics():
    return {"exchange": EXCHANGE, "mode": MODE, "model_loaded": os.path.exists(MODEL_PATH)}

@app.get("/status")
async def status():
    pnl = 0
    if os.path.exists(TRADE_LOG):
        df = pd.read_csv(TRADE_LOG)
        pnl = df["pnl"].sum()
    return {"mode": MODE, "exchange": EXCHANGE, "pnl": pnl, "trades": len(df) if os.path.exists(TRADE_LOG) else 0}

@app.post("/webhook/{token}")
async def telegram_webhook(token: str, req: Request):
    if token != BOT_TOKEN:
        return JSONResponse({"error": "Invalid token"}, status_code=403)
    data = await req.json()
    message = data.get("message", {}).get("text", "")
    if "/learn" in message:
        await train_model()
        return {"msg": "Training triggered"}
    elif "/status" in message:
        s = await status()
        await telegram_send(f"📊 Status: {json.dumps(s, indent=2)}")
        return {"msg": "Status sent"}
    else:
        await telegram_send("🤖 Commands:\n/status – show info\n/learn – retrain model")
    return {"ok": True}

# ───────────────────────────────
# BACKGROUND JOB
# ───────────────────────────────
scheduler = AsyncIOScheduler()

@scheduler.scheduled_job("interval", minutes=2)
async def trade_loop():
    entry = await execute_trade()
    if entry:
        pnl = await update_pnl()
        if pnl is not None:
            logger.info(f"Trade executed {entry['side']} @ {entry['price']:.2f} | PnL {pnl:.5f}")
            await train_model()

@app.on_event("startup")
async def on_startup():
    logger.info(f"[INIT] HPB–TCT Phase 11 started in {MODE.upper()} mode")
    scheduler.start()
    await telegram_send("✅ HPB–TCT Phase 11 initialized successfully.")

@app.on_event("shutdown")
async def on_shutdown():
    scheduler.shutdown()
    logger.info("[STOP] Server shutdown")

# ───────────────────────────────
# END OF FILE
# ───────────────────────────────
