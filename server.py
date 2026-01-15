# ============================================================
# HPB–TCT Server | Phase 11.5 – Live TCT Signal Integration (Paper)
# ============================================================

import os
import json
import asyncio
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from loguru import logger
import httpx
import ccxt

# ───────────────────────────────
# CONFIGURATION
# ───────────────────────────────
MODE = os.getenv("TRADE_MODE", "paper")
EXCHANGE = os.getenv("EXCHANGE", "mexc").lower()
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
SYMBOL = "BTC/USDT"

# Folders for logs and models
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)
TRADE_LOG = "data/trade_log.csv"
MODEL_PATH = "models/tct_model.pt"

# ───────────────────────────────
# SIMPLE TENSOR MODEL (LEARNING)
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
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH))
    logger.info("Loaded previous model weights")

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
    except Exception as e:
        logger.error(f"Exchange init failed: {e}")
        return None

exchange = get_exchange()

# ───────────────────────────────
# MARKET DATA FUNCTIONS
# ───────────────────────────────
async def get_ohlcv(symbol=SYMBOL, timeframe="15m", limit=100):
    try:
        data = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(data, columns=["time","open","high","low","close","volume"])
        df["time"] = pd.to_datetime(df["time"], unit="ms")
        return df
    except Exception as e:
        logger.warning(f"OHLCV fetch failed: {e}")
        return None

# ───────────────────────────────
# TCT SIGNAL DETECTION (LOCAL)
# ───────────────────────────────
async def detect_tct_signal():
    """
    Combines 15m and 1h structural bias for TCT-style directional signal
    """
    short_df = await get_ohlcv(SYMBOL, "15m", 50)
    long_df = await get_ohlcv(SYMBOL, "1h", 50)
    if short_df is None or long_df is None:
        return "hold"

    short_ma = short_df["close"].tail(10).mean()
    long_ma = long_df["close"].tail(10).mean()
    current = short_df["close"].iloc[-1]

    # Calculate momentum & volatility bias
    momentum = (current - short_ma) / short_ma
    volatility = (short_df["high"].max() - short_df["low"].min()) / short_ma

    # TCT-style directional bias
    if momentum > 0.002 and long_ma < current:
        signal = "buy"
    elif momentum < -0.002 and long_ma > current:
        signal = "sell"
    else:
        signal = "hold"

    logger.info(f"[TCT] Signal {signal.upper()} | Momentum={momentum:.5f} | Vol={volatility:.5f}")
    return signal

# ───────────────────────────────
# PAPER TRADING CORE
# ───────────────────────────────
async def get_price():
    try:
        ticker = exchange.fetch_ticker(SYMBOL)
        return float(ticker["last"])
    except Exception as e:
        logger.warning(f"Price fetch failed: {e}")
        return None

async def execute_trade():
    price = await get_price()
    if not price:
        return None

    signal = await detect_tct_signal()
    if signal == "hold":
        return None

    size = 0.001
    entry = {
        "utc": datetime.utcnow().isoformat(),
        "symbol": SYMBOL,
        "side": signal,
        "size": size,
        "price": price,
        "pnl": 0.0,
        "note": "TCT"
    }

    df = pd.DataFrame([entry])
    if os.path.exists(TRADE_LOG):
        df.to_csv(TRADE_LOG, mode="a", header=False, index=False)
    else:
        df.to_csv(TRADE_LOG, index=False)

    await telegram_send(f"📊 <b>TCT {signal.upper()}</b> {SYMBOL} @ {price:.2f}")
    logger.info(f"[TRADE] Paper {signal.upper()} @ {price:.2f}")

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
    x = torch.tensor(df[["price","size","pnl"]].values, dtype=torch.float32)
    y = torch.tensor(df["pnl"].values, dtype=torch.float32).unsqueeze(1)

    model.train()
    optimizer.zero_grad()
    out = model(x)
    loss = criterion(out, y)
    loss.backward()
    optimizer.step()
    torch.save(model.state_dict(), MODEL_PATH)

    logger.info(f"[LEARN] Model updated – loss={loss.item():.6f}")
    await telegram_send(f"🧠 Model trained | loss={loss.item():.6f}")

# ───────────────────────────────
# FASTAPI APP
# ───────────────────────────────
app = FastAPI(title="HPB–TCT Phase 11.5")

@app.get("/")
async def root():
    return {"phase": "11.5", "mode": MODE, "symbol": SYMBOL, "exchange": EXCHANGE}

@app.get("/diagnostics")
async def diagnostics():
    return {"exchange": EXCHANGE, "mode": MODE, "symbol": SYMBOL, "model": os.path.exists(MODEL_PATH)}

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
    msg = data.get("message", {}).get("text", "")
    if "/status" in msg:
        s = await status()
        await telegram_send(f"📊 Status:\n{json.dumps(s, indent=2)}")
        return {"msg": "Status sent"}
    elif "/learn" in msg:
        await train_model()
        return {"msg": "Model retrained"}
    else:
        await telegram_send("Commands:\n/status – show PnL\n/learn – retrain model")
    return {"ok": True}

# ───────────────────────────────
# BACKGROUND JOB
# ───────────────────────────────
scheduler = AsyncIOScheduler()

@scheduler.scheduled_job("interval", minutes=3)
async def trade_loop():
    await execute_trade()
    pnl = await update_pnl()
    if pnl is not None:
        logger.info(f"PNL Update: {pnl:.6f}")
        await train_model()

@app.on_event("startup")
async def startup_event():
    scheduler.start()
    logger.info(f"🚀 Phase 11.5 (Paper) launched for {SYMBOL}")
    await telegram_send(f"🚀 HPB–TCT Phase 11.5 started in {MODE.upper()} mode for {SYMBOL}")

@app.on_event("shutdown")
async def shutdown_event():
    scheduler.shutdown()
    logger.info("🛑 Server stopped")

# ============================================================
# END OF FILE
# ============================================================
