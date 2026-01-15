# ============================================================
# HPB–TCT Server | Phase 12–14 (Visualizer + Adaptive Execution + Reinforcement Learning)
# ============================================================

import os, io, json, asyncio, random
import pandas as pd, numpy as np, torch
import torch.nn as nn, torch.optim as optim
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from loguru import logger
import httpx, ccxt

# ───────────────────────────────
# CONFIG
# ───────────────────────────────
MODE = os.getenv("TRADE_MODE", "paper")
EXCHANGE = os.getenv("EXCHANGE", "mexc").lower()
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
SYMBOL = "BTC/USDT"
DAILY_LOSS_LIMIT = -15.0
RISK_SUSPEND_MINUTES = 30

os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)
TRADE_LOG = "data/trade_log.csv"
MODEL_PATH = "models/tct_model.pt"

# ───────────────────────────────
# MODEL
# ───────────────────────────────
class TCTModel(nn.Module):
    def __init__(self, input_size=3, hidden_size=32):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Tanh()
        )
    def forward(self, x):
        return self.fc(x)

model = TCTModel()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH))
    logger.info("Model weights loaded")

# ───────────────────────────────
# TELEGRAM
# ───────────────────────────────
async def telegram_send(msg: str):
    if not BOT_TOKEN or not CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    async with httpx.AsyncClient() as c:
        try:
            await c.post(url, data={"chat_id": CHAT_ID, "text": msg, "parse_mode": "HTML"})
        except Exception:
            pass

# ───────────────────────────────
# EXCHANGE
# ───────────────────────────────
def get_exchange():
    if EXCHANGE == "mexc":
        return ccxt.mexc()
    elif EXCHANGE == "okx":
        return ccxt.okx()
    elif EXCHANGE == "binance":
        return ccxt.binance()
exchange = get_exchange()

# ───────────────────────────────
# MARKET DATA
# ───────────────────────────────
async def get_ohlcv(symbol=SYMBOL, tf="15m", limit=100):
    try:
        data = exchange.fetch_ohlcv(symbol, timeframe=tf, limit=limit)
        df = pd.DataFrame(data, columns=["time","open","high","low","close","volume"])
        df["time"] = pd.to_datetime(df["time"], unit="ms")
        return df
    except Exception:
        return None

# ───────────────────────────────
# SIGNAL GENERATION
# ───────────────────────────────
async def detect_tct_signal():
    short = await get_ohlcv(SYMBOL, "15m", 80)
    long = await get_ohlcv(SYMBOL, "1h", 80)
    if short is None or long is None:
        return "hold"
    ma_s = short["close"].tail(10).mean()
    ma_l = long["close"].tail(10).mean()
    cur = short["close"].iloc[-1]
    momentum = (cur - ma_s) / ma_s
    if momentum > 0.002 and cur > ma_l:
        return "buy"
    elif momentum < -0.002 and cur < ma_l:
        return "sell"
    return "hold"

# ───────────────────────────────
# RISK SENTINEL
# ───────────────────────────────
class RiskSentinel:
    def __init__(self):
        self.paused_until = None
        self.loss_streak = 0
    async def check(self, pnl_today, pnl_recent):
        if pnl_today <= DAILY_LOSS_LIMIT:
            self.paused_until = datetime.utcnow() + timedelta(minutes=RISK_SUSPEND_MINUTES)
            await telegram_send(f"⚠️ Risk Sentinel: Daily PnL {pnl_today:.2f} below limit, paused for {RISK_SUSPEND_MINUTES}m.")
            return True
        if len(pnl_recent) >= 3 and all(p < 0 for p in pnl_recent[-3:]):
            self.loss_streak += 1
            if self.loss_streak >= 2:
                self.paused_until = datetime.utcnow() + timedelta(minutes=15)
                await telegram_send("⚠️ Risk Sentinel: consecutive loss streak detected, pausing 15m.")
                self.loss_streak = 0
                return True
        return False
    def is_paused(self):
        return self.paused_until and datetime.utcnow() < self.paused_until

sentinel = RiskSentinel()

# ───────────────────────────────
# PAPER TRADING CORE
# ───────────────────────────────
async def get_price():
    try:
        t = exchange.fetch_ticker(SYMBOL)
        return float(t["last"])
    except Exception:
        return None

async def daily_pnl():
    if not os.path.exists(TRADE_LOG): return 0
    df = pd.read_csv(TRADE_LOG)
    df["utc"] = pd.to_datetime(df["utc"])
    today = datetime.utcnow().date()
    return df[df["utc"].dt.date == today]["pnl"].sum()

async def execute_trade():
    if sentinel.is_paused():
        logger.info("Risk sentinel paused trading.")
        return
    signal = await detect_tct_signal()
    if signal == "hold": return
    price = await get_price()
    if not price: return
    df = pd.read_csv(TRADE_LOG) if os.path.exists(TRADE_LOG) else pd.DataFrame(columns=["utc","symbol","side","size","price","pnl"])
    note = "entry" if len(df)==0 else ("reverse" if df.iloc[-1]["side"] != signal else "scale")
    size = 0.001
    entry = {"utc": datetime.utcnow().isoformat(),"symbol":SYMBOL,"side":signal,"size":size,"price":price,"pnl":0.0,"note":note}
    df = pd.concat([df, pd.DataFrame([entry])], ignore_index=True)
    df.to_csv(TRADE_LOG, index=False)
    await telegram_send(f"📊 {signal.upper()} {SYMBOL} @ {price:.2f} ({note})")

async def update_pnl():
    if not os.path.exists(TRADE_LOG): return
    df = pd.read_csv(TRADE_LOG)
    if len(df)<2: return
    pnl = (df.iloc[-1]["price"] - df.iloc[-2]["price"]) * df.iloc[-2]["size"]
    if df.iloc[-2]["side"] == "sell": pnl *= -1
    df.loc[df.index[-1], "pnl"] = pnl
    df.to_csv(TRADE_LOG, index=False)
    return pnl

# ───────────────────────────────
# REINFORCEMENT LEARNING UPDATE
# ───────────────────────────────
async def reinforce_model():
    if not os.path.exists(TRADE_LOG): return
    df = pd.read_csv(TRADE_LOG)
    if len(df)<10: return
    rewards = df["pnl"] - abs(df["size"])*0.1
    x = torch.tensor(df[["price","size","pnl"]].values, dtype=torch.float32)
    y = torch.tensor(rewards.values, dtype=torch.float32).unsqueeze(1)
    model.train(); optimizer.zero_grad()
    out = model(x); loss = criterion(out, y)
    loss.backward(); optimizer.step()
    torch.save(model.state_dict(), MODEL_PATH)
    await telegram_send(f"🤖 Reinforced | reward-loss={loss.item():.6f}")

# ───────────────────────────────
# CHART VISUALIZER
# ───────────────────────────────
async def render_chart():
    df = await get_ohlcv(SYMBOL, "15m", 100)
    if df is None: return None
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(df["time"], df["close"], color="cyan")
    ax.fill_between(df["time"], df["low"], df["high"], color="gray", alpha=0.2)
    ax.set_title(f"{SYMBOL} | TCT Range Heatmap")
    buf = io.BytesIO(); plt.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    plt.close(fig); buf.seek(0)
    return buf

# ───────────────────────────────
# FASTAPI
# ───────────────────────────────
app = FastAPI(title="HPB–TCT Phase 12–14")

@app.get("/")
async def root(): return {"phase":"12–14","symbol":SYMBOL,"mode":MODE}

@app.get("/chart")
async def chart():
    buf = await render_chart()
    if not buf: return JSONResponse({"error":"chart unavailable"},status_code=500)
    return StreamingResponse(buf,media_type="image/png")

@app.get("/status")
async def status():
    pnl = await daily_pnl()
    trades = len(pd.read_csv(TRADE_LOG)) if os.path.exists(TRADE_LOG) else 0
    return {"pnl_today":pnl,"trades":trades,"paused":sentinel.is_paused()}

# ───────────────────────────────
# SCHEDULE LOOP
# ───────────────────────────────
scheduler = AsyncIOScheduler()

@scheduler.scheduled_job("interval", minutes=3)
async def loop():
    pnl_today = await daily_pnl()
    df = pd.read_csv(TRADE_LOG) if os.path.exists(TRADE_LOG) else pd.DataFrame()
    pnl_recent = df["pnl"].tail(5).tolist() if "pnl" in df else []
    if await sentinel.check(pnl_today, pnl_recent): return
    await execute_trade()
    pnl = await update_pnl()
    if pnl is not None:
        await reinforce_model()

@app.on_event("startup")
async def start():
    scheduler.start()
    await telegram_send("🚀 Phase 12–14 Hybrid+RL Started")

@app.on_event("shutdown")
async def stop():
    scheduler.shutdown()
    await telegram_send("🛑 Server stopped")
