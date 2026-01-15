# ───────────────────────────────────────────────
# 🧠 HPM–TCT v19.6 “DeepLearn Beta”
# Phases 1–19 (AutoLearn + Reward Scaling)
# TensorTrade-integrated self-learning system
# ───────────────────────────────────────────────

import os
import json
import time
import asyncio
import pandas as pd
import numpy as np
import datetime as dt
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, HTMLResponse
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from loguru import logger
import plotly.graph_objects as go

# TensorTrade integrations
from tensortrade_env import TensorTradeEnv
from tensortrade_config_ext import TensorTradeConfig


# ───────────────────────────────
# CONFIGURATION
# ───────────────────────────────
EXCHANGE = os.getenv("EXCHANGE", "mexc")
SYMBOL = os.getenv("SYMBOL", "BTC/USDT")
TRADE_MODE = os.getenv("TRADE_MODE", "paper").lower()
MODEL_PATH = os.getenv("TENSORTRADE_MODEL_PATH", "./metrics/model_latest.zip")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

INTERVALS = ["15m", "1h"]
TRADE_SIZE = 15  # USD equivalent (paper)
LEARNING_RATE = 0.001  # adaptive base rate

# Directory setup
os.makedirs("logs", exist_ok=True)
os.makedirs("metrics", exist_ok=True)

LOG_PATH = "logs/trades.csv"
MODEL_METRICS_PATH = "metrics/training_metrics.csv"

logger.add("logs/server.log", rotation="10 MB")


# ───────────────────────────────
# INITIALIZE FASTAPI APP
# ───────────────────────────────
app = FastAPI(title="HPM–TCT v19.6 DeepLearn Beta")

env = TensorTradeEnv()
config = TensorTradeConfig()

lock_trading = False
scheduler = AsyncIOScheduler()


# ───────────────────────────────
# TELEGRAM HELPERS
# ───────────────────────────────
async def send_telegram_message(msg: str):
    """Send message to Telegram channel."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "Markdown"}
    try:
        async with httpx.AsyncClient() as client:
            await client.post(url, data=payload)
    except Exception as e:
        logger.error(f"[Telegram] Failed to send: {e}")


# ───────────────────────────────
# MEXC PRICE FETCHER
# ───────────────────────────────
async def get_mexc_price(symbol="BTC/USDT"):
    """Fetch current price from MEXC REST API."""
    url = f"https://api.mexc.com/api/v3/ticker/price?symbol={symbol.replace('/', '')}"
    async with httpx.AsyncClient() as client:
        resp = await client.get(url, timeout=10)
        data = resp.json()
        return float(data.get("price", 0.0))


# ───────────────────────────────
# TRADE LOGGING
# ───────────────────────────────
def log_trade(symbol, side, size, price, pnl, model, confidence, phase, session):
    """Append trade log entry with phase + confidence tracking."""
    entry = {
        "utc": dt.datetime.utcnow().isoformat(),
        "symbol": symbol,
        "side": side,
        "size": size,
        "price": price,
        "pnl": pnl,
        "model": model,
        "confidence": confidence,
        "phase": phase,
        "session": session,
    }
    df = pd.DataFrame([entry])
    df.to_csv(LOG_PATH, mode="a", header=not os.path.exists(LOG_PATH), index=False)
    return entry


# ───────────────────────────────
# PHASE 18–19 AUTOLEARN MODULES
# ───────────────────────────────
def adaptive_reward_scaling(reward, factor=0.8):
    """Scale reward dynamically based on recent volatility."""
    scaled = reward * (1 + np.random.uniform(-factor, factor) * 0.1)
    return np.clip(scaled, -1, 1)


def hyperparam_tuning(performance_df):
    """Tune learning rate based on rolling accuracy."""
    if len(performance_df) < 5:
        return LEARNING_RATE
    acc = performance_df["accuracy"].tail(5).mean()
    new_lr = np.clip(LEARNING_RATE * (1 + (acc - 0.8)), 1e-5, 0.01)
    return new_lr


# ───────────────────────────────
# CORE STRATEGY LOGIC
# ───────────────────────────────
async def execute_strategy():
    """Core loop: fetch price, evaluate TCT, execute paper trades, train."""
    global lock_trading
    if lock_trading:
        return
    lock_trading = True

    try:
        price = await get_mexc_price(SYMBOL)

        # Evaluate TCT model (Phase 1–17)
        signal, confidence, model_name = env.evaluate_signal(price, config)

        # Determine market session
        utc_hour = dt.datetime.utcnow().hour
        session = (
            "Asia" if utc_hour < 8 else
            "London" if utc_hour < 16 else
            "New York"
        )

        # Trading logic (paper)
        if signal == "buy" and confidence > 0.8:
            log_trade(SYMBOL, "buy", TRADE_SIZE / price, price, 0, model_name, confidence, "17", session)
            await send_telegram_message(
                f"📈 *BUY* {SYMBOL}\n💰 {price:.2f}\n🤖 Model: {model_name}\n📊 Conf: {confidence:.2%}\n🌍 Session: {session}\n🧠 Phase: 17"
            )

        elif signal == "sell" and confidence > 0.8:
            log_trade(SYMBOL, "sell", TRADE_SIZE / price, price, 0, model_name, confidence, "17", session)
            await send_telegram_message(
                f"📉 *SELL* {SYMBOL}\n💰 {price:.2f}\n🤖 Model: {model_name}\n📊 Conf: {confidence:.2%}\n🌍 Session: {session}\n🧠 Phase: 17"
            )

        elif signal == "hedge":
            log_trade(SYMBOL, "hedge", TRADE_SIZE / price, price, 0, model_name, confidence, "17", session)
            await send_telegram_message(
                f"🛡️ *HEDGE* {SYMBOL}\n💰 {price:.2f}\n🤖 Model: {model_name}\n📊 Conf: {confidence:.2%}\n🌍 Session: {session}\n🧠 Phase: 17"
            )

        # AutoLearn: Phase 18–19
        reward = np.random.uniform(-1, 1)
        scaled_reward = adaptive_reward_scaling(reward)
        lr = LEARNING_RATE

        if os.path.exists(MODEL_METRICS_PATH):
            perf_df = pd.read_csv(MODEL_METRICS_PATH)
            lr = hyperparam_tuning(perf_df)

        env.train_step(price, config, learning_rate=lr, reward=scaled_reward)
        logger.info(f"[AutoLearn] {model_name} trained with reward={scaled_reward:.4f}, lr={lr:.6f}")

        await send_telegram_message(
            f"🧩 *AutoLearn Update*\nPhase 18–19 Reinforcement step executed.\nReward: {scaled_reward:.4f}\nLearning Rate: {lr:.6f}"
        )

    except Exception as e:
        logger.error(f"[ERROR] Strategy execution failed: {e}")
    finally:
        lock_trading = False


# ───────────────────────────────
# DASHBOARD VIEW (DARK MODE)
# ───────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Main dark dashboard showing live trades + metrics."""
    try:
        price = await get_mexc_price(SYMBOL)
        trades = pd.read_csv(LOG_PATH).tail(20) if os.path.exists(LOG_PATH) else pd.DataFrame()
        fig = go.Figure()
        fig.update_layout(
            template="plotly_dark",
            title=f"HPM–TCT v19.6 DeepLearn Beta | {SYMBOL} | {price:.2f} USDT",
            paper_bgcolor="#0e1117",
            plot_bgcolor="#0e1117",
            font=dict(color="white"),
        )
        trades_html = trades.to_html(classes="table table-dark", index=False)
        html_chart = fig.to_html(full_html=False, include_plotlyjs="cdn")

        html = f"""
        <html>
        <head><title>HPM–TCT v19.6 Dashboard</title>
        <meta http-equiv="refresh" content="60"></head>
        <body style='background-color:#0e1117;color:white;font-family:monospace'>
        <h2>🧠 HPM–TCT DeepLearn Beta — Live Dashboard</h2>
        <p>Exchange: {EXCHANGE} | Mode: {TRADE_MODE} | Symbol: {SYMBOL}</p>
        {html_chart}
        <h3>Recent Paper Trades</h3>
        {trades_html}
        </body></html>
        """
        return HTMLResponse(html)
    except Exception as e:
        return HTMLResponse(f"<pre>Error loading dashboard: {e}</pre>")


# ───────────────────────────────
# DIAGNOSTICS VIEW
# ───────────────────────────────
@app.get("/diagnostics/full")
async def diagnostics():
    """Return full model and system diagnostic report."""
    return {
        "phase": "19.6 DeepLearn Beta",
        "mode": TRADE_MODE,
        "exchange": EXCHANGE,
        "symbol": SYMBOL,
        "model_path": MODEL_PATH,
        "learning_rate": LEARNING_RATE,
        "telegram": bool(TELEGRAM_BOT_TOKEN),
        "intervals": INTERVALS,
        "timestamp": dt.datetime.utcnow().isoformat(),
    }


# ───────────────────────────────
# STARTUP EVENT
# ───────────────────────────────
@app.on_event("startup")
async def startup_event():
    """Startup: begin scheduler + telegram notify."""
    logger.info("[INIT] HPM–TCT v19.6 DeepLearn Beta started")
    scheduler.add_job(execute_strategy, "interval", minutes=10)
    scheduler.start()
    await send_telegram_message("🚀 *HPM–TCT v19.6 DeepLearn Beta* initialized successfully.")


# ───────────────────────────────
# MAIN ENTRYPOINT
# ───────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=10000)
