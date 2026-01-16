# ================================================================
# telegram_bot.py — HPB–TCT v19.2 Telegram Bridge
# ================================================================
# Connects Telegram commands to your Render API endpoints
# ================================================================

import os
import requests
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

# ────────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────────
API_BASE = "https://hpb-tct-server.onrender.com"
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")  # set in Render or local .env

# ────────────────────────────────────────────────
# Command Handlers
# ────────────────────────────────────────────────

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 HPB–TCT AutoLearn v19.2 Bot Active\n"
        "Available commands:\n"
        "• /ping — check backend health\n"
        "• /status — system summary\n"
        "• /train <episodes> — run training cycles\n"
        "• /reset — reset AutoLearn state"
    )

async def ping(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        r = requests.get(f"{API_BASE}/bot/ping", timeout=10)
        data = r.json()
        text = (
            f"✅ Server: {data.get('server','?')}\n"
            f"Status: {data.get('bot_status','offline')}\n"
            f"Heartbeat: {data.get('heartbeat','N/A')}"
        )
    except Exception as e:
        text = f"❌ Ping failed: {e}"
    await update.message.reply_text(text)

async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        r = requests.get(f"{API_BASE}/bot/status", timeout=10)
        data = r.json()
        text = (
            f"📊 HPB–TCT Status\n"
            f"Bias: {data.get('bias','?')}\n"
            f"Confidence: {data.get('confidence','?')}\n"
            f"RIG: {data.get('RIG_status','?')}\n"
            f"Cycles: {data.get('train_cycles_completed','?')}\n"
            f"Heartbeat: {data.get('heartbeat','N/A')}"
        )
    except Exception as e:
        text = f"❌ Status fetch failed: {e}"
    await update.message.reply_text(text)

async def train(update: Update, context: ContextTypes.DEFAULT_TYPE):
    episodes = 5
    if context.args:
        try:
            episodes = int(context.args[0])
        except ValueError:
            pass
    try:
        r = requests.get(f"{API_BASE}/bot/train?episodes={episodes}", timeout=60)
        data = r.json()
        text = (
            f"🚀 Training Complete\n"
            f"Episodes: {data.get('episodes', episodes)}\n"
            f"Total Cycles: {data.get('state',{}).get('train_cycles_completed','?')}\n"
            f"Bias: {data.get('state',{}).get('last_bias','?')}\n"
            f"Confidence: {data.get('state',{}).get('last_confidence','?')}"
        )
    except Exception as e:
        text = f"❌ Training error: {e}"
    await update.message.reply_text(text)

async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        r = requests.post(f"{API_BASE}/reset_state", timeout=10)
        data = r.json()
        text = f"🧹 State reset complete.\nNew cycles: {data.get('state',{}).get('train_cycles_completed',0)}"
    except Exception as e:
        text = f"❌ Reset failed: {e}"
    await update.message.reply_text(text)

# ────────────────────────────────────────────────
# Main Entrypoint
# ────────────────────────────────────────────────

def main():
    if not BOT_TOKEN:
        print("❌ Missing TELEGRAM_BOT_TOKEN environment variable.")
        return

    print("🤖 Starting HPB–TCT v19.2 Telegram Bot...")
    app = ApplicationBuilder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("ping", ping))
    app.add_handler(CommandHandler("status", status))
    app.add_handler(CommandHandler("train", train))
    app.add_handler(CommandHandler("reset", reset))

    app.run_polling()

if __name__ == "__main__":
    main()
