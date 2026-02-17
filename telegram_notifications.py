# ================================================================
# telegram_notifications.py — Proactive Telegram Trade Alerts
# ================================================================
# Sends trade setup notifications to your Telegram chat when the
# TensorTCT trader enters, closes, or force-closes a position.
# ================================================================

import os
import logging
import requests
from typing import Dict, Optional

logger = logging.getLogger("TelegramNotify")

TELEGRAM_API = "https://api.telegram.org"


def _get_credentials():
    """Read Telegram credentials from env at call time (not import time)."""
    return os.getenv("TELEGRAM_BOT_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")


def _is_configured() -> bool:
    """Check if Telegram credentials are available."""
    token, chat_id = _get_credentials()
    if not token or not chat_id:
        logger.warning("[TELEGRAM] Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID — notifications disabled")
        return False
    return True


def send_message(text: str, parse_mode: str = "HTML") -> bool:
    """Send a message to the configured Telegram chat."""
    token, chat_id = _get_credentials()
    if not token or not chat_id:
        logger.warning("[TELEGRAM] Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID — notifications disabled")
        return False
    try:
        url = f"{TELEGRAM_API}/bot{token}/sendMessage"
        payload = {"chat_id": chat_id, "text": text, "parse_mode": parse_mode}
        resp = requests.post(url, json=payload, timeout=10)
        if resp.status_code != 200:
            logger.error(f"[TELEGRAM] Send failed: HTTP {resp.status_code} — {resp.text}")
            return False
        return True
    except Exception as e:
        logger.error(f"[TELEGRAM] Send error: {e}")
        return False


# ────────────────────────────────────────────────
# Trade Notification Formatters
# ────────────────────────────────────────────────

def notify_trade_entered(trade: Dict) -> bool:
    """Send simple alert when a new trade is entered."""
    direction = trade.get("direction", "?")
    arrow = "BUY" if direction == "bullish" else "SELL"

    text = (
        f"<b>{arrow} — Trade Entered</b>\n"
        f"Entry: ${trade.get('entry_price', 0):,.2f}\n"
        f"Stop: ${trade.get('stop_price', 0):,.2f}\n"
        f"Target: ${trade.get('target_price', 0):,.2f}\n"
    )

    return send_message(text)


def notify_trade_closed(closed_trade: Dict) -> bool:
    """Send simple alert when a trade is closed (TP or SL hit)."""
    is_win = closed_trade.get("is_win", False)
    result = "WIN" if is_win else "LOSS"
    pnl_pct = closed_trade.get("pnl_pct", 0)
    pnl_dollars = closed_trade.get("pnl_dollars", 0)
    pnl_sign = "+" if pnl_dollars >= 0 else ""

    text = (
        f"<b>Trade Closed — {result}</b>\n"
        f"Entry: ${closed_trade.get('entry_price', 0):,.2f}\n"
        f"Exit: ${closed_trade.get('exit_price', 0):,.2f}\n"
        f"Stop: ${closed_trade.get('stop_price', 0):,.2f}\n"
        f"Target: ${closed_trade.get('target_price', 0):,.2f}\n"
        f"P&L: {pnl_sign}{pnl_pct:.2f}% ({pnl_sign}${pnl_dollars:,.2f})\n"
    )

    return send_message(text)


def notify_trade_force_closed(closed_trade: Dict) -> bool:
    """Send simple alert when a trade is force-closed."""
    pnl_pct = closed_trade.get("pnl_pct", 0)
    pnl_dollars = closed_trade.get("pnl_dollars", 0)
    pnl_sign = "+" if pnl_dollars >= 0 else ""

    text = (
        f"<b>Trade Force-Closed</b>\n"
        f"Entry: ${closed_trade.get('entry_price', 0):,.2f}\n"
        f"Exit: ${closed_trade.get('exit_price', 0):,.2f}\n"
        f"Stop: ${closed_trade.get('stop_price', 0):,.2f}\n"
        f"Target: ${closed_trade.get('target_price', 0):,.2f}\n"
        f"P&L: {pnl_sign}{pnl_pct:.2f}% ({pnl_sign}${pnl_dollars:,.2f})\n"
    )

    return send_message(text)
