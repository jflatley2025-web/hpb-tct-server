# ================================================================
# telegram_notifications.py — Proactive Telegram Trade Alerts
# ================================================================
# Sends trade setup notifications to your Telegram chat when the
# TensorTCT trader enters, closes, or force-closes a position.
# ================================================================

import os
import requests
import logging
import threading
from typing import Dict, Optional

logger = logging.getLogger("TelegramNotify")
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())

TELEGRAM_API = "https://api.telegram.org"

# Test notification
def send_test_notification():
    url = "https://api.telegram.org"
    response = requests.get(url)
    logger.debug(f"Response status code: {response.status_code}")
    if response.status_code == 200:
        send_message("Test Notification")
    else:
        logger.error(f"Failed to access Telegram API: {response.status_code}")




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


def _send_sync(url: str, payload: dict) -> None:
    """Blocking HTTP POST to Telegram API — runs in a background thread."""
    try:
        resp = requests.post(url, json=payload, timeout=5)
        if resp.status_code != 200:
            logger.error(f"[TELEGRAM] Send failed: HTTP {resp.status_code} — {resp.text}")
    except Exception as e:
        logger.error(f"[TELEGRAM] Send error: {e}")


def send_message(text: str, parse_mode: str = "HTML") -> bool:
    """Fire-and-forget: dispatches the Telegram message to a background thread
    so the caller (trade execution) is never blocked."""
    token, chat_id = _get_credentials()
    if not token or not chat_id:
        logger.warning("[TELEGRAM] Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID — notifications disabled")
        return False

    url = f"{TELEGRAM_API}/bot{token}/sendMessage"

    # Send to primary chat
    payload = {"chat_id": chat_id, "text": text, "parse_mode": parse_mode}
    t = threading.Thread(target=_send_sync, args=(url, payload), daemon=True)
    t.start()

    # Send to secondary chat if configured
    chat_id_2 = os.getenv("TELEGRAM_CHAT_ID_2")
    if chat_id_2:
        payload_2 = {"chat_id": chat_id_2, "text": text, "parse_mode": parse_mode}
        t2 = threading.Thread(target=_send_sync, args=(url, payload_2), daemon=True)
        t2.start()

    return True


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


def notify_half_tp_taken(trade: Dict, exit_price: float, pnl_dollars: float) -> bool:
    """Send alert when the half take profit is triggered and stop moved to break even."""
    direction = trade.get("direction", "?")
    arrow = "BUY" if direction == "bullish" else "SELL"

    text = (
        f"<b>Half TP Hit — {arrow}</b>\n"
        f"Symbol: {trade.get('symbol', 'BTCUSDT')}\n"
        f"Entry: ${trade.get('entry_price', 0):,.2f}\n"
        f"Half TP exit: ${exit_price:,.2f}\n"
        f"Half P&amp;L: +${pnl_dollars:,.2f}\n"
        f"Stop moved to break even: ${trade.get('stop_price', 0):,.2f}\n"
        f"Target: ${trade.get('target_price', 0):,.2f}\n"
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
