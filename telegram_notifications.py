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

# ────────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────────
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
TELEGRAM_API = "https://api.telegram.org"


def _is_configured() -> bool:
    """Check if Telegram credentials are available."""
    if not BOT_TOKEN or not CHAT_ID:
        logger.warning("[TELEGRAM] Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID — notifications disabled")
        return False
    return True


def send_message(text: str, parse_mode: str = "HTML") -> bool:
    """Send a message to the configured Telegram chat."""
    if not _is_configured():
        return False
    try:
        url = f"{TELEGRAM_API}/bot{BOT_TOKEN}/sendMessage"
        payload = {"chat_id": CHAT_ID, "text": text, "parse_mode": parse_mode}
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
    """Send alert when a new trade setup is executed."""
    direction = trade.get("direction", "?")
    arrow = "\U0001F7E2" if direction == "bullish" else "\U0001F534"  # green/red circle
    model = trade.get("model", "?")
    tf = trade.get("timeframe", "?")

    text = (
        f"{arrow} <b>NEW TRADE ENTERED</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"<b>Symbol:</b> {trade.get('symbol', 'BTCUSDT')}\n"
        f"<b>Direction:</b> {direction.upper()}\n"
        f"<b>Model:</b> {model}\n"
        f"<b>Timeframe:</b> {tf}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"<b>Entry:</b> ${trade.get('entry_price', 0):,.2f}\n"
        f"<b>Stop Loss:</b> ${trade.get('stop_price', 0):,.2f}\n"
        f"<b>Target:</b> ${trade.get('target_price', 0):,.2f}\n"
        f"<b>R:R:</b> {trade.get('rr', 0):.1f}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"<b>Position Size:</b> ${trade.get('position_size', 0):,.2f}\n"
        f"<b>Risk:</b> ${trade.get('risk_amount', 0):,.2f} ({trade.get('leverage', 10)}x)\n"
        f"<b>Entry Score:</b> {trade.get('entry_score', 0)}/100\n"
        f"<b>Bias:</b> {trade.get('reward_bias', {}).get('bias', '?')} "
        f"({trade.get('reward_bias', {}).get('confidence', 0):.0f}% conf)\n"
    )

    reasons = trade.get("entry_reasons", [])
    if reasons:
        text += f"\n<b>Reasons:</b>\n"
        for r in reasons[:8]:
            text += f"  • {r}\n"

    return send_message(text)


def notify_trade_closed(closed_trade: Dict) -> bool:
    """Send alert when a trade is closed (TP or SL hit)."""
    is_win = closed_trade.get("is_win", False)
    icon = "\u2705" if is_win else "\u274C"  # checkmark / X
    result = "WIN" if is_win else "LOSS"
    reason = closed_trade.get("exit_reason", "?").replace("_", " ").upper()
    direction = closed_trade.get("direction", "?")
    pnl_pct = closed_trade.get("pnl_pct", 0)
    pnl_dollars = closed_trade.get("pnl_dollars", 0)
    pnl_sign = "+" if pnl_dollars >= 0 else ""

    text = (
        f"{icon} <b>TRADE CLOSED — {result}</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"<b>Symbol:</b> {closed_trade.get('symbol', 'BTCUSDT')}\n"
        f"<b>Direction:</b> {direction.upper()}\n"
        f"<b>Reason:</b> {reason}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"<b>Entry:</b> ${closed_trade.get('entry_price', 0):,.2f}\n"
        f"<b>Exit:</b> ${closed_trade.get('exit_price', 0):,.2f}\n"
        f"<b>P&L:</b> {pnl_sign}{pnl_pct:.2f}% ({pnl_sign}${pnl_dollars:,.2f})\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"<b>Balance:</b> ${closed_trade.get('balance_after', 0):,.2f}\n"
        f"<b>Total Trades:</b> W{closed_trade.get('_wins', '?')} / L{closed_trade.get('_losses', '?')}\n"
    )

    analysis = closed_trade.get("analysis", "")
    solution = closed_trade.get("solution", "")
    if analysis:
        text += f"\n<b>Analysis:</b> {analysis}\n"
    if solution:
        text += f"<b>Adaptation:</b> {solution}\n"

    return send_message(text)


def notify_trade_force_closed(closed_trade: Dict) -> bool:
    """Send alert when a trade is force-closed."""
    pnl_pct = closed_trade.get("pnl_pct", 0)
    pnl_dollars = closed_trade.get("pnl_dollars", 0)
    pnl_sign = "+" if pnl_dollars >= 0 else ""

    text = (
        f"\u26A0\uFE0F <b>TRADE FORCE-CLOSED</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"<b>Symbol:</b> {closed_trade.get('symbol', 'BTCUSDT')}\n"
        f"<b>Direction:</b> {closed_trade.get('direction', '?').upper()}\n"
        f"<b>Entry:</b> ${closed_trade.get('entry_price', 0):,.2f}\n"
        f"<b>Exit:</b> ${closed_trade.get('exit_price', 0):,.2f}\n"
        f"<b>P&L:</b> {pnl_sign}{pnl_pct:.2f}% ({pnl_sign}${pnl_dollars:,.2f})\n"
        f"<b>Balance:</b> ${closed_trade.get('balance_after', 0):,.2f}\n"
    )

    return send_message(text)
