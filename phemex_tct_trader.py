"""
phemex_tct_trader.py — Phemex TCT 6-Gate Simulated Trading Engine

Standalone background trader. No dependency on tensor_tct_trader or
schematics_5b_trader. Uses:
  - phemex_feed.py   : TTL-cached OHLCV candles from Phemex
  - phemex_tct_algo  : run_pipeline() — 6-gate signal generation
  - tct_pdf_rules    : load_tct_rules() — rule set loaded once at startup
  - github_storage   : push_phemex_log() / fetch_phemex_log()
  - telegram_notifications : send_message()

Trade lifecycle (paper trading, no real orders):
  NO_TRADE
    → pipeline emits LONG or SHORT
    → open simulated position (entry, stop, target, risk $)
    → each scan: check stop/target vs latest LTF close
    → close position (WIN or LOSS)
    → log + GitHub sync + Telegram notify

State is persisted to phemex_trade_log.json locally and on the GitHub
`data` branch after every trade close and on graceful shutdown.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from typing import Optional

from phemex_tct_algo import PipelineResult, run_pipeline
from phemex_feed import fetch_all as feed_fetch_all, LTF_TF
from tct_pdf_rules import load_tct_rules, TCTRuleSet

logger = logging.getLogger("PhemexTCTTrader")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

STARTING_BALANCE: float = float(os.getenv("PHEMEX_TCT_STARTING_BALANCE", "5000"))
RISK_PER_TRADE_PCT: float = 1.0          # 1 % of balance risked per trade
SCAN_INTERVAL: int = int(os.getenv("PHEMEX_TCT_SCAN_INTERVAL", "900"))  # 15 min

_DIR = os.path.dirname(os.path.abspath(__file__))
TRADE_LOG_PATH = os.path.join(_DIR, "phemex_trade_log.json")


# ---------------------------------------------------------------------------
# PhemexTCTTrader
# ---------------------------------------------------------------------------

class PhemexTCTTrader:
    """
    Paper-trading engine for the Phemex 6-gate TCT pipeline.

    Thread-safe via _lock. All mutations go through _open_trade(),
    _close_trade(), and save_state().
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._rules: Optional[TCTRuleSet] = None

        # Mutable state — always access under _lock
        self.balance: float = STARTING_BALANCE
        self.starting_balance: float = STARTING_BALANCE
        self.current_trade: Optional[dict] = None
        self.trade_history: list[dict] = []
        self.last_scan_time: Optional[float] = None
        self.last_signal: str = "NO_TRADE"
        self.last_pipeline_result: Optional[dict] = None
        self.consecutive_errors: int = 0

        self._load_state()

    # ------------------------------------------------------------------
    # Rule loading
    # ------------------------------------------------------------------

    def _ensure_rules(self) -> TCTRuleSet:
        """Load TCT rules once; re-use on every subsequent call."""
        if self._rules is None:
            try:
                self._rules = load_tct_rules()
                logger.info("[PHEMEX-TCT] TCT rules loaded")
            except Exception as exc:
                logger.warning("[PHEMEX-TCT] Could not load TCT rules: %s — using empty ruleset", exc)
                from tct_pdf_rules import TCTRuleSet
                self._rules = TCTRuleSet(rules={})
        return self._rules

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load_state(self) -> None:
        """Restore state from phemex_trade_log.json on disk."""
        try:
            from github_storage import fetch_phemex_log
            fetch_phemex_log(TRADE_LOG_PATH)
        except Exception as exc:
            logger.warning("[PHEMEX-TCT] GitHub fetch skipped: %s", exc)

        if not os.path.exists(TRADE_LOG_PATH):
            logger.info("[PHEMEX-TCT] No trade log found — starting fresh")
            return

        try:
            with open(TRADE_LOG_PATH, "r") as f:
                data = json.load(f)
            self.balance = data.get("balance", STARTING_BALANCE)
            self.starting_balance = data.get("starting_balance", STARTING_BALANCE)
            self.current_trade = data.get("current_trade")
            self.trade_history = data.get("trade_history", [])
            logger.info(
                "[PHEMEX-TCT] State restored — balance=%.2f trades=%d",
                self.balance, len(self.trade_history),
            )
        except Exception as exc:
            logger.error("[PHEMEX-TCT] State load failed: %s", exc)

    def save_state(self) -> None:
        """Write current state to disk (call under _lock)."""
        data = {
            "balance": self.balance,
            "starting_balance": self.starting_balance,
            "current_trade": self.current_trade,
            "trade_history": self.trade_history,
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }
        try:
            with open(TRADE_LOG_PATH, "w") as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as exc:
            logger.error("[PHEMEX-TCT] State save failed: %s", exc)

    def _push_to_github(self) -> None:
        """Push trade log to GitHub in a background thread."""
        def _push():
            try:
                from github_storage import push_phemex_log
                push_phemex_log(TRADE_LOG_PATH)
            except Exception as exc:
                logger.warning("[PHEMEX-TCT] GitHub push failed: %s", exc)
        threading.Thread(target=_push, daemon=True).start()

    # ------------------------------------------------------------------
    # Telegram
    # ------------------------------------------------------------------

    def _notify(self, text: str) -> None:
        """Fire-and-forget Telegram notification."""
        try:
            from telegram_notifications import send_message
            send_message(text)
        except Exception as exc:
            logger.warning("[PHEMEX-TCT] Telegram notify failed: %s", exc)

    # ------------------------------------------------------------------
    # Trade lifecycle
    # ------------------------------------------------------------------

    def _open_trade(self, result: PipelineResult) -> None:
        """
        Open a simulated position from a pipeline result.
        Call under _lock.
        """
        risk_amount = self.balance * (RISK_PER_TRADE_PCT / 100.0)

        trade = {
            "id": f"PHEMEX-{int(time.time())}",
            "signal": result.signal,
            "entry": result.entry,
            "stop": result.stop,
            "target": result.target,
            "rr": result.rr,
            "confidence": result.confidence,
            "risk_amount": round(risk_amount, 2),
            "opened_at": datetime.now(timezone.utc).isoformat(),
            "status": "OPEN",
        }
        self.current_trade = trade
        self.save_state()

        logger.info(
            "[PHEMEX-TCT] Trade OPENED — %s entry=%.4f stop=%.4f target=%.4f rr=%.2f conf=%.2f",
            result.signal, result.entry, result.stop, result.target,
            result.rr, result.confidence,
        )

        direction = "LONG 🟢" if result.signal == "LONG" else "SHORT 🔴"
        self._notify(
            f"<b>📈 Phemex TCT — Trade Opened</b>\n"
            f"Direction: <b>{direction}</b>\n"
            f"Entry: <code>{result.entry:.4f}</code>\n"
            f"Stop: <code>{result.stop:.4f}</code>\n"
            f"Target: <code>{result.target:.4f}</code>\n"
            f"R:R: <code>{result.rr:.2f}</code>   Confidence: <code>{result.confidence:.0%}</code>\n"
            f"Risk: <code>${risk_amount:.2f}</code>"
        )

    def _close_trade(self, outcome: str, exit_price: float) -> None:
        """
        Close the current trade as WIN or LOSS.
        Updates balance and history. Call under _lock.
        """
        if self.current_trade is None:
            return

        trade = self.current_trade
        risk = trade["risk_amount"]
        rr = trade["rr"]

        if outcome == "WIN":
            pnl = round(risk * rr, 2)
        else:
            pnl = round(-risk, 2)

        self.balance = round(self.balance + pnl, 2)

        closed = {
            **trade,
            "status": "CLOSED",
            "outcome": outcome,
            "exit_price": exit_price,
            "pnl": pnl,
            "balance_after": self.balance,
            "closed_at": datetime.now(timezone.utc).isoformat(),
        }
        self.trade_history.append(closed)
        self.current_trade = None
        self.save_state()
        self._push_to_github()

        emoji = "✅" if outcome == "WIN" else "❌"
        logger.info(
            "[PHEMEX-TCT] Trade CLOSED — %s pnl=%.2f balance=%.2f",
            outcome, pnl, self.balance,
        )
        self._notify(
            f"<b>{emoji} Phemex TCT — Trade Closed ({outcome})</b>\n"
            f"Exit: <code>{exit_price:.4f}</code>\n"
            f"P&L: <code>${pnl:+.2f}</code>\n"
            f"Balance: <code>${self.balance:.2f}</code>"
        )

    # ------------------------------------------------------------------
    # Stop / target check
    # ------------------------------------------------------------------

    def _check_position(self, current_price: float) -> None:
        """
        Evaluate whether an open position's stop or target has been reached.
        Call under _lock.
        """
        trade = self.current_trade
        if trade is None:
            return

        signal = trade["signal"]
        stop = trade["stop"]
        target = trade["target"]

        if signal == "LONG":
            if current_price <= stop:
                self._close_trade("LOSS", current_price)
            elif current_price >= target:
                self._close_trade("WIN", current_price)
        else:  # SHORT
            if current_price >= stop:
                self._close_trade("LOSS", current_price)
            elif current_price <= target:
                self._close_trade("WIN", current_price)

    # ------------------------------------------------------------------
    # Scan cycle
    # ------------------------------------------------------------------

    def scan(self) -> dict:
        """
        Run one complete scan cycle:
          1. Fetch candles (TTL-cached)
          2. If open trade: check stop/target
          3. If no trade: run pipeline and open on signal

        Returns a summary dict for the API endpoint.
        """
        rules = self._ensure_rules()

        candles = feed_fetch_all()
        htf = candles.get("4h")
        mtf = candles.get("1h")
        ltf = candles.get("15m")

        if htf is None or mtf is None or ltf is None:
            logger.warning("[PHEMEX-TCT] Candle fetch incomplete — skipping scan")
            return {"action": "skip", "reason": "candle_fetch_failed"}

        # Current price proxy: last close on LTF
        current_price = float(ltf.iloc[-1]["close"])

        with self._lock:
            self.last_scan_time = time.time()

            # Priority: manage open position before looking for new signals
            if self.current_trade is not None:
                self._check_position(current_price)
                action = "monitor"
            else:
                # Run the full 6-gate pipeline
                try:
                    result: PipelineResult = run_pipeline(htf, mtf, ltf, rules)
                except Exception as exc:
                    logger.error("[PHEMEX-TCT] Pipeline error: %s", exc, exc_info=True)
                    return {"action": "error", "reason": str(exc)}

                self.last_signal = result.signal
                self.last_pipeline_result = {
                    "signal": result.signal,
                    "confidence": result.confidence,
                    "entry": result.entry,
                    "stop": result.stop,
                    "target": result.target,
                    "rr": result.rr,
                    "blocking_gate": result.blocking_gate,
                    "gates": [
                        {"layer": g.layer, "name": g.name, "passed": g.passed}
                        for g in result.gate_results
                    ],
                }

                if result.is_trade:
                    self._open_trade(result)
                    action = "trade_opened"
                else:
                    action = "no_trade"

        return {"action": action, "price": current_price}

    # ------------------------------------------------------------------
    # Snapshot (read-only, for the API)
    # ------------------------------------------------------------------

    def snapshot(self) -> dict:
        """Return a thread-safe read-only snapshot of current state."""
        with self._lock:
            wins = sum(1 for t in self.trade_history if t.get("outcome") == "WIN")
            losses = sum(1 for t in self.trade_history if t.get("outcome") == "LOSS")
            total = wins + losses
            win_rate = round(wins / total * 100, 1) if total else 0.0
            total_pnl = round(sum(t.get("pnl", 0) for t in self.trade_history), 2)

            return {
                "balance": self.balance,
                "starting_balance": self.starting_balance,
                "pnl_total": total_pnl,
                "pnl_pct": round((self.balance - self.starting_balance) / self.starting_balance * 100, 2),
                "wins": wins,
                "losses": losses,
                "total_trades": total,
                "win_rate": win_rate,
                "current_trade": self.current_trade,
                "trade_history": list(reversed(self.trade_history)),  # newest first
                "last_scan": self.last_scan_time,
                "last_signal": self.last_signal,
                "scan_interval": SCAN_INTERVAL,
            }

    def debug(self) -> dict:
        """Return last pipeline result gate breakdown for the debug API."""
        with self._lock:
            return {
                "last_pipeline_result": self.last_pipeline_result,
                "last_signal": self.last_signal,
                "last_scan": self.last_scan_time,
            }


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_trader: Optional[PhemexTCTTrader] = None
_trader_lock = threading.Lock()


def get_trader() -> PhemexTCTTrader:
    """Return the PhemexTCTTrader singleton, creating it on first call."""
    global _trader
    if _trader is None:
        with _trader_lock:
            if _trader is None:
                _trader = PhemexTCTTrader()
    return _trader
