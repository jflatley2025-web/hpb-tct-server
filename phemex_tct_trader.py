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
from tct_snapshot import tct_store

logger = logging.getLogger("PhemexTCTTrader")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

STARTING_BALANCE: float = float(os.getenv("PHEMEX_TCT_STARTING_BALANCE", "5000"))
RISK_PER_TRADE_PCT: float = 1.0          # 1 % of balance risked per trade
SCAN_INTERVAL: int = int(os.getenv("PHEMEX_TCT_SCAN_INTERVAL", "900"))  # 15 min

# Trade log stored on Render's persistent chroma_db directory so it
# survives restarts between deploys.  Override via PHEMEX_TRADE_LOG_DIR
# for local development.
_DEFAULT_TRADE_LOG_DIR = "/opt/render/project/chroma_db"
_TRADE_LOG_DIR = (os.getenv("PHEMEX_TRADE_LOG_DIR", _DEFAULT_TRADE_LOG_DIR) or "").strip() or _DEFAULT_TRADE_LOG_DIR
try:
    os.makedirs(_TRADE_LOG_DIR, exist_ok=True)
except OSError as exc:
    logger.error("[PHEMEX-TCT] Invalid PHEMEX_TRADE_LOG_DIR=%r: %s", _TRADE_LOG_DIR, exc)
    raise
TRADE_LOG_PATH = os.path.join(_TRADE_LOG_DIR, "phemex_trade_log.json")


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
        # Write initial state to disk immediately so the file exists on
        # the Render persistent volume even before the first trade occurs.
        self.save_state()

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
                self._rules = TCTRuleSet()
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
            self.last_scan_time = data.get("last_scan_time")
            self.last_signal = data.get("last_signal", "NO_TRADE")
            self.last_pipeline_result = data.get("last_pipeline_result")
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
            "last_scan_time": self.last_scan_time,
            "last_signal": self.last_signal,
            "last_pipeline_result": self.last_pipeline_result,
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
                # _check_position calls save_state only on close; persist
                # scan timestamp on every monitor tick as well.
                self.save_state()
            else:
                # Run the full 6-gate pipeline
                try:
                    result: PipelineResult = run_pipeline(htf, mtf, ltf, rules)
                except Exception as exc:
                    logger.error("[PHEMEX-TCT] Pipeline error: %s", exc, exc_info=True)
                    return {"action": "error", "reason": str(exc)}

                # ── RIG: Range Integrity Gate ────────────────────────
                rig_result = self._evaluate_rig(result, current_price)
                rig_blocked = rig_result.get("status") != "VALID"

                if rig_blocked:
                    logger.info("RIG BLOCK: %s", rig_result)

                self.last_signal = "NO_TRADE" if rig_blocked else result.signal
                self.last_pipeline_result = {
                    "signal": "NO_TRADE" if rig_blocked else result.signal,
                    "confidence": 0.0 if rig_blocked else result.confidence,
                    "entry": result.entry,
                    "stop": result.stop,
                    "target": result.target,
                    "rr": result.rr,
                    "blocking_gate": "RIG" if rig_blocked else result.blocking_gate,
                    "gates": [
                        {"layer": g.layer, "name": g.name, "passed": g.passed}
                        for g in result.gate_results
                    ],
                    "rig": rig_result,
                }

                if result.is_trade and not rig_blocked:
                    self._open_trade(result)
                    action = "trade_opened"
                else:
                    action = "no_trade"
                    # _open_trade already calls save_state; persist here
                    # for the no-trade path so last_scan_time is on disk.
                    self.save_state()

                # ── TCT Snapshot: capture EXACT gate values from this scan ──
                self._record_snapshot(result, current_price, rig_result)

        return {"action": action, "price": current_price}

    # ------------------------------------------------------------------
    # RIG — Range Integrity Gate
    # ------------------------------------------------------------------

    def _evaluate_rig(self, result: PipelineResult, current_price: float = None) -> dict:
        """
        Run the RIG validator using real pipeline gate data.
        Returns the full RIG result dict (status, reason, htf_bias, etc.).

        Computes displacement from real range data and current price.
        Fail-closed: errors return status=ERROR which is NOT treated as VALID.
        """
        try:
            from hpb_rig_validator import range_integrity_validator, compute_displacement

            # Extract gate data from pipeline result
            gates_by_layer: dict[int, dict] = {}
            for g in result.gate_results:
                gates_by_layer[g.layer] = g.data or {}

            g1_data = gates_by_layer.get(1, {})
            g2_data = gates_by_layer.get(2, {})

            htf_bias = g1_data.get("trend", "neutral")
            rcm_valid = g2_data.get("valid", False)
            range_high = g2_data.get("range_high")
            range_low = g2_data.get("range_low")
            range_duration = g2_data.get("range_duration_hours", 0)

            # Compute displacement from real range data
            displacement = compute_displacement(current_price, range_high, range_low) if rcm_valid else None

            # Do not evaluate RIG if RCM is invalid or displacement is missing
            if not rcm_valid or displacement is None:
                return {
                    "status": "NOT_EVALUATED",
                    "Gate": "RIG",
                    "reason": f"Missing: rcm_valid={rcm_valid} displacement={displacement is not None}",
                    "confidence": 0.0,
                    "displacement": displacement,
                    "htf_bias": htf_bias,
                    "session_bias": None,
                    "timestamp": None,
                }

            # Build RIG context from real execution data
            rig_context = {
                "gates": {
                    "1A": {"bias": htf_bias},
                    "RCM": {
                        "valid": rcm_valid,
                        "range_duration_hours": range_duration,
                    },
                    "MSCE": {
                        "session_bias": "neutral",  # MSCE not implemented in phemex pipeline
                        "session": "Unknown",
                    },
                    "1D": {"score": result.confidence},
                },
                "local_range_displacement": displacement,
            }

            rig_result = range_integrity_validator(rig_context)
            rig_result["displacement"] = displacement
            return rig_result
        except Exception as exc:
            logger.exception("[PHEMEX-TCT] RIG evaluation failed")
            return {
                "status": "ERROR",
                "Gate": "RIG",
                "reason": str(exc),
                "confidence": 0.0,
                "displacement": None,
                "htf_bias": None,
                "session_bias": None,
                "timestamp": None,
            }

    # ------------------------------------------------------------------
    # TCT Decision Snapshot
    # ------------------------------------------------------------------

    def _record_snapshot(self, result: PipelineResult, current_price: float, rig_result: dict | None = None) -> None:
        """
        Capture a TCT decision snapshot from the pipeline result.

        Uses EXACT values from the pipeline gate_results — never
        recomputed or approximated. Missing HPB components (1A BTC
        macro, 1B USDT.D, 1C alt alignment) are set to None
        because they are not implemented in this pipeline.

        When RIG blocks, ALL nested structures reflect the post-RIG
        state — no contradictory signals anywhere in the snapshot.
        """
        # Extract gate data by layer number
        gates_by_layer: dict[int, dict] = {}
        for g in result.gate_results:
            gates_by_layer[g.layer] = {
                "name": g.name,
                "passed": g.passed,
                "data": g.data,
            }

        g1 = gates_by_layer.get(1, {})
        g2 = gates_by_layer.get(2, {})
        g3 = gates_by_layer.get(3, {})
        g4 = gates_by_layer.get(4, {})
        g5 = gates_by_layer.get(5, {})
        g6 = gates_by_layer.get(6, {})

        g1_data = g1.get("data", {})
        g2_data = g2.get("data", {})
        g6_data = g6.get("data", {})

        # Single source of truth for RIG pass/block
        rig_passed = (rig_result.get("status") == "VALID") if rig_result else False
        rig_blocked = not rig_passed and rig_result is not None

        # 1D execution — override when RIG blocks to prevent contradictory signals
        if rig_blocked:
            gate_1d = {
                "status": "BLOCKED_BY_RIG",
                "blocking_gate": "RIG",
                "signal": "NO_TRADE",
                "supply_demand": {
                    "passed": g3.get("passed"),
                    "zone": g3.get("data", {}),
                },
                "liquidity": {
                    "passed": g4.get("passed"),
                    "tap_count": g4.get("data", {}).get("tap_count"),
                    "tap_price": g4.get("data", {}).get("tap_price"),
                },
                "schematics": {
                    "passed": g5.get("passed"),
                    "valid_count": g5.get("data", {}).get("valid_count"),
                    "best_score": g5.get("data", {}).get("best_score"),
                },
                "final": {
                    "passed": False,
                    "signal": "NO_TRADE",
                    "entry": g6_data.get("entry", result.entry),
                    "stop": g6_data.get("stop", result.stop),
                    "target": g6_data.get("target", result.target),
                    "rr": g6_data.get("rr", result.rr),
                    "confidence": 0.0,
                },
                "bos_confirmed": g1_data.get("bos_count", 0) > 0,
            }
        else:
            gate_1d = {
                "status": "ACTIVE",
                "supply_demand": {
                    "passed": g3.get("passed"),
                    "zone": g3.get("data", {}),
                },
                "liquidity": {
                    "passed": g4.get("passed"),
                    "tap_count": g4.get("data", {}).get("tap_count"),
                    "tap_price": g4.get("data", {}).get("tap_price"),
                },
                "schematics": {
                    "passed": g5.get("passed"),
                    "valid_count": g5.get("data", {}).get("valid_count"),
                    "best_score": g5.get("data", {}).get("best_score"),
                },
                "final": {
                    "passed": g6.get("passed"),
                    "signal": g6_data.get("signal", result.signal),
                    "entry": g6_data.get("entry", result.entry),
                    "stop": g6_data.get("stop", result.stop),
                    "target": g6_data.get("target", result.target),
                    "rr": g6_data.get("rr", result.rr),
                    "confidence": g6_data.get("confidence", result.confidence),
                },
                "bos_confirmed": g1_data.get("bos_count", 0) > 0,
            }

        pipeline_passed = sum(1 for g in result.gate_results if g.passed)

        snapshot = {
            "source": "phemex_tct",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "price": current_price,

            # 1A — BTC structure/bias (from Gate 1 market structure on LTF)
            # NOTE: This is the TRADING PAIR's structure, not a separate
            # BTC macro bias anchor. True 1A (BTC as macro context for
            # altcoins) is NOT IMPLEMENTED in this pipeline.
            "gate_1A_btc_structure": {
                "status": "PARTIAL",
                "note": "Uses trading pair structure, not dedicated BTC macro anchor",
                "trend": g1_data.get("trend"),
                "bos_count": g1_data.get("bos_count"),
                "rtz_valid": (g1_data.get("rtz") or {}).get("valid"),
                "eof": g1_data.get("eof"),
                "passed": g1.get("passed"),
            },

            # 1B — USDT.D inverse correlation
            "gate_1B_usdt_d": {
                "status": "NOT_IMPLEMENTED",
                "trend": None,
                "correlation": None,
                "passed": None,
            },

            # 1C — Altcoin alignment
            "gate_1C_alt_alignment": {
                "status": "NOT_IMPLEMENTED",
                "aligned": None,
                "passed": None,
            },

            # RCM — Range context (Gate 2)
            "gate_RCM_range": {
                "status": "ACTIVE" if g2 else "NOT_EVALUATED",
                "valid": g2_data.get("valid"),
                "range_high": g2_data.get("range_high"),
                "range_low": g2_data.get("range_low"),
                "range_size_pct": g2_data.get("range_size_pct"),
                "high_touches": g2_data.get("high_touches"),
                "low_touches": g2_data.get("low_touches"),
                "passed": g2.get("passed"),
            },

            # RIG — Counter-bias filter (wired to hpb_rig_validator)
            "gate_RIG": {
                "status": rig_result.get("status", "ERROR") if rig_result else "NOT_EVALUATED",
                "reason": rig_result.get("reason") if rig_result else None,
                "displacement": rig_result.get("displacement") if rig_result else None,
                "htf_bias": rig_result.get("htf_bias") if rig_result else None,
                "session_bias": rig_result.get("session_bias") if rig_result else None,
                "confidence": rig_result.get("confidence") if rig_result else None,
                "passed": rig_passed,
            },

            # MSCE — Session logic
            "gate_MSCE": {
                "status": "NOT_IMPLEMENTED",
                "note": "session_manipulation.py exists but is not called by phemex_tct_algo",
                "session": None,
                "multiplier": None,
                "passed": None,
            },

            # 1D — Execution (Gates 3-6 combined, overridden when RIG blocks)
            "gate_1D_execution": gate_1d,

            # Overall result — RIG override applied when it does not PASS
            "signal": "NO_TRADE" if rig_blocked else result.signal,
            "confidence": 0.0 if rig_blocked else result.confidence,
            "blocking_gate": "RIG" if rig_blocked else result.blocking_gate,
            "gates_passed": pipeline_passed + int(rig_passed),
            "gates_total": len(result.gate_results) + 1,  # +1 for RIG gate
        }

        try:
            tct_store.update(snapshot)
        except Exception as exc:
            logger.warning("[PHEMEX-TCT] Snapshot store update failed: %s", exc)

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
