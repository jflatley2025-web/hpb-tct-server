"""
tensor_tct_trader.py — TensorTrade TCT Simulated Trading Engine
================================================================
Uses TCT schematics (Lecture 5A) to detect trade setups on BTCUSDT,
executes simulated trades with a $5,000 starting balance, and integrates
the HPBContextualReward system for adaptive learning.

Each cycle:
1. Fetch live BTCUSDT candles from MEXC
2. Run TCT schematic detection (accumulation & distribution models)
3. Evaluate active/confirmed schematics for trade entry
4. Simulate position management (entry, SL, TP)
5. Log results with reward-based performance tracking
"""

import os
import json
import time
import asyncio
import logging
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from typing import Dict, List, Optional
from pathlib import Path

from tct_schematics import detect_tct_schematics
from tensortrade_env import HPBContextualReward, HPB_TensorTrade_Env
from trade_execution import (
    calculate_position_size,
    calculate_margin,
    calculate_liquidation_price,
    check_liquidation_safety,
)
from telegram_notifications import (
    notify_trade_entered,
    notify_trade_closed,
    notify_trade_force_closed,
)

logger = logging.getLogger("TensorTCT")

# ================================================================
# CONFIGURATION
# ================================================================
MEXC_URL_BASE = os.getenv("MEXC_URL_BASE", "https://api.mexc.com")
STARTING_BALANCE = 5000.0
RISK_PER_TRADE_PCT = 1.0  # 1% of balance per trade
DEFAULT_LEVERAGE = 10
SYMBOL = "BTCUSDT"
TIMEFRAME = "4h"  # legacy default (multi-TF scanning overrides this)
# Scan these timeframes from highest to lowest — first qualifying setup wins
SCAN_TIMEFRAMES = ["4h", "1h", "15m", "5m", "1m"]
# How often the background auto-scan loop runs (seconds)
AUTO_SCAN_INTERVAL = int(os.getenv("TENSOR_SCAN_INTERVAL", "60"))
TRADE_LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tensor_trade_log.json")
TRADE_LOG_BACKUP_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tensor_trade_log_backup.json")

# Deduplication: minimum seconds between closing one trade and entering the next
# with the same entry price/direction (prevents re-entering the same schematic)
DUPLICATE_COOLDOWN_SECONDS = 300  # 5 minutes
DUPLICATE_PRICE_TOLERANCE = 0.002  # 0.2% price tolerance for "same entry"


# ================================================================
# TRADE STATE MANAGER
# ================================================================
class TradeState:
    """Persistent trade state — serialized to JSON for the dashboard."""

    def __init__(self):
        self.balance = STARTING_BALANCE
        self.starting_balance = STARTING_BALANCE
        self.current_trade: Optional[Dict] = None
        self.trade_history: List[Dict] = []
        self.reward_history: List[float] = []
        self.total_wins = 0
        self.total_losses = 0
        self.solutions_applied: List[str] = []
        self.last_scan_time: Optional[str] = None
        self.last_scan_action: Optional[str] = None
        self.last_error: Optional[str] = None
        self._load()

    def _log_path(self):
        return TRADE_LOG_PATH

    def _backup_path(self):
        return TRADE_LOG_BACKUP_PATH

    def _load(self):
        """Load state from disk if available."""
        try:
            path = self._log_path()
            if os.path.exists(path):
                with open(path, "r") as f:
                    data = json.load(f)
                self.balance = data.get("balance", STARTING_BALANCE)
                self.starting_balance = data.get("starting_balance", STARTING_BALANCE)
                self.current_trade = data.get("current_trade")
                self.trade_history = data.get("trade_history", [])
                self.reward_history = data.get("reward_history", [])
                self.total_wins = data.get("total_wins", 0)
                self.total_losses = data.get("total_losses", 0)
                self.solutions_applied = data.get("solutions_applied", [])
                self.last_scan_time = data.get("last_scan_time")
                self.last_scan_action = data.get("last_scan_action")
                logger.info(f"[STATE] Loaded trade state — balance=${self.balance:.2f}, trades={len(self.trade_history)}")
                # One-time deduplication of trade history
                deduped = self._deduplicate_history()
                if deduped:
                    self.save()
        except Exception as e:
            logger.warning(f"[STATE] Could not load trade state: {e}")

    def _deduplicate_history(self) -> bool:
        """
        Remove duplicate trades from history. Two trades are considered
        duplicates if they share the same entry_price and direction.
        Keeps the first occurrence of each unique (entry_price, direction) pair.
        Returns True if any duplicates were removed.
        """
        if len(self.trade_history) <= 1:
            return False

        seen = set()
        unique = []
        removed = 0
        for t in self.trade_history:
            key = (round(t.get("entry_price", 0), 2), t.get("direction", ""))
            if key in seen:
                removed += 1
                continue
            seen.add(key)
            unique.append(t)

        if removed == 0:
            return False

        logger.info(f"[STATE] Deduplicated trade history: removed {removed} duplicates, kept {len(unique)}")
        self.trade_history = unique
        # Re-number trade IDs
        for i, t in enumerate(self.trade_history):
            t["id"] = i + 1
        # Recalculate stats from clean history
        self.total_wins = sum(1 for t in self.trade_history if t.get("is_win"))
        self.total_losses = sum(1 for t in self.trade_history if not t.get("is_win") and t.get("status") == "closed")
        self.reward_history = [t.get("reward_value", 0) for t in self.trade_history if "reward_value" in t]
        # Set balance to the last trade's balance_after, or starting balance
        if self.trade_history:
            self.balance = self.trade_history[-1].get("balance_after", self.balance)
        self.solutions_applied = [t.get("solution", "") for t in self.trade_history if t.get("solution")]
        return True

    def _backup(self):
        """Create a backup copy of the trade log before writing."""
        import shutil
        try:
            src = self._log_path()
            if os.path.exists(src):
                shutil.copy2(src, self._backup_path())
        except Exception as e:
            logger.warning(f"[STATE] Backup failed: {e}")

    def restore_from_backup(self) -> bool:
        """Restore trade state from the backup file."""
        try:
            bak = self._backup_path()
            if not os.path.exists(bak):
                logger.warning("[STATE] No backup file found")
                return False
            import shutil
            shutil.copy2(bak, self._log_path())
            self._load()
            logger.info("[STATE] Restored from backup")
            return True
        except Exception as e:
            logger.error(f"[STATE] Restore failed: {e}")
            return False

    def save(self):
        """Persist state to disk (with automatic backup)."""
        try:
            self._backup()
            data = {
                "balance": round(self.balance, 2),
                "starting_balance": self.starting_balance,
                "current_trade": self.current_trade,
                "trade_history": self.trade_history,
                "reward_history": self.reward_history,
                "total_wins": self.total_wins,
                "total_losses": self.total_losses,
                "solutions_applied": self.solutions_applied,
                "last_scan_time": self.last_scan_time,
                "last_scan_action": self.last_scan_action,
                "last_error": self.last_error,
            }
            with open(self._log_path(), "w") as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"[STATE] Failed to save: {e}")

    def snapshot(self) -> Dict:
        """Return JSON-safe state snapshot for the dashboard."""
        win_rate = (self.total_wins / max(len(self.trade_history), 1)) * 100
        pnl_total = self.balance - self.starting_balance
        avg_reward = float(np.mean(self.reward_history)) if self.reward_history else 0.0
        return {
            "balance": round(self.balance, 2),
            "starting_balance": self.starting_balance,
            "pnl_total": round(pnl_total, 2),
            "pnl_pct": round((pnl_total / self.starting_balance) * 100, 2),
            "total_trades": len(self.trade_history),
            "wins": self.total_wins,
            "losses": self.total_losses,
            "win_rate": round(win_rate, 2),
            "avg_reward": round(avg_reward, 6),
            "current_trade": self.current_trade,
            "trade_history": self.trade_history[-50:],  # last 50 trades
            "solutions_applied": self.solutions_applied[-20:],
            "last_scan_time": self.last_scan_time,
            "last_scan_action": self.last_scan_action,
            "last_error": self.last_error,
        }


# ================================================================
# MEXC DATA FETCH (sync for use in background thread)
# ================================================================
def fetch_candles_sync(symbol: str = SYMBOL, interval: str = TIMEFRAME, limit: int = 200, _retries: int = 3) -> Optional[pd.DataFrame]:
    """Fetch OHLCV candles from MEXC (synchronous) with retry logic."""
    # Normalize intervals to MEXC-supported values (MEXC spot uses 60m not 1h)
    _MEXC_INTERVAL_MAP = {"1h": "60m", "2h": "4h"}
    mexc_interval = _MEXC_INTERVAL_MAP.get(interval, interval)
    url = f"{MEXC_URL_BASE}/api/v3/klines"
    params = {"symbol": symbol, "interval": mexc_interval, "limit": limit}

    last_error = None
    for attempt in range(_retries):
        try:
            res = requests.get(url, params=params, timeout=20)
            if res.status_code != 200:
                body = res.text[:300] if res.text else "(empty)"
                logger.error(f"[FETCH] {symbol}/{interval} HTTP {res.status_code} — {body}")
                last_error = f"HTTP {res.status_code}"
                time.sleep(1.5 * (attempt + 1))
                continue
            data = res.json()
            if not isinstance(data, list) or len(data) == 0:
                logger.warning(f"[FETCH] {symbol}/{interval} returned empty data (attempt {attempt + 1}/{_retries})")
                last_error = "empty response"
                time.sleep(1.5 * (attempt + 1))
                continue
            # MEXC spot /api/v3/klines returns variable column counts (8 or 12).
            # Build column names dynamically to match what the API actually returns.
            _ALL_COLS = [
                "open_time", "open", "high", "low", "close", "volume",
                "close_time", "quote_vol", "trades", "taker_base",
                "taker_quote", "ignore"
            ]
            ncols = len(data[0]) if data else 0
            col_names = _ALL_COLS[:ncols] if ncols <= len(_ALL_COLS) else _ALL_COLS
            df = pd.DataFrame(data, columns=col_names)
            df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
            df = df.astype({"open": float, "high": float, "low": float, "close": float, "volume": float})
            df = df[["open_time", "open", "high", "low", "close", "volume"]].sort_values("open_time")
            return df
        except requests.exceptions.Timeout:
            logger.warning(f"[FETCH] {symbol}/{interval} timeout (attempt {attempt + 1}/{_retries})")
            last_error = "timeout"
            time.sleep(1.5 * (attempt + 1))
        except Exception as e:
            logger.error(f"[FETCH] {symbol}/{interval}: {e}", exc_info=True)
            last_error = str(e)
            time.sleep(1.5 * (attempt + 1))

    logger.error(f"[FETCH] {symbol}/{interval} failed after {_retries} attempts — last error: {last_error}")
    return None


def fetch_live_price(symbol: str = SYMBOL) -> Optional[float]:
    """Get current price from MEXC ticker."""
    try:
        url = f"{MEXC_URL_BASE}/api/v3/ticker/price"
        res = requests.get(url, params={"symbol": symbol}, timeout=10)
        if res.status_code != 200:
            logger.error(f"[PRICE] {symbol} HTTP {res.status_code}")
            return None
        price = float(res.json().get("price", 0))
        if price <= 0:
            logger.warning(f"[PRICE] {symbol} returned invalid price: {price}")
            return None
        return price
    except Exception as e:
        logger.error(f"[PRICE] {symbol} fetch failed: {e}")
        return None


# ================================================================
# REWARD-ENHANCED TRADE EVALUATOR
# ================================================================
class TCTTradeEvaluator:
    """
    Evaluates TCT schematics and decides whether to enter trades.
    Integrates HPBContextualReward for adaptive bias.
    """

    def __init__(self):
        self.reward_scheme = HPBContextualReward()
        self.env = HPB_TensorTrade_Env(symbol=SYMBOL, interval=TIMEFRAME, window=100)
        self.consecutive_losses = 0
        self.adaptation_notes: List[str] = []

    def compute_reward_bias(self, prices: np.ndarray) -> Dict:
        """Run reward scheme over recent prices to determine market bias."""
        self.reward_scheme = HPBContextualReward()  # reset
        rewards = []
        for p in prices:
            r = self.reward_scheme.compute(float(p))
            rewards.append(r)
        avg = float(np.mean(rewards)) if rewards else 0.0
        bias = "bullish" if avg > 0 else "bearish" if avg < 0 else "neutral"
        confidence = min(abs(avg) * 1000, 100.0)  # scale to 0-100
        return {"bias": bias, "confidence": round(confidence, 2), "avg_reward": avg}

    def evaluate_schematic(self, schematic: Dict, reward_bias: Dict, current_price: float) -> Dict:
        """
        Score a schematic for trade entry quality.
        Returns a dict with score (0-100), direction, and reasoning.
        """
        score = 0
        reasons = []

        direction = schematic.get("direction", "unknown")
        model = schematic.get("model_type", "unknown")
        is_confirmed = schematic.get("is_confirmed", False)
        rr = schematic.get("risk_reward", 0) or 0

        # Must be confirmed (BOS happened)
        if not is_confirmed:
            return {"score": 0, "direction": direction, "model": model, "rr": rr, "required_score": 50, "pass": False, "reasons": ["No BOS confirmation"]}

        # BOS confirmation = base score
        score += 30
        reasons.append("BOS confirmed")

        # R:R quality
        if rr >= 3.0:
            score += 25
            reasons.append(f"Excellent R:R ({rr:.1f})")
        elif rr >= 2.0:
            score += 15
            reasons.append(f"Good R:R ({rr:.1f})")
        elif rr >= 1.5:
            score += 5
            reasons.append(f"Acceptable R:R ({rr:.1f})")
        else:
            return {"score": 0, "direction": direction, "model": model, "rr": rr, "required_score": 50, "pass": False, "reasons": [f"R:R too low ({rr:.1f})"]}

        # Reward bias alignment
        if direction == "bullish" and reward_bias["bias"] == "bullish":
            score += 20
            reasons.append("Reward bias aligned (bullish)")
        elif direction == "bearish" and reward_bias["bias"] == "bearish":
            score += 20
            reasons.append("Reward bias aligned (bearish)")
        elif reward_bias["bias"] == "neutral":
            score += 5
            reasons.append("Neutral reward bias — proceed with caution")
        else:
            score -= 10
            reasons.append(f"Reward bias conflicting ({reward_bias['bias']} vs {direction})")

        # Confidence boost
        if reward_bias["confidence"] > 60:
            score += 10
            reasons.append(f"High reward confidence ({reward_bias['confidence']}%)")
        elif reward_bias["confidence"] > 30:
            score += 5
            reasons.append(f"Moderate reward confidence ({reward_bias['confidence']}%)")

        # Model type preference
        if "Model_1" in model:
            score += 5
            reasons.append("Model 1 — deeper deviation (stronger)")
        elif "Model_2" in model:
            score += 3
            reasons.append("Model 2 — higher low / lower high")

        # Consecutive loss adaptation — tighten entry requirements
        if self.consecutive_losses >= 3:
            required = 70
            reasons.append(f"Tightened entry (3+ losses) — need score >= {required}")
        elif self.consecutive_losses >= 2:
            required = 60
            reasons.append(f"Cautious mode (2 losses) — need score >= {required}")
        else:
            required = 50

        # Check price relative to entry
        entry_info = schematic.get("entry", {})
        entry_price = entry_info.get("price")
        if entry_price:
            if direction == "bullish" and current_price > entry_price * 1.02:
                score -= 15
                reasons.append("Price already moved 2%+ above entry — late")
            elif direction == "bearish" and current_price < entry_price * 0.98:
                score -= 15
                reasons.append("Price already moved 2%+ below entry — late")

        score = max(0, min(100, score))
        return {
            "score": score,
            "direction": direction,
            "model": model,
            "rr": rr,
            "required_score": required,
            "pass": score >= required,
            "reasons": reasons,
        }

    def adapt_after_loss(self, trade: Dict) -> str:
        """Analyze a losing trade and generate an adaptation solution."""
        self.consecutive_losses += 1
        direction = trade.get("direction", "unknown")
        rr = trade.get("rr", 0)
        entry_score = trade.get("entry_score", 0)

        solutions = []
        if self.consecutive_losses >= 3:
            solutions.append("Switched to high-confidence-only mode (score >= 70) after 3 consecutive losses")
        if rr < 2.5:
            solutions.append(f"Raising minimum R:R from {rr:.1f} — will require R:R >= 2.5 for next trade")
        if entry_score < 60:
            solutions.append(f"Previous entry score was only {entry_score} — tightening minimum entry score")
        if not solutions:
            solutions.append("No specific adaptation needed — loss within normal parameters")

        note = " | ".join(solutions)
        self.adaptation_notes.append(note)
        return note

    def adapt_after_win(self):
        """Reset consecutive loss counter on win."""
        self.consecutive_losses = max(0, self.consecutive_losses - 1)


# ================================================================
# MAIN TRADING ENGINE
# ================================================================
class TensorTCTTrader:
    """
    Simulated TCT trading engine using TensorTrade reward system.
    Manages the full lifecycle: scan → evaluate → enter → manage → exit.
    """

    def __init__(self):
        self.state = TradeState()
        self.evaluator = TCTTradeEvaluator()
        self.last_debug: Dict = {}  # detailed debug from last scan cycle

    def scan_and_trade(self) -> Dict:
        """
        Main cycle: fetch data across multiple timeframes, detect schematics,
        evaluate, and trade.  Scans SCAN_TIMEFRAMES from highest to lowest —
        the first qualifying setup wins.
        Returns a summary of what happened this cycle.
        """
        cycle_result = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": "none",
            "details": {},
        }

        try:
            # 1. Get current price from the fastest feed (1m)
            df_price = fetch_candles_sync(SYMBOL, "1m", 10)
            if df_price is None or len(df_price) == 0:
                self.state.last_error = "Could not fetch price data"
                self.state.save()
                cycle_result["action"] = "error"
                cycle_result["details"] = {"error": "Could not fetch price data"}
                return cycle_result

            current_price = float(df_price.iloc[-1]["close"])

            # 2. If we have an open trade — manage it (check SL/TP) and return
            if self.state.current_trade:
                result = self._manage_open_trade(current_price)
                cycle_result["action"] = result.get("action", "manage")
                cycle_result["details"] = result
                self.state.last_scan_time = cycle_result["timestamp"]
                self.state.last_scan_action = cycle_result["action"]
                self.state.save()
                return cycle_result

            # 3. Scan across all timeframes for the best qualifying setup
            best_setup = None
            best_score = 0
            best_tf = None
            best_reward_bias = None
            all_tf_results = {}

            for tf in SCAN_TIMEFRAMES:
                try:
                    df = fetch_candles_sync(SYMBOL, tf, 200)
                    if df is None or len(df) < 50:
                        all_tf_results[tf] = {
                            "status": "insufficient_data",
                            "candles": 0 if df is None else len(df),
                            "fetch_error": df is None,
                        }
                        continue

                    prices = df["close"].values
                    reward_bias = self.evaluator.compute_reward_bias(prices)

                    schematics_result = detect_tct_schematics(df, [])
                    all_schematics = (
                        schematics_result.get("accumulation_schematics", [])
                        + schematics_result.get("distribution_schematics", [])
                    )

                    tf_evals = []
                    for s in all_schematics:
                        if not isinstance(s, dict):
                            continue
                        eval_result = self.evaluator.evaluate_schematic(s, reward_bias, current_price)
                        tf_evals.append(eval_result)
                        if eval_result["pass"] and eval_result["score"] > best_score:
                            best_score = eval_result["score"]
                            best_setup = (s, eval_result)
                            best_tf = tf
                            best_reward_bias = reward_bias

                    all_tf_results[tf] = {
                        "status": "scanned",
                        "candles": len(df),
                        "schematics_found": len(all_schematics),
                        "confirmed": sum(1 for s in all_schematics if isinstance(s, dict) and s.get("is_confirmed")),
                        "best_score": max((e["score"] for e in tf_evals), default=0),
                        "reward_bias": reward_bias["bias"],
                        "reward_confidence": reward_bias["confidence"],
                        "evaluations": tf_evals[:10],  # top 10 for debug
                        "detection_error": schematics_result.get("error"),
                    }

                except Exception as tf_err:
                    logger.warning(f"[TRADE] Error scanning {tf}: {tf_err}", exc_info=True)
                    all_tf_results[tf] = {"status": "error", "error": str(tf_err), "fetch_error": True}

            # Detect total fetch failure — all timeframes returned errors or no data
            failed_tfs = [tf for tf, r in all_tf_results.items() if r.get("status") in ("insufficient_data", "error") and r.get("fetch_error", False)]
            if len(failed_tfs) == len(SCAN_TIMEFRAMES):
                logger.error(f"[TRADE] ALL {len(SCAN_TIMEFRAMES)} timeframes failed to fetch — possible MEXC API outage or network issue")
                self.state.last_error = f"All timeframes failed: {', '.join(failed_tfs)}"

            # Store debug info from this scan cycle
            self.last_debug = {
                "timestamp": cycle_result["timestamp"],
                "current_price": current_price,
                "timeframes": all_tf_results,
                "best_tf": best_tf,
                "best_score": best_score,
                "consecutive_losses": self.evaluator.consecutive_losses,
            }

            # 4. Enter trade if we found a qualifying setup on any timeframe
            if best_setup:
                schematic, evaluation = best_setup
                entry_info = schematic.get("entry", {})
                candidate_price = entry_info.get("price", current_price)

                # Deduplication: skip if this is the same setup we just traded
                if self._is_duplicate_setup(candidate_price, evaluation["direction"]):
                    cycle_result["action"] = "duplicate_setup_skipped"
                    cycle_result["details"] = {
                        "price": current_price,
                        "skipped_entry": candidate_price,
                        "skipped_direction": evaluation["direction"],
                        "reason": "Same entry price/direction as recent trade — cooldown active",
                    }
                    logger.info(f"[TRADE] Duplicate setup skipped: {evaluation['direction']} @ {candidate_price}")
                else:
                    trade = self._enter_trade(schematic, evaluation, current_price, best_reward_bias)
                    trade["timeframe"] = best_tf
                    cycle_result["action"] = "trade_entered"
                    cycle_result["details"] = trade
                    logger.info(f"[TRADE] Setup found on {best_tf} — entering trade")
            else:
                cycle_result["action"] = "no_qualifying_setups"
                cycle_result["details"] = {
                    "price": current_price,
                    "timeframes_scanned": all_tf_results,
                }

            self.state.last_scan_time = cycle_result["timestamp"]
            self.state.last_scan_action = cycle_result["action"]
            self.state.last_error = None
            self.state.save()

        except Exception as e:
            logger.error(f"[TRADE] Scan error: {e}", exc_info=True)
            self.state.last_error = str(e)
            self.state.last_scan_action = "error"
            self.state.save()
            cycle_result["action"] = "error"
            cycle_result["details"] = {"error": str(e)}

        return cycle_result

    def _enter_trade(self, schematic: Dict, evaluation: Dict, current_price: float, reward_bias: Dict) -> Dict:
        """Open a simulated trade based on a qualifying schematic."""
        direction = evaluation["direction"]
        entry_info = schematic.get("entry", {})
        stop_info = schematic.get("stop_loss", {})
        target_info = schematic.get("target", {})

        entry_price = current_price  # always enter at actual market price
        stop_price = stop_info.get("price")
        target_price = target_info.get("price")

        if not stop_price or not target_price:
            return {"error": "Missing stop or target price"}

        # Calculate position sizing (1% risk)
        risk_amount = self.state.balance * (RISK_PER_TRADE_PCT / 100)
        if direction == "bullish":
            sl_pct = abs((entry_price - stop_price) / entry_price) * 100
        else:
            sl_pct = abs((stop_price - entry_price) / entry_price) * 100

        if sl_pct <= 0:
            sl_pct = 1.0  # safety floor

        position_size = calculate_position_size(risk_amount, sl_pct)
        margin = calculate_margin(position_size, DEFAULT_LEVERAGE)

        # Check liquidation safety
        liq_dir = "long" if direction == "bullish" else "short"
        liq_price = calculate_liquidation_price(entry_price, DEFAULT_LEVERAGE, liq_dir)
        safety = check_liquidation_safety(liq_price, stop_price, entry_price, liq_dir)

        trade = {
            "id": len(self.state.trade_history) + 1,
            "symbol": SYMBOL,
            "direction": direction,
            "model": evaluation.get("model", "unknown"),
            "entry_price": round(entry_price, 2),
            "stop_price": round(stop_price, 2),
            "target_price": round(target_price, 2),
            "position_size": round(position_size, 2),
            "margin": round(margin, 2),
            "risk_amount": round(risk_amount, 2),
            "leverage": DEFAULT_LEVERAGE,
            "rr": evaluation.get("rr", 0),
            "entry_score": evaluation["score"],
            "entry_reasons": evaluation["reasons"],
            "reward_bias": reward_bias,
            "liquidation_price": round(liq_price, 2),
            "liquidation_safe": safety["is_safe"],
            "opened_at": datetime.now(timezone.utc).isoformat(),
            "status": "open",
        }

        self.state.current_trade = trade
        self.state.save()
        logger.info(f"[TRADE] Entered {direction} @ {entry_price} | SL={stop_price} | TP={target_price} | Score={evaluation['score']}")
        notify_trade_entered(trade)
        return trade

    def _manage_open_trade(self, current_price: float) -> Dict:
        """Check if the open trade hit SL or TP."""
        trade = self.state.current_trade
        if not trade:
            return {"action": "no_trade"}

        direction = trade["direction"]
        entry_price = trade["entry_price"]
        stop_price = trade["stop_price"]
        target_price = trade["target_price"]
        risk_amount = trade["risk_amount"]

        # Calculate current P&L
        if direction == "bullish":
            pnl_pct = ((current_price - entry_price) / entry_price) * 100
            hit_target = current_price >= target_price
            hit_stop = current_price <= stop_price
        else:
            pnl_pct = ((entry_price - current_price) / entry_price) * 100
            hit_target = current_price <= target_price
            hit_stop = current_price >= stop_price

        trade["live_pnl_pct"] = round(pnl_pct, 2)
        trade["current_price"] = round(current_price, 2)

        if hit_target:
            return self._close_trade(current_price, "target_hit")
        elif hit_stop:
            return self._close_trade(current_price, "stop_hit")
        else:
            self.state.save()
            return {
                "action": "holding",
                "pnl_pct": round(pnl_pct, 2),
                "current_price": current_price,
                "entry_price": entry_price,
                "direction": direction,
            }

    def _close_trade(self, exit_price: float, reason: str) -> Dict:
        """Close the current trade and record results."""
        trade = self.state.current_trade
        if not trade:
            return {"action": "no_trade"}

        direction = trade["direction"]
        entry_price = trade["entry_price"]
        risk_amount = trade["risk_amount"]
        rr = trade.get("rr", 1)

        # Calculate actual P&L
        if direction == "bullish":
            pnl_pct = ((exit_price - entry_price) / entry_price) * 100
        else:
            pnl_pct = ((entry_price - exit_price) / entry_price) * 100

        is_win = reason == "target_hit"

        # Calculate dollar P&L based on position size
        position_size = trade.get("position_size", 0)
        pnl_dollars = position_size * (pnl_pct / 100)

        # Update balance
        self.state.balance += pnl_dollars

        # Compute reward for this trade
        reward_value = pnl_pct / 100  # normalized

        # Generate loss analysis / win confirmation
        if is_win:
            self.state.total_wins += 1
            self.evaluator.adapt_after_win()
            analysis = f"WIN: {trade['model']} {direction} trade hit target. R:R={rr:.1f}, reward bias was aligned."
            solution = "Continue with current strategy — reward alignment confirmed."
        else:
            self.state.total_losses += 1
            solution = self.evaluator.adapt_after_loss(trade)
            analysis = f"LOSS: {trade['model']} {direction} trade stopped out. Price moved against position."
            if trade.get("reward_bias", {}).get("bias") != direction.replace("ish", ""):
                analysis += " Reward bias was conflicting — consider only trading with aligned bias."

        closed_trade = {
            **trade,
            "exit_price": round(exit_price, 2),
            "exit_reason": reason,
            "pnl_pct": round(pnl_pct, 2),
            "pnl_dollars": round(pnl_dollars, 2),
            "is_win": is_win,
            "closed_at": datetime.now(timezone.utc).isoformat(),
            "status": "closed",
            "analysis": analysis,
            "solution": solution,
            "balance_after": round(self.state.balance, 2),
            "reward_value": round(reward_value, 6),
        }

        self.state.trade_history.append(closed_trade)
        self.state.reward_history.append(reward_value)
        self.state.solutions_applied.append(solution)
        self.state.current_trade = None
        self.state.save()

        logger.info(f"[TRADE] Closed: {reason} | P&L={pnl_pct:.2f}% (${pnl_dollars:.2f}) | Balance=${self.state.balance:.2f}")

        # Include win/loss totals for the notification
        closed_trade["_wins"] = self.state.total_wins
        closed_trade["_losses"] = self.state.total_losses

        if reason == "force_closed":
            notify_trade_force_closed(closed_trade)
        else:
            notify_trade_closed(closed_trade)

        return {
            "action": "trade_closed",
            "trade": closed_trade,
        }

    def _is_duplicate_setup(self, entry_price: float, direction: str) -> bool:
        """
        Check if this setup duplicates a recently closed trade.
        Returns True if the same entry price + direction was traded recently
        (within DUPLICATE_COOLDOWN_SECONDS and DUPLICATE_PRICE_TOLERANCE).
        """
        if not self.state.trade_history:
            return False

        now = datetime.now(timezone.utc)
        # Check last 5 closed trades for matching entry price + direction
        for t in reversed(self.state.trade_history[-5:]):
            t_entry = t.get("entry_price", 0)
            t_dir = t.get("direction", "")
            if t_dir != direction:
                continue
            # Price within tolerance?
            if t_entry > 0 and abs(t_entry - entry_price) / t_entry < DUPLICATE_PRICE_TOLERANCE:
                # Check cooldown timer from when the trade was closed
                closed_at = t.get("closed_at")
                if closed_at:
                    try:
                        closed_time = datetime.fromisoformat(closed_at)
                        if closed_time.tzinfo is None:
                            closed_time = closed_time.replace(tzinfo=timezone.utc)
                        elapsed = (now - closed_time).total_seconds()
                        if elapsed < DUPLICATE_COOLDOWN_SECONDS:
                            logger.info(
                                f"[DEDUP] Blocking duplicate: {direction} @ {entry_price} "
                                f"matches trade #{t.get('id')} closed {elapsed:.0f}s ago "
                                f"(cooldown {DUPLICATE_COOLDOWN_SECONDS}s)"
                            )
                            return True
                    except (ValueError, TypeError):
                        pass
                else:
                    # No closed_at timestamp — treat as recent duplicate to be safe
                    return True
        return False

    def force_close(self) -> Dict:
        """Force-close the current trade at market price."""
        if not self.state.current_trade:
            return {"action": "no_trade"}
        price = fetch_live_price(SYMBOL)
        if not price:
            return {"action": "error", "details": "Could not fetch live price"}
        return self._close_trade(price, "force_closed")



# ================================================================
# SINGLETON
# ================================================================
_trader_instance: Optional[TensorTCTTrader] = None


def get_trader() -> TensorTCTTrader:
    global _trader_instance
    if _trader_instance is None:
        _trader_instance = TensorTCTTrader()
    return _trader_instance
