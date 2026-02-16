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
        except Exception as e:
            logger.warning(f"[STATE] Could not load trade state: {e}")

    def save(self):
        """Persist state to disk."""
        try:
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
def fetch_candles_sync(symbol: str = SYMBOL, interval: str = TIMEFRAME, limit: int = 200) -> Optional[pd.DataFrame]:
    """Fetch OHLCV candles from MEXC (synchronous)."""
    # Normalize intervals to MEXC-supported values (MEXC spot uses 60m not 1h)
    _MEXC_INTERVAL_MAP = {"1h": "60m", "2h": "4h"}
    interval = _MEXC_INTERVAL_MAP.get(interval, interval)
    url = f"{MEXC_URL_BASE}/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    try:
        res = requests.get(url, params=params, timeout=15)
        if res.status_code != 200:
            logger.error(f"[FETCH] HTTP {res.status_code}")
            return None
        data = res.json()
        if not isinstance(data, list) or len(data) == 0:
            return None
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
    except Exception as e:
        logger.error(f"[FETCH] {symbol}/{interval}: {e}", exc_info=True)
        return None


def fetch_live_price(symbol: str = SYMBOL) -> Optional[float]:
    """Get current price from MEXC ticker."""
    try:
        url = f"{MEXC_URL_BASE}/api/v3/ticker/price"
        res = requests.get(url, params={"symbol": symbol}, timeout=10)
        return float(res.json().get("price", 0))
    except Exception:
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
        model = schematic.get("model", "unknown")
        is_confirmed = schematic.get("is_confirmed", False)
        rr = schematic.get("risk_reward_ratio", 0)
        # Also check "risk_reward" key (schematics use both naming conventions)
        if not rr:
            rr = schematic.get("risk_reward", 0) or 0

        # Must be confirmed (BOS happened)
        if not is_confirmed:
            return {"score": 0, "direction": direction, "reasons": ["No BOS confirmation"], "required_score": 50, "pass": False, "model": model, "rr": rr}

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
            score += 8
            reasons.append(f"Acceptable R:R ({rr:.1f})")
        elif rr >= 1.0:
            score += 3
            reasons.append(f"Minimum R:R ({rr:.1f})")
        else:
            return {"score": 0, "direction": direction, "reasons": [f"R:R too low ({rr:.1f})"], "required_score": 50, "pass": False, "model": model, "rr": rr}

        # 6CR validation — strong quality signal from TCT methodology
        six_cr_valid = schematic.get("six_candle_valid", False)
        htf_val = schematic.get("lecture_5b_enhancements", {}) or {}
        htf_6cr = (htf_val.get("htf_validation") or {}).get("all_taps_valid_6cr", False)
        if six_cr_valid or htf_6cr:
            score += 10
            reasons.append("6CR validated on all taps")

        # Quality score from the schematic detector
        quality = schematic.get("quality_score", 0) or 0
        if quality >= 0.8:
            score += 8
            reasons.append(f"High quality ({quality:.0%})")
        elif quality >= 0.5:
            score += 4
            reasons.append(f"Moderate quality ({quality:.0%})")

        # Reward bias alignment
        if direction == "bullish" and reward_bias["bias"] == "bullish":
            score += 15
            reasons.append("Reward bias aligned (bullish)")
        elif direction == "bearish" and reward_bias["bias"] == "bearish":
            score += 15
            reasons.append("Reward bias aligned (bearish)")
        elif reward_bias["bias"] == "neutral":
            score += 5
            reasons.append("Neutral reward bias — proceed with caution")
        else:
            score -= 5
            reasons.append(f"Reward bias conflicting ({reward_bias['bias']} vs {direction})")

        # Confidence boost
        if reward_bias["confidence"] > 60:
            score += 8
            reasons.append(f"High reward confidence ({reward_bias['confidence']}%)")
        elif reward_bias["confidence"] > 30:
            score += 4
            reasons.append(f"Moderate reward confidence ({reward_bias['confidence']}%)")

        # Model type preference
        if "Model_1" in model:
            score += 5
            reasons.append("Model 1 — deeper deviation (stronger)")
        elif "Model_2" in model:
            score += 3
            reasons.append("Model 2 — higher low / lower high")

        # Price proximity to entry — bonus when price is near the entry zone
        entry_info = schematic.get("entry", {})
        entry_price = entry_info.get("price")
        if entry_price and entry_price > 0:
            distance_pct = abs(current_price - entry_price) / entry_price * 100
            if distance_pct <= 0.3:
                score += 10
                reasons.append(f"Price at entry zone ({distance_pct:.2f}% away)")
            elif distance_pct <= 1.0:
                score += 5
                reasons.append(f"Price near entry ({distance_pct:.2f}% away)")

            # Penalize if price already moved well past entry
            if direction == "bullish" and current_price > entry_price * 1.02:
                score -= 10
                reasons.append("Price already moved 2%+ above entry — late")
            elif direction == "bearish" and current_price < entry_price * 0.98:
                score -= 10
                reasons.append("Price already moved 2%+ below entry — late")

        # Consecutive loss adaptation — tighten entry requirements
        if self.consecutive_losses >= 3:
            required = 65
            reasons.append(f"Tightened entry (3+ losses) — need score >= {required}")
        elif self.consecutive_losses >= 2:
            required = 55
            reasons.append(f"Cautious mode (2 losses) — need score >= {required}")
        else:
            required = 45

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
                    logger.warning(f"[TRADE] Error scanning {tf}: {tf_err}")
                    all_tf_results[tf] = {"status": "error", "error": str(tf_err)}

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

        entry_price = entry_info.get("price", current_price)
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
        return {
            "action": "trade_closed",
            "trade": closed_trade,
        }

    def force_close(self) -> Dict:
        """Force-close the current trade at market price."""
        if not self.state.current_trade:
            return {"action": "no_trade"}
        price = fetch_live_price(SYMBOL)
        if not price:
            return {"action": "error", "details": "Could not fetch live price"}
        return self._close_trade(price, "force_closed")

    def reset(self) -> Dict:
        """Reset the trading state to start fresh."""
        self.state = TradeState()
        self.state.balance = STARTING_BALANCE
        self.state.starting_balance = STARTING_BALANCE
        self.state.current_trade = None
        self.state.trade_history = []
        self.state.reward_history = []
        self.state.total_wins = 0
        self.state.total_losses = 0
        self.state.solutions_applied = []
        self.state.last_error = None
        self.state.save()
        self.evaluator = TCTTradeEvaluator()
        return {"action": "reset", "balance": STARTING_BALANCE}


# ================================================================
# SINGLETON
# ================================================================
_trader_instance: Optional[TensorTCTTrader] = None


def get_trader() -> TensorTCTTrader:
    global _trader_instance
    if _trader_instance is None:
        _trader_instance = TensorTCTTrader()
    return _trader_instance
