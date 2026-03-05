"""
5A_tct_trader.py — TCT Schematics Simulated Trading Engine (Lecture 5A)
=======================================================================
Deterministic paper-trading engine implementing Lecture 5A methodology.
No ML, no reward learning — fixed 50-point entry threshold.

Imported via tensor_tct_trader (thin shim) by server_mexc and schematics_5b_trader.

Trade lifecycle:
  NO_TRADE → qualified schematic → enter position → manage (half-TP, stop, target) → close
"""

from __future__ import annotations

import importlib
import json
import logging
import os
import shutil
import threading
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional

import pandas as pd
import requests

from tct_schematics import detect_tct_schematics

logger = logging.getLogger("5ATCTTrader")

# ================================================================
# CONFIGURATION
# ================================================================

STARTING_BALANCE: float = float(os.getenv("TENSOR_TCT_STARTING_BALANCE", "5000"))
RISK_PER_TRADE_PCT: float = 1.0          # % of balance risked per trade
AUTO_SCAN_INTERVAL: int = int(os.getenv("TENSOR_TCT_SCAN_INTERVAL", "300"))
DUPLICATE_COOLDOWN_SECONDS: int = 3600  # 1 hour
DUPLICATE_PRICE_TOLERANCE: float = 0.001  # 0.1%
MIN_RR: float = 1.5
MIN_QUALITY_SCORE: float = 0.70
ENTRY_THRESHOLD: int = 50  # Fixed — never adapts

MEXC_KLINES_URL = "https://api.mexc.com/api/v3/klines"
MEXC_TICKER_URL = "https://api.mexc.com/api/v3/ticker/price"

_DEFAULT_LOG_DIR = "/opt/render/project/chroma_db"
_LOG_DIR = (os.getenv("TENSOR_TCT_LOG_DIR", _DEFAULT_LOG_DIR) or "").strip() or _DEFAULT_LOG_DIR
try:
    os.makedirs(_LOG_DIR, exist_ok=True)
except OSError:
    _LOG_DIR = os.path.dirname(os.path.abspath(__file__))

TRADE_LOG_PATH = os.path.join(_LOG_DIR, "tensor_trade_log.json")
TRADE_LOG_BACKUP_PATH = os.path.join(_LOG_DIR, "tensor_trade_log_backup.json")

# MTF timeframes to scan (ordered highest → lowest)
_MTF_TIMEFRAMES = ["1d", "4h", "1h", "30m"]
_MTF_CANDLE_LIMITS: Dict[str, int] = {
    "1d": 200, "4h": 300, "1h": 300, "30m": 400,
}
_MAX_STALE: Dict[str, int] = {
    "1d": 2, "4h": 3, "1h": 3, "30m": 4,
}

# HTF bias cache TTL (seconds)
_HTF_BIAS_TTL: float = 3600.0


# ================================================================
# MEXC DATA FETCHING
# ================================================================

def fetch_candles_sync(symbol: str, tf: str, limit: int = 300) -> Optional[pd.DataFrame]:
    """
    Fetch OHLCV candles from MEXC synchronously.

    Args:
        symbol: e.g. "BTCUSDT"
        tf: timeframe e.g. "4h"
        limit: number of candles (max 1000)

    Returns:
        DataFrame with columns [open_time, open, high, low, close, volume], or None on error.
    """
    try:
        resp = requests.get(
            MEXC_KLINES_URL,
            params={"symbol": symbol, "interval": tf, "limit": limit},
            timeout=20,
            headers={"User-Agent": "5A-TCT-Trader/1.0"},
        )
        resp.raise_for_status()
        raw = resp.json()
        if not isinstance(raw, list) or not raw:
            logger.warning("[5A-FETCH] Empty response for %s/%s", symbol, tf)
            return None

        rows = [[r[0], r[1], r[2], r[3], r[4], r[5]] for r in raw]
        df = pd.DataFrame(rows, columns=["open_time", "open", "high", "low", "close", "volume"])
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        for col in ("open", "high", "low", "close", "volume"):
            df[col] = df[col].astype(float)
        df.reset_index(drop=True, inplace=True)
        return df
    except Exception as exc:
        logger.warning("[5A-FETCH] Candle fetch failed %s/%s: %s", symbol, tf, exc)
        return None


def fetch_live_price(symbol: str = "BTCUSDT") -> Optional[float]:
    """
    Fetch the current mid price from MEXC ticker.

    Returns:
        Price as float, or None on error.
    """
    try:
        resp = requests.get(
            MEXC_TICKER_URL,
            params={"symbol": symbol},
            timeout=10,
            headers={"User-Agent": "5A-TCT-Trader/1.0"},
        )
        resp.raise_for_status()
        data = resp.json()
        return float(data["price"])
    except Exception as exc:
        logger.warning("[5A-FETCH] Live price fetch failed %s: %s", symbol, exc)
        return None


# ================================================================
# TELEGRAM NOTIFICATIONS
# ================================================================

def _telegram_send(text: str) -> None:
    """Fire-and-forget Telegram message."""
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return

    def _send():
        try:
            url = f"https://api.telegram.org/bot{token}/sendMessage"
            requests.post(url, json={"chat_id": chat_id, "text": text, "parse_mode": "HTML"}, timeout=5)
        except Exception as exc:
            logger.warning("[5A-TELEGRAM] Send error: %s", exc)

    threading.Thread(target=_send, daemon=True).start()


def notify_trade_entered(trade: dict) -> None:
    direction = "LONG 🟢" if trade.get("direction") == "bullish" else "SHORT 🔴"
    _telegram_send(
        f"<b>5A Trade Entered — {direction}</b>\n"
        f"Entry: ${trade.get('entry_price', 0):,.2f}\n"
        f"Stop: ${trade.get('stop_price', 0):,.2f}\n"
        f"Target: ${trade.get('target_price', 0):,.2f}\n"
        f"R:R: {trade.get('rr', 0):.1f} | Score: {trade.get('entry_score', 0)}\n"
        f"Model: {trade.get('model', '?')} | TF: {trade.get('timeframe', '?')}"
    )


def notify_trade_closed(trade: dict) -> None:
    result = "WIN ✅" if trade.get("is_win") else "LOSS ❌"
    pnl = trade.get("pnl_dollars", 0)
    sign = "+" if pnl >= 0 else ""
    _telegram_send(
        f"<b>5A Trade Closed — {result}</b>\n"
        f"Exit: ${trade.get('exit_price', 0):,.2f}\n"
        f"P&L: {sign}${pnl:,.2f} ({sign}{trade.get('pnl_pct', 0):.2f}%)\n"
        f"Balance: ${trade.get('balance_after', 0):,.2f}"
    )


def notify_trade_force_closed(trade: dict) -> None:
    pnl = trade.get("pnl_dollars", 0)
    sign = "+" if pnl >= 0 else ""
    _telegram_send(
        f"<b>5A Trade Force-Closed ⚠️</b>\n"
        f"Reason: {trade.get('close_reason', '?')}\n"
        f"P&L: {sign}${pnl:,.2f}"
    )


def notify_half_tp_taken(trade: dict) -> None:
    pnl = trade.get("half_tp_pnl_dollars", 0)
    sign = "+" if pnl >= 0 else ""
    _telegram_send(
        f"<b>5A Half TP Taken 🎯</b>\n"
        f"At: ${trade.get('half_tp_price', 0):,.2f}\n"
        f"½ P&L: {sign}${pnl:,.2f}\n"
        f"Stop → Break-Even @ ${trade.get('entry_price', 0):,.2f}"
    )


# ================================================================
# TRADE STATE MANAGER
# ================================================================

class TradeState:
    """
    Persistent trade state for the 5A trader.

    Fields are restored from TRADE_LOG_PATH on init.
    """

    def __init__(self) -> None:
        self.balance: float = STARTING_BALANCE
        self.starting_balance: float = STARTING_BALANCE
        self.current_trade: Optional[dict] = None
        self.trade_history: List[dict] = []
        self.reward_history: List[float] = []
        self.total_wins: int = 0
        self.total_losses: int = 0
        self.solutions_applied: List[str] = []
        self.last_scan_time: Optional[str] = None
        self.last_scan_action: Optional[str] = None
        self.last_error: Optional[str] = None
        self._load()

    def _load(self) -> None:
        """Restore state from GitHub then local disk."""
        try:
            from github_storage import fetch_trade_log
            fetch_trade_log(TRADE_LOG_PATH)
        except Exception as exc:
            logger.warning("[5A-STATE] GitHub restore skipped: %s", exc)

        if not os.path.exists(TRADE_LOG_PATH):
            logger.info("[5A-STATE] No trade log found — starting fresh")
            return

        try:
            with open(TRADE_LOG_PATH, "r") as f:
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
            self.last_error = data.get("last_error")
            self._deduplicate_history()
            logger.info(
                "[5A-STATE] Loaded — balance=$%.2f, trades=%d",
                self.balance, len(self.trade_history),
            )
        except Exception as exc:
            logger.warning("[5A-STATE] Could not load trade state: %s", exc)

    def _deduplicate_history(self) -> None:
        """
        Remove duplicate trade records that may accumulate from repeated
        GitHub syncs or state restores.  Deduplicates on (entry_price,
        direction, closed_at) — trades that share all three are the same.
        """
        seen = set()
        deduped = []
        for t in self.trade_history:
            key = (
                t.get("entry_price"),
                t.get("direction"),
                t.get("closed_at"),
            )
            if key not in seen:
                seen.add(key)
                deduped.append(t)
        self.trade_history = deduped

    def save(self) -> None:
        """Write current state to disk and create a backup."""
        try:
            if os.path.exists(TRADE_LOG_PATH):
                shutil.copy2(TRADE_LOG_PATH, TRADE_LOG_BACKUP_PATH)

            data = {
                "balance": round(self.balance, 4),
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
                "last_updated": datetime.now(timezone.utc).isoformat(),
            }
            with open(TRADE_LOG_PATH, "w") as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as exc:
            logger.error("[5A-STATE] Save failed: %s", exc)

    def restore_from_backup(self) -> bool:
        """Restore state from the most recent backup file."""
        if not os.path.exists(TRADE_LOG_BACKUP_PATH):
            logger.warning("[5A-STATE] No backup file found at %s", TRADE_LOG_BACKUP_PATH)
            return False
        try:
            shutil.copy2(TRADE_LOG_BACKUP_PATH, TRADE_LOG_PATH)
            self._load()
            logger.info("[5A-STATE] Restored from backup")
            return True
        except Exception as exc:
            logger.error("[5A-STATE] Backup restore failed: %s", exc)
            return False

    def snapshot(self) -> dict:
        """Thread-safe read-only snapshot of current state."""
        total = max(len(self.trade_history), 1)
        win_rate = (self.total_wins / total) * 100
        pnl_total = self.balance - self.starting_balance
        return {
            "balance": round(self.balance, 2),
            "starting_balance": self.starting_balance,
            "pnl_total": round(pnl_total, 2),
            "pnl_pct": round((pnl_total / self.starting_balance) * 100, 2),
            "total_trades": len(self.trade_history),
            "wins": self.total_wins,
            "losses": self.total_losses,
            "win_rate": round(win_rate, 2),
            "current_trade": self.current_trade,
            "trade_history": self.trade_history[-50:],
            "last_scan_time": self.last_scan_time,
            "last_scan_action": self.last_scan_action,
            "last_error": self.last_error,
        }

    def derive_consecutive_losses(self) -> int:
        """
        Count consecutive losses from the tail of trade_history.

        Iterates backwards until a win is found.  Survives restarts
        because it is re-derived from persisted history every time.
        """
        count = 0
        for trade in reversed(self.trade_history):
            if trade.get("status") == "closed" and not trade.get("is_win"):
                count += 1
            else:
                break
        return count

    def compute_model_weights(self, min_trades: int = 10) -> Dict[str, int]:
        """
        Compute per-model performance bonus/penalty scores.

        Formula: win_rate = wins / total; bonus = round((win_rate - 0.5) * 20)
        Returns {} if any model has fewer than min_trades closed trades.
        """
        from collections import defaultdict
        model_wins: Dict[str, int] = defaultdict(int)
        model_total: Dict[str, int] = defaultdict(int)

        for t in self.trade_history:
            if t.get("status") != "closed":
                continue
            model = t.get("model", "unknown")
            model_total[model] += 1
            if t.get("is_win"):
                model_wins[model] += 1

        weights: Dict[str, int] = {}
        for model, total in model_total.items():
            if total < min_trades:
                return {}
            win_rate = model_wins[model] / total
            weights[model] = round((win_rate - 0.5) * 20)
        return weights


# ================================================================
# TCT TRADE EVALUATOR
# ================================================================

class TCTTradeEvaluator:
    """
    Scores TCT schematics for trade entry quality.

    Fixed 50-point threshold — reward learning is disabled.
    No `env` attribute, no `compute_reward_bias` method.
    """

    def __init__(self) -> None:
        self.consecutive_losses: int = 0
        self.model_weights: Dict[str, int] = {}
        self.adaptation_notes: List[str] = []

    def evaluate_schematic(
        self,
        schematic: dict,
        htf_bias: str,
        current_price: float,
        total_candles: int = 200,
        max_stale_candles: int = 5,
    ) -> dict:
        """
        Evaluate a TCT schematic for trade entry quality.

        Gates (in order):
          1. BOS confirmation (hard fail — returns immediately)
          2. Quality score floor ≥ 0.70 (hard fail — returns immediately)
          3. Stale BOS (hard fail — returns immediately)
          4. Live R:R at current price (hard fail — returns immediately)
          5. HTF bias alignment (hard fail on neutral; misalignment subtracts)

        Scoring:
          - Base: 30 pts for BOS confirmation
          - R:R tiers: 5 / 15 / 25 pts
          - HTF aligned: +20; misaligned: −10
          - Quality bonus: round(quality * 15)
          - Model type: +5 (Model_1), +3 (Model_2)
          - Late entry: −15 if >2% past ideal entry

        Required score is always 50 (no adaptive learning).
        """
        direction = schematic.get("direction", "unknown")
        model = schematic.get("model", schematic.get("schematic_type", "unknown"))
        is_confirmed = schematic.get("is_confirmed", False)
        tap3 = schematic.get("tap3", {})
        quality_score = schematic.get("quality_score", 0.0)

        # Compute live R:R at current price
        stop_price = (schematic.get("stop_loss") or {}).get("price")
        target_price = (schematic.get("target") or {}).get("price")
        if stop_price and target_price and current_price > 0:
            if direction == "bullish":
                live_risk = current_price - stop_price
                live_reward = target_price - current_price
            else:
                live_risk = stop_price - current_price
                live_reward = current_price - target_price
            rr = (live_reward / live_risk) if live_risk > 0 else 0.0
        else:
            rr = float(schematic.get("risk_reward", 0) or 0)

        _fail = {
            "score": 0,
            "direction": direction,
            "model": model,
            "rr": rr,
            "required_score": ENTRY_THRESHOLD,
            "pass": False,
        }

        # Gate 1: BOS confirmation
        bos = schematic.get("bos_confirmation") or {}
        if not is_confirmed or not bos.get("confirmed"):
            return {**_fail, "reasons": ["No BOS confirmation"]}

        # Gate 2: Quality floor
        if quality_score < MIN_QUALITY_SCORE:
            return {**_fail, "reasons": [f"Quality too low ({quality_score:.2f} < {MIN_QUALITY_SCORE:.2f})"]}

        # Gate 3: Stale BOS
        bos_idx = bos.get("bos_idx")
        if bos_idx is not None and bos_idx < total_candles - max_stale_candles:
            return {
                **_fail,
                "reasons": [f"Stale BOS: {total_candles - bos_idx} candles ago (max {max_stale_candles})"],
            }

        # Gate 4: Model structure validation
        if "Model_1_from_M2_failure" not in model:
            if "Model_1" in model:
                if tap3.get("type") != "tap3_model1":
                    return {**_fail, "reasons": ["Model 1: Tap3 must be deeper deviation"]}
            elif "Model_2" in model:
                if direction == "bullish" and not tap3.get("is_higher_low"):
                    return {**_fail, "reasons": ["Model 2: Tap3 must be higher low for accumulation"]}
                if direction == "bearish" and not tap3.get("is_lower_high"):
                    return {**_fail, "reasons": ["Model 2: Tap3 must be lower high for distribution"]}

        # Gate 5: Live R:R
        score = 0
        reasons: List[str] = []

        score += 30
        reasons.append("BOS confirmed")

        if rr >= 3.0:
            score += 25
            reasons.append(f"Excellent R:R ({rr:.1f})")
        elif rr >= 2.0:
            score += 15
            reasons.append(f"Good R:R ({rr:.1f})")
        elif rr >= MIN_RR:
            score += 5
            reasons.append(f"Acceptable R:R ({rr:.1f})")
        else:
            return {**_fail, "reasons": [f"R:R too low ({rr:.2f}, min {MIN_RR})"]}

        # Gate 6: HTF bias alignment
        if htf_bias == "neutral":
            return {**_fail, "reasons": ["HTF bias is neutral — no directional confirmation"]}
        elif (direction == "bullish" and htf_bias == "bullish") or \
                (direction == "bearish" and htf_bias == "bearish"):
            score += 20
            reasons.append(f"HTF bias aligned ({htf_bias})")
        else:
            score -= 10
            reasons.append(f"HTF bias conflict ({htf_bias} vs {direction})")

        # Quality bonus
        quality_bonus = round(quality_score * 15)
        score += quality_bonus
        reasons.append(f"Quality {quality_score:.2f} (+{quality_bonus})")

        # Model bonus
        if "Model_1" in model:
            score += 5
            reasons.append("Model 1 — deeper deviation")
        elif "Model_2" in model:
            score += 3
            reasons.append("Model 2 — HL/LH")

        # Late entry penalty
        entry_info = schematic.get("entry", {})
        entry_price = entry_info.get("price")
        if entry_price:
            if direction == "bullish" and current_price > entry_price * 1.02:
                score -= 15
                reasons.append("Price 2%+ above entry — late")
            elif direction == "bearish" and current_price < entry_price * 0.98:
                score -= 15
                reasons.append("Price 2%+ below entry — late")

        score = max(0, min(100, score))
        return {
            "score": score,
            "direction": direction,
            "model": model,
            "rr": rr,
            "required_score": ENTRY_THRESHOLD,
            "pass": score >= ENTRY_THRESHOLD,
            "reasons": reasons,
        }


# ================================================================
# TENSOR TCT TRADER
# ================================================================

class TensorTCTTrader:
    """
    Paper-trading engine implementing Lecture 5A TCT methodology.

    Thread-safe. All state mutations go through _enter_trade(),
    _close_trade(), and state.save().
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.state = TradeState()
        self.evaluator = TCTTradeEvaluator()
        self.last_debug: dict = {}

        # HTF bias cache
        self._htf_bias_cache: Optional[str] = None
        self._htf_bias_expiry: float = 0.0

        # Derive consecutive_losses from persisted history (Issue 5)
        self.evaluator.consecutive_losses = self.state.derive_consecutive_losses()

    # ------------------------------------------------------------------
    # HTF bias (cached)
    # ------------------------------------------------------------------

    def _get_htf_bias(self, symbol: str = "BTCUSDT") -> str:
        """
        Fetch HTF (1d) bias: "bullish", "bearish", or "neutral".
        Cached for _HTF_BIAS_TTL seconds.
        """
        now = time.time()
        if self._htf_bias_cache is not None and now < self._htf_bias_expiry:
            return self._htf_bias_cache

        try:
            df = fetch_candles_sync(symbol, "1d", 200)
            if df is None or len(df) < 10:
                bias = "neutral"
            else:
                result = detect_tct_schematics(df)
                acc = [s for s in result.get("accumulation_schematics", []) if s.get("is_confirmed")]
                dist = [s for s in result.get("distribution_schematics", []) if s.get("is_confirmed")]
                if acc and not dist:
                    bias = "bullish"
                elif dist and not acc:
                    bias = "bearish"
                else:
                    bias = "neutral"
        except Exception as exc:
            logger.warning("[5A-HTF] HTF bias fetch failed: %s", exc)
            bias = "neutral"

        self._htf_bias_cache = bias
        self._htf_bias_expiry = now + _HTF_BIAS_TTL
        return bias

    # ------------------------------------------------------------------
    # Duplicate suppression
    # ------------------------------------------------------------------

    def _is_duplicate_setup(self, entry_price: float, direction: str) -> bool:
        """
        Return True if a matching trade was closed within DUPLICATE_COOLDOWN_SECONDS.
        Matching: same direction + entry_price within DUPLICATE_PRICE_TOLERANCE.
        """
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=DUPLICATE_COOLDOWN_SECONDS)
        for t in reversed(self.state.trade_history):
            closed_at_raw = t.get("closed_at")
            if not closed_at_raw:
                continue
            try:
                closed_at = datetime.fromisoformat(closed_at_raw)
                if closed_at.tzinfo is None:
                    closed_at = closed_at.replace(tzinfo=timezone.utc)
            except (ValueError, TypeError):
                continue

            if closed_at < cutoff:
                break  # History is chronological; older entries won't match

            if t.get("direction") != direction:
                continue

            prev_entry = t.get("entry_price", 0.0)
            if prev_entry and abs(entry_price - prev_entry) / prev_entry <= DUPLICATE_PRICE_TOLERANCE:
                return True
        return False

    # ------------------------------------------------------------------
    # Trade lifecycle
    # ------------------------------------------------------------------

    def _enter_trade(
        self,
        schematic: dict,
        evaluation: dict,
        entry_price: float,
        htf_bias: str,
    ) -> dict:
        """
        Open a simulated trade position.

        Validates stop/target geometry and minimum R:R before entry.
        Returns an error dict if geometry is invalid.
        """
        direction = evaluation.get("direction", schematic.get("direction", "bullish"))
        stop_price = float((schematic.get("stop_loss") or {}).get("price", 0))
        target_price = float((schematic.get("target") or {}).get("price", 0))

        # Geometry validation
        if direction == "bullish":
            if stop_price >= entry_price:
                return {"error": f"Invalid stop: stop {stop_price} >= entry {entry_price} for long"}
            if target_price <= entry_price:
                return {"error": f"Invalid target: target {target_price} <= entry {entry_price} for long"}
            risk = entry_price - stop_price
            reward = target_price - entry_price
        else:  # bearish
            if stop_price <= entry_price:
                return {"error": f"Invalid stop: stop {stop_price} <= entry {entry_price} for short"}
            if target_price >= entry_price:
                return {"error": f"Invalid target: target {target_price} >= entry {entry_price} for short"}
            risk = stop_price - entry_price
            reward = entry_price - target_price

        actual_rr = reward / risk if risk > 0 else 0.0
        if actual_rr < MIN_RR:
            return {
                "error": f"R:R {actual_rr:.2f} below minimum {MIN_RR} — trade rejected"
            }

        # Position sizing: risk 1% of balance
        risk_amount = self.state.balance * (RISK_PER_TRADE_PCT / 100.0)
        stop_distance_pct = risk / entry_price
        position_size = risk_amount / stop_distance_pct if stop_distance_pct > 0 else 0.0

        # Half-TP at 49% of distance to target
        if direction == "bullish":
            half_tp_price = round(entry_price + (target_price - entry_price) * 0.49, 2)
        else:
            half_tp_price = round(entry_price - (entry_price - target_price) * 0.49, 2)

        trade = {
            "id": int(time.time() * 1000),
            "direction": direction,
            "model": evaluation.get("model", schematic.get("model", "unknown")),
            "timeframe": schematic.get("timeframe", "unknown"),
            "symbol": schematic.get("symbol", "BTCUSDT"),
            "entry_price": entry_price,
            "stop_price": stop_price,
            "target_price": target_price,
            "rr": round(actual_rr, 2),
            "position_size": round(position_size, 4),
            "original_position_size": round(position_size, 4),
            "risk_amount": round(risk_amount, 2),
            "entry_score": evaluation.get("score", 0),
            "htf_bias": htf_bias,
            "half_tp_price": half_tp_price,
            "half_tp_taken": False,
            "half_tp_pnl_dollars": 0.0,
            "stop_is_breakeven": False,
            "status": "open",
            "opened_at": datetime.now(timezone.utc).isoformat(),
            "pnl_dollars": 0.0,
            "pnl_pct": 0.0,
        }
        self.state.current_trade = trade
        self.state.save()
        notify_trade_entered(trade)
        logger.info(
            "[5A] Trade ENTERED — %s entry=%.2f stop=%.2f target=%.2f R:R=%.2f",
            direction, entry_price, stop_price, target_price, actual_rr,
        )
        return {"action": "trade_entered", "trade": trade}

    def _close_trade(self, exit_price: float, reason: str) -> dict:
        """
        Close the current open trade.

        P&L accounts for half-TP already credited to balance.
        Returns result dict with closed trade.
        """
        trade = self.state.current_trade
        if trade is None:
            return {"action": "no_trade"}

        entry_price = trade["entry_price"]
        direction = trade["direction"]
        position_size = trade["position_size"]  # already halved if half-TP fired

        if direction == "bullish":
            remaining_pnl_pct = (exit_price - entry_price) / entry_price
        else:
            remaining_pnl_pct = (entry_price - exit_price) / entry_price

        remaining_pnl_dollars = round(position_size * remaining_pnl_pct, 4)

        # Total P&L includes half-TP already booked + remaining position
        total_pnl_dollars = round(trade["half_tp_pnl_dollars"] + remaining_pnl_dollars, 4)
        is_win = total_pnl_dollars > 0

        # Update balance (only remaining; half-TP was already credited)
        self.state.balance = round(self.state.balance + remaining_pnl_dollars, 4)

        closed = {
            **trade,
            "status": "closed",
            "exit_price": exit_price,
            "close_reason": reason,
            "pnl_dollars": total_pnl_dollars,
            "pnl_pct": round(total_pnl_dollars / trade["risk_amount"] * RISK_PER_TRADE_PCT, 4)
            if trade["risk_amount"] else 0.0,
            "is_win": is_win,
            "balance_after": round(self.state.balance, 2),
            "closed_at": datetime.now(timezone.utc).isoformat(),
        }

        self.state.trade_history.append(closed)
        self.state.current_trade = None

        if is_win:
            self.state.total_wins += 1
            self.evaluator.consecutive_losses = 0
        else:
            self.state.total_losses += 1
            self.evaluator.consecutive_losses += 1

        self.state.save()
        notify_trade_closed(closed)

        logger.info(
            "[5A] Trade CLOSED — %s exit=%.2f P&L=$%.2f balance=$%.2f",
            "WIN" if is_win else "LOSS",
            exit_price, total_pnl_dollars, self.state.balance,
        )
        return {"action": "trade_closed", "trade": closed}

    def _manage_open_trade(self, current_price: float) -> dict:
        """
        Check stop/target/half-TP for the current open trade.

        Returns action: "holding" | "half_tp_taken" | "stop_hit" | "target_hit"
        """
        trade = self.state.current_trade
        if trade is None:
            return {"action": "no_trade"}

        direction = trade["direction"]
        entry_price = trade["entry_price"]
        stop_price = trade["stop_price"]
        target_price = trade["target_price"]
        half_tp_price = trade["half_tp_price"]
        half_tp_taken = trade["half_tp_taken"]

        # Half-TP trigger (fires once)
        if not half_tp_taken:
            triggered = (direction == "bullish" and current_price >= half_tp_price) or \
                        (direction == "bearish" and current_price <= half_tp_price)
            if triggered:
                original_size = trade["position_size"]
                half_size = original_size / 2

                if direction == "bullish":
                    half_pnl_pct = (half_tp_price - entry_price) / entry_price
                else:
                    half_pnl_pct = (entry_price - half_tp_price) / entry_price

                half_pnl_dollars = round(half_size * half_pnl_pct, 4)

                # Credit balance immediately
                self.state.balance = round(self.state.balance + half_pnl_dollars, 4)

                # Update trade state
                trade["half_tp_taken"] = True
                trade["half_tp_pnl_dollars"] = half_pnl_dollars
                trade["position_size"] = round(half_size, 4)
                trade["stop_price"] = entry_price  # move stop to break-even
                trade["stop_is_breakeven"] = True

                self.state.current_trade = trade
                self.state.save()
                notify_half_tp_taken(trade)

                logger.info(
                    "[5A] Half-TP triggered at %.2f — P&L=$%.2f, stop→B/E=%.2f",
                    half_tp_price, half_pnl_dollars, entry_price,
                )
                return {"action": "half_tp_taken", "price": current_price, "half_pnl": half_pnl_dollars}

        # Re-read stop (may have moved to break-even)
        stop_price = trade["stop_price"]

        # Stop hit
        stop_triggered = (direction == "bullish" and current_price <= stop_price) or \
                         (direction == "bearish" and current_price >= stop_price)
        if stop_triggered:
            return self._close_trade(current_price, "stop_hit")

        # Target hit
        target_triggered = (direction == "bullish" and current_price >= target_price) or \
                           (direction == "bearish" and current_price <= target_price)
        if target_triggered:
            return self._close_trade(current_price, "target_hit")

        return {"action": "holding", "price": current_price}

    # ------------------------------------------------------------------
    # Main scan cycle
    # ------------------------------------------------------------------

    def scan_and_trade(self, symbol: str = "BTCUSDT") -> dict:
        """
        Run one complete scan-and-trade cycle.

        1. If open trade: manage it and return.
        2. Fetch HTF bias (cached).
        3. Scan each MTF timeframe for qualified schematics.
        4. Enter the best qualifying trade if found.

        Returns a detailed result dict with diagnostics.
        """
        with self._lock:
            timestamp = datetime.now(timezone.utc).isoformat()
            self.state.last_scan_time = timestamp

            # Manage existing open trade first
            if self.state.current_trade is not None:
                price = fetch_live_price(symbol) or 0.0
                result = self._manage_open_trade(price)
                self.state.last_scan_action = result.get("action", "managing_open_trade")
                self.state.save()
                return {
                    "action": self.state.last_scan_action,
                    "timestamp": timestamp,
                    "price": price,
                }

            # Fetch HTF directional bias
            htf_bias = self._get_htf_bias(symbol)

            # Scan each MTF timeframe
            timeframes_scanned: Dict[str, dict] = {}
            best_schematic: Optional[dict] = None
            best_evaluation: Optional[dict] = None
            best_score: int = -1
            best_tf: Optional[str] = None

            for tf in _MTF_TIMEFRAMES:
                limit = _MTF_CANDLE_LIMITS.get(tf, 300)
                max_stale = _MAX_STALE.get(tf, 5)

                try:
                    df = fetch_candles_sync(symbol, tf, limit)
                    if df is None or len(df) < 20:
                        timeframes_scanned[tf] = {"status": "skip", "reason": "no_candles"}
                        continue

                    total_candles = len(df)
                    current_price = float(df.iloc[-1]["close"])

                    schematics_result = detect_tct_schematics(df)
                    all_schematics = (
                        schematics_result.get("accumulation_schematics", []) +
                        schematics_result.get("distribution_schematics", [])
                    )
                    confirmed = [s for s in all_schematics if s.get("is_confirmed")]

                    tf_best_score = -1
                    tf_best_eval = None
                    tf_best_sch = None

                    for sch in confirmed:
                        eval_result = self.evaluator.evaluate_schematic(
                            sch, htf_bias, current_price,
                            total_candles=total_candles,
                            max_stale_candles=max_stale,
                        )
                        if eval_result["pass"] and eval_result["score"] > tf_best_score:
                            tf_best_score = eval_result["score"]
                            tf_best_eval = eval_result
                            tf_best_sch = sch

                    if tf_best_sch is not None and tf_best_score > best_score:
                        tf_best_sch["timeframe"] = tf
                        tf_best_sch["symbol"] = symbol
                        best_score = tf_best_score
                        best_schematic = tf_best_sch
                        best_evaluation = tf_best_eval
                        best_tf = tf

                    timeframes_scanned[tf] = {
                        "status": "ok",
                        "candles": total_candles,
                        "confirmed_schematics": len(confirmed),
                        "htf_bias": htf_bias,
                        "current_price": current_price,
                    }

                except Exception as exc:
                    logger.error("[5A-SCAN] Error scanning %s/%s: %s", symbol, tf, exc, exc_info=True)
                    timeframes_scanned[tf] = {
                        "status": "error",
                        "fetch_error": True,
                        "error": str(exc),
                    }

            self.last_debug = {
                "timestamp": timestamp,
                "htf_bias": htf_bias,
                "timeframes_scanned": timeframes_scanned,
            }

            # Enter trade if a qualifying schematic was found
            action = "no_qualifying_setups"
            if best_schematic is not None and best_evaluation is not None:
                current_price = fetch_live_price(symbol) or float(
                    fetch_candles_sync(symbol, best_tf or "1h", 1).iloc[-1]["close"]
                    if fetch_candles_sync(symbol, best_tf or "1h", 1) is not None
                    else 0.0
                )
                entry_price = float(
                    (best_schematic.get("entry") or {}).get("price") or current_price
                )

                if self._is_duplicate_setup(entry_price, best_evaluation["direction"]):
                    action = "duplicate_suppressed"
                    logger.info("[5A-SCAN] Duplicate setup suppressed — %s @ %.2f",
                                best_evaluation["direction"], entry_price)
                else:
                    enter_result = self._enter_trade(best_schematic, best_evaluation, entry_price, htf_bias)
                    if "error" in enter_result:
                        action = f"entry_rejected: {enter_result['error']}"
                    else:
                        action = "trade_entered"

            self.state.last_scan_action = action
            self.state.save()

            return {
                "action": action,
                "timestamp": timestamp,
                "htf_bias": htf_bias,
                "details": {
                    "timeframes_scanned": timeframes_scanned,
                    "best_tf": best_tf,
                    "best_score": best_score if best_schematic else None,
                },
            }


# ================================================================
# MODULE-LEVEL SINGLETON
# ================================================================

_trader: Optional[TensorTCTTrader] = None
_trader_lock = threading.Lock()


def get_trader() -> TensorTCTTrader:
    """Return the TensorTCTTrader singleton, creating it on first call."""
    global _trader
    if _trader is None:
        with _trader_lock:
            if _trader is None:
                _trader = TensorTCTTrader()
    return _trader
