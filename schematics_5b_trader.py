"""
schematics_5b_trader.py — Schematics 5B Simulated Trading Engine
================================================================
Deterministic TCT trading engine using Lecture 5B methodology on BTCUSDT.
No learning, no reward system — fixed 50-point entry threshold forever.

Each cycle:
1. Fetch live BTCUSDT candles from MEXC
2. Run TCT schematic detection
3. Evaluate confirmed schematics for trade entry (fixed threshold)
4. Simulate position management (entry, SL, TP)
5. Log W/L results, persist to GitHub, notify via Telegram
"""

import os
import json
import time
import asyncio
import logging
import base64
import shutil
import threading
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from tct_schematics import detect_tct_schematics, TCTSchematicDetector
from concurrent.futures import ThreadPoolExecutor, as_completed
from trade_execution import (
    calculate_position_size,
    calculate_margin,
    calculate_liquidation_price,
    check_liquidation_safety,
)

# Reuse MEXC fetch helpers from tensor trader (no duplication)
from tensor_tct_trader import fetch_candles_sync, fetch_live_price

logger = logging.getLogger("Schematics5B")

# ================================================================
# CONFIGURATION
# ================================================================
SYMBOL = "BTCUSDT"
STARTING_BALANCE = 5000.0
RISK_PER_TRADE_PCT = 1.0  # 1% of balance per trade
DEFAULT_LEVERAGE = 10
# HTF bias gate — daily candle tells us the dominant directional context.
# 4h was too narrow; 1d changes once a day so the cache TTL matches.
HTF_TIMEFRAME = "1d"
# Scan TFs — ordered high to low.  4h is now scanned for setups (not just bias);
# 3D is not available on MEXC Spot so 1d is the longest granularity available.
# 30m added to give the HTF cascade a stepping-stone between 15m and 1h.
MTF_TIMEFRAMES = ["1d", "4h", "1h", "30m", "15m"]
AUTO_SCAN_INTERVAL = int(os.getenv("SCHEMATICS_5B_SCAN_INTERVAL", "60"))
ENTRY_THRESHOLD = 50  # Fixed — never changes

# TF hierarchy for the upward cascade (ordered highest → lowest).
# When a schematic is confirmed at a lower TF, we try to upgrade it to the
# highest TF that contains the same pattern in the same price area.
_TF_HIERARCHY = ["1W", "1d", "4h", "1h", "30m", "15m", "5m", "1m"]

# Candle limits per TF — sized so the BOS window never falls off the array
# even when tap3 is a recent candle.
_MTF_CANDLE_LIMITS = {
    "1d": 200,  # ~200 days
    "4h": 300,  # ~50 days
    "1h": 300,  # ~12.5 days
    "30m": 400, # ~8.3 days
    "15m": 500, # ~5.2 days
}

# Max candles since BOS before a setup is considered stale
_MAX_STALE: Dict[str, int] = {
    "1d": 2, "4h": 3, "1h": 3, "30m": 4, "15m": 5,
}

# LTF cascade: fetch lower-TF candles once per cycle for BOS refinement.
# Per TCT model, cascade from highest TF down to the lowest that confirms BOS;
# the earliest (lowest-TF) BOS gives the best entry and best R:R.
LTF_BOS_TIMEFRAMES = ["5m", "1m"]  # highest → lowest; we keep overwriting so lowest TF wins
_LTF_CANDLE_LIMITS = {"5m": 500, "1m": 1000}  # 500×5m ≈ 41 h; 1000×1m ≈ 17 h

_DIR = os.path.dirname(os.path.abspath(__file__))
TRADE_LOG_PATH = os.path.join(_DIR, "schematics_5b_trade_log.json")
TRADE_LOG_BACKUP_PATH = os.path.join(_DIR, "schematics_5b_trade_log_backup.json")

# Deduplication
DUPLICATE_COOLDOWN_SECONDS = 300
DUPLICATE_PRICE_TOLERANCE = 0.002


# ================================================================
# GITHUB STORAGE (uses GITHUB_TOKEN_2)
# ================================================================
_GITHUB_API = "https://api.github.com"
_DATA_BRANCH = "data"
_GITHUB_LOG_FILENAME = "schematics_5b_trade_log.json"


def _github_configured() -> bool:
    return bool(os.getenv("GITHUB_TOKEN_2") and os.getenv("GITHUB_REPO"))


def _github_headers() -> dict:
    return {
        "Authorization": f"token {os.getenv('GITHUB_TOKEN_2')}",
        "Accept": "application/vnd.github.v3+json",
    }


def github_fetch_5b_log(local_path: str) -> bool:
    """Pull schematics_5b_trade_log.json from the data branch on GitHub."""
    if not _github_configured():
        return False
    if os.path.exists(local_path):
        return True

    repo = os.getenv("GITHUB_REPO")
    url = f"{_GITHUB_API}/repos/{repo}/contents/{_GITHUB_LOG_FILENAME}"
    try:
        resp = requests.get(url, headers=_github_headers(), params={"ref": _DATA_BRANCH}, timeout=15)
        if resp.status_code == 404:
            logger.info("[5B-GITHUB] No trade log on data branch yet — starting fresh")
            return False
        resp.raise_for_status()
        content = resp.json().get("content", "")
        decoded = base64.b64decode(content).decode("utf-8")
        with open(local_path, "w") as f:
            f.write(decoded)
        trades = len(json.loads(decoded).get("trade_history", []))
        logger.info(f"[5B-GITHUB] Trade log restored — {trades} trades recovered")
        return True
    except Exception as e:
        logger.warning(f"[5B-GITHUB] Fetch failed: {e}")
        return False


def github_push_5b_log(local_path: str) -> bool:
    """Push schematics_5b_trade_log.json to the data branch on GitHub."""
    if not _github_configured():
        return False
    if not os.path.exists(local_path):
        return False

    repo = os.getenv("GITHUB_REPO")
    url = f"{_GITHUB_API}/repos/{repo}/contents/{_GITHUB_LOG_FILENAME}"
    try:
        with open(local_path, "r") as f:
            raw = f.read()
        encoded = base64.b64encode(raw.encode("utf-8")).decode("utf-8")

        sha = None
        get_resp = requests.get(url, headers=_github_headers(), params={"ref": _DATA_BRANCH}, timeout=10)
        if get_resp.status_code == 200:
            sha = get_resp.json().get("sha")

        payload = {
            "message": "chore: 5B trade log sync",
            "content": encoded,
            "branch": _DATA_BRANCH,
        }
        if sha:
            payload["sha"] = sha

        put_resp = requests.put(url, headers=_github_headers(), json=payload, timeout=15)
        put_resp.raise_for_status()
        logger.info("[5B-GITHUB] Trade log pushed to data branch")
        return True
    except Exception as e:
        logger.warning(f"[5B-GITHUB] Push failed: {e}")
        return False


# ================================================================
# TELEGRAM NOTIFICATIONS (uses TELEGRAM_CHAT_ID_3)
# ================================================================
_TELEGRAM_API = "https://api.telegram.org"


def _telegram_5b_send(text: str) -> bool:
    """Fire-and-forget Telegram message to TELEGRAM_CHAT_ID_3."""
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID_3")
    if not token or not chat_id:
        return False

    url = f"{_TELEGRAM_API}/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text, "parse_mode": "HTML"}

    def _send():
        try:
            requests.post(url, json=payload, timeout=5)
        except Exception as e:
            logger.error(f"[5B-TELEGRAM] Send error: {e}")

    threading.Thread(target=_send, daemon=True).start()
    return True


def _notify_5b_entry(trade: Dict) -> None:
    direction = trade.get("direction", "?")
    arrow = "BUY" if direction == "bullish" else "SELL"
    text = (
        f"<b>5B {arrow} — Trade Entered</b>\n"
        f"Entry: ${trade.get('entry_price', 0):,.2f}\n"
        f"Stop: ${trade.get('stop_price', 0):,.2f}\n"
        f"Target: ${trade.get('target_price', 0):,.2f}\n"
        f"R:R: {trade.get('rr', 0):.1f} | Score: {trade.get('entry_score', 0)}/100\n"
    )
    _telegram_5b_send(text)


def _notify_5b_exit(trade: Dict) -> None:
    result = "WIN" if trade.get("is_win") else "LOSS"
    pnl = trade.get("pnl_dollars", 0)
    sign = "+" if pnl >= 0 else ""
    text = (
        f"<b>5B Trade Closed — {result}</b>\n"
        f"Entry: ${trade.get('entry_price', 0):,.2f}\n"
        f"Exit: ${trade.get('exit_price', 0):,.2f}\n"
        f"P&L: {sign}${pnl:,.2f} ({sign}{trade.get('pnl_pct', 0):.2f}%)\n"
    )
    _telegram_5b_send(text)


# ================================================================
# TRADE STATE MANAGER
# ================================================================
class Schematics5BTradeState:
    """Persistent trade state — no reward/learning fields."""

    def __init__(self):
        self.balance = STARTING_BALANCE
        self.starting_balance = STARTING_BALANCE
        self.current_trade: Optional[Dict] = None
        self.trade_history: List[Dict] = []
        self.total_wins = 0
        self.total_losses = 0
        self.last_scan_time: Optional[str] = None
        self.last_scan_action: Optional[str] = None
        self.last_error: Optional[str] = None
        self._load()

    def _load(self):
        try:
            github_fetch_5b_log(TRADE_LOG_PATH)
        except Exception as e:
            logger.warning(f"[5B-STATE] GitHub restore skipped: {e}")

        try:
            if os.path.exists(TRADE_LOG_PATH):
                with open(TRADE_LOG_PATH, "r") as f:
                    data = json.load(f)
                self.balance = data.get("balance", STARTING_BALANCE)
                self.starting_balance = data.get("starting_balance", STARTING_BALANCE)
                self.current_trade = data.get("current_trade")
                self.trade_history = data.get("trade_history", [])
                self.total_wins = data.get("total_wins", 0)
                self.total_losses = data.get("total_losses", 0)
                self.last_scan_time = data.get("last_scan_time")
                self.last_scan_action = data.get("last_scan_action")
                logger.info(f"[5B-STATE] Loaded — balance=${self.balance:.2f}, trades={len(self.trade_history)}")
        except Exception as e:
            logger.warning(f"[5B-STATE] Could not load trade state: {e}")

    def save(self):
        try:
            # Backup before write
            if os.path.exists(TRADE_LOG_PATH):
                shutil.copy2(TRADE_LOG_PATH, TRADE_LOG_BACKUP_PATH)

            data = {
                "balance": round(self.balance, 2),
                "starting_balance": self.starting_balance,
                "current_trade": self.current_trade,
                "trade_history": self.trade_history,
                "total_wins": self.total_wins,
                "total_losses": self.total_losses,
                "last_scan_time": self.last_scan_time,
                "last_scan_action": self.last_scan_action,
                "last_error": self.last_error,
            }
            with open(TRADE_LOG_PATH, "w") as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"[5B-STATE] Failed to save: {e}")

    def snapshot(self) -> Dict:
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


# ================================================================
# DETERMINISTIC EVALUATOR (fixed threshold, no learning)
# ================================================================
class Schematics5BEvaluator:
    """
    Scores TCT schematics for trade entry quality.
    Fixed 50-point threshold — never adapts.
    """

    def evaluate_schematic(self, schematic: Dict, htf_bias: str, current_price: float,
                           total_candles: int = 200, max_stale_candles: int = 5) -> Dict:
        score = 0
        reasons = []

        direction = schematic.get("direction", "unknown")
        model = schematic.get("model", schematic.get("schematic_type", "unknown"))
        is_confirmed = schematic.get("is_confirmed", False)
        tap3 = schematic.get("tap3", {})

        # Live R:R calculation
        stop_price = (schematic.get("stop_loss") or {}).get("price")
        target_price_val = (schematic.get("target") or {}).get("price")
        if stop_price and target_price_val and current_price > 0:
            if direction == "bullish":
                live_risk = current_price - stop_price
                live_reward = target_price_val - current_price
            else:
                live_risk = stop_price - current_price
                live_reward = current_price - target_price_val
            rr = (live_reward / live_risk) if live_risk > 0 else 0
        else:
            rr = schematic.get("risk_reward", 0) or 0

        fail = {"score": 0, "direction": direction, "model": model, "rr": rr,
                "required_score": ENTRY_THRESHOLD, "pass": False}

        # Gate: must be confirmed
        if not is_confirmed:
            return {**fail, "reasons": ["No BOS confirmation"]}

        # Gate: stale BOS
        bos = schematic.get("bos_confirmation") or {}
        bos_idx = bos.get("bos_idx")
        if bos_idx is not None and bos_idx < total_candles - max_stale_candles:
            return {**fail, "reasons": [f"Stale BOS: {total_candles - bos_idx} candles ago (max {max_stale_candles})"]}

        # Gate: model structure validation
        if "Model_1_from_M2_failure" not in model:
            if "Model_1" in model:
                if tap3.get("type") != "tap3_model1":
                    return {**fail, "reasons": ["Model 1: Tap3 must be deeper deviation"]}
            elif "Model_2" in model:
                if direction == "bullish" and not tap3.get("is_higher_low"):
                    return {**fail, "reasons": ["Model 2: Tap3 must be higher low for accumulation"]}
                if direction == "bearish" and not tap3.get("is_lower_high"):
                    return {**fail, "reasons": ["Model 2: Tap3 must be lower high for distribution"]}

        # Gate: quality floor
        quality_score = schematic.get("quality_score", 0.0)
        if quality_score < 0.70:
            return {**fail, "reasons": [f"Quality too low ({quality_score:.2f} < 0.70)"]}

        # ---- Scoring ----

        # BOS confirmed = base
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
            return {**fail, "reasons": [f"R:R too low ({rr:.1f})"]}

        # HTF alignment
        if direction == "bullish" and htf_bias == "bullish":
            score += 20
            reasons.append("HTF bias aligned (bullish)")
        elif direction == "bearish" and htf_bias == "bearish":
            score += 20
            reasons.append("HTF bias aligned (bearish)")
        elif htf_bias == "neutral":
            return {**fail, "reasons": ["HTF bias neutral — no directional clarity"]}
        else:
            score -= 10
            reasons.append(f"HTF bias conflict ({htf_bias} vs {direction})")

        # Quality bonus
        quality_bonus = round(quality_score * 15)
        score += quality_bonus
        reasons.append(f"Quality {quality_score:.2f} (+{quality_bonus})")

        # Model type (fixed defaults — no learning)
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
# SHARED LTF BOS REFINEMENT HELPER
# Extracted as a module-level function so the schematics-5A page can reuse it
# without duplicating the cascade logic.
# ================================================================

def refine_schematic_bos_with_ltf(
    schematic: Dict,
    ltf_dfs: Dict[str, Optional[pd.DataFrame]],
    *,
    label: str = "LTF",
) -> Dict:
    """
    Refine the BOS entry price using lower-timeframe data.

    TCT methodology: after identifying tap2/tap3, cascade from the highest TF
    down to the lowest to find the EARLIEST confirmed BOS.  A lower-TF BOS
    closes sooner in real time, giving an entry closer to tap3 and therefore
    a smaller stop distance and better R:R.

    The MTF schematic's bos_confirmation already contains the reference prices
    (highest/lowest point between tap2 and tap3) so we reuse them directly —
    no need to re-scan the tap2-tap3 window on the lower TF.

    Writes ltf_refined metadata into bos_confirmation and updates entry.price.
    Returns a shallow copy of the schematic (original is never mutated).

    Args:
        schematic: A confirmed TCT schematic dict.
        ltf_dfs:   Mapping of timeframe label → DataFrame (e.g. {"5m": df, "1m": df}).
        label:     Log prefix for tracing (e.g. "5A-LTF" or "5B-LTF").
    """
    bos_conf = schematic.get("bos_confirmation")
    if not bos_conf:
        return schematic

    tap3 = schematic.get("tap3") or {}
    tap3_time_str = tap3.get("time", "")
    if not tap3_time_str:
        return schematic

    try:
        tap3_time = pd.Timestamp(tap3_time_str)
        if tap3_time.tzinfo is None:
            tap3_time = tap3_time.tz_localize("UTC")
    except Exception:
        return schematic

    direction = schematic.get("direction")
    equilibrium = (schematic.get("range") or {}).get("equilibrium")
    tap3_price = tap3.get("price")

    # Reference prices already computed by MTF BOS detection — reuse them.
    if direction == "bullish":
        ref_high = (bos_conf.get("highest_point_between_tabs") or {}).get("price")
        ref_low = tap3_price
    elif direction == "bearish":
        ref_low = (bos_conf.get("lowest_point_between_tabs") or {}).get("price")
        ref_high = tap3_price
    else:
        return schematic

    if not ref_high or not ref_low:
        return schematic

    best_bos = None
    best_ltf = None

    # LTF_BOS_TIMEFRAMES is ordered highest → lowest (e.g. ["5m", "1m"]).
    # We keep overwriting best_bos so the LOWEST TF that confirms wins.
    for ltf in LTF_BOS_TIMEFRAMES:
        ltf_df = ltf_dfs.get(ltf)
        if ltf_df is None or len(ltf_df) < 20:
            continue

        ltf_df_reset = ltf_df.reset_index(drop=True)

        # Find the first LTF candle whose open_time is at or after tap3's bar open.
        # BOS is only valid once tap3's deviation has completed.
        after_tap3 = ltf_df_reset["open_time"] >= tap3_time
        if not after_tap3.any():
            logger.debug(
                f"[{label}] {ltf}: tap3_time {tap3_time} predates all {len(ltf_df)} candles — skipping"
            )
            continue

        tap3_ltf_pos = int(after_tap3.idxmax())

        try:
            detector = TCTSchematicDetector(ltf_df_reset)
            if direction == "bullish":
                bos = detector._find_bullish_bos(
                    tap3_ltf_pos, ref_high, ref_low, equilibrium=equilibrium
                )
            else:
                bos = detector._find_bearish_bos(
                    tap3_ltf_pos, ref_low, ref_high, equilibrium=equilibrium
                )
        except Exception as e:
            logger.warning(f"[{label}] BOS detection error on {ltf}: {e}")
            continue

        if bos:
            best_bos = bos
            best_ltf = ltf
            # Do NOT break — continue to lower TFs so the lowest confirmed TF wins.

    if not best_bos:
        return schematic

    mtf_bos_price = bos_conf.get("bos_price")
    mtf_bos_str = f"{mtf_bos_price:.2f}" if mtf_bos_price else "?"
    logger.debug(
        f"[{label}] Refined BOS on {best_ltf}: entry={best_bos['price']:.2f} "
        f"(MTF BOS was {mtf_bos_str})"
    )

    refined = {**schematic}
    refined["bos_confirmation"] = {
        **bos_conf,
        "ltf_refined": True,
        "ltf_timeframe": best_ltf,
        "ltf_bos_price": best_bos["price"],
        "ltf_bos_idx": best_bos["idx"],
    }
    refined["entry"] = {
        **(schematic.get("entry") or {}),
        "price": best_bos["price"],
        "type": f"LTF_BOS_{best_ltf}",
        "description": f"Enter on {best_ltf} BOS — earliest confirmation, best R:R",
    }
    return refined


# ================================================================
# HTF UPGRADE CASCADE HELPER
# ================================================================

def _find_htf_upgrade(
    schematic: Dict,
    this_tf: str,
    all_schematics_by_tf: Dict[str, List[Dict]],
) -> Tuple[Dict, str]:
    """
    Walk UP the TF hierarchy from this_tf, looking for a same-direction
    confirmed schematic at a higher timeframe whose range bracket contains
    this schematic's tap3 price.

    TCT principle: a confirmed model at a higher TF is higher probability
    because it represents a larger structural imbalance.  When the same
    pattern is visible on multiple TFs, prefer the highest-TF version.

    The search stops at the first (= highest) TF that satisfies both:
      - same direction (bullish / bearish)
      - range [low, high] contains tap3 price of the *original* schematic

    Returns (schematic, effective_tf).  If no upgrade found, returns the
    original (schematic, this_tf).
    """
    tap3_price = (schematic.get("tap3") or {}).get("price")
    direction = schematic.get("direction")
    if not tap3_price or not direction or this_tf not in _TF_HIERARCHY:
        return schematic, this_tf

    this_idx = _TF_HIERARCHY.index(this_tf)
    if this_idx == 0:
        return schematic, this_tf  # already at highest TF

    # _TF_HIERARCHY[:this_idx] = all TFs above this_tf, already in high→low order.
    # Iterating in that order means we return the HIGHEST match immediately.
    for higher_tf in _TF_HIERARCHY[:this_idx]:
        higher_schematics = all_schematics_by_tf.get(higher_tf, [])
        for s in higher_schematics:
            if not isinstance(s, dict) or not s.get("is_confirmed"):
                continue
            if s.get("direction") != direction:
                continue
            r = s.get("range") or {}
            r_high = r.get("high")
            r_low = r.get("low")
            if r_high is not None and r_low is not None and r_low <= tap3_price <= r_high:
                logger.info(
                    f"[5B-HTF] Upgraded {this_tf}→{higher_tf}: "
                    f"tap3={tap3_price:.2f} ∈ [{r_low:.2f},{r_high:.2f}] "
                    f"({direction})"
                )
                return s, higher_tf

    return schematic, this_tf


# ================================================================
# MAIN TRADING ENGINE
# ================================================================
class Schematics5BTrader:
    """
    Deterministic TCT trading engine — BTCUSDT only, fixed threshold,
    no learning/reward system.
    """

    # 1d candle → bias stable for 24h; neutral re-checks every hour
    _HTF_CACHE_TTL = {"bullish": 24 * 3600, "bearish": 24 * 3600, "neutral": 3600}

    def __init__(self):
        self.state = Schematics5BTradeState()
        self.evaluator = Schematics5BEvaluator()
        self.last_debug: Dict = {}
        self._htf_bias_cache: str = "neutral"
        self._htf_bias_expiry: float = 0.0

    def scan_and_trade(self) -> Dict:
        """Main cycle: fetch, detect, evaluate, trade."""
        cycle_result = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": "none",
            "details": {},
        }

        try:
            # 1. Current price
            df_price = fetch_candles_sync(SYMBOL, "1m", 10)
            if df_price is None or len(df_price) == 0:
                self.state.last_error = "Could not fetch price data"
                self.state.save()
                cycle_result["action"] = "error"
                cycle_result["details"] = {"error": "Could not fetch price data"}
                return cycle_result

            current_price = float(df_price.iloc[-1]["close"])

            # 2. Manage open trade
            if self.state.current_trade:
                result = self._manage_open_trade(current_price)
                cycle_result["action"] = result.get("action", "manage")
                cycle_result["details"] = result
                self.state.last_scan_time = cycle_result["timestamp"]
                self.state.last_scan_action = cycle_result["action"]
                self.state.save()
                return cycle_result

            # 3. HTF bias gate (with cache)
            now_ts = time.time()
            if now_ts < self._htf_bias_expiry:
                htf_bias = self._htf_bias_cache
                htf_debug = {"status": "cached", "htf_bias": htf_bias,
                             "expires_in_s": round(self._htf_bias_expiry - now_ts)}
            else:
                htf_bias = "neutral"
                htf_debug = {"status": "not_fetched"}
                try:
                    df_htf = fetch_candles_sync(SYMBOL, HTF_TIMEFRAME, 200)
                    if df_htf is not None and len(df_htf) >= 50:
                        htf_schematics = detect_tct_schematics(df_htf, [])
                        htf_acc = [s for s in htf_schematics.get("accumulation_schematics", [])
                                   if isinstance(s, dict) and s.get("is_confirmed")]
                        htf_dist = [s for s in htf_schematics.get("distribution_schematics", [])
                                    if isinstance(s, dict) and s.get("is_confirmed")]
                        if htf_acc and not htf_dist:
                            htf_bias = "bullish"
                        elif htf_dist and not htf_acc:
                            htf_bias = "bearish"
                        htf_debug = {"status": "scanned", "candles": len(df_htf),
                                     "confirmed_acc": len(htf_acc), "confirmed_dist": len(htf_dist),
                                     "htf_bias": htf_bias}
                    else:
                        htf_debug = {"status": "insufficient_data",
                                     "candles": 0 if df_htf is None else len(df_htf)}
                except Exception as e:
                    logger.warning(f"[5B] HTF gate error: {e}", exc_info=True)
                    htf_debug = {"status": "error", "error": str(e), "fetch_error": True}

                self._htf_bias_cache = htf_bias
                self._htf_bias_expiry = now_ts + self._HTF_CACHE_TTL.get(htf_bias, 900)

            # 4. Parallel fetch — MTF at increased limits + LTF for BOS cascade
            best_setup = None
            best_score = 0
            best_tf = None
            # HTF_TIMEFRAME bias result is always shown first in the debug output
            all_tf_results = {HTF_TIMEFRAME: htf_debug}

            mtf_dfs: Dict[str, Optional[pd.DataFrame]] = {}
            ltf_dfs: Dict[str, Optional[pd.DataFrame]] = {}
            all_tfs = {
                **{tf: _MTF_CANDLE_LIMITS.get(tf, 300) for tf in MTF_TIMEFRAMES},
                **{ltf: _LTF_CANDLE_LIMITS[ltf] for ltf in LTF_BOS_TIMEFRAMES},
            }
            with ThreadPoolExecutor(max_workers=len(all_tfs)) as ex:
                futures = {ex.submit(fetch_candles_sync, SYMBOL, tf, lim): tf
                           for tf, lim in all_tfs.items()}
                for future in as_completed(futures):
                    tf = futures[future]
                    try:
                        df_result = future.result()
                        if tf in MTF_TIMEFRAMES:
                            mtf_dfs[tf] = df_result
                        else:
                            ltf_dfs[tf] = df_result
                    except Exception as e:
                        logger.warning(f"[5B] Fetch failed for {tf}: {e}")
                        if tf in MTF_TIMEFRAMES:
                            mtf_dfs[tf] = None
                        else:
                            ltf_dfs[tf] = None

            # Phase A: collect all detected schematics per TF (needed for HTF cascade)
            all_schematics_by_tf: Dict[str, List[Dict]] = {}
            for tf in MTF_TIMEFRAMES:
                df = mtf_dfs.get(tf)
                if df is None or len(df) < 50:
                    all_schematics_by_tf[tf] = []
                    all_tf_results[tf] = {
                        "status": "insufficient_data",
                        "candles": 0 if df is None else len(df),
                        "fetch_error": df is None,
                    }
                    continue
                try:
                    result = detect_tct_schematics(df, [])
                    all_schematics_by_tf[tf] = (
                        result.get("accumulation_schematics", [])
                        + result.get("distribution_schematics", [])
                    )
                except Exception as e:
                    logger.warning(f"[5B] Detection error on {tf}: {e}", exc_info=True)
                    all_schematics_by_tf[tf] = []
                    all_tf_results[tf] = {"status": "error", "error": str(e)}

            # Phase B: evaluate with HTF cascade.
            # Walk MTF_TIMEFRAMES in reverse (lowest → highest TF) so the cascade
            # check finds upgrades from the smaller-granularity schematics first.
            for tf in reversed(MTF_TIMEFRAMES):
                df = mtf_dfs.get(tf)
                if tf not in all_schematics_by_tf or "status" in all_tf_results.get(tf, {}):
                    continue  # already recorded as error/insufficient above

                all_sch = all_schematics_by_tf[tf]
                tf_evals = []
                htf_upgraded_count = 0

                for s in all_sch:
                    if not isinstance(s, dict):
                        continue
                    s_dir = s.get("direction", "unknown")
                    if htf_bias == "bullish" and s_dir == "bearish":
                        continue
                    if htf_bias == "bearish" and s_dir == "bullish":
                        continue

                    # LTF BOS refinement: sharpen the entry price.
                    if s.get("is_confirmed"):
                        s = self._refine_schematic_bos_with_ltf(s, ltf_dfs)

                    # HTF upgrade cascade: prefer the highest TF that covers this pattern.
                    effective_tf = tf
                    if s.get("is_confirmed"):
                        s, effective_tf = _find_htf_upgrade(s, tf, all_schematics_by_tf)
                        if effective_tf != tf:
                            htf_upgraded_count += 1

                    # Evaluate against the *effective* TF's candle count and stale limit.
                    eff_df = mtf_dfs.get(effective_tf) or df
                    eval_result = self.evaluator.evaluate_schematic(
                        s, htf_bias, current_price,
                        total_candles=len(eff_df),
                        max_stale_candles=_MAX_STALE.get(effective_tf, 5),
                    )
                    eval_result["source_tf"] = tf
                    eval_result["effective_tf"] = effective_tf
                    if effective_tf != tf:
                        eval_result["htf_upgraded"] = True

                    tf_evals.append(eval_result)
                    if eval_result["pass"] and eval_result["score"] > best_score:
                        best_score = eval_result["score"]
                        best_setup = (s, eval_result)
                        best_tf = effective_tf

                total_candles = len(df) if df is not None else 0
                all_tf_results[tf] = {
                    "status": "scanned",
                    "candles": total_candles,
                    "schematics_found": len(all_sch),
                    "confirmed": sum(
                        1 for s in all_sch if isinstance(s, dict) and s.get("is_confirmed")
                    ),
                    "htf_upgraded": htf_upgraded_count,
                    "htf_bias": htf_bias,
                    "best_score": max((e["score"] for e in tf_evals), default=0),
                    "evaluations": tf_evals[:10],
                }

            # Store debug
            self.last_debug = {
                "timestamp": cycle_result["timestamp"],
                "current_price": current_price,
                "timeframes": all_tf_results,
                "best_tf": best_tf,
                "best_score": best_score,
                "htf_cascade_active": True,
            }

            # 5. Enter trade if qualifying setup found
            if best_setup:
                schematic, evaluation = best_setup
                entry_info = schematic.get("entry", {})
                candidate_price = entry_info.get("price", current_price)

                if self._is_duplicate_setup(candidate_price, evaluation["direction"]):
                    cycle_result["action"] = "duplicate_setup_skipped"
                    cycle_result["details"] = {
                        "price": current_price,
                        "skipped_entry": candidate_price,
                        "reason": "Same setup as recent trade — cooldown active",
                    }
                else:
                    trade = self._enter_trade(schematic, evaluation, current_price, htf_bias)
                    trade["timeframe"] = best_tf
                    cycle_result["action"] = "trade_entered"
                    cycle_result["details"] = trade
            else:
                cycle_result["action"] = "no_qualifying_setups"
                cycle_result["details"] = {"price": current_price, "timeframes_scanned": all_tf_results}

            self.state.last_scan_time = cycle_result["timestamp"]
            self.state.last_scan_action = cycle_result["action"]
            self.state.last_error = None
            self.state.save()

        except Exception as e:
            logger.error(f"[5B] Scan error: {e}", exc_info=True)
            self.state.last_error = str(e)
            self.state.last_scan_action = "error"
            self.state.save()
            cycle_result["action"] = "error"
            cycle_result["details"] = {"error": str(e)}

        return cycle_result

    def _refine_schematic_bos_with_ltf(
        self, schematic: Dict, ltf_dfs: Dict[str, Optional[pd.DataFrame]]
    ) -> Dict:
        """Delegate to the shared module-level helper with a 5B-specific log label."""
        return refine_schematic_bos_with_ltf(schematic, ltf_dfs, label="5B-LTF")

    def _enter_trade(self, schematic: Dict, evaluation: Dict, current_price: float, htf_bias: str) -> Dict:
        direction = evaluation["direction"]
        stop_info = schematic.get("stop_loss", {})
        target_info = schematic.get("target", {})

        entry_price = current_price
        stop_price = stop_info.get("price")
        target_price = target_info.get("price")

        if not stop_price or not target_price:
            return {"error": "Missing stop or target price"}

        # Validate direction consistency
        if direction == "bearish":
            if target_price >= entry_price:
                return {"error": "Invalid short: target above entry"}
            if stop_price <= entry_price:
                return {"error": "Invalid short: stop below entry"}
        else:
            if target_price <= entry_price:
                return {"error": "Invalid long: target below entry"}
            if stop_price >= entry_price:
                return {"error": "Invalid long: stop above entry"}

        # Live R:R
        if direction == "bearish":
            actual_risk = stop_price - entry_price
            actual_reward = entry_price - target_price
        else:
            actual_risk = entry_price - stop_price
            actual_reward = target_price - entry_price

        actual_rr = actual_reward / actual_risk if actual_risk > 0 else 0
        if actual_rr < 1.0:
            return {"error": f"R:R too low at market ({actual_rr:.2f}:1)"}

        # Position sizing (1% risk)
        risk_amount = self.state.balance * (RISK_PER_TRADE_PCT / 100)
        if direction == "bullish":
            sl_pct = abs((entry_price - stop_price) / entry_price) * 100
        else:
            sl_pct = abs((stop_price - entry_price) / entry_price) * 100
        if sl_pct <= 0:
            sl_pct = 1.0

        position_size = calculate_position_size(risk_amount, sl_pct)
        margin = calculate_margin(position_size, DEFAULT_LEVERAGE)

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
            "rr": round(actual_rr, 2),
            "entry_score": evaluation["score"],
            "entry_reasons": evaluation["reasons"],
            "htf_bias": htf_bias,
            "liquidation_price": round(liq_price, 2),
            "liquidation_safe": safety["is_safe"],
            "opened_at": datetime.now(timezone.utc).isoformat(),
            "status": "open",
        }

        self.state.current_trade = trade
        self.state.save()
        logger.info(f"[5B] Entered {direction} @ {entry_price} | SL={stop_price} | TP={target_price} | Score={evaluation['score']}")
        _notify_5b_entry(trade)
        return trade

    def _manage_open_trade(self, current_price: float) -> Dict:
        trade = self.state.current_trade
        if not trade:
            return {"action": "no_trade"}

        direction = trade["direction"]
        entry_price = trade["entry_price"]
        stop_price = trade["stop_price"]
        target_price = trade["target_price"]

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
            return {"action": "holding", "pnl_pct": round(pnl_pct, 2),
                    "current_price": current_price, "direction": direction}

    def _close_trade(self, exit_price: float, reason: str) -> Dict:
        trade = self.state.current_trade
        if not trade:
            return {"action": "no_trade"}

        direction = trade["direction"]
        entry_price = trade["entry_price"]

        if direction == "bullish":
            pnl_pct = ((exit_price - entry_price) / entry_price) * 100
        else:
            pnl_pct = ((entry_price - exit_price) / entry_price) * 100

        is_win = reason == "target_hit"
        position_size = trade.get("position_size", 0)
        pnl_dollars = position_size * (pnl_pct / 100)
        self.state.balance += pnl_dollars

        if is_win:
            self.state.total_wins += 1
        else:
            self.state.total_losses += 1

        closed_trade = {
            **trade,
            "exit_price": round(exit_price, 2),
            "exit_reason": reason,
            "pnl_pct": round(pnl_pct, 2),
            "pnl_dollars": round(pnl_dollars, 2),
            "is_win": is_win,
            "closed_at": datetime.now(timezone.utc).isoformat(),
            "status": "closed",
            "balance_after": round(self.state.balance, 2),
        }

        self.state.trade_history.append(closed_trade)
        self.state.current_trade = None
        self.state.save()

        logger.info(f"[5B] Closed: {reason} | P&L={pnl_pct:.2f}% (${pnl_dollars:.2f}) | Balance=${self.state.balance:.2f}")
        _notify_5b_exit(closed_trade)

        return {"action": "trade_closed", "trade": closed_trade}

    def _is_duplicate_setup(self, entry_price: float, direction: str) -> bool:
        if not self.state.trade_history:
            return False
        now = datetime.now(timezone.utc)
        for t in reversed(self.state.trade_history[-5:]):
            if t.get("direction") != direction:
                continue
            t_entry = t.get("entry_price", 0)
            if t_entry > 0 and abs(t_entry - entry_price) / t_entry < DUPLICATE_PRICE_TOLERANCE:
                closed_at = t.get("closed_at")
                if closed_at:
                    try:
                        closed_time = datetime.fromisoformat(closed_at)
                        if closed_time.tzinfo is None:
                            closed_time = closed_time.replace(tzinfo=timezone.utc)
                        if (now - closed_time).total_seconds() < DUPLICATE_COOLDOWN_SECONDS:
                            return True
                    except (ValueError, TypeError):
                        pass
                else:
                    return True
        return False

    def force_close(self) -> Dict:
        if not self.state.current_trade:
            return {"action": "no_trade"}
        price = fetch_live_price(SYMBOL)
        if not price:
            return {"action": "error", "details": "Could not fetch live price"}
        return self._close_trade(price, "force_closed")


# ================================================================
# SINGLETON
# ================================================================
_trader_5b: Optional[Schematics5BTrader] = None


def get_5b_trader() -> Schematics5BTrader:
    global _trader_5b
    if _trader_5b is None:
        _trader_5b = Schematics5BTrader()
    return _trader_5b
