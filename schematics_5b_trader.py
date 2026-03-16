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
from pivot_cache import PivotCache
from concurrent.futures import ThreadPoolExecutor, as_completed
from trade_execution import (
    calculate_position_size,
    calculate_margin,
    calculate_liquidation_price,
    check_liquidation_safety,
)
from decision_tree_bridge import DecisionTreeEvaluator
from jack_tct_evaluator import JackTCTEvaluator

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
# Scan TFs — 30m and above only.  Lower TFs (15m, 5m, 1m) produce too many
# low-quality setups across 5 pairs; 30m+ keeps signal quality high.
# Note: 5m/1m are still fetched separately for LTF BOS *entry refinement*
# after a setup is confirmed on one of these TFs (see LTF_BOS_TIMEFRAMES).
MTF_TIMEFRAMES = ["1d", "4h", "1h", "30m"]
AUTO_SCAN_INTERVAL = int(os.getenv("SCHEMATICS_5B_SCAN_INTERVAL", "60"))
ENTRY_THRESHOLD = 60  # v2 pipeline: raised from 50 to 60

# TF hierarchy for the upward cascade (ordered highest → lowest).
# When a schematic is confirmed at a lower TF, we try to upgrade it to the
# highest TF that contains the same pattern in the same price area.
_TF_HIERARCHY = ["1W", "1d", "4h", "1h", "30m", "15m", "5m", "1m"]

# Candle limits per TF — sized so the BOS window never falls off the array
# even when tap3 is a recent candle.
_MTF_CANDLE_LIMITS = {
    "1d": 200,   # ~200 days
    "4h": 300,   # ~50 days
    "1h": 300,   # ~12.5 days
    "30m": 400,  # ~8.3 days
    "15m": 500,  # ~5.2 days
    "5m": 1000,  # ~83 h — same as LTF limit; cascade reuses these candles
    "1m": 1000,  # ~17 h — same as LTF limit
}

# Max candles since BOS before a setup is considered stale
_MAX_STALE: Dict[str, int] = {
    "1d": 2, "4h": 3, "1h": 3, "30m": 4, "15m": 5, "5m": 12, "1m": 30,
}

# LTF cascade: fetch lower-TF candles once per cycle for BOS refinement.
# Per TCT model, cascade from highest TF down to the lowest that confirms BOS;
# the earliest (lowest-TF) BOS gives the best entry and best R:R.
LTF_BOS_TIMEFRAMES = ["5m", "1m"]  # highest → lowest; we keep overwriting so lowest TF wins
_LTF_CANDLE_LIMITS = {"5m": 200, "1m": 100}  # 200×5m ≈ 17h; 100×1m ≈ 1.7h — sufficient for recent BOS

_DIR = os.path.dirname(os.path.abspath(__file__))
TRADE_LOG_PATH = os.path.join(_DIR, "schematics_5b_trade_log.json")
TRADE_LOG_BACKUP_PATH = os.path.join(_DIR, "schematics_5b_trade_log_backup.json")

# Deduplication
DUPLICATE_COOLDOWN_SECONDS = 300
DUPLICATE_PRICE_TOLERANCE = 0.002


# ================================================================
# SHARED HELPERS
# ================================================================

def _build_range_data_for_bos(rng: Dict) -> Dict:
    """
    Build the range_data dict needed for BOS validation from a range/
    schematic dict that may use either "high"/"low" or "range_high"/
    "range_low" key conventions.
    """
    return {
        "range_high": rng.get("high") or rng.get("range_high"),
        "range_low": rng.get("low") or rng.get("range_low"),
        "range_size": rng.get("size") or rng.get("range_size", 0),
        "equilibrium": rng.get("equilibrium", 0),
        "range_high_idx": rng.get("range_high_idx", 0),
        "range_low_idx": rng.get("range_low_idx", 0),
    }


def _get_entry_session_context(timestamp: Optional[str] = None) -> Dict:
    """
    Return session context for an entry timestamp.

    Attempts to import session_manipulation and apply session multiplier.
    Falls back to a neutral default on ImportError or runtime errors.
    """
    default = {"session": None, "boost_applied": False, "multiplier": 1.0}

    try:
        from session_manipulation import apply_session_multiplier
    except (ImportError, ModuleNotFoundError):
        return default

    try:
        return apply_session_multiplier(timestamp)
    except (TypeError, ValueError):
        return default


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
    """Fire-and-forget Telegram message.
    Uses TELEGRAM_CHAT_ID_3 if set; falls back to TELEGRAM_CHAT_ID."""
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID_3") or os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        logger.warning("[5B-TELEGRAM] Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID — notifications disabled")
        return False

    url = f"{_TELEGRAM_API}/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text, "parse_mode": "HTML"}

    def _send():
        try:
            resp = requests.post(url, json=payload, timeout=5)
            if resp.status_code != 200:
                logger.error(f"[5B-TELEGRAM] API error {resp.status_code}: {resp.text}")
            else:
                logger.info(f"[5B-TELEGRAM] Sent OK to chat_id={chat_id[:6]}…")
        except Exception as e:
            logger.error(f"[5B-TELEGRAM] Send error: {e}")

    threading.Thread(target=_send, daemon=True).start()
    return True


def _notify_5b_entry(trade: Dict) -> None:
    direction = trade.get("direction", "?")
    symbol = trade.get("symbol", "BTCUSDT")
    timeframe = (trade.get("timeframe") or "?").upper()
    arrow = "BUY" if direction == "bullish" else "SELL"
    text = (
        f"<b>5B {arrow} — {symbol} | {timeframe}</b>\n"
        f"Entry: ${trade.get('entry_price', 0):,.2f}\n"
        f"Stop: ${trade.get('stop_price', 0):,.2f}\n"
        f"Target: ${trade.get('target_price', 0):,.2f}\n"
        f"R:R: {trade.get('rr', 0):.1f} | Score: {trade.get('entry_score', 0)}/100\n"
    )
    _telegram_5b_send(text)


def _notify_5b_tp1(trade: Dict) -> None:
    """Telegram alert when TP1 (halfway) is hit and half position closed."""
    direction = trade.get("direction", "?")
    symbol = trade.get("symbol", "BTCUSDT")
    timeframe = (trade.get("timeframe") or "?").upper()
    arrow = "BUY" if direction == "bullish" else "SELL"
    tp1_pnl = trade.get("tp1_pnl_dollars", 0)
    sign = "+" if tp1_pnl >= 0 else ""
    text = (
        f"<b>5B {arrow} — {symbol} | {timeframe} — TP1 Hit (½ Closed)</b>\n"
        f"Entry: ${trade.get('entry_price', 0):,.2f}\n"
        f"TP1: ${trade.get('tp1_exit_price', 0):,.2f}\n"
        f"½ P&L: {sign}${tp1_pnl:,.2f} ({sign}{trade.get('tp1_pnl_pct', 0):.2f}%)\n"
        f"Stop → Break-Even @ ${trade.get('entry_price', 0):,.2f}\n"
        f"Remaining ½ targets ${trade.get('target_price', 0):,.2f}\n"
    )
    _telegram_5b_send(text)


def _notify_5b_exit(trade: Dict) -> None:
    result = "WIN" if trade.get("is_win") else "LOSS"
    symbol = trade.get("symbol", "BTCUSDT")
    timeframe = (trade.get("timeframe") or "?").upper()
    pnl = trade.get("pnl_dollars", 0)
    sign = "+" if pnl >= 0 else ""
    if trade.get("tp1_hit"):
        tp1_pnl = trade.get("tp1_pnl_dollars", 0)
        tp1_sign = "+" if tp1_pnl >= 0 else ""
        total_pnl = pnl + tp1_pnl
        total_sign = "+" if total_pnl >= 0 else ""
        text = (
            f"<b>5B Trade Closed — {symbol} | {timeframe} — {result}</b>\n"
            f"Entry: ${trade.get('entry_price', 0):,.2f}\n"
            f"Exit: ${trade.get('exit_price', 0):,.2f}\n"
            f"TP1 P&L (½): {tp1_sign}${tp1_pnl:,.2f}\n"
            f"Final P&L (½): {sign}${pnl:,.2f} ({sign}{trade.get('pnl_pct', 0):.2f}%)\n"
            f"Total P&L: {total_sign}${total_pnl:,.2f}\n"
        )
    else:
        text = (
            f"<b>5B Trade Closed — {symbol} | {timeframe} — {result}</b>\n"
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
        self.trading_mode: str = "claude"  # "claude" | "jack"
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
                self.trading_mode = data.get("trading_mode", "claude")
                # Validate current_trade — discard if essential fields are missing/zero
                if self.current_trade:
                    ep = self.current_trade.get("entry_price", 0)
                    sp = self.current_trade.get("stop_price", 0)
                    tp = self.current_trade.get("target_price", 0)
                    if not ep or not sp or not tp:
                        logger.error(f"[5B-STATE] Discarding corrupt current_trade on load: entry={ep}, stop={sp}, target={tp}")
                        self.current_trade = None
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
                "trading_mode": self.trading_mode,
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
# DECISION TREE DATA EXTRACTOR
# Pulls the key pass/fail fields from a schematic for the debug UI.
# ================================================================

def _build_dt_data(schematic: Dict, htf_bias: str) -> Dict:
    """Extract decision-tree-relevant data from the best schematic for a TF.

    Returns a flat dict consumed by the schematics-5B UI debug panel.
    Includes both legacy keys (backward compat) and v2 phase fields.
    """
    enh = schematic.get("lecture_5b_enhancements") or {}
    rng = schematic.get("range") or {}
    tap_spacing = enh.get("tap_spacing") or {}
    range_quality = enh.get("range_quality") or {}
    sd_check = enh.get("supply_demand_check") or {}
    trendline = enh.get("trendline_liquidity") or {}
    overlapping = enh.get("overlapping_structure") or {}

    direction = schematic.get("direction", "unknown")

    # Range size as a % of mid-price
    rng_size = rng.get("size", 0)
    eq = rng.get("equilibrium") or ((rng.get("high", 0) + rng.get("low", 0)) / 2) or 1
    rng_size_pct = round((rng_size / eq) * 100, 2) if eq else None

    entry_p = (schematic.get("entry") or {}).get("price")
    sl_p = (schematic.get("stop_loss") or {}).get("price")
    tgt_p = (schematic.get("target") or {}).get("price")

    # Tap time displacement (v2 Phase 2)
    tap1 = schematic.get("tap1") or {}
    tap2 = schematic.get("tap2") or {}
    tap3 = schematic.get("tap3") or {}
    t1_idx = tap1.get("idx", 0)
    t2_idx = tap2.get("idx", 0)
    t3_idx = tap3.get("idx", 0)
    t12_gap = abs(t2_idx - t1_idx) if t1_idx and t2_idx else None
    t23_gap = abs(t3_idx - t2_idx) if t2_idx and t3_idx else None

    # Model type for v2 Phase 3
    model_str = schematic.get("model", "")
    if "Model_1" in model_str:
        model_type = "Model_1"
    elif "Model_2" in model_str:
        model_type = "Model_2"
    else:
        model_type = model_str or "unknown"

    return {
        # --- Ranges (v2 Phase 2) ---
        "range_high": rng.get("high"),
        "range_low": rng.get("low"),
        "range_size_pct": rng_size_pct,
        "range_horizontal": range_quality.get("is_horizontal", False),
        "range_has_clean_pivots": range_quality.get("has_clean_pivots", False),
        "range_quality_score": range_quality.get("quality_score", 0),
        "range_quality_factors": range_quality.get("quality_factors", []),
        "six_candle_valid": schematic.get("six_candle_valid", False),
        "time_displacement_t12": t12_gap,
        "time_displacement_t23": t23_gap,
        # --- Market Structure (v2 Phase 1/7) ---
        "direction": direction,
        "htf_bias": htf_bias,
        "htf_aligned": direction == htf_bias or htf_bias == "neutral",
        "bos_confirmed": schematic.get("is_confirmed", False),
        "model": schematic.get("model", ""),
        "model_type": model_type,
        # --- POI / Supply & Demand (v2 Phase 6) ---
        "sd_conflict": sd_check.get("has_conflict", False),
        "target_clear": not sd_check.get("opposing_zone_blocks_target", False),
        # --- Liquidity (v2 Phase 4) ---
        "trendline_confluence": trendline.get("provides_confluence", False),
        "tap_spacing_valid": tap_spacing.get("spacing_valid", False),
        "spacing_ratio": tap_spacing.get("spacing_ratio"),
        "tap1_to_tap2": tap_spacing.get("tap1_to_tap2_candles") or t12_gap,
        "tap2_to_tap3": tap_spacing.get("tap2_to_tap3_candles") or t23_gap,
        "tap_is_horizontal": tap_spacing.get("is_horizontal", False),
        # --- Tap Structure (v2 Phase 3) ---
        "tap1_price": tap1.get("price"),
        "tap2_price": tap2.get("price"),
        "tap3_price": tap3.get("price"),
        # --- Trade Setup (v2 Phase 8) ---
        "rr": schematic.get("risk_reward"),
        "rr_meets_minimum": enh.get("meets_minimum_rr", False) or (schematic.get("risk_reward", 0) or 0) >= 1.5,
        "quality_score": schematic.get("quality_score", 0),
        "entry": entry_p,
        "sl": sl_p,
        "target": tgt_p,
        # --- 5B Schematics ---
        "overlapping_structure": overlapping.get("has_overlapping_structure", False) if isinstance(overlapping, dict) else False,
        "optimized_entry_price": (enh.get("optimized_entry") or {}).get("entry") if isinstance(enh.get("optimized_entry"), dict) else None,
        "domino_levels": (overlapping.get("domino_levels") or []) if isinstance(overlapping, dict) else [],
        # --- Advanced TCT (Lecture 6) ---
        "schematic_conversion": bool(schematic.get("schematic_conversion")),
        "multi_tf_valid": bool(schematic.get("multi_tf_validity")),
        "wov_in_wov": bool(schematic.get("wov_in_wov")),
        "m1_to_m2_flow": bool(schematic.get("m1_to_m2_flow")),
    }


# ================================================================
# DETERMINISTIC EVALUATOR (fixed threshold, no learning)
# ================================================================
class Schematics5BEvaluator:
    """
    DEPRECATED — kept as fallback reference only.
    Use DecisionTreeEvaluator (from decision_tree_bridge) instead.
    """
    pass


# ================================================================
# LTF BOS CONFIRMATION FOR FORMING SCHEMATICS
# ================================================================

def _try_confirm_with_ltf_bos(
    schematic: Dict,
    ltf_dfs: Dict[str, Optional[pd.DataFrame]],
) -> Dict:
    """
    Attempt to confirm a forming schematic (all 3 taps present but no MTF BOS)
    using lower-timeframe candles.

    TCT methodology: BOS entry should be on the lowest timeframe possible,
    looking for the first bullish/bearish BOS *prior to EQ* (in the discount/
    premium zone).  On MTF candles (4H etc.) the first internal swing high
    after a Model 2 Tap3 may not form below EQ before price reaches EQ —
    the MTF candles are too coarse.  LTF data (5m/1m) sees smaller swings
    and can confirm the BOS while price is still in the discount zone.

    EQ filter is kept: BOS must confirm below EQ (above for distribution).
    Only the lowest TF that confirms is used (cascade overwrites upward).

    Returns the original schematic unchanged if no LTF BOS is found.
    Returns a shallow copy with is_confirmed=True and bos_confirmation set
    if a valid LTF BOS below EQ is found.
    """
    tap2 = schematic.get("tap2") or {}
    tap3 = schematic.get("tap3") or {}
    direction = schematic.get("direction")
    tap3_price = tap3.get("price")
    tap3_time_str = tap3.get("time", "")

    if not tap3_time_str or not tap3_price or direction not in ("bullish", "bearish"):
        return schematic

    try:
        tap3_time = pd.Timestamp(tap3_time_str)
        if tap3_time.tzinfo is None:
            tap3_time = tap3_time.tz_localize("UTC")
    except Exception:
        return schematic

    rng = schematic.get("range") or {}
    equilibrium = rng.get("equilibrium")

    tap2_time: Optional[pd.Timestamp] = None
    try:
        tap2_time_str = tap2.get("time", "")
        if tap2_time_str:
            tap2_time = pd.Timestamp(tap2_time_str)
            if tap2_time.tzinfo is None:
                tap2_time = tap2_time.tz_localize("UTC")
    except Exception:
        pass

    best_bos: Optional[Dict] = None
    best_ltf: Optional[str] = None
    best_ref_high: Optional[float] = None

    for ltf in LTF_BOS_TIMEFRAMES:
        ltf_df = ltf_dfs.get(ltf)
        if ltf_df is None or len(ltf_df) < 20:
            continue

        ltf_df_reset = ltf_df.reset_index(drop=True)

        after_tap3 = ltf_df_reset["open_time"] >= tap3_time
        if not after_tap3.any():
            continue
        tap3_ltf_pos = int(after_tap3.idxmax())

        tap2_ltf_pos: Optional[int] = None
        if tap2_time is not None:
            after_tap2 = ltf_df_reset["open_time"] >= tap2_time
            if after_tap2.any():
                tap2_ltf_pos = int(after_tap2.idxmax())

        bos = None
        ref_high_used: Optional[float] = None
        try:
            detector = TCTSchematicDetector(ltf_df_reset)
            _ltf_highs = ltf_df_reset["high"].to_numpy()
            _ltf_lows = ltf_df_reset["low"].to_numpy()

            ltf_window = max(len(ltf_df_reset) - tap3_ltf_pos, 25)

            if direction == "bullish":
                # Highest point between tap2 and tap3 on LTF — this is the
                # market structure reference from which the BOS is drawn.
                if tap2_ltf_pos is not None and tap2_ltf_pos < tap3_ltf_pos:
                    ref_high_used = float(_ltf_highs[tap2_ltf_pos:tap3_ltf_pos + 1].max())
                else:
                    ref_high_used = float(_ltf_highs[:tap3_ltf_pos + 1].max())

                # Look for first bullish BOS below EQ after Tap3.
                # equilibrium filter is intentional: "first BOS prior to EQ".
                bos = detector._find_bullish_bos(
                    tap3_ltf_pos, ref_high_used, tap3_price,
                    equilibrium=equilibrium,
                    window=ltf_window,
                )

            else:  # bearish
                if tap2_ltf_pos is not None and tap2_ltf_pos < tap3_ltf_pos:
                    ref_low_used = float(_ltf_lows[tap2_ltf_pos:tap3_ltf_pos + 1].min())
                else:
                    ref_low_used = float(_ltf_lows[:tap3_ltf_pos + 1].min())

                bos = detector._find_bearish_bos(
                    tap3_ltf_pos, ref_low_used, tap3_price,
                    equilibrium=equilibrium,
                    window=ltf_window,
                )
                ref_high_used = ref_low_used  # reuse field for logging

        except Exception as e:
            logger.warning(f"[5B-FORMING-LTF] BOS detection error on {ltf}: {e}")
            continue

        if bos:
            best_bos = bos
            best_ltf = ltf
            best_ref_high = ref_high_used
            # Continue — lower TF overwrites so lowest confirmed TF wins.

    if not best_bos:
        return schematic

    # Build bos_confirmation mirroring the structure from MTF detection.
    if direction == "bullish":
        bos_conf: Dict = {
            "type": "bullish_bos",
            "highest_point_between_tabs": {"price": best_ref_high},
            "bos_idx": best_bos["idx"],
            "bos_price": best_bos["price"],
            "is_inside_range": best_bos.get("is_inside_range", True),
            "confirmed": True,
            "ltf_initial_confirm": True,
            "ltf_timeframe": best_ltf,
        }
    else:
        bos_conf = {
            "type": "bearish_bos",
            "lowest_point_between_tabs": {"price": best_ref_high},
            "bos_idx": best_bos["idx"],
            "bos_price": best_bos["price"],
            "is_inside_range": best_bos.get("is_inside_range", True),
            "confirmed": True,
            "ltf_initial_confirm": True,
            "ltf_timeframe": best_ltf,
        }

    confirmed = {**schematic}
    confirmed["is_confirmed"] = True
    confirmed["bos_confirmation"] = bos_conf
    confirmed["entry"] = {
        **(schematic.get("entry") or {}),
        "price": best_bos["price"],
        "type": f"LTF_BOS_{best_ltf}",
        "description": (
            f"First {direction} BOS below EQ on {best_ltf} after Tap3 — "
            "earliest confirmation, best R:R"
        ),
    }
    logger.info(
        f"[5B-FORMING-LTF] Confirmed via {best_ltf} BOS at {best_bos['price']:.2f} "
        f"(dir={direction}, model={schematic.get('model', '?')})"
    )
    return confirmed


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

    # Extract tap2 time to bound the internal LTF structure window.
    tap2 = schematic.get("tap2") or {}
    tap2_time: Optional[pd.Timestamp] = None
    try:
        tap2_time_str = (tap2.get("time") or "")
        if tap2_time_str:
            tap2_time = pd.Timestamp(tap2_time_str)
            if tap2_time.tzinfo is None:
                tap2_time = tap2_time.tz_localize("UTC")
    except Exception:
        pass

    best_bos = None
    best_ltf = None

    # LTF_BOS_TIMEFRAMES is ordered highest → lowest (e.g. ["5m", "1m"]).
    # We keep overwriting best_bos so the LOWEST TF that confirms wins.
    for ltf in LTF_BOS_TIMEFRAMES:
        ltf_df = ltf_dfs.get(ltf)
        if ltf_df is None or len(ltf_df) < 20:
            continue

        ltf_df_reset = ltf_df.reset_index(drop=True)

        # Find tap3 position in LTF data.
        after_tap3 = ltf_df_reset["open_time"] >= tap3_time
        if not after_tap3.any():
            logger.debug(
                f"[{label}] {ltf}: tap3_time {tap3_time} predates all {len(ltf_df)} candles — skipping"
            )
            continue

        tap3_ltf_pos = int(after_tap3.idxmax())

        # Find tap2 position in LTF data (bounds the between-tap internal structure window).
        tap2_ltf_pos: Optional[int] = None
        if tap2_time is not None:
            after_tap2 = ltf_df_reset["open_time"] >= tap2_time
            if after_tap2.any():
                tap2_ltf_pos = int(after_tap2.idxmax())

        bos = None
        try:
            detector = TCTSchematicDetector(ltf_df_reset)

            # Pre-extract column arrays once to avoid repeated Series creation
            # inside the per-candle loops below (.iloc[i]["col"] allocates a
            # Series per call; direct array indexing is ~10x faster).
            _ltf_highs = ltf_df_reset["high"].to_numpy()
            _ltf_lows = ltf_df_reset["low"].to_numpy()
            _ltf_closes = ltf_df_reset["close"].to_numpy()

            # PRIMARY: scan LTF candles BETWEEN tap2 and tap3 for internal
            # swing structure.  TCT methodology: the market structure drawn
            # between tap2 and tap3 (swing highs for bullish, swing lows for
            # bearish) is the correct BOS reference — these levels are BELOW
            # the MTF structural high, giving an earlier and lower entry.
            # The first LTF close that breaks one of those levels AFTER tap3
            # is the entry signal.
            if tap2_ltf_pos is not None and tap2_ltf_pos < tap3_ltf_pos - 2:
                if direction == "bullish":
                    ltf_swings = []
                    for i in range(tap2_ltf_pos + 1, tap3_ltf_pos):
                        if detector._is_swing_high(i, lookback=1):
                            sh_price = _ltf_highs[i]
                            # Must be within the tap2-tap3 window (below MTF struct high)
                            if ref_low < sh_price < ref_high:
                                ltf_swings.append({"idx": i, "price": sh_price})
                    if not ltf_swings:
                        # Fallback to coarser swings within the same window
                        for i in range(tap2_ltf_pos + 2, tap3_ltf_pos - 1):
                            if detector._is_swing_high(i, lookback=2):
                                sh_price = _ltf_highs[i]
                                if ref_low < sh_price < ref_high:
                                    ltf_swings.append({"idx": i, "price": sh_price})
                    # Lowest first → earliest entry at best R:R
                    ltf_swings.sort(key=lambda s: s["price"])
                    for sh in ltf_swings:
                        for i in range(tap3_ltf_pos, len(ltf_df_reset)):
                            if _ltf_closes[i] > sh["price"]:
                                bos = {
                                    "idx": i,
                                    "price": sh["price"],
                                    "confirmation_close": _ltf_closes[i],
                                    "is_inside_range": True,
                                    "bos_method": "ltf_internal_structure",
                                }
                                break
                        if bos:
                            break

                else:  # bearish
                    ltf_swings = []
                    for i in range(tap2_ltf_pos + 1, tap3_ltf_pos):
                        if detector._is_swing_low(i, lookback=1):
                            sl_price = _ltf_lows[i]
                            if ref_low < sl_price < ref_high:
                                ltf_swings.append({"idx": i, "price": sl_price})
                    if not ltf_swings:
                        for i in range(tap2_ltf_pos + 2, tap3_ltf_pos - 1):
                            if detector._is_swing_low(i, lookback=2):
                                sl_price = _ltf_lows[i]
                                if ref_low < sl_price < ref_high:
                                    ltf_swings.append({"idx": i, "price": sl_price})
                    # Highest first → earliest entry at best R:R
                    ltf_swings.sort(key=lambda s: s["price"], reverse=True)
                    for sl in ltf_swings:
                        for i in range(tap3_ltf_pos, len(ltf_df_reset)):
                            if _ltf_closes[i] < sl["price"]:
                                bos = {
                                    "idx": i,
                                    "price": sl["price"],
                                    "confirmation_close": _ltf_closes[i],
                                    "is_inside_range": True,
                                    "bos_method": "ltf_internal_structure",
                                }
                                break
                        if bos:
                            break

            if bos is None:
                # FALLBACK: tap2 not in LTF window or no internal swings found.
                # Search post-tap3 LTF swings with no EQ filter and full window
                # so we still get a better entry than the MTF BOS.
                ltf_window = max(len(ltf_df_reset) - tap3_ltf_pos, 25)
                if direction == "bullish":
                    bos = detector._find_bullish_bos(
                        tap3_ltf_pos, ref_high, ref_low,
                        equilibrium=None, window=ltf_window,
                    )
                else:
                    bos = detector._find_bearish_bos(
                        tap3_ltf_pos, ref_low, ref_high,
                        equilibrium=None, window=ltf_window,
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
        self.evaluator = DecisionTreeEvaluator()
        self._jack_evaluator = JackTCTEvaluator()
        self.last_debug: Dict = {}
        self._lock = threading.Lock()
        # Per-symbol HTF bias cache: symbol → (bias_str, expiry_timestamp)
        self._htf_bias_cache: Dict[str, str] = {}
        self._htf_bias_expiry: Dict[str, float] = {}
        # Guard against overlapping scan_and_trade runs.  asyncio.wait_for
        # cancels the *await* but cannot kill the executor thread, so a
        # timed-out scan keeps running while the loop dispatches a new one.
        # This lock ensures only one thread mutates state at a time.
        self._scan_lock = threading.Lock()

    def get_mode(self) -> str:
        """Return the current trading mode: 'claude' or 'jack'."""
        return self.state.trading_mode

    def set_mode(self, mode: str) -> None:
        """Set trading mode ('claude' or 'jack') and persist to disk."""
        if mode not in ("claude", "jack"):
            raise ValueError(f"Invalid mode '{mode}' — must be 'claude' or 'jack'")
        with self._lock:
            self.state.trading_mode = mode
            self.state.save()
        logger.info(f"[5B] Trading mode changed to '{mode}'")

    def _get_evaluator(self, mode: str):
        """Return the evaluator for the given snapshotted mode string."""
        if mode == "jack":
            return self._jack_evaluator
        return self.evaluator

    def debug_snapshot(self) -> Dict:
        """Return a consistent snapshot of last_debug + state fields under the lock.

        The background scan loop writes last_debug and mutates state concurrently.
        Reading them separately risks mixing data from different scan cycles.
        Taking both under one lock guarantees the response is internally consistent.
        Uses Schematics5BTradeState.snapshot() so all state fields are captured in
        one call rather than being read individually across separate attribute accesses.
        """
        with self._lock:
            debug = dict(self.last_debug)
            debug["state_summary"] = self.state.snapshot()
        return debug

    def scan_and_trade(self, top_5_pairs=None) -> Dict:
        """Main cycle: fetch, detect, evaluate, trade.

        BTCUSDT only.  top_5_pairs parameter is accepted for backward
        compatibility but ignored — all scanning is done on BTCUSDT.

        Thread-safe: if another thread is already inside this method
        (e.g. the previous timed-out executor thread hasn't finished),
        this call returns immediately with action="scan_in_progress".
        """
        if not self._scan_lock.acquire(blocking=False):
            logger.warning("[5B] scan_and_trade skipped — previous scan still in progress")
            return {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "action": "scan_in_progress",
                "details": {"reason": "Previous scan cycle still running"},
            }

        try:
            return self._scan_and_trade_locked()
        finally:
            self._scan_lock.release()

    def _scan_and_trade_locked(self) -> Dict:
        """Actual scan logic, called only while self._scan_lock is held."""
        cycle_result = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": "none",
            "details": {},
        }

        try:
            # 1. Manage open trade first
            if self.state.current_trade:
                df_price = fetch_candles_sync(SYMBOL, "1m", 10)
                if df_price is None or len(df_price) == 0:
                    self.state.last_error = "Could not fetch price data"
                    self.state.save()
                    cycle_result["action"] = "error"
                    cycle_result["details"] = {"error": "Could not fetch price data"}
                    return cycle_result

                current_price = float(df_price.iloc[-1]["close"])
                result = self._manage_open_trade(current_price)
                cycle_result["action"] = result.get("action", "manage")
                cycle_result["details"] = result
                self.state.last_scan_time = cycle_result["timestamp"]
                self.state.last_scan_action = cycle_result["action"]
                self.state.save()
                return cycle_result

            # 2. Scan BTCUSDT for qualifying TCT setups.
            # Snapshot mode here (under _scan_lock) so the entire scan cycle uses
            # one consistent evaluator even if set_mode() is called concurrently.
            scan_mode = self.state.trading_mode
            scan_tfs = ["4h"] if scan_mode == "jack" else None
            sym_result = self._scan_single_symbol(SYMBOL, mode=scan_mode, timeframes=scan_tfs)
            best_setup = sym_result.get("best_setup")
            best_score = sym_result.get("best_score", 0)
            best_tf = sym_result.get("best_tf")
            best_current_price = sym_result.get("current_price", 0.0)
            best_htf_bias = sym_result.get("htf_bias", "neutral")
            all_forming = sym_result.get("forming", [])

            with self._lock:
                self.last_debug = {
                    "timestamp": cycle_result["timestamp"],
                    "trading_mode": scan_mode,
                    "symbols_scanned": [SYMBOL],
                    "current_price": best_current_price,
                    "best_symbol": SYMBOL,
                    "best_tf": best_tf,
                    "best_score": best_score,
                    "htf_cascade_active": True,
                    "forming_schematics": all_forming[:5],
                    "per_symbol": {SYMBOL: sym_result},
                }

            # 3. Enter trade on highest-scoring qualifying setup.
            if best_setup:
                schematic, evaluation = best_setup
                entry_info = schematic.get("entry", {})
                candidate_price = entry_info.get("price", best_current_price)

                if self._is_duplicate_setup(candidate_price, evaluation["direction"]):
                    cycle_result["action"] = "duplicate_setup_skipped"
                    cycle_result["details"] = {
                        "price": best_current_price,
                        "symbol": SYMBOL,
                        "skipped_entry": candidate_price,
                        "reason": "Same setup as recent trade — cooldown active",
                    }
                else:
                    trade = self._enter_trade(
                        schematic, evaluation, best_current_price, best_htf_bias,
                        SYMBOL, best_tf,
                    )
                    cycle_result["action"] = "trade_entered"
                    cycle_result["details"] = trade
            else:
                cycle_result["action"] = "no_qualifying_setups"
                cycle_result["details"] = {
                    "symbols": [SYMBOL],
                    "best_score": best_score,
                    "htf_bias": best_htf_bias,
                    "error": sym_result.get("error"),
                }

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

    # ----------------------------------------------------------------
    # HTF BIAS HELPER — per-symbol TTL cache
    # ----------------------------------------------------------------

    def _get_htf_bias(self, symbol: str) -> Tuple[str, Dict]:
        """Return (bias_str, debug_dict) for symbol, using a per-symbol TTL cache.

        Uses pivot-based market structure detection (not schematic detection)
        to determine directional bias from the Daily timeframe.
        """
        now_ts = time.time()
        expiry = self._htf_bias_expiry.get(symbol, 0.0)
        if now_ts < expiry:
            cached = self._htf_bias_cache.get(symbol, "neutral")
            return cached, {
                "status": "cached", "htf_bias": cached,
                "expires_in_s": round(expiry - now_ts),
            }

        htf_bias = "neutral"
        htf_debug: Dict = {"status": "not_fetched"}
        try:
            from decision_tree_bridge import detect_htf_market_structure
            df_htf = fetch_candles_sync(symbol, HTF_TIMEFRAME, 200)
            if df_htf is not None and len(df_htf) >= 20:
                ms_result = detect_htf_market_structure(df_htf, lookback=min(100, len(df_htf)))
                htf_bias = ms_result["bias"]
                htf_debug = {
                    "status": "scanned",
                    "method": "market_structure",
                    "candles": len(df_htf),
                    "htf_bias": htf_bias,
                    "swing_highs": len(ms_result.get("swing_highs", [])),
                    "swing_lows": len(ms_result.get("swing_lows", [])),
                    "structure_break": ms_result.get("structure_break"),
                    "reason": ms_result.get("reason", ""),
                }
            else:
                htf_debug = {
                    "status": "insufficient_data",
                    "candles": 0 if df_htf is None else len(df_htf),
                }
        except Exception as e:
            logger.warning(f"[5B] HTF gate error for {symbol}: {e}", exc_info=True)
            htf_debug = {"status": "error", "error": str(e), "fetch_error": True}
            # Don't cache error results — next cycle should retry immediately
            return htf_bias, htf_debug

        self._htf_bias_cache[symbol] = htf_bias
        self._htf_bias_expiry[symbol] = now_ts + self._HTF_CACHE_TTL.get(htf_bias, 900)
        return htf_bias, htf_debug

    # ----------------------------------------------------------------
    # SINGLE-SYMBOL SCAN — extracted so scan_and_trade can loop over pairs
    # ----------------------------------------------------------------

    def _scan_single_symbol(self, symbol: str, mode: str = "claude",
                            timeframes: Optional[List[str]] = None) -> Dict:
        """
        Fetch candles, detect TCT schematics, and evaluate all candidates for
        one symbol.  Returns a result dict keyed by: current_price, htf_bias,
        best_setup, best_score, best_tf, forming, timeframes, error.

        Args:
            mode:       Snapshotted trading mode ('claude' or 'jack').  Must be
                        passed by the caller so evaluator choice is deterministic
                        for the full scan cycle even if set_mode() runs concurrently.
            timeframes: MTF timeframes to scan.  None → use MTF_TIMEFRAMES (all).
                        Pass ["4h"] for Jack's mode (4H-only scan).
        """
        out: Dict = {
            "symbol": symbol,
            "current_price": 0.0,
            "htf_bias": "neutral",
            "best_setup": None,
            "best_score": 0,
            "best_tf": None,
            "forming": [],
            "timeframes": {},
            "error": None,
        }

        try:
            _t0 = time.time()
            logger.info(f"[5B] _scan_single_symbol started for {symbol}")
            # Current price
            df_price = fetch_candles_sync(symbol, "1m", 10)
            if df_price is None or len(df_price) == 0:
                out["error"] = f"No price data for {symbol}"
                return out
            current_price = float(df_price.iloc[-1]["close"])
            out["current_price"] = current_price
            logger.info(f"[5B] price fetch ok ({time.time()-_t0:.1f}s) — ${current_price:,.2f}")

            # HTF bias (per-symbol cache)
            htf_bias, htf_debug = self._get_htf_bias(symbol)
            out["htf_bias"] = htf_bias
            all_tf_results: Dict = {HTF_TIMEFRAME: htf_debug}
            logger.info(f"[5B] HTF bias done ({time.time()-_t0:.1f}s) — {htf_bias}")

            # Resolve which MTF timeframes to scan
            active_mtf_tfs = timeframes if timeframes is not None else MTF_TIMEFRAMES

            # Parallel MTF + LTF candle fetch
            mtf_dfs: Dict[str, Optional[pd.DataFrame]] = {}
            ltf_dfs: Dict[str, Optional[pd.DataFrame]] = {}
            all_tfs = {
                **{tf: _MTF_CANDLE_LIMITS.get(tf, 300) for tf in active_mtf_tfs},
                **{ltf: _LTF_CANDLE_LIMITS[ltf] for ltf in LTF_BOS_TIMEFRAMES},
            }
            with ThreadPoolExecutor(max_workers=len(all_tfs)) as ex:
                futures = {
                    ex.submit(fetch_candles_sync, symbol, tf, lim): tf
                    for tf, lim in all_tfs.items()
                }
                for future in as_completed(futures):
                    tf = futures[future]
                    try:
                        df_result = future.result()
                        if tf in active_mtf_tfs:
                            mtf_dfs[tf] = df_result
                        if tf in LTF_BOS_TIMEFRAMES:
                            ltf_dfs[tf] = df_result
                    except Exception as e:
                        logger.warning(f"[5B] Fetch failed for {symbol}/{tf}: {e}")
                        if tf in active_mtf_tfs:
                            mtf_dfs[tf] = None
                        if tf in LTF_BOS_TIMEFRAMES:
                            ltf_dfs[tf] = None

            logger.info(f"[5B] parallel candle fetch done ({time.time()-_t0:.1f}s)")
            # Phase A: collect all schematics per TF (needed for HTF cascade)
            all_schematics_by_tf: Dict[str, List[Dict]] = {}
            for tf in active_mtf_tfs:
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
                    det = detect_tct_schematics(df, [])
                    all_schematics_by_tf[tf] = (
                        det.get("accumulation_schematics", [])
                        + det.get("distribution_schematics", [])
                    )
                except Exception as e:
                    logger.warning(f"[5B] Detection error on {symbol}/{tf}: {e}", exc_info=True)
                    all_schematics_by_tf[tf] = []
                    all_tf_results[tf] = {"status": "error", "error": str(e)}

            # Phase A.5: LTF BOS confirmation for forming schematics.
            # MTF candles (4H etc.) are too coarse to see the first internal
            # swing high that forms below EQ after Tap3 — especially for
            # Model 2 where Tap3 is a higher low and price recovers quickly.
            # Use actual 5m/1m candles to find the first bullish/bearish BOS
            # prior to EQ and promote the schematic to is_confirmed=True so
            # Phase B can evaluate and enter it.
            for tf in active_mtf_tfs:
                updated: List[Dict] = []
                for s in all_schematics_by_tf.get(tf, []):
                    if (
                        isinstance(s, dict)
                        and not s.get("is_confirmed")
                        and s.get("tap1") and s.get("tap2") and s.get("tap3")
                    ):
                        s = _try_confirm_with_ltf_bos(s, ltf_dfs)
                    updated.append(s)
                all_schematics_by_tf[tf] = updated
            logger.info(f"[5B] Phase A.5 (LTF BOS confirm) done ({time.time()-_t0:.1f}s)")

            # Phase B: evaluate with HTF cascade (lowest → highest TF walk)
            logger.info(f"[5B] Phase A (detection) done ({time.time()-_t0:.1f}s)")
            best_setup: Optional[Tuple] = None
            best_score = 0
            best_tf_local: Optional[str] = None

            for tf in reversed(active_mtf_tfs):
                df = mtf_dfs.get(tf)
                if tf not in all_schematics_by_tf or all_tf_results.get(tf, {}).get("status") in {"error", "insufficient_data"}:
                    continue

                all_sch = all_schematics_by_tf[tf]
                tf_evals: List[Dict] = []
                htf_upgraded_count = 0

                for s in all_sch:
                    if not isinstance(s, dict):
                        continue

                    # NOTE: No caller-side HTF directional filter here.
                    # Phase 7 in the v2 pipeline handles HTF alignment,
                    # including the reversal exception for strong counter-HTF setups.

                    if s.get("is_confirmed"):
                        s = self._refine_schematic_bos_with_ltf(s, ltf_dfs)

                    effective_tf = tf
                    if s.get("is_confirmed"):
                        s, effective_tf = _find_htf_upgrade(s, tf, all_schematics_by_tf)
                        if effective_tf != tf:
                            htf_upgraded_count += 1

                    eff_df = mtf_dfs.get(effective_tf) if mtf_dfs.get(effective_tf) is not None else df
                    eval_result = self._get_evaluator(mode).evaluate_schematic(
                        s, htf_bias, current_price,
                        total_candles=len(eff_df),
                        max_stale_candles=_MAX_STALE.get(effective_tf, 5),
                        candle_df=eff_df,
                    )
                    eval_result["source_tf"] = tf
                    eval_result["effective_tf"] = effective_tf
                    if effective_tf != tf:
                        eval_result["htf_upgraded"] = True

                    tf_evals.append(eval_result)
                    if eval_result["pass"] and eval_result["score"] > best_score:
                        best_score = eval_result["score"]
                        best_setup = (s, eval_result)
                        best_tf_local = effective_tf

                # Pick the best schematic for decision-tree display (highest quality_score
                # among confirmed ones; fall back to highest quality_score overall).
                dt_best: Optional[Dict] = None
                for _s in all_sch:
                    if not isinstance(_s, dict):
                        continue
                    if dt_best is None or _s.get("quality_score", 0) > dt_best.get("quality_score", 0):
                        if _s.get("is_confirmed") or dt_best is None:
                            dt_best = _s

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
                    "dt_data": _build_dt_data(dt_best, htf_bias) if dt_best else None,
                }

            # Collect forming (unconfirmed, all 3 taps present) schematics for display
            forming: List[Dict] = []
            for _ftf, _fsch_list in all_schematics_by_tf.items():
                for _fs in _fsch_list:
                    if not isinstance(_fs, dict) or _fs.get("is_confirmed", False):
                        continue
                    _fdir = _fs.get("direction", "unknown")
                    if htf_bias == "bullish" and _fdir == "bearish":
                        continue
                    if htf_bias == "bearish" and _fdir == "bullish":
                        continue
                    if not (_fs.get("tap1") and _fs.get("tap2") and _fs.get("tap3")):
                        continue
                    _range = _fs.get("range") or {}
                    _sl = _fs.get("stop_loss")
                    _tgt = _fs.get("target")
                    forming.append({
                        "symbol": symbol,
                        "tf": _ftf,
                        "direction": _fdir,
                        "model": _fs.get("model", ""),
                        "tap1": _fs.get("tap1"),
                        "tap2": _fs.get("tap2"),
                        "tap3": _fs.get("tap3"),
                        "range_high": _range.get("high") if _range else None,
                        "range_low": _range.get("low") if _range else None,
                        "target": _tgt.get("price") if isinstance(_tgt, dict) else _tgt,
                        "stop_loss": _sl.get("price") if isinstance(_sl, dict) else _sl,
                        "quality_score": _fs.get("quality_score", 0),
                    })
            forming.sort(key=lambda x: (x.get("tap3") or {}).get("idx", 0), reverse=True)

            out.update({
                "current_price": current_price,
                "htf_bias": htf_bias,
                "best_setup": best_setup,
                "best_score": best_score,
                "best_tf": best_tf_local,
                "forming": forming[:5],
                "timeframes": all_tf_results,
            })
            logger.info(
                f"[5B] _scan_single_symbol done ({time.time()-_t0:.1f}s) — "
                f"best_tf={best_tf_local}, best_score={best_score}, "
                f"timeframes={list(all_tf_results.keys())}"
            )

        except Exception as e:
            logger.error(f"[5B] _scan_single_symbol error for {symbol}: {e}", exc_info=True)
            out["error"] = str(e)

        return out

    def _refine_schematic_bos_with_ltf(
        self, schematic: Dict, ltf_dfs: Dict[str, Optional[pd.DataFrame]]
    ) -> Dict:
        """Delegate to the shared module-level helper with a 5B-specific log label."""
        return refine_schematic_bos_with_ltf(schematic, ltf_dfs, label="5B-LTF")

    def _enter_trade(self, schematic: Dict, evaluation: Dict, current_price: float, htf_bias: str,
                     symbol: str = SYMBOL, timeframe: str = "unknown") -> Dict:
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

        # TP1 = halfway between entry and final target
        if direction == "bullish":
            tp1_price = round(entry_price + (target_price - entry_price) / 2, 2)
        else:
            tp1_price = round(entry_price - (entry_price - target_price) / 2, 2)

        trade = {
            "id": len(self.state.trade_history) + 1,
            "symbol": symbol,
            "timeframe": timeframe,
            "direction": direction,
            "model": evaluation.get("model", "unknown"),
            "entry_price": round(entry_price, 2),
            "stop_price": round(stop_price, 2),
            "target_price": round(target_price, 2),
            "tp1_price": tp1_price,
            "tp1_hit": False,
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
        logger.info(f"[5B] Entered {direction} @ {entry_price} | {symbol} {timeframe} | SL={stop_price} | TP={target_price} | Score={evaluation['score']}")
        _notify_5b_entry(trade)
        return trade

    def _manage_open_trade(self, current_price: float) -> Dict:
        trade = self.state.current_trade
        if not trade:
            return {"action": "no_trade"}

        # Guard: discard corrupt trades with missing/zero entry_price
        entry_price = trade.get("entry_price", 0)
        if not entry_price:
            logger.error(f"[5B] Discarding corrupt trade with entry_price={entry_price!r}: {trade}")
            self.state.current_trade = None
            self.state.save()
            return {"action": "corrupt_trade_discarded", "details": "entry_price was 0 or missing"}

        direction = trade["direction"]
        stop_price = trade["stop_price"]
        target_price = trade["target_price"]
        tp1_price = trade.get("tp1_price")
        tp1_hit = trade.get("tp1_hit", False)

        if direction == "bullish":
            pnl_pct = ((current_price - entry_price) / entry_price) * 100
            hit_target = current_price >= target_price
            hit_stop = current_price <= stop_price
            hit_tp1 = tp1_price is not None and not tp1_hit and current_price >= tp1_price
        else:
            pnl_pct = ((entry_price - current_price) / entry_price) * 100
            hit_target = current_price <= target_price
            hit_stop = current_price >= stop_price
            hit_tp1 = tp1_price is not None and not tp1_hit and current_price <= tp1_price

        trade["live_pnl_pct"] = round(pnl_pct, 2)
        trade["current_price"] = round(current_price, 2)

        if hit_target:
            return self._close_trade(current_price, "target_hit")
        elif hit_tp1:
            # Take half off at TP1, slide SL to break-even, keep trade open
            return self._take_partial_profit(current_price)
        elif hit_stop:
            return self._close_trade(current_price, "stop_hit")
        else:
            return {"action": "holding", "pnl_pct": round(pnl_pct, 2),
                    "current_price": current_price, "direction": direction}

    def _take_partial_profit(self, exit_price: float) -> Dict:
        """Close half the position at TP1 and move the stop to break-even."""
        trade = self.state.current_trade
        if not trade:
            return {"action": "no_trade"}

        direction = trade["direction"]
        entry_price = trade["entry_price"]
        half_size = trade["position_size"] / 2

        if direction == "bullish":
            pnl_pct = ((exit_price - entry_price) / entry_price) * 100
        else:
            pnl_pct = ((entry_price - exit_price) / entry_price) * 100

        pnl_dollars = half_size * (pnl_pct / 100)
        self.state.balance += pnl_dollars

        # Record TP1 details in the live trade dict and halve the remaining size
        trade["tp1_hit"] = True
        trade["tp1_exit_price"] = round(exit_price, 2)
        trade["tp1_pnl_pct"] = round(pnl_pct, 2)
        trade["tp1_pnl_dollars"] = round(pnl_dollars, 2)
        trade["position_size"] = round(half_size, 2)   # remaining half
        trade["stop_price"] = round(entry_price, 2)    # slide SL to break-even

        self.state.save()
        logger.info(
            f"[5B] TP1 hit @ {exit_price} | ½ P&L={pnl_pct:.2f}% (${pnl_dollars:.2f}) "
            f"| SL → BE={entry_price} | Balance=${self.state.balance:.2f}"
        )
        _notify_5b_tp1(trade)

        return {
            "action": "tp1_hit",
            "exit_price": round(exit_price, 2),
            "pnl_pct": round(pnl_pct, 2),
            "pnl_dollars": round(pnl_dollars, 2),
            "new_stop": round(entry_price, 2),
        }

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

        # A stop hit after TP1 lands at break-even — still a net win overall
        is_win = reason == "target_hit" or (
            reason == "stop_hit" and trade.get("tp1_hit", False)
        )
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
        trade_sym = self.state.current_trade.get("symbol", SYMBOL)
        price = fetch_live_price(trade_sym)
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
