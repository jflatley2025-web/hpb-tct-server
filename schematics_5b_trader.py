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
from backtest.config import timeframe_to_seconds as _tf_seconds
from portfolio_manager import (
    USE_PORTFOLIO_LAYER as _USE_PORTFOLIO_LAYER,
    PortfolioState as _PortfolioState,
    PortfolioPosition as _PortfolioPosition,
    can_open_trade as _pm_can_open_trade,
    open_position as _pm_open_position,
    close_position as _pm_close_position,
    debug_snapshot as _pm_debug_snapshot,
)

# Reuse MEXC fetch helpers from tensor trader (no duplication)
from mexc_data import (
    fetch_candles_sync as _mexc_fetch_candles,
    fetch_live_price as _mexc_fetch_live_price,
)
import moondev_feed as _moondev


def fetch_candles_sync(symbol: str, tf: str, limit: int = 300):
    """
    Fetch OHLCV candles — routes to MoonDev when MOONDEV_PAPER_TRADING=true,
    falling back to MEXC on any failure.  Signature is identical to the MEXC
    version so all existing call sites work without change.
    """
    if _moondev.is_enabled():
        df = _moondev.fetch_candles(symbol, tf, limit)
        if df is not None and len(df) > 0:
            return df
        logger.warning("[5B] MoonDev candle fetch failed for %s/%s — falling back to MEXC", symbol, tf)
    return _mexc_fetch_candles(symbol, tf, limit)


def fetch_live_price(symbol: str = "BTCUSDT"):
    """
    Fetch the current price — routes to MoonDev when MOONDEV_PAPER_TRADING=true,
    falling back to MEXC on any failure.
    """
    if _moondev.is_enabled():
        price = _moondev.fetch_live_price(symbol)
        if price is not None:
            return price
        logger.warning("[5B] MoonDev live price fetch failed for %s — falling back to MEXC", symbol)
    return _mexc_fetch_live_price(symbol)

logger = logging.getLogger("Schematics5B")


def _compute_rr(entry: float, stop: float, target: float) -> float:
    """Compute reward-to-risk ratio from price levels. Returns 0.0 on invalid inputs."""
    if not entry or not stop or not target:
        return 0.0
    sl_dist = abs(entry - stop)
    if sl_dist == 0:
        return 0.0
    return abs(target - entry) / sl_dist


# ================================================================
# CONFIGURATION
# ================================================================
# ── Live scan symbol modes (env-configurable) ───────────────────
# Modes:
#   eth_only     → ETHUSDT only (fastest cycle, proven backtest edge)
#   primary_only → BTCUSDT, ETHUSDT, SOLUSDT
#   full         → all 12 symbols (monitoring/debug only)
_ALL_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT",
    "BCHUSDT", "WIFUSDT", "DOGEUSDT", "HBARUSDT", "FETUSDT",
    "XMRUSDT", "FARTCOINUSDT", "PEPEUSDT", "XRPUSDT",
]
_SYMBOL_MODES = {
    "eth_only": ["ETHUSDT"],
    "primary_only": ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
    "full": _ALL_SYMBOLS,
}
LIVE_SCAN_SYMBOL_MODE = os.getenv("LIVE_SCAN_SYMBOL_MODE", "primary_only")
TRADING_SYMBOLS = _SYMBOL_MODES.get(LIVE_SCAN_SYMBOL_MODE, _SYMBOL_MODES["eth_only"])
DEFAULT_SYMBOL = TRADING_SYMBOLS[0]
TRADEABLE_SYMBOLS = list(TRADING_SYMBOLS)  # all scanned symbols are tradeable in narrowed modes

# ── L3 relaxed BOS override (ETH-only live feature flag) ─────────
L3_RELAXED_BOS = os.getenv("L3_RELAXED_BOS", "true").lower() == "true"
L3_RELAXED_BOS_TOLERANCE_PCT = float(os.getenv("L3_RELAXED_BOS_TOLERANCE_PCT", "0.0025"))

# ── SCCE-gated L3 override (shadow mode — no live entries yet) ────
# When True: track what would pass L3 if SCCE phase is used to set tolerance.
# Does NOT affect real eval results or entry decisions.
# qualified phase  → 0.0025 tolerance (0.25%)
# bos_pending phase → 0.0015 tolerance (0.15%)
# all others       → unchanged (0.0)
SCCE_L3_OVERRIDE_SHADOW = os.getenv("SCCE_L3_OVERRIDE_SHADOW", "true").lower() == "true"
_SCCE_L3_TOL_QUALIFIED = 0.0025   # tolerance when SCCE phase == "qualified"
_SCCE_L3_TOL_BOS_PENDING = 0.0015 # tolerance when SCCE phase == "bos_pending"

# ── SCCE-gated L3 COMPRESSION override (shadow mode) ─────────────
# Report BE showed tolerance relaxation has zero effect because the real
# blocker is COMPRESSION (3 higher-lows / lower-highs), not micro-BOS
# distance.  This flag enables a shadow eval that skips compression
# entirely when SCCE confirms the structural BOS exists.
# Does NOT change live behaviour — shadow telemetry only.
SCCE_L3_COMP_OVERRIDE_SHADOW = os.getenv("SCCE_L3_COMP_OVERRIDE_SHADOW", "true").lower() == "true"

# ── Staged pair expansion plan ───────────────────────────────────
# Phase 1 candidates are NOT active. They are queued for rollout
# after ETH-only live mode proves stable (readiness gate).
PAIR_ROLLOUT_PLAN = {
    "phase_0_live_focus": ["ETHUSDT"],
    "phase_1_candidates": [
        {"symbol": "AAVEUSDT", "source_name": "AAVEUSDT.P", "order": 1},
        {"symbol": "ADAUSDT", "source_name": "ADAUSDT.P", "order": 2},
        {"symbol": "XLMUSDT", "source_name": "XLMUSDT.P", "order": 3},
        {"symbol": "CRVUSDT", "source_name": "CRVUSDT.P", "order": 4},
        {"symbol": "VIRTUALUSDT", "source_name": "VIRTUALUSDT.P", "order": 5},
    ],
    "readiness_gate": {
        "eth_live_trade_count_min": 1,
        "live_cycle_duration_max": 300,
        "scanner_healthy": True,
        "no_critical_errors": True,
    },
    "test_order_note": "Start with established/liquid names; VIRTUAL last (newer/noisier)",
}

logger.info(
    "[CONFIG] LIVE_SCAN_SYMBOL_MODE=%s symbols=%s L3_RELAXED_BOS=%s tolerance=%.4f",
    LIVE_SCAN_SYMBOL_MODE, TRADING_SYMBOLS, L3_RELAXED_BOS, L3_RELAXED_BOS_TOLERANCE_PCT,
)
STARTING_BALANCE = 5000.0
RISK_PER_TRADE_PCT = 1.0  # 1% of balance per trade
DEFAULT_LEVERAGE = 10
# Trailing stop — activated after TP1 hit.  Ratchets the stop in the
# profit direction by (target-entry)*TRAIL_FACTOR from the best price
# seen since TP1.  Matches backtest/config.py for parity with Run 40.
TRAIL_FACTOR = 0.50
# ── Scan throughput config (env-configurable for live A/B testing) ──
MAX_SYMBOL_CONCURRENCY = int(os.getenv("MAX_SYMBOL_CONCURRENCY", "1"))
PER_SYMBOL_TIMEOUT_SECONDS = int(os.getenv("PER_SYMBOL_TIMEOUT_SECONDS", "120"))
# HTF bias gate — daily candle tells us the dominant directional context.
# 4h was too narrow; 1d changes once a day so the cache TTL matches.
HTF_TIMEFRAME = "1d"
# Scan TFs — includes 15m to match backtest configuration.
# 15m has strict quality gates in decision_engine_v2 (RR>=0.8, range>=0.3%,
# displacement>=0.65) that filter low-quality setups.
# Note: 5m/1m are still fetched separately for LTF BOS *entry refinement*
# after a setup is confirmed on one of these TFs (see LTF_BOS_TIMEFRAMES).
MTF_TIMEFRAMES = ["4h", "1h", "30m", "15m"]
def _parse_int_env(key: str, default: int) -> int:
    raw = os.getenv(key, str(default))
    try:
        return int(raw)
    except (ValueError, TypeError):
        logger.warning("[CONFIG] Malformed env %s=%r — using default %d", key, raw, default)
        return default

AUTO_SCAN_INTERVAL = _parse_int_env("SCHEMATICS_5B_SCAN_INTERVAL", 60)
ACTIVE_SCHEMATIC_INTERVAL = _parse_int_env("SCHEMATICS_5B_ACTIVE_INTERVAL", 15)
ENTRY_THRESHOLD = 50  # aligned with backtest Run 38 (97.8% WR at threshold=50)

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
HTF_BIAS_CACHE_PATH = os.path.join(_DIR, "htf_bias_cache.json")

# Parity validation log (JSONL) — persistent storage for live vs backtest decision comparison.
# Written every cycle that reaches the decision gate; never overwritten (append-only).
# Read by scripts/analyze_parity.py and scripts/replay_parity_test.py.
_PARITY_LOG_DIR = os.path.join(_DIR, "logs")
_PARITY_LOG_PATH = os.path.join(_PARITY_LOG_DIR, "decision_parity.jsonl")

# Deduplication
DUPLICATE_COOLDOWN_SECONDS = 300
DUPLICATE_PRICE_TOLERANCE = 0.002

# ── Startup warmup gate ──────────────────────────────────────────
# Prevents premature trades on cold start by requiring N clean scan
# cycles with valid data before enabling trade execution.
# Set to False for instant rollback (no warmup, original behavior).
ENABLE_STARTUP_WARMUP = True
WARMUP_CYCLES_REQUIRED = 2          # observation-only cycles before trading
WARMUP_MIN_CANDLES_PER_TF = 50      # minimum candles per required TF
WARMUP_REQUIRED_TFS = ["1h", "4h"]  # TFs that must have sufficient data
WARMUP_MAX_CYCLES = 10              # log warning if warmup stuck beyond this
# Symbols validated during warmup — core liquidity pairs that must have data
# before trading is enabled.  Not all 12 to avoid fragile alt-coin API failures.
# Override via env: WARMUP_VALIDATION_SYMBOLS_OVERRIDE="BTCUSDT,ETHUSDT,SOLUSDT"
_warmup_sym_override = os.getenv("WARMUP_VALIDATION_SYMBOLS_OVERRIDE", "")
WARMUP_VALIDATION_SYMBOLS = (
    [s.strip() for s in _warmup_sym_override.split(",") if s.strip()]
    if _warmup_sym_override
    else ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
)
# Validate configured symbols exist in TRADING_SYMBOLS
WARMUP_VALIDATION_SYMBOLS = [s for s in WARMUP_VALIDATION_SYMBOLS if s in TRADING_SYMBOLS]


# ================================================================
# DECISION ENGINE AUDIT LOG
# Writes one JSON line per cycle to schematics_5b_decision_audit.log
# so shadow-mode comparisons can be reviewed without DB changes.
# ================================================================
_AUDIT_LOG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "schematics_5b_decision_audit.log",
)
_audit_logger = logging.getLogger("5B.audit")
_audit_logger.setLevel(logging.DEBUG)
# Lazy-add a file handler only when we first write (avoids creating the file
# on import if audit logging is never triggered).
_audit_fh_added = False


def _5b_audit_log(entry: Dict) -> None:
    """Append one JSON-encoded decision-comparison record to the audit log.

    Each record contains:
        symbol, timestamp, legacy_decision, v2_decision, match,
        model, timeframe, score, v2_failure_code, v2_reason,
        htf_bias, use_unified_engine
    """
    global _audit_fh_added
    if not _audit_fh_added:
        try:
            fh = logging.FileHandler(_AUDIT_LOG_PATH, encoding="utf-8")
            fh.setFormatter(logging.Formatter("%(message)s"))
            _audit_logger.addHandler(fh)
            _audit_fh_added = True
        except Exception as _e:
            logger.warning("[5B-AUDIT] Could not open audit log: %s", _e)
            return
    try:
        _audit_logger.debug(json.dumps(entry, default=str))
    except Exception as _e:
        logger.warning("[5B-AUDIT] Write failed: %s", _e)


def _log_decision_parity(entry: Dict) -> None:
    """Append one JSON-encoded parity record to logs/decision_parity.jsonl.

    Written on every cycle that reaches the decision gate — regardless of match/mismatch.
    On mismatch (match=False), the entry also contains a ``gate_debug`` key with the
    full v2 metadata dict (RIG result, RR check, displacement, model gates, session
    filters) for root-cause analysis.

    Read by: scripts/analyze_parity.py, scripts/replay_parity_test.py
    """
    try:
        os.makedirs(_PARITY_LOG_DIR, exist_ok=True)
        with open(_PARITY_LOG_PATH, "a", encoding="utf-8") as _pf:
            _pf.write(json.dumps(entry, default=str) + "\n")
    except Exception as _pe:
        logger.warning("[5B-PARITY] Write failed: %s", _pe)


def _get_entry_session_context(base_score: float) -> Dict:
    """Get session manipulation context for trade entry (MSCE integration)."""
    try:
        from session_manipulation import apply_session_multiplier
        return apply_session_multiplier(base_score)
    except Exception:
        return {"session": None, "boost_applied": False, "multiplier": 1.0}


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


def _notify_5b_shadow_candidate(shadow: Dict, open_trade: Dict) -> None:
    """Telegram alert when a qualified setup is blocked by an open trade."""
    open_sym = open_trade.get("symbol", "?")
    text = (
        f"<b>5B SHADOW — {shadow.get('symbol', '?')} | {shadow.get('timeframe', '?').upper()}</b>\n"
        f"Model: {shadow.get('model', '?')}\n"
        f"Score: {shadow.get('score', 0)}/100\n"
        f"⚠️ Blocked by open {open_sym} trade\n"
        f"<i>Would have entered if no position was open</i>"
    )
    _telegram_5b_send(text)


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
        # Issue 3: DD protection state
        self.peak_balance: float = STARTING_BALANCE  # highest balance seen
        self.dd_triggered_at: Optional[str] = None   # ISO UTC; None = not in hard block
        self.dd_trough_balance: Optional[float] = None  # lowest balance since hard block trigger
        # v15: compression state
        self.last_accepted_trade_ts: Optional[str] = None  # ISO UTC of last executed trade
        self.last_accepted_trade_priority: float = 0.0     # priority_score of that trade
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
                # Issue 3: DD protection state (default to current balance as peak on first load)
                self.peak_balance = data.get("peak_balance", self.balance)
                self.dd_triggered_at = data.get("dd_triggered_at")
                self.dd_trough_balance = data.get("dd_trough_balance")
                # v15: compression state
                self.last_accepted_trade_ts = data.get("last_accepted_trade_ts")
                self.last_accepted_trade_priority = data.get("last_accepted_trade_priority", 0.0)
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
                # Issue 3: DD protection state
                "peak_balance": round(self.peak_balance, 2),
                "dd_triggered_at": self.dd_triggered_at,
                "dd_trough_balance": round(self.dd_trough_balance, 2) if self.dd_trough_balance is not None else None,
                # v15: compression state
                "last_accepted_trade_ts": self.last_accepted_trade_ts,
                "last_accepted_trade_priority": round(self.last_accepted_trade_priority, 4),
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

    # Build range_data dict for demand-path ranking in BOS detection
    range_data_for_bos = _build_range_data_for_bos(rng)

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
            ltf_pc = PivotCache(ltf_df_reset, lookback=3)
            detector = TCTSchematicDetector(ltf_df_reset, pivot_cache=ltf_pc)
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
                    range_data=range_data_for_bos,
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
    _rng_refine = schematic.get("range") or {}
    equilibrium = _rng_refine.get("equilibrium")
    tap3_price = tap3.get("price")

    # Build range_data for demand-path ranking in BOS detection
    _range_data_refine = _build_range_data_for_bos(_rng_refine)

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
            ltf_pc = PivotCache(ltf_df_reset, lookback=3)
            detector = TCTSchematicDetector(ltf_df_reset, pivot_cache=ltf_pc)

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
                        range_data=_range_data_refine,
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
def _build_eth_funnel(per_sym_perf: list) -> dict:
    """Aggregate ETH funnel rejections from per-symbol perf data."""
    merged_rej = {}
    merged_l3 = {}
    all_l3_traces = []
    for p in per_sym_perf:
        for k, v in p.get("rejections", {}).items():
            merged_rej[k] = merged_rej.get(k, 0) + v
        for k, v in p.get("l3_sub_failures", {}).items():
            merged_l3[k] = merged_l3.get(k, 0) + v
        all_l3_traces.extend(p.get("l3_traces", []))
    return {
        "confirmed_evaluated": sum(p.get("confirmed_evaluated", 0) for p in per_sym_perf),
        "passed_eval": sum(p.get("passed_eval", 0) for p in per_sym_perf),
        "rejections": dict(sorted(merged_rej.items(), key=lambda x: -x[1])),
        "l3_sub_failures": dict(sorted(merged_l3.items(), key=lambda x: -x[1])),
        "l3_traces": all_l3_traces[:5],
    }


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
        self._has_forming_schematics: bool = False  # signals scan loop to use faster interval
        self._lock = threading.Lock()
        # Lifetime gate-block counters — incremented inside _scan_lock (no race).
        # One counter per failure_context label + "passes" for the success path.
        self._gate_metrics: Dict[str, int] = {
            "l2_blocks": 0,
            "l3_failures": 0,
            "rig_blocks": 0,
            "range_failures": 0,
            "tap_failures": 0,
            "liquidity_failures": 0,
            "bos_failures": 0,
            "htf_failures": 0,
            "rr_failures": 0,
            "passes": 0,
        }
        # ── Scan performance: last completed cycle (immutable until next cycle finishes) ──
        self._last_completed_scan_perf: Optional[Dict] = None
        self._scan_cycle_id = 0

        # ── Scan loop health trace ────────────────────────────────
        self._scan_trace = {
            "startup_fired": False,
            "task_created": False,
            "loop_entered": False,
            "loop_iteration_count": 0,
            "loop_iteration_started": False,
            "symbols_loaded": len(TRADING_SYMBOLS),
            "symbol_list": list(TRADING_SYMBOLS),
            "fetch_attempted": False,
            "fetch_success": False,
            "fetched_symbol": None,
            "candle_count": None,
            "evaluation_entered": False,
            "evaluation_completed": False,
            "last_exception": None,
            "last_exception_stage": None,
            "last_success_stage": "init",
            "heartbeat_time": None,
            "process_start_time": datetime.now(timezone.utc).isoformat(),
        }

        # ── Live execution health telemetry ──────────────────────
        self._live_health = {
            "signals_seen": 0,
            "after_rig": 0,
            "after_v2_gate": 0,
            "order_attempts": 0,
            "orders_submitted": 0,
            "orders_rejected": 0,
            "last_signal_time": None,
            "last_candidate_time": None,
            "last_order_attempt_time": None,
            "last_successful_order_time": None,
            "top_block_reasons": {},
            "recent_non_executions": [],  # last 10
            "conditional_seen": 0,
            "conditional_passed_floor": 0,
            "conditional_blocked_floor": 0,
            "l3_relaxed_bos_enabled": L3_RELAXED_BOS,
            "l3_relaxed_bos_tolerance": L3_RELAXED_BOS_TOLERANCE_PCT,
            "l3_relaxed_bos_seen": 0,
            "l3_relaxed_bos_passed": 0,
            "l3_relaxed_bos_failed": 0,
            # L3 near-miss buckets
            "l3_near_miss": {
                "strict_fail_relaxed_pass": 0,
                "within_0_10_pct": 0,
                "within_0_15_pct": 0,
                "within_0_25_pct": 0,
                "beyond_0_25_pct": 0,
            },
            # SCCE × L3 cross-telemetry: tracks SCCE phase at time of L3 failure
            # seed/tap1/tap2 → structure genuinely incomplete (L3 correct to block)
            # tap3/bos_pending/qualified → mature structure narrowly missing (L3 too tight)
            "scce_l3_cross": {
                "seed": 0,
                "tap1": 0,
                "tap2": 0,
                "tap3": 0,
                "bos_pending": 0,
                "qualified": 0,
                "no_match": 0,
                "examples": [],
            },
            # SCCE-gated L3 override shadow telemetry
            # SHADOW MODE ONLY — does not affect live entries.
            # Measures what WOULD pass L3 if SCCE phase were used to gate tolerance.
            "l3_override": {
                "shadow_mode": True,
                "enabled": SCCE_L3_OVERRIDE_SHADOW,
                "total_seen": 0,         # confirmed schematics that reached SCCE lookup
                "scce_match": 0,         # matched an active SCCE candidate
                "override_applied": 0,   # override tolerance > 0.0 was used
                "would_pass": 0,         # shadow eval passed with override tolerance
                "would_fail": 0,         # shadow eval still failed with override tolerance
                "no_override_needed": 0, # strict eval already passed (no help needed)
                "by_phase": {
                    "qualified": {"seen": 0, "would_pass": 0, "would_fail": 0},
                    "bos_pending": {"seen": 0, "would_pass": 0, "would_fail": 0},
                },
                "examples": [],  # last 10: {symbol, tf, model, scce_phase, bos_dist_pct, override_used, result}
            },
            # SCCE-gated L3 COMPRESSION override shadow telemetry
            # Bypasses compression requirement for SCCE-qualified candidates.
            # Runs full downstream funnel to track would_be_qualified.
            "l3_compression_override": {
                "shadow_mode": True,
                "enabled": SCCE_L3_COMP_OVERRIDE_SHADOW,
                "total_seen": 0,
                "scce_match": 0,
                "override_applied": 0,
                "would_pass_l3": 0,
                "would_fail_l3": 0,
                "would_be_qualified": 0,       # pass L3 AND all downstream gates
                "would_fail_downstream": 0,    # pass L3 but fail later gate
                "no_override_needed": 0,
                "next_blocker": {},             # {gate: count} after L3 pass
                "by_symbol": {},
                "examples": [],  # last 10
            },
        }
        # ── Per-symbol execution funnel (since boot) ─────────────
        self._symbol_funnels: Dict[str, Dict] = {}
        for _s in TRADING_SYMBOLS:
            self._symbol_funnels[_s] = {
                "schematics_detected": 0, "confirmed": 0,
                "after_l3": 0, "qualified": 0,
                "order_attempts": 0, "orders_submitted": 0,
                "top_block_reasons": {},
            }
        # ── Neutral HTF passthrough telemetry ─────────────────────
        self._neutral_htf = {
            "seen": 0,
            "would_have_blocked": 0,
            "allowed_with_penalty": 0,
            "qualified": 0,
            "order_attempts": 0,
            "examples": [],  # last 10
        }
        # ── Shadow candidates (blocked by open trade) ────────────
        self._shadow_candidates: List[Dict] = []

        # ── Global first-hit markers ─────────────────────────────
        self._global_first_events = {
            "first_schematic_detected": None,
            "first_confirmed": None,
            "first_l3_pass": None,
            "first_qualified": None,
            "first_order_attempt": None,
            "first_order_submitted": None,
        }
        # ── Rolling ETH override monitoring ─────────────────────
        self._eth_rollup_boot = self._new_eth_rollup()  # since boot
        self._eth_rollup_1h: List[Dict] = []            # per-cycle snapshots for rolling 1h
        self._eth_rollup_session_label = ""
        self._eth_rollup_session = self._new_eth_rollup()
        self._eth_cycle_archive: List[Dict] = []        # last 20 per-cycle records

        # ── First-hit events ─────────────────────────────────────
        self._eth_first_events = {
            "first_confirmed": None,
            "first_l3_relaxed_seen": None,
            "first_l3_relaxed_passed": None,
            "first_qualified": None,
            "first_order_attempt": None,
            "first_order_submitted": None,
        }

        # ── Portfolio reference (set externally for multi-symbol mode) ──
        self._portfolio = None

        # Per-symbol HTF bias cache: symbol → (bias_str, expiry_timestamp)
        self._htf_bias_cache: Dict[str, str] = {}
        self._htf_bias_expiry: Dict[str, float] = {}
        self._load_htf_cache()  # restore from disk if available

        # Guard against overlapping scan_and_trade runs.  asyncio.wait_for
        # cancels the *await* but cannot kill the executor thread, so a
        # timed-out scan keeps running while the loop dispatches a new one.
        # This lock ensures only one thread mutates state at a time.
        # _scan_lock_acquired_at tracks when the lock was taken so callers
        # can force-release a zombie lock after a timeout (see force_release_scan_lock).
        self._scan_lock = threading.Lock()
        self._scan_lock_acquired_at: Optional[float] = None

        # ── Startup warmup gate ───────────────────────────────────────
        self._warmup_cycles_completed = 0
        self._warmup_total_attempts = 0   # monotonic — never decremented (for stuck detection)
        self._is_ready = not ENABLE_STARTUP_WARMUP  # True if warmup disabled
        if ENABLE_STARTUP_WARMUP:
            logger.info(
                "[WARMUP] Startup warmup ENABLED — %d observation cycles "
                "required before trading (TFs: %s, min candles: %d)",
                WARMUP_CYCLES_REQUIRED, WARMUP_REQUIRED_TFS,
                WARMUP_MIN_CANDLES_PER_TF,
            )

        # ── Issue 5: portfolio layer ───────────────────────────────────
        # Equity is synced from self.state.balance before every can_open_trade()
        # call so this object never drifts silently.  open_positions is
        # reconstructed from current_trade (at most one) on startup.
        self._portfolio: Optional[_PortfolioState] = None
        if _USE_PORTFOLIO_LAYER:
            self._portfolio = _PortfolioState(
                equity=self.state.balance,
                peak_equity=self.state.peak_balance,
            )
            # Reconstruct any open position so the risk cap is correct
            # immediately — avoids a stale-zero total_risk_pct on the
            # first cycle after a restart with an open trade.
            _ct = self.state.current_trade
            if _ct and _ct.get("entry_price"):
                # Prefer the stored risk_amount (exact).  Fall back to
                # position_size * RISK_PER_TRADE_PCT / leverage as a best-effort
                # estimate so old-format trades (pre-risk_amount field) still
                # register a non-zero exposure in the portfolio.
                _ct_risk: Optional[float] = None
                if _ct.get("risk_amount"):
                    _ct_risk = float(_ct["risk_amount"])
                elif _ct.get("position_size") and _ct.get("entry_price"):
                    # position_size is denominated in the base asset (contracts).
                    # Notional risk ≈ balance × RISK_PER_TRADE_PCT / 100 as a
                    # conservative floor; use that rather than leaving it zero.
                    _ct_risk = self.state.balance * (RISK_PER_TRADE_PCT / 100)
                    logger.warning(
                        "[5B] Portfolio layer: current_trade missing "
                        "'risk_amount' — using balance-based fallback "
                        "$%.2f for position registration; "
                        "this may be an old-format trade record",
                        _ct_risk,
                    )

                if _ct_risk is not None and _ct_risk > 0:
                    _pm_open_position(
                        self._portfolio,
                        symbol=_ct.get("symbol", DEFAULT_SYMBOL),
                        direction=_ct.get("direction", "bullish"),
                        notional_risk=_ct_risk,
                        entry_price=float(_ct["entry_price"]),
                        model=_ct.get("model", "unknown"),
                        timeframe=_ct.get("timeframe", "unknown"),
                        opened_at=None,  # exact timestamp unavailable after reload
                    )
                    logger.info(
                        "[5B] Portfolio layer: reconstructed open position "
                        "from current_trade (symbol=%s risk=$%.2f)",
                        _ct.get("symbol", DEFAULT_SYMBOL), _ct_risk,
                    )
            logger.info(
                "[5B] Portfolio layer ENABLED — %s",
                _pm_debug_snapshot(self._portfolio),
            )

        # Log the active data source so it's visible in server logs at startup.
        if _moondev.is_enabled():
            logger.info(
                "[5B] MoonDev paper trading ENABLED — using MoonDev API for price/candle data "
                "(MEXC is fallback). Simulated order execution unchanged."
            )
        else:
            logger.info("[5B] MoonDev paper trading DISABLED — using MEXC for price/candle data.")

    def has_forming_schematics(self) -> bool:
        """Thread-safe check for active forming schematics (Tap 3 present, BOS imminent)."""
        with self._lock:
            return self._has_forming_schematics

    def _dd_risk_multiplier(self) -> float:
        """
        Hybrid DD risk tier from live balance state.

        Returns:
            1.0  — no DD concern, full position size
            0.5  — soft throttle (DD ≥ 2%), halve position size
            0.0  — hard block (DD ≥ 4%), do NOT enter trade

        Hard-block reset requires BOTH:
            1. hours_since_trigger >= _DD_RESET_HOURS
            2. equity recovered >= _DD_RECOVERY_PCT of (peak→trough) gap

        Reads all thresholds from decision_engine_v2 constants so the
        live bot and backtest are always governed by the same values.
        Fails open (returns 1.0) if the import is unavailable.
        """
        try:
            from decision_engine_v2 import (
                USE_DD_PROTECTION,
                _DD_SOFT_THRESHOLD,
                _DD_HARD_THRESHOLD,
                _DD_RISK_SCALE,
                _DD_RESET_HOURS,
                _DD_RECOVERY_PCT,
            )
        except ImportError as _ie:
            logger.error(
                "[5B] _dd_risk_multiplier: cannot import DD constants (%s) — "
                "failing open (1.0); DD protection is INACTIVE",
                _ie,
            )
            return 1.0

        if not USE_DD_PROTECTION or self.state.peak_balance <= 0:
            return 1.0

        dd_pct = (self.state.peak_balance - self.state.balance) / self.state.peak_balance

        if dd_pct >= _DD_HARD_THRESHOLD:
            # Track trough — keep the lowest balance seen since hard block started.
            if (
                self.state.dd_trough_balance is None
                or self.state.balance < self.state.dd_trough_balance
            ):
                self.state.dd_trough_balance = self.state.balance

            if self.state.dd_triggered_at is not None:
                try:
                    triggered_dt = datetime.fromisoformat(self.state.dd_triggered_at)
                    hours_in_dd = (
                        datetime.now(timezone.utc) - triggered_dt
                    ).total_seconds() / 3600

                    # Compute recovery fraction: what fraction of peak→trough has been regained.
                    _trough = self.state.dd_trough_balance
                    _recovery = 0.0
                    if _trough is not None and self.state.peak_balance > _trough:
                        _recovery = (
                            (self.state.balance - _trough)
                            / (self.state.peak_balance - _trough)
                        )

                    if hours_in_dd >= _DD_RESET_HOURS and _recovery >= _DD_RECOVERY_PCT:
                        logger.info(
                            "[5B] DD hard reset: %.0fh elapsed, recovery=%.0f%% ≥ %.0f%% "
                            "(trough=$%.2f) — re-baselining peak $%.2f → $%.2f",
                            hours_in_dd, _recovery * 100, _DD_RECOVERY_PCT * 100,
                            (_trough or 0), self.state.peak_balance, self.state.balance,
                        )
                        self.state.peak_balance = self.state.balance
                        self.state.dd_triggered_at = None
                        self.state.dd_trough_balance = None
                        self.state.save()
                        return 1.0
                    else:
                        if hours_in_dd >= _DD_RESET_HOURS:
                            # Time elapsed but recovery not sufficient — log diagnostic.
                            logger.debug(
                                "[5B] DD hard block: %.0fh elapsed but recovery=%.2f%% < %.0f%% "
                                "(trough=$%.2f balance=$%.2f) — blocked",
                                hours_in_dd, _recovery * 100, _DD_RECOVERY_PCT * 100,
                                (_trough or 0), self.state.balance,
                            )

                except (ValueError, TypeError):
                    pass  # malformed timestamp — treat as not yet triggered

            return 0.0  # hard block

        if dd_pct >= _DD_SOFT_THRESHOLD:
            return _DD_RISK_SCALE  # 0.5

        return 1.0

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
            held_for = (time.time() - self._scan_lock_acquired_at
                        if self._scan_lock_acquired_at else None)
            logger.warning(
                "[5B] scan_and_trade skipped — previous scan still in progress"
                " (held %.0fs)" % (held_for or 0,))
            return {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "action": "scan_in_progress",
                "details": {
                    "reason": "Previous scan cycle still running",
                    "held_seconds": round(held_for) if held_for else None,
                },
            }

        self._scan_lock_acquired_at = time.time()
        try:
            return self._scan_and_trade_locked()
        finally:
            self._scan_lock_acquired_at = None
            self._scan_lock.release()

    def force_release_scan_lock(self, max_age_seconds: int = 0) -> bool:
        """Force-release _scan_lock if held (idempotent, safe).

        Args:
            max_age_seconds: If >0, only release if lock held longer than this.
                             If 0 (default), release unconditionally.

        Returns True if the lock was force-released, False otherwise.
        This is a safety valve for zombie threads left behind by asyncio.wait_for
        cancelling the executor wrapper while the thread keeps running.
        """
        try:
            if not self._scan_lock.locked():
                return False

            if max_age_seconds > 0:
                acquired_at = self._scan_lock_acquired_at
                if acquired_at is not None:
                    held = time.time() - acquired_at
                    if held < max_age_seconds:
                        return False
                    logger.warning(
                        "[5B] Force-releasing _scan_lock held for %.0fs (limit %ds)",
                        held, max_age_seconds,
                    )
                else:
                    logger.warning(
                        "[5B] Force-releasing _scan_lock (no acquisition timestamp)",
                    )
            else:
                logger.warning("[5B] Force-releasing _scan_lock (unconditional)")

            self._scan_lock.release()
            self._scan_lock_acquired_at = None
            return True

        except RuntimeError:
            # Lock already released or invalid state
            return False

    def _scan_and_trade_locked(self) -> Dict:
        """Actual scan logic, called only while self._scan_lock is held."""
        cycle_result = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": "none",
            "details": {},
        }

        # Update scan trace heartbeat
        self._scan_trace["loop_iteration_started"] = True
        self._scan_trace["loop_iteration_count"] += 1
        self._scan_trace["heartbeat_time"] = datetime.now(timezone.utc).isoformat()
        self._scan_trace["last_success_stage"] = "LOOP_ENTER"

        # Initialize perf vars before try so the except handler can access them
        _cycle_start = time.time()
        _per_sym_perf = []
        _timed_out_symbols = []

        # Reset forming-schematics flag at cycle start so it can't remain
        # latched from a prior sweep through early-return branches (open-trade
        # management, fetch errors, warmup gate).  The flag is set to the
        # actual result after the full symbol sweep completes.
        with self._lock:
            self._has_forming_schematics = False

        try:
            # 1. Manage open trade first (if any)
            _has_open_trade = self.state.current_trade is not None
            if _has_open_trade:
                _trade_sym = self.state.current_trade.get("symbol", DEFAULT_SYMBOL)
                df_price = fetch_candles_sync(_trade_sym, "1m", 10)
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

                # If trade just closed, clear the flag so entry logic can run below
                if self.state.current_trade is None:
                    _has_open_trade = False

                # PHASE 1: Continue scanning/context-building even while trade is open.
                # Do NOT return early — fall through to symbol scan below.
                # Entry will be blocked by the _has_open_trade guard at the entry point.

            # 2. Scan all trading symbols for qualifying TCT setups.
            # Snapshot mode here (under _scan_lock) so the entire scan cycle uses
            # one consistent evaluator even if set_mode() is called concurrently.
            scan_mode = self.state.trading_mode
            scan_tfs = ["4h"] if scan_mode == "jack" else None

            best_setup = None
            best_score = 0
            best_tf = None
            best_current_price = 0.0
            best_htf_bias = "neutral"
            best_symbol = DEFAULT_SYMBOL
            # Separate tracking for tradeable symbols — used for trade execution.
            # best_setup/best_symbol track the global best (for monitoring/debug).
            best_tradeable_setup = None
            best_tradeable_score = 0
            best_tradeable_tf = None
            best_tradeable_price = 0.0
            best_tradeable_htf_bias = "neutral"
            best_tradeable_symbol = DEFAULT_SYMBOL
            all_forming = []
            all_forming_ranges = []
            all_symbol_results: Dict[str, Dict] = {}

            self._scan_trace["last_success_stage"] = "LOAD_SYMBOLS"
            self._scan_trace["symbols_loaded"] = len(TRADING_SYMBOLS)

            # ── Parallel symbol scanning with bounded concurrency ──
            _cycle_schematics = 0
            _cycle_confirmed = 0
            _cycle_qualified = 0
            _cycle_by_model = {}

            def _scan_one_symbol(_sym):
                """Scan a single symbol with hard timeout. Thread-safe."""
                _sym_start = time.time()
                try:
                    _result = self._scan_single_symbol(_sym, mode=scan_mode, timeframes=scan_tfs)
                except Exception as _sym_err:
                    logger.error("[5B] _scan_single_symbol(%s) crashed: %s", _sym, _sym_err, exc_info=True)
                    _result = {"error": str(_sym_err), "current_price": 0, "best_setup": None, "best_score": 0}
                _dur = round(time.time() - _sym_start, 1)
                return _sym, _result, _dur

            self._scan_trace["fetch_attempted"] = True
            with ThreadPoolExecutor(max_workers=MAX_SYMBOL_CONCURRENCY) as _pool:
                _futures = {
                    _pool.submit(_scan_one_symbol, _sym): _sym
                    for _sym in TRADING_SYMBOLS
                }
                # Collect results as futures complete (no global timeout —
                # each symbol is bounded by PER_SYMBOL_TIMEOUT internally)
                for _fut in as_completed(_futures):
                    _sym_name = _futures[_fut]
                    try:
                        _sym, _result, _dur = _fut.result(timeout=PER_SYMBOL_TIMEOUT_SECONDS)
                    except Exception as _fut_err:
                        _sym = _sym_name
                        _dur = PER_SYMBOL_TIMEOUT_SECONDS
                        _result = {"error": f"timeout/crash: {_fut_err}", "current_price": 0, "best_setup": None, "best_score": 0}
                        _timed_out_symbols.append(_sym)
                        logger.warning("[5B] Symbol %s timed out or crashed: %s", _sym, _fut_err)

                    self._scan_trace["fetched_symbol"] = _sym
                    self._scan_trace["fetch_success"] = _result.get("error") is None

                    # Collect per-symbol stats
                    _sym_tfs = _result.get("timeframes", {})
                    _sym_sch_count = sum(v.get("schematics_found", 0) for v in _sym_tfs.values() if isinstance(v, dict))
                    _sym_conf_count = sum(v.get("confirmed", 0) for v in _sym_tfs.values() if isinstance(v, dict))
                    _sym_setup = _result.get("best_setup") is not None
                    _cycle_schematics += _sym_sch_count
                    _cycle_confirmed += _sym_conf_count
                    if _sym_setup:
                        _cycle_qualified += 1
                        _m = ((_result.get("best_setup") or (None, {}))[1] or {}).get("model", "unknown") if isinstance(_result.get("best_setup"), tuple) else "unknown"
                        _cycle_by_model[_m] = _cycle_by_model.get(_m, 0) + 1

                    # Per-symbol funnel update
                    _sf = self._symbol_funnels.get(_sym)
                    if _sf is not None:
                        _sf["schematics_detected"] += _sym_sch_count
                        _sf["confirmed"] += _sym_conf_count
                        _sym_ef = _result.get("eth_funnel", {})
                        _sf["after_l3"] += _sym_ef.get("passed_eval", 0)
                        if _sym_setup:
                            _sf["qualified"] += 1
                        for _rk, _rv in _sym_ef.get("rejections", {}).items():
                            _sf["top_block_reasons"][_rk] = _sf["top_block_reasons"].get(_rk, 0) + _rv

                    # Global first-hit events
                    _ts_now = datetime.now(timezone.utc).isoformat()
                    if _sym_sch_count > 0 and not self._global_first_events["first_schematic_detected"]:
                        self._global_first_events["first_schematic_detected"] = _ts_now
                    if _sym_conf_count > 0 and not self._global_first_events["first_confirmed"]:
                        self._global_first_events["first_confirmed"] = _ts_now
                    if _sym_ef.get("passed_eval", 0) > 0 and not self._global_first_events["first_l3_pass"]:
                        self._global_first_events["first_l3_pass"] = _ts_now
                    if _sym_setup and not self._global_first_events["first_qualified"]:
                        self._global_first_events["first_qualified"] = _ts_now

                    _sym_fetch_perf = _result.get("fetch_perf", {})
                    _sym_eth_funnel = _result.get("eth_funnel", {})
                    _per_sym_perf.append({
                        "symbol": _sym,
                        "duration_seconds": _dur,
                        "fetch_seconds": _sym_fetch_perf.get("fetch_seconds", 0),
                        "requests_total": _sym_fetch_perf.get("requests_total", 0),
                        "requests_failed": _sym_fetch_perf.get("requests_failed", 0),
                        "setup_found": _sym_setup,
                        "schematics": _sym_sch_count,
                        "confirmed": _sym_conf_count,
                        "confirmed_evaluated": _sym_eth_funnel.get("confirmed_evaluated", 0),
                        "passed_eval": _sym_eth_funnel.get("passed_eval", 0),
                        "rejections": _sym_eth_funnel.get("rejections", {}),
                        "l3_sub_failures": _sym_eth_funnel.get("l3_sub_failures", {}),
                        "l3_traces": _sym_eth_funnel.get("l3_traces", []),
                        "timed_out": _sym in _timed_out_symbols,
                    })
                    all_symbol_results[_sym] = _result

                    # Track best setup across all symbols
                    _sym_best = _result.get("best_setup")
                    _sym_score = _result.get("best_score", 0)
                    if _sym_best and _sym_score > best_score:
                        best_setup = _sym_best
                        best_score = _sym_score
                        best_tf = _result.get("best_tf")
                        best_current_price = _result.get("current_price", 0.0)
                        best_htf_bias = _result.get("htf_bias", "neutral")
                        best_symbol = _sym
                    # Track best tradeable setup separately
                    if _sym_best and _sym_score > best_tradeable_score and _sym in TRADEABLE_SYMBOLS:
                        best_tradeable_setup = _sym_best
                        best_tradeable_score = _sym_score
                        best_tradeable_tf = _result.get("best_tf")
                        best_tradeable_price = _result.get("current_price", 0.0)
                        best_tradeable_htf_bias = _result.get("htf_bias", "neutral")
                        best_tradeable_symbol = _sym
                    all_forming.extend(_result.get("forming", []))
                    all_forming_ranges.extend(_result.get("forming_all_ranges", []))
                    all_forming_ranges.extend(_result.get("confirmed_ranges", []))

            self._scan_trace["evaluation_entered"] = True
            self._scan_trace["evaluation_completed"] = True
            self._scan_trace["last_success_stage"] = "EVALUATE_COMPLETE"

            # Finalize scan performance — snapshot into last_completed (immutable until next cycle ends)
            _cycle_dur = round(time.time() - _cycle_start, 1)
            _sorted_perf = sorted(_per_sym_perf, key=lambda x: x["duration_seconds"])
            self._scan_cycle_id += 1
            self._last_completed_scan_perf = {
                "scan_mode": LIVE_SCAN_SYMBOL_MODE,
                "cycle_id": self._scan_cycle_id,
                "cycle_start": datetime.fromtimestamp(_cycle_start, tz=timezone.utc).isoformat(),
                "cycle_end": datetime.now(timezone.utc).isoformat(),
                "cycle_duration_seconds": _cycle_dur,
                "symbols_total": len(TRADING_SYMBOLS),
                "symbols_completed": len(_per_sym_perf),
                "symbols_timed_out": len(_timed_out_symbols),
                "timed_out_symbols": _timed_out_symbols,
                "avg_symbol_seconds": round(sum(p["duration_seconds"] for p in _per_sym_perf) / max(len(_per_sym_perf), 1), 1),
                "slowest_symbol": _sorted_perf[-1]["symbol"] if _sorted_perf else None,
                "slowest_symbol_seconds": _sorted_perf[-1]["duration_seconds"] if _sorted_perf else 0,
                "fastest_symbol": _sorted_perf[0]["symbol"] if _sorted_perf else None,
                "fastest_symbol_seconds": _sorted_perf[0]["duration_seconds"] if _sorted_perf else 0,
                "max_symbol_concurrency": MAX_SYMBOL_CONCURRENCY,
                "per_symbol_timeout": PER_SYMBOL_TIMEOUT_SECONDS,
                "per_symbol": _per_sym_perf,
                "provider_perf": {
                    "requests_total": sum(p.get("requests_total", 0) for p in _per_sym_perf),
                    "requests_failed": sum(p.get("requests_failed", 0) for p in _per_sym_perf),
                    "avg_request_seconds": round(
                        sum(p.get("avg_request_seconds", 0) for p in _per_sym_perf) / max(len(_per_sym_perf), 1), 1
                    ),
                    "max_request_seconds": max((p.get("max_request_seconds", 0) for p in _per_sym_perf), default=0),
                    "fetch_time_total": round(sum(p.get("fetch_seconds", 0) for p in _per_sym_perf), 1),
                },
                "schematics_detected_total": _cycle_schematics,
                "confirmed_schematics_total": _cycle_confirmed,
                "qualified_setups_total": _cycle_qualified,
                "by_model": _cycle_by_model,
                "last_cycle_result": "candidate_found" if best_tradeable_setup else "no_qualifying_setups",
                "eth_funnel": _build_eth_funnel(_per_sym_perf),
            }
            logger.info(
                "[5B] SCAN_PERF: %.1fs total | %d symbols | slowest=%s (%.1fs) | "
                "schematics=%d confirmed=%d qualified=%d",
                _cycle_dur, len(_per_sym_perf),
                _sorted_perf[-1]["symbol"] if _sorted_perf else "none",
                _sorted_perf[-1]["duration_seconds"] if _sorted_perf else 0,
                _cycle_schematics, _cycle_confirmed, _cycle_qualified,
            )
            # Update rolling ETH override monitoring
            self._update_eth_rollups(self._last_completed_scan_perf)

            # If no setup found, use the first symbol's price for debug display
            if best_current_price == 0.0 and all_symbol_results:
                _first = all_symbol_results.get(DEFAULT_SYMBOL) or next(iter(all_symbol_results.values()))
                best_current_price = _first.get("current_price", 0.0)

            with self._lock:
                # Use unfiltered set (all_forming_ranges) so interval switching
                # catches forming schematics on either bias direction, not just
                # the HTF-filtered display list (all_forming).
                self._has_forming_schematics = bool(all_forming_ranges)
                self.last_debug = {
                    "timestamp": cycle_result["timestamp"],
                    "trading_mode": scan_mode,
                    "symbols_scanned": list(TRADING_SYMBOLS),
                    "current_price": best_current_price,
                    "best_symbol": best_symbol,
                    "best_tf": best_tf,
                    "best_score": best_score,
                    "htf_cascade_active": True,
                    "forming_schematics": all_forming[:5],
                    "per_symbol": all_symbol_results,
                    # Lifetime gate-block counters (since process start).
                    "gate_metrics": dict(self._gate_metrics),
                }

            # ── STARTUP WARMUP GATE ───────────────────────────────────────
            # Block trade evaluation for the first N cycles after cold start.
            # Data is fetched and scanned (proving connectivity + context build),
            # but no decision engine runs and no trades are entered.
            # Open-trade management (step 1 above) is NOT blocked — we always
            # manage exits even during warmup.
            if not self._is_ready:
                self._warmup_cycles_completed += 1
                self._warmup_total_attempts += 1

                # Validate data quality for core symbols (BTC, ETH, SOL)
                # In Jack mode only "4h" is fetched, so skip checks for TFs
                # that weren't requested — otherwise warmup never advances.
                _warmup_data_ok = True
                for _vsym in WARMUP_VALIDATION_SYMBOLS:
                    _vsym_result = all_symbol_results.get(_vsym, {})
                    _warmup_mtf = _vsym_result.get("mtf_dfs") or {}
                    _effective_tfs = [
                        tf for tf in WARMUP_REQUIRED_TFS if tf in _warmup_mtf
                    ]
                    if not _effective_tfs:
                        # None of the required TFs were fetched (shouldn't happen)
                        logger.warning(
                            "[WARMUP] Cycle %d/%d — no required TFs fetched for %s "
                            "(mode=%s) — not advancing",
                            self._warmup_cycles_completed,
                            WARMUP_CYCLES_REQUIRED,
                            _vsym, scan_mode,
                        )
                        _warmup_data_ok = False
                        continue
                    for _wtf in _effective_tfs:
                        _wdf = _warmup_mtf.get(_wtf)
                        if _wdf is None or len(_wdf) < WARMUP_MIN_CANDLES_PER_TF:
                            _wcount = 0 if _wdf is None else len(_wdf)
                            logger.warning(
                                "[WARMUP] Cycle %d/%d — insufficient data for "
                                "%s/%s: %d candles (need %d) — not advancing",
                                self._warmup_cycles_completed,
                                WARMUP_CYCLES_REQUIRED,
                                _vsym, _wtf, _wcount,
                                WARMUP_MIN_CANDLES_PER_TF,
                            )
                            _warmup_data_ok = False

                if not _warmup_data_ok:
                    # Data incomplete — don't count this cycle toward readiness
                    self._warmup_cycles_completed -= 1
                    # Stuck detection (uses monotonic counter that is never decremented)
                    if self._warmup_total_attempts >= WARMUP_MAX_CYCLES:
                        logger.error(
                            "[WARMUP] STUCK — %d total attempts, %d successful. "
                            "Check data feeds. Warmup will NOT auto-advance.",
                            self._warmup_total_attempts,
                            self._warmup_cycles_completed,
                        )
                    cycle_result["action"] = "warmup_data_incomplete"
                    cycle_result["details"] = {
                        "warmup_cycle": self._warmup_cycles_completed,
                        "required": WARMUP_CYCLES_REQUIRED,
                    }
                    self.state.last_scan_time = cycle_result["timestamp"]
                    self.state.last_scan_action = cycle_result["action"]
                    self.state.save()
                    return cycle_result

                logger.info(
                    "[WARMUP] Cycle %d/%d — observation mode "
                    "(data OK, %d symbols scanned, price=$%.2f)",
                    self._warmup_cycles_completed,
                    WARMUP_CYCLES_REQUIRED,
                    len(all_symbol_results),
                    best_current_price,
                )

                if self._warmup_cycles_completed >= WARMUP_CYCLES_REQUIRED:
                    self._is_ready = True
                    logger.info(
                        "[WARMUP COMPLETE] Trading ENABLED after %d clean cycles "
                        "— engine is ready",
                        self._warmup_cycles_completed,
                    )
                else:
                    cycle_result["action"] = "warmup_observation"
                    cycle_result["details"] = {
                        "warmup_cycle": self._warmup_cycles_completed,
                        "required": WARMUP_CYCLES_REQUIRED,
                        "symbols_scanned": len(all_symbol_results),
                        "best_price": best_current_price,
                    }
                    self.state.last_scan_time = cycle_result["timestamp"]
                    self.state.last_scan_action = cycle_result["action"]
                    self.state.save()
                    return cycle_result
            # ── END WARMUP GATE ───────────────────────────────────────────

            # ── Unified engine shadow run ──────────────────────────────────
            # Always run decision_engine_v2.decide() in parallel with the legacy
            # pipeline so we accumulate a shadow comparison log before activating.
            # USE_UNIFIED_ENGINE = False → shadow only (no behaviour change).
            # USE_UNIFIED_ENGINE = True  → v2 result gates the actual trade.
            try:
                from decision_engine_v2 import (
                    decide as _v2_decide,
                    USE_UNIFIED_ENGINE as _USE_V2,
                    USE_TRADE_COMPRESSION as _USE_COMPRESSION,
                    COMPRESSION_WINDOW_BARS as _COMPRESSION_BARS,
                    compute_priority_score as _cps,
                )
                _v2_available = True
            except ImportError:
                _v2_available = False
                _USE_V2 = False
                _USE_COMPRESSION = False
                _COMPRESSION_BARS = 6
                _cps = None

            _v2_result: Optional[Dict] = None
            if _v2_available:
                # Pass all fetched MTF candles (1d/4h/1h/30m) — decide() skips
                # TFs with insufficient data so missing 15m is handled gracefully.
                _best_sym_result = all_symbol_results.get(best_symbol, {})
                _candles_by_tf = dict(_best_sym_result.get("mtf_dfs") or {})
                # Guard the ISO timestamp parse — persisted state can be malformed if the
                # trade log was manually edited or written by an older version of the bot.
                # A bad value degrades to None (no prior hard-block trigger known).
                try:
                    _parsed_dd_triggered_at = (
                        datetime.fromisoformat(self.state.dd_triggered_at)
                        if self.state.dd_triggered_at
                        else None
                    )
                except (ValueError, TypeError) as _dte:
                    logger.warning(
                        "[5B] malformed dd_triggered_at=%r — treating as not triggered: %s",
                        self.state.dd_triggered_at, _dte,
                    )
                    _parsed_dd_triggered_at = None
                _v2_ctx = {
                    "current_price": best_current_price,
                    "current_time": datetime.now(timezone.utc),
                    # Wire live DD state so shadow comparisons reflect actual protection status.
                    # NOTE: _v2_result DD outputs (new_peak / new_trough) are NOT consumed here
                    # in shadow mode — _dd_risk_multiplier() manages live state independently.
                    "peak_equity": self.state.peak_balance,
                    "equity": self.state.balance,
                    "dd_protection_triggered_at": _parsed_dd_triggered_at,
                    "dd_trough_equity": self.state.dd_trough_balance,
                }
                try:
                    _v2_result = _v2_decide(_candles_by_tf, _v2_ctx)
                    logger.debug(
                        "[5B] V2 shadow: decision=%s score=%s tf=%s model=%s failure=%s",
                        _v2_result.get("decision"),
                        _v2_result.get("score"),
                        _v2_result.get("timeframe"),
                        _v2_result.get("model"),
                        _v2_result.get("failure_code"),
                    )
                except Exception as _ve:
                    logger.warning("[5B] V2 shadow error: %s", _ve)
                    _v2_result = None

            # ── Sync v2 DD state when unified engine is active ────────────
            # When USE_UNIFIED_ENGINE=True, v2's 3-tier DD protection (hard block /
            # soft throttle / clear) is authoritative. Persist its outputs so the
            # next cycle receives accurate DD context.
            # IMPORTANT: always persist including explicit None — v2 returns
            # new_trough=None and new_dd_protection_triggered_at=None on DD
            # recovery to signal "clear the state."  Guarding on `is not None`
            # would leave stale hard-block state and prevent recovery.
            if _USE_V2 and _v2_result is not None:
                self.state.peak_balance = _v2_result.get("new_peak") or self.state.peak_balance
                self.state.dd_trough_balance = _v2_result.get("new_trough")  # None = cleared
                _v2_new_dd_ts = _v2_result.get("new_dd_protection_triggered_at")
                self.state.dd_triggered_at = (
                    _v2_new_dd_ts.isoformat()
                    if isinstance(_v2_new_dd_ts, datetime)
                    else (str(_v2_new_dd_ts) if _v2_new_dd_ts is not None else None)
                )
                self.state.save()

            # 3. RIG: Build real MSCE context and evaluate globally.
            #    Uses canonical rig_engine — no fake session/RCM inputs.
            from rig_engine import evaluate_rig_global
            from msce_engine import get_msce_context

            msce = get_msce_context(best_htf_bias)

            # Select dominant range from unfiltered forming pool.
            # Rank by range span (widest = most significant structural range),
            # then use conservative (min) displacement across all valid ranges.
            from hpb_rig_validator import compute_displacement as _cd

            # Collect valid ranges from forming pool first, then fall back to
            # confirmed schematics so RIG always has range context.
            _valid_ranges = []
            for _fs in (all_forming_ranges or []):
                _rh = _fs.get("range_high")
                _rl = _fs.get("range_low")
                if _rh is not None and _rl is not None and _rh > _rl:
                    _valid_ranges.append((_rh, _rl, _rh - _rl))

            # No separate fallback needed — confirmed_ranges are merged into
            # all_forming_ranges at aggregation (line above), so the primary
            # loop always includes both forming AND confirmed schematic ranges.

            # Primary range: widest span (most significant structural range)
            _rig_range_high = None
            _rig_range_low = None
            if _valid_ranges:
                _valid_ranges.sort(key=lambda x: x[2], reverse=True)
                _rig_range_high = _valid_ranges[0][0]
                _rig_range_low = _valid_ranges[0][1]

            # Conservative displacement: minimum across ALL valid ranges.
            # If price is trapped in ANY range, RIG should know.
            _all_displacements = [
                _cd(best_current_price, rh, rl)
                for rh, rl, _ in _valid_ranges
            ]
            _all_displacements = [d for d in _all_displacements if d is not None]
            _conservative_disp = min(_all_displacements) if _all_displacements else None

            logger.debug(
                "[5B] RIG range context: range=[%s, %s] disp=%s sources=%d price=%.2f",
                _rig_range_high, _rig_range_low, _conservative_disp,
                len(_valid_ranges), best_current_price or 0,
            )

            rig_result = evaluate_rig_global(
                htf_bias=best_htf_bias,
                session_name=msce["session"],
                session_bias=msce["session_bias"],
                range_high=_rig_range_high,
                range_low=_rig_range_low,
                current_price=best_current_price,
                displacement_override=_conservative_disp,
            )
            with self._lock:
                self.last_debug["rig"] = rig_result
                self.last_debug["msce"] = msce

            # 4. DD hard-block timestamp — start the 72h reset clock as soon as equity
            #    crosses the hard threshold, REGARDLESS of whether any setup qualifies.
            #    Previously this only ran inside `if best_setup:` so scan cycles with no
            #    candidates would never set dd_triggered_at and the clock never started.
            _dd_mult = self._dd_risk_multiplier()
            if _dd_mult == 0.0 and self.state.dd_triggered_at is None:
                # Prefer the timestamp returned by the v2 engine (first-crossing detection
                # in decide()) if available; fall back to now.
                _v2_dd_ts = (
                    _v2_result.get("new_dd_protection_triggered_at")
                    if _v2_result
                    else None
                )
                self.state.dd_triggered_at = (
                    _v2_dd_ts.isoformat()
                    if isinstance(_v2_dd_ts, datetime)
                    else datetime.now(timezone.utc).isoformat()
                )
                self.state.save()
                logger.info(
                    "[5B] DD hard block: starting 72h reset clock "
                    "(balance=$%.2f peak=$%.2f)",
                    self.state.balance, self.state.peak_balance,
                )

            # 5. Enter trade on highest-scoring qualifying TRADEABLE setup.
            #    best_setup tracks the global best (any symbol) for monitoring.
            #    best_tradeable_setup tracks the best among TRADEABLE_SYMBOLS only.

            # Live health: track signal presence
            if best_tradeable_setup:
                self._live_health["signals_seen"] += 1
                self._live_health["last_signal_time"] = datetime.now(timezone.utc).isoformat()

            # ── PHASE 1 GUARD: block new entries while a trade is open ──
            # Scanning/context-building ran above even with an open trade.
            # But we still enforce single-trade-at-a-time for entry.
            if _has_open_trade and best_tradeable_setup:
                _shadow = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "symbol": best_tradeable_symbol,
                    "model": (best_tradeable_setup[1] or {}).get("model", "?") if isinstance(best_tradeable_setup, tuple) else "?",
                    "timeframe": best_tradeable_tf or "?",
                    "score": best_tradeable_score,
                    "qualified_setup": True,
                    "blocked_only_by_open_trade": True,
                }
                self._shadow_candidates.append(_shadow)
                if len(self._shadow_candidates) > 20:
                    self._shadow_candidates = self._shadow_candidates[-20:]
                logger.info(
                    "[5B] SHADOW_CANDIDATE: %s %s score=%s — blocked by open %s trade",
                    best_tradeable_symbol, _shadow["model"], best_tradeable_score,
                    self.state.current_trade.get("symbol", "?"),
                )
                _notify_5b_shadow_candidate(_shadow, self.state.current_trade)
                cycle_result["action"] = "shadow_candidate_blocked"
                cycle_result["details"] = _shadow
                self.state.last_scan_time = cycle_result["timestamp"]
                self.state.last_scan_action = cycle_result["action"]
                self.state.save()
                # Skip all entry logic below but scanning completed above
            elif best_tradeable_setup:
                schematic, evaluation = best_tradeable_setup
                # Use tradeable candidate's context for the execution path
                best_current_price = best_tradeable_price
                best_htf_bias = best_tradeable_htf_bias
                best_symbol = best_tradeable_symbol
                best_tf = best_tradeable_tf
                best_score = best_tradeable_score
                entry_info = schematic.get("entry") or {}
                candidate_price = entry_info.get("price", best_current_price)

                # Re-evaluate RIG with the actual schematic range (more precise)
                sch_rng = schematic.get("range") or {}
                sch_rh = sch_rng.get("high") or sch_rng.get("range_high")
                sch_rl = sch_rng.get("low") or sch_rng.get("range_low")

                # Estimate range duration from candle span + timeframe
                _TF_TO_HOURS = {
                    "1d": 24, "4h": 4, "1h": 1,
                    "30m": 0.5, "15m": 0.25,
                }
                sch_duration = 48  # conservative default for confirmed schematic
                high_idx = sch_rng.get("range_high_idx") or sch_rng.get("high_idx")
                low_idx = sch_rng.get("range_low_idx") or sch_rng.get("low_idx")
                if high_idx is not None and low_idx is not None:
                    candle_span = abs(high_idx - low_idx)
                    if candle_span > 0:
                        hours_per_candle = _TF_TO_HOURS.get(best_tf, 1)
                        sch_duration = candle_span * hours_per_candle

                rig_result = evaluate_rig_global(
                    htf_bias=best_htf_bias,
                    session_name=msce["session"],
                    session_bias=msce["session_bias"],
                    range_high=sch_rh,
                    range_low=sch_rl,
                    current_price=best_current_price,
                    range_duration_hours=sch_duration,
                    exec_score=evaluation.get("score", 0) / 100.0,
                )
                with self._lock:
                    self.last_debug["rig"] = rig_result

                rig_blocked = rig_result.get("status") not in (
                    "VALID", "NOT_EVALUATED", "CONDITIONAL", None,
                )

                # ── RIG + Execution Trace (synchronized with backtest) ──
                _exec_block_reason = None
                if rig_blocked:
                    _exec_block_reason = "FAIL_RIG_COUNTER_BIAS"
                elif self._is_duplicate_setup(candidate_price, evaluation["direction"]):
                    _exec_block_reason = "FAIL_DUPLICATE_SETUP"

                rig_trace = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "symbol": best_symbol,
                    "htf_bias": best_htf_bias,
                    "ltf_direction": evaluation.get("direction", "unknown"),
                    "rig_status": rig_result.get("status"),
                    "confidence_modifier": rig_result.get("confidence_modifier", 1.0),
                    "position": rig_result.get("position"),
                    "local_displacement": rig_result.get("displacement"),
                    "range_duration": rig_result.get("range_duration", 0),
                    "session": msce.get("session"),
                    "counter_bias": rig_result.get("counter_bias", False),
                    "execution_allowed": _exec_block_reason is None,
                    "execution_block_reason": _exec_block_reason,
                    "bos_confirmed": bool(schematic.get("bos_confirmation")),
                    "poi_valid": bool(schematic.get("quality_score", 0) >= 0.6),
                    "session_alignment": msce.get("session_bias") == best_htf_bias,
                    "final_confidence": evaluation.get("score", 0) / 100.0,
                }
                logger.info("[5B] RIG_TRACE: %s", rig_trace)
                with self._lock:
                    self.last_debug["rig_trace"] = rig_trace

                if not rig_blocked:
                    self._live_health["after_rig"] += 1
                    self._live_health["last_candidate_time"] = datetime.now(timezone.utc).isoformat()

                if rig_blocked:
                    self._record_non_execution(best_symbol, "RIG_BLOCK", rig_result.get("reason"))
                    logger.info("[5B] RIG BLOCK: %s", rig_result)
                    cycle_result["action"] = "rig_blocked"
                    cycle_result["details"] = {
                        "price": best_current_price,
                        "symbol": best_symbol,
                        "rig_status": rig_result.get("status"),
                        "rig_reason": rig_result.get("reason"),
                        "displacement": rig_result.get("displacement"),
                    }
                elif self._is_duplicate_setup(candidate_price, evaluation["direction"]):
                    self._record_non_execution(best_symbol, "DUPLICATE_SETUP", "cooldown active")
                    cycle_result["action"] = "duplicate_setup_skipped"
                    cycle_result["details"] = {
                        "price": best_current_price,
                        "symbol": best_symbol,
                        "skipped_entry": candidate_price,
                        "reason": "Same setup as recent trade — cooldown active",
                    }
                else:
                    # ── Shadow log + v2 gate ───────────────────────────────
                    # Legacy pipeline says TAKE. Check whether the unified engine
                    # agrees. If USE_UNIFIED_ENGINE=True and v2 says PASS, we block
                    # the trade and record the failure code. If USE_UNIFIED_ENGINE=False
                    # we always execute (shadow mode — log only).
                    _v2_decision = _v2_result.get("decision", "PASS") if _v2_result else "not_run"
                    _v2_blocks = _USE_V2 and _v2_result is not None and _v2_decision != "TAKE"

                    # Track conditional flow through v2 gate
                    if rig_result.get("status") == "CONDITIONAL":
                        self._live_health["conditional_seen"] += 1
                        if _v2_blocks:
                            _v2_fc = _v2_result.get("failure_code", "") if _v2_result else ""
                            if "SCORE" in _v2_fc:
                                self._live_health["conditional_blocked_floor"] += 1
                        else:
                            self._live_health["conditional_passed_floor"] += 1
                    self._live_health["after_v2_gate"] += (0 if _v2_blocks else 1)

                    # Neutral HTF passthrough telemetry
                    if best_htf_bias == "neutral":
                        self._neutral_htf["seen"] += 1
                        self._neutral_htf["would_have_blocked"] += 1  # old logic would have blocked
                        _nhtf_example = {
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "symbol": best_symbol,
                            "model": evaluation.get("model", "?"),
                            "tf": best_tf or "?",
                            "score": best_score,
                            "rr": round(evaluation.get("rr", 0) or 0, 1),
                            "htf_bias": "neutral",
                            "allowed_with_penalty": not _v2_blocks,
                            "final_block_reason": _v2_result.get("failure_code") if _v2_blocks else None,
                        }
                        if not _v2_blocks:
                            self._neutral_htf["allowed_with_penalty"] += 1
                        self._neutral_htf["examples"].append(_nhtf_example)
                        if len(self._neutral_htf["examples"]) > 10:
                            self._neutral_htf["examples"] = self._neutral_htf["examples"][-10:]

                    # Build unified parity entry (Steps 1 + 5).
                    # Includes all fields required by analyze_parity.py and the
                    # gate_debug block on mismatch for root-cause triage.
                    _parity_match = _v2_decision == "TAKE" or _v2_decision == "not_run"
                    _v2_meta = (_v2_result.get("metadata") or {}) if _v2_result else {}
                    _parity_entry = {
                        "type": "decision_parity",
                        "timestamp": cycle_result["timestamp"],
                        "symbol": best_symbol,
                        "timeframe": best_tf,
                        "model": evaluation.get("model", "unknown"),
                        # Decision comparison
                        "legacy_decision": "TAKE",
                        "v2_decision": _v2_decision,
                        "match": _parity_match,
                        # v2 scores / reason
                        "score_v2": _v2_result.get("score", 0) if _v2_result else None,
                        "reason_v2": _v2_result.get("reason") if _v2_result else None,
                        # Critical debug fields for mismatch analysis
                        "rr": _v2_result.get("rr") if _v2_result else None,
                        "displacement": _v2_meta.get("local_displacement"),
                        "session": _v2_meta.get("session"),
                        "htf_bias": best_htf_bias,
                        # Legacy audit fields (backward compat)
                        "score": evaluation.get("score", 0),
                        "v2_score": _v2_result.get("score", 0) if _v2_result else None,
                        "v2_failure_code": _v2_result.get("failure_code") if _v2_result else None,
                        "v2_reason": _v2_result.get("reason") if _v2_result else None,
                        "entry_price": candidate_price,
                        "use_unified_engine": _USE_V2,
                    }
                    # Step 5 — gate-level debug on mismatch only (avoids log bloat on matches).
                    # Includes: RIG result, RR check, displacement, model gates, session filters.
                    if not _parity_match:
                        _parity_entry["gate_debug"] = _v2_meta
                    _5b_audit_log(_parity_entry)
                    _log_decision_parity(_parity_entry)

                    # ── Trade-level INFO with v2 shadow ───────────────
                    _td_stop = (schematic.get("stop_loss") or {}).get("price") or 0
                    _td_tp1 = (schematic.get("target") or {}).get("price") or 0
                    _td_rr = _compute_rr(candidate_price, _td_stop, _td_tp1)
                    logger.info(
                        "[5B] [%s] TRADE_EVAL HTF=%s Model=%s Decision=%s "
                        "Entry=%.4f Stop=%.4f TP1=%.4f RR=%.2f "
                        "v2_shadow=%s v2_failure=%s",
                        best_symbol, best_htf_bias,
                        evaluation.get("model", "unknown"),
                        "V2_BLOCK" if _v2_blocks else "TAKE",
                        candidate_price, _td_stop, _td_tp1, _td_rr,
                        _v2_decision,
                        _v2_result.get("failure_code") if _v2_result else "n/a",
                    )

                    if _v2_blocks:
                        # v2 gate fired — trade blocked by unified engine
                        self._record_non_execution(
                            best_symbol,
                            _v2_result.get("failure_code", "V2_BLOCK"),
                            _v2_result.get("reason"),
                        )
                        logger.info(
                            "[5B] V2 GATE BLOCK: legacy=TAKE v2=PASS | "
                            "failure=%s reason=%s | score=%s model=%s tf=%s",
                            _v2_result.get("failure_code"),
                            _v2_result.get("reason"),
                            evaluation.get("score"),
                            evaluation.get("model"),
                            best_tf,
                        )
                        cycle_result["action"] = "v2_gate_blocked"
                        cycle_result["details"] = {
                            "price": best_current_price,
                            "symbol": best_symbol,
                            "legacy_would_take": True,
                            "v2_decision": _v2_decision,
                            "v2_failure_code": _v2_result.get("failure_code"),
                            "v2_reason": _v2_result.get("reason"),
                            "score": evaluation.get("score"),
                            "model": evaluation.get("model"),
                            "timeframe": best_tf,
                        }
                    else:
                        if _v2_result and _v2_decision != "TAKE":
                            # Shadow-mode mismatch: log but still execute
                            logger.info(
                                "[5B] SHADOW MISMATCH (not blocking): legacy=TAKE v2=%s | "
                                "failure=%s | score=%s model=%s tf=%s",
                                _v2_decision,
                                _v2_result.get("failure_code"),
                                evaluation.get("score"),
                                evaluation.get("model"),
                                best_tf,
                            )

                        # Issue 3: DD risk multiplier — already computed above before this block
                        # (_dd_mult = self._dd_risk_multiplier()) so the 72h clock starts even
                        # on cycles where no schematic reaches this point.
                        if _dd_mult == 0.0:
                            self._record_non_execution(best_symbol, "DD_HARD_BLOCK", "drawdown protection")
                            logger.info(
                                "[5B] DD hard block: suppressing legacy TAKE — "
                                "balance=$%.2f peak=$%.2f",
                                self.state.balance, self.state.peak_balance,
                            )
                            # Record first trigger time so the 72h reset can fire
                            if self.state.dd_triggered_at is None:
                                self.state.dd_triggered_at = datetime.now(timezone.utc).isoformat()
                                self.state.save()
                            cycle_result["action"] = "dd_hard_blocked"
                            cycle_result["details"] = {
                                "balance": self.state.balance,
                                "peak_balance": self.state.peak_balance,
                                "dd_pct": round(
                                    (self.state.peak_balance - self.state.balance)
                                    / self.state.peak_balance * 100, 2
                                ),
                            }
                        else:
                            # ── v15: compute priority score ───────────────────────
                            # Single source: decision_engine_v2.compute_priority_score.
                            # Falls back to inline formula if module is unavailable.
                            _p_stop = (schematic.get("stop_loss") or {}).get("price") or 0
                            _p_tgt = (schematic.get("target") or {}).get("price") or 0
                            _p_risk = abs(best_current_price - _p_stop) if _p_stop else 0
                            _p_rr = (
                                abs(_p_tgt - best_current_price) / _p_risk
                                if _p_risk > 0 and _p_tgt else 0.0
                            )
                            _p_rng = schematic.get("range") or {}
                            _p_rh = _p_rng.get("high") or _p_rng.get("range_high")
                            _p_rl = _p_rng.get("low") or _p_rng.get("range_low")
                            _p_disp = _cd(best_current_price, _p_rh, _p_rl)
                            if _p_disp is None:
                                _p_disp = 0.5
                            _p_score = evaluation.get("score", 0)
                            _p_rcm = schematic.get("quality_score", 0.0)
                            _priority = (
                                _cps(_p_score, _p_rcm, _p_rr, _p_disp)
                                if _cps is not None
                                else (
                                    _p_score * 0.5
                                    + _p_rcm * 100 * 0.2
                                    + _p_rr * 10 * 0.2
                                    + _p_disp * 100 * 0.1
                                )
                            )

                            # ── v15: compression check (post-gate execution filter) ─
                            _compress_block = False
                            if _USE_COMPRESSION and self.state.last_accepted_trade_ts is not None:
                                try:
                                    _last_dt = datetime.fromisoformat(
                                        self.state.last_accepted_trade_ts
                                    )
                                    _elapsed = (
                                        datetime.now(timezone.utc) - _last_dt
                                    ).total_seconds()
                                    _bar_secs = _tf_seconds(best_tf) if best_tf else 3600
                                    _bars_since = _elapsed / _bar_secs
                                    if (
                                        _bars_since < _COMPRESSION_BARS
                                        and _priority <= self.state.last_accepted_trade_priority
                                    ):
                                        _compress_block = True
                                        logger.info(
                                            "[5B] COMPRESSION_SUPPRESSED | "
                                            "priority=%.1f <= last=%.1f | "
                                            "bars_since=%.1f/%d | model=%s tf=%s",
                                            _priority,
                                            self.state.last_accepted_trade_priority,
                                            _bars_since, _COMPRESSION_BARS,
                                            evaluation.get("model"), best_tf,
                                        )
                                        self._record_non_execution(best_symbol, "COMPRESSION", "priority too low within window")
                                        cycle_result["action"] = "compression_suppressed"
                                        cycle_result["details"] = {
                                            "compression_blocked": True,
                                            "priority_score": round(_priority, 2),
                                            "last_priority": round(
                                                self.state.last_accepted_trade_priority, 2
                                            ),
                                            "bars_since_last": round(_bars_since, 2),
                                        }
                                except (ValueError, TypeError):
                                    pass  # malformed timestamp — skip compression check

                            # ── Issue 5: portfolio correlation check ───────────────
                            # Use a dedicated flag so portfolio suppression is never
                            # conflated with compression suppression — they have
                            # different semantics and different cycle_result actions.
                            # Evaluated even when _compress_block is True so the
                            # portfolio state stays visible in logs on every cycle.
                            _portfolio_block = False
                            _pm_scaling = 1.0
                            if not _compress_block and self._portfolio is not None:
                                # Sync equity so total_risk_pct is accurate
                                self._portfolio.equity = self.state.balance
                                _base_risk = (
                                    self.state.balance * (RISK_PER_TRADE_PCT / 100) * _dd_mult
                                )
                                _pm_result = _pm_can_open_trade(
                                    best_symbol, _base_risk, self._portfolio
                                )
                                if not _pm_result["allowed"]:
                                    logger.info(
                                        "[5B] PORTFOLIO_BLOCK | reason=%s | "
                                        "portfolio_risk=%.2f%%",
                                        _pm_result["reason"],
                                        _pm_result["adjusted_portfolio_risk"],
                                    )
                                    _portfolio_block = True
                                    cycle_result["action"] = "portfolio_blocked"
                                    cycle_result["details"] = {
                                        "reason": _pm_result["reason"],
                                        "adjusted_portfolio_risk": round(
                                            _pm_result["adjusted_portfolio_risk"], 2
                                        ),
                                        "portfolio_snapshot": _pm_debug_snapshot(
                                            self._portfolio
                                        ),
                                    }
                                else:
                                    _pm_scaling = _pm_result["scaling_factor"]

                            if not _compress_block and not _portfolio_block:
                                self._live_health["order_attempts"] += 1
                                self._live_health["last_order_attempt_time"] = datetime.now(timezone.utc).isoformat()
                                if not self._eth_first_events["first_order_attempt"]:
                                    self._eth_first_events["first_order_attempt"] = datetime.now(timezone.utc).isoformat()
                                if not self._global_first_events["first_order_attempt"]:
                                    self._global_first_events["first_order_attempt"] = datetime.now(timezone.utc).isoformat()
                                trade = self._enter_trade(
                                    schematic, evaluation, best_current_price, best_htf_bias,
                                    best_symbol, best_tf,
                                    risk_multiplier=_dd_mult * _pm_scaling,
                                )
                                if trade.get("error"):
                                    self._live_health["orders_rejected"] += 1
                                    self._record_non_execution(best_symbol, "ENTRY_REJECTED", trade["error"])
                                    logger.warning(
                                        "[5B] _enter_trade failed: %s", trade["error"]
                                    )
                                    cycle_result["action"] = "entry_failed"
                                    cycle_result["details"] = trade
                                else:
                                    self._live_health["orders_submitted"] += 1
                                    self._live_health["last_successful_order_time"] = datetime.now(timezone.utc).isoformat()
                                    if not self._eth_first_events["first_order_submitted"]:
                                        self._eth_first_events["first_order_submitted"] = datetime.now(timezone.utc).isoformat()
                                    if not self._global_first_events["first_order_submitted"]:
                                        self._global_first_events["first_order_submitted"] = datetime.now(timezone.utc).isoformat()
                                    # v15: only record accepted trade when order was placed
                                    self.state.last_accepted_trade_ts = (
                                        datetime.now(timezone.utc).isoformat()
                                    )
                                    self.state.last_accepted_trade_priority = _priority
                                    cycle_result["action"] = "trade_entered"
                                    cycle_result["details"] = trade

                                    # ── Issue 5: register in portfolio after confirmed entry
                                    if self._portfolio is not None:
                                        _pm_open_position(
                                            self._portfolio,
                                            symbol=best_symbol,
                                            direction=evaluation["direction"],
                                            notional_risk=trade.get("risk_amount", 0.0),
                                            entry_price=trade.get("entry_price", 0.0),
                                            model=evaluation.get("model", "unknown"),
                                            timeframe=best_tf or "unknown",
                                        )
            else:
                # Legacy: no qualifying setups found.
                # Log if v2 independently found a signal (informational — not executed).
                if _v2_result is not None and _v2_result.get("decision") == "TAKE":
                    logger.info(
                        "[5B] V2 FOUND SETUP (legacy missed): model=%s tf=%s score=%s — "
                        "not executing (legacy must confirm first)",
                        _v2_result.get("model"),
                        _v2_result.get("timeframe"),
                        _v2_result.get("score"),
                    )
                    _5b_audit_log({
                        "symbol": best_symbol,
                        "timestamp": cycle_result["timestamp"],
                        "legacy_decision": "PASS",
                        "v2_decision": "TAKE",
                        "match": False,
                        "model": _v2_result.get("model"),
                        "timeframe": _v2_result.get("timeframe"),
                        "score": _v2_result.get("score"),
                        "v2_failure_code": None,
                        "v2_reason": "v2_found_setup_legacy_missed",
                        "htf_bias": best_htf_bias,
                        "entry_price": _v2_result.get("entry_price"),
                        "use_unified_engine": _USE_V2,
                    })

                cycle_result["action"] = "no_qualifying_setups"
                cycle_result["details"] = {
                    "symbols": list(TRADING_SYMBOLS),
                    "best_score": best_score,
                    "best_symbol": best_symbol,
                    "htf_bias": best_htf_bias,
                }

            self.state.last_scan_time = cycle_result["timestamp"]
            self.state.last_scan_action = cycle_result["action"]
            self.state.last_error = None
            self.state.save()

        except Exception as e:
            logger.error(f"[5B] Scan error: {e}", exc_info=True)
            self._scan_trace["last_exception"] = str(e)[:200]
            self._scan_trace["last_exception_stage"] = "SCAN_CYCLE"
            # Save partial perf even on crash so we can diagnose
            self._scan_cycle_id += 1
            self._last_completed_scan_perf = {
                "cycle_id": self._scan_cycle_id,
                "cycle_duration_seconds": round(time.time() - _cycle_start, 1),
                "symbols_completed": len(_per_sym_perf),
                "symbols_total": len(TRADING_SYMBOLS),
                "symbols_timed_out": len(_timed_out_symbols),
                "error": str(e)[:200],
                "per_symbol": _per_sym_perf,
                "last_cycle_result": "error",
            }
            self.state.last_error = str(e)
            self.state.last_scan_action = "error"
            self.state.save()
            cycle_result["action"] = "error"
            cycle_result["details"] = {"error": str(e)}

        self._scan_trace["heartbeat_time"] = datetime.now(timezone.utc).isoformat()
        return cycle_result

    # ----------------------------------------------------------------
    # HTF BIAS CACHE PERSISTENCE
    # ----------------------------------------------------------------

    @staticmethod
    def _new_eth_rollup() -> Dict:
        return {
            "cycles_completed": 0,
            "schematics_detected": 0,
            "confirmed_schematics": 0,
            "l3_relaxed_bos_seen": 0,
            "l3_relaxed_bos_passed": 0,
            "qualified_setups": 0,
            "after_v2_gate": 0,
            "order_attempts": 0,
            "orders_submitted": 0,
            "orders_rejected": 0,
            "top_block_reasons": {},
        }

    def _get_session_label(self) -> str:
        from datetime import datetime, timezone
        h = datetime.now(timezone.utc).hour
        if 0 <= h < 8:
            return "Asia"
        elif 8 <= h < 16:
            return "London"
        return "New York"

    def _update_eth_rollups(self, cycle_perf: Dict):
        """Update rolling aggregation after each completed cycle."""
        ef = cycle_perf.get("eth_funnel", {})
        rej = ef.get("rejections", {})

        increment = {
            "cycles_completed": 1,
            "schematics_detected": cycle_perf.get("schematics_detected_total", 0),
            "confirmed_schematics": cycle_perf.get("confirmed_schematics_total", 0),
            "qualified_setups": cycle_perf.get("qualified_setups_total", 0),
        }

        for rollup in [self._eth_rollup_boot]:
            for k, v in increment.items():
                rollup[k] = rollup.get(k, 0) + v
            for k, v in rej.items():
                rollup["top_block_reasons"][k] = rollup["top_block_reasons"].get(k, 0) + v

        # Session rollup — reset on session change
        current_session = self._get_session_label()
        if current_session != self._eth_rollup_session_label:
            self._eth_rollup_session = self._new_eth_rollup()
            self._eth_rollup_session_label = current_session
        for k, v in increment.items():
            self._eth_rollup_session[k] = self._eth_rollup_session.get(k, 0) + v
        for k, v in rej.items():
            self._eth_rollup_session["top_block_reasons"][k] = self._eth_rollup_session["top_block_reasons"].get(k, 0) + v

        # Rolling 1h — keep last 40 cycle snapshots (~60min at 90s/cycle)
        self._eth_rollup_1h.append({**increment, "ts": datetime.now(timezone.utc).isoformat()})
        if len(self._eth_rollup_1h) > 40:
            self._eth_rollup_1h = self._eth_rollup_1h[-40:]

        # Per-cycle archive (ring buffer, last 20)
        _detected = increment["schematics_detected"]
        _confirmed = increment["confirmed_schematics"]
        _qualified = increment["qualified_setups"]
        _l3_seen = ef.get("l3_sub_failures", {})
        _l3_relaxed_seen = sum(1 for p in cycle_perf.get("per_symbol", []) if p.get("l3_sub_failures"))

        # Determine zero-schematic reason
        if _detected == 0:
            _per_sym = cycle_perf.get("per_symbol", [])
            _any_error = any(p.get("error") for p in _per_sym if isinstance(p, dict))
            if not _per_sym:
                _zero_reason = "data_gap"
            elif _any_error:
                _zero_reason = "data_gap"
            else:
                _zero_reason = "market_absence"
        else:
            _zero_reason = None

        # Determine last_result classification
        if _qualified > 0:
            _last_result = "qualified"
        elif _confirmed > 0:
            _last_result = "confirmed_only"
        elif _detected > 0:
            _last_result = "detected_only"
        else:
            _last_result = "no_schematics"

        _top_rej = max(rej, key=rej.get) if rej else None

        self._eth_cycle_archive.append({
            "cycle_id": cycle_perf.get("cycle_id", 0),
            "cycle_end": cycle_perf.get("cycle_end", ""),
            "session": current_session,
            "schematics_detected": _detected,
            "confirmed_schematics": _confirmed,
            "qualified_setups": _qualified,
            "l3_relaxed_bos_seen": sum(ef.get("l3_sub_failures", {}).values()),
            "top_block_reason": _top_rej,
            "zero_schematic_reason": _zero_reason,
            "last_result": _last_result,
        })
        if len(self._eth_cycle_archive) > 20:
            self._eth_cycle_archive = self._eth_cycle_archive[-20:]

        # First-hit events
        ts_now = datetime.now(timezone.utc).isoformat()
        if increment["confirmed_schematics"] > 0 and not self._eth_first_events["first_confirmed"]:
            self._eth_first_events["first_confirmed"] = ts_now
        if increment["qualified_setups"] > 0 and not self._eth_first_events["first_qualified"]:
            self._eth_first_events["first_qualified"] = ts_now

    def _record_non_execution(self, symbol: str, reason: str, detail: str = None):
        """Record a non-execution event for live health telemetry."""
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
        entry = f"{ts} {symbol} -> blocked: {reason}"
        if detail:
            entry += f" ({detail[:60]})"
        self._live_health["recent_non_executions"].append(entry)
        # Keep last 10
        if len(self._live_health["recent_non_executions"]) > 10:
            self._live_health["recent_non_executions"] = self._live_health["recent_non_executions"][-10:]
        # Track top block reasons
        self._live_health["top_block_reasons"][reason] = self._live_health["top_block_reasons"].get(reason, 0) + 1

    def get_scan_perf(self) -> dict:
        """Return last completed scan performance + current trace.

        last_completed persists until the NEXT cycle finishes.
        current_scan_trace shows in-progress state.
        """
        return {
            "last_completed": dict(self._last_completed_scan_perf) if self._last_completed_scan_perf else None,
            "current_scan_trace": dict(self._scan_trace),
        }

    def get_scan_trace(self) -> dict:
        """Return scan loop health trace for debugging."""
        return dict(self._scan_trace)

    def get_live_health(self) -> dict:
        """Return live trading health snapshot for dashboard."""
        h = dict(self._live_health)
        h["trading_enabled"] = True  # paper trading is always enabled
        h["exchange_mode"] = "paper (MoonDev)" if os.getenv("MOONDEV_PAPER_TRADING", "").lower() == "true" else "paper (MEXC)"
        h["current_trade"] = self.state.current_trade is not None
        h["balance"] = self.state.balance
        h["peak_balance"] = self.state.peak_balance
        h["dd_active"] = self.state.dd_triggered_at is not None
        # Sort block reasons by count
        h["top_block_reasons"] = dict(
            sorted(h.get("top_block_reasons", {}).items(), key=lambda x: -x[1])
        )
        # Conditional conversion rate
        _cs = h.get("conditional_seen", 0)
        _cp = h.get("conditional_passed_floor", 0)
        h["conditional_conversion_pct"] = round(_cp / _cs * 100, 1) if _cs > 0 else 0
        # Rolling ETH override monitoring
        h["eth_rollup_boot"] = dict(self._eth_rollup_boot)
        _1h_agg = self._new_eth_rollup()
        for snap in self._eth_rollup_1h:
            for k in ("cycles_completed", "schematics_detected", "confirmed_schematics", "qualified_setups"):
                _1h_agg[k] += snap.get(k, 0)
        h["eth_rollup_1h"] = _1h_agg
        h["eth_rollup_session"] = dict(self._eth_rollup_session)
        h["eth_rollup_session_label"] = self._eth_rollup_session_label or self._get_session_label()
        h["eth_first_events"] = dict(self._eth_first_events)
        h["eth_cycle_archive"] = list(self._eth_cycle_archive)
        h["symbol_funnels"] = {sym: dict(f) for sym, f in self._symbol_funnels.items()}
        h["global_first_events"] = dict(self._global_first_events)
        # Scan-while-trade-open telemetry
        h["scan_while_trade_open"] = {
            "current_trade_open": self.state.current_trade is not None,
            "scan_running_while_trade_open": True,  # always True after Phase 1
            "entry_blocked_due_to_open_trade": self.state.current_trade is not None,
            "context_updates_running": True,
            "schematic_detection_running": True,
        }
        h["shadow_candidates"] = list(self._shadow_candidates)
        h["neutral_htf"] = dict(self._neutral_htf)
        # SCCE snapshot
        try:
            from scce_engine import get_scce, SCCE_ENABLED
            if SCCE_ENABLED:
                h["scce"] = get_scce().get_snapshot()
            else:
                h["scce"] = {"enabled": False}
        except Exception:
            h["scce"] = {"enabled": False, "error": "import_failed"}

        # ETH HTF/warmup diagnostic
        h["eth_context_debug"] = {
            "warmup_completed": self._is_ready,
            "warmup_cycles_done": self._warmup_cycles_completed,
            "warmup_required": WARMUP_CYCLES_REQUIRED,
            "htf_bias_cache": {
                sym: {"bias": bias, "expires_in": round(self._htf_bias_expiry.get(sym, 0) - time.time())}
                for sym, bias in self._htf_bias_cache.items()
            },
            "scan_mode": LIVE_SCAN_SYMBOL_MODE,
            "trading_symbols": list(TRADING_SYMBOLS),
            "warmup_validation_symbols": list(WARMUP_VALIDATION_SYMBOLS),
            "context_mode": "mixed: HTF bias cached (24h TTL), pivots/ranges/schematics recomputed each cycle",
            "mtf_timeframes": list(MTF_TIMEFRAMES),
            "ltf_bos_timeframes": list(LTF_BOS_TIMEFRAMES),
            "candle_limits": dict(_MTF_CANDLE_LIMITS),
        }

        # Pair rollout queue status
        h["pair_rollout"] = {
            "active_phase": "phase_0_live_focus",
            "active_symbols": list(TRADING_SYMBOLS),
            "queued_pairs": [p["source_name"] for p in PAIR_ROLLOUT_PLAN.get("phase_1_candidates", [])],
            "expansion_ready": False,  # gate not yet evaluated live
        }
        return h

    def _load_htf_cache(self) -> None:
        """Load HTF bias cache from disk if available, skipping expired entries."""
        try:
            if not os.path.exists(HTF_BIAS_CACHE_PATH):
                logger.info("[5B] HTF bias cache not found — starting fresh")
                return
            with open(HTF_BIAS_CACHE_PATH, "r") as f:
                data = json.load(f)
            now_ts = time.time()
            loaded = 0
            expired = 0
            for symbol, entry in data.items():
                expiry = entry.get("expiry", 0.0)
                if expiry > now_ts:
                    self._htf_bias_cache[symbol] = entry.get("bias", "neutral")
                    self._htf_bias_expiry[symbol] = expiry
                    loaded += 1
                else:
                    expired += 1
            logger.info(
                "[5B] HTF bias cache loaded: %d/%d symbols cached (%d expired)",
                loaded, loaded + expired, expired,
            )
        except Exception as e:
            logger.warning("[5B] HTF bias cache load failed — starting fresh: %s", e)

    def _save_htf_cache(self) -> None:
        """Persist current HTF bias cache to disk (atomic write)."""
        try:
            data = {}
            for symbol, bias in self._htf_bias_cache.items():
                expiry = self._htf_bias_expiry.get(symbol, 0.0)
                data[symbol] = {"bias": bias, "expiry": expiry}
            tmp_path = HTF_BIAS_CACHE_PATH + ".tmp"
            with open(tmp_path, "w") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp_path, HTF_BIAS_CACHE_PATH)
        except Exception as e:
            logger.warning("[5B] HTF bias cache save failed: %s", e)

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
                # Don't overwrite a valid cached bias with "neutral" fallback —
                # preserve existing cache entry if we have one.
                if symbol in self._htf_bias_cache:
                    htf_bias = self._htf_bias_cache[symbol]
                    logger.info(
                        "[5B] HTF insufficient data for %s — retaining cached bias '%s'",
                        symbol, htf_bias,
                    )
                # Don't persist insufficient-data results to disk
                return htf_bias, htf_debug
        except Exception as e:
            logger.warning(f"[5B] HTF gate error for {symbol}: {e}", exc_info=True)
            htf_debug = {"status": "error", "error": str(e), "fetch_error": True}
            # Don't cache error results — next cycle should retry immediately
            return htf_bias, htf_debug

        self._htf_bias_cache[symbol] = htf_bias
        self._htf_bias_expiry[symbol] = now_ts + self._HTF_CACHE_TTL.get(htf_bias, 900)
        self._save_htf_cache()
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
            # Populated below after parallel fetch — exposed so _scan_and_trade_locked
            # can pass candles to decision_engine_v2.decide() without re-fetching.
            "mtf_dfs": {},
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

            # Parallel MTF + LTF candle fetch with provider telemetry
            mtf_dfs: Dict[str, Optional[pd.DataFrame]] = {}
            ltf_dfs: Dict[str, Optional[pd.DataFrame]] = {}
            all_tfs = {
                **{tf: _MTF_CANDLE_LIMITS.get(tf, 300) for tf in active_mtf_tfs},
                **{ltf: _LTF_CANDLE_LIMITS[ltf] for ltf in LTF_BOS_TIMEFRAMES},
            }
            _fetch_times = []  # (tf, seconds, success)
            _fetch_start = time.time()
            with ThreadPoolExecutor(max_workers=len(all_tfs)) as ex:
                futures = {
                    ex.submit(fetch_candles_sync, symbol, tf, lim): (tf, time.time())
                    for tf, lim in all_tfs.items()
                }
                try:
                    for future in as_completed(futures, timeout=120):
                        tf, _sub_time = futures[future]
                        _req_dur = round(time.time() - _sub_time, 1)
                        try:
                            df_result = future.result(timeout=30)
                            _fetch_times.append((tf, _req_dur, df_result is not None))
                            if tf in active_mtf_tfs:
                                mtf_dfs[tf] = df_result
                            if tf in LTF_BOS_TIMEFRAMES:
                                ltf_dfs[tf] = df_result
                        except Exception as e:
                            _fetch_times.append((tf, _req_dur, False))
                            logger.warning(f"[5B] Fetch failed for {symbol}/{tf}: {e}")
                            if tf in active_mtf_tfs:
                                mtf_dfs[tf] = None
                            if tf in LTF_BOS_TIMEFRAMES:
                                ltf_dfs[tf] = None
                except TimeoutError:
                    _fetch_times.append(("TIMEOUT", 120.0, False))
                    logger.error(f"[5B] Parallel candle fetch timed out for {symbol}")
                    for f in futures:
                        f.cancel()

            _fetch_total = round(time.time() - _fetch_start, 1)
            _fetch_ok = sum(1 for _, _, ok in _fetch_times if ok)
            _fetch_fail = sum(1 for _, _, ok in _fetch_times if not ok)
            _fetch_durations = [d for _, d, _ in _fetch_times]
            out["fetch_perf"] = {
                "fetch_seconds": _fetch_total,
                "requests_total": len(_fetch_times),
                "requests_ok": _fetch_ok,
                "requests_failed": _fetch_fail,
                "avg_request_seconds": round(sum(_fetch_durations) / max(len(_fetch_durations), 1), 1),
                "max_request_seconds": round(max(_fetch_durations, default=0), 1),
            }
            logger.info(f"[5B] fetch done ({_fetch_total:.1f}s) ok={_fetch_ok} fail={_fetch_fail}")
            # Expose fetched MTF DataFrames so the caller can pass them to
            # decision_engine_v2.decide() without a second round-trip to the API.
            out["mtf_dfs"] = {
                tf: df for tf, df in mtf_dfs.items() if df is not None
            }
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
                    # Create PivotCache per TF to eliminate pivot drift
                    pc = PivotCache(df, lookback=3)
                    det = detect_tct_schematics(df, [], pivot_cache=pc)
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

            # ── SCCE shadow update ────────────────────────────────
            try:
                from scce_engine import get_scce, SCCE_ENABLED
                if SCCE_ENABLED:
                    _scce = get_scce()
                    for _scce_tf in active_mtf_tfs:
                        _scce_sch = all_schematics_by_tf.get(_scce_tf, [])
                        if _scce_sch:
                            _scce.update_from_schematics(symbol, _scce_tf, _scce_sch, current_price)
            except Exception as _scce_err:
                logger.debug("[SCCE] Shadow update error: %s", _scce_err)

            # Phase B: evaluate with HTF cascade (lowest → highest TF walk)
            logger.info(f"[5B] Phase A (detection) done ({time.time()-_t0:.1f}s)")
            best_setup: Optional[Tuple] = None
            best_score = 0
            best_tf_local: Optional[str] = None

            # ETH funnel tracking
            _eth_funnel_rejections: Dict[str, int] = {}
            _eth_funnel_confirmed = 0
            _eth_funnel_passed_eval = 0
            _eth_l3_sub_failures: Dict[str, int] = {}
            _eth_l3_traces: List[Dict] = []

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
                    # ETH-only L3 relaxed BOS: pass tolerance only for ETH in eth_only mode
                    _l3_tol = (
                        L3_RELAXED_BOS_TOLERANCE_PCT
                        if L3_RELAXED_BOS and LIVE_SCAN_SYMBOL_MODE == "eth_only" and symbol == "ETHUSDT"
                        else 0.0
                    )
                    eval_result = self._get_evaluator(mode).evaluate_schematic(
                        s, htf_bias, current_price,
                        total_candles=len(eff_df),
                        max_stale_candles=_MAX_STALE.get(effective_tf, 5),
                        candle_df=eff_df,
                        l3_relaxed_bos_tolerance=_l3_tol,
                    )

                    # ── SCCE-gated L3 shadow override ──────────────────
                    # SHADOW MODE ONLY: run a second eval with SCCE-derived tolerance.
                    # Does NOT alter eval_result or any entry decision.
                    # Populates l3_override telemetry for Report BD.
                    if SCCE_L3_OVERRIDE_SHADOW and s.get("is_confirmed"):
                        try:
                            from scce_engine import get_scce, SCCE_ENABLED
                            if SCCE_ENABLED:
                                _ov = self._live_health["l3_override"]
                                _ov["total_seen"] += 1

                                # Look up matching SCCE candidate
                                _rng_ov = s.get("range") or {}
                                _rh_ov = _rng_ov.get("high")
                                _rl_ov = _rng_ov.get("low")
                                _dir_ov = s.get("direction", "")
                                _fam_ov = (
                                    "accumulation" if _dir_ov == "bullish"
                                    else "distribution" if _dir_ov == "bearish"
                                    else None
                                )
                                _scce_cand = None
                                _scce_phase_ov = None
                                if _fam_ov and _rh_ov and _rl_ov:
                                    for _c in get_scce().get_active_candidates(symbol):
                                        # Safety guards: active, not stale, range_valid
                                        if _c.get("stale") or not _c.get("range_valid"):
                                            continue
                                        if _c.get("model_family") != _fam_ov:
                                            continue
                                        _c_rh = _c.get("range_high")
                                        _c_rl = _c.get("range_low")
                                        if (
                                            _c_rh and _c_rl
                                            and abs(_c_rh - _rh_ov) / max(_rh_ov, 1) < 0.02
                                            and abs(_c_rl - _rl_ov) / max(_rl_ov, 1) < 0.02
                                        ):
                                            _scce_cand = _c
                                            _scce_phase_ov = _c.get("phase", "seed")
                                            break

                                # Determine override tolerance from SCCE phase
                                _override_tol = 0.0
                                if _scce_cand:
                                    _ov["scce_match"] += 1
                                    if _scce_phase_ov == "qualified":
                                        _override_tol = _SCCE_L3_TOL_QUALIFIED
                                    elif _scce_phase_ov == "bos_pending":
                                        _override_tol = _SCCE_L3_TOL_BOS_PENDING

                                # Run shadow eval only when override would make a difference
                                if _override_tol > 0.0 and not eval_result.get("pass"):
                                    # Strict eval failed — check if override would rescue it
                                    _ov["override_applied"] += 1
                                    _phase_bucket = _scce_phase_ov if _scce_phase_ov in ("qualified", "bos_pending") else None

                                    # Shadow evaluation — does NOT touch eval_result
                                    _shadow_eval = self._get_evaluator(mode).evaluate_schematic(
                                        s, htf_bias, current_price,
                                        total_candles=len(eff_df),
                                        max_stale_candles=_MAX_STALE.get(effective_tf, 5),
                                        candle_df=eff_df,
                                        l3_relaxed_bos_tolerance=_override_tol,
                                    )
                                    _shadow_pass = _shadow_eval.get("pass", False)
                                    _shadow_fc = _shadow_eval.get("failure_context", "unknown")

                                    # Track l3_override telemetry
                                    if _shadow_pass:
                                        _ov["would_pass"] += 1
                                    else:
                                        _ov["would_fail"] += 1
                                    if _phase_bucket:
                                        _pb = _ov["by_phase"][_phase_bucket]
                                        _pb["seen"] += 1
                                        if _shadow_pass:
                                            _pb["would_pass"] += 1
                                        else:
                                            _pb["would_fail"] += 1

                                    # BOS distance for example record
                                    _l3_pr_ov = (eval_result.get("phase_results") or {}).get("l3", {})
                                    _l3_tr_ov = _l3_pr_ov.get("trace", {})
                                    _bos_d_ov = abs(_l3_tr_ov.get("bos_distance_pct", 99))

                                    _ov["examples"] = _ov["examples"][-9:] + [{
                                        "ts": datetime.now(timezone.utc).isoformat(),
                                        "symbol": symbol,
                                        "tf": effective_tf,
                                        "model": eval_result.get("model") or s.get("model"),
                                        "direction": _dir_ov,
                                        "scce_phase": _scce_phase_ov,
                                        "override_tol": _override_tol,
                                        "bos_dist_pct": round(_bos_d_ov, 4),
                                        "strict_result": "fail",
                                        "override_result": "pass" if _shadow_pass else "fail",
                                        "override_fc": _shadow_fc if not _shadow_pass else None,
                                        "score": _shadow_eval.get("score", 0),
                                        "rr": round(_shadow_eval.get("rr", 0), 2),
                                    }]
                                    logger.debug(
                                        "[L3-OVERRIDE-SHADOW] %s %s %s phase=%s "
                                        "bos_dist=%.4f%% shadow=%s",
                                        symbol, effective_tf,
                                        eval_result.get("model") or s.get("model"),
                                        _scce_phase_ov, _bos_d_ov,
                                        "PASS" if _shadow_pass else "FAIL",
                                    )
                                elif eval_result.get("pass"):
                                    # Strict eval already passed — no override needed
                                    _ov["no_override_needed"] += 1
                        except Exception as _ov_err:
                            logger.debug("[L3-OVERRIDE-SHADOW] error: %s", _ov_err)

                    # ── SCCE-gated L3 COMPRESSION override (shadow) ────
                    # Separate experiment: skip compression entirely when
                    # SCCE confirms structural BOS.  Measures whether
                    # compression is the true bottleneck to tradeable setups.
                    if SCCE_L3_COMP_OVERRIDE_SHADOW and s.get("is_confirmed") and not eval_result.get("pass"):
                        try:
                            from scce_engine import get_scce, SCCE_ENABLED
                            if SCCE_ENABLED:
                                _co = self._live_health["l3_compression_override"]
                                _co["total_seen"] += 1

                                # SCCE candidate lookup (same logic as tolerance override)
                                _rng_co = s.get("range") or {}
                                _rh_co = _rng_co.get("high")
                                _rl_co = _rng_co.get("low")
                                _dir_co = s.get("direction", "")
                                _fam_co = (
                                    "accumulation" if _dir_co == "bullish"
                                    else "distribution" if _dir_co == "bearish"
                                    else None
                                )
                                _co_phase = None
                                if _fam_co and _rh_co and _rl_co:
                                    for _cc in get_scce().get_active_candidates(symbol):
                                        if _cc.get("stale") or not _cc.get("range_valid"):
                                            continue
                                        if _cc.get("model_family") != _fam_co:
                                            continue
                                        _cc_rh = _cc.get("range_high")
                                        _cc_rl = _cc.get("range_low")
                                        if (
                                            _cc_rh and _cc_rl
                                            and abs(_cc_rh - _rh_co) / max(_rh_co, 1) < 0.02
                                            and abs(_cc_rl - _rl_co) / max(_rl_co, 1) < 0.02
                                        ):
                                            _co_phase = _cc.get("phase", "seed")
                                            break

                                if _co_phase == "qualified":
                                    _co["scce_match"] += 1
                                    _co["override_applied"] += 1

                                    # Shadow eval with compression SKIPPED + relaxed BOS
                                    _co_eval = self._get_evaluator(mode).evaluate_schematic(
                                        s, htf_bias, current_price,
                                        total_candles=len(eff_df),
                                        max_stale_candles=_MAX_STALE.get(effective_tf, 5),
                                        candle_df=eff_df,
                                        l3_relaxed_bos_tolerance=_SCCE_L3_TOL_QUALIFIED,
                                        l3_skip_compression=True,
                                    )
                                    _co_pass = _co_eval.get("pass", False)
                                    _co_fc = _co_eval.get("failure_context", "unknown")
                                    _co_score = _co_eval.get("score", 0)
                                    _co_rr = round(_co_eval.get("rr", 0), 2)

                                    if _co_pass:
                                        _co["would_pass_l3"] += 1
                                        _co["would_be_qualified"] += 1
                                    else:
                                        # Check if L3 itself still failed (data issue)
                                        # vs passed L3 but failed downstream
                                        _co_l3_pr = (_co_eval.get("phase_results") or {}).get("l3", {})
                                        if not _co_l3_pr.get("passed", False):
                                            _co["would_fail_l3"] += 1
                                        else:
                                            # Passed L3 but failed downstream gate
                                            _co["would_pass_l3"] += 1
                                            _co["would_fail_downstream"] += 1
                                            _co["next_blocker"][_co_fc] = _co["next_blocker"].get(_co_fc, 0) + 1

                                    # Per-symbol tracking
                                    _sym_co = _co["by_symbol"].setdefault(symbol, {
                                        "override_applied": 0, "would_pass_l3": 0,
                                        "would_be_qualified": 0, "dominant_blocker": {},
                                    })
                                    _sym_co["override_applied"] += 1
                                    if _co_pass or (_co_l3_pr if not _co_pass else {}).get("passed", False):
                                        _sym_co["would_pass_l3"] += 1
                                    if _co_pass:
                                        _sym_co["would_be_qualified"] += 1
                                    elif not _co_pass and (_co_eval.get("phase_results") or {}).get("l3", {}).get("passed"):
                                        _sym_co["dominant_blocker"][_co_fc] = _sym_co["dominant_blocker"].get(_co_fc, 0) + 1

                                    # Original L3 trace for context
                                    _l3_pr_co = (eval_result.get("phase_results") or {}).get("l3", {})
                                    _l3_tr_co = _l3_pr_co.get("trace", {})

                                    _co["examples"] = _co["examples"][-9:] + [{
                                        "ts": datetime.now(timezone.utc).isoformat(),
                                        "symbol": symbol,
                                        "tf": effective_tf,
                                        "model": eval_result.get("model") or s.get("model"),
                                        "direction": _dir_co,
                                        "scce_phase": _co_phase,
                                        "orig_compression_count": _l3_tr_co.get("compression_count", 0),
                                        "orig_compression_pass": _l3_tr_co.get("compression_pass", False),
                                        "orig_bos_pass": _l3_tr_co.get("micro_bos_pass", False),
                                        "shadow_l3_pass": (_co_eval.get("phase_results") or {}).get("l3", {}).get("passed", False),
                                        "shadow_final_pass": _co_pass,
                                        "shadow_fc": _co_fc if not _co_pass else None,
                                        "score": _co_score,
                                        "rr": _co_rr,
                                    }]
                                    logger.debug(
                                        "[L3-COMP-OVERRIDE-SHADOW] %s %s %s comp=%d "
                                        "bos=%s shadow_l3=%s final=%s fc=%s score=%d rr=%.2f",
                                        symbol, effective_tf,
                                        eval_result.get("model") or s.get("model"),
                                        _l3_tr_co.get("compression_count", 0),
                                        _l3_tr_co.get("micro_bos_pass"),
                                        (_co_eval.get("phase_results") or {}).get("l3", {}).get("passed"),
                                        _co_pass, _co_fc if not _co_pass else "n/a",
                                        _co_score, _co_rr,
                                    )
                                elif eval_result.get("pass"):
                                    _co["no_override_needed"] += 1
                        except Exception as _co_err:
                            logger.debug("[L3-COMP-OVERRIDE-SHADOW] error: %s", _co_err)

                    eval_result["source_tf"] = tf
                    eval_result["effective_tf"] = effective_tf
                    if effective_tf != tf:
                        eval_result["htf_upgraded"] = True

                    tf_evals.append(eval_result)

                    # ── Per-schematic INFO log (all pairs) ────────────
                    _eval_model = eval_result.get("model", s.get("model", "unknown"))
                    _eval_dir = eval_result.get("direction", "unknown")
                    _eval_action = "PASS" if eval_result.get("pass") else "BLOCKED"
                    _eval_entry = (s.get("entry") or {}).get("price") or 0
                    _eval_stop = (s.get("stop_loss") or {}).get("price") or 0
                    _eval_tp1 = (s.get("target") or {}).get("price") or 0
                    _eval_rr = _compute_rr(_eval_entry, _eval_stop, _eval_tp1)
                    logger.info(
                        "[5B] [%s] HTF=%s Model=%s Direction=%s Decision=%s "
                        "Score=%s TF=%s Entry=%.4f Stop=%.4f TP1=%.4f RR=%.2f",
                        symbol, htf_bias, _eval_model, _eval_dir, _eval_action,
                        int(eval_result.get("score") or 0), effective_tf,
                        _eval_entry, _eval_stop, _eval_tp1, _eval_rr,
                    )

                    # ETH funnel: track confirmed and rejection reasons
                    if s.get("is_confirmed"):
                        _eth_funnel_confirmed += 1
                        if eval_result.get("pass"):
                            _eth_funnel_passed_eval += 1
                            # Track if L3 passed via relaxed BOS
                            _l3_pr_pass = (eval_result.get("phase_results") or {}).get("l3", {})
                            _l3_tr_pass = _l3_pr_pass.get("trace", {})
                            if _l3_tr_pass.get("relaxed_bos_tolerance", 0) > 0 and _l3_tr_pass.get("micro_bos_relaxed_pass"):
                                self._live_health["l3_relaxed_bos_seen"] += 1
                                self._live_health["l3_relaxed_bos_passed"] += 1
                                # This is the exact class of fix: strict would fail but relaxed passes
                                if not _l3_tr_pass.get("micro_bos_pass"):
                                    # Wait — micro_bos_pass is set True on relaxed pass too.
                                    # Check if strict would have failed by checking distance.
                                    pass
                                self._live_health["l3_near_miss"]["strict_fail_relaxed_pass"] += 1
                                _ts_now = datetime.now(timezone.utc).isoformat()
                                if not self._eth_first_events["first_l3_relaxed_seen"]:
                                    self._eth_first_events["first_l3_relaxed_seen"] = _ts_now
                                if not self._eth_first_events["first_l3_relaxed_passed"]:
                                    self._eth_first_events["first_l3_relaxed_passed"] = _ts_now
                        else:
                            _rej = eval_result.get("failure_context") or "unknown"
                            _eth_funnel_rejections[_rej] = _eth_funnel_rejections.get(_rej, 0) + 1
                            # L3 sub-failure tracking
                            if _rej == "L3":
                                _l3_pr = (eval_result.get("phase_results") or {}).get("l3", {})
                                _l3_tr = _l3_pr.get("trace", {})
                                _l3_sub = _l3_tr.get("first_failed", "unknown")
                                _eth_l3_sub_failures[_l3_sub] = _eth_l3_sub_failures.get(_l3_sub, 0) + 1
                                # Track relaxed BOS usage + near-miss buckets
                                _bos_dist = abs(_l3_tr.get("bos_distance_pct", 99))
                                if _l3_tr.get("relaxed_bos_tolerance", 0) > 0:
                                    self._live_health["l3_relaxed_bos_seen"] += 1
                                    self._live_health["l3_relaxed_bos_failed"] += 1
                                    if not self._eth_first_events["first_l3_relaxed_seen"]:
                                        self._eth_first_events["first_l3_relaxed_seen"] = datetime.now(timezone.utc).isoformat()
                                # Near-miss buckets (distance from breakout level)
                                nm = self._live_health["l3_near_miss"]
                                if _bos_dist <= 0.10:
                                    nm["within_0_10_pct"] += 1
                                elif _bos_dist <= 0.15:
                                    nm["within_0_15_pct"] += 1
                                elif _bos_dist <= 0.25:
                                    nm["within_0_25_pct"] += 1
                                else:
                                    nm["beyond_0_25_pct"] += 1
                                if len(_eth_l3_traces) < 5:
                                    _eth_l3_traces.append({
                                        "model": eval_result.get("model", "?"),
                                        "direction": eval_result.get("direction", "?"),
                                        "compression": _l3_tr.get("compression_count", 0),
                                        "comp_pass": _l3_tr.get("compression_pass"),
                                        "bos_pass": _l3_tr.get("micro_bos_pass"),
                                        "relaxed_pass": _l3_tr.get("micro_bos_relaxed_pass"),
                                        "bos_dist_pct": _l3_tr.get("bos_distance_pct"),
                                        "first_failed": _l3_sub,
                                    })
                                # ── SCCE × L3 cross-telemetry ──────────────
                                # Find matching SCCE candidate and record its phase.
                                # Phase distribution = L3 opportunity map:
                                #   seed/tap1/tap2  → structure immature, L3 correct
                                #   tap3/bos_pending/qualified → mature miss, tolerance candidate
                                try:
                                    from scce_engine import get_scce, SCCE_ENABLED
                                    if SCCE_ENABLED:
                                        _scce_phase = "no_match"
                                        _rng_s = s.get("range") or {}
                                        _rh_s = _rng_s.get("high")
                                        _rl_s = _rng_s.get("low")
                                        _dir_s = eval_result.get("direction", "")
                                        _fam_s = (
                                            "accumulation" if _dir_s == "bullish"
                                            else "distribution" if _dir_s == "bearish"
                                            else None
                                        )
                                        if _fam_s and _rh_s and _rl_s:
                                            for _sc in get_scce().get_active_candidates(symbol):
                                                if _sc.get("model_family") != _fam_s:
                                                    continue
                                                _sc_rh = _sc.get("range_high")
                                                _sc_rl = _sc.get("range_low")
                                                if (
                                                    _sc_rh and _sc_rl
                                                    and abs(_sc_rh - _rh_s) / max(_rh_s, 1) < 0.02
                                                    and abs(_sc_rl - _rl_s) / max(_rl_s, 1) < 0.02
                                                ):
                                                    _scce_phase = _sc.get("phase", "seed")
                                                    break
                                        _cross = self._live_health["scce_l3_cross"]
                                        _cross[_scce_phase] = _cross.get(_scce_phase, 0) + 1
                                        _cross["examples"] = _cross["examples"][-9:] + [{
                                            "ts": datetime.now(timezone.utc).isoformat(),
                                            "symbol": symbol,
                                            "tf": effective_tf,
                                            "model": eval_result.get("model"),
                                            "direction": _dir_s,
                                            "scce_phase": _scce_phase,
                                            "l3_sub": _l3_sub,
                                            "bos_dist_pct": round(_bos_dist, 4),
                                        }]
                                except Exception:
                                    pass

                    # Gate metrics — _scan_lock is held so no race condition.
                    _fc = eval_result.get("failure_context")
                    _fc_map = {
                        "L2":        "l2_blocks",
                        "L3":        "l3_failures",
                        "RIG":       "rig_blocks",
                        "range":     "range_failures",
                        "taps":      "tap_failures",
                        "liquidity": "liquidity_failures",
                        "BOS":       "bos_failures",
                        "HTF":       "htf_failures",
                        "RR":        "rr_failures",
                    }
                    if _fc in _fc_map:
                        self._gate_metrics[_fc_map[_fc]] += 1
                    if eval_result.get("pass"):
                        self._gate_metrics["passes"] += 1
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
            # and confirmed schematic ranges for RIG fallback
            forming: List[Dict] = []
            forming_all_ranges: List[Dict] = []
            confirmed_ranges: List[Dict] = []
            for _ftf, _fsch_list in all_schematics_by_tf.items():
                for _fs in _fsch_list:
                    if not isinstance(_fs, dict):
                        continue
                    _range = _fs.get("range") or {}
                    _fdir = _fs.get("direction", "unknown")

                    # Confirmed schematics: extract range for RIG fallback
                    if _fs.get("is_confirmed", False):
                        _cr_rh = _range.get("high")
                        _cr_rl = _range.get("low")
                        if _cr_rh is not None and _cr_rl is not None and _cr_rh > _cr_rl:
                            confirmed_ranges.append({
                                "range_high": _cr_rh,
                                "range_low": _cr_rl,
                            })
                        continue

                    if not (_fs.get("tap1") and _fs.get("tap2") and _fs.get("tap3")):
                        continue
                    _sl = _fs.get("stop_loss")
                    _tgt = _fs.get("target")
                    _forming_entry = {
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
                    }
                    # Unfiltered list for RIG (all directions, all ranges)
                    forming_all_ranges.append(_forming_entry)
                    # Display list: filtered by HTF bias alignment
                    if htf_bias == "bullish" and _fdir == "bearish":
                        continue
                    if htf_bias == "bearish" and _fdir == "bullish":
                        continue
                    forming.append(_forming_entry)
            forming.sort(key=lambda x: (x.get("tap3") or {}).get("idx", 0), reverse=True)

            # Session context (MSCE integration)
            try:
                from session_manipulation import get_session_info
                session_info = get_session_info()
            except ImportError:
                session_info = None

            out.update({
                "current_price": current_price,
                "htf_bias": htf_bias,
                "best_setup": best_setup,
                "best_score": best_score,
                "best_tf": best_tf_local,
                "forming": forming[:5],
                "forming_all_ranges": forming_all_ranges,
                "confirmed_ranges": confirmed_ranges,
                "timeframes": all_tf_results,
                "session": session_info,
                "eth_funnel": {
                    "confirmed_evaluated": _eth_funnel_confirmed,
                    "passed_eval": _eth_funnel_passed_eval,
                    "rejections": dict(sorted(_eth_funnel_rejections.items(), key=lambda x: -x[1])),
                    "l3_sub_failures": dict(sorted(_eth_l3_sub_failures.items(), key=lambda x: -x[1])),
                    "l3_traces": _eth_l3_traces,
                },
            })
            logger.info(
                f"[5B] _scan_single_symbol done ({time.time()-_t0:.1f}s) — "
                f"best_tf={best_tf_local}, best_score={best_score}, "
                f"confirmed_eval={_eth_funnel_confirmed}, passed={_eth_funnel_passed_eval}, "
                f"rejections={_eth_funnel_rejections}"
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

    # ----------------------------------------------------------------
    # RIG — Range Integrity Gate (5B)
    # ----------------------------------------------------------------
    # Canonical RIG evaluation is now in rig_engine.py (evaluate_rig_global).
    # Called directly from _scan_and_trade_locked() with real MSCE context.
    # No per-class RIG methods needed.

    # ----------------------------------------------------------------
    # TRADE ENTRY
    # ----------------------------------------------------------------

    def _enter_trade(self, schematic: Dict, evaluation: Dict, current_price: float, htf_bias: str,
                     symbol: str = DEFAULT_SYMBOL, timeframe: str = "unknown",
                     risk_multiplier: float = 1.0) -> Dict:
        """
        Open a new trade.

        ``risk_multiplier`` is supplied by the DD protection layer:
            1.0 = full risk (default)
            0.5 = soft throttle — halve the position size
        Callers must not pass 0.0 here; hard-blocked trades must be
        caught before calling _enter_trade.
        """
        direction = evaluation["direction"]
        stop_info = schematic.get("stop_loss", {})
        target_info = schematic.get("target", {})

        # Capture a fresh price snapshot at entry time so stale-entry checks
        # and market_price_at_entry reflect the actual moment of execution,
        # not the scan-start price that may be minutes old.
        try:
            _fresh_df = fetch_candles_sync(symbol, "1m", 2)
            entry_snapshot_price = float(_fresh_df.iloc[-1]["close"]) if _fresh_df is not None and len(_fresh_df) > 0 else current_price
        except Exception:
            entry_snapshot_price = current_price

        # Use the schematic's BOS level as entry (where a limit order would fill)
        # instead of current_price which may have displaced past the entry zone.
        # Falls back to entry_snapshot_price if no schematic entry price is available.
        schematic_entry = (schematic.get("entry") or {}).get("price")
        entry_price = schematic_entry if schematic_entry else entry_snapshot_price
        stop_price = stop_info.get("price")
        target_price = target_info.get("price")

        if not stop_price or not target_price:
            return {"error": "Missing stop or target price"}

        # Reject entries where the market has already breached the stop.
        # This prevents retroactive fills that are DOA (dead on arrival).
        if direction == "bullish" and entry_snapshot_price <= stop_price:
            logger.warning(
                "[5B] Rejecting long: market %.4f already at/below stop %.4f",
                entry_snapshot_price, stop_price,
            )
            return {"error": "Market already at/below stop — entry DOA"}
        if direction == "bearish" and entry_snapshot_price >= stop_price:
            logger.warning(
                "[5B] Rejecting short: market %.4f already at/above stop %.4f",
                entry_snapshot_price, stop_price,
            )
            return {"error": "Market already at/above stop — entry DOA"}

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

        # Position sizing — base risk 1%, then apply DD risk_multiplier
        risk_amount = self.state.balance * (RISK_PER_TRADE_PCT / 100) * risk_multiplier
        if direction == "bullish":
            sl_pct = abs((entry_price - stop_price) / entry_price) * 100
        else:
            sl_pct = abs((stop_price - entry_price) / entry_price) * 100
        if sl_pct <= 0:
            sl_pct = 1.0

        if risk_multiplier < 1.0:
            logger.info(
                "[5B] DD soft throttle applied: risk_multiplier=%.2f — "
                "risk_amount=$%.2f (base would be $%.2f)",
                risk_multiplier,
                risk_amount,
                self.state.balance * (RISK_PER_TRADE_PCT / 100),
            )

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

        # Stale entry guard — if market already displaced past TP1, the trade
        # has played out and entering retroactively would be inaccurate.
        # Uses entry_snapshot_price (fresh) instead of current_price (scan-start).
        if direction == "bearish" and entry_snapshot_price < tp1_price:
            return {"error": "Stale entry: price %.4f already below TP1 %.4f" % (entry_snapshot_price, tp1_price)}
        if direction == "bullish" and entry_snapshot_price > tp1_price:
            return {"error": "Stale entry: price %.4f already above TP1 %.4f" % (entry_snapshot_price, tp1_price)}

        trade = {
            "id": len(self.state.trade_history) + 1,
            "symbol": symbol,
            "timeframe": timeframe,
            "direction": direction,
            "model": evaluation.get("model", "unknown"),
            "entry_price": entry_price,
            "market_price_at_entry": entry_snapshot_price,
            "stop_price": stop_price,
            "target_price": target_price,
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
            # MSCE session context
            "session_context": _get_entry_session_context(evaluation["score"]),
        }

        self.state.current_trade = trade
        self.state.save()
        _mkt_delta = " (mkt=%.4f)" % entry_snapshot_price if abs(entry_snapshot_price - entry_price) > 0.01 else ""
        logger.info(
            "[5B] Entered %s @ %.4f%s | %s %s | SL=%.4f | TP=%.4f | Score=%s",
            direction, entry_price, _mkt_delta, symbol, timeframe,
            stop_price, target_price, evaluation["score"],
        )
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

        # ── Step 1: SL priority — if SL hit before TP1, stop takes precedence ──
        if not tp1_hit and hit_stop and hit_tp1:
            return self._close_trade(current_price, "stop_hit")

        # ── Step 2: Check TP1 trigger ──
        if hit_tp1:
            return self._take_partial_profit(current_price)

        # ── Step 3: Update trailing stop after TP1 ──
        if tp1_hit:
            trail_distance = abs(target_price - entry_price) * TRAIL_FACTOR
            if direction == "bullish":
                best = trade.get("highest_since_tp1", current_price)
                best = max(best, current_price)
                trade["highest_since_tp1"] = best
                new_trail = round(best - trail_distance, 2)
                if new_trail > stop_price:
                    trade["stop_price"] = new_trail
                    stop_price = new_trail  # update local var for exit check below
            else:
                best = trade.get("lowest_since_tp1", current_price)
                best = min(best, current_price)
                trade["lowest_since_tp1"] = best
                new_trail = round(best + trail_distance, 2)
                if new_trail < stop_price:
                    trade["stop_price"] = new_trail
                    stop_price = new_trail

            # Re-check SL after trailing update
            if direction == "bullish":
                hit_stop = current_price <= stop_price
            else:
                hit_stop = current_price >= stop_price

        # ── Step 4: Final exit checks (dual-hit: SL takes priority) ──
        if hit_stop and hit_target:
            if tp1_hit:
                if abs(stop_price - entry_price) < 0.01:
                    return self._close_trade(stop_price, "breakeven_after_tp1")
                else:
                    return self._close_trade(stop_price, "trailing_stop")
            return self._close_trade(stop_price, "stop_hit")

        if hit_target:
            return self._close_trade(current_price, "target_hit")

        if hit_stop:
            if tp1_hit:
                if abs(stop_price - entry_price) < 0.01:
                    return self._close_trade(stop_price, "breakeven_after_tp1")
                else:
                    return self._close_trade(stop_price, "trailing_stop")
            return self._close_trade(current_price, "stop_hit")

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
        # Initialize trailing stop tracker — best price seen since TP1
        if direction == "bullish":
            trade["highest_since_tp1"] = exit_price
        else:
            trade["lowest_since_tp1"] = exit_price

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

        # A stop hit after TP1 lands at break-even or trailing — still a net win
        # from the TP1 partial profit.  New exit reasons from trailing stop logic:
        # "breakeven_after_tp1", "trailing_stop" are both post-TP1 exits.
        is_win = reason in ("target_hit", "breakeven_after_tp1", "trailing_stop") or (
            reason == "stop_hit" and trade.get("tp1_hit", False)
        )
        position_size = trade.get("position_size", 0)
        pnl_dollars = position_size * (pnl_pct / 100)
        self.state.balance += pnl_dollars

        # ── Issue 5: deregister from portfolio on close ────────────────
        if self._portfolio is not None:
            _pm_close_position(self._portfolio, trade.get("symbol", DEFAULT_SYMBOL))

        # Issue 3: update peak on every close; clear all DD state on new equity high
        if self.state.balance > self.state.peak_balance:
            self.state.peak_balance = self.state.balance
            if self.state.dd_triggered_at is not None:
                logger.info(
                    "[5B] DD reset: equity set new high $%.2f — clearing hard-block timer and trough",
                    self.state.balance,
                )
                self.state.dd_triggered_at = None
            self.state.dd_trough_balance = None  # clear trough — back above previous peak

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
        trade_sym = self.state.current_trade.get("symbol", DEFAULT_SYMBOL)
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
