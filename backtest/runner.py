"""
backtest/runner.py — Stateful Walk-Forward Backtest Engine
============================================================
Deterministic, HPB-enforced backtesting engine. Walks forward through
historical candle data step-by-step, running the full gate pipeline
(MSCE → 1A → 1B → 1C → RCM → RIG → 1D) at each step.

Zero changes to live trading code. Imports live modules directly.

NON-NEGOTIABLE RULES:
- Closed candles ONLY (open_time < current_time)
- BOS = close-based only
- One signal per symbol per step max
- NO future leakage
"""

import argparse
import hashlib
import json
import logging
import sys
import time as _time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from backtest.config import (
    DEFAULT_LEVERAGE,
    ENTRY_THRESHOLD,
    EXECUTION_SLIPPAGE_PCT,
    FEE_PCT,
    HTF_TIMEFRAME,
    LTF_BOS_TIMEFRAMES,
    MIN_BARS_BETWEEN_TRADES,
    MIN_PIVOT_CONFIRM,
    MIN_RR,
    MTF_TIMEFRAMES,
    RISK_PER_TRADE_PCT,
    STARTING_BALANCE,
    TP1_POSITION_CLOSE_PCT,
    TRAIL_FACTOR,
    VALID_TIMEFRAMES,
    WARMUP_DAYS,
    timeframe_to_seconds,
)
from backtest.db import (
    complete_run,
    create_run,
    create_schema,
    fail_run,
    get_connection,
    insert_signal,
    insert_trade,
)
from backtest.ingest import load_candles
from backtest.session import get_session

logger = logging.getLogger("backtest.runner")

# ── v12/v13/v14 filter constants ──────────────────────────────────────
# 15m quality gates (local overrides — global MIN_RR and threshold unchanged)
_MIN_RR_15M = 0.8            # tighter RR for 15m entries
_MIN_RR_15M_RELAXED = 0.7    # fallback for re-run if 15m trade count < 3
_MIN_RANGE_PCT_15M = 0.003   # range must be >= 0.3% of price on 15m

# Model 3 quality gates
_TREND_LOOKBACK = 50         # candles to measure slope and volatility over
_TREND_SLOPE_FLOOR = 0.003   # v14: raised from 0.0015 — filters weaker continuation trends
_TREND_VOL_MULTIPLIER = 0.5  # slope threshold = max(floor, vol * multiplier)
_MODEL3_MAX_DISTANCE_PCT = 0.015  # entry must be within 1.5% of range midpoint

# v14: global displacement quality gate (applied after RR filter, before score)
_MIN_DISPLACEMENT = 0.50     # minimum local_displacement for any signal
_MIN_DISPLACEMENT_15M = 0.65 # stricter displacement floor for 15m specifically

# v14: hard score floor (below threshold trades are removed before score gate)
_MIN_SCORE_HARD = 65         # score 57-64 confirmed losers — blocked regardless of threshold

# Issue 3: 3-tier drawdown control (mirror of decision_engine_v2.py constants)
# DO NOT change these values without a matching change in decision_engine_v2.py.
_DD_SOFT_THRESHOLD = 0.02    # soft throttle: halve position size when DD ≥ 2%
_DD_HARD_THRESHOLD = 0.04    # hard block: skip new entries entirely when DD ≥ 4%
_DD_RISK_SCALE = 0.5         # position size multiplier applied during soft throttle
_DD_RESET_HOURS = 72         # minimum hours in hard block before reset is even evaluated
_DD_RECOVERY_PCT = 0.5       # equity must recover ≥ 50% of peak→trough gap before reset
                              # e.g. peak=$10k, trough=$9.6k, must recover to $9.8k

# Optional 15m NY overtrade guard — set True for re-run only if trade count > 55
_ENABLE_15M_NY_OVERTRADE_FILTER = False


def _compute_volatility(closes, lookback: int = _TREND_LOOKBACK) -> float:
    """Return average absolute 1-bar return over the last `lookback` bars."""
    if len(closes) < lookback:
        return 0.0
    returns = [
        abs((float(closes[i]) - float(closes[i - 1])) / float(closes[i - 1]))
        for i in range(-lookback + 1, 0)
        if float(closes[i - 1]) > 0
    ]
    return sum(returns) / len(returns) if returns else 0.0


def _is_trending_environment(closes, lookback: int = _TREND_LOOKBACK) -> tuple:
    """Return (trend_ok: bool, slope: float, min_slope: float).

    Uses volatility-adjusted threshold: min_slope = max(floor, vol * multiplier).
    This prevents the gate from being trivially easy in low-vol regimes or
    impossibly tight in high-vol regimes.
    """
    if len(closes) < lookback:
        return False, 0.0, _TREND_SLOPE_FLOOR
    start = float(closes[-lookback])
    end = float(closes[-1])
    if start <= 0:
        return False, 0.0, _TREND_SLOPE_FLOOR
    slope = (end - start) / start
    vol = _compute_volatility(closes, lookback)
    min_slope = max(_TREND_SLOPE_FLOOR, vol * _TREND_VOL_MULTIPLIER)
    return abs(slope) >= min_slope, slope, min_slope


def _range_size_pct(range_info, current_price: float) -> float:
    """Return (range_high - range_low) / current_price, or 0.0 on bad input."""
    if not isinstance(range_info, dict) or current_price <= 0:
        return 0.0
    r_high = range_info.get("high") or 0
    r_low = range_info.get("low") or 0
    if r_high <= r_low:
        return 0.0
    return (r_high - r_low) / current_price


# ── Backtest State ────────────────────────────────────────────────────

@dataclass
class OpenTrade:
    """An active backtest trade."""
    trade_num: int
    symbol: str
    timeframe: str
    direction: str
    model: str
    entry_price: float
    stop_price: float
    target_price: float
    tp1_price: Optional[float] = None
    tp1_hit: bool = False
    position_size: float = 0.0
    original_position_size: float = 0.0  # before TP1 partial close
    risk_amount: float = 0.0
    leverage: int = DEFAULT_LEVERAGE
    rr: float = 0.0
    entry_score: int = 0
    entry_reasons: list = field(default_factory=list)
    mfe: float = 0.0
    mae: float = 0.0
    opened_at: Optional[datetime] = None
    # Slippage-adjusted prices
    effective_entry: float = 0.0
    original_stop_price: float = 0.0     # before BE adjustment
    # TP1 / trailing state
    realized_pnl: float = 0.0           # accumulated from partial exits
    highest_since_tp1: float = 0.0      # for trailing stop (longs)
    lowest_since_tp1: float = float('inf')  # for trailing stop (shorts)


@dataclass
class BacktestState:
    """Full state of a backtest run."""
    current_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    open_trade: Optional[OpenTrade] = None
    trade_history: list = field(default_factory=list)
    equity: float = STARTING_BALANCE
    peak_equity: float = STARTING_BALANCE
    drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    trade_count: int = 0
    wins: int = 0
    losses: int = 0
    # HPB state
    last_htf_bias: str = "neutral"
    # Range context keyed by timeframe to prevent leaking range age across TFs
    active_range_context: dict = field(default_factory=dict)   # tf -> range dict
    range_first_seen: dict = field(default_factory=dict)       # tf -> datetime
    last_signal_time: Optional[datetime] = None
    last_trade_close_step: int = 0  # step index of last trade close
    current_step: int = 0
    # BOS fingerprint dedup: prevents re-entering the same schematic
    # Key: (tf, model, direction, bos_idx) — cleared after 48h or when BOS changes
    traded_bos_fingerprints: dict = field(default_factory=dict)  # fp -> timestamp traded
    # v13 funnel diagnostics — accumulated across the full run
    model3_detected: int = 0    # Model_3 schematics that entered gate pipeline
    model3_tf_pass: int = 0     # passed TF restriction (1h only)
    model3_trend_pass: int = 0  # passed adaptive trend gate
    model3_trades: int = 0      # resulted in a TAKE trade
    signals_15m_detected: int = 0  # 15m schematics that entered gate pipeline
    signals_15m_passed: int = 0    # survived all 15m-specific gates
    trades_15m: int = 0            # resulted in a TAKE trade
    # v14: DD protection state — tracks when protection first triggered
    dd_protection_triggered_at: Optional[datetime] = None  # None = not in protection
    # Lowest equity recorded since hard block first triggered.
    # Required for hybrid reset: reset only when time AND partial recovery both met.
    dd_trough_equity: Optional[float] = None  # None = not in hard block


# ── Multi-TF Synchronization ─────────────────────────────────────────

def get_last_closed(df: pd.DataFrame, tf: str, current_time: datetime) -> pd.DataFrame:
    """
    Return only candles whose timeframe period has fully closed
    relative to current_time. Prevents partial HTF candle leakage.
    """
    tf_seconds = timeframe_to_seconds(tf)
    current_ts = current_time.timestamp()
    aligned_ts = current_ts - (current_ts % tf_seconds)
    aligned_time = datetime.fromtimestamp(aligned_ts, tz=timezone.utc)
    return df[df["open_time"] < aligned_time].copy()


# ── Execution Helpers ─────────────────────────────────────────────────

def apply_slippage(price: float, direction: str, is_entry: bool) -> float:
    """Apply execution slippage (adverse direction)."""
    if is_entry:
        # Entry: LONG buys higher, SHORT sells lower
        if direction == "bullish":
            return price * (1 + EXECUTION_SLIPPAGE_PCT)
        else:
            return price * (1 - EXECUTION_SLIPPAGE_PCT)
    else:
        # Exit: LONG sells lower, SHORT buys higher
        if direction == "bullish":
            return price * (1 - EXECUTION_SLIPPAGE_PCT)
        else:
            return price * (1 + EXECUTION_SLIPPAGE_PCT)


def apply_fees(pnl: float, position_size: float) -> float:
    """Deduct round-trip fees from P&L."""
    total_fee = position_size * FEE_PCT * 2  # entry + exit
    return pnl - total_fee


def calculate_position_size(equity: float, sl_pct: float) -> Tuple[float, float]:
    """
    Calculate position size and risk amount.
    Returns (position_size, risk_amount).
    """
    risk_amount = equity * (RISK_PER_TRADE_PCT / 100)
    if sl_pct <= 0:
        return 0.0, risk_amount
    position_size = (risk_amount / sl_pct) * 100
    return position_size, risk_amount


def update_mfe_mae(trade: OpenTrade, high: float, low: float):
    """Update direction-aware MFE/MAE for an open trade."""
    entry = trade.effective_entry
    if trade.direction == "bullish":
        trade.mfe = max(trade.mfe, high - entry)
        trade.mae = min(trade.mae, low - entry)
    else:
        trade.mfe = max(trade.mfe, entry - low)
        trade.mae = min(trade.mae, entry - high)


def check_trade_exit(
    trade: OpenTrade, high: float, low: float, close: float,
    tp1_close_pct: float = TP1_POSITION_CLOSE_PCT,
    trail_factor: float = TRAIL_FACTOR,
) -> Optional[Tuple[str, float]]:
    """
    Check if an open trade hits TP1, TP2/target, or SL on this candle.
    Handles TP1 partial close, breakeven stop, and trailing stop.

    Returns (exit_reason, exit_price) or None.
    Intra-candle conflict: if both TP and SL are hit, assume worst case (SL first).

    TP1 Logic:
    - When price reaches tp1_price: close 50% at TP1, move SL to breakeven
    - After TP1: activate trailing stop at 50% of (target - entry) distance
    - Final exit: target_hit (remaining 50%) or stop_hit (breakeven/trailing)
    """
    # ── Pre-check: SL takes precedence over TP1 on same candle ────────
    # Compute SL hit BEFORE any state mutation. If both SL and TP1 are
    # reachable on this candle, the conservative assumption is SL hit first.
    if not trade.tp1_hit and trade.tp1_price is not None:
        if trade.direction == "bullish":
            original_sl_hit = low <= trade.stop_price
        else:
            original_sl_hit = high >= trade.stop_price
        if original_sl_hit:
            return "stop_hit", trade.stop_price

    # ── Step 1: Check TP1 trigger (if not already hit) ──────────────
    if not trade.tp1_hit and trade.tp1_price is not None:
        tp1_triggered = False
        if trade.direction == "bullish":
            tp1_triggered = high >= trade.tp1_price
        else:
            tp1_triggered = low <= trade.tp1_price

        if tp1_triggered:
            # TP1 hit — close partial position
            tp1_exit_price = trade.tp1_price
            partial_size = trade.original_position_size * tp1_close_pct

            # Calculate P&L on the partial close
            if trade.direction == "bullish":
                partial_pnl = (tp1_exit_price - trade.effective_entry) * (partial_size / trade.effective_entry)
            else:
                partial_pnl = (trade.effective_entry - tp1_exit_price) * (partial_size / trade.effective_entry)

            # Deduct fees on the partial close only
            partial_fee = partial_size * FEE_PCT
            trade.realized_pnl += (partial_pnl - partial_fee)

            # Reduce remaining position
            trade.position_size = trade.original_position_size - partial_size
            trade.tp1_hit = True

            # Move SL to breakeven (entry price)
            trade.stop_price = trade.effective_entry

            # Initialize trailing price tracker
            trade.highest_since_tp1 = high
            trade.lowest_since_tp1 = low

            logger.debug(
                f"  TP1 HIT #{trade.trade_num}: partial PnL=${partial_pnl - partial_fee:.2f}, "
                f"remaining size={trade.position_size:.2f}, SL -> BE @ {trade.effective_entry:.2f}"
            )

    # ── Step 2: Update trailing stop (after TP1) ────────────────────
    if trade.tp1_hit:
        trail_distance = abs(trade.target_price - trade.effective_entry) * trail_factor

        if trade.direction == "bullish":
            trade.highest_since_tp1 = max(trade.highest_since_tp1, high)
            trailing_stop = trade.highest_since_tp1 - trail_distance
            # Only ratchet up, never down
            trade.stop_price = max(trade.stop_price, trailing_stop)
        else:
            trade.lowest_since_tp1 = min(trade.lowest_since_tp1, low)
            trailing_stop = trade.lowest_since_tp1 + trail_distance
            # Only ratchet down, never up
            trade.stop_price = min(trade.stop_price, trailing_stop)

    # ── Step 3: Check SL and TP on remaining position ───────────────
    sl_hit = False
    tp_hit = False

    if trade.direction == "bullish":
        sl_hit = low <= trade.stop_price
        tp_hit = high >= trade.target_price
    else:
        sl_hit = high >= trade.stop_price
        tp_hit = low <= trade.target_price

    if sl_hit and tp_hit:
        # Worst case: SL first
        if trade.tp1_hit:
            return "breakeven_after_tp1", trade.stop_price
        return "stop_hit", trade.stop_price

    if sl_hit:
        if trade.tp1_hit:
            # Determine if this is breakeven or trailing stop
            if abs(trade.stop_price - trade.effective_entry) < 0.01:
                return "breakeven_after_tp1", trade.stop_price
            else:
                return "trailing_stop", trade.stop_price
        return "stop_hit", trade.stop_price

    if tp_hit:
        return "target_hit", trade.target_price

    return None


# ── Gate Pipeline ─────────────────────────────────────────────────────

def _get_live_modules():
    """Import live modules once, cache them."""
    if not hasattr(_get_live_modules, "_cache"):
        from tct_schematics import detect_tct_schematics
        from pivot_cache import PivotCache
        from decision_tree_bridge import DecisionTreeEvaluator
        from hpb_rig_validator import range_integrity_validator
        from session_manipulation import apply_session_multiplier
        _get_live_modules._cache = {
            "detect": detect_tct_schematics,
            "PivotCache": PivotCache,
            "evaluator": DecisionTreeEvaluator(),
            "rig": range_integrity_validator,
            "msce": apply_session_multiplier,
        }
    return _get_live_modules._cache


# Max candles to pass to detection (avoids processing entire history)
DETECTION_WINDOW = 200

# Max seconds per step before skipping detection
STEP_TIMEOUT_SECONDS = 10


def run_gate_pipeline(
    state: BacktestState,
    candles_by_tf: Dict[str, pd.DataFrame],
    current_price: float,
    current_time: datetime,
    run_id: int,
    conn,
    replay_mode: bool = False,
    entry_threshold: int = ENTRY_THRESHOLD,
    min_rr: float = MIN_RR,
) -> Optional[dict]:
    """
    Run the full HPB gate pipeline at the current step.
    Logs every signal (TAKE or SKIP) to the database.

    Gate order: MSCE → 1A → 1B → 1C → RCM → RIG → 1D

    Returns signal dict if a trade should be taken, None otherwise.
    """
    try:
        mods = _get_live_modules()
    except ImportError as e:
        logger.error(f"Failed to import live module: {e}")
        return None

    detect_tct_schematics = mods["detect"]
    PivotCache = mods["PivotCache"]
    evaluator = mods["evaluator"]
    range_integrity_validator = mods["rig"]
    apply_session_multiplier = mods["msce"]

    # ── Signal deduplication ──────────────────────────────────────
    if state.last_signal_time == current_time:
        return None

    # ── Trade collision ───────────────────────────────────────────
    if state.open_trade is not None:
        return None

    # ── Trade cooldown ────────────────────────────────────────────
    if (state.current_step - state.last_trade_close_step) < MIN_BARS_BETWEEN_TRADES:
        if state.last_trade_close_step > 0:
            return None

    # ── MSCE Session ──────────────────────────────────────────────
    session_info = get_session(current_time)
    session_name = session_info["session"]
    session_multiplier = session_info["confidence_multiplier"]

    # ── HTF Bias (Gate 1A) ────────────────────────────────────────
    # Try 1D first, fall back to 4H for more pivot data
    htf_bias = "neutral"
    for htf_candidate in [HTF_TIMEFRAME, "4h"]:
        htf_df = candles_by_tf.get(htf_candidate)
        if htf_df is None or len(htf_df) < MIN_PIVOT_CONFIRM:
            continue
        try:
            from market_structure import find_6cr_pivots, confirm_structure_points, classify_trend
            pivots = find_6cr_pivots(htf_df)
            ph = pivots.get("highs", [])
            pl = pivots.get("lows", [])
            if len(ph) < 2 and len(pl) < 2:
                continue
            ms_points = confirm_structure_points(htf_df, ph, pl)
            htf_bias = classify_trend(
                ms_points.get("ms_highs", []),
                ms_points.get("ms_lows", []),
            )
            if htf_bias not in ("neutral", "ranging"):
                break
        except Exception as e:
            logger.debug(f"HTF bias detection error on {htf_candidate}: {e}")
            continue

    if htf_bias == "neutral":
        htf_bias = state.last_htf_bias
    else:
        state.last_htf_bias = htf_bias

    gate_1a_pass = htf_bias != "neutral"

    # ── Scan MTF timeframes for schematics ────────────────────────
    best_signal = None
    best_score = 0

    import signal as _signal
    import threading

    for tf in MTF_TIMEFRAMES:
        df_tf_full = candles_by_tf.get(tf)
        if df_tf_full is None or len(df_tf_full) < MIN_PIVOT_CONFIRM:
            continue

        # Limit detection window for performance
        df_tf = df_tf_full.tail(DETECTION_WINDOW).reset_index(drop=True)

        try:
            pc = PivotCache(df_tf, lookback=3)
            # Run detection with timeout via thread
            det_result = [None]
            det_error = [None]

            def _detect():
                try:
                    det_result[0] = detect_tct_schematics(df_tf, [], pivot_cache=pc)
                except Exception as e:
                    det_error[0] = e

            t = threading.Thread(target=_detect, daemon=True)
            t.start()
            t.join(timeout=STEP_TIMEOUT_SECONDS)
            if t.is_alive():
                logger.debug(f"Detection timeout on {tf} (>{STEP_TIMEOUT_SECONDS}s), skipping")
                continue
            if det_error[0]:
                raise det_error[0]
            det = det_result[0]
            if det is None:
                continue

            all_schematics = (
                det.get("accumulation_schematics", [])
                + det.get("distribution_schematics", [])
            )
        except Exception as e:
            logger.debug(f"Detection error on {tf}: {e}")
            continue

        # Dedup: keep only the schematic with the most recent BOS
        # per model/direction to avoid re-evaluating stale ranges
        best_by_key: Dict[str, dict] = {}
        for s in all_schematics:
            if not isinstance(s, dict) or not s.get("is_confirmed"):
                continue
            key = f"{s.get('direction', '')}_{s.get('model', s.get('schematic_type', ''))}"
            bos_info = s.get("bos_confirmation") or {}
            bos_idx = bos_info.get("bos_idx", -1) or -1
            if key not in best_by_key or bos_idx > (best_by_key[key].get("bos_confirmation") or {}).get("bos_idx", -1):
                best_by_key[key] = s
        deduped_schematics = list(best_by_key.values())

        for schematic in deduped_schematics:

            # Evaluate through decision tree
            try:
                eval_result = evaluator.evaluate_schematic(
                    schematic, htf_bias, current_price,
                    total_candles=len(df_tf),
                    candle_df=df_tf,
                )
            except Exception as e:
                logger.debug(f"Evaluation error: {e}")
                continue

            score = eval_result.get("score", 0)
            direction = eval_result.get("direction", "unknown")
            model = eval_result.get("model", "unknown")
            rr = eval_result.get("rr", 0)
            reasons = eval_result.get("reasons", [])

            # TEMP DIAGNOSTIC: log why score=0 for first N signals
            if score == 0 and not hasattr(state, '_diag_count'):
                state._diag_count = 0
            if score == 0 and getattr(state, '_diag_count', 0) < 30:
                state._diag_count = getattr(state, '_diag_count', 0) + 1
                bos_info = schematic.get("bos_confirmation") or {}
                logger.warning(
                    f"DIAG score=0 #{state._diag_count}: time={current_time} tf={tf} model={model} "
                    f"dir={direction} confirmed={schematic.get('is_confirmed')} "
                    f"bos_idx={bos_info.get('bos_idx')} total_candles={len(df_tf)} "
                    f"reasons={reasons} phase_results={list(eval_result.get('phase_results', {}).keys())}"
                )

            # ── RCM (Range Context) ──────────────────────────────
            rcm_score = schematic.get("quality_score", 0.0)
            range_info = schematic.get("range", {})
            rcm_valid = rcm_score >= 0.6

            # Compute range_duration_hours from range candle data
            range_duration_hours = 0.0
            if isinstance(range_info, dict):
                # Try explicit duration first
                range_duration_hours = range_info.get("duration_hours", 0)
                # If not available, compute from range start/end indices
                if range_duration_hours == 0:
                    r_start_idx = range_info.get("start_idx")
                    r_end_idx = range_info.get("end_idx")
                    if r_start_idx is not None and r_end_idx is not None and len(df_tf) > 0:
                        try:
                            start_time = df_tf.iloc[max(0, int(r_start_idx))]["open_time"]
                            end_time = df_tf.iloc[min(len(df_tf)-1, int(r_end_idx))]["open_time"]
                            range_duration_hours = (end_time - start_time).total_seconds() / 3600
                        except (IndexError, KeyError):
                            pass
                    # Fallback: estimate from num_candles * timeframe
                    if range_duration_hours == 0:
                        num_candles = range_info.get("num_candles", range_info.get("candle_count", 0))
                        if num_candles > 0:
                            try:
                                tf_secs = timeframe_to_seconds(tf)
                                range_duration_hours = (num_candles * tf_secs) / 3600
                            except ValueError:
                                pass

            # Compute local_displacement from range high/low vs current price
            local_displacement = 0.0
            if isinstance(range_info, dict):
                local_displacement = range_info.get("displacement", 0.0)
                if local_displacement == 0.0:
                    r_high = range_info.get("high", 0)
                    r_low = range_info.get("low", 0)
                    if r_high and r_low and r_high > r_low:
                        range_size = r_high - r_low
                        if direction == "bullish":
                            # How far price has moved from range low relative to range size
                            local_displacement = (current_price - r_low) / range_size
                        else:
                            local_displacement = (r_high - current_price) / range_size
                        local_displacement = max(0.0, min(1.0, local_displacement))

            # Range persistence lock + duration tracking (keyed by TF to avoid cross-TF leakage)
            if state.active_range_context.get(tf) and rcm_valid:
                # Reuse existing range — compute duration from first detection
                if state.range_first_seen.get(tf):
                    range_duration_hours = (current_time - state.range_first_seen[tf]).total_seconds() / 3600
            elif rcm_valid:
                # New range detected — record first-seen time
                state.range_first_seen[tf] = current_time
                state.active_range_context[tf] = {
                    "score": rcm_score,
                    "duration_hours": range_duration_hours,
                }
            else:
                # Range invalidated — reset for this TF only
                state.active_range_context.pop(tf, None)
                state.range_first_seen.pop(tf, None)

            # ── RIG (Range Integrity Gate) ────────────────────────
            previous_htf_bias = state.last_htf_bias
            rig_context = {
                "gates": {
                    "RCM": {
                        "valid": rcm_valid,
                        "range_duration_hours": range_duration_hours,
                    },
                    "MSCE": {
                        "session_bias": session_info.get("bias", "neutral"),
                        "session": session_name,
                    },
                    "1A": {"bias": htf_bias},
                    "1D": {"score": score},
                },
                "local_range_displacement": local_displacement,
            }

            rig_result = range_integrity_validator(rig_context)
            rig_status = rig_result.get("status", "VALID")
            rig_reason = rig_result.get("reason")

            # Apply session multiplier to confidence
            msce_result = apply_session_multiplier(float(score), current_time)
            execution_confidence = msce_result.get("adjusted_confidence", float(score))

            # ── Build signal record ───────────────────────────────
            stop_info = schematic.get("stop_loss", {})
            target_info = schematic.get("target", {})
            entry_price = schematic.get("entry", {}).get("price", current_price)
            stop_price = stop_info.get("price", 0) if isinstance(stop_info, dict) else 0
            target_price = target_info.get("price", 0) if isinstance(target_info, dict) else 0

            # Structure snapshot for debugging
            structure_state = {
                "last_bos_level": schematic.get("bos_confirmation", {}).get("bos_price"),
                "range_high": range_info.get("high") if isinstance(range_info, dict) else None,
                "range_low": range_info.get("low") if isinstance(range_info, dict) else None,
                "pivot_count": len(PivotCache(df_tf, lookback=3).get_pivot_highs()),
            }

            # ── Compute actual R:R from price levels ────────────────
            actual_rr = 0.0
            if entry_price and stop_price and target_price:
                sl_distance = abs(entry_price - stop_price)
                tp_distance = abs(target_price - entry_price)
                if sl_distance > 0:
                    actual_rr = tp_distance / sl_distance

            # ── RIG Debug Logging ───────────────────────────────────
            rig_should_block = (
                range_info is not None
                and isinstance(range_info, dict)
                and rcm_valid
                and range_duration_hours >= 24
                and local_displacement < 0.25
                and session_info.get("bias", "neutral") != htf_bias
            )
            logger.debug(
                f"RIG_DEBUG time={current_time} htf_bias={htf_bias} "
                f"session_bias={session_info.get('bias', 'neutral')} "
                f"range_valid={rcm_valid} duration={range_duration_hours:.1f}h "
                f"displacement={local_displacement:.3f} "
                f"rig_status={rig_status} rig_should_block={rig_should_block}"
            )

            # v13 funnel: count every Model_3 / 15m signal that reaches the gate
            if "Model_3" in model:
                state.model3_detected += 1
            if tf == "15m":
                state.signals_15m_detected += 1

            # ── Hybrid DD tier — computed per schematic, consistent per step ──
            # Evaluated BEFORE the gate chain so the elif can reference _risk_multiplier.
            # Reset requires BOTH: time elapsed AND partial equity recovery.
            _risk_multiplier = 1.0
            if state.peak_equity > 0:
                _step_dd = (state.peak_equity - state.equity) / state.peak_equity
                if _step_dd >= _DD_HARD_THRESHOLD:
                    # Record first trigger and initialise trough on first crossing.
                    if state.dd_protection_triggered_at is None:
                        state.dd_protection_triggered_at = current_time
                        state.dd_trough_equity = state.equity
                        logger.info(
                            "DD_PROTECTION | hard block triggered | dd=%.2f%% | trough=$%.2f",
                            _step_dd * 100, state.equity,
                        )
                    else:
                        # Keep trough at the lowest point seen since trigger.
                        if state.dd_trough_equity is None or state.equity < state.dd_trough_equity:
                            state.dd_trough_equity = state.equity

                    _hours_in_dd = (
                        current_time - state.dd_protection_triggered_at
                    ).total_seconds() / 3600

                    # Recovery fraction: how much of peak→trough gap has been reclaimed.
                    _recovery = 0.0
                    _trough = state.dd_trough_equity
                    if _trough is not None and state.peak_equity > _trough:
                        _recovery = (state.equity - _trough) / (state.peak_equity - _trough)

                    if _hours_in_dd >= _DD_RESET_HOURS and _recovery >= _DD_RECOVERY_PCT:
                        logger.info(
                            "DD_PROTECTION | reset: %.0fh elapsed, recovery=%.0f%% ≥ %.0f%% "
                            "(trough=$%.2f) — resetting peak $%.2f → $%.2f",
                            _hours_in_dd, _recovery * 100, _DD_RECOVERY_PCT * 100,
                            (_trough or 0), state.peak_equity, state.equity,
                        )
                        state.peak_equity = state.equity
                        state.dd_protection_triggered_at = None
                        state.dd_trough_equity = None
                        # _risk_multiplier stays 1.0 — full risk after reset
                    else:
                        _risk_multiplier = 0.0  # hard block remains
                        if _hours_in_dd >= _DD_RESET_HOURS:
                            # Time elapsed but recovery not sufficient — log diagnostic.
                            logger.debug(
                                "DD_PROTECTION | %.0fh elapsed but recovery=%.2f%% < %.0f%% "
                                "(trough=$%.2f equity=$%.2f) — blocked",
                                _hours_in_dd, _recovery * 100, _DD_RECOVERY_PCT * 100,
                                (_trough or 0), state.equity,
                            )
                elif _step_dd >= _DD_SOFT_THRESHOLD:
                    _risk_multiplier = _DD_RISK_SCALE  # soft throttle: 50% risk
                    logger.info(
                        "DD_PROTECTION | soft throttle | dd=%.2f%% — risk scaled to %.0f%%",
                        _step_dd * 100, _DD_RISK_SCALE * 100,
                    )

            # Determine failure code
            failure_code = None
            skip_reason = None
            final_decision = "TAKE"

            if rig_status == "BLOCK":
                final_decision = "SKIP"
                skip_reason = "RIG_BLOCK"
                failure_code = "FAIL_RIG_COUNTER_BIAS"
                execution_confidence = 0.0
                state.last_htf_bias = previous_htf_bias  # RIG bias freeze
            elif not gate_1a_pass:
                final_decision = "SKIP"
                skip_reason = "NO_HTF_BIAS"
                failure_code = "FAIL_1A_BIAS"
            elif not rcm_valid:
                final_decision = "SKIP"
                skip_reason = "RCM_INVALID"
                failure_code = "FAIL_RCM_DURATION"
            elif actual_rr < min_rr:
                final_decision = "SKIP"
                skip_reason = "RR_TOO_LOW ({:.2f} < {:.2f})".format(actual_rr, min_rr)
                failure_code = "FAIL_RR_FILTER"

            # ── Issue 3: DD hard block ─────────────────────────────────────
            # Tier computed above. 0.0 = hard block; 0.5 = soft throttle (trade allowed,
            # position scaled); 1.0 = no DD concern.
            elif _risk_multiplier == 0.0:
                final_decision = "SKIP"
                skip_reason = (
                    f"DD_HARD_BLOCK (dd={_step_dd * 100:.2f}%"
                    f" >= {_DD_HARD_THRESHOLD * 100:.0f}%)"
                )
                failure_code = "FAIL_DD_PROTECTION"

            # ── v14: global displacement quality gate ──────────────────
            elif local_displacement < _MIN_DISPLACEMENT:
                final_decision = "SKIP"
                skip_reason = "LOW_DISPLACEMENT ({:.3f} < {:.2f})".format(local_displacement, _MIN_DISPLACEMENT)
                failure_code = "FAIL_LOW_DISPLACEMENT"

            # ── v12/v14: 15m hardening gates ──────────────────────────
            elif tf == "15m" and actual_rr < _MIN_RR_15M:
                final_decision = "SKIP"
                skip_reason = "RR_15M_STRICT ({:.2f} < {:.2f})".format(actual_rr, _MIN_RR_15M)
                failure_code = "FAIL_RR_15M_STRICT"
                logger.info(
                    "15M_FILTER | rr=%.2f (need %.2f) | session=%s",
                    actual_rr, _MIN_RR_15M, session_name,
                )
            elif tf == "15m" and session_name == "asia":
                final_decision = "SKIP"
                skip_reason = "15M_ASIA_FILTER"
                failure_code = "FAIL_15M_ASIA_FILTER"
                logger.info(
                    "15M_FILTER | session=Asia blocked | rr=%.2f", actual_rr,
                )
            elif tf == "15m" and _range_size_pct(range_info, current_price) < _MIN_RANGE_PCT_15M:
                _rpct = _range_size_pct(range_info, current_price)
                final_decision = "SKIP"
                skip_reason = "RANGE_TOO_SMALL_15M ({:.4f} < {:.4f})".format(_rpct, _MIN_RANGE_PCT_15M)
                failure_code = "FAIL_RANGE_TOO_SMALL_15M"
                logger.info(
                    "15M_FILTER | range_pct=%.4f (need %.4f) | session=%s | rr=%.2f",
                    _rpct, _MIN_RANGE_PCT_15M, session_name, actual_rr,
                )
            elif tf == "15m" and session_name == "new_york" and _ENABLE_15M_NY_OVERTRADE_FILTER:
                final_decision = "SKIP"
                skip_reason = "15M_NY_OVERTRADE_GUARD"
                failure_code = "FAIL_15M_NY_OVERTRADE"
                logger.info("15M_FILTER | NY overtrade guard active | rr=%.2f", actual_rr)

            # ── v14: 15m stricter displacement gate (3A) ──────────────
            elif tf == "15m" and local_displacement < _MIN_DISPLACEMENT_15M:
                final_decision = "SKIP"
                skip_reason = "15M_LOW_DISPLACEMENT ({:.3f} < {:.2f})".format(local_displacement, _MIN_DISPLACEMENT_15M)
                failure_code = "FAIL_15M_LOW_DISPLACEMENT"
                logger.info(
                    "15M_FILTER | low_displacement=%.3f (need %.2f) | session=%s",
                    local_displacement, _MIN_DISPLACEMENT_15M, session_name,
                )

            # ── v14: 15m entry location gate (3B) ─────────────────────
            # Standalone if-guard (not elif) so it never short-circuits the
            # Model_2 block, score gates, or BOS dedup below.
            if final_decision == "TAKE" and tf == "15m" and isinstance(range_info, dict) and range_info.get("high") and range_info.get("low"):
                _r_high_loc = float(range_info["high"])
                _r_low_loc = float(range_info["low"])
                if _r_high_loc > _r_low_loc and entry_price > 0:
                    _pos = (entry_price - _r_low_loc) / (_r_high_loc - _r_low_loc)
                    _loc_fail = (
                        (direction == "bullish" and _pos > 0.4) or
                        (direction == "bearish" and _pos < 0.6)
                    )
                    if _loc_fail:
                        final_decision = "SKIP"
                        skip_reason = "15M_POOR_ENTRY_LOCATION (pos={:.3f}, dir={})".format(_pos, direction)
                        failure_code = "FAIL_15M_POOR_ENTRY_LOCATION"
                        logger.info(
                            "15M_FILTER | poor_entry_location pos=%.3f | dir=%s | session=%s",
                            _pos, direction, session_name,
                        )

            # ── v14: block Model_2 on 15m (3C) — confirmed weak ───────
            # Standalone if-guard: runs even when the location check above passed.
            if final_decision == "TAKE" and tf == "15m" and model == "Model_2":
                final_decision = "SKIP"
                skip_reason = "MODEL2_15M_BLOCK"
                failure_code = "FAIL_MODEL2_15M_BLOCK"
                logger.info("15M_FILTER | Model_2 blocked on 15m | session=%s", session_name)

            # ── v14: hard score floor (BEFORE model-specific branches) ──
            # Standalone guard so it applies to ALL models including Model_3,
            # which previously bypassed it via the elif chain.
            if final_decision == "TAKE" and score < _MIN_SCORE_HARD:
                final_decision = "SKIP"
                skip_reason = "SCORE_HARD_FLOOR ({} < {})".format(score, _MIN_SCORE_HARD)
                failure_code = "FAIL_SCORE_HARD_FLOOR"

            # ── v12/v13: Model 3 quality gates ────────────────────────
            if final_decision == "TAKE" and "Model_3" in model:
                if tf != "1h":
                    final_decision = "SKIP"
                    skip_reason = "MODEL3_TF_FILTER (tf={}, only 1h allowed)".format(tf)
                    failure_code = "FAIL_MODEL3_TF_FILTER"
                    logger.info(
                        "MODEL3_CHECK | tf=%s BLOCKED (1h only) | trend not evaluated", tf,
                    )
                else:
                    state.model3_tf_pass += 1
                    _closes = df_tf["close"].values if df_tf is not None and len(df_tf) > 0 else []
                    _trend_ok, _slope, _min_slope = _is_trending_environment(_closes)
                    logger.info(
                        "MODEL3_CHECK | tf=%s | trend_ok=%s | slope=%.4f | min_slope=%.4f",
                        tf, _trend_ok, _slope, _min_slope,
                    )
                    if not _trend_ok:
                        final_decision = "SKIP"
                        skip_reason = "MODEL3_NO_TREND (slope={:.4f} < adaptive {:.4f})".format(
                            abs(_slope), _min_slope
                        )
                        failure_code = "FAIL_MODEL3_NO_TREND"
                    else:
                        state.model3_trend_pass += 1
                        # Entry distance gate: reject if entry is > 1.5% from range midpoint
                        _r_high = range_info.get("high", 0) if isinstance(range_info, dict) else 0
                        _r_low = range_info.get("low", 0) if isinstance(range_info, dict) else 0
                        if _r_high > _r_low and entry_price > 0:
                            _range_mid = (_r_high + _r_low) / 2
                            _dist_pct = abs(entry_price - _range_mid) / _range_mid
                            if _dist_pct > _MODEL3_MAX_DISTANCE_PCT:
                                final_decision = "SKIP"
                                skip_reason = "MODEL3_EXTENDED (dist={:.4f} > {:.4f})".format(
                                    _dist_pct, _MODEL3_MAX_DISTANCE_PCT
                                )
                                failure_code = "FAIL_MODEL3_EXTENDED"
                                logger.info(
                                    "MODEL3_CHECK | tf=%s | EXTENDED dist=%.4f (max %.4f)",
                                    tf, _dist_pct, _MODEL3_MAX_DISTANCE_PCT,
                                )

            # ── score threshold ────────────────────────────────────────
            if final_decision == "TAKE" and score < entry_threshold:
                final_decision = "SKIP"
                skip_reason = "SCORE_BELOW_THRESHOLD ({} < {})".format(score, entry_threshold)
                failure_code = "FAIL_1D_SCORE"

            # ── BOS fingerprint dedup ──────────────────────────────────
            # Keyed on entry_price (rounded) + BOS price — both absolute and
            # stable across detection window shifts (unlike bos_idx).
            if final_decision == "TAKE":
                bos_info = schematic.get("bos_confirmation") or {}
                bos_price = round(float(bos_info.get("bos_price") or 0), 0)
                entry_snap = round(float(
                    schematic.get("entry", {}).get("price") or current_price
                ), 0)
                fp = (tf, model, direction, entry_snap, bos_price)
                fp_traded_at = state.traded_bos_fingerprints.get(fp)
                # Expire fingerprints older than 48 hours so valid re-entries are allowed
                if fp_traded_at is not None:
                    age_hours = (current_time - fp_traded_at).total_seconds() / 3600
                    if age_hours < 48:
                        final_decision = "SKIP"
                        skip_reason = "DUPLICATE_BOS (same schematic traded {:.0f}h ago)".format(age_hours)
                        failure_code = "FAIL_DUPLICATE_BOS"

            # v13: track 15m signals that survived all gates (including BOS dedup)
            if tf == "15m" and final_decision == "TAKE":
                state.signals_15m_passed += 1

            signal = {
                "signal_time": current_time,
                "price_at_signal": current_price,
                "timeframe": tf,
                "direction": direction,
                "model": model,
                "gate_1a_bias": htf_bias,
                "gate_1a_pass": gate_1a_pass,
                "gate_1b_pass": True,  # placeholder — USDT.D not yet integrated
                "gate_1c_pass": True,  # placeholder — alt alignment optional
                "rcm_score": rcm_score,
                "rcm_valid": rcm_valid,
                "range_duration_hours": range_duration_hours,
                "local_displacement": local_displacement,
                "htf_bias": htf_bias,
                "msce_session": session_name,
                "msce_confidence": execution_confidence,
                "session_bias": session_info.get("bias", "neutral"),
                "rig_status": rig_status,
                "rig_reason": rig_reason,
                "score_1d": score,
                "execution_confidence": execution_confidence,
                "latency_to_entry_seconds": 0,  # backtest has no latency
                "final_score": score,
                "final_decision": final_decision,
                "skip_reason": skip_reason,
                "failure_code": failure_code,
                "risk_multiplier": _risk_multiplier,
                "structure_state": structure_state,
                "entry_price": entry_price,
                "stop_price": stop_price,
                "target_price": target_price,
                "rr": actual_rr if actual_rr > 0 else rr,
                "schematic_json": {
                    # bos_confirmation sub-dict: {"bos_idx": int, "bos_price": float, ...}
                    "bos_idx":   (schematic.get("bos_confirmation") or {}).get("bos_idx"),
                    "bos_price": (schematic.get("bos_confirmation") or {}).get("bos_price"),
                    # tap dicts: {"price": float, "idx": int, "type": str, ...}
                    "tap1_price": (schematic.get("tap1") or {}).get("price"),
                    "tap1_idx":   (schematic.get("tap1") or {}).get("idx"),
                    "tap2_price": (schematic.get("tap2") or {}).get("price"),
                    "tap2_idx":   (schematic.get("tap2") or {}).get("idx"),
                    "tap3_price": (schematic.get("tap3") or {}).get("price"),
                    "tap3_idx":   (schematic.get("tap3") or {}).get("idx"),
                    # range sub-dict: {"high": float, "low": float, "equilibrium": float, ...}
                    "range_high": (schematic.get("range") or {}).get("high"),
                    "range_low":  (schematic.get("range") or {}).get("low"),
                    "sweep_type": schematic.get("sweep_type"),
                } if schematic else None,
            }

            # Log the signal
            insert_signal(conn, run_id, signal)
            state.last_signal_time = current_time

            if replay_mode:
                _print_replay_detail(signal, current_time)

            if final_decision == "TAKE" and score > best_score:
                best_score = score
                signal["_eval_result"] = eval_result
                signal["_schematic"] = schematic
                signal["_tf_df"] = df_tf
                best_signal = signal

    return best_signal


def _print_replay_detail(signal: dict, current_time: datetime):
    """Print detailed gate info for --replay-at mode."""
    print(f"\n{'='*60}")
    print(f"REPLAY @ {current_time.isoformat()}")
    print(f"{'='*60}")
    print(f"  Price:     {signal.get('price_at_signal', 0):.2f}")
    print(f"  TF:        {signal.get('timeframe')}")
    print(f"  Direction: {signal.get('direction')}")
    print(f"  Model:     {signal.get('model')}")
    print(f"  HTF Bias:  {signal.get('htf_bias')} (pass={signal.get('gate_1a_pass')})")
    print(f"  Session:   {signal.get('msce_session')} (conf={signal.get('msce_confidence', 0):.1f})")
    print(f"  RCM:       score={signal.get('rcm_score', 0):.2f} valid={signal.get('rcm_valid')}")
    print(f"  RIG:       {signal.get('rig_status')} — {signal.get('rig_reason', 'none')}")
    print(f"  Score 1D:  {signal.get('score_1d', 0)}")
    print(f"  Decision:  {signal.get('final_decision')}")
    if signal.get("skip_reason"):
        print(f"  Skip:      {signal['skip_reason']}")
    if signal.get("failure_code"):
        print(f"  Fail Code: {signal['failure_code']}")
    print(f"  R:R:       {signal.get('rr', 0):.2f}")
    print(f"  Entry:     {signal.get('entry_price', 0):.2f}")
    print(f"  SL:        {signal.get('stop_price', 0):.2f}")
    print(f"  TP:        {signal.get('target_price', 0):.2f}")
    if signal.get("structure_state"):
        ss = signal["structure_state"]
        print(f"  Structure: BOS={ss.get('last_bos_level')}, "
              f"Range=[{ss.get('range_low')}, {ss.get('range_high')}]")
    print()


# ── Main Backtest Loop ────────────────────────────────────────────────

def run_backtest(
    symbol: str = "BTCUSDT",
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    step_interval: str = "1h",
    starting_balance: float = STARTING_BALANCE,
    entry_threshold: Optional[int] = None,
    replay_at: Optional[datetime] = None,
    warmup_days: int = 0,
    conn=None,
    min_rr: Optional[float] = None,
    tp1_close_pct: Optional[float] = None,
    tp1_level_pct: Optional[float] = None,
    trail_factor: Optional[float] = None,
) -> dict:
    """
    Run the full walk-forward backtest.

    Args:
        symbol: Trading pair
        start_date: Backtest start
        end_date: Backtest end
        step_interval: Time between evaluation steps
        starting_balance: Initial equity
        entry_threshold: Override ENTRY_THRESHOLD from config
        replay_at: If set, run only this single timestamp in deep debug mode
        conn: DB connection (creates one if not provided)

    Returns:
        Summary dict with metrics.
    """
    # Resolve overrides (fall back to config defaults)
    effective_threshold = entry_threshold if entry_threshold is not None else ENTRY_THRESHOLD
    effective_min_rr = min_rr if min_rr is not None else MIN_RR
    effective_tp1_close_pct = tp1_close_pct if tp1_close_pct is not None else TP1_POSITION_CLOSE_PCT
    effective_tp1_level_pct = tp1_level_pct if tp1_level_pct is not None else 0.5  # default: midpoint
    effective_trail_factor = trail_factor if trail_factor is not None else TRAIL_FACTOR

    own_conn = conn is None
    if own_conn:
        conn = get_connection()

    if end_date is None:
        end_date = datetime.now(timezone.utc)
    if start_date is None:
        start_date = end_date - timedelta(days=90)

    if start_date.tzinfo is None:
        start_date = start_date.replace(tzinfo=timezone.utc)
    if end_date.tzinfo is None:
        end_date = end_date.replace(tzinfo=timezone.utc)

    step_seconds = timeframe_to_seconds(step_interval)
    replay_mode = replay_at is not None

    # Create run record
    config = {
        "symbol": symbol,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "step_interval": step_interval,
        "starting_balance": starting_balance,
        "entry_threshold": effective_threshold,
        "slippage_pct": EXECUTION_SLIPPAGE_PCT,
        "fee_pct": FEE_PCT,
        "min_pivot_confirm": MIN_PIVOT_CONFIRM,
        "min_bars_between_trades": MIN_BARS_BETWEEN_TRADES,
        "min_rr": effective_min_rr,
        "warmup_days": warmup_days,
        "tp1_close_pct": effective_tp1_close_pct,
        "tp1_level_pct": effective_tp1_level_pct,
        "trail_factor": effective_trail_factor,
        "engine_version": 15,  # v15: gate-chain elif→if fix (score floor now runs for Model_3), DD time-reset (72h), schematic_json real keys
    }

    run_id = create_run(
        conn, name=f"backtest_{symbol}_{step_interval}",
        start_date=start_date, end_date=end_date,
        step_interval=step_interval,
        starting_balance=starting_balance, config=config,
    )

    # Pre-load all candle data from DB
    logger.info("Loading candle data from database...")
    candles_all: Dict[str, pd.DataFrame] = {}
    # Need data before start_date for structure building
    lookback_start = start_date - timedelta(days=30)
    all_tfs = list(set([HTF_TIMEFRAME] + MTF_TIMEFRAMES + LTF_BOS_TIMEFRAMES))

    for tf in all_tfs:
        df = load_candles(conn, symbol, tf, lookback_start, end_date)
        candles_all[tf] = df
        logger.info(f"  Loaded {tf}: {len(df)} candles")

    # Initialize state
    state = BacktestState(
        current_time=start_date,
        equity=starting_balance,
        peak_equity=starting_balance,
    )

    # Determine step times
    if replay_at:
        step_times = [replay_at]
    else:
        step_times = []
        t = start_date
        while t < end_date:
            step_times.append(t)
            t += timedelta(seconds=step_seconds)

    total_steps = len(step_times)
    logger.info(f"Starting backtest: {total_steps} steps, "
                f"{start_date.date()} to {end_date.date()}")

    # Warmup: skip signal evaluation until warmup period is over
    warmup_end = start_date + timedelta(days=warmup_days) if warmup_days > 0 else start_date
    logger.info(f"Warmup period: {warmup_days} days (signals enabled after {warmup_end.date()})")

    # Pre-compute numpy arrays for fast price lookup from 1m/5m
    price_source_tf = "1m" if "1m" in candles_all and not candles_all["1m"].empty else "5m"
    price_df = candles_all.get(price_source_tf, pd.DataFrame())
    if not price_df.empty:
        price_times = price_df["open_time"].values  # numpy datetime64 array
        price_closes = price_df["close"].values.astype(float)
    else:
        price_times = None
        price_closes = None

    # Include step_interval so exit-checking can access its candles
    detection_tfs = list({HTF_TIMEFRAME, *MTF_TIMEFRAMES, step_interval})

    try:
        for step_idx, current_time in enumerate(step_times):
            state.current_time = current_time
            state.current_step = step_idx

            # Fast price lookup via binary search on pre-sorted 1m array
            # Use side="left" - 1 to ensure we only access fully-closed candles
            # (never the candle whose open_time == current_time)
            if price_times is not None:
                ct_np = pd.Timestamp(current_time).to_datetime64()
                idx = np.searchsorted(price_times, ct_np, side="left") - 1
                if idx < 0:
                    continue
                current_price = float(price_closes[idx])
            else:
                continue

            # Slice only detection TFs (not 1m/5m — too expensive)
            candles_by_tf: Dict[str, pd.DataFrame] = {}
            for tf in detection_tfs:
                df = candles_all.get(tf)
                if df is not None and not df.empty:
                    candles_by_tf[tf] = get_last_closed(df, tf, current_time)

            # ── Check open trade exit ─────────────────────────────
            if state.open_trade is not None:
                # Use the step interval candle for TP/SL checking
                step_tf = step_interval
                df_step = candles_by_tf.get(step_tf)
                if df_step is not None and not df_step.empty:
                    last_candle = df_step.iloc[-1]
                    h = float(last_candle["high"])
                    l = float(last_candle["low"])
                    c = float(last_candle["close"])

                    update_mfe_mae(state.open_trade, h, l)

                    exit_result = check_trade_exit(
                        state.open_trade, h, l, c,
                        tp1_close_pct=effective_tp1_close_pct,
                        trail_factor=effective_trail_factor,
                    )
                    if exit_result:
                        exit_reason, raw_exit_price = exit_result
                        _close_trade(state, raw_exit_price, exit_reason,
                                     current_time, conn, run_id)

            # ── Run gate pipeline (skip during warmup) ─────────────
            if state.open_trade is None and current_time >= warmup_end:
                signal = run_gate_pipeline(
                    state, candles_by_tf, current_price, current_time,
                    run_id, conn, replay_mode=replay_mode,
                    entry_threshold=effective_threshold,
                    min_rr=effective_min_rr,
                )
                if signal and signal.get("final_decision") == "TAKE":
                    # Record BOS fingerprint to prevent same-schematic re-entry.
                    # Use entry_price + BOS price (absolute, stable across window shifts).
                    sch = signal.get("_schematic") or {}
                    bos_info_fp = (sch.get("bos_confirmation") or {})
                    bos_fp = (
                        signal.get("timeframe"),
                        signal.get("model"),
                        signal.get("direction"),
                        round(float(signal.get("entry_price") or current_price), 0),
                        round(float(bos_info_fp.get("bos_price") or 0), 0),
                    )
                    state.traded_bos_fingerprints[bos_fp] = current_time
                    _trade_opened = _open_trade(state, signal, current_time, current_price,
                                               tp1_level_pct=effective_tp1_level_pct)
                    # v13: funnel trade counters — only increment if trade was actually opened
                    if _trade_opened:
                        if "Model_3" in signal.get("model", ""):
                            state.model3_trades += 1
                        if signal.get("timeframe") == "15m":
                            state.trades_15m += 1

            # Progress logging
            if step_idx > 0 and step_idx % 100 == 0:
                pct = step_idx / total_steps * 100
                logger.info(
                    f"Step {step_idx}/{total_steps} ({pct:.0f}%) — "
                    f"equity=${state.equity:.2f}, trades={state.trade_count}, "
                    f"dd={state.max_drawdown_pct:.2f}%"
                )

        # Close any remaining open trade at last price
        if state.open_trade is not None:
            last_price = current_price
            _close_trade(state, last_price, "backtest_end", end_date, conn, run_id)

        # v13: emit funnel diagnostics before finalizing
        logger.info(
            "MODEL3_FUNNEL | detected=%d | passed_tf=%d | passed_trend=%d | trades=%d",
            state.model3_detected, state.model3_tf_pass,
            state.model3_trend_pass, state.model3_trades,
        )
        logger.info(
            "15M_FUNNEL | detected=%d | passed_filters=%d | trades=%d",
            state.signals_15m_detected, state.signals_15m_passed, state.trades_15m,
        )

        # Complete run
        complete_run(
            conn, run_id,
            final_balance=state.equity,
            total_trades=state.trade_count,
            wins=state.wins,
            losses=state.losses,
            max_drawdown_pct=state.max_drawdown_pct,
        )

        summary = {
            "run_id": run_id,
            "final_balance": state.equity,
            "total_trades": state.trade_count,
            "wins": state.wins,
            "losses": state.losses,
            "win_rate": (state.wins / state.trade_count * 100) if state.trade_count > 0 else 0,
            "max_drawdown_pct": state.max_drawdown_pct,
            "pnl_pct": ((state.equity - starting_balance) / starting_balance) * 100,
        }
        logger.info(f"Backtest complete: {json.dumps(summary, indent=2)}")
        return summary

    except Exception as e:
        fail_run(conn, run_id, str(e))
        logger.error(f"Backtest failed: {e}", exc_info=True)
        raise
    finally:
        if own_conn:
            conn.close()


# ── Trade Management ──────────────────────────────────────────────────

def _open_trade(state: BacktestState, signal: dict, current_time: datetime,
                current_price: float, tp1_level_pct: float = 0.5):
    """Open a new trade from a qualifying signal."""
    direction = signal["direction"]
    entry_price = signal.get("entry_price", current_price)
    stop_price = signal["stop_price"]
    target_price = signal["target_price"]

    if not stop_price or not target_price:
        return

    # Apply slippage to entry
    effective_entry = apply_slippage(entry_price, direction, is_entry=True)

    # Calculate SL percentage for position sizing
    if direction == "bullish":
        sl_pct = abs(effective_entry - stop_price) / effective_entry * 100
    else:
        sl_pct = abs(stop_price - effective_entry) / effective_entry * 100

    if sl_pct <= 0:
        return

    position_size, risk_amount = calculate_position_size(state.equity, sl_pct)

    # Issue 3: apply DD risk multiplier (1.0 = full, 0.5 = soft throttle)
    risk_multiplier = signal.get("risk_multiplier", 1.0)
    if 0 < risk_multiplier < 1.0:
        position_size = round(position_size * risk_multiplier, 4)
        risk_amount = round(risk_amount * risk_multiplier, 4)
        logger.info(
            "DD soft throttle: risk_multiplier=%.2f applied — "
            "position_size=%.4f risk_amount=%.4f",
            risk_multiplier, position_size, risk_amount,
        )

    # Calculate TP1 price: configurable % of entry→target move
    tp1_price = signal.get("tp1_price")
    if tp1_price is None or tp1_price == 0:
        tp1_price = effective_entry + (target_price - effective_entry) * tp1_level_pct

    state.trade_count += 1
    state.open_trade = OpenTrade(
        trade_num=state.trade_count,
        symbol=signal.get("_schematic", {}).get("symbol", "BTCUSDT"),
        timeframe=signal["timeframe"],
        direction=direction,
        model=signal.get("model", "unknown"),
        entry_price=entry_price,
        stop_price=stop_price,
        target_price=target_price,
        tp1_price=tp1_price,
        position_size=position_size,
        original_position_size=position_size,
        risk_amount=risk_amount,
        leverage=DEFAULT_LEVERAGE,
        rr=signal.get("rr", 0),
        entry_score=signal.get("score_1d", 0),
        entry_reasons=signal.get("_eval_result", {}).get("reasons", []),
        mfe=0.0,
        mae=0.0,
        opened_at=current_time,
        effective_entry=effective_entry,
        original_stop_price=stop_price,
    )

    logger.debug(
        f"OPEN #{state.trade_count}: {direction} @ {effective_entry:.2f} "
        f"(SL={stop_price:.2f}, TP={target_price:.2f}, R:R={signal.get('rr', 0):.2f})"
    )
    return True


def _close_trade(state: BacktestState, raw_exit_price: float, exit_reason: str,
                 current_time: datetime, conn, run_id: int):
    """Close the current open trade and record it."""
    trade = state.open_trade
    if trade is None:
        return

    # Apply slippage to exit
    effective_exit = apply_slippage(raw_exit_price, trade.direction, is_entry=False)

    # Calculate raw P&L on REMAINING position (after any TP1 partial close)
    if trade.direction == "bullish":
        raw_pnl = (effective_exit - trade.effective_entry) * (trade.position_size / trade.effective_entry)
    else:
        raw_pnl = (trade.effective_entry - effective_exit) * (trade.position_size / trade.effective_entry)

    # Apply fees on remaining position exit
    remaining_fee = trade.position_size * FEE_PCT
    remaining_pnl = raw_pnl - remaining_fee

    # Total P&L = TP1 realized + remaining position P&L
    # Entry fee was already charged on original position at open, deducted here proportionally
    entry_fee = trade.original_position_size * FEE_PCT
    # We charged partial exit fee in TP1 handler; charge remaining exit fee here
    # Total = realized_pnl (from TP1, already has its exit fee) + remaining_pnl - entry_fee
    pnl_dollars = trade.realized_pnl + remaining_pnl - entry_fee
    pnl_pct = (pnl_dollars / state.equity) * 100 if state.equity > 0 else 0

    is_win = pnl_dollars > 0
    state.equity += pnl_dollars

    if is_win:
        state.wins += 1
    else:
        state.losses += 1

    # Update drawdown — also clear DD protection when equity sets a new high
    if state.equity > state.peak_equity:
        state.peak_equity = state.equity
        state.dd_protection_triggered_at = None  # equity recovered; reset protection
        state.dd_trough_equity = None            # clear trough — back above previous peak
    current_dd = ((state.peak_equity - state.equity) / state.peak_equity) * 100
    state.drawdown = current_dd
    if current_dd > state.max_drawdown_pct:
        state.max_drawdown_pct = current_dd

    # Record trade
    trade_record = {
        "trade_num": trade.trade_num,
        "symbol": trade.symbol,
        "timeframe": trade.timeframe,
        "direction": trade.direction,
        "model": trade.model,
        "entry_price": trade.entry_price,
        "stop_price": trade.original_stop_price,
        "target_price": trade.target_price,
        "tp1_price": trade.tp1_price,
        "tp1_hit": trade.tp1_hit,
        "position_size": trade.original_position_size,
        "risk_amount": trade.risk_amount,
        "leverage": trade.leverage,
        "rr": trade.rr,
        "entry_score": trade.entry_score,
        "entry_reasons": trade.entry_reasons,
        "mfe": trade.mfe,
        "mae": trade.mae,
        "opened_at": trade.opened_at,
        "closed_at": current_time,
        "exit_price": raw_exit_price,
        "exit_reason": exit_reason,
        "pnl_pct": round(pnl_pct, 4),
        "pnl_dollars": round(pnl_dollars, 2),
        "is_win": is_win,
        "balance_after": round(state.equity, 2),
    }

    insert_trade(conn, run_id, trade_record)
    state.trade_history.append(trade_record)
    state.open_trade = None
    state.last_trade_close_step = state.current_step

    logger.debug(
        f"CLOSE #{trade.trade_num}: {exit_reason} @ {effective_exit:.2f} "
        f"P&L=${pnl_dollars:.2f} ({pnl_pct:+.2f}%) — balance=${state.equity:.2f}"
    )


# ── CLI Entrypoint ────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="HPB-TCT Backtest Runner")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--start", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", help="End date YYYY-MM-DD")
    parser.add_argument("--step", default="1h",
                        help="Step interval (1m, 5m, 15m, 30m, 1h)")
    parser.add_argument("--balance", type=float, default=STARTING_BALANCE)
    parser.add_argument("--replay-at", help="Single timestamp deep debug (ISO format)")
    parser.add_argument("--threshold", type=int, default=None,
                        help="Override ENTRY_THRESHOLD (default: from config)")
    parser.add_argument("--warmup", type=int, default=0,
                        help="Warmup days before evaluating signals (default: 0)")
    parser.add_argument("--execution-mode", choices=["strict", "fast"], default="strict")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    start = (datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
             if args.start else None)
    end = (datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
           if args.end else None)
    replay = (datetime.fromisoformat(args.replay_at).replace(tzinfo=timezone.utc)
              if args.replay_at else None)

    conn = get_connection()
    create_schema(conn)

    try:
        summary = run_backtest(
            symbol=args.symbol,
            start_date=start,
            end_date=end,
            step_interval=args.step,
            starting_balance=args.balance,
            entry_threshold=args.threshold,
            warmup_days=args.warmup,
            replay_at=replay,
            conn=conn,
        )
        print(f"\n{'='*40}")
        print("BACKTEST SUMMARY")
        print(f"{'='*40}")
        for k, v in summary.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.2f}")
            else:
                print(f"  {k}: {v}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
