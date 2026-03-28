"""
decision_engine_v2.py — Unified Decision Engine (v14 logic)
============================================================

Single source of truth for all entry decisions in both backtest and live trading.

Extracted from backtest/runner.py::run_gate_pipeline() — v14 gate chain.
Logic is identical to the backtest engine; do NOT modify gate thresholds,
order, or structure without a corresponding change to runner.py.

Gate pipeline (in order):
  ELIF chain:
    RIG          → FAIL_RIG_COUNTER_BIAS      (range integrity, counter-bias block)
    1A           → FAIL_1A_BIAS               (HTF market structure bias)
    RCM          → FAIL_RCM_DURATION          (range context / quality gate)
    RR global    → FAIL_RR_FILTER             (minimum R:R for any signal)
    DD soft      → FAIL_DD_PROTECTION         (drawdown protection, optional)
    displacement → FAIL_LOW_DISPLACEMENT      (v14: global displacement floor)
    15m RR       → FAIL_RR_15M_STRICT         (v12: tighter 15m RR gate)
    15m Asia     → FAIL_15M_ASIA_FILTER       (v12: no 15m entries in Asia)
    15m range sz → FAIL_RANGE_TOO_SMALL_15M   (v12: 15m range must be ≥0.3%)
    15m NY OT    → FAIL_15M_NY_OVERTRADE      (optional NY overtrade guard)
    15m disp     → FAIL_15M_LOW_DISPLACEMENT  (v14: stricter 15m displacement)

  Standalone IF guards (run even when prior elif skipped):
    15m location → FAIL_15M_POOR_ENTRY_LOCATION (v14: entry must be in valid zone)
    Model_2 15m  → FAIL_MODEL2_15M_BLOCK         (v14: Model 2 blocked on 15m)
    score floor  → FAIL_SCORE_HARD_FLOOR          (v14: block scores 57-64)
    Model 3      → FAIL_MODEL3_TF_FILTER / _NO_TREND / _EXTENDED (v12/v13)
    threshold    → FAIL_1D_SCORE                  (score < entry_threshold)
    BOS dedup    → FAIL_DUPLICATE_BOS             (same schematic ≤48h ago)

Stateful gates that require caller-managed context:
  DD protection: pass peak_equity, equity, dd_protection_triggered_at via context.
                 If omitted, DD gate is skipped (always passes).
  BOS dedup:     pass traded_bos_fingerprints (dict of fp→datetime) via context.
                 If omitted, dedup is skipped.

Usage:
    from decision_engine_v2 import decide, USE_UNIFIED_ENGINE

    result = decide(candles_by_tf, context)
    # result["decision"] in ("TAKE", "PASS")

DO NOT modify gate values here without updating runner.py to match.
"""

import concurrent.futures
import logging
from datetime import datetime, timezone
from typing import Optional

from backtest.config import (
    ENTRY_THRESHOLD,
    HTF_TIMEFRAME,
    MIN_PIVOT_CONFIRM,
    MIN_RR,
    MTF_TIMEFRAMES,
    timeframe_to_seconds,
)

logger = logging.getLogger("decision_engine_v2")

# ── Feature flag ──────────────────────────────────────────────────────
# Set True to activate the unified engine in live trading.
# The backtest runner has its own independent copy of this flag in runner.py.
# Default: False — existing code paths remain active until explicitly switched.
USE_UNIFIED_ENGINE = False

# ── Gate constants (v12/v13/v14) ─────────────────────────────────────
# Mirror of backtest/runner.py — keep in sync. Do NOT change values here
# without a matching change in runner.py and a full parity validation.
_MIN_RR_15M = 0.8             # tighter RR for 15m entries
_MIN_RANGE_PCT_15M = 0.003    # range must be >= 0.3% of price on 15m
_TREND_LOOKBACK = 50          # candles to measure slope and volatility over
_TREND_SLOPE_FLOOR = 0.003    # v14: raised from 0.0015
_TREND_VOL_MULTIPLIER = 0.5   # slope threshold = max(floor, vol * multiplier)
_MODEL3_MAX_DISTANCE_PCT = 0.015  # entry must be within 1.5% of range midpoint
_MIN_DISPLACEMENT = 0.50      # minimum local_displacement for any signal
_MIN_DISPLACEMENT_15M = 0.65  # stricter displacement floor for 15m specifically
_MIN_SCORE_HARD = 65          # hard score floor — scores 57-64 blocked regardless of threshold
_MAX_DD_SOFT = 0.04           # halt new signals when current DD exceeds 4%
_DD_RESET_HOURS = 72          # after this many hours in DD, reset peak to allow trading again
_ENABLE_15M_NY_OVERTRADE_FILTER = False  # optional NY overtrade guard

# Detection window limits (performance)
_DETECTION_WINDOW = 200
_STEP_TIMEOUT_SECONDS = 10

# Bounded thread pool for schematic detection.
# Prevents unbounded daemon-thread accumulation on repeated timeouts in long-running processes.
_DETECT_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=4, thread_name_prefix="tct_detect"
)


# ── Helper functions (verbatim from runner.py) ────────────────────────

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
    """Return (trend_ok, slope, min_slope). Volatility-adjusted threshold."""
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


# ── Module import cache ───────────────────────────────────────────────

def _get_modules() -> dict:
    """Import live modules once and cache. Mirrors runner._get_live_modules()."""
    if not hasattr(_get_modules, "_cache"):
        from tct_schematics import detect_tct_schematics
        from pivot_cache import PivotCache
        from decision_tree_bridge import DecisionTreeEvaluator
        from hpb_rig_validator import range_integrity_validator
        from session_manipulation import apply_session_multiplier
        from backtest.session import get_session
        _get_modules._cache = {
            "detect": detect_tct_schematics,
            "PivotCache": PivotCache,
            "evaluator": DecisionTreeEvaluator(),
            "rig": range_integrity_validator,
            "msce": apply_session_multiplier,
            "get_session": get_session,
        }
    return _get_modules._cache


# ── Unified decision function ─────────────────────────────────────────

def decide(
    candles_by_tf: dict,
    context: dict,
) -> dict:
    """
    Unified decision engine — v14 gate logic.

    Scans all available MTF timeframes for qualifying TCT schematics and
    returns the highest-scoring signal that passes all gates, or a PASS result.

    Args:
        candles_by_tf:
            dict mapping timeframe string → pd.DataFrame.
            Keys from MTF_TIMEFRAMES (["4h", "1h", "30m", "15m"]) are scanned.
            HTF_TIMEFRAME ("1d") or "4h" are used for bias detection.
            Missing timeframes are silently skipped.

        context:
            Required:
                current_price (float)
                current_time (datetime, UTC)
            Optional:
                entry_threshold (int)      — default: ENTRY_THRESHOLD (60)
                min_rr (float)             — default: MIN_RR (0.5)
                traded_bos_fingerprints (dict[tuple, datetime])
                                           — for BOS dedup; default: {} (disabled)
                peak_equity (float)        — for DD protection; None = skip gate
                equity (float)             — for DD protection; None = skip gate
                dd_protection_triggered_at (datetime | None)

    Returns:
        {
            "decision": "TAKE" | "PASS",
            "reason": str,               # gate that fired or "all_gates_passed"
            "failure_code": str | None,  # FAIL_* constant or None
            "score": float,
            "model": str,
            "timeframe": str,
            "direction": str,
            "entry_price": float,
            "stop_price": float,
            "target_price": float,
            "rr": float,
            "metadata": {
                "htf_bias": str,
                "gate_1a_pass": bool,
                "rcm_valid": bool,
                "rcm_score": float,
                "range_duration_hours": float,
                "local_displacement": float,
                "rig_status": str,
                "rig_reason": str | None,
                "session": str,
                "execution_confidence": float,
            },
        }

    The function is stateless. Callers that need gate-level state (DD protection,
    BOS dedup, range persistence) must manage that state externally and pass it
    in via context on each call.
    """
    _PASS = {
        "decision": "PASS",
        "reason": "no_signal",
        "failure_code": None,
        "score": 0,
        "model": "",
        "timeframe": "",
        "direction": "",
        "entry_price": 0.0,
        "stop_price": 0.0,
        "target_price": 0.0,
        "rr": 0.0,
        "metadata": {},
    }

    current_price: float = context.get("current_price", 0.0)
    current_time: Optional[datetime] = context.get("current_time")
    entry_threshold: int = context.get("entry_threshold", ENTRY_THRESHOLD)
    min_rr: float = context.get("min_rr", MIN_RR)
    traded_bos_fingerprints: dict = context.get("traded_bos_fingerprints") or {}

    # DD protection (stateful — omit keys to disable this gate)
    peak_equity: Optional[float] = context.get("peak_equity")
    equity: Optional[float] = context.get("equity")
    dd_protection_triggered_at: Optional[datetime] = context.get("dd_protection_triggered_at")

    if not current_price or not current_time:
        return {**_PASS, "reason": "missing_required_context (current_price or current_time)"}

    try:
        mods = _get_modules()
    except ImportError as e:
        logger.exception("decide(): module import failed: %s", e)
        return {**_PASS, "reason": f"import_error: {e}"}

    detect_tct_schematics = mods["detect"]
    PivotCache = mods["PivotCache"]
    evaluator = mods["evaluator"]
    range_integrity_validator = mods["rig"]
    apply_session_multiplier = mods["msce"]
    get_session = mods["get_session"]

    # ── MSCE Session ──────────────────────────────────────────────
    session_info = get_session(current_time)
    session_name: str = session_info["session"]

    # ── HTF Bias (Gate 1A) ────────────────────────────────────────
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
            logger.debug("HTF bias error on %s: %s", htf_candidate, e)
            continue

    gate_1a_pass: bool = htf_bias != "neutral"

    # ── Scan MTF timeframes ───────────────────────────────────────
    best_signal: Optional[dict] = None
    best_score: float = 0

    for tf in MTF_TIMEFRAMES:
        df_tf_full = candles_by_tf.get(tf)
        if df_tf_full is None or len(df_tf_full) < MIN_PIVOT_CONFIRM:
            continue

        df_tf = df_tf_full.tail(_DETECTION_WINDOW).reset_index(drop=True)

        # ── Schematic detection (with timeout) ───────────────────
        try:
            pc = PivotCache(df_tf, lookback=3)

            def _run_detect(df=df_tf, p=pc):
                return detect_tct_schematics(df, [], pivot_cache=p)

            _fut = _DETECT_EXECUTOR.submit(_run_detect)
            try:
                det = _fut.result(timeout=_STEP_TIMEOUT_SECONDS)
            except concurrent.futures.TimeoutError:
                logger.debug("decide(): detection timeout on %s", tf)
                continue
            if det is None:
                continue

            all_schematics = (
                det.get("accumulation_schematics", [])
                + det.get("distribution_schematics", [])
            )
        except Exception as e:
            logger.debug("decide(): detection error on %s: %s", tf, e)
            continue

        # ── Per-key dedup: keep most-recent BOS per model/direction ──
        best_by_key: dict[str, dict] = {}
        for s in all_schematics:
            if not isinstance(s, dict) or not s.get("is_confirmed"):
                continue
            key = f"{s.get('direction', '')}_{s.get('model', s.get('schematic_type', ''))}"
            bos_info = s.get("bos_confirmation") or {}
            bos_idx = bos_info.get("bos_idx", -1) or -1
            existing_bos_idx = (best_by_key.get(key, {}).get("bos_confirmation") or {}).get("bos_idx", -1)
            if key not in best_by_key or bos_idx > existing_bos_idx:
                best_by_key[key] = s
        deduped_schematics = list(best_by_key.values())

        for schematic in deduped_schematics:

            # ── Score via decision tree ───────────────────────────
            try:
                eval_result = evaluator.evaluate_schematic(
                    schematic, htf_bias, current_price,
                    total_candles=len(df_tf),
                    candle_df=df_tf,
                )
            except Exception as e:
                logger.debug("decide(): evaluation error: %s", e)
                continue

            score: float = eval_result.get("score", 0)
            direction: str = eval_result.get("direction", "unknown")
            model: str = eval_result.get("model", "unknown")
            rr: float = eval_result.get("rr", 0)

            # ── RCM (Range Context) ───────────────────────────────
            rcm_score: float = schematic.get("quality_score", 0.0)
            range_info = schematic.get("range", {})
            rcm_valid: bool = rcm_score >= 0.6

            # Compute range_duration_hours (no state persistence — pure from schematic data)
            range_duration_hours: float = 0.0
            if isinstance(range_info, dict):
                range_duration_hours = range_info.get("duration_hours", 0)
                if range_duration_hours == 0:
                    r_start_idx = range_info.get("start_idx")
                    r_end_idx = range_info.get("end_idx")
                    if r_start_idx is not None and r_end_idx is not None and len(df_tf) > 0:
                        try:
                            start_t = df_tf.iloc[max(0, int(r_start_idx))]["open_time"]
                            end_t = df_tf.iloc[min(len(df_tf) - 1, int(r_end_idx))]["open_time"]
                            range_duration_hours = (end_t - start_t).total_seconds() / 3600
                        except (IndexError, KeyError):
                            pass
                    if range_duration_hours == 0:
                        num_candles = range_info.get("num_candles", range_info.get("candle_count", 0))
                        if num_candles > 0:
                            try:
                                tf_secs = timeframe_to_seconds(tf)
                                range_duration_hours = (num_candles * tf_secs) / 3600
                            except ValueError:
                                pass

            # Compute local_displacement from range levels vs current price
            local_displacement: float = 0.0
            if isinstance(range_info, dict):
                local_displacement = range_info.get("displacement", 0.0)
                if local_displacement == 0.0:
                    r_high = range_info.get("high", 0)
                    r_low = range_info.get("low", 0)
                    if r_high and r_low and r_high > r_low:
                        range_size = r_high - r_low
                        if direction == "bullish":
                            local_displacement = (current_price - r_low) / range_size
                        else:
                            local_displacement = (r_high - current_price) / range_size
                        local_displacement = max(0.0, min(1.0, local_displacement))

            # ── RIG (Range Integrity Gate) ────────────────────────
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
            rig_status: str = rig_result.get("status", "VALID")
            rig_reason: Optional[str] = rig_result.get("reason")

            # ── MSCE session multiplier ───────────────────────────
            msce_result = apply_session_multiplier(float(score), current_time)
            execution_confidence: float = msce_result.get("adjusted_confidence", float(score))

            # ── Price levels + R:R ────────────────────────────────
            stop_info = schematic.get("stop_loss", {})
            target_info = schematic.get("target", {})
            entry_price: float = schematic.get("entry", {}).get("price", current_price)
            stop_price: float = stop_info.get("price", 0) if isinstance(stop_info, dict) else 0
            target_price: float = target_info.get("price", 0) if isinstance(target_info, dict) else 0

            actual_rr: float = 0.0
            if entry_price and stop_price and target_price:
                sl_dist = abs(entry_price - stop_price)
                tp_dist = abs(target_price - entry_price)
                if sl_dist > 0:
                    actual_rr = tp_dist / sl_dist

            # ══════════════════════════════════════════════════════
            # GATE CHAIN — must match runner.py gate order exactly.
            # ELIF chain: first matching condition fires, rest skip.
            # Standalone IF guards run regardless of prior chain.
            # ══════════════════════════════════════════════════════
            failure_code: Optional[str] = None
            skip_reason: Optional[str] = None
            final_decision = "TAKE"

            # ── RIG ───────────────────────────────────────────────
            if rig_status == "BLOCK":
                final_decision = "PASS"
                skip_reason = "RIG_BLOCK"
                failure_code = "FAIL_RIG_COUNTER_BIAS"
                execution_confidence = 0.0

            # ── 1A: HTF bias ──────────────────────────────────────
            elif not gate_1a_pass:
                final_decision = "PASS"
                skip_reason = "NO_HTF_BIAS"
                failure_code = "FAIL_1A_BIAS"

            # ── RCM: range quality ────────────────────────────────
            elif not rcm_valid:
                final_decision = "PASS"
                skip_reason = "RCM_INVALID"
                failure_code = "FAIL_RCM_DURATION"

            # ── Global RR filter ──────────────────────────────────
            elif actual_rr < min_rr:
                final_decision = "PASS"
                skip_reason = f"RR_TOO_LOW ({actual_rr:.2f} < {min_rr:.2f})"
                failure_code = "FAIL_RR_FILTER"

            # ── v14: soft DD protection (optional — needs equity state) ──
            elif (
                peak_equity is not None
                and equity is not None
                and peak_equity > 0
                and (peak_equity - equity) / peak_equity > _MAX_DD_SOFT
            ):
                _cur_dd = (peak_equity - equity) / peak_equity * 100
                # Only block if within the reset window (72h)
                _in_window = True
                if dd_protection_triggered_at is not None:
                    _hours = (current_time - dd_protection_triggered_at).total_seconds() / 3600
                    if _hours >= _DD_RESET_HOURS:
                        _in_window = False  # reset elapsed — fall through to next gate
                if _in_window:
                    final_decision = "PASS"
                    skip_reason = f"DD_PROTECTION (dd={_cur_dd:.2f}% > {_MAX_DD_SOFT * 100:.0f}%)"
                    failure_code = "FAIL_DD_PROTECTION"

            # ── v14: global displacement floor ────────────────────
            elif local_displacement < _MIN_DISPLACEMENT:
                final_decision = "PASS"
                skip_reason = f"LOW_DISPLACEMENT ({local_displacement:.3f} < {_MIN_DISPLACEMENT:.2f})"
                failure_code = "FAIL_LOW_DISPLACEMENT"

            # ── v12: 15m RR strict ────────────────────────────────
            elif tf == "15m" and actual_rr < _MIN_RR_15M:
                final_decision = "PASS"
                skip_reason = f"RR_15M_STRICT ({actual_rr:.2f} < {_MIN_RR_15M:.2f})"
                failure_code = "FAIL_RR_15M_STRICT"

            # ── v12: 15m Asia session block ───────────────────────
            elif tf == "15m" and session_name == "asia":
                final_decision = "PASS"
                skip_reason = "15M_ASIA_FILTER"
                failure_code = "FAIL_15M_ASIA_FILTER"

            # ── v12: 15m range size gate ──────────────────────────
            elif tf == "15m" and _range_size_pct(range_info, current_price) < _MIN_RANGE_PCT_15M:
                _rpct = _range_size_pct(range_info, current_price)
                final_decision = "PASS"
                skip_reason = f"RANGE_TOO_SMALL_15M ({_rpct:.4f} < {_MIN_RANGE_PCT_15M:.4f})"
                failure_code = "FAIL_RANGE_TOO_SMALL_15M"

            # ── Optional: 15m NY overtrade guard ─────────────────
            elif tf == "15m" and session_name == "new_york" and _ENABLE_15M_NY_OVERTRADE_FILTER:
                final_decision = "PASS"
                skip_reason = "15M_NY_OVERTRADE_GUARD"
                failure_code = "FAIL_15M_NY_OVERTRADE"

            # ── v14: 15m stricter displacement (3A) ──────────────
            elif tf == "15m" and local_displacement < _MIN_DISPLACEMENT_15M:
                final_decision = "PASS"
                skip_reason = f"15M_LOW_DISPLACEMENT ({local_displacement:.3f} < {_MIN_DISPLACEMENT_15M:.2f})"
                failure_code = "FAIL_15M_LOW_DISPLACEMENT"

            # ═══════════════════════════════════════════════════════
            # STANDALONE IF GUARDS — each runs independently, even if
            # a prior guard already set final_decision = "PASS".
            # This matches the runner.py gate-chain exactly (v15 fix).
            # ═══════════════════════════════════════════════════════

            # ── v14: 15m entry location gate (3B) ─────────────────
            if (
                final_decision == "TAKE"
                and tf == "15m"
                and isinstance(range_info, dict)
                and range_info.get("high")
                and range_info.get("low")
            ):
                _r_high_loc = float(range_info["high"])
                _r_low_loc = float(range_info["low"])
                if _r_high_loc > _r_low_loc and entry_price > 0:
                    _pos = (entry_price - _r_low_loc) / (_r_high_loc - _r_low_loc)
                    _loc_fail = (
                        (direction == "bullish" and _pos > 0.4)
                        or (direction == "bearish" and _pos < 0.6)
                    )
                    if _loc_fail:
                        final_decision = "PASS"
                        skip_reason = f"15M_POOR_ENTRY_LOCATION (pos={_pos:.3f}, dir={direction})"
                        failure_code = "FAIL_15M_POOR_ENTRY_LOCATION"

            # ── v14: block Model_2 on 15m (3C) ───────────────────
            if final_decision == "TAKE" and tf == "15m" and model == "Model_2":
                final_decision = "PASS"
                skip_reason = "MODEL2_15M_BLOCK"
                failure_code = "FAIL_MODEL2_15M_BLOCK"

            # ── v14: hard score floor (ALL models) ───────────────
            if final_decision == "TAKE" and score < _MIN_SCORE_HARD:
                final_decision = "PASS"
                skip_reason = f"SCORE_HARD_FLOOR ({score} < {_MIN_SCORE_HARD})"
                failure_code = "FAIL_SCORE_HARD_FLOOR"

            # ── v12/v13: Model 3 quality gates ───────────────────
            if final_decision == "TAKE" and "Model_3" in model:
                if tf != "1h":
                    final_decision = "PASS"
                    skip_reason = f"MODEL3_TF_FILTER (tf={tf}, only 1h allowed)"
                    failure_code = "FAIL_MODEL3_TF_FILTER"
                else:
                    _closes = df_tf["close"].values if df_tf is not None and len(df_tf) > 0 else []
                    _trend_ok, _slope, _min_slope = _is_trending_environment(_closes)
                    if not _trend_ok:
                        final_decision = "PASS"
                        skip_reason = f"MODEL3_NO_TREND (slope={abs(_slope):.4f} < adaptive {_min_slope:.4f})"
                        failure_code = "FAIL_MODEL3_NO_TREND"
                    else:
                        _r_high = range_info.get("high", 0) if isinstance(range_info, dict) else 0
                        _r_low = range_info.get("low", 0) if isinstance(range_info, dict) else 0
                        if _r_high > _r_low and entry_price > 0:
                            _range_mid = (_r_high + _r_low) / 2
                            _dist_pct = abs(entry_price - _range_mid) / _range_mid
                            if _dist_pct > _MODEL3_MAX_DISTANCE_PCT:
                                final_decision = "PASS"
                                skip_reason = f"MODEL3_EXTENDED (dist={_dist_pct:.4f} > {_MODEL3_MAX_DISTANCE_PCT:.4f})"
                                failure_code = "FAIL_MODEL3_EXTENDED"

            # ── Score threshold ───────────────────────────────────
            if final_decision == "TAKE" and score < entry_threshold:
                final_decision = "PASS"
                skip_reason = f"SCORE_BELOW_THRESHOLD ({score} < {entry_threshold})"
                failure_code = "FAIL_1D_SCORE"

            # ── BOS fingerprint dedup ─────────────────────────────
            if final_decision == "TAKE":
                bos_info = schematic.get("bos_confirmation") or {}
                bos_price = round(float(bos_info.get("bos_price") or 0), 0)
                entry_snap = round(float(
                    schematic.get("entry", {}).get("price") or current_price
                ), 0)
                fp = (tf, model, direction, entry_snap, bos_price)
                fp_traded_at = traded_bos_fingerprints.get(fp)
                if isinstance(fp_traded_at, datetime):
                    age_hours = (current_time - fp_traded_at).total_seconds() / 3600
                    if age_hours < 48:
                        final_decision = "PASS"
                        skip_reason = f"DUPLICATE_BOS (same schematic {age_hours:.0f}h ago)"
                        failure_code = "FAIL_DUPLICATE_BOS"
                elif fp_traded_at is not None:
                    # Caller stored a non-datetime value — log and skip dedup rather than crash.
                    logger.warning(
                        "decide(): traded_bos_fingerprints[%s] has unexpected type %s — ignoring entry",
                        fp, type(fp_traded_at).__name__,
                    )

            logger.debug(
                "decide(): tf=%s model=%s dir=%s score=%s rr=%.2f "
                "disp=%.3f rcm=%s rig=%s → %s (%s)",
                tf, model, direction, score, actual_rr,
                local_displacement, rcm_valid, rig_status,
                final_decision, failure_code or "ok",
            )

            # ── Keep best TAKE ────────────────────────────────────
            if final_decision == "TAKE" and score > best_score:
                best_score = score
                best_signal = {
                    "decision": "TAKE",
                    "reason": "all_gates_passed",
                    "failure_code": None,
                    "score": score,
                    "model": model,
                    "timeframe": tf,
                    "direction": direction,
                    "entry_price": entry_price,
                    "stop_price": stop_price,
                    "target_price": target_price,
                    "rr": actual_rr if actual_rr > 0 else rr,
                    "metadata": {
                        "htf_bias": htf_bias,
                        "gate_1a_pass": gate_1a_pass,
                        "rcm_valid": rcm_valid,
                        "rcm_score": rcm_score,
                        "range_duration_hours": range_duration_hours,
                        "local_displacement": local_displacement,
                        "rig_status": rig_status,
                        "rig_reason": rig_reason,
                        "session": session_name,
                        "execution_confidence": execution_confidence,
                        # Raw objects for callers that need them (e.g. trade execution)
                        "_eval_result": eval_result,
                        "_schematic": schematic,
                        "_tf_df": df_tf,
                    },
                }

    return best_signal if best_signal is not None else _PASS
