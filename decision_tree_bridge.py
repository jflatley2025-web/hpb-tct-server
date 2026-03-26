"""
decision_tree_bridge.py — Bridge between raw candle/schematic data and the Decision Trees
============================================================================================

Provides two scoring pipelines:

**v2 (active)** — 9-phase sequential pipeline following correct TCT logic:

    Phase 1: HTF Context         → Market-structure-based bias (not schematic detection)
    Phase 2: Range Detection     → Time displacement, liquidity stacking, V-shape rejection
    Phase 3: Tap Structure       → Model 1 / Model 2 validation (before BOS)
    Phase 4: Liquidity           → Sweep or interaction (slight close beyond OK)
    Phase 5: Break of Structure  → Internal MS break after Tap3
    Phase 6: POI Validation      → FVG / OB / MM block (FVG optional, not hard gate)
    Phase 7: Directional Filter  → HTF alignment (with reversal exception)
    Phase 8: Risk Filter         → Minimum R:R 1.5
    Phase 9: Confidence Scoring  → 6-component score, threshold 60

**v1 (legacy, kept for reference)** — Original 6-tree pipeline.

Key differences vs v1:
  - HTF bias uses market structure, NOT schematic detection
  - Tap detection occurs BEFORE BOS validation
  - Range validation uses time displacement + liquidity stacking
  - Liquidity sweeps allow slight close beyond range boundary
  - FVG is optional (confidence contributor, not hard gate)
  - Threshold raised from 50 → 60
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from decision_trees.ranges_decision_tree import (
    RangeInputs, RangeEvaluation, Trend, TradeBias as RangeTradeBias,
    DeviationType, evaluate_range_setup,
)
from decision_trees.supply_demand_decision_tree import (
    SDZoneInputs, SDZoneEvaluation, MarketContext, ZoneType, ZoneDirection,
    MitigationStatus, TradeBias as SDTradeBias, evaluate_sd_zone,
)
from decision_trees.liquidity_decision_tree import (
    LiquidityInputs, LiquidityEvaluation, LiquidityPoolType, SweepSide,
    PathQuality, TradeBias as LiqTradeBias, evaluate_liquidity_setup,
)
from decision_trees.tct_5a_schematics_decision_tree import (
    TCTSchematicInputs, TCTSchematicEvaluation, SchematicDirection as Dir5A,
    ModelType as MT5A, BOSLocation, BOSTimeframe,
    SchematicStatus, TradeBias as TB5A, evaluate_tct_schematic,
)
from decision_trees.tct_5b_schematics_real_examples_decision_tree import (
    TCT5BInputs, TCT5BEvaluation, EntryTimeframe,
    SchematicDirection as Dir5B, ModelType as MT5B,
    SchematicStatus as Status5B, TradeBias as TB5B, evaluate_5b_schematic,
)
from decision_trees.tct_6_advanced_schematics_decision_tree import (
    FlipInputs, FlipEvaluation, EscalationInputs, EscalationEvaluation,
    SchematicDirection as Dir6, OutcomeStatus,
    evaluate_schematic_flip, evaluate_ltf_htf_escalation,
)

logger = logging.getLogger("DecisionTreeBridge")

# ================================================================
# CANDLE ANALYSIS HELPERS
# ================================================================

def _detect_trend(df: pd.DataFrame, lookback: int = 20) -> Tuple[bool, bool]:
    """Detect HH/HL (uptrend) and LH/LL (downtrend) from recent swing points."""
    if df is None or len(df) < lookback:
        return False, False

    highs = df["high"].values[-lookback:]
    lows = df["low"].values[-lookback:]

    # Simple swing detection: compare rolling windows
    window = max(3, lookback // 5)
    swing_highs = []
    swing_lows = []
    for i in range(window, len(highs) - window):
        if highs[i] == max(highs[i - window:i + window + 1]):
            swing_highs.append(highs[i])
        if lows[i] == min(lows[i - window:i + window + 1]):
            swing_lows.append(lows[i])

    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return False, False

    hh = swing_highs[-1] > swing_highs[-2]
    hl = swing_lows[-1] > swing_lows[-2]
    lh = swing_highs[-1] < swing_highs[-2]
    ll = swing_lows[-1] < swing_lows[-2]

    return (hh and hl), (lh and ll)


def _six_candle_rule(df: pd.DataFrame, range_high: float, range_low: float) -> bool:
    """Check if 2+2+2 candle structure is satisfied within the range.

    The six-candle rule requires: at least 2 candles near range high, 2 near range
    low, and 2 in between (showing price has visited all zones of the range).
    """
    if df is None or len(df) < 6:
        return False

    closes = df["close"].values[-min(60, len(df)):]
    rng = range_high - range_low
    if rng <= 0:
        return False

    upper_third = range_high - rng * 0.33
    lower_third = range_low + rng * 0.33

    near_high = sum(1 for c in closes if c >= upper_third)
    near_low = sum(1 for c in closes if c <= lower_third)
    mid = sum(1 for c in closes if lower_third < c < upper_third)

    return near_high >= 2 and near_low >= 2 and mid >= 2


def _check_equilibrium_touched(df: pd.DataFrame, range_high: float, range_low: float) -> bool:
    """Check if price has returned to the 0.5 Fib (equilibrium)."""
    if df is None or len(df) < 3:
        return False
    eq = (range_high + range_low) / 2.0
    tolerance = (range_high - range_low) * 0.05
    closes = df["close"].values[-min(60, len(df)):]
    return any(abs(c - eq) <= tolerance for c in closes)


def _range_looks_horizontal(df: pd.DataFrame, range_high: float, range_low: float) -> bool:
    """Check if the range is genuinely horizontal (not V-shaped / impulsive)."""
    if df is None or len(df) < 6:
        return False

    closes = df["close"].values[-min(40, len(df)):]
    rng = range_high - range_low
    if rng <= 0:
        return False

    # A range is horizontal if most closes stay within the range boundaries
    inside_count = sum(1 for c in closes if range_low <= c <= range_high)
    return (inside_count / len(closes)) >= 0.6


# ================================================================
# PHASE 1 — HTF MARKET STRUCTURE BIAS (v2 pipeline)
# Uses pivot-based structure, NOT schematic detection.
# ================================================================

def detect_htf_market_structure(df: pd.DataFrame, lookback: int = 100) -> Dict:
    """Determine HTF bias from pure market structure (pivots → structure breaks).

    Steps:
      1. Identify valid pivots using the 6-candle rule (3 candles each side).
      2. Build swing sequence: alternating highs and lows.
      3. Classify: HH+HL = bullish, LH+LL = bearish.
      4. Confirm via most recent structure break (close beyond prior swing).
      5. Return bias + diagnostic detail.

    Returns:
        {
            "bias": "bullish" | "bearish" | "neutral",
            "swing_highs": [(idx, price), ...],
            "swing_lows": [(idx, price), ...],
            "structure_break": {"type": "bullish"|"bearish", "level": float, "idx": int} | None,
            "reason": str,
        }
    """
    neutral = {
        "bias": "neutral", "swing_highs": [], "swing_lows": [],
        "structure_break": None, "reason": "insufficient data",
    }
    if df is None or len(df) < 20:
        return neutral

    highs = df["high"].values[-min(lookback, len(df)):]
    lows = df["low"].values[-min(lookback, len(df)):]
    closes = df["close"].values[-min(lookback, len(df)):]
    n = len(highs)
    offset = len(df) - n  # to convert local idx → df idx

    # Step 1: Pivot detection using 6-candle rule (3 candles each side)
    pivot_window = 3
    swing_highs: List[Tuple[int, float]] = []
    swing_lows: List[Tuple[int, float]] = []

    for i in range(pivot_window, n - pivot_window):
        left_h = highs[i - pivot_window:i]
        right_h = highs[i + 1:i + pivot_window + 1]
        if highs[i] >= max(left_h) and highs[i] >= max(right_h):
            swing_highs.append((i + offset, float(highs[i])))

        left_l = lows[i - pivot_window:i]
        right_l = lows[i + 1:i + pivot_window + 1]
        if lows[i] <= min(left_l) and lows[i] <= min(right_l):
            swing_lows.append((i + offset, float(lows[i])))

    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return {**neutral, "swing_highs": swing_highs, "swing_lows": swing_lows,
                "reason": "not enough swing points"}

    # Step 2: Classify the last two swing highs and lows
    sh1_price = swing_highs[-2][1]
    sh2_price = swing_highs[-1][1]
    sl1_price = swing_lows[-2][1]
    sl2_price = swing_lows[-1][1]

    hh = sh2_price > sh1_price
    hl = sl2_price > sl1_price
    lh = sh2_price < sh1_price
    ll = sl2_price < sl1_price

    # Step 3: Determine structure direction
    if hh and hl:
        structure_dir = "bullish"
    elif lh and ll:
        structure_dir = "bearish"
    else:
        return {
            "bias": "neutral", "swing_highs": swing_highs, "swing_lows": swing_lows,
            "structure_break": None,
            "reason": f"mixed structure (HH={hh}, HL={hl}, LH={lh}, LL={ll})",
        }

    # Step 4: Confirm via structure break — a candle must CLOSE beyond
    # the prior swing level to confirm the break.
    structure_break = None
    if structure_dir == "bullish":
        # Need a close above the prior swing high (sh1) to confirm bullish break
        break_level = sh1_price
        for i in range(swing_highs[-2][0] - offset, n):
            if closes[i] > break_level:
                structure_break = {
                    "type": "bullish", "level": break_level, "idx": i + offset,
                }
                break
    else:
        # Need a close below the prior swing low (sl1) to confirm bearish break
        break_level = sl1_price
        for i in range(swing_lows[-2][0] - offset, n):
            if closes[i] < break_level:
                structure_break = {
                    "type": "bearish", "level": break_level, "idx": i + offset,
                }
                break

    if structure_break is None:
        return {
            "bias": "neutral", "swing_highs": swing_highs, "swing_lows": swing_lows,
            "structure_break": None,
            "reason": f"{structure_dir} structure detected but no confirmed break",
        }

    return {
        "bias": structure_dir,
        "swing_highs": swing_highs,
        "swing_lows": swing_lows,
        "structure_break": structure_break,
        "reason": f"confirmed {structure_dir} structure break at {break_level:.2f}",
    }


# ================================================================
# PHASE 2 — RANGE VALIDATION HELPERS (v2 pipeline)
# ================================================================

def _check_time_displacement(schematic: Dict, min_candles: int = 8) -> Tuple[bool, int]:
    """Check that time displacement between Tap1 and Tap2 is significant.

    Returns (passes, candle_gap).
    """
    tap1 = schematic.get("tap1") or {}
    tap2 = schematic.get("tap2") or {}
    t1_idx = tap1.get("idx", 0)
    t2_idx = tap2.get("idx", 0)
    gap = abs(t2_idx - t1_idx)
    return gap >= min_candles, gap


def _detect_liquidity_stacking(df: pd.DataFrame, range_high: float,
                                range_low: float, tolerance_pct: float = 0.002) -> Dict:
    """Detect equal highs or equal lows near the range boundary (liquidity stacking).

    Returns {"equal_highs": int, "equal_lows": int, "has_stacking": bool}.
    """
    if df is None or len(df) < 6:
        return {"equal_highs": 0, "equal_lows": 0, "has_stacking": False}

    highs = df["high"].values[-min(60, len(df)):]
    lows = df["low"].values[-min(60, len(df)):]

    tol_h = range_high * tolerance_pct
    tol_l = range_low * tolerance_pct

    # Count highs clustering near range_high
    equal_highs = sum(1 for h in highs if abs(h - range_high) <= tol_h)
    # Count lows clustering near range_low
    equal_lows = sum(1 for l in lows if abs(l - range_low) <= tol_l)

    # Liquidity stacking requires at least 2 touches on either boundary
    return {
        "equal_highs": int(equal_highs),
        "equal_lows": int(equal_lows),
        "has_stacking": equal_highs >= 2 or equal_lows >= 2,
    }


def _reject_v_shape(df: pd.DataFrame, range_high: float, range_low: float) -> bool:
    """Return True if the range is a V-shape / impulsive move that should be rejected.

    A V-shape is detected when price moves impulsively through the range without
    oscillation — i.e., it trends in one direction rather than consolidating.
    """
    if df is None or len(df) < 6:
        return True  # Not enough data → reject

    rng = range_high - range_low
    if rng <= 0:
        return True

    closes = df["close"].values[-min(40, len(df)):]

    # Count direction changes (oscillations) — sideways ranges have many,
    # impulsive moves have few.
    direction_changes = 0
    for i in range(2, len(closes)):
        prev_dir = closes[i - 1] - closes[i - 2]
        curr_dir = closes[i] - closes[i - 1]
        if prev_dir * curr_dir < 0:  # sign change = direction reversal
            direction_changes += 1

    # Count half-crossings (price crosses EQ from one half to the other)
    eq = (range_high + range_low) / 2
    half_crossings = 0
    for i in range(1, len(closes)):
        above_prev = closes[i - 1] > eq
        above_curr = closes[i] > eq
        if above_prev != above_curr:
            half_crossings += 1

    # V-shape: few oscillations AND few (or zero) EQ crossings.
    # A real range has price crossing EQ many times; an impulse crosses it
    # at most once (on the way through).
    min_crossings = max(2, len(closes) * 0.10)
    min_dir_changes = max(3, len(closes) * 0.15)

    if direction_changes < min_dir_changes and half_crossings < min_crossings:
        return True

    return False


# ================================================================
# PHASE 4 — LIQUIDITY TOLERANCE (v2 pipeline)
# ================================================================

def _check_wick_rejection(highs, lows, closes, direction: str,
                          opens=None,
                          lookback: int = 5, ratio: float = 1.5) -> bool:
    """Check if recent candles show wick rejection of a price break.

    For bullish (sell-side sweep): look for long lower wicks (rejection of downside)
    For bearish (buy-side sweep): look for long upper wicks (rejection of upside)

    A wick rejection suggests the market is rejecting the break — the move beyond
    the boundary is being faded, not sustained.

    Args:
        highs, lows, closes: numpy arrays of recent candle data
        direction: "bullish" or "bearish"
        opens: numpy array of open prices (if None, falls back to close-based approx)
        lookback: number of recent candles to check
        ratio: minimum wick-to-body ratio to qualify as rejection

    Returns:
        True if wick rejection pattern found in recent candles.
    """
    n = min(lookback, len(closes))
    if n < 2:
        return False

    rejection_count = 0
    for i in range(-n, 0):
        h = float(highs[i])
        l = float(lows[i])
        c = float(closes[i])
        o = float(opens[i]) if opens is not None else c  # fallback if no opens
        body = abs(c - o)
        candle_range = h - l
        if candle_range <= 0:
            continue

        if direction == "bullish":
            # Sell-side sweep: look for long lower wicks (rejection of downside)
            lower_wick = min(c, o) - l
            if lower_wick > 0 and body > 0:
                if lower_wick > body * ratio:
                    rejection_count += 1
        else:
            # Buy-side sweep: look for long upper wicks (rejection of upside)
            upper_wick = h - max(c, o)
            if upper_wick > 0 and body > 0:
                if upper_wick > body * ratio:
                    rejection_count += 1

    # Require at least 1 rejection candle in last N candles
    return rejection_count >= 1


def _detect_liquidity_sweep_v2(df: pd.DataFrame, range_high: float, range_low: float,
                                direction: str) -> Dict:
    """Detect liquidity sweep with tolerance for slight closes beyond boundary.

    Unlike v1, does NOT require strict wick-only behavior. A close slightly
    beyond the range boundary (within tolerance) still qualifies as a sweep.
    """
    if df is None or len(df) < 5:
        return {"swept": False, "classification": "no_data"}

    rng = range_high - range_low
    if rng <= 0:
        return {"swept": False, "classification": "no_range"}

    # Tolerance: 0.3% of range size — close within this is still a sweep
    tolerance = rng * 0.003
    dl2 = rng * 0.30
    dl2_above = range_high + dl2
    dl2_below = range_low - dl2

    recent = df.tail(20) if len(df) >= 20 else df
    highs = recent["high"].values
    lows = recent["low"].values
    closes = recent["close"].values
    opens = recent["open"].values

    # Sustained acceptance threshold: if the deepest close beyond DL2 was
    # significant (> 1% of range), require multiple candles back inside the
    # range to confirm acceptance, not just one.
    sustained_candles = 3

    if direction == "bullish":
        # Sell-side sweep (below range low)
        exceeded = any(l < range_low for l in lows)
        # Decisive break: full body close below DL2 AND no immediate acceptance back
        close_beyond_dl2 = any(c < dl2_below for c in closes)
        # Slight close beyond range low but within tolerance
        slight_close = any(c < range_low and c >= range_low - tolerance for c in closes)
        # Accepted back inside range — require sustained acceptance for deep DL2 breaks
        deepest_close = min(closes)
        deep_break = deepest_close < dl2_below and (dl2_below - deepest_close) > rng * 0.01
        if deep_break:
            # Sustained acceptance: last N candles must all be back inside
            accepted = (len(closes) >= sustained_candles
                        and all(c >= range_low for c in closes[-sustained_candles:]))
        else:
            accepted = len(closes) >= 2 and closes[-1] >= range_low
        sweep_count = sum(1 for l in lows if l < range_low)

        if close_beyond_dl2 and not accepted:
            # Wick rejection check: if recent candles show strong rejection wicks
            # (long lower wicks relative to body), price is rejecting the break
            wick_rejection = _check_wick_rejection(highs, lows, closes, direction, opens=opens)
            if wick_rejection:
                classification = "sweep_with_rejection"
            else:
                classification = "true_break"
        elif exceeded:
            classification = "sweep"
        else:
            classification = "no_sweep"

        return {
            "swept": exceeded,
            "sweep_side": "sell_side",
            "pool_type": "range_low",
            "close_beyond_dl2": close_beyond_dl2,
            "slight_close_beyond": slight_close,
            "accepted_back": accepted,
            "sweep_count": sweep_count,
            "classification": classification,
        }
    else:
        # Buy-side sweep (above range high)
        exceeded = any(h > range_high for h in highs)
        close_beyond_dl2 = any(c > dl2_above for c in closes)
        slight_close = any(c > range_high and c <= range_high + tolerance for c in closes)
        # Sustained acceptance for deep DL2 breaks
        deepest_close = max(closes)
        deep_break = deepest_close > dl2_above and (deepest_close - dl2_above) > rng * 0.01
        if deep_break:
            accepted = (len(closes) >= sustained_candles
                        and all(c <= range_high for c in closes[-sustained_candles:]))
        else:
            accepted = len(closes) >= 2 and closes[-1] <= range_high
        sweep_count = sum(1 for h in highs if h > range_high)

        if close_beyond_dl2 and not accepted:
            # Wick rejection check: if recent candles show strong rejection wicks
            # (long upper wicks relative to body), price is rejecting the break
            wick_rejection = _check_wick_rejection(highs, lows, closes, direction, opens=opens)
            if wick_rejection:
                classification = "sweep_with_rejection"
            else:
                classification = "true_break"
        elif exceeded:
            classification = "sweep"
        else:
            classification = "no_sweep"

        return {
            "swept": exceeded,
            "sweep_side": "buy_side",
            "pool_type": "range_high",
            "close_beyond_dl2": close_beyond_dl2,
            "slight_close_beyond": slight_close,
            "accepted_back": accepted,
            "sweep_count": sweep_count,
            "classification": classification,
        }


# ================================================================
# PHASE 9 — SESSION TIMING HELPER
# ================================================================

def _score_session_timing() -> Tuple[int, str]:
    """Score based on whether we're in a high-volume trading session.

    London (07:00-16:00 UTC) and NY (13:00-21:00 UTC) overlap is best.
    Returns (points, description).
    """
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    hour = now.hour

    # London+NY overlap (13:00-16:00 UTC) — highest volume
    if 13 <= hour < 16:
        return 10, "London/NY overlap (peak)"
    # NY session (13:00-21:00 UTC)
    if 13 <= hour < 21:
        return 7, "NY session"
    # London session (07:00-16:00 UTC)
    if 7 <= hour < 16:
        return 7, "London session"
    # Asian session (00:00-07:00 UTC)
    if 0 <= hour < 7:
        return 3, "Asian session (low volume)"
    # Off-hours
    return 2, "off-session"


# ================================================================
# V1 CANDLE ANALYSIS HELPERS (kept for legacy pipeline)
# ================================================================

def _detect_deviation(df: pd.DataFrame, range_high: float, range_low: float,
                      current_price: float) -> Dict:
    """Classify price action relative to range extremes as deviation or break."""
    rng = range_high - range_low
    dl_extension = rng * 0.30
    dl_above = range_high + dl_extension
    dl_below = range_low - dl_extension

    recent = df.tail(10) if len(df) >= 10 else df

    at_high = current_price >= range_high * 0.998
    at_low = current_price <= range_low * 1.002

    # Check for closes beyond range
    highs = recent["high"].values
    lows = recent["low"].values
    closes = recent["close"].values

    any_close_above = any(c > range_high for c in closes)
    any_close_below = any(c < range_low for c in closes)
    wick_above = any(h > range_high for h in highs) and not any_close_above
    wick_below = any(low < range_low for low in lows) and not any_close_below
    # DL breach
    close_beyond_dl = any(c > dl_above for c in closes) or any(c < dl_below for c in closes)

    # Immediate reversal (next candle closed back inside)
    immediate_reversal = False
    for i in range(len(closes) - 1):
        if closes[i] > range_high and closes[i + 1] <= range_high:
            immediate_reversal = True
        if closes[i] < range_low and closes[i + 1] >= range_low:
            immediate_reversal = True

    return {
        "at_high": at_high,
        "at_low": at_low,
        "wick_only": wick_above or wick_below,
        "close_outside": any_close_above or any_close_below,
        "immediate_reversal": immediate_reversal,
        "close_beyond_dl": close_beyond_dl,
        "dl_above": dl_above,
        "dl_below": dl_below,
    }


def _detect_fvg(df: pd.DataFrame, zone_idx: int) -> bool:
    """Detect a Fair Value Gap at or near the given candle index.

    FVG exists when the wick of candle 1 and wick of candle 3 do NOT connect
    (there is a gap between them).
    """
    if df is None or zone_idx < 1 or zone_idx >= len(df) - 1:
        return False

    # Check a small window around zone_idx
    for i in range(max(1, zone_idx - 2), min(len(df) - 1, zone_idx + 3)):
        if i < 1 or i >= len(df) - 1:
            continue
        c1_low = df.iloc[i - 1]["low"]
        c1_high = df.iloc[i - 1]["high"]
        c3_low = df.iloc[i + 1]["low"]
        c3_high = df.iloc[i + 1]["high"]

        # Bullish FVG: gap between c1 high and c3 low
        if c3_low > c1_high:
            return True
        # Bearish FVG: gap between c3 high and c1 low
        if c1_low > c3_high:
            return True

    return False


def _find_order_block_near_tap(df: pd.DataFrame, tap_price: float,
                               direction: str, lookback: int = 10) -> Dict:
    """Find an order block (last opposing candle before expansion) near a tap price."""
    if df is None or len(df) < lookback:
        return {"found": False}

    recent = df.tail(lookback)
    best_ob = None
    tolerance = abs(tap_price * 0.005)

    for i in range(len(recent) - 1):
        row = recent.iloc[i]
        next_row = recent.iloc[i + 1]
        ob_idx = len(df) - lookback + i

        if direction == "bullish":
            # Demand OB: last bearish candle before bullish expansion
            if row["close"] < row["open"] and next_row["close"] > next_row["open"]:
                if abs(row["low"] - tap_price) < tolerance or row["low"] <= tap_price:
                    fvg = _detect_fvg(df, ob_idx)
                    best_ob = {
                        "found": True,
                        "price": float(row["low"]),
                        "fvg_confirmed": fvg,
                        "idx": ob_idx,
                    }
        else:
            # Supply OB: last bullish candle before bearish expansion
            if row["close"] > row["open"] and next_row["close"] < next_row["open"]:
                if abs(row["high"] - tap_price) < tolerance or row["high"] >= tap_price:
                    fvg = _detect_fvg(df, ob_idx)
                    best_ob = {
                        "found": True,
                        "price": float(row["high"]),
                        "fvg_confirmed": fvg,
                        "idx": ob_idx,
                    }

    return best_ob or {"found": False}


def _detect_liquidity_sweep(df: pd.DataFrame, range_high: float, range_low: float,
                            direction: str) -> Dict:
    """Detect whether a liquidity sweep occurred at the range extreme."""
    if df is None or len(df) < 5:
        return {"swept": False}

    rng = range_high - range_low
    dl2 = rng * 0.30
    dl2_above = range_high + dl2
    dl2_below = range_low - dl2

    recent = df.tail(20) if len(df) >= 20 else df
    highs = recent["high"].values
    lows = recent["low"].values
    closes = recent["close"].values
    opens = recent["open"].values

    if direction == "bullish":
        # Sell-side swept (below range low)
        exceeded = any(l < range_low for l in lows)
        close_beyond_dl2 = any(c < dl2_below for c in closes)
        wick_beyond_dl2 = any(l < dl2_below for l in lows) and not close_beyond_dl2
        # Acceptance back inside
        accepted = len(closes) >= 2 and closes[-1] >= range_low
        sweep_count = sum(1 for l in lows if l < range_low)

        return {
            "swept": exceeded,
            "sweep_side": "sell_side",
            "pool_type": "range_low",
            "close_beyond_dl2": close_beyond_dl2,
            "wick_beyond_dl2": wick_beyond_dl2,
            "accepted_back": accepted,
            "sweep_count": sweep_count,
        }
    else:
        # Buy-side swept (above range high)
        exceeded = any(h > range_high for h in highs)
        close_beyond_dl2 = any(c > dl2_above for c in closes)
        wick_beyond_dl2 = any(h > dl2_above for h in highs) and not close_beyond_dl2
        accepted = len(closes) >= 2 and closes[-1] <= range_high
        sweep_count = sum(1 for h in highs if h > range_high)

        return {
            "swept": exceeded,
            "sweep_side": "buy_side",
            "pool_type": "range_high",
            "close_beyond_dl2": close_beyond_dl2,
            "wick_beyond_dl2": wick_beyond_dl2,
            "accepted_back": accepted,
            "sweep_count": sweep_count,
        }


def _estimate_path_quality(df: pd.DataFrame, direction: str,
                           entry_price: float, target_price: float) -> PathQuality:
    """Estimate path quality between entry and target by looking for unmitigated OBs."""
    if df is None or len(df) < 10:
        return PathQuality.PARTIAL

    # Simple heuristic: count opposing candle clusters in the path
    closes = df["close"].values[-30:] if len(df) >= 30 else df["close"].values
    opens = df["open"].values[-30:] if len(df) >= 30 else df["open"].values

    if direction == "bullish":
        # Count bearish (supply) candle clusters between entry and target
        obstacles = sum(
            1 for c, o in zip(closes, opens)
            if c < o and entry_price < c < target_price
        )
    else:
        obstacles = sum(
            1 for c, o in zip(closes, opens)
            if c > o and target_price < c < entry_price
        )

    if obstacles <= 2:
        return PathQuality.CLEAN
    elif obstacles <= 5:
        return PathQuality.PARTIAL
    return PathQuality.OBSTRUCTED


def range_integrity_gate(range_high: float, range_low: float, current_price: float,
                         threshold: float = 0.20) -> bool:
    """Return False if current price is near the range equilibrium (0.5 Fib).

    The gate passes (True) when price is in the outer region of the range, and
    fails (False) when price is inside the equilibrium zone.

    Args:
        threshold: Fraction of range size defining the equilibrium zone.
                   Default 0.20 (central 20%). MarketStructureEngine.range_integrity_gate
                   uses 0.10 for a tighter gate at execution level.
    """
    if range_high <= range_low or current_price <= 0:
        return True  # Cannot determine — allow through
    rng = range_high - range_low
    eq = (range_high + range_low) / 2.0
    eq_zone = rng * threshold
    return abs(current_price - eq) > eq_zone


# ================================================================
# BUILDER FUNCTIONS — convert candles+schematic → decision tree inputs
# ================================================================

def build_range_inputs(df: pd.DataFrame, schematic: Dict, current_price: float) -> RangeInputs:
    """Build RangeInputs from candle data and a TCT schematic dict."""
    range_info = schematic.get("range") or {}
    range_high = range_info.get("high", 0)
    range_low = range_info.get("low", 0)

    hh_hl, lh_ll = _detect_trend(df)
    six_candle = _six_candle_rule(df, range_high, range_low)
    eq_touched = _check_equilibrium_touched(df, range_high, range_low)
    horizontal = _range_looks_horizontal(df, range_high, range_low)
    dev = _detect_deviation(df, range_high, range_low, current_price)

    # Timeframe category from schematic or default
    tf = schematic.get("timeframe", "1h")
    tf_cat = "mid"
    if tf in ("5m", "10m", "15m"):
        tf_cat = "low"
    elif tf in ("1d", "3d", "1W"):
        tf_cat = "high"

    # S/D zone overlap near DL (simplified: check if any OB exists near DL)
    direction = schematic.get("direction", "bullish")
    tap3 = schematic.get("tap3") or {}
    tap3_price = tap3.get("price", 0)
    ob_info = _find_order_block_near_tap(df, tap3_price, direction)

    return RangeInputs(
        higher_highs_higher_lows=hh_hl,
        lower_highs_lower_lows=lh_ll,
        six_candle_rule_passes=six_candle,
        equilibrium_touched=eq_touched,
        highest_valid_timeframe=tf.upper(),
        range_looks_horizontal=horizontal,
        price_at_range_high=dev["at_high"],
        price_at_range_low=dev["at_low"],
        is_wick_only=dev["wick_only"],
        close_outside_range=dev["close_outside"],
        immediate_reversal_next_candle=dev["immediate_reversal"],
        close_beyond_dl=dev["close_beyond_dl"],
        higher_tf_looks_like_wick=dev["wick_only"],  # Simplified: use same TF wick check
        demand_zone_overlaps_dl_below=(ob_info.get("found", False) and direction == "bullish"),
        supply_zone_overlaps_dl_above=(ob_info.get("found", False) and direction == "bearish"),
        timeframe_category=tf_cat,
    )


def build_sd_inputs(df: pd.DataFrame, schematic: Dict, range_eval: RangeEvaluation) -> SDZoneInputs:
    """Build SDZoneInputs from candle data, schematic, and range evaluation."""
    direction = schematic.get("direction", "bullish")
    tap3 = schematic.get("tap3") or {}
    tap3_price = tap3.get("price", 0)
    tap3_idx = tap3.get("idx", len(df) - 5 if df is not None else 0)

    # Derive market context from range evaluation.
    # When the range tree couldn't determine trend, infer from schematic direction
    # since the schematic detector already validated the range structure.
    if range_eval.range_valid:
        ctx = MarketContext.RANGE
    elif range_eval.trend == Trend.UPTREND:
        ctx = MarketContext.UPTREND
    elif range_eval.trend == Trend.DOWNTREND:
        ctx = MarketContext.DOWNTREND
    elif direction == "bullish":
        ctx = MarketContext.UPTREND  # Schematic detector found accumulation → imply uptrend context
    elif direction == "bearish":
        ctx = MarketContext.DOWNTREND
    else:
        ctx = MarketContext.UNCLEAR

    zone_dir = ZoneDirection.DEMAND if direction == "bullish" else ZoneDirection.SUPPLY
    ob_info = _find_order_block_near_tap(df, tap3_price, direction, lookback=15)

    fvg_confirmed = ob_info.get("fvg_confirmed", False)
    # If no OB found, try detecting FVG directly near tap3
    if not ob_info.get("found"):
        fvg_confirmed = _detect_fvg(df, tap3_idx)

    return SDZoneInputs(
        market_context=ctx,
        zone_direction=zone_dir,
        context_reason_exists=range_eval.range_valid or range_eval.trend is not None,
        zone_type=ZoneType.ORDER_BLOCK if ob_info.get("found") else ZoneType.STRUCTURE_ZONE,
        fvg_confirmed=fvg_confirmed,
        fvg_tapped_from_top_down=(direction == "bearish"),
        adjacent_candle_has_more_extreme_wick=False,  # Conservative default
        mitigation_status=MitigationStatus.UNMITIGATED,  # Fresh tap
        is_only_zone_in_area=True,  # Simplified: assume primary zone
        refined_ob_found_on_lower_tf=False,  # Not checking LTF here
        higher_tf_ob_unmitigated=False,
        higher_tf_ob_mitigated_on_lower=False,
        is_supply_chain_second_ob=False,
        is_extreme_zone=True,  # At range extreme by definition (TCT tap)
        price_inside_zone=True,  # Price is at the tap
        tct_schematic_confirmed=schematic.get("is_confirmed", False),
    )


def build_liquidity_inputs(df: pd.DataFrame, schematic: Dict,
                           range_eval: RangeEvaluation) -> LiquidityInputs:
    """Build LiquidityInputs from candle data and schematic."""
    direction = schematic.get("direction", "bullish")
    range_info = schematic.get("range") or {}
    range_high = range_info.get("high", 0)
    range_low = range_info.get("low", 0)

    sweep = _detect_liquidity_sweep(df, range_high, range_low, direction)

    if direction == "bullish":
        sweep_side = SweepSide.SELL_SIDE
        pool_type = LiquidityPoolType.RANGE_LOW
    else:
        sweep_side = SweepSide.BUY_SIDE
        pool_type = LiquidityPoolType.RANGE_HIGH

    entry_price = (schematic.get("entry") or {}).get("price", 0)
    target_price = (schematic.get("target") or {}).get("price", 0)
    path_q = _estimate_path_quality(df, direction, entry_price, target_price)

    return LiquidityInputs(
        pool_type=pool_type,
        sweep_side=sweep_side,
        is_trend_line_or_curve=False,
        no_sd_backing_the_level=True,
        times_this_side_swept=sweep.get("sweep_count", 1),
        other_side_untouched=True,  # Simplified
        price_exceeded_range_extreme=sweep.get("swept", False),
        any_candle_closed_beyond_dl2=sweep.get("close_beyond_dl2", False),
        wick_only_beyond_dl2=sweep.get("wick_beyond_dl2", False),
        accepted_back_inside_range=sweep.get("accepted_back", False),
        path_quality=path_q,
        retail_trapped_in_wrong_direction=False,  # Conservative default
        tct_schematic_confirmed=schematic.get("is_confirmed", False),
    )


def build_5a_inputs(schematic: Dict, range_eval: RangeEvaluation) -> TCTSchematicInputs:
    """Build TCT5A inputs directly from schematic dict fields."""
    direction = schematic.get("direction", "bullish")
    model = schematic.get("model", schematic.get("schematic_type", ""))
    tap2 = schematic.get("tap2") or {}
    tap3 = schematic.get("tap3") or {}
    bos = schematic.get("bos_confirmation") or {}

    is_acc = direction == "bullish"
    sch_dir = Dir5A.ACCUMULATION if is_acc else Dir5A.DISTRIBUTION

    # Determine model type
    if "Model_3" in model:
        # Model 3 (continuation) uses same tap structure — map to MODEL_1
        # for the 5A decision tree which only knows MODEL_1/MODEL_2.
        model_type = MT5A.MODEL_1
    elif "Model_1" in model:
        model_type = MT5A.MODEL_1
    elif "Model_2" in model:
        model_type = MT5A.MODEL_2
    else:
        model_type = MT5A.UNKNOWN

    # Tap 2 validation
    tap2_exceeded = bool(tap2.get("price"))  # Tap 2 exists = exceeded range extreme
    tap2_close_beyond_dl2 = False  # If schematic exists, tap2 passed DL2 check
    tap2_accepted = bool(tap2.get("price"))

    # Tap 3 fields
    tap3_m1_exceeded = tap3.get("type") == "tap3_model1"
    tap3_m2_hl_lh = tap3.get("is_higher_low", False) or tap3.get("is_lower_high", False)

    # BOS
    bos_confirmed = schematic.get("is_confirmed", False)
    bos_loc = BOSLocation.INSIDE_RANGE if bos_confirmed else BOSLocation.NOT_YET
    bos_tf = BOSTimeframe.BLACK_HIGH_TF if bos_confirmed else BOSTimeframe.NOT_YET

    # The schematic detector already validated range structure; trust it
    # even if the range decision tree couldn't confirm via the 6-candle rule
    # (the tree's stricter criteria may not match the detector's logic).
    range_confirmed_for_5a = range_eval.range_valid or schematic.get("is_confirmed", False)

    return TCTSchematicInputs(
        range_confirmed=range_confirmed_for_5a,
        dl2_drawn=True,  # DL2 is always computed by the detector
        direction=sch_dir,
        tap2_price_exceeded_extreme=tap2_exceeded,
        tap2_close_beyond_dl2=tap2_close_beyond_dl2,
        tap2_accepted_back_inside=tap2_accepted,
        tap2_grabbed_major_liquidity=False,
        tap2_mitigated_strong_sd=False,
        model_type=model_type,
        tap3_m1_exceeded_tap2=tap3_m1_exceeded,
        tap3_m1_close_beyond_dl2=False,
        tap3_m2_is_higher_low_or_lower_high=tap3_m2_hl_lh,
        tap3_m2_req_a_extreme_liq=tap3_m2_hl_lh,  # Simplified: HL/LH implies requirement met
        tap3_m2_req_b_extreme_sd=False,
        tap3_m2_sd_in_extreme_zone=False,
        tap3_m2_sd_tf_proportional=True,
        bos_confirmed=bos_confirmed,
        bos_wrong_direction=False,
        bos_location=bos_loc,
        bos_timeframe=bos_tf,
        black_bos_poor_rr=False,
        red_bos_inside_range=False,
    )


def build_5b_inputs(schematic: Dict, eval_5a: TCTSchematicEvaluation,
                    range_eval: RangeEvaluation, rr: float) -> TCT5BInputs:
    """Build TCT5B inputs from schematic, 5A result, and range evaluation."""
    direction = schematic.get("direction", "bullish")
    model = schematic.get("model", schematic.get("schematic_type", ""))

    is_acc = direction == "bullish"
    sch_dir = Dir5B.ACCUMULATION if is_acc else Dir5B.DISTRIBUTION

    if "Model_3" in model:
        model_type = MT5B.MODEL_1  # Model 3 uses same structure as MODEL_1
    elif "Model_1" in model:
        model_type = MT5B.MODEL_1
    elif "Model_2" in model:
        model_type = MT5B.MODEL_2
    else:
        model_type = MT5B.MODEL_1  # Default

    # Tap spacing: check if tap2-tap3 gap is reasonable
    tap2 = schematic.get("tap2") or {}
    tap3 = schematic.get("tap3") or {}
    tap1 = schematic.get("tap1") or {}
    t1_idx = tap1.get("idx", 0)
    t2_idx = tap2.get("idx", 0)
    t3_idx = tap3.get("idx", 0)
    t12_gap = max(1, t2_idx - t1_idx)
    t23_gap = max(1, t3_idx - t2_idx)
    tap_spacing_ok = t23_gap >= t12_gap * 0.3  # At least 30% of T1-T2 gap

    # Deviations are wicks or bad breaks if schematic passed the detector
    deviations_ok = schematic.get("quality_score", 0) >= 0.5

    tf = schematic.get("timeframe", "1h")

    return TCT5BInputs(
        direction=sch_dir,
        model_type=model_type,
        tap2_valid=eval_5a.status != SchematicStatus.INVALID,
        tap3_valid=eval_5a.status != SchematicStatus.INVALID,
        range_looks_horizontal=range_eval.range_rational if range_eval.range_valid else True,
        deviations_are_wicks_or_bad_breaks=deviations_ok,
        highest_valid_tf=tf.upper(),
        tap23_gap_reasonable=tap_spacing_ok,
        extreme_liq_obvious=True,  # Simplified: assume detector found valid setup
        extreme_sd_obvious=False,
        sd_tf_proportional_to_range=True,
        trendline_liquidity_present=False,
        primary_bos_rr=rr,
        lower_tf_bos_available=(rr < 1.5),  # Available if we need it
        lower_tf_bos_entry=EntryTimeframe.BLUE_LOW if rr < 1.5 else EntryTimeframe.BLACK_PRIMARY,
        lower_tf_bos_inside_range=True,
        bos_inside_sd_zone=False,
        retest_occurred=False,
        retest_blue_confirmed=False,
        bos_confirmed=schematic.get("is_confirmed", False),
        ltf_bullish_break_then_new_low=False,
    )


def build_flip_inputs(schematic: Dict, active_trade: Optional[Dict]) -> Optional[FlipInputs]:
    """Build FlipInputs for Tree 6 — only when there is an active trade."""
    if not active_trade:
        return None

    trade_dir = active_trade.get("direction", "")
    if trade_dir == "bullish":
        active_dir = Dir6.ACCUMULATION
    elif trade_dir == "bearish":
        active_dir = Dir6.DISTRIBUTION
    else:
        return None

    # Check if opposing extreme was deviated in the schematic
    sch_dir = schematic.get("direction", "")
    opposing = (
        (trade_dir == "bullish" and sch_dir == "bearish") or
        (trade_dir == "bearish" and sch_dir == "bullish")
    )

    return FlipInputs(
        active_schematic_direction=active_dir,
        opposite_extreme_deviated_before_confirm=opposing,
        opposing_bos_confirmed=opposing and schematic.get("is_confirmed", False),
        supply_demand_at_opposite_extreme=opposing,
    )


# ================================================================
# COMPOSITE SCORING — run all trees, produce final evaluation dict
# ================================================================

def compute_composite_score(
    df: pd.DataFrame,
    schematic: Dict,
    htf_bias: str,
    current_price: float,
    active_trade: Optional[Dict] = None,
) -> Dict:
    """
    Run the 6-tree decision pipeline on a schematic and return a score dict
    compatible with the existing _enter_trade() interface.

    Returns:
        {
            "score": int (0-100),
            "pass": bool,
            "direction": str,
            "model": str,
            "rr": float,
            "required_score": 50,
            "reasons": [str],
            "tree_results": {tree_name: {passed, phases, failed_at}},
        }
    """
    direction = schematic.get("direction", "unknown")
    model = schematic.get("model", schematic.get("schematic_type", "unknown"))
    is_confirmed = schematic.get("is_confirmed", False)

    # Live R:R calculation (same as old evaluator)
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

    fail = {
        "score": 0, "direction": direction, "model": model, "rr": rr,
        "required_score": 50, "pass": False, "tree_results": {},
    }

    if not is_confirmed:
        return {**fail, "reasons": ["No BOS confirmation"]}

    # Pre-gate: HTF alignment
    if direction == "bullish" and htf_bias == "bearish":
        return {**fail, "reasons": ["HTF bias conflict (bearish vs bullish)"]}
    if direction == "bearish" and htf_bias == "bullish":
        return {**fail, "reasons": ["HTF bias conflict (bullish vs bearish)"]}
    if htf_bias == "neutral":
        return {**fail, "reasons": ["HTF bias neutral — no directional clarity"]}

    # Pre-gate: minimum R:R
    if rr < 1.5:
        return {**fail, "reasons": [f"R:R too low ({rr:.1f})"]}

    reasons = []
    tree_results = {}
    score = 0

    # ── Tree 1: Ranges ──
    try:
        range_inputs = build_range_inputs(df, schematic, current_price)
        range_eval = evaluate_range_setup(range_inputs)
        tree_passed = range_eval.failed_at_phase is None
        tree_results["ranges"] = {
            "passed": tree_passed,
            "phases": range_eval.passed_phases,
            "failed_at": range_eval.failed_at_phase,
            "trade_bias": range_eval.trade_bias.value if range_eval.trade_bias else None,
            "deviation_type": range_eval.deviation_type.value if range_eval.deviation_type else None,
        }
        if tree_passed:
            score += 20
            reasons.append(f"Range valid ({range_eval.deviation_type.value})")
        else:
            # Range tree failure is soft — the schematic detector already validated range
            # structure, so we downgrade rather than hard-fail
            score += 5
            reasons.append(f"Range: {range_eval.failed_at_phase}")
    except Exception as e:
        logger.warning(f"[BRIDGE] Range tree error: {e}", exc_info=True)
        tree_results["ranges"] = {"passed": False, "error": str(e)}
        score += 5  # Soft pass
        reasons.append(f"Range tree error: {e}")

    # ── Tree 2: Supply & Demand ──
    try:
        sd_inputs = build_sd_inputs(df, schematic, range_eval)
        sd_eval = evaluate_sd_zone(sd_inputs)
        tree_passed = sd_eval.failed_at_phase is None
        tree_results["supply_demand"] = {
            "passed": tree_passed,
            "phases": sd_eval.passed_phases,
            "failed_at": sd_eval.failed_at_phase,
            "fvg_valid": sd_eval.fvg_valid,
            "priority": sd_eval.priority.value if sd_eval.priority else None,
        }
        if tree_passed:
            score += 20
            reasons.append(f"S/D zone valid (FVG={'YES' if sd_eval.fvg_valid else 'NO'})")
        else:
            # FVG failure is hard — no valid zone means no trade
            if not sd_eval.fvg_valid:
                return {**fail, "reasons": [f"S/D: No FVG — {sd_eval.failed_at_phase}"],
                        "tree_results": tree_results}
            score += 5
            reasons.append(f"S/D: {sd_eval.failed_at_phase}")
    except Exception as e:
        logger.warning(f"[BRIDGE] S/D tree error: {e}", exc_info=True)
        tree_results["supply_demand"] = {"passed": False, "error": str(e)}
        score += 5
        reasons.append(f"S/D tree error: {e}")

    # ── Tree 3: Liquidity ──
    try:
        liq_inputs = build_liquidity_inputs(df, schematic, range_eval)
        liq_eval = evaluate_liquidity_setup(liq_inputs)
        tree_passed = liq_eval.failed_at_phase is None
        tree_results["liquidity"] = {
            "passed": tree_passed,
            "phases": liq_eval.passed_phases,
            "failed_at": liq_eval.failed_at_phase,
            "sweep_class": liq_eval.sweep_classification.value,
            "path_quality": liq_eval.path_quality.value if liq_eval.path_quality else None,
            "conviction": liq_eval.conviction_level,
        }
        if tree_passed:
            score += 15
            reasons.append(f"Liquidity confirmed ({liq_eval.sweep_classification.value})")
        else:
            # True break is a hard fail
            from decision_trees.liquidity_decision_tree import SweepClassification
            if liq_eval.sweep_classification == SweepClassification.TRUE_BREAK:
                return {**fail, "reasons": [f"Liquidity: TRUE BREAK — {liq_eval.failed_at_phase}"],
                        "tree_results": tree_results}
            score += 5
            reasons.append(f"Liquidity: {liq_eval.failed_at_phase}")
    except Exception as e:
        logger.warning(f"[BRIDGE] Liquidity tree error: {e}", exc_info=True)
        tree_results["liquidity"] = {"passed": False, "error": str(e)}
        score += 5
        reasons.append(f"Liquidity tree error: {e}")

    # ── Tree 4: TCT 5A ──
    try:
        inputs_5a = build_5a_inputs(schematic, range_eval)
        eval_5a = evaluate_tct_schematic(inputs_5a)
        tree_passed = eval_5a.status == SchematicStatus.VALID_ENTRY
        tree_results["tct_5a"] = {
            "passed": tree_passed,
            "phases": eval_5a.passed_phases,
            "failed_at": eval_5a.failed_at_phase,
            "status": eval_5a.status.value,
            "model_type": eval_5a.model_type.value if eval_5a.model_type else None,
        }
        if tree_passed:
            score += 20
            reasons.append(f"5A schematic valid ({eval_5a.model_type.value if eval_5a.model_type else model})")
        else:
            # 5A failure is hard if schematic is INVALID, soft if just FORMING
            if eval_5a.status == SchematicStatus.INVALID:
                return {**fail, "reasons": [f"5A: {eval_5a.failed_at_phase}"],
                        "tree_results": tree_results}
            score += 5
            reasons.append(f"5A: {eval_5a.failed_at_phase}")
    except Exception as e:
        logger.warning(f"[BRIDGE] 5A tree error: {e}", exc_info=True)
        tree_results["tct_5a"] = {"passed": False, "error": str(e)}
        eval_5a = TCTSchematicEvaluation()
        score += 5
        reasons.append(f"5A tree error: {e}")

    # ── Tree 5: TCT 5B ──
    try:
        inputs_5b = build_5b_inputs(schematic, eval_5a, range_eval, rr)
        eval_5b = evaluate_5b_schematic(inputs_5b)
        tree_passed = eval_5b.status == Status5B.VALID_ENTRY
        tree_results["tct_5b"] = {
            "passed": tree_passed,
            "phases": eval_5b.passed_phases,
            "failed_at": eval_5b.failed_at_phase,
            "status": eval_5b.status.value,
            "entry_tf": eval_5b.entry_timeframe.value if eval_5b.entry_timeframe else None,
            "primary_rr": eval_5b.primary_bos_rr,
        }
        if tree_passed:
            score += 15
            reasons.append(f"5B real-world checks pass (entry: {eval_5b.entry_timeframe.value})")
        else:
            # Skip bad R:R is hard fail
            if eval_5b.status == Status5B.SKIP_BAD_RR:
                return {**fail, "reasons": [f"5B: {eval_5b.failed_at_phase}"],
                        "tree_results": tree_results}
            score += 5
            reasons.append(f"5B: {eval_5b.failed_at_phase}")
    except Exception as e:
        logger.warning(f"[BRIDGE] 5B tree error: {e}", exc_info=True)
        tree_results["tct_5b"] = {"passed": False, "error": str(e)}
        score += 5
        reasons.append(f"5B tree error: {e}")

    # ── Tree 6: Advanced (enhancement layer) ──
    try:
        flip_inputs = build_flip_inputs(schematic, active_trade)
        if flip_inputs:
            flip_eval = evaluate_schematic_flip(flip_inputs)
            tree_results["advanced_flip"] = {
                "status": flip_eval.status.value,
                "notes": flip_eval.notes,
                "warnings": flip_eval.warnings,
            }
            if flip_eval.status == OutcomeStatus.FLIP_EXIT_AND_ENTER:
                # Flip detected — flag it but don't block new entry
                score += 5
                reasons.append("FLIP detected — opposing schematic confirmed")
            elif flip_eval.status == OutcomeStatus.HOLD_TRADE:
                score += 5
                reasons.append("No flip risk")
        else:
            tree_results["advanced_flip"] = {"status": "not_applicable"}
            score += 5  # No active trade, no flip check needed
    except Exception as e:
        logger.warning(f"[BRIDGE] Advanced tree error: {e}", exc_info=True)
        tree_results["advanced_flip"] = {"error": str(e)}
        score += 5

    # R:R bonus
    if rr >= 3.0:
        score += 5
        reasons.append(f"Excellent R:R ({rr:.1f})")
    elif rr >= 2.0:
        score += 3
        reasons.append(f"Good R:R ({rr:.1f})")
    else:
        reasons.append(f"Acceptable R:R ({rr:.1f})")

    score = max(0, min(100, score))
    return {
        "score": score,
        "direction": direction,
        "model": model,
        "rr": rr,
        "required_score": 50,
        "pass": score >= 50,
        "reasons": reasons,
        "tree_results": tree_results,
    }


# ================================================================
# V2 COMPOSITE SCORING — 9-phase pipeline (active)
# ================================================================

V2_THRESHOLD = 60

def compute_composite_score_v2(
    df: pd.DataFrame,
    schematic: Dict,
    htf_bias: str,
    current_price: float,
    active_trade: Optional[Dict] = None,
) -> Dict:
    """Run the 9-phase decision pipeline on a schematic.

    Phase order:
        1. HTF Context        — already resolved by caller (htf_bias param)
        2. Range Detection    — time displacement, liquidity stacking, horizontal, V-shape
        3. Tap Structure      — Model 1/2 validation (BEFORE BOS)
        4. Liquidity          — sweep with tolerance for slight closes beyond
        5. Break of Structure — internal MS break after Tap3
        6. POI Validation     — FVG / OB / MM block (optional, not hard gate)
        7. Directional Filter — HTF alignment (with reversal exception)
        8. Risk Filter        — minimum R:R 1.5
        9. Confidence Scoring — 6-component score, threshold 60

    Returns same shape as v1 for backward compatibility:
        {"score", "pass", "direction", "model", "rr", "required_score",
         "reasons", "tree_results", "phase_results"}
    """
    direction = schematic.get("direction", "unknown")
    model = schematic.get("model", schematic.get("schematic_type", "unknown"))
    is_confirmed = schematic.get("is_confirmed", False)

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

    fail = {
        "score": 0, "direction": direction, "model": model, "rr": rr,
        "required_score": V2_THRESHOLD, "pass": False, "tree_results": {},
        "phase_results": {},
    }

    reasons: List[str] = []
    phase_results: Dict = {}
    score = 0

    # ── Phase 1: HTF Context ──
    phase_results["htf_context"] = {"bias": htf_bias}

    # ============================================================
    # MARKET STRUCTURE ENGINE INIT
    # ============================================================
    from decision_trees.market_structure_engine import MarketStructureEngine
    mse = MarketStructureEngine()

    # ============================================================
    # L2 STRUCTURE BLOCK (COUNTER-STRUCTURE FILTER)
    # ============================================================
    l2 = mse.detect_l2_structure(df, htf_bias)

    if l2.get("exists"):
        phase_results["l2"] = {
            "passed": False,
            "reason": "L2 counter-structure (internal reversal active)",
            "data": l2
        }
        return {**fail,
                "reasons": ["L2 counter-structure (internal reversal active)"],
                "phase_results": phase_results}

    phase_results["l2"] = {"passed": True}

    # ── Phase 2: Range Detection ──
    range_info = schematic.get("range") or {}
    range_high = range_info.get("high", 0)
    range_low = range_info.get("low", 0)

    # ============================================================
    # RIG (RANGE INTEGRITY GATE)
    # ============================================================
    if not range_integrity_gate(range_high, range_low, current_price):
        phase_results["rig"] = {
            "passed": False,
            "reason": "RIG: price at equilibrium (no edge)"
        }
        return {**fail,
                "reasons": ["RIG: price at equilibrium (no edge)"],
                "phase_results": phase_results}

    phase_results["rig"] = {"passed": True}

    time_ok, time_gap = _check_time_displacement(schematic)
    liq_stack = _detect_liquidity_stacking(df, range_high, range_low)
    is_v_shape = _reject_v_shape(df, range_high, range_low)
    horizontal = _range_looks_horizontal(df, range_high, range_low)
    six_candle = _six_candle_rule(df, range_high, range_low)

    range_score = 0
    range_checks = {
        "time_displacement_ok": time_ok,
        "time_gap_candles": time_gap,
        "liquidity_stacking": liq_stack,
        "v_shape_rejected": is_v_shape,
        "horizontal": horizontal,
        "six_candle_rule": six_candle,
    }

    if is_v_shape:
        phase_results["range"] = {**range_checks, "passed": False, "reason": "V-shape / impulsive move"}
        return {**fail,
                "reasons": ["Phase 2: Range rejected — V-shape / impulsive move"],
                "phase_results": phase_results}

    if not time_ok:
        phase_results["range"] = {**range_checks, "passed": False, "reason": "insufficient time displacement"}
        return {**fail,
                "reasons": [f"Phase 2: Insufficient time displacement ({time_gap} candles)"],
                "phase_results": phase_results}

    if horizontal and six_candle:
        range_score = 20
    elif horizontal or six_candle:
        range_score = 14
    else:
        range_score = 8

    if liq_stack["has_stacking"]:
        range_score = min(20, range_score + 3)

    phase_results["range"] = {**range_checks, "passed": True, "score": range_score}
    score += range_score
    reasons.append(f"Range: {range_score}/20")

    # ── Phase 3: Tap Structure ──
    tap1 = schematic.get("tap1") or {}
    tap2 = schematic.get("tap2") or {}
    tap3 = schematic.get("tap3") or {}

    if not tap1.get("price") or not tap2.get("price") or not tap3.get("price"):
        phase_results["tap_structure"] = {"passed": False}
        return {**fail,
                "score": score,
                "reasons": reasons + ["Missing tap structure"],
                "phase_results": phase_results}

    model_str = schematic.get("model", "")

    if "Model_3" in model_str:
        model_type = "Model_3"
    elif "Model_1" in model_str:
        model_type = "Model_1"
    elif "Model_2" in model_str:
        model_type = "Model_2"
    else:
        model_type = "unknown"

    tap_valid = True

    # tap prices are guaranteed non-None by the guard above
    if model_type == "Model_1":
        if direction == "bullish":
            tap_valid = tap3["price"] < tap2["price"]
        else:
            tap_valid = tap3["price"] > tap2["price"]

    elif model_type == "Model_2":
        if direction == "bullish":
            tap_valid = tap3["price"] > tap2["price"]
        else:
            tap_valid = tap3["price"] < tap2["price"]

    elif model_type == "Model_3":
        # Model 3 uses same tap structure as Model 1 or 2 —
        # already validated by _build_accumulation/distribution_schematic.
        # Accept both deviation patterns (M1-style lower/higher OR M2-style HL/LH).
        tap_valid = True

    if not tap_valid:
        phase_results["tap_structure"] = {"passed": False}
        return {**fail,
                "score": score,
                "reasons": reasons + ["Invalid tap structure"],
                "phase_results": phase_results}

    tap_score = 20
    phase_results["tap_structure"] = {"passed": True, "score": tap_score, "model": model_type}
    score += tap_score
    reasons.append(f"Taps: {tap_score}/20")

    # ── Phase 4: Liquidity ──
    sweep_v2 = _detect_liquidity_sweep_v2(df, range_high, range_low, direction)

    if sweep_v2["classification"] == "true_break":
        phase_results["liquidity"] = {"passed": False}
        return {**fail,
                "score": score,
                "reasons": reasons + ["True break — not a sweep"],
                "phase_results": phase_results}

    liq_score = 20 if sweep_v2["swept"] else 5

    phase_results["liquidity"] = {**sweep_v2, "passed": True, "score": liq_score}
    score += liq_score
    reasons.append(f"Liquidity: {liq_score}/20")

    # ── Phase 4.5: L3 EXECUTION STRUCTURE (REAL) ──
    l3_valid = mse.detect_l3_structure(df, direction)

    if not l3_valid:
        phase_results["l3"] = {"passed": False}
        return {**fail,
                "score": score,
                "reasons": reasons + ["No L3 execution confirmation"],
                "phase_results": phase_results}

    phase_results["l3"] = {"passed": True}

    # ── Phase 5: BOS ──
    bos = schematic.get("bos_confirmation") or {}
    bos_idx = bos.get("bos_idx")
    tap3_idx = tap3.get("idx")  # No default — missing idx must be treated as a failure

    if not is_confirmed or bos_idx is None or tap3_idx is None or bos_idx < tap3_idx:
        phase_results["bos"] = {"passed": False}
        return {**fail,
                "score": score,
                "reasons": reasons + ["Invalid BOS sequence"],
                "phase_results": phase_results}

    bos_score = 20
    phase_results["bos"] = {"passed": True, "score": bos_score}
    score += bos_score
    reasons.append(f"BOS: {bos_score}/20")

    # ── Phase 6: POI Validation ──
    # FVG is optional — increases confidence but is not mandatory
    tap3_price = tap3.get("price", 0)
    tap3_df_idx = tap3.get("idx", len(df) - 5 if df is not None else 0)
    ob_info = _find_order_block_near_tap(df, tap3_price, direction, lookback=15)
    fvg_found = _detect_fvg(df, tap3_df_idx)

    poi_score = 0
    poi_type = "none"
    if fvg_found and ob_info.get("found"):
        poi_score = 10
        poi_type = "FVG + OB"
    elif fvg_found:
        poi_score = 10
        poi_type = "FVG"
    elif ob_info.get("found"):
        poi_score = 7
        poi_type = "OB only"
    else:
        poi_score = 0
        poi_type = "none"
        # NOT a hard gate — just means less confluence

    phase_results["poi"] = {"passed": True, "score": poi_score, "type": poi_type,
                            "fvg_found": fvg_found, "ob_found": ob_info.get("found", False)}
    score += poi_score
    reasons.append(f"POI: {poi_score}/10 ({poi_type})")

    # ── Phase 7: Directional Filter ──
    # Trade direction must align with HTF bias UNLESS the setup is a confirmed
    # HTF reversal schematic (e.g., a confirmed accumulation in a bearish trend
    # that signals a structural reversal).
    aligned = True
    is_reversal = False
    if direction == "bullish" and htf_bias == "bearish":
        # Check for reversal exception: if this is a confirmed accumulation
        # schematic with strong BOS, it may be a valid reversal
        if is_confirmed and bos_score >= 20 and tap_score >= 20:
            is_reversal = True
            reasons.append("Directional: HTF reversal exception (confirmed accumulation vs bearish HTF)")
        else:
            aligned = False
    elif direction == "bearish" and htf_bias == "bullish":
        if is_confirmed and bos_score >= 20 and tap_score >= 20:
            is_reversal = True
            reasons.append("Directional: HTF reversal exception (confirmed distribution vs bullish HTF)")
        else:
            aligned = False
    elif htf_bias == "neutral":
        aligned = False

    if not aligned:
        phase_results["directional"] = {"passed": False, "aligned": False,
                                         "direction": direction, "htf_bias": htf_bias}
        return {**fail, "score": score,
                "reasons": reasons + [f"Phase 7: HTF bias conflict ({htf_bias} vs {direction})"],
                "phase_results": phase_results}

    phase_results["directional"] = {"passed": True, "aligned": not is_reversal,
                                     "reversal": is_reversal,
                                     "direction": direction, "htf_bias": htf_bias}

    # ── Phase 8: Risk Filter ──
    if rr < 1.5:
        phase_results["risk"] = {"passed": False, "rr": rr}
        return {**fail, "score": score,
                "reasons": reasons + [f"Phase 8: R:R too low ({rr:.1f} < 1.5)"],
                "phase_results": phase_results}

    phase_results["risk"] = {"passed": True, "rr": round(rr, 2)}

    # ── Phase 9: Confidence Scoring ──
    # Session timing component (max 10 points)
    session_pts, session_desc = _score_session_timing()
    score += session_pts
    reasons.append(f"Session: {session_pts}/10 ({session_desc})")
    phase_results["session"] = {"score": session_pts, "session": session_desc}

    # R:R bonus (folded into BOS quality — already scored above)
    if rr >= 3.0:
        score += 3
        reasons.append(f"R:R bonus: +3 ({rr:.1f})")
    elif rr >= 2.0:
        score += 1
        reasons.append(f"R:R bonus: +1 ({rr:.1f})")

    # Reversal penalty — aligned trades score higher than reversals
    if is_reversal:
        penalty = 5
        score -= penalty
        reasons.append(f"Reversal penalty: -{penalty}")

    score = max(0, min(100, score))

    return {
        "score": score,
        "direction": direction,
        "model": model,
        "rr": rr,
        "required_score": V2_THRESHOLD,
        "pass": score >= V2_THRESHOLD,
        "reasons": reasons,
        "tree_results": {
            # Backward-compatible tree_results shape for UI
            "ranges": {"passed": not is_v_shape and time_ok and (horizontal or six_candle),
                       "score": range_score},
            "market_structure": {"passed": htf_bias in ("bullish", "bearish"),
                                 "bias": htf_bias},
            "supply_demand": {"passed": poi_score > 0, "fvg_valid": fvg_found,
                              "ob_found": ob_info.get("found", False)},
            "liquidity": {"passed": sweep_v2["classification"] != "true_break",
                          "sweep_class": sweep_v2["classification"]},
            "schematics_5a": {"passed": tap_valid and is_confirmed,
                              "model_type": model_type, "status": "VALID_ENTRY" if tap_valid else "INVALID"},
            "schematics_5b": {"passed": bos_score > 0,
                              "entry_tf": "primary", "primary_rr": rr},
            "advanced_flip": {"status": "not_applicable"},
        },
        "phase_results": phase_results,
    }


# ================================================================
# PUBLIC EVALUATOR CLASS — drop-in replacement for Schematics5BEvaluator
# ================================================================

class DecisionTreeEvaluator:
    """
    Evaluates TCT schematics using the 9-phase decision pipeline (v2).
    Drop-in replacement for Schematics5BEvaluator — same evaluate_schematic interface.
    """

    def __init__(self):
        pass  # Stateless — v2 pipeline does not use flip detection
        # TODO: re-implement flip detection for v2 if needed

    def evaluate_schematic(self, schematic: Dict, htf_bias: str, current_price: float,
                           total_candles: int = 200, max_stale_candles: int = 5,
                           candle_df: Optional[pd.DataFrame] = None) -> Dict:
        """
        Evaluate a schematic using the 9-phase v2 pipeline.

        Same return format as before:
            {"score", "pass", "direction", "model", "rr", "required_score", "reasons"}

        Plus "tree_results" (backward-compatible) and "phase_results" (v2 detail).
        """
        direction = schematic.get("direction", "unknown")
        model = schematic.get("model", schematic.get("schematic_type", "unknown"))
        is_confirmed = schematic.get("is_confirmed", False)

        # Compute R:R first for early-exit checks
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

        fail = {
            "score": 0, "direction": direction, "model": model, "rr": rr,
            "required_score": V2_THRESHOLD, "pass": False, "tree_results": {},
            "phase_results": {},
        }

        # Pre-gate: BOS must be confirmed
        if not is_confirmed:
            return {**fail, "reasons": ["No BOS confirmation"]}

        # Pre-gate: stale BOS check
        bos = schematic.get("bos_confirmation") or {}
        bos_idx = bos.get("bos_idx")
        if bos_idx is not None and bos_idx < total_candles - max_stale_candles:
            return {**fail, "reasons": [f"Stale BOS: {total_candles - bos_idx} candles ago (max {max_stale_candles})"]}

        # Run the v2 9-phase pipeline if candle data is available
        if candle_df is not None and len(candle_df) > 0:
            return compute_composite_score_v2(
                candle_df, schematic, htf_bias, current_price,
            )

        # Fallback: no candle data — simplified scoring
        return self._fallback_score(schematic, htf_bias, current_price, rr)

    def _fallback_score(self, schematic: Dict, htf_bias: str,
                        current_price: float, rr: float) -> Dict:
        """Simplified scoring when candle data is not available."""
        direction = schematic.get("direction", "unknown")
        model = schematic.get("model", schematic.get("schematic_type", "unknown"))
        score = 0
        reasons = []

        fail = {
            "score": 0, "direction": direction, "model": model, "rr": rr,
            "required_score": V2_THRESHOLD, "pass": False, "tree_results": {},
            "phase_results": {},
        }

        # R:R gate
        if rr < 1.5:
            return {**fail, "reasons": [f"R:R too low ({rr:.1f})"]}

        # Quality gate
        quality_score = schematic.get("quality_score", 0.0)
        if quality_score < 0.70:
            return {**fail, "reasons": [f"Quality too low ({quality_score:.2f} < 0.70)"]}

        # BOS confirmed base
        score += 30
        reasons.append("BOS confirmed")

        # R:R scoring
        if rr >= 3.0:
            score += 25
            reasons.append(f"Excellent R:R ({rr:.1f})")
        elif rr >= 2.0:
            score += 15
            reasons.append(f"Good R:R ({rr:.1f})")
        elif rr >= 1.5:
            score += 5
            reasons.append(f"Acceptable R:R ({rr:.1f})")

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
            return {**fail, "reasons": [f"HTF bias conflict ({htf_bias} vs {direction})"]}

        # Quality bonus
        quality_bonus = round(quality_score * 15)
        score += quality_bonus
        reasons.append(f"Quality {quality_score:.2f} (+{quality_bonus})")

        score = max(0, min(100, score))
        return {
            "score": score, "direction": direction, "model": model, "rr": rr,
            "required_score": V2_THRESHOLD, "pass": score >= V2_THRESHOLD, "reasons": reasons,
            "tree_results": {"mode": "fallback_no_candle_data"},
            "phase_results": {"mode": "fallback"},
        }
