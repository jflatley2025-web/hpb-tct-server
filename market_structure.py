"""
market_structure.py – TCT Lecture 1 Market Structure Engine (Rebuilt)

Implements the full market structure detection system from:
  - TCT Mentorship Lecture 1: Market Structure
  - 2025 MS 1T TCT
  - 2025 MS 1 EREVIEW (32 Pages)

Core concepts:
  - 6-Candle Rule pivot detection (inside bar exclusion)
  - MSH / MSL confirmation (revisit rule)
  - BOS detection with Good/Bad quality classification
  - Wick / SFP (Swing Failure Pattern) detection with 3 scenarios
  - CHoCH (Change of Character) via domino effect
  - Level 1 / 2 / 3 structure hierarchy
  - Domino Effect confirmation chain (L3 → L2 → L1)
  - RTZ (Reaction Trading Zone) quality scoring
  - Trend classification and trend shift detection
  - Expectational Order Flow (EOF)
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np


# ====================================================================
# CONSTANTS
# ====================================================================

# Minimum candles needed to form a valid 6-candle rule pivot
MIN_CANDLES_FOR_PIVOT = 6

# BOS quality thresholds
BOS_GOOD_DISTANCE_PCT = 0.002      # 0.2% away from broken level = "good distance"
BOS_MIN_CANDLES_ABOVE = 2          # Candles that must remain above/below for good BOS

# Wick detection
WICK_PROXIMITY_PCT = 0.001         # Within 0.1% of level counts as "touching"

# RTZ quality
RTZ_MAX_ZONES_FOR_CLEAN = 0        # 0 demand/supply zones = clean RTZ


# ====================================================================
# HELPERS
# ====================================================================

def _is_inside_bar(candle, prev_candle) -> bool:
    """Inside bar: high and low completely inside the previous bar."""
    return (float(candle["high"]) <= float(prev_candle["high"]) and
            float(candle["low"]) >= float(prev_candle["low"]))


def _is_bullish(candle) -> bool:
    """Candle closed higher than it opened."""
    if "open" in (candle.index if hasattr(candle, "index") else candle):
        return float(candle["close"]) > float(candle["open"])
    return float(candle["close"]) > (float(candle["high"]) + float(candle["low"])) / 2


def _is_bearish(candle) -> bool:
    """Candle closed lower than it opened."""
    if "open" in (candle.index if hasattr(candle, "index") else candle):
        return float(candle["close"]) < float(candle["open"])
    return float(candle["close"]) < (float(candle["high"]) + float(candle["low"])) / 2


def _candle_directions(candles: pd.DataFrame) -> List[Tuple[int, str]]:
    """
    Pre-compute candle direction for every bar.
    Returns list of (original_idx, 'bull' | 'bear' | 'inside').
    Inside bars and dojis are tagged 'inside'.
    """
    n = len(candles)
    dirs = []
    for i in range(n):
        c = candles.iloc[i]
        if i > 0 and _is_inside_bar(c, candles.iloc[i - 1]):
            dirs.append((i, "inside"))
        elif _is_bullish(c):
            dirs.append((i, "bull"))
        elif _is_bearish(c):
            dirs.append((i, "bear"))
        else:
            dirs.append((i, "inside"))  # doji
    return dirs


# ====================================================================
# 6-CANDLE RULE PIVOT DETECTION
# ====================================================================

def find_6cr_pivots(candles: pd.DataFrame) -> Dict[str, list]:
    """
    Find valid pivot highs and lows using the TCT 6-candle rule.

    Pivot HIGH: 2+ consecutive non-inside bullish candles → 2+ consecutive
                non-inside bearish candles.  Pivot price = highest high in
                the transition zone (including any embedded inside bars).

    Pivot LOW:  2+ consecutive non-inside bearish candles → 2+ consecutive
                non-inside bullish candles.  Pivot price = lowest low.

    Inside bars are skipped — they do not count toward the 2+2 requirement
    but they do NOT break the sequence.
    """
    n = len(candles)
    if n < MIN_CANDLES_FOR_PIVOT:
        return {"highs": [], "lows": []}

    directions = _candle_directions(candles)
    non_inside = [(idx, d) for idx, d in directions if d != "inside"]

    pivot_highs: List[Dict] = []
    pivot_lows: List[Dict] = []

    for k in range(len(non_inside) - 3):
        i0, d0 = non_inside[k]
        i1, d1 = non_inside[k + 1]
        i2, d2 = non_inside[k + 2]
        i3, d3 = non_inside[k + 3]

        # Pivot HIGH: 2 bull → 2 bear
        if d0 == "bull" and d1 == "bull" and d2 == "bear" and d3 == "bear":
            best_idx, best_high = i0, float(candles.iloc[i0]["high"])
            for j in range(i0, min(i3 + 1, n)):
                h = float(candles.iloc[j]["high"])
                if h > best_high:
                    best_high = h
                    best_idx = j
            if not pivot_highs or pivot_highs[-1]["idx"] != best_idx:
                pivot_highs.append({
                    "idx": int(best_idx),
                    "price": best_high,
                })

        # Pivot LOW: 2 bear → 2 bull
        if d0 == "bear" and d1 == "bear" and d2 == "bull" and d3 == "bull":
            best_idx, best_low = i0, float(candles.iloc[i0]["low"])
            for j in range(i0, min(i3 + 1, n)):
                lo = float(candles.iloc[j]["low"])
                if lo < best_low:
                    best_low = lo
                    best_idx = j
            if not pivot_lows or pivot_lows[-1]["idx"] != best_idx:
                pivot_lows.append({
                    "idx": int(best_idx),
                    "price": best_low,
                })

    return {"highs": pivot_highs, "lows": pivot_lows}


# ====================================================================
# MSH / MSL CONFIRMATION
# ====================================================================

def confirm_structure_points(
    candles: pd.DataFrame,
    pivot_highs: List[Dict],
    pivot_lows: List[Dict],
) -> Dict[str, list]:
    """
    Confirm Market Structure Highs and Lows per Lecture 1.

    MSH = highest point between two consecutive pivot lows.
          Confirmed when the second low is created (price revisits low area).
    MSL = lowest point between two consecutive pivot highs.
          Confirmed when the second high is created (price revisits high area).

    Each confirmed point records:
      - idx, price: where the extreme occurred
      - confirmed_at_idx: when confirmation happened
      - confirmed: True (all points returned are confirmed)
    """
    ms_highs: List[Dict] = []
    ms_lows: List[Dict] = []
    n = len(candles)

    # MSH: between consecutive pivot lows
    for i in range(len(pivot_lows) - 1):
        low1 = pivot_lows[i]
        low2 = pivot_lows[i + 1]
        start, end = low1["idx"], low2["idx"]
        if start >= end:
            continue
        best_high, best_idx = -float("inf"), start
        for j in range(start, min(end + 1, n)):
            h = float(candles.iloc[j]["high"])
            if h > best_high:
                best_high = h
                best_idx = j
        ms_highs.append({
            "idx": int(best_idx),
            "price": best_high,
            "confirmed_at_idx": int(end),
            "confirmed_by_low_idx": int(low2["idx"]),
            "confirmed": True,
        })

    # MSL: between consecutive pivot highs
    for i in range(len(pivot_highs) - 1):
        high1 = pivot_highs[i]
        high2 = pivot_highs[i + 1]
        start, end = high1["idx"], high2["idx"]
        if start >= end:
            continue
        best_low, best_idx = float("inf"), start
        for j in range(start, min(end + 1, n)):
            lo = float(candles.iloc[j]["low"])
            if lo < best_low:
                best_low = lo
                best_idx = j
        ms_lows.append({
            "idx": int(best_idx),
            "price": best_low,
            "confirmed_at_idx": int(end),
            "confirmed_by_high_idx": int(high1["idx"]),
            "confirmed": True,
        })

    return {"ms_highs": ms_highs, "ms_lows": ms_lows}


# ====================================================================
# BOS DETECTION — GOOD vs BAD
# ====================================================================

def _classify_bos_quality(
    candles: pd.DataFrame,
    bos_idx: int,
    broken_level: float,
    direction: str,
) -> Dict:
    """
    Classify BOS quality per PDF rules:

    GOOD BOS:
      - Close has good distance from the broken level (>= 0.2%)
      - At least 2 candles remain above/below the level after break candle
      - Break candle itself does NOT count

    BAD BOS:
      - Close barely above/below level (< 0.2% distance)
      - OR 2nd candle immediately closes back below/above level
      - Indicates low conviction; wick-like behavior
    """
    n = len(candles)
    bos_close = float(candles.iloc[bos_idx]["close"])

    # Distance from broken level
    if broken_level != 0:
        distance_pct = abs(bos_close - broken_level) / abs(broken_level)
    else:
        distance_pct = 0.0

    good_distance = distance_pct >= BOS_GOOD_DISTANCE_PCT

    # Count candles that remain above/below level after the break candle
    candles_holding = 0
    immediate_rejection = False

    for j in range(bos_idx + 1, min(bos_idx + 1 + BOS_MIN_CANDLES_ABOVE + 2, n)):
        c_close = float(candles.iloc[j]["close"])
        if direction == "bullish":
            if c_close > broken_level:
                candles_holding += 1
            else:
                if j == bos_idx + 1:
                    immediate_rejection = True
                break
        else:  # bearish
            if c_close < broken_level:
                candles_holding += 1
            else:
                if j == bos_idx + 1:
                    immediate_rejection = True
                break

    sustained = candles_holding >= BOS_MIN_CANDLES_ABOVE

    if good_distance and sustained and not immediate_rejection:
        quality = "good"
    elif immediate_rejection:
        quality = "bad"
    elif not good_distance:
        quality = "bad"
    else:
        quality = "moderate"

    return {
        "quality": quality,
        "distance_pct": round(distance_pct, 6),
        "candles_holding": candles_holding,
        "immediate_rejection": immediate_rejection,
    }


def detect_bos_events(
    candles: pd.DataFrame,
    ms_highs: List[Dict],
    ms_lows: List[Dict],
) -> List[Dict]:
    """
    Detect all Break of Structure events with quality classification.

    Bullish BOS: candle CLOSE above confirmed MSH (not wick).
    Bearish BOS: candle CLOSE below confirmed MSL (not wick).

    Each BOS event includes:
      - type: 'bullish' | 'bearish'
      - quality: 'good' | 'moderate' | 'bad'
      - bos_idx, bos_price: where and at what price
      - broken_level, broken_level_idx: which MS level was broken
      - distance_pct: how far the close was from the broken level
      - candles_holding: how many subsequent candles held above/below
      - immediate_rejection: True if next candle closed back across
    """
    bos_events: List[Dict] = []
    n = len(candles)

    for ms_h in ms_highs:
        search_start = ms_h.get("confirmed_at_idx", ms_h["idx"]) + 1
        for j in range(search_start, n):
            close_price = float(candles.iloc[j]["close"])
            if close_price > ms_h["price"]:
                quality_info = _classify_bos_quality(
                    candles, j, ms_h["price"], "bullish"
                )
                bos_events.append({
                    "type": "bullish",
                    "bos_idx": int(j),
                    "bos_price": close_price,
                    "broken_level": ms_h["price"],
                    "broken_level_idx": ms_h["idx"],
                    **quality_info,
                })
                break

    for ms_l in ms_lows:
        search_start = ms_l.get("confirmed_at_idx", ms_l["idx"]) + 1
        for j in range(search_start, n):
            close_price = float(candles.iloc[j]["close"])
            if close_price < ms_l["price"]:
                quality_info = _classify_bos_quality(
                    candles, j, ms_l["price"], "bearish"
                )
                bos_events.append({
                    "type": "bearish",
                    "bos_idx": int(j),
                    "bos_price": close_price,
                    "broken_level": ms_l["price"],
                    "broken_level_idx": ms_l["idx"],
                    **quality_info,
                })
                break

    bos_events.sort(key=lambda x: x["bos_idx"])
    return bos_events


# ====================================================================
# WICK / SFP (SWING FAILURE PATTERN) DETECTION
# ====================================================================

def detect_wicks(
    candles: pd.DataFrame,
    ms_highs: List[Dict],
    ms_lows: List[Dict],
) -> List[Dict]:
    """
    Detect Wick / SFP events per Lecture 1:

    A wick occurs when price TOUCHES (via wick) an MSH or MSL but
    FAILS TO CLOSE above/below it.

    Effect: makes the OPPOSITE level weak.
      - Wick of MSH → MSL becomes weak
      - Wick of MSL → MSH becomes weak

    Three post-wick scenarios are tracked:
      1. Full rotation: price rotates fully back to opposite level
      2. Reaction away: price bounces but doesn't reach opposite, creates HL/LH
      3. Immediate break: very next candle closes through the wick level

    Returns list of wick events with scenario classification.
    """
    wicks: List[Dict] = []
    n = len(candles)

    # Wicks of MS Highs (price wicks above MSH but closes below)
    for ms_h in ms_highs:
        search_start = ms_h.get("confirmed_at_idx", ms_h["idx"]) + 1
        for j in range(search_start, n):
            c = candles.iloc[j]
            high = float(c["high"])
            close = float(c["close"])

            if high >= ms_h["price"] and close < ms_h["price"]:
                # Wick detected — classify scenario
                wick_high = high
                scenario = _classify_wick_scenario(
                    candles, j, ms_h["price"], "high", ms_lows
                )
                wicks.append({
                    "type": "high_wick",
                    "wick_idx": int(j),
                    "wick_price": wick_high,
                    "level_price": ms_h["price"],
                    "level_idx": ms_h["idx"],
                    "weakens": "msl",  # MSL becomes weak
                    "scenario": scenario["scenario"],
                    "scenario_detail": scenario,
                })
                break

    # Wicks of MS Lows (price wicks below MSL but closes above)
    for ms_l in ms_lows:
        search_start = ms_l.get("confirmed_at_idx", ms_l["idx"]) + 1
        for j in range(search_start, n):
            c = candles.iloc[j]
            low = float(c["low"])
            close = float(c["close"])

            if low <= ms_l["price"] and close > ms_l["price"]:
                wick_low = low
                scenario = _classify_wick_scenario(
                    candles, j, ms_l["price"], "low", ms_highs
                )
                wicks.append({
                    "type": "low_wick",
                    "wick_idx": int(j),
                    "wick_price": wick_low,
                    "level_price": ms_l["price"],
                    "level_idx": ms_l["idx"],
                    "weakens": "msh",  # MSH becomes weak
                    "scenario": scenario["scenario"],
                    "scenario_detail": scenario,
                })
                break

    wicks.sort(key=lambda x: x["wick_idx"])
    return wicks


def _classify_wick_scenario(
    candles: pd.DataFrame,
    wick_idx: int,
    level_price: float,
    level_type: str,      # "high" or "low"
    opposite_levels: List[Dict],
) -> Dict:
    """
    Classify which of the three post-wick scenarios occurs:

    Scenario 1 (full_rotation):
      Price rotates fully back to the opposite MS level.
    Scenario 2 (reaction_continuation):
      Price reacts away from wick, does NOT touch opposite level,
      creates HL (if high wick) or LH (if low wick) — continuation.
    Scenario 3 (immediate_break):
      Very next candle closes through the wick level — becomes BOS.
    """
    n = len(candles)

    # Scenario 3: check if very next candle closes through
    if wick_idx + 1 < n:
        next_close = float(candles.iloc[wick_idx + 1]["close"])
        if level_type == "high" and next_close > level_price:
            return {"scenario": "immediate_break", "break_idx": wick_idx + 1}
        if level_type == "low" and next_close < level_price:
            return {"scenario": "immediate_break", "break_idx": wick_idx + 1}

    # Check for full rotation to opposite level (scenario 1)
    # or reaction away without reaching it (scenario 2)
    if opposite_levels:
        closest_opposite = None
        for opp in opposite_levels:
            if opp["idx"] < wick_idx:
                closest_opposite = opp
        if closest_opposite:
            opp_price = closest_opposite["price"]
            # Scan forward from wick to see if price reaches opposite
            look_ahead = min(wick_idx + 60, n)
            reached_opposite = False
            created_continuation = False

            for j in range(wick_idx + 1, look_ahead):
                c = candles.iloc[j]
                if level_type == "high":
                    # Wick was at a high; opposite level is a low
                    if float(c["low"]) <= opp_price:
                        reached_opposite = True
                        return {
                            "scenario": "full_rotation",
                            "rotation_idx": int(j),
                            "opposite_price": opp_price,
                        }
                else:
                    # Wick was at a low; opposite level is a high
                    if float(c["high"]) >= opp_price:
                        reached_opposite = True
                        return {
                            "scenario": "full_rotation",
                            "rotation_idx": int(j),
                            "opposite_price": opp_price,
                        }

            # Didn't reach opposite → scenario 2
            return {"scenario": "reaction_continuation"}

    return {"scenario": "undetermined"}


# ====================================================================
# TREND CLASSIFICATION
# ====================================================================

def classify_trend(ms_highs: List[Dict], ms_lows: List[Dict]) -> str:
    """
    Classify trend from confirmed MS points.

    Bullish: HH + HL (higher highs AND higher lows)
    Bearish: LH + LL (lower highs AND lower lows)
    Ranging: mixed / conflicting signals
    Neutral: insufficient data
    """
    if len(ms_highs) < 2 or len(ms_lows) < 2:
        if len(ms_highs) >= 2:
            h1, h2 = ms_highs[-2]["price"], ms_highs[-1]["price"]
            return "bullish" if h2 > h1 else "bearish"
        if len(ms_lows) >= 2:
            l1, l2 = ms_lows[-2]["price"], ms_lows[-1]["price"]
            return "bullish" if l2 > l1 else "bearish"
        return "neutral"

    h1, h2 = ms_highs[-2]["price"], ms_highs[-1]["price"]
    l1, l2 = ms_lows[-2]["price"], ms_lows[-1]["price"]

    hh = h2 > h1
    hl = l2 > l1
    lh = h2 < h1
    ll = l2 < l1

    if hh and hl:
        return "bullish"
    elif lh and ll:
        return "bearish"
    else:
        return "ranging"


# ====================================================================
# TREND SHIFT DETECTION
# ====================================================================

def detect_trend_shifts(
    candles: pd.DataFrame,
    ms_highs: List[Dict],
    ms_lows: List[Dict],
    bos_events: List[Dict],
    trend: str,
) -> List[Dict]:
    """
    Detect trend shifts per PDF rules:

    In BULLISH trend: bearish BOS below MSL → potential shift.
      Confirmed when LL forms, then LH forms after it.

    In BEARISH trend: bullish BOS above MSH → potential shift.
      Confirmed when HH forms, then HL forms after it.

    A bad BOS in the shift direction is a warning sign but not confirmed.
    """
    shifts: List[Dict] = []

    for bos in bos_events:
        is_counter_trend = False

        if trend == "bullish" and bos["type"] == "bearish":
            is_counter_trend = True
        elif trend == "bearish" and bos["type"] == "bullish":
            is_counter_trend = True

        if not is_counter_trend:
            continue

        # Check for confirming structure after the BOS
        bos_idx = bos["bos_idx"]
        confirmed = False
        confirming_points = []

        if bos["type"] == "bearish":
            # Need LL then LH to confirm bearish shift
            for ms_l in ms_lows:
                if ms_l["idx"] > bos_idx:
                    # Check if this is a lower low
                    prev_lows = [l for l in ms_lows if l["idx"] < ms_l["idx"]]
                    if prev_lows and ms_l["price"] < prev_lows[-1]["price"]:
                        confirming_points.append(("LL", ms_l))
                        break
            for ms_h in ms_highs:
                if ms_h["idx"] > bos_idx:
                    prev_highs = [h for h in ms_highs if h["idx"] < ms_h["idx"]]
                    if prev_highs and ms_h["price"] < prev_highs[-1]["price"]:
                        confirming_points.append(("LH", ms_h))
                        break

            confirmed = len(confirming_points) >= 2

        elif bos["type"] == "bullish":
            # Need HH then HL to confirm bullish shift
            for ms_h in ms_highs:
                if ms_h["idx"] > bos_idx:
                    prev_highs = [h for h in ms_highs if h["idx"] < ms_h["idx"]]
                    if prev_highs and ms_h["price"] > prev_highs[-1]["price"]:
                        confirming_points.append(("HH", ms_h))
                        break
            for ms_l in ms_lows:
                if ms_l["idx"] > bos_idx:
                    prev_lows = [l for l in ms_lows if l["idx"] < ms_l["idx"]]
                    if prev_lows and ms_l["price"] > prev_lows[-1]["price"]:
                        confirming_points.append(("HL", ms_l))
                        break

            confirmed = len(confirming_points) >= 2

        quality = bos.get("quality", "moderate")
        shifts.append({
            "bos_event": bos,
            "from_trend": trend,
            "to_trend": "bearish" if bos["type"] == "bearish" else "bullish",
            "confirmed": confirmed,
            "confirming_points": [
                {"label": label, "idx": pt["idx"], "price": pt["price"]}
                for label, pt in confirming_points
            ],
            "bos_quality": quality,
            "warning": quality == "bad",
        })

    return shifts


# ====================================================================
# EXPECTATIONAL ORDER FLOW (EOF)
# ====================================================================

def get_eof(trend: str, bos_events: List[Dict]) -> Dict:
    """
    Determine Expectational Order Flow.

    Bullish trend + bullish BOS → expect HL for HH (continuation)
    Bearish trend + bearish BOS → expect LH for LL (continuation)
    Bullish trend + bearish BOS → trend shift, expect LH for LL
    Bearish trend + bullish BOS → trend shift, expect HL for HH
    """
    last_bos = bos_events[-1] if bos_events else None

    if not last_bos:
        if trend == "bullish":
            return {"expectation": "higher_low_for_higher_high", "trend_shift": False, "bias": "bullish"}
        elif trend == "bearish":
            return {"expectation": "lower_high_for_lower_low", "trend_shift": False, "bias": "bearish"}
        return {"expectation": "undetermined", "trend_shift": False, "bias": "neutral"}

    bos_quality = last_bos.get("quality", "moderate")

    if trend == "bullish" and last_bos["type"] == "bullish":
        return {
            "expectation": "higher_low_for_higher_high",
            "trend_shift": False,
            "bias": "bullish",
            "bos_quality": bos_quality,
        }
    elif trend == "bearish" and last_bos["type"] == "bearish":
        return {
            "expectation": "lower_high_for_lower_low",
            "trend_shift": False,
            "bias": "bearish",
            "bos_quality": bos_quality,
        }
    elif trend == "bullish" and last_bos["type"] == "bearish":
        return {
            "expectation": "lower_high_for_lower_low",
            "trend_shift": True,
            "bias": "bearish",
            "bos_quality": bos_quality,
            "caution": bos_quality == "bad",
        }
    elif trend == "bearish" and last_bos["type"] == "bullish":
        return {
            "expectation": "higher_low_for_higher_high",
            "trend_shift": True,
            "bias": "bullish",
            "bos_quality": bos_quality,
            "caution": bos_quality == "bad",
        }
    return {"expectation": "undetermined", "trend_shift": False, "bias": "neutral"}


# ====================================================================
# CHoCH (CHANGE OF CHARACTER) VIA DOMINO EFFECT
# ====================================================================

def detect_choch_events(
    bos_events: List[Dict],
    levels: Dict,
) -> List[Dict]:
    """
    Detect Change of Character events via the domino effect.

    CHoCH occurs when:
      - L3 BOS breaks back to opposite direction → confirms rotation to L2
      - L2 BOS breaks back to opposite direction → confirms rotation to L1

    This is the domino effect: L3 → L2 → L1 confirmation chain.
    """
    choch_events: List[Dict] = []

    l3_bos = levels.get("level_3", {}).get("bos", [])
    l2_bos = levels.get("level_2", {}).get("bos", [])
    l2_trend = levels.get("level_2", {}).get("trend", "")
    l3_trend = levels.get("level_3", {}).get("trend", "")

    # L3 break against L2's trend → rotation to L2
    for bos in l3_bos:
        if l2_trend and bos["type"] != l2_trend:
            choch_events.append({
                "type": "l3_to_l2",
                "direction": bos["type"],
                "bos_idx": bos["bos_idx"],
                "bos_price": bos["bos_price"],
                "quality": bos.get("quality", "moderate"),
                "description": (
                    f"L3 {bos['type']} BOS against L2 {l2_trend} trend → "
                    f"confirms rotation to L2"
                ),
            })

    # L2 break against L1's trend → rotation to L1
    l1_trend = levels.get("level_1", {}).get("trend", "")
    for bos in l2_bos:
        if l1_trend and bos["type"] != l1_trend:
            choch_events.append({
                "type": "l2_to_l1",
                "direction": bos["type"],
                "bos_idx": bos["bos_idx"],
                "bos_price": bos["bos_price"],
                "quality": bos.get("quality", "moderate"),
                "description": (
                    f"L2 {bos['type']} BOS against L1 {l1_trend} trend → "
                    f"confirms rotation to L1"
                ),
            })

    choch_events.sort(key=lambda x: x["bos_idx"])
    return choch_events


# ====================================================================
# LEVEL 1 / 2 / 3 STRUCTURE HIERARCHY
# ====================================================================

def _extract_level_structure(
    candles: pd.DataFrame,
    start_idx: int,
    end_idx: int,
    parent_offset: int = 0,
) -> Dict:
    """
    Extract pivots, confirmed MS points, and BOS within a candle subset.
    Adjusts all indices back to the parent candle frame.
    """
    if end_idx - start_idx < MIN_CANDLES_FOR_PIVOT:
        return {"highs": [], "lows": [], "bos": [], "trend": "neutral"}

    subset = candles.iloc[start_idx:end_idx + 1].reset_index(drop=True)
    pivots = find_6cr_pivots(subset)
    confirmed = confirm_structure_points(subset, pivots["highs"], pivots["lows"])

    # Offset indices back to original frame
    offset = start_idx + parent_offset
    for h in confirmed["ms_highs"]:
        h["idx"] += offset
        h["confirmed_at_idx"] += offset
    for lo in confirmed["ms_lows"]:
        lo["idx"] += offset
        lo["confirmed_at_idx"] += offset

    # BOS detection on the full candle set with offset-corrected levels
    bos = detect_bos_events(candles, confirmed["ms_highs"], confirmed["ms_lows"])

    trend = classify_trend(confirmed["ms_highs"], confirmed["ms_lows"])

    return {
        "highs": confirmed["ms_highs"],
        "lows": confirmed["ms_lows"],
        "bos": bos,
        "trend": trend,
    }


def classify_levels(
    candles: pd.DataFrame,
    ms_highs: List[Dict],
    ms_lows: List[Dict],
    bos_events: List[Dict],
    trend: str,
) -> Dict:
    """
    Classify market structure into Level 1, 2, and 3.

    Level 1: Primary trend direction (most important).
      - Bullish: drawn bottom-to-top (L → H → HL → HH)
      - Bearish: drawn top-to-bottom (H → L → LH → LL)

    Level 2: Always OPPOSITE direction of Level 1.
      - Focuses on the pullback from the most recent L1 extreme.

    Level 3: Refined structure of the most recent Level 2 expansion.
      - Used via domino effect for early entry confirmation.
      - Only valid if RTZ between L2 and L3 is clean (no opposing zones).

    Levels are NOT defined by timeframe — they are defined by
    direction of structural pull.
    """
    result = {
        "level_1": {
            "trend": trend,
            "highs": ms_highs,
            "lows": ms_lows,
            "bos": bos_events,
        },
        "level_2": {"trend": "", "highs": [], "lows": [], "bos": []},
        "level_3": {"trend": "", "highs": [], "lows": [], "bos": []},
    }

    if not ms_highs and not ms_lows:
        return result

    n = len(candles)
    l2_trend = "bearish" if trend == "bullish" else "bullish" if trend == "bearish" else "neutral"
    result["level_2"]["trend"] = l2_trend

    # Level 2: counter-trend structure within the pullback from L1's last extreme
    l2_start = None
    if trend == "bullish" and ms_highs:
        l2_start = ms_highs[-1]["idx"]
    elif trend == "bearish" and ms_lows:
        l2_start = ms_lows[-1]["idx"]

    if l2_start is not None and n - 1 - l2_start > MIN_CANDLES_FOR_PIVOT:
        l2 = _extract_level_structure(candles, l2_start, n - 1)
        result["level_2"]["highs"] = l2["highs"]
        result["level_2"]["lows"] = l2["lows"]
        result["level_2"]["bos"] = l2["bos"]
        if l2["trend"] != "neutral":
            result["level_2"]["trend"] = l2["trend"]

    # Level 3: refined structure of L2's most recent expansion
    l2_highs = result["level_2"]["highs"]
    l2_lows = result["level_2"]["lows"]
    l3_trend = trend  # L3 refines L2's expansion; same dir as L1

    l3_start = None
    l3_end = None
    if l2_trend == "bearish" and l2_lows:
        last_l2_low = l2_lows[-1]
        l3_start = max(0, last_l2_low["idx"] - 20)
        l3_end = min(n - 1, last_l2_low["idx"] + 10)
    elif l2_trend == "bullish" and l2_highs:
        last_l2_high = l2_highs[-1]
        l3_start = max(0, last_l2_high["idx"] - 20)
        l3_end = min(n - 1, last_l2_high["idx"] + 10)

    if l3_start is not None and l3_end is not None and l3_end - l3_start > MIN_CANDLES_FOR_PIVOT:
        l3 = _extract_level_structure(candles, l3_start, l3_end)
        result["level_3"]["highs"] = l3["highs"]
        result["level_3"]["lows"] = l3["lows"]
        result["level_3"]["bos"] = l3["bos"]
        result["level_3"]["trend"] = l3_trend if l3["trend"] == "neutral" else l3["trend"]

    return result


# ====================================================================
# DOMINO EFFECT CONFIRMATION CHAIN
# ====================================================================

def evaluate_domino_effect(levels: Dict) -> Dict:
    """
    Evaluate the domino effect confirmation chain.

    The domino effect progression:
      1. L3 BOS back to bullish/bearish → confirms rotation to L2
      2. L2 BOS back to bullish/bearish → confirms rotation to L1
      3. Each level breaking increases confidence in HTF pivot

    Returns domino state: which levels have confirmed and the
    overall confidence in an HTF pivot forming.

    IMPORTANT: Breaking L3 and L2 does NOT guarantee L1 break —
    these are confirmations of rotations, not guaranteed outcomes.
    """
    l1_trend = levels.get("level_1", {}).get("trend", "neutral")
    l2_bos = levels.get("level_2", {}).get("bos", [])
    l3_bos = levels.get("level_3", {}).get("bos", [])

    l3_confirmed = False
    l2_confirmed = False
    target_direction = l1_trend  # We want L2/L3 to break BACK toward L1

    # L3 confirms rotation to L2 when it breaks back toward L1 direction
    for bos in l3_bos:
        if bos["type"] == target_direction:
            l3_confirmed = True
            break

    # L2 confirms rotation to L1 when it breaks back toward L1 direction
    for bos in l2_bos:
        if bos["type"] == target_direction:
            l2_confirmed = True
            break

    # Confidence scoring
    if l2_confirmed and l3_confirmed:
        confidence = "high"
        stage = "l1_rotation_expected"
    elif l3_confirmed:
        confidence = "moderate"
        stage = "l2_rotation_confirmed"
    else:
        confidence = "low"
        stage = "waiting_for_l3"

    return {
        "l3_confirmed": l3_confirmed,
        "l2_confirmed": l2_confirmed,
        "target_direction": target_direction,
        "confidence": confidence,
        "stage": stage,
        "note": (
            "Breaking L3 and L2 confirms rotations but does NOT guarantee "
            "L1 break. These are confirmations, not certainties."
        ),
    }


# ====================================================================
# RTZ (REACTION TRADING ZONE) QUALITY
# ====================================================================

def evaluate_rtz(
    levels: Dict,
    supply_zones: Optional[List[Dict]] = None,
    demand_zones: Optional[List[Dict]] = None,
) -> Dict:
    """
    Evaluate RTZ quality — the space between Level 2 high and Level 3 high.

    Good RTZ: empty space (no demand/supply zones) → Level 3 entry valid.
    Poor RTZ: demand/supply zones present → Level 3 entry too risky.

    RTZ is the area where liquidity builds up. When clean, price can
    break through to Level 1. When congested, price may bounce at
    intermediate support/resistance.
    """
    l2_highs = levels.get("level_2", {}).get("highs", [])
    l3_highs = levels.get("level_3", {}).get("highs", [])
    l2_lows = levels.get("level_2", {}).get("lows", [])
    l3_lows = levels.get("level_3", {}).get("lows", [])

    l1_trend = levels.get("level_1", {}).get("trend", "neutral")

    rtz_top = None
    rtz_bottom = None

    if l1_trend == "bullish":
        # For bullish: RTZ is between L2 low and L3 low (pullback zone)
        if l2_lows and l3_lows:
            rtz_top = max(l2_lows[-1]["price"], l3_lows[-1]["price"])
            rtz_bottom = min(l2_lows[-1]["price"], l3_lows[-1]["price"])
    elif l1_trend == "bearish":
        # For bearish: RTZ is between L2 high and L3 high (pullback zone)
        if l2_highs and l3_highs:
            rtz_top = max(l2_highs[-1]["price"], l3_highs[-1]["price"])
            rtz_bottom = min(l2_highs[-1]["price"], l3_highs[-1]["price"])

    if rtz_top is None or rtz_bottom is None:
        return {
            "valid": False,
            "quality": 0.0,
            "clean": False,
            "rtz_top": None,
            "rtz_bottom": None,
            "blocking_zones": 0,
            "can_use_l3": False,
        }

    # Count blocking zones within RTZ
    blocking = 0
    all_zones = (supply_zones or []) + (demand_zones or [])
    for zone in all_zones:
        zone_top = zone.get("top", zone.get("high", 0))
        zone_bottom = zone.get("bottom", zone.get("low", 0))
        # Zone overlaps with RTZ
        if zone_top >= rtz_bottom and zone_bottom <= rtz_top:
            blocking += 1

    clean = blocking <= RTZ_MAX_ZONES_FOR_CLEAN
    quality = 1.0 / (1.0 + blocking)

    return {
        "valid": True,
        "quality": round(quality, 3),
        "clean": clean,
        "rtz_top": rtz_top,
        "rtz_bottom": rtz_bottom,
        "blocking_zones": blocking,
        "can_use_l3": clean,
        "note": (
            "Clean RTZ (no blocking zones) → Level 3 entry is valid. "
            "Blocking zones present → too risky for Level 3 entry."
            if not clean else
            "RTZ is clean — Level 3 entry is valid."
        ),
    }


# ====================================================================
# MAIN PUBLIC CLASS
# ====================================================================

class MarketStructure:
    """
    TCT Lecture 1 – Full Market Structure Detection Engine (Rebuilt).

    Implements all concepts from the three PDF lectures:
      - 6-Candle Rule pivot detection (inside bar exclusion)
      - MSH / MSL confirmation
      - BOS with Good/Bad quality classification
      - Wick / SFP detection (3 post-wick scenarios)
      - CHoCH via domino effect
      - Level 1/2/3 hierarchy (direction-based, not timeframe-based)
      - Domino Effect confirmation chain
      - RTZ quality scoring
      - Trend classification + trend shift detection
      - Expectational Order Flow
    """

    # Expose module-level helpers as static methods for backward compat
    _is_inside_bar = staticmethod(_is_inside_bar)
    _is_bullish_candle = staticmethod(_is_bullish)
    _is_bearish_candle = staticmethod(_is_bearish)
    _find_6cr_pivots = staticmethod(find_6cr_pivots)
    _confirm_structure_points = staticmethod(confirm_structure_points)
    _detect_bos_events = staticmethod(detect_bos_events)
    _classify_trend = staticmethod(classify_trend)
    _get_eof = staticmethod(get_eof)
    _classify_levels = staticmethod(classify_levels)

    @staticmethod
    def detect_pivots(candles: pd.DataFrame) -> Dict:
        """
        Full TCT Lecture 1 market structure analysis.

        Returns a dict with both backward-compatible fields and new
        rebuilt fields including BOS quality, wicks, CHoCH, domino
        effect, trend shifts, and RTZ.
        """
        empty = {
            "highs": [], "lows": [], "trend": "neutral",
            "ms_highs": [], "ms_lows": [],
            "bos_events": [],
            "eof": {"expectation": "undetermined", "trend_shift": False, "bias": "neutral"},
            "levels": {
                "level_1": {"trend": "neutral", "highs": [], "lows": [], "bos": []},
                "level_2": {"trend": "neutral", "highs": [], "lows": [], "bos": []},
                "level_3": {"trend": "neutral", "highs": [], "lows": [], "bos": []},
            },
            "pivot_highs_6cr": [], "pivot_lows_6cr": [],
            "wicks": [],
            "choch_events": [],
            "domino_effect": {
                "l3_confirmed": False, "l2_confirmed": False,
                "confidence": "low", "stage": "waiting_for_l3",
            },
            "trend_shifts": [],
            "rtz": {"valid": False, "quality": 0.0, "can_use_l3": False},
        }

        if len(candles) < MIN_CANDLES_FOR_PIVOT:
            return empty

        # Step 1: 6-candle-rule pivots
        raw_pivots = find_6cr_pivots(candles)
        pivot_highs = raw_pivots["highs"]
        pivot_lows = raw_pivots["lows"]

        # Step 2: Confirm MS highs and lows
        confirmed = confirm_structure_points(candles, pivot_highs, pivot_lows)
        ms_highs = confirmed["ms_highs"]
        ms_lows = confirmed["ms_lows"]

        # Step 3: Trend
        trend = classify_trend(ms_highs, ms_lows)

        # Step 4: BOS with quality classification
        bos_events = detect_bos_events(candles, ms_highs, ms_lows)

        # Step 5: Wick / SFP detection
        wicks = detect_wicks(candles, ms_highs, ms_lows)

        # Step 6: EOF
        eof = get_eof(trend, bos_events)

        # Step 7: Level 1/2/3 classification
        levels = classify_levels(candles, ms_highs, ms_lows, bos_events, trend)

        # Step 8: CHoCH via domino effect
        choch_events = detect_choch_events(bos_events, levels)

        # Step 9: Domino effect evaluation
        domino = evaluate_domino_effect(levels)

        # Step 10: Trend shift detection
        trend_shifts = detect_trend_shifts(
            candles, ms_highs, ms_lows, bos_events, trend
        )

        # Step 11: RTZ quality (without external zone data; callers can
        # pass supply/demand zones separately via evaluate_rtz())
        rtz = evaluate_rtz(levels)

        return {
            # Backward-compatible fields
            "highs": pivot_highs,
            "lows": pivot_lows,
            "trend": trend,
            # Confirmed structure
            "ms_highs": ms_highs,
            "ms_lows": ms_lows,
            # BOS with quality
            "bos_events": bos_events,
            # Wick / SFP
            "wicks": wicks,
            # EOF
            "eof": eof,
            # Level hierarchy
            "levels": levels,
            # CHoCH events
            "choch_events": choch_events,
            # Domino effect
            "domino_effect": domino,
            # Trend shifts
            "trend_shifts": trend_shifts,
            # RTZ
            "rtz": rtz,
            # Raw pivots
            "pivot_highs_6cr": pivot_highs,
            "pivot_lows_6cr": pivot_lows,
        }

    @staticmethod
    def detect_bos(candles: pd.DataFrame, pivots: Dict) -> Optional[Dict]:
        """
        Backward-compatible BOS detection.
        Returns the last BOS event with quality info.
        """
        bos_events = pivots.get("bos_events", [])
        if bos_events:
            last_bos = bos_events[-1]
            return {
                "type": last_bos["type"],
                "price": last_bos["bos_price"],
                "quality": last_bos.get("quality", "moderate"),
            }

        ms_highs = pivots.get("ms_highs", pivots.get("highs", []))
        ms_lows = pivots.get("ms_lows", pivots.get("lows", []))
        if not ms_highs and not ms_lows:
            return None

        price = float(candles.iloc[-1]["close"])
        if ms_highs and price > ms_highs[-1]["price"]:
            return {"type": "bullish", "price": price, "quality": "unconfirmed"}
        if ms_lows and price < ms_lows[-1]["price"]:
            return {"type": "bearish", "price": price, "quality": "unconfirmed"}
        return None
