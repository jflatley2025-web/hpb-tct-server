"""
jack_tct_evaluator.py — Jack's TCT Mode evaluator (5-tree pipeline)
====================================================================

Evaluates confirmed TCT schematics using Jack's 5-decision-tree framework.
Operates exclusively on the 4H timeframe (per Jack's methodology).

Trees:
    1. Market Structure  — Confirmed MSH/MSL present; direction aligned.  Hard gate.
    2. Ranges            — Price in correct premium/discount zone; range horizontal.
    3. Supply & Demand   — OB + FVG near tap zones.  Hard gate if no FVG.
    4. Liquidity         — Swing-point trendline (2 confirmed swing highs or lows).
    5. TCT 5A            — BOS confirmed, model valid.  Hard gate.

Scoring:
    Tree 1: 20 pts (hard gate — if no valid structure, reject)
    Tree 2: 20 pts
    Tree 3: 20 pts (hard gate — if no FVG, reject)
    Tree 4: 15 pts
    Tree 5: 25 pts (hard gate — must be confirmed)
    Threshold: 50

Return format matches DecisionTreeEvaluator.evaluate_schematic exactly.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from market_structure import find_6cr_pivots, confirm_structure_points
from decision_tree_bridge import (
    _detect_fvg,
    _find_order_block_near_tap,
    _range_looks_horizontal,
)

logger = logging.getLogger("JackTCTEvaluator")

REQUIRED_SCORE = 50


# ================================================================
# TREE 1 — Market Structure
# ================================================================

def _eval_market_structure(df: pd.DataFrame, direction: str) -> Dict:
    """
    Check that confirmed MSH/MSL points exist and support the trade direction.

    Bullish setup: needs at least one confirmed MSL (swing low held) forming
    a higher-low structure — accumulation context.
    Bearish setup: needs at least one confirmed MSH (swing high held) forming
    a lower-high structure — distribution context.
    """
    result = {
        "passed": False,
        "reason": "No candle data",
        "ms_highs": 0,
        "ms_lows": 0,
        "trend": "unknown",
    }

    if df is None or len(df) < 30:
        result["reason"] = f"Insufficient candle data ({0 if df is None else len(df)} candles)"
        return result

    try:
        pivots = find_6cr_pivots(df)
        ph = pivots.get("highs", [])
        pl = pivots.get("lows", [])

        if len(ph) < 2 or len(pl) < 2:
            result["reason"] = f"Not enough pivots (highs={len(ph)}, lows={len(pl)})"
            result["ms_highs"] = len(ph)
            result["ms_lows"] = len(pl)
            return result

        ms = confirm_structure_points(df, ph, pl)
        ms_highs = ms.get("ms_highs", [])
        ms_lows = ms.get("ms_lows", [])

        result["ms_highs"] = len(ms_highs)
        result["ms_lows"] = len(ms_lows)

        # Determine trend from most-recent 2 confirmed highs + lows
        if len(ms_highs) >= 2 and len(ms_lows) >= 2:
            hh = ms_highs[-1]["price"] > ms_highs[-2]["price"]
            hl = ms_lows[-1]["price"] > ms_lows[-2]["price"]
            lh = ms_highs[-1]["price"] < ms_highs[-2]["price"]
            ll = ms_lows[-1]["price"] < ms_lows[-2]["price"]

            if hh and hl:
                result["trend"] = "bullish"
            elif lh and ll:
                result["trend"] = "bearish"
            else:
                result["trend"] = "transitional"
        elif len(ms_lows) >= 2:
            result["trend"] = "bullish" if ms_lows[-1]["price"] > ms_lows[-2]["price"] else "bearish"
        elif len(ms_highs) >= 2:
            result["trend"] = "bearish" if ms_highs[-1]["price"] < ms_highs[-2]["price"] else "bullish"

        # Gate: trend must support trade direction
        trend = result["trend"]
        if direction == "bullish":
            ok = trend in ("bullish", "transitional") and len(ms_lows) >= 1
        else:
            ok = trend in ("bearish", "transitional") and len(ms_highs) >= 1

        result["passed"] = ok
        result["reason"] = (
            f"Trend={trend}, MS highs={len(ms_highs)}, MS lows={len(ms_lows)}"
        )
    except Exception as e:
        logger.warning(f"[JACK-T1] Market structure error: {e}", exc_info=True)
        result["reason"] = f"Error: {e}"

    return result


# ================================================================
# TREE 2 — Ranges
# ================================================================

def _eval_ranges(df: pd.DataFrame, schematic: Dict, current_price: float) -> Dict:
    """
    Check price position relative to the schematic range (premium/discount/EQ)
    and validate that the range is horizontal.

    Bullish entry: price should be at/below equilibrium (discount zone) — tap3 at range low.
    Bearish entry: price should be at/above equilibrium (premium zone) — tap3 at range high.
    """
    result = {
        "passed": False,
        "reason": "No range data",
        "range_high": None,
        "range_low": None,
        "equilibrium": None,
        "price_zone": "unknown",
        "is_horizontal": False,
    }

    rng = schematic.get("range") or {}
    range_high = rng.get("high", 0)
    range_low = rng.get("low", 0)
    if not range_high or not range_low or range_high <= range_low:
        result["reason"] = "Invalid or missing range"
        return result

    eq = (range_high + range_low) / 2.0
    rng_size = range_high - range_low
    dl2_above = range_high + rng_size * 0.30
    dl2_below = range_low - rng_size * 0.30

    result["range_high"] = round(range_high, 2)
    result["range_low"] = round(range_low, 2)
    result["equilibrium"] = round(eq, 2)
    result["dl2_above"] = round(dl2_above, 2)
    result["dl2_below"] = round(dl2_below, 2)

    direction = schematic.get("direction", "bullish")

    # Determine price zone
    if current_price >= range_high:
        zone = "premium"
    elif current_price <= range_low:
        zone = "discount"
    elif current_price >= eq:
        zone = "upper_half"
    else:
        zone = "lower_half"
    result["price_zone"] = zone

    # Horizontal check
    if df is not None and len(df) >= 6:
        is_horiz = _range_looks_horizontal(df, range_high, range_low)
        result["is_horizontal"] = is_horiz
    else:
        is_horiz = True  # no data to disprove; assume flat per schematic
        result["is_horizontal"] = True

    # Pass logic: bullish → price in discount (at/below EQ), bearish → price in premium
    if direction == "bullish":
        in_zone = zone in ("discount", "lower_half")
        result["reason"] = f"Direction=bullish, zone={zone}, horizontal={is_horiz}"
    else:
        in_zone = zone in ("premium", "upper_half")
        result["reason"] = f"Direction=bearish, zone={zone}, horizontal={is_horiz}"

    result["passed"] = in_zone and is_horiz
    return result


# ================================================================
# TREE 3 — Supply & Demand
# ================================================================

def _eval_supply_demand(df: pd.DataFrame, schematic: Dict) -> Dict:
    """
    Detect OB + FVG near tap1, tap2, tap3 zones.
    Hard gate: at least one FVG must be present.
    """
    result = {
        "passed": False,
        "reason": "No candle data",
        "fvg_found": False,
        "ob_found": False,
        "taps_checked": 0,
        "zone_type": "unknown",
    }

    if df is None or len(df) < 10:
        result["reason"] = f"Insufficient data ({0 if df is None else len(df)} candles)"
        return result

    direction = schematic.get("direction", "bullish")
    result["zone_type"] = "demand" if direction == "bullish" else "supply"

    tap_keys = ["tap1", "tap2", "tap3"]
    fvg_found = False
    ob_found = False
    taps_checked = 0

    for key in tap_keys:
        tap = schematic.get(key) or {}
        tap_price = tap.get("price")
        tap_idx = tap.get("idx")
        if not tap_price:
            continue
        taps_checked += 1

        ob = _find_order_block_near_tap(df, tap_price, direction, lookback=15)
        if ob.get("found"):
            ob_found = True
            if ob.get("fvg_confirmed"):
                fvg_found = True

        # Also check FVG directly at/near the tap index
        if not fvg_found and tap_idx is not None:
            idx = int(tap_idx) if isinstance(tap_idx, (int, float)) else 0
            if _detect_fvg(df, idx):
                fvg_found = True

    result["taps_checked"] = taps_checked
    result["ob_found"] = ob_found
    result["fvg_found"] = fvg_found

    if not fvg_found:
        result["passed"] = False
        result["reason"] = f"No FVG found near {taps_checked} taps (hard gate)"
    else:
        result["passed"] = True
        result["reason"] = f"FVG confirmed near tap zones (OB={'yes' if ob_found else 'no'})"

    return result


# ================================================================
# TREE 4 — Liquidity (Swing-Point Trendline)
# ================================================================

def _eval_liquidity_trendline(df: pd.DataFrame, schematic: Dict) -> Dict:
    """
    Connect the last 2 confirmed swing highs (bearish) or swing lows (bullish)
    and check if price near tap1/tap2 lies within ~1% of that trendline.

    Uses find_6cr_pivots for swing-point detection (same approach as market_structure.py).
    """
    result = {
        "passed": False,
        "reason": "No candle data",
        "trendline_slope": None,
        "swing_points_used": 0,
        "price_within_pct": None,
    }

    if df is None or len(df) < 30:
        result["reason"] = f"Insufficient data ({0 if df is None else len(df)} candles)"
        return result

    direction = schematic.get("direction", "bullish")

    try:
        pivots = find_6cr_pivots(df)

        if direction == "bullish":
            swings = pivots.get("lows", [])
            result["swing_type"] = "lows"
        else:
            swings = pivots.get("highs", [])
            result["swing_type"] = "highs"

        if len(swings) < 2:
            result["reason"] = f"Not enough confirmed swing {'lows' if direction == 'bullish' else 'highs'} ({len(swings)})"
            result["swing_points_used"] = len(swings)
            return result

        # Use the two most recent swing points
        s1, s2 = swings[-2], swings[-1]
        result["swing_points_used"] = 2
        result["swing_point_1"] = {"idx": s1["idx"], "price": round(s1["price"], 2)}
        result["swing_point_2"] = {"idx": s2["idx"], "price": round(s2["price"], 2)}

        # Slope per candle
        idx_diff = s2["idx"] - s1["idx"]
        if idx_diff == 0:
            result["reason"] = "Swing points at same candle index"
            return result

        slope = (s2["price"] - s1["price"]) / idx_diff
        result["trendline_slope"] = round(slope, 4)

        # Slope direction check: bullish → rising lows (slope > 0), bearish → falling highs (slope < 0)
        slope_valid = (direction == "bullish" and slope > 0) or (direction == "bearish" and slope < 0)
        if not slope_valid:
            result["reason"] = f"Trendline slope {slope:.4f} does not match direction={direction}"
            return result

        # Check if tap1 or tap2 price lies within 1% of the projected trendline
        def projected_price_at(target_idx: int) -> float:
            return s1["price"] + slope * (target_idx - s1["idx"])

        proximity_threshold = 0.01  # 1%
        tap1 = schematic.get("tap1") or {}
        tap2 = schematic.get("tap2") or {}

        hits = []
        for tap_key, tap in [("tap1", tap1), ("tap2", tap2)]:
            tap_price = tap.get("price")
            tap_idx = tap.get("idx")
            if tap_price and tap_idx is not None:
                projected = projected_price_at(int(tap_idx))
                if projected > 0:
                    pct_diff = abs(tap_price - projected) / projected
                    hits.append((tap_key, round(pct_diff * 100, 2)))
                    if pct_diff <= proximity_threshold:
                        result["passed"] = True
                        result["price_within_pct"] = round(pct_diff * 100, 2)
                        result["reason"] = f"Price within {pct_diff*100:.2f}% of trendline at {tap_key}"
                        return result

        # Also check current price against most-recent trendline extension
        n = len(df)
        proj_current = projected_price_at(n - 1)
        current_price = float(df.iloc[-1]["close"])
        if proj_current > 0:
            pct_diff = abs(current_price - proj_current) / proj_current
            result["price_within_pct"] = round(pct_diff * 100, 2)
            if pct_diff <= proximity_threshold:
                result["passed"] = True
                result["reason"] = f"Current price within {pct_diff*100:.2f}% of trendline"
                return result

        tap_details = ", ".join(f"{k}={v:.2f}%" for k, v in hits) if hits else "no taps matched"
        result["reason"] = f"Price not within 1% of trendline ({tap_details})"

    except Exception as e:
        logger.warning(f"[JACK-T4] Trendline error: {e}", exc_info=True)
        result["reason"] = f"Error: {e}"

    return result


# ================================================================
# TREE 5 — TCT 5A
# ================================================================

def _eval_tct_5a(schematic: Dict) -> Dict:
    """
    Check the TCT 5A schematic directly.  Hard gate: BOS must be confirmed.
    """
    direction = schematic.get("direction", "unknown")
    model = schematic.get("model", schematic.get("schematic_type", "unknown"))
    is_confirmed = schematic.get("is_confirmed", False)
    bos = schematic.get("bos_confirmation") or {}
    bos_price = bos.get("bos_price")
    tf = schematic.get("timeframe", "unknown")

    tap1 = (schematic.get("tap1") or {}).get("price")
    tap2 = (schematic.get("tap2") or {}).get("price")
    tap3 = (schematic.get("tap3") or {}).get("price")

    result = {
        "passed": is_confirmed,
        "reason": "BOS confirmed" if is_confirmed else "No BOS confirmation (hard gate)",
        "direction": direction,
        "model": model,
        "bos_confirmed": is_confirmed,
        "bos_price": round(bos_price, 2) if bos_price else None,
        "timeframe": tf,
        "tap1_price": round(tap1, 2) if tap1 else None,
        "tap2_price": round(tap2, 2) if tap2 else None,
        "tap3_price": round(tap3, 2) if tap3 else None,
    }
    return result


# ================================================================
# PUBLIC EVALUATOR CLASS
# ================================================================

class JackTCTEvaluator:
    """
    Evaluates TCT schematics using Jack's 5-tree decision pipeline.
    Drop-in interface: same evaluate_schematic signature as DecisionTreeEvaluator.
    """

    def evaluate_schematic(
        self,
        schematic: Dict,
        htf_bias: str,
        current_price: float,
        total_candles: int = 200,
        max_stale_candles: int = 5,
        candle_df: Optional[pd.DataFrame] = None,
    ) -> Dict:
        direction = schematic.get("direction", "unknown")
        model = schematic.get("model", schematic.get("schematic_type", "unknown"))
        is_confirmed = schematic.get("is_confirmed", False)

        # R:R
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
            "required_score": REQUIRED_SCORE, "pass": False, "tree_results": {},
        }

        # Stale BOS check
        bos = schematic.get("bos_confirmation") or {}
        bos_idx = bos.get("bos_idx")
        if bos_idx is not None and bos_idx < total_candles - max_stale_candles:
            return {
                **fail,
                "reasons": [f"Stale BOS: {total_candles - bos_idx} candles ago (max {max_stale_candles})"],
            }

        score = 0
        reasons: List[str] = []
        tree_results: Dict = {}
        df = candle_df

        # ── Tree 1: Market Structure (hard gate, 20 pts) ──
        t1 = _eval_market_structure(df, direction)
        tree_results["market_structure"] = t1
        if not t1["passed"]:
            return {
                **fail,
                "reasons": [f"MS gate: {t1['reason']}"],
                "tree_results": tree_results,
            }
        score += 20
        reasons.append(f"Market structure valid ({t1['reason']})")

        # ── Tree 2: Ranges (20 pts) ──
        t2 = _eval_ranges(df, schematic, current_price)
        tree_results["ranges"] = t2
        if t2["passed"]:
            score += 20
            reasons.append(f"Range valid — zone={t2.get('price_zone')}, horizontal={t2.get('is_horizontal')}")
        else:
            score += 5
            reasons.append(f"Range partial: {t2['reason']}")

        # ── Tree 3: Supply & Demand (hard gate, 20 pts) ──
        t3 = _eval_supply_demand(df, schematic)
        tree_results["supply_demand"] = t3
        if not t3["fvg_found"]:
            return {
                **fail,
                "score": score,
                "reasons": [f"S/D gate: {t3['reason']}"],
                "tree_results": tree_results,
            }
        score += 20
        reasons.append(f"S/D confirmed — FVG found, OB={'yes' if t3.get('ob_found') else 'no'}")

        # ── Tree 4: Liquidity trendline (15 pts) ──
        t4 = _eval_liquidity_trendline(df, schematic)
        tree_results["liquidity"] = t4
        if t4["passed"]:
            score += 15
            reasons.append(f"Liquidity trendline confirmed ({t4['reason']})")
        else:
            reasons.append(f"Liquidity trendline not found: {t4['reason']}")

        # ── Tree 5: TCT 5A (hard gate, 25 pts) ──
        t5 = _eval_tct_5a(schematic)
        tree_results["tct_5a"] = t5
        if not t5["passed"]:
            return {
                **fail,
                "score": score,
                "reasons": [f"5A gate: {t5['reason']}"],
                "tree_results": tree_results,
            }
        score += 25
        reasons.append(f"TCT 5A confirmed — model={t5.get('model')}, BOS @ {t5.get('bos_price')}")

        # R:R bonus
        if rr >= 3.0:
            score += 5
            reasons.append(f"Excellent R:R ({rr:.1f})")
        elif rr >= 2.0:
            score += 3
            reasons.append(f"Good R:R ({rr:.1f})")
        else:
            reasons.append(f"R:R: {rr:.1f}")

        score = max(0, min(100, score))
        return {
            "score": score,
            "direction": direction,
            "model": model,
            "rr": rr,
            "required_score": REQUIRED_SCORE,
            "pass": score >= REQUIRED_SCORE,
            "reasons": reasons,
            "tree_results": tree_results,
        }
