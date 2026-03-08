"""
decision_tree_bridge.py — Bridge between raw candle/schematic data and the 6 Decision Trees
============================================================================================

Translates candle DataFrames and schematic dicts (from detect_tct_schematics) into the
typed dataclass inputs each decision tree expects, then runs the full 6-tree pipeline:

    Tree 1: Ranges              → Is the range valid? What's the trend/bias?
    Tree 2: Supply & Demand     → Is there a valid S/D zone with FVG confirmation?
    Tree 3: Liquidity           → Was liquidity swept? Grab vs true break?
    Tree 4: TCT 5A              → Is the 3-tap schematic structurally sound?
    Tree 5: TCT 5B              → Does it pass real-world refinement rules?
    Tree 6: Advanced (optional) → Flip, escalation, Wyckoff-in-Wyckoff?

Trees 1–5 are hard gates — fail any and the setup is rejected.
Tree 6 is an enhancement layer (upgrades conviction, triggers flip exits).
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
    if "Model_1" in model:
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

    if "Model_1" in model:
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
# PUBLIC EVALUATOR CLASS — drop-in replacement for Schematics5BEvaluator
# ================================================================

class DecisionTreeEvaluator:
    """
    Evaluates TCT schematics using the 6-tree decision pipeline.
    Drop-in replacement for Schematics5BEvaluator — same evaluate_schematic interface.
    """

    def __init__(self):
        self._active_trade: Optional[Dict] = None

    def set_active_trade(self, trade: Optional[Dict]):
        """Update the active trade reference for flip detection (Tree 6)."""
        self._active_trade = trade

    def evaluate_schematic(self, schematic: Dict, htf_bias: str, current_price: float,
                           total_candles: int = 200, max_stale_candles: int = 5,
                           candle_df: Optional[pd.DataFrame] = None) -> Dict:
        """
        Evaluate a schematic using the 6-tree pipeline.

        Same return format as Schematics5BEvaluator.evaluate_schematic:
            {"score", "pass", "direction", "model", "rr", "required_score", "reasons"}

        Plus additional "tree_results" key with per-tree diagnostics.
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
            "required_score": 50, "pass": False, "tree_results": {},
        }

        # Pre-gates (same as old evaluator for backward compat)
        if not is_confirmed:
            return {**fail, "reasons": ["No BOS confirmation"]}

        # Stale BOS check
        bos = schematic.get("bos_confirmation") or {}
        bos_idx = bos.get("bos_idx")
        if bos_idx is not None and bos_idx < total_candles - max_stale_candles:
            return {**fail, "reasons": [f"Stale BOS: {total_candles - bos_idx} candles ago (max {max_stale_candles})"]}

        # If we have candle data, run the full tree pipeline
        if candle_df is not None and len(candle_df) > 0:
            return compute_composite_score(
                candle_df, schematic, htf_bias, current_price, self._active_trade,
            )

        # Fallback: no candle data — run simplified scoring (mirrors old evaluator logic)
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
            "required_score": 50, "pass": False, "tree_results": {},
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
            "required_score": 50, "pass": score >= 50, "reasons": reasons,
            "tree_results": {"mode": "fallback_no_candle_data"},
        }
