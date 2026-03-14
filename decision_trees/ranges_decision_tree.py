"""
TCT Lecture 2 — Ranges: Decision Tree (Python)

Encodes the full decision logic from TCT 2024 Mentorship Lecture 2 | Ranges.
Includes bonus internal ranges, liquidity stacking, and S/D confluence.

Usage:
    from ranges_decision_tree import evaluate_range_setup, print_evaluation
    result = evaluate_range_setup(inputs)
"""
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

# ──────────────────────────────────────────────────────────
# Enumerations
# ──────────────────────────────────────────────────────────
class Trend(Enum):
    UPTREND = auto()
    DOWNTREND = auto()
    UNCLEAR = auto()

class DeviationType(Enum):
    TYPE1_WICK = "Type 1 — Wick deviation (highest confidence)"
    TYPE2_BAD_BOS = "Type 2 — Bad BOS candle close deviation (high confidence)"
    MULTI_CLOSE_WITHIN_DL = "Multi-close within DL (moderate — requires higher TF confirmation)"
    RANGE_BREAK = "Range break — DL exceeded (do NOT trade as deviation)"
    NO_DEVIATION = "No deviation yet"

class TradeBias(Enum):
    LONG = "Long / Accumulation (buy)"
    SHORT = "Short / Distribution (sell)"
    WAIT = "Wait — no setup"

# ──────────────────────────────────────────────────────────
# Input / Output data classes
# ──────────────────────────────────────────────────────────
@dataclass
class RangeInputs:
    higher_highs_higher_lows: bool
    lower_highs_lower_lows: bool
    six_candle_rule_passes: bool
    equilibrium_touched: bool
    highest_valid_timeframe: str
    range_looks_horizontal: bool
    price_at_range_high: bool
    price_at_range_low: bool
    is_wick_only: bool
    close_outside_range: bool
    immediate_reversal_next_candle: bool
    close_beyond_dl: bool
    higher_tf_looks_like_wick: bool
    demand_zone_overlaps_dl_below: bool
    supply_zone_overlaps_dl_above: bool
    timeframe_category: str
    # Bonus internal range
    inside_macro_range_not_near_extremes: bool = False
    recent_internal_expansion_up: bool = False
    recent_internal_expansion_down: bool = False

@dataclass
class RangeEvaluation:
    trend: Optional[Trend] = None
    fib_direction: Optional[str] = None
    range_valid: bool = False
    range_rational: bool = False
    deviation_type: DeviationType = DeviationType.NO_DEVIATION
    trade_bias: TradeBias = TradeBias.WAIT
    primary_target: str = ""
    minimum_target: str = ""
    extend_range_boundary: bool = False
    dl_adjustment: str = ""
    entry_note: str = ""
    internal_range_note: str = ""
    warnings: list = field(default_factory=list)
    passed_phases: list = field(default_factory=list)
    failed_at_phase: Optional[str] = None

# ──────────────────────────────────────────────────────────
# Phase functions
# ──────────────────────────────────────────────────────────
def phase1_identify_trend(inputs: RangeInputs, result: RangeEvaluation) -> bool:
    if inputs.higher_highs_higher_lows:
        result.trend = Trend.UPTREND
        result.fib_direction = "Top → Bottom (high to low)"
        result.passed_phases.append("Phase 1: UPTREND confirmed — Fib Top → Bottom")
        return True
    if inputs.lower_highs_lower_lows:
        result.trend = Trend.DOWNTREND
        result.fib_direction = "Bottom → Top (low to high)"
        result.passed_phases.append("Phase 1: DOWNTREND confirmed — Fib Bottom → Top")
        return True
    result.trend = Trend.UNCLEAR
    result.failed_at_phase = "Phase 1: Trend unclear — step up one timeframe and re-assess"
    result.trade_bias = TradeBias.WAIT
    return False

def phase2_confirm_range(inputs: RangeInputs, result: RangeEvaluation) -> bool:
    if not inputs.six_candle_rule_passes:
        result.failed_at_phase = "Phase 2: Six candle rule not satisfied"
        return False
    if not inputs.equilibrium_touched:
        result.failed_at_phase = "Phase 2: EQ not yet touched"
        result.warnings.append("Phase 2: Range tentative until equilibrium touched")
        return False
    result.range_valid = True
    result.passed_phases.append(f"Phase 2: Range confirmed on {inputs.highest_valid_timeframe}")
    return True

def phase3_rationality_check(inputs: RangeInputs, result: RangeEvaluation) -> bool:
    if not inputs.range_looks_horizontal:
        result.range_rational = False
        result.failed_at_phase = "Phase 3: V-shaped / irrational range"
        result.warnings.append("V-shape test failed — do not trade this range directly")
        return False
    result.range_rational = True
    result.passed_phases.append("Phase 3: Range rationality confirmed")
    return True

def phase4_map_zones(inputs: RangeInputs, result: RangeEvaluation):
    result.passed_phases.append(
        "Phase 4: Zones mapped — 0.0=Range High, 0.5=EQ, 1.0=Range Low, -0.3/1.3=DL"
    )

def phase5_breach_direction(inputs: RangeInputs, result: RangeEvaluation) -> bool:
    if inputs.price_at_range_high:
        result.trade_bias = TradeBias.SHORT
        result.passed_phases.append("Phase 5: Price at Range High — bearish setup")
        return True
    if inputs.price_at_range_low:
        result.trade_bias = TradeBias.LONG
        result.passed_phases.append("Phase 5: Price at Range Low — bullish setup")
        return True
    result.failed_at_phase = "Phase 5: Price not at range extreme"
    result.trade_bias = TradeBias.WAIT
    return False

def phase6_classify_deviation(inputs: RangeInputs, result: RangeEvaluation) -> bool:
    if inputs.is_wick_only:
        result.deviation_type = DeviationType.TYPE1_WICK
        result.extend_range_boundary = True
        result.passed_phases.append("Phase 6: TYPE 1 WICK DEVIATION")
        return True
    if inputs.close_outside_range and inputs.immediate_reversal_next_candle:
        result.deviation_type = DeviationType.TYPE2_BAD_BOS
        result.extend_range_boundary = True
        result.passed_phases.append("Phase 6: TYPE 2 BAD BOS DEVIATION")
        return True
    if inputs.close_beyond_dl:
        result.deviation_type = DeviationType.RANGE_BREAK
        result.failed_at_phase = "Phase 6: Close beyond DL — RANGE BREAK"
        result.trade_bias = TradeBias.WAIT
        return False
    if inputs.close_outside_range and not inputs.close_beyond_dl:
        if inputs.higher_tf_looks_like_wick:
            result.deviation_type = DeviationType.MULTI_CLOSE_WITHIN_DL
            result.extend_range_boundary = True
            result.passed_phases.append(
                f"Phase 6: Multi-close deviation within DL on {inputs.highest_valid_timeframe}"
            )
            return True
        else:
            result.deviation_type = DeviationType.RANGE_BREAK
            result.failed_at_phase = "Phase 6: Multi-close outside range — structural break"
            result.trade_bias = TradeBias.WAIT
            return False
    result.failed_at_phase = "Phase 6: No clear deviation yet"
    return False

def phase7_adjust_range(inputs: RangeInputs, result: RangeEvaluation):
    if not result.extend_range_boundary:
        return
    if inputs.price_at_range_high:
        result.passed_phases.append("Phase 7: Extend Range High and recalc DL")
    elif inputs.price_at_range_low:
        result.passed_phases.append("Phase 7: Extend Range Low and recalc DL")

def phase8_sd_confluence(inputs: RangeInputs, result: RangeEvaluation):
    if inputs.price_at_range_low and inputs.demand_zone_overlaps_dl_below:
        result.dl_adjustment = "Demand zone below Range Low moves DL to bottom of zone"
        result.passed_phases.append("Phase 8: DL adjusted for demand zone")
    elif inputs.price_at_range_high and inputs.supply_zone_overlaps_dl_above:
        result.dl_adjustment = "Supply zone above Range High moves DL to top of zone"
        result.passed_phases.append("Phase 8: DL adjusted for supply zone")
    else:
        result.passed_phases.append("Phase 8: No DL adjustment — use raw DL")

def phase9_set_targets(inputs: RangeInputs, result: RangeEvaluation):
    tf_notes = {
        "low": "Lower TF — quick reversal, be strict",
        "mid": "Mid TF — moderate tolerance",
        "high": "High TF — deeper, slower deviations allowed",
    }
    note = tf_notes.get(inputs.timeframe_category, "")
    if result.trade_bias == TradeBias.SHORT:
        result.primary_target = "Range Low"
        result.minimum_target = "Lower portion (below EQ)"
        result.entry_note = f"Distribution setup via TCT schematic. {notePerfect — I see what you’re asking: you want your **pretty-print helper and example usage** fully integrated with the complete TCT Ranges Python code, including **all phases and the bonus internal range logic**. Here’s a reconstructed and fully integrated module you can copy and use directly:

```python
# ──────────────────────────────────────────────────────────
# TCT Lecture 2 — Ranges: Decision Tree (Python)
# Full decision tree including bonus internal range logic
# ──────────────────────────────────────────────────────────

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

# ──────────────────────────────────────────────────────────
# Enumerations
# ──────────────────────────────────────────────────────────
class Trend(Enum):
    UPTREND = auto()
    DOWNTREND = auto()
    UNCLEAR = auto()

class DeviationType(Enum):
    TYPE1_WICK = "Type 1 — Wick deviation (highest confidence)"
    TYPE2_BAD_BOS = "Type 2 — Bad BOS candle close deviation (high confidence)"
    MULTI_CLOSE_WITHIN_DL = "Multi-close within DL (moderate — requires higher TF confirmation)"
    RANGE_BREAK = "Range break — DL exceeded (do NOT trade as deviation)"
    NO_DEVIATION = "No deviation yet"

class TradeBias(Enum):
    LONG = "Long / Accumulation (buy)"
    SHORT = "Short / Distribution (sell)"
    WAIT = "Wait — no setup"

# ──────────────────────────────────────────────────────────
# Input / Output data classes
# ──────────────────────────────────────────────────────────
@dataclass
class RangeInputs:
    higher_highs_higher_lows: bool
    lower_highs_lower_lows: bool
    six_candle_rule_passes: bool
    equilibrium_touched: bool
    highest_valid_timeframe: str
    range_looks_horizontal: bool
    price_at_range_high: bool
    price_at_range_low: bool
    is_wick_only: bool
    close_outside_range: bool
    immediate_reversal_next_candle: bool
    close_beyond_dl: bool
    higher_tf_looks_like_wick: bool
    demand_zone_overlaps_dl_below: bool
    supply_zone_overlaps_dl_above: bool
    timeframe_category: str
    # Bonus internal range
    inside_macro_range_not_near_extremes: bool = False
    recent_internal_expansion_up: bool = False
    recent_internal_expansion_down: bool = False

@dataclass
class RangeEvaluation:
    trend: Optional[Trend] = None
    fib_direction: Optional[str] = None
    range_valid: bool = False
    range_rational: bool = False
    deviation_type: DeviationType = DeviationType.NO_DEVIATION
    trade_bias: TradeBias = TradeBias.WAIT
    primary_target: str = ""
    minimum_target: str = ""
    extend_range_boundary: bool = False
    dl_adjustment: str = ""
    entry_note: str = ""
    internal_range_note: str = ""
    warnings: list = field(default_factory=list)
    passed_phases: list = field(default_factory=list)
    failed_at_phase: Optional[str] = None

# ──────────────────────────────────────────────────────────
# Phase functions
# ──────────────────────────────────────────────────────────
def phase1_identify_trend(inputs: RangeInputs, result: RangeEvaluation) -> bool:
    if inputs.higher_highs_higher_lows:
        result.trend = Trend.UPTREND
        result.fib_direction = "Top → Bottom (high to low)"
        result.passed_phases.append("Phase 1: UPTREND confirmed — Fib Top → Bottom")
        return True
    if inputs.lower_highs_lower_lows:
        result.trend = Trend.DOWNTREND
        result.fib_direction = "Bottom → Top (low to high)"
        result.passed_phases.append("Phase 1: DOWNTREND confirmed — Fib Bottom → Top")
        return True
    result.trend = Trend.UNCLEAR
    result.failed_at_phase = "Phase 1: Trend unclear — step up one timeframe and re-assess"
    result.trade_bias = TradeBias.WAIT
    return False

def phase2_confirm_range(inputs: RangeInputs, result: RangeEvaluation) -> bool:
    if not inputs.six_candle_rule_passes:
        result.failed_at_phase = "Phase 2: Six candle rule not satisfied"
        return False
    if not inputs.equilibrium_touched:
        result.failed_at_phase = "Phase 2: EQ not yet touched"
        result.warnings.append("Phase 2: Range tentative until equilibrium touched")
        return False
    result.range_valid = True
    result.passed_phases.append(f"Phase 2: Range confirmed on {inputs.highest_valid_timeframe}")
    return True

def phase3_rationality_check(inputs: RangeInputs, result: RangeEvaluation) -> bool:
    if not inputs.range_looks_horizontal:
        result.range_rational = False
        result.failed_at_phase = "Phase 3: V-shaped / irrational range"
        result.warnings.append("V-shape test failed — do not trade this range directly")
        return False
    result.range_rational = True
    result.passed_phases.append("Phase 3: Range rationality confirmed")
    return True

def phase4_map_zones(inputs: RangeInputs, result: RangeEvaluation):
    result.passed_phases.append(
        "Phase 4: Zones mapped — 0.0=Range High, 0.5=EQ, 1.0=Range Low, -0.3/1.3=DL"
    )

def phase5_breach_direction(inputs: RangeInputs, result: RangeEvaluation) -> bool:
    if inputs.price_at_range_high:
        result.trade_bias = TradeBias.SHORT
        result.passed_phases.append("Phase 5: Price at Range High — bearish setup")
        return True
    if inputs.price_at_range_low:
        result.trade_bias = TradeBias.LONG
        result.passed_phases.append("Phase 5: Price at Range Low — bullish setup")
        return True
    result.failed_at_phase = "Phase 5: Price not at range extreme"
    result.trade_bias = TradeBias.WAIT
    return False

def phase6_classify_deviation(inputs: RangeInputs, result: RangeEvaluation) -> bool:
    if inputs.is_wick_only:
        result.deviation_type = DeviationType.TYPE1_WICK
        result.extend_range_boundary = True
        result.passed_phases.append("Phase 6: TYPE 1 WICK DEVIATION")
        return True
    if inputs.close_outside_range and inputs.immediate_reversal_next_candle:
        result.deviation_type = DeviationType.TYPE2_BAD_BOS
        result.extend_range_boundary = True
        result.passed_phases.append("Phase 6: TYPE 2 BAD BOS DEVIATION")
        return True
    if inputs.close_beyond_dl:
        result.deviation_type = DeviationType.RANGE_BREAK
        result.failed_at_phase = "Phase 6: Close beyond DL — RANGE BREAK"
        result.trade_bias = TradeBias.WAIT
        return False
    if inputs.close_outside_range and not inputs.close_beyond_dl:
        if inputs.higher_tf_looks_like_wick:
            result.deviation_type = DeviationType.MULTI_CLOSE_WITHIN_DL
            result.extend_range_boundary = True
            result.passed_phases.append(
                f"Phase 6: Multi-close deviation within DL on {inputs.highest_valid_timeframe}"
            )
            return True
        else:
            result.deviation_type = DeviationType.RANGE_BREAK
            result.failed_at_phase = "Phase 6: Multi-close outside range — structural break"
            result.trade_bias = TradeBias.WAIT
            return False
    result.failed_at_phase = "Phase 6: No clear deviation yet"
    return False

def phase7_adjust_range(inputs: RangeInputs, result: RangeEvaluation):
    if not result.extend_range_boundary:
        return
    if inputs.price_at_range_high:
        result.passed_phases.append("Phase 7: Extend Range High and recalc DL")
    elif inputs.price_at_range_low:
        result.passed_phases.append("Phase 7: Extend Range Low and recalc DL")

def phase8_sd_confluence(inputs: RangeInputs, result: RangeEvaluation):
    if inputs.price_at_range_low and inputs.demand_zone_overlaps_dl_below:
        result.dl_adjustment = "Demand zone below Range Low moves DL to bottom of zone"
        result.passed_phases.append("Phase 8: DL adjusted for demand zone")
    elif inputs.price_at_range_high and inputs.supply_zone_overlaps_dl_above:
        result.dl_adjustment = "Supply zone above Range High moves DL to top of zone"
        result.passed_phases.append("Phase 8: DL adjusted for supply zone")
    else:
        result.passed_phases.append("Phase 8: No DL adjustment — use raw DL")

def phase9_set_targets(inputs: RangeInputs, result: RangeEvaluation):
    tf_notes = {
        "low": "Lower TF — quick reversal, be strict",
        "mid": "Mid TF — moderate tolerance",
        "high": "High TF — deeper, slower deviations allowed",
    }
    note = tf_notes.get(inputs.timeframe_category, "")
    if result.trade_bias == TradeBias.SHORT:
        result.primary_target = "Range Low"
        result.minimum_target = "Lower portion (below EQ)"
        result.entry_note = f"Distribution setup via TCT schematic. {note}"
        result.passed_phases.append("Phase 9Perfect — I can finish integrating the **pretty-print helper** and **example usage** while keeping the full TCT/bonus range logic. Here's the completed Python module:

```python
# ──────────────────────────────────────────────────────────
# Master evaluation entry point
# ──────────────────────────────────────────────────────────
def evaluate_range_setup(inputs: RangeInputs) -> RangeEvaluation:
    result = RangeEvaluation()

    if not phase1_identify_trend(inputs, result):
        return result
    if not phase2_confirm_range(inputs, result):
        return result
    if not phase3_rationality_check(inputs, result):
        return result
    phase4_map_zones(inputs, result)
    if not phase5_breach_direction(inputs, result):
        return result
    if not phase6_classify_deviation(inputs, result):
        return result
    phase7_adjust_range(inputs, result)
    phase8_sd_confluence(inputs, result)
    phase9_set_targets(inputs, result)

    # Bonus: internal range hierarchy
    if inputs.inside_macro_range_not_near_extremes:
        if inputs.recent_internal_expansion_up:
            result.internal_range_note = "Internal uptrend sub-range: Top → Bottom"
        elif inputs.recent_internal_expansion_down:
            result.internal_range_note = "Internal downtrend sub-range: Bottom → Top"
        result.passed_phases.append("Bonus Phase: Internal sub-range determined")

    return result

# ──────────────────────────────────────────────────────────
# Pretty-print helper
# ──────────────────────────────────────────────────────────
def print_evaluation(result: RangeEvaluation):
    """Print a human-readable summary of the evaluation result."""
    print("\n" + "=" * 60)
    print("  TCT RANGES DECISION TREE — EVALUATION RESULT")
    print("=" * 60)
    print(f"  Trend:           {result.trend.name if result.trend else 'N/A'}")
    print(f"  Fib Direction:   {result.fib_direction or 'N/A'}")
    print(f"  Range Valid:     {'YES' if result.range_valid else 'NO'}")
    print(f"  Range Rational:  {'YES' if result.range_rational else 'NO'}")
    print(f"  Deviation Type:  {result.deviation_type.value}")
    print(f"  Trade Bias:      {result.trade_bias.value}")
    print(f"  Primary Target:  {result.primary_target or 'N/A'}")
    print(f"  Minimum Target:  {result.minimum_target or 'N/A'}")
    if result.dl_adjustment:
        print(f"  DL Adjustment:   {result.dl_adjustment}")
    if result.entry_note:
        print(f"  Entry Note:      {result.entry_note}")
    if result.internal_range_note:
        print(f"  Internal Range:  {result.internal_range_note}")
    print()
    print("  Phases Passed:")
    for p in result.passed_phases:
        print(f"    ✓ {p}")
    if result.failed_at_phase:
        print(f"  Failed At:  ✗ {result.failed_at_phase}")
    if result.warnings:
        print("  Warnings:")
        for w in result.warnings:
            print(f"    ⚠ {w}")
    print("=" * 60)

# ──────────────────────────────────────────────────────────
# Example usage
# ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Bullish deviation at Range Low on a 4H range
    example_bullish = RangeInputs(
        higher_highs_higher_lows=True,
        lower_highs_lower_lows=False,
        six_candle_rule_passes=True,
        equilibrium_touched=True,
        highest_valid_timeframe="4H",
        range_looks_horizontal=True,
        price_at_range_high=False,
        price_at_range_low=True,
        is_wick_only=True,
        close_outside_range=False,
        immediate_reversal_next_candle=False,
        close_beyond_dl=False,
        higher_tf_looks_like_wick=True,
        demand_zone_overlaps_dl_below=True,
        supply_zone_overlaps_dl_above=False,
        timeframe_category="mid",
        inside_macro_range_not_near_extremes=True,
        recent_internal_expansion_up=True
    )

    result = evaluate_range_setup(example_bullish)
    print_evaluation(result)

    # Bearish deviation at Range High — Type 2 bad BOS
    example_bearish = RangeInputs(
        higher_highs_higher_lows=False,
        lower_highs_lower_lows=True,
        six_candle_rule_passes=True,
        equilibrium_touched=True,
        highest_valid_timeframe="1H",
        range_looks_horizontal=True,
        price_at_range_high=True,
        price_at_range_low=False,
        is_wick_only=False,
        close_outside_range=True,
        immediate_reversal_next_candle=True,
        close_beyond_dl=False,
        higher_tf_looks_like_wick=True,
        demand_zone_overlaps_dl_below=False,
        supply_zone_overlaps_dl_above=False,
        timeframe_category="mid",
        inside_macro_range_not_near_extremes=True,
        recent_internal_expansion_down=True
    )

    result2 = evaluate_range_setup(example_bearish)
    print_evaluation(result2)