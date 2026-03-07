"""
TCT Lecture 2 — Ranges: Decision Tree (Python)

Encodes the full decision logic from TCT 2024 Mentorship Lecture 2 | Ranges.
Each function corresponds to one phase of the decision tree.

Usage:
    from decision_trees.ranges_decision_tree import evaluate_range_setup
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
    """All observable market inputs needed to evaluate a range setup."""

    # Phase 1 — Trend
    higher_highs_higher_lows: bool       # True = uptrend; False = evaluate lower field
    lower_highs_lower_lows: bool         # True = downtrend

    # Phase 2 — Range confirmation
    six_candle_rule_passes: bool         # True if 2+2+2 structure is satisfied
    equilibrium_touched: bool            # Price has returned to the 0.5 Fib
    highest_valid_timeframe: str         # e.g. "4H", "Daily", "3D"

    # Phase 3 — Rationality
    range_looks_horizontal: bool         # False = V-shaped / irrational

    # Phase 5–6 — Deviation or break?
    price_at_range_high: bool            # Price approaching or exceeding Range High
    price_at_range_low: bool             # Price approaching or exceeding Range Low
    is_wick_only: bool                   # Price wicked but candle closed inside
    close_outside_range: bool            # At least one candle close outside the range
    immediate_reversal_next_candle: bool # Next candle closed back inside the range
    close_beyond_dl: bool                # Any close beyond the 30% DL extension
    higher_tf_looks_like_wick: bool      # On the highest valid TF, looks like wick/brief excursion

    # Phase 8 — S/D confluence
    demand_zone_overlaps_dl_below: bool  # Demand zone near/overlapping the DL below Range Low
    supply_zone_overlaps_dl_above: bool  # Supply zone near/overlapping the DL above Range High

    # Phase 9 — Timeframe context
    timeframe_category: str              # "low" (5m/10m), "mid" (1h/4h), "high" (Daily/3D)


@dataclass
class RangeEvaluation:
    """Result of the full decision tree evaluation."""
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
    warnings: list = field(default_factory=list)
    passed_phases: list = field(default_factory=list)
    failed_at_phase: Optional[str] = None


# ──────────────────────────────────────────────────────────
# Phase functions
# ──────────────────────────────────────────────────────────

def phase1_identify_trend(inputs: RangeInputs, result: RangeEvaluation) -> bool:
    """Phase 1: Determine macro trend direction and Fibonacci draw direction."""
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
    """Phase 2: Validate range via six candle rule and equilibrium touch."""
    if not inputs.six_candle_rule_passes:
        result.failed_at_phase = (
            "Phase 2: Six candle rule not satisfied on this timeframe. "
            "Drop one timeframe and re-check."
        )
        return False

    if not inputs.equilibrium_touched:
        result.warnings.append(
            "Phase 2: EQ not yet touched — range is tentative. "
            "Wait for the 0.5 Fib to be reached before confirming."
        )
        result.failed_at_phase = "Phase 2: Range pending — equilibrium not yet confirmed"
        return False

    result.range_valid = True
    result.passed_phases.append(
        f"Phase 2: Range confirmed on timeframe. "
        f"Highest valid TF: {inputs.highest_valid_timeframe}"
    )
    return True


def phase3_rationality_check(inputs: RangeInputs, result: RangeEvaluation) -> bool:
    """Phase 3: Ensure the range is visually rational (horizontal, not V-shaped)."""
    if not inputs.range_looks_horizontal:
        result.range_rational = False
        result.failed_at_phase = (
            "Phase 3: Range is V-shaped / irrational on this timeframe. "
            "Drop to a lower timeframe and find a cleaner sub-range."
        )
        result.warnings.append(
            "V-shape test failed: structure looks like trending price action, "
            "not consolidation. Do not trade this range directly."
        )
        return False

    result.range_rational = True
    result.passed_phases.append("Phase 3: Range rationality confirmed — genuinely horizontal")
    return True


def phase4_map_zones(inputs: RangeInputs, result: RangeEvaluation):
    """Phase 4: Document zone mapping (informational — always passes)."""
    result.passed_phases.append(
        "Phase 4: Zones mapped — "
        "0.0=Range High (sell), 0.5=EQ, 1.0=Range Low (buy), "
        "-0.3/1.3=DL (30% extension)"
    )


def phase5_breach_direction(inputs: RangeInputs, result: RangeEvaluation) -> bool:
    """Phase 5: Determine which range extreme is being approached."""
    if inputs.price_at_range_high:
        result.trade_bias = TradeBias.SHORT
        result.passed_phases.append("Phase 5: Price at Range High — watching for bearish deviation")
        return True

    if inputs.price_at_range_low:
        result.trade_bias = TradeBias.LONG
        result.passed_phases.append("Phase 5: Price at Range Low — watching for bullish deviation")
        return True

    result.failed_at_phase = "Phase 5: Price not at a range extreme — wait"
    result.trade_bias = TradeBias.WAIT
    return False


def phase6_classify_deviation(inputs: RangeInputs, result: RangeEvaluation) -> bool:
    """Phase 6: Classify the breach as a deviation or range break."""

    # Type 1: Pure wick
    if inputs.is_wick_only:
        result.deviation_type = DeviationType.TYPE1_WICK
        result.extend_range_boundary = True
        result.passed_phases.append(
            "Phase 6: TYPE 1 WICK DEVIATION confirmed — highest confidence"
        )
        return True

    # Type 2: One bad close then immediate reversal
    if inputs.close_outside_range and inputs.immediate_reversal_next_candle:
        result.deviation_type = DeviationType.TYPE2_BAD_BOS
        result.extend_range_boundary = True
        result.passed_phases.append(
            "Phase 6: TYPE 2 BAD BOS DEVIATION confirmed — next candle reversed"
        )
        return True

    # DL exceeded → range break
    if inputs.close_beyond_dl:
        result.deviation_type = DeviationType.RANGE_BREAK
        result.failed_at_phase = (
            "Phase 6: Close beyond DL — this is a RANGE BREAK, not a deviation. "
            "Reassess macro trend."
        )
        result.trade_bias = TradeBias.WAIT
        return False

    # Multiple closes outside but within DL → check highest TF
    if inputs.close_outside_range and not inputs.close_beyond_dl:
        if inputs.higher_tf_looks_like_wick:
            result.deviation_type = DeviationType.MULTI_CLOSE_WITHIN_DL
            result.extend_range_boundary = True
            result.passed_phases.append(
                f"Phase 6: Multi-close deviation within DL — "
                f"confirmed as deviation on {inputs.highest_valid_timeframe}"
            )
            return True
        else:
            result.deviation_type = DeviationType.RANGE_BREAK
            result.failed_at_phase = (
                f"Phase 6: Closes outside range AND looks like structural break on "
                f"{inputs.highest_valid_timeframe} — treat as range break"
            )
            result.trade_bias = TradeBias.WAIT
            return False

    result.failed_at_phase = "Phase 6: No clear deviation yet — wait for range extreme to be tested"
    return False


def phase7_adjust_range(inputs: RangeInputs, result: RangeEvaluation):
    """Phase 7: Extend range boundary and recalculate DL after confirmed deviation."""
    if not result.extend_range_boundary:
        return

    if inputs.price_at_range_high:
        result.passed_phases.append(
            "Phase 7: Extend Range High to deviation high. "
            "Recalculate DL at 30% above new Range High."
        )
    elif inputs.price_at_range_low:
        result.passed_phases.append(
            "Phase 7: Extend Range Low to deviation low. "
            "Recalculate DL at 30% below new Range Low."
        )


def phase8_sd_confluence(inputs: RangeInputs, result: RangeEvaluation):
    """Phase 8: Adjust DL based on supply/demand zone confluence."""
    if inputs.price_at_range_low and inputs.demand_zone_overlaps_dl_below:
        result.dl_adjustment = (
            "Demand zone overlaps DL below Range Low. "
            "Effective DL moves to the BOTTOM of the demand zone. "
            "A tap into the demand zone remains a valid deviation."
        )
        result.passed_phases.append("Phase 8: DL adjusted — demand zone below Range Low extends the tolerance")

    elif inputs.price_at_range_high and inputs.supply_zone_overlaps_dl_above:
        result.dl_adjustment = (
            "Supply zone overlaps DL above Range High. "
            "Effective DL moves to the TOP of the supply zone. "
            "A tap into the supply zone remains a valid deviation."
        )
        result.passed_phases.append("Phase 8: DL adjusted — supply zone above Range High extends the tolerance")

    else:
        result.passed_phases.append("Phase 8: No S/D zone adjustment — use raw DL line")


def phase9_set_targets(inputs: RangeInputs, result: RangeEvaluation):
    """Phase 9: Set trade bias and targets based on deviation direction and TF."""
    tf_notes = {
        "low": "Lower TF — require quick, sharp reversal. Be strict.",
        "mid": "Mid TF — moderate tolerance. A few outside closes acceptable.",
        "high": "Higher TF — allow deeper, slower deviations. Be lenient.",
    }

    if result.trade_bias == TradeBias.SHORT:
        result.primary_target = "Range Low"
        result.minimum_target = "Lower portion of the range (below EQ)"
        result.entry_note = (
            "Distribution setup. Wait for TCT schematic (Model 1 or Model 2) "
            "to form at the Range High deviation before entering. "
            + tf_notes.get(inputs.timeframe_category, "")
        )
        result.passed_phases.append(
            "Phase 9: SHORT bias — target Range Low. Entry via TCT schematic at Range High."
        )

    elif result.trade_bias == TradeBias.LONG:
        result.primary_target = "Range High"
        result.minimum_target = "Upper portion of the range (above EQ)"
        result.entry_note = (
            "Accumulation setup. Wait for TCT schematic (Model 1 or Model 2) "
            "to form at the Range Low deviation before entering. "
            + tf_notes.get(inputs.timeframe_category, "")
        )
        result.passed_phases.append(
            "Phase 9: LONG bias — target Range High. Entry via TCT schematic at Range Low."
        )


# ──────────────────────────────────────────────────────────
# Master evaluation entry point
# ──────────────────────────────────────────────────────────

def evaluate_range_setup(inputs: RangeInputs) -> RangeEvaluation:
    """
    Run the full TCT Ranges decision tree against the provided market inputs.

    Returns a RangeEvaluation describing:
      - Trend direction and Fib draw direction
      - Whether the range is valid and rational
      - Deviation type (or range break)
      - Trade bias (long/short/wait)
      - Primary and minimum targets
      - Any DL adjustments for S/D confluence
      - Which phases passed and where (if anywhere) evaluation failed
    """
    result = RangeEvaluation()

    # Phase 1 — Trend
    if not phase1_identify_trend(inputs, result):
        return result

    # Phase 2 — Range confirmation
    if not phase2_confirm_range(inputs, result):
        return result

    # Phase 3 — Rationality check
    if not phase3_rationality_check(inputs, result):
        return result

    # Phase 4 — Zone mapping (informational, always continues)
    phase4_map_zones(inputs, result)

    # Phase 5 — Breach direction
    if not phase5_breach_direction(inputs, result):
        return result

    # Phase 6 — Classify deviation or break
    if not phase6_classify_deviation(inputs, result):
        return result

    # Phase 7 — Adjust range boundary
    phase7_adjust_range(inputs, result)

    # Phase 8 — S/D zone confluence
    phase8_sd_confluence(inputs, result)

    # Phase 9 — Set targets and entry note
    phase9_set_targets(inputs, result)

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
    # Example: bullish deviation at Range Low on a 4H range
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
    )

    result = evaluate_range_setup(example_bullish)
    print_evaluation(result)

    # Example: bearish deviation at Range High — Type 2 bad BOS
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
    )

    result2 = evaluate_range_setup(example_bearish)
    print_evaluation(result2)
