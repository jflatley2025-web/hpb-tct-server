"""
TCT Lecture 4 — Liquidity: Decision Tree (Python)

Encodes the full decision logic from TCT 2024 Mentorship Lecture 4 | Liquidity.

Core mental model: market makers engineer price to stop-loss clusters, sweep them
to fill their own opposing position, then reverse price to the next liquidity pool.

Usage:
    from decision_trees.liquidity_decision_tree import evaluate_liquidity_setup
    result = evaluate_liquidity_setup(inputs)
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional


# ──────────────────────────────────────────────────────────
# Enumerations
# ──────────────────────────────────────────────────────────

class LiquidityPoolType(Enum):
    RANGE_HIGH = "Range High — buy-side liquidity above upper range boundary"
    RANGE_LOW = "Range Low — sell-side liquidity below lower range boundary"
    MS_HIGH = "Market Structure High — buy-side (downtrend pivot highs)"
    MS_LOW = "Market Structure Low — sell-side (uptrend pivot lows)"
    TREND_LINE = "Trend Line Liquidity — stop clusters along a connected series of highs/lows"
    LIQUIDITY_CURVE = "Liquidity Curve — curved series of highs/lows (price too steep for straight line)"
    INTERNAL = "Internal Range Liquidity — higher lows / lower highs inside a range"


class SweepSide(Enum):
    BUY_SIDE = "Buy-side liquidity swept (above highs — shorts' stops)"
    SELL_SIDE = "Sell-side liquidity swept (below lows — longs' stops)"


class SweepClassification(Enum):
    LIQUIDITY_GRAB = "Liquidity grab / deviation — DL2 not closed beyond"
    BAD_BOS = "Bad BOS / grab — immediate reversal after close outside"
    TRUE_BREAK = "True range break — candle closed convincingly beyond DL2"
    UNCLASSIFIED = "Not yet determined"


class PathQuality(Enum):
    CLEAN = "Clean — only liquidity between entry and target (cascade expected)"
    OBSTRUCTED = "Obstructed — active S/D zones sit between entry and target"
    PARTIAL = "Partially obstructed — mitigated S/D zones (lower concern)"


class TradeBias(Enum):
    LONG = "Long / Buy — sell-side swept, accepted back inside, target Range High"
    SHORT = "Short / Sell — buy-side swept, accepted back inside, target Range Low"
    WAIT = "Wait — conditions not yet met"


# ──────────────────────────────────────────────────────────
# Input / Output dataclasses
# ──────────────────────────────────────────────────────────

@dataclass
class LiquidityInputs:
    """All observable conditions needed to evaluate a liquidity-based trade setup."""

    # Phase 2 — Pool identification
    pool_type: LiquidityPoolType
    sweep_side: SweepSide

    # Trend line / curve validation (Phase 2)
    is_trend_line_or_curve: bool            # True if the pool is a TL or curve
    no_sd_backing_the_level: bool           # True if no supply/demand backs the level being swept

    # Phase 3 — Repeated sweep asymmetry
    times_this_side_swept: int              # How many times this side has been grabbed without DL2 close
    other_side_untouched: bool              # True if opposite liquidity pool has not been swept

    # Phase 4 — DL2 classification
    price_exceeded_range_extreme: bool      # Price moved beyond Range High or Low
    any_candle_closed_beyond_dl2: bool      # True = true break; False = deviation
    wick_only_beyond_dl2: bool             # Wick through DL2 but no close (still valid)

    # Phase 5 — Acceptance
    accepted_back_inside_range: bool        # Price closed back inside the range after deviation

    # Phase 6 — Path quality
    path_quality: PathQuality               # CLEAN, OBSTRUCTED, or PARTIAL

    # Phase 7 — Retail trap / compounding
    retail_trapped_in_wrong_direction: bool # Retail longs trapped above Range High (or shorts below Low)

    # Phase 8 — Entry trigger
    tct_schematic_confirmed: bool           # TCT Model 1 or Model 2 confirmation received


@dataclass
class LiquidityEvaluation:
    """Result of the full liquidity decision tree."""
    pool_type: Optional[LiquidityPoolType] = None
    sweep_side: Optional[SweepSide] = None
    sweep_classification: SweepClassification = SweepClassification.UNCLASSIFIED
    accepted_back_inside: bool = False
    path_quality: Optional[PathQuality] = None
    retail_compounding: bool = False
    trade_bias: TradeBias = TradeBias.WAIT
    primary_target: str = ""
    conviction_level: str = ""
    entry_note: str = ""
    path_score: float = 0.0                     # Numeric path quality: CLEAN=1.0, PARTIAL=0.65, OBSTRUCTED=0.3
    entry_ready: bool = False                     # True only when all pre-execution phases pass
    liquidity_valid: bool = False                 # True if Phase 2 + Phase 4 + Phase 5 all passed (audit/5B gating)
    warnings: list = field(default_factory=list)
    passed_phases: list = field(default_factory=list)
    failed_at_phase: Optional[str] = None


# ──────────────────────────────────────────────────────────
# Phase functions
# ──────────────────────────────────────────────────────────

def phase2_identify_pool(inputs: LiquidityInputs, result: LiquidityEvaluation) -> bool:
    """Phase 2: Validate the liquidity pool type. Extra check for TL/curves."""
    result.pool_type = inputs.pool_type
    result.sweep_side = inputs.sweep_side

    # Trend line / curve require no S/D backing the level
    if inputs.is_trend_line_or_curve and not inputs.no_sd_backing_the_level:
        result.failed_at_phase = (
            "Phase 2: Trend line / liquidity curve INVALID — supply or demand zone exists "
            "at the level being swept. Something is holding price. Not a clean sweep setup."
        )
        return False

    note = ""
    if inputs.is_trend_line_or_curve:
        note = (
            " No S/D backing confirmed — valid TL/curve. "
            "Do NOT trade the bounce at each point. Wait for ALL levels to be swept."
        )

    result.passed_phases.append(
        f"Phase 2: Pool identified — {inputs.pool_type.value}. "
        f"{inputs.sweep_side.value}.{note}"
    )
    return True


def phase3_sweep_asymmetry(inputs: LiquidityInputs, result: LiquidityEvaluation):
    """Phase 3: Assess which side has been more swept (directional bias context)."""
    if inputs.times_this_side_swept >= 3 and inputs.other_side_untouched:
        result.passed_phases.append(
            f"Phase 3: HIGH CONVICTION directional read — this side swept "
            f"{inputs.times_this_side_swept}x while opposite side remains untouched. "
            f"Market maker is stacking position against this side."
        )
    elif inputs.times_this_side_swept >= 2:
        result.passed_phases.append(
            f"Phase 3: Repeated sweep ({inputs.times_this_side_swept}x) — buy-side "
            f"exhausting. Opposite pool is likely the next major target."
        )
    else:
        result.passed_phases.append(
            f"Phase 3: First or second sweep of this side "
            f"({inputs.times_this_side_swept}x). Directional bias developing."
        )


def phase4_classify_sweep(inputs: LiquidityInputs, result: LiquidityEvaluation) -> bool:
    """Phase 4: Classify the move as a grab (deviation) or a true break via DL2."""
    if not inputs.price_exceeded_range_extreme:
        result.failed_at_phase = (
            "Phase 4: Price has not exceeded the range extreme. "
            "No sweep in progress — wait."
        )
        return False

    if inputs.any_candle_closed_beyond_dl2:
        result.sweep_classification = SweepClassification.TRUE_BREAK
        # Hard invalidate: downstream systems must not misinterpret partial result
        result.trade_bias = TradeBias.WAIT
        result.conviction_level = "INVALID — TRUE BREAK"
        result.failed_at_phase = (
            "Phase 4: TRUE RANGE BREAK — candle closed beyond DL2. "
            "Do not apply reversal logic. Reassess macro trend direction."
        )
        return False

    # Wick through DL2 is explicitly fine
    if inputs.wick_only_beyond_dl2:
        result.warnings.append(
            "Phase 4: Wick extended beyond DL2 but no close — still a valid grab. "
            "'A wick is completely fine. You need a close above it for it to be invalidated.'"
        )

    result.sweep_classification = SweepClassification.LIQUIDITY_GRAB
    result.passed_phases.append(
        "Phase 4: LIQUIDITY GRAB confirmed — no close beyond DL2. "
        "Range is still intact. Market maker swept the stop cluster."
    )
    return True


def phase5_acceptance(inputs: LiquidityInputs, result: LiquidityEvaluation) -> bool:
    """Phase 5: Confirm price accepted back inside the range."""
    # TODO: Replace with structural confirmation from market_structure_engine
    # accepted_back_inside_range is currently binary — needs L1/L2 structure validation
    if not inputs.accepted_back_inside_range:
        result.failed_at_phase = (
            "Phase 5: No acceptance back inside range yet. "
            "Wait — acceptance is required before applying reversal logic."
        )
        return False

    result.accepted_back_inside = True

    # Assign directional bias based on which side was swept
    if inputs.sweep_side == SweepSide.BUY_SIDE:
        result.trade_bias = TradeBias.SHORT
        result.primary_target = "Range Low (sell-side liquidity)"
        result.passed_phases.append(
            "Phase 5: Accepted back inside — BUY-SIDE swept + accepted. "
            "Market maker filled SHORT positions. Bias: SHORT. Target: Range Low."
        )
    else:
        result.trade_bias = TradeBias.LONG
        result.primary_target = "Range High (buy-side liquidity)"
        result.passed_phases.append(
            "Phase 5: Accepted back inside — SELL-SIDE swept + accepted. "
            "Market maker filled LONG positions. Bias: LONG. Target: Range High."
        )

    return True


def phase6_path_quality(inputs: LiquidityInputs, result: LiquidityEvaluation):
    """Phase 6: Evaluate the path between entry and target."""
    result.path_quality = inputs.path_quality

    # Numeric path score for downstream 1D scoring integration
    _PATH_SCORE_MAP = {PathQuality.CLEAN: 1.0, PathQuality.PARTIAL: 0.65, PathQuality.OBSTRUCTED: 0.3}
    result.path_score = _PATH_SCORE_MAP.get(inputs.path_quality, 0.0)

    if inputs.path_quality == PathQuality.CLEAN:
        result.conviction_level = "HIGH — cascade / snowball effect expected"
        result.passed_phases.append(
            "Phase 6: CLEAN path — only liquidity between entry and target. "
            "Cascade effect likely: each stop cluster triggered amplifies the next. "
            "High conviction."
        )
    elif inputs.path_quality == PathQuality.PARTIAL:
        result.conviction_level = "MODERATE — mitigated S/D zones present but less obstructive"
        result.warnings.append(
            "Phase 6: Partially obstructed path — previously mitigated S/D zones between "
            "entry and target. Lower obstruction risk but monitor price action at those levels."
        )
        result.passed_phases.append("Phase 6: Partially clear path — moderate conviction.")
    else:  # OBSTRUCTED
        result.conviction_level = "LOW — active S/D zone(s) may stall or absorb the move"
        result.warnings.append(
            "Phase 6: Obstructed path — active unmitigated supply (for a short) or "
            "demand (for a long) sits between entry and target. The move may stall. "
            "Downgrade conviction."
        )
        result.passed_phases.append("Phase 6: Obstructed path — lower conviction.")


def phase7_retail_trap(inputs: LiquidityInputs, result: LiquidityEvaluation):
    """Phase 7: Assess whether retail traders are trapped and adding fuel."""
    if inputs.retail_trapped_in_wrong_direction:
        result.retail_compounding = True
        if inputs.sweep_side == SweepSide.BUY_SIDE:
            result.passed_phases.append(
                "Phase 7: Retail trap confirmed — longs entered above Range High "
                "(perceived BOS). Their future stops = extra sell orders = "
                "additional fuel for short cascade."
            )
        else:
            result.passed_phases.append(
                "Phase 7: Retail trap confirmed — shorts entered below Range Low "
                "(perceived BOS). Their future stops = extra buy orders = "
                "additional fuel for long cascade."
            )
    else:
        result.passed_phases.append(
            "Phase 7: No significant retail trap detected. "
            "Standard cascade expected (no compounding from trapped positions)."
        )


def phase8_entry(inputs: LiquidityInputs, result: LiquidityEvaluation):
    """Phase 8: Prepare entry — liquidity feeds INTO 1D, not blocked by TCT confirmation."""
    # Build conviction string — include sweep asymmetry impact (Phase 3 fix)
    conviction_parts = [result.conviction_level]
    if inputs.times_this_side_swept >= 3 and inputs.other_side_untouched:
        conviction_parts.append("sweep asymmetry confirmed")
    elif inputs.times_this_side_swept >= 2:
        conviction_parts.append(f"sweep #{inputs.times_this_side_swept} — moderate asymmetry")
    if result.retail_compounding:
        conviction_parts.append("retail trap adds fuel")

    if not inputs.tct_schematic_confirmed:
        # Keep bias from Phase 5 — do NOT reset to WAIT
        result.entry_ready = False
        result.entry_note = "Awaiting TCT confirmation"
        result.passed_phases.append(
            "Phase 8: Liquidity validated — awaiting TCT schematic for entry trigger"
        )
    else:
        result.entry_ready = True
        result.entry_note = (
            f"{result.trade_bias.value}. "
            f"Target: {result.primary_target}. "
            f"Conviction: {', '.join(conviction_parts)}."
        )
        result.passed_phases.append(
            f"Phase 8: Entry confirmed — {result.trade_bias.value}"
        )

    result.conviction_level = ", ".join(conviction_parts)


# ──────────────────────────────────────────────────────────
# Master evaluation entry point
# ──────────────────────────────────────────────────────────

def evaluate_liquidity_setup(inputs: LiquidityInputs) -> LiquidityEvaluation:
    """
    Run the full TCT Liquidity decision tree.

    Returns a LiquidityEvaluation describing:
      - Liquidity pool type and sweep side
      - Sweep classification (grab vs true break)
      - Trade bias and target
      - Path quality (clean / partial / obstructed)
      - Conviction level
      - Which phases passed and where evaluation stopped
    """
    result = LiquidityEvaluation()

    # Phase 2 — Pool identification + TL/curve validation
    if not phase2_identify_pool(inputs, result):
        return result

    # Phase 3 — Sweep asymmetry (informational, always continues)
    phase3_sweep_asymmetry(inputs, result)

    # Phase 4 — Classify sweep via DL2
    if not phase4_classify_sweep(inputs, result):
        return result

    # Phase 5 — Acceptance back inside range
    if not phase5_acceptance(inputs, result):
        return result

    # Phases 2, 4, 5 all passed — mark liquidity as structurally valid for audit/5B gating
    result.liquidity_valid = True

    # Phase 6 — Path quality
    phase6_path_quality(inputs, result)

    # Phase 7 — Retail trap
    phase7_retail_trap(inputs, result)

    # Phase 8 — TCT entry trigger
    phase8_entry(inputs, result)

    return result


# ──────────────────────────────────────────────────────────
# Pretty-print helper
# ──────────────────────────────────────────────────────────

def print_evaluation(result: LiquidityEvaluation):
    """Print a human-readable summary of the liquidity evaluation."""
    print("\n" + "=" * 64)
    print("  TCT LIQUIDITY DECISION TREE — EVALUATION RESULT")
    print("=" * 64)
    print(f"  Pool Type:       {result.pool_type.value if result.pool_type else 'N/A'}")
    print(f"  Sweep Side:      {result.sweep_side.value if result.sweep_side else 'N/A'}")
    print(f"  Classification:  {result.sweep_classification.value}")
    print(f"  Accepted Inside: {'YES' if result.accepted_back_inside else 'NO'}")
    print(f"  Path Quality:    {result.path_quality.value if result.path_quality else 'N/A'}")
    print(f"  Retail Trap:     {'YES — adds fuel' if result.retail_compounding else 'No'}")
    print(f"  Trade Bias:      {result.trade_bias.value}")
    print(f"  Primary Target:  {result.primary_target or 'N/A'}")
    print(f"  Conviction:      {result.conviction_level or 'N/A'}")
    if result.entry_note:
        print(f"  Entry Note:      {result.entry_note}")
    print()
    print("  Phases Passed:")
    for p in result.passed_phases:
        print(f"    ✓ {p}")
    if result.failed_at_phase:
        print(f"  Stopped At: ✗ {result.failed_at_phase}")
    if result.warnings:
        print("  Warnings:")
        for w in result.warnings:
            print(f"    ⚠ {w}")
    print("=" * 64)


# ──────────────────────────────────────────────────────────
# Example usage
# ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Example 1: High conviction short — Range High swept 3x, clean path, TCT confirmed
    example_short = LiquidityInputs(
        pool_type=LiquidityPoolType.RANGE_HIGH,
        sweep_side=SweepSide.BUY_SIDE,
        is_trend_line_or_curve=False,
        no_sd_backing_the_level=True,
        times_this_side_swept=3,
        other_side_untouched=True,
        price_exceeded_range_extreme=True,
        any_candle_closed_beyond_dl2=False,
        wick_only_beyond_dl2=True,
        accepted_back_inside_range=True,
        path_quality=PathQuality.CLEAN,
        retail_trapped_in_wrong_direction=True,
        tct_schematic_confirmed=True,
    )
    print_evaluation(evaluate_liquidity_setup(example_short))

    # Example 2: Long — Range Low sell-side swept, obstructed path, waiting for TCT
    example_long_wait = LiquidityInputs(
        pool_type=LiquidityPoolType.RANGE_LOW,
        sweep_side=SweepSide.SELL_SIDE,
        is_trend_line_or_curve=False,
        no_sd_backing_the_level=True,
        times_this_side_swept=1,
        other_side_untouched=False,
        price_exceeded_range_extreme=True,
        any_candle_closed_beyond_dl2=False,
        wick_only_beyond_dl2=False,
        accepted_back_inside_range=True,
        path_quality=PathQuality.OBSTRUCTED,
        retail_trapped_in_wrong_direction=False,
        tct_schematic_confirmed=False,
    )
    print_evaluation(evaluate_liquidity_setup(example_long_wait))

    # Example 3: True break — DL2 closed beyond, do not trade reversal
    example_break = LiquidityInputs(
        pool_type=LiquidityPoolType.RANGE_HIGH,
        sweep_side=SweepSide.BUY_SIDE,
        is_trend_line_or_curve=False,
        no_sd_backing_the_level=True,
        times_this_side_swept=1,
        other_side_untouched=False,
        price_exceeded_range_extreme=True,
        any_candle_closed_beyond_dl2=True,
        wick_only_beyond_dl2=False,
        accepted_back_inside_range=False,
        path_quality=PathQuality.CLEAN,
        retail_trapped_in_wrong_direction=False,
        tct_schematic_confirmed=False,
    )
    print_evaluation(evaluate_liquidity_setup(example_break))

    # Example 4: Invalid trend line — supply backs the highs
    example_tl_invalid = LiquidityInputs(
        pool_type=LiquidityPoolType.TREND_LINE,
        sweep_side=SweepSide.BUY_SIDE,
        is_trend_line_or_curve=True,
        no_sd_backing_the_level=False,      # Supply exists at highs → invalid
        times_this_side_swept=2,
        other_side_untouched=True,
        price_exceeded_range_extreme=True,
        any_candle_closed_beyond_dl2=False,
        wick_only_beyond_dl2=False,
        accepted_back_inside_range=True,
        path_quality=PathQuality.CLEAN,
        retail_trapped_in_wrong_direction=False,
        tct_schematic_confirmed=False,
    )
    print_evaluation(evaluate_liquidity_setup(example_tl_invalid))
