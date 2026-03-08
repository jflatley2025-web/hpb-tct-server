"""
TCT Lecture 5B — Real Trading Examples: Decision Tree (Python)

Builds on the 5A schematic theory with practical rules learned from real chart examples:
  - Rationality rule (range must look horizontal, deviations must be wicks/bad breaks)
  - Highest valid timeframe detection
  - Tap spacing check
  - Extreme liquidity vs extreme supply/demand priority
  - R:R evaluation before entry
  - Overlapping structure / Domino effect (black → red → blue → entry)
  - Reconfirmation tool (never enter directly on BOS inside a S/D zone)
  - Range formation signal (lower-TF bullish break + immediate new low = bigger range forming)

Usage:
    from decision_trees.tct_5b_schematics_real_examples_decision_tree import (
        evaluate_5b_schematic, TCT5BInputs
    )
    result = evaluate_5b_schematic(inputs)
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional


# ──────────────────────────────────────────────────────────
# Enumerations
# ──────────────────────────────────────────────────────────

class SchematicDirection(Enum):
    ACCUMULATION = "Accumulation / Re-accumulation"
    DISTRIBUTION = "Distribution / Re-distribution"


class ModelType(Enum):
    MODEL_1 = "Model 1"
    MODEL_2 = "Model 2"
    MODEL_2_FAILED_TO_M1 = "Model 2 failed → becoming Model 1"


class EntryTimeframe(Enum):
    BLACK_PRIMARY   = "Black (primary / highest-TF) BOS"
    RED_MID         = "Red (mid-TF) BOS — overlapping structure refinement"
    BLUE_LOW        = "Blue (low-TF) BOS — overlapping structure refinement"
    RECONFIRM_BLUE  = "Blue BOS after reconfirmation tool (retest + higher low / lower high)"
    NONE            = "No valid entry identified"


class SchematicStatus(Enum):
    VALID_ENTRY         = "Valid entry — all conditions met"
    WAIT_FOR_BOS        = "Wait — conditions met but BOS not yet confirmed"
    INVALID_RATIONALITY = "Invalid — range does not look like horizontal price action"
    INVALID_TAP_SPACING = "Invalid — tap spacing too compressed (Tap 3 not visible on schematic TF)"
    INVALID_NO_M2_TAP3  = "Invalid — Model 2 Tap 3 does not meet requirements"
    INVALID_RECONFIRM   = "Invalid — BOS in S/D zone, reconfirmation failed (blue broke wrong way)"
    WATCH_FOR_M1        = "Model 2 failed — watching for Model 1 to form"
    RANGE_FORMATION     = "Range formation signal — bigger range likely forming on higher TF"
    SKIP_BAD_RR         = "Skip — primary BOS R:R is unacceptable and no lower-TF refinement available"


class TradeBias(Enum):
    LONG  = "Long — Accumulation"
    SHORT = "Short — Distribution"
    WAIT  = "Wait / no trade"


# ──────────────────────────────────────────────────────────
# Dataclasses
# ──────────────────────────────────────────────────────────

@dataclass
class TCT5BInputs:
    """
    All conditions needed to evaluate a TCT schematic using the 5B real-example ruleset.
    Assumes range is already confirmed and direction (Acc/Dist) is already identified.
    """

    # ── 5A prerequisites (assumed already evaluated; pass outcomes here) ──
    direction: SchematicDirection
    model_type: ModelType
    tap2_valid: bool             # Tap 2 passed 5A DL2 + acceptance checks
    tap3_valid: bool             # Tap 3 passed 5A Model 1 or Model 2 checks

    # ── 5B Step 0: Rationality ──
    range_looks_horizontal: bool         # Range is sideways consolidation (not impulsive)
    deviations_are_wicks_or_bad_breaks: bool  # Taps are wicks or "bad breaks" (not massive moves)

    # ── 5B Step 1: Highest TF ──
    highest_valid_tf: str                # e.g. "15min", "45min", "2H", "4H"

    # ── 5B Step 2: Tap spacing ──
    tap23_gap_reasonable: bool           # T2-T3 gap is proportional to T1-T2; Tap 3 visible on schematic TF

    # ── 5B Step 3: Model 2 extreme liq vs S/D priority ──
    # (only evaluated if model_type == MODEL_2)
    extreme_liq_obvious: bool            # Obvious liquidity trail / first MSH/MSL clearly identifiable
    extreme_sd_obvious: bool             # Demand/supply zone obviously clean and proportional
    sd_tf_proportional_to_range: bool    # Zone TF matches schematic TF size
    trendline_liquidity_present: bool    # A trendline liquidity trail exists (diagonal sweep of highs/lows)

    # ── 5B Step 4: R:R assessment ──
    primary_bos_rr: float                # Estimated R:R if entering on main (black) BOS
    lower_tf_bos_available: bool         # Valid overlapping structure found in last expansion leg

    # ── 5B Step 5: Overlapping structure ──
    # (only used when primary_bos_rr is insufficient)
    lower_tf_bos_entry: EntryTimeframe   # Which TF the lower entry BOS is on
    lower_tf_bos_inside_range: bool      # Lower-TF BOS is inside original range values (critical for M1)

    # ── 5B Step 6: Reconfirmation tool ──
    bos_inside_sd_zone: bool             # BOS candle is inside a supply or demand zone
    retest_occurred: bool                # Price pulled back after BOS (for reconfirmation)
    retest_blue_confirmed: bool          # Blue structure confirmed higher low / lower high on retest

    # ── 5B BOS status ──
    bos_confirmed: bool                  # Main or lower-TF BOS has occurred in schematic direction

    # ── 5B Range formation signal ──
    ltf_bullish_break_then_new_low: bool  # Lower-TF bullish BOS immediately followed by new lower low


from typing import Optional, List

@dataclass
class TCT5BEvaluation:
    """Result of the 5B decision tree evaluation."""
    direction: Optional[SchematicDirection] = None
    model_type: Optional[ModelType] = None
    status: SchematicStatus = SchematicStatus.WAIT_FOR_BOS
    trade_bias: TradeBias = TradeBias.WAIT
    entry_timeframe: EntryTimeframe = EntryTimeframe.NONE
    highest_valid_tf: str = ""
    primary_bos_rr: float = 0.0
    stop_loss_note: str = ""
    primary_target: str = ""
    passed_phases: List[str] = field(default_factory=list)
    failed_at_phase: Optional[str] = None
    warnings: List[str] = field(default_factory=list)

# ──────────────────────────────────────────────────────────
# Phase functions
# ──────────────────────────────────────────────────────────

def phase_rationality(inputs: TCT5BInputs, result: TCT5BEvaluation) -> bool:
    """Step 0: Rationality check — does the range look like horizontal price action?"""
    if not inputs.range_looks_horizontal:
        result.status = SchematicStatus.INVALID_RATIONALITY
        result.failed_at_phase = (
            "Rationality: Range does not look like horizontal price action. "
            "Deviations on impulsive/trending moves are not valid TCT ranges. "
            "Wait for genuine sideways consolidation."
        )
        return False
    if not inputs.deviations_are_wicks_or_bad_breaks:
        result.warnings.append(
            "Rationality: Deviations are not clean wicks or bad breaks — they appear impulsive. "
            "Be cautious. Prefer wicks or slight closes beyond the extreme (quickly reversed) as deviations."
        )
    result.passed_phases.append(
        "Rationality: Range looks horizontal. Deviations are wicks or bad breaks. "
        "Valid TCT schematic shape."
    )
    return True


def phase_highest_tf(inputs: TCT5BInputs, result: TCT5BEvaluation):
    """Step 1: Record the highest valid TF for this schematic."""
    result.highest_valid_tf = inputs.highest_valid_tf
    result.passed_phases.append(
        f"Highest valid TF: {inputs.highest_valid_tf} TCT schematic. "
        f"Use proportional demand/supply zone TFs for Model 2 validation."
    )


def phase_tap_spacing(inputs: TCT5BInputs, result: TCT5BEvaluation) -> bool:
    """Step 2: Tap spacing check."""
    if not inputs.tap23_gap_reasonable:
        result.status = SchematicStatus.INVALID_TAP_SPACING
        result.failed_at_phase = (
            "Tap spacing: Tap 2–3 gap is extremely compressed vs Tap 1–2. "
            "Tap 3 does not register as a visible swing on the schematic TF. "
            "Not a valid Tap 3. Continue watching — schematic still forming."
        )
        return False
    result.passed_phases.append(
        "Tap spacing: Tap 2–3 gap is reasonably proportional. "
        "Tap 3 is a visible swing on the schematic TF."
    )
    return True


def phase_model2_sd_priority(inputs: TCT5BInputs, result: TCT5BEvaluation):
    """Step 3 (Model 2 only): Determine whether extreme liq or extreme S/D is the dominant validator."""
    notes = []

    if inputs.trendline_liquidity_present:
        notes.append(
            "Trendline liquidity trail identified — this is the most obvious extreme liquidity form. "
            "Prioritize the trendline liq sweep even if no supply/demand zone is clearly present."
        )

    if inputs.extreme_liq_obvious and not inputs.extreme_sd_obvious:
        notes.append(
            "Extreme liquidity is the dominant validator. "
            "Focus on it; pay less attention to S/D zone alignment."
        )
    elif inputs.extreme_sd_obvious and not inputs.extreme_liq_obvious:
        notes.append(
            "Extreme demand/supply is the dominant validator. "
            "Focus on it; pay less attention to liquidity alignment."
        )
    elif inputs.extreme_liq_obvious and inputs.extreme_sd_obvious:
        notes.append(
            "Both extreme liquidity AND extreme demand/supply are obvious — "
            "maximum confluence. High confidence in Model 2 Tap 3."
        )
    else:
        result.warnings.append(
            "Model 2 S/D priority: Neither extreme liquidity nor extreme S/D is clearly obvious. "
            "Re-evaluate whether this is a valid Model 2 Tap 3."
        )

    if not inputs.sd_tf_proportional_to_range:
        result.warnings.append(
            "Model 2: Demand/supply zone timeframe is not proportional to the schematic range size. "
            f"For a {result.highest_valid_tf} schematic, use proportionally sized zones "
            "(e.g., 4H range → 3H OBs; 15min range → 5min OBs; 45min range → 10min OBs)."
        )

    result.passed_phases.append("Model 2 S/D priority: " + " | ".join(notes) if notes else "Model 2 S/D priority: assessed.")


def phase_rr_check(inputs: TCT5BInputs, result: TCT5BEvaluation) -> bool:
    """Step 4: Assess R:R and determine whether to use primary BOS or overlapping structure."""
    result.primary_bos_rr = inputs.primary_bos_rr
    MIN_ACCEPTABLE_RR = 1.5

    if inputs.primary_bos_rr >= MIN_ACCEPTABLE_RR:
        result.passed_phases.append(
            f"R:R check: Primary BOS R:R = {inputs.primary_bos_rr:.2f} — acceptable. "
            "Using primary BOS entry."
        )
        result.entry_timeframe = EntryTimeframe.BLACK_PRIMARY
        return True

    # Poor R:R — need lower TF
    result.warnings.append(
        f"R:R check: Primary BOS R:R = {inputs.primary_bos_rr:.2f} — too low. "
        "Checking for overlapping structure / domino effect entry."
    )

    if not inputs.lower_tf_bos_available:
        result.status = SchematicStatus.SKIP_BAD_RR
        result.failed_at_phase = (
            f"R:R check: Primary BOS R:R = {inputs.primary_bos_rr:.2f} and no valid lower-TF "
            "overlapping structure was found. Cannot improve entry. Skip or wait."
        )
        return False

    result.passed_phases.append(
        f"R:R check: Primary BOS R:R = {inputs.primary_bos_rr:.2f} — insufficient. "
        "Lower-TF overlapping structure available. Proceeding to domino effect refinement."
    )
    result.entry_timeframe = inputs.lower_tf_bos_entry
    return True


def phase_overlapping_structure(inputs: TCT5BInputs, result: TCT5BEvaluation) -> bool:
    """Step 5: Validate the lower-TF overlapping structure entry."""
    if inputs.primary_bos_rr >= 1.5:
        # No refinement needed — already using primary BOS
        return True

    if not inputs.lower_tf_bos_available:
        return False  # Already handled in phase_rr_check

    # For Model 1: lower-TF BOS must be inside original range values
    if inputs.model_type == ModelType.MODEL_1:
        if not inputs.lower_tf_bos_inside_range:
            result.warnings.append(
                "Overlapping structure (M1): Lower-TF BOS is outside original range values. "
                "This is riskier. Try stepping down to a lower TF to find a BOS inside the range, "
                "or accept the primary BOS R:R."
            )
        else:
            result.passed_phases.append(
                f"Overlapping structure (M1): Lower-TF BOS ({result.entry_timeframe.value}) "
                "is inside original range values — valid, lower-TF entry confirmed."
            )

    # For Model 2: BOS should look reasonable (no supply/demand obstruction check here — that's reconfirmation)
    if inputs.model_type in (ModelType.MODEL_2, ModelType.MODEL_2_FAILED_TO_M1):
        result.passed_phases.append(
            f"Overlapping structure (M2): Using {result.entry_timeframe.value} "
            "for improved R:R entry."
        )

    return True


def phase_reconfirmation(inputs: TCT5BInputs, result: TCT5BEvaluation) -> bool:
    """Step 6: Reconfirmation tool — check if BOS candle is inside a S/D zone."""
    if not inputs.bos_inside_sd_zone:
        result.passed_phases.append(
            "Reconfirmation: BOS candle is NOT inside a S/D zone — enter directly on BOS."
        )
        return True

    # BOS is inside a S/D zone
    result.warnings.append(
        "Reconfirmation: BOS candle is inside a supply/demand zone. "
        "Do NOT enter directly on this BOS candle. Apply reconfirmation tool."
    )

    if not inputs.retest_occurred:
        result.status = SchematicStatus.WAIT_FOR_BOS
        result.failed_at_phase = (
            "Reconfirmation: BOS in S/D zone but price has not yet retested the BOS level. "
            "Wait for the retest."
        )
        return False

    if not inputs.retest_blue_confirmed:
        result.status = SchematicStatus.INVALID_RECONFIRM
        result.failed_at_phase = (
            "Reconfirmation: Retest occurred but blue (lowest-TF) structure did NOT confirm "
            "a higher low (Acc) or lower high (Dist). The BOS was likely false. "
            "Do NOT enter. Watch for Model 1 formation if price takes out Tap 2 extreme."
        )
        return False

    result.entry_timeframe = EntryTimeframe.RECONFIRM_BLUE
    result.passed_phases.append(
        "Reconfirmation: Retest occurred AND blue structure confirmed higher low / lower high. "
        "Valid reconfirmation entry on blue BOS."
    )
    return True


def phase_range_formation_signal(inputs: TCT5BInputs, result: TCT5BEvaluation) -> bool:
    """Bonus: Detect range formation signal — lower-TF bullish BOS then immediate new lower low."""
    if inputs.ltf_bullish_break_then_new_low:
        result.status = SchematicStatus.RANGE_FORMATION
        result.failed_at_phase = (
            "Range formation signal: Lower-TF bullish BOS followed immediately by a new lower low. "
            "This is a typical sign of a bigger range forming. "
            "Zoom out to a higher TF — this is likely one downward impulse on the higher TF. "
            "Apply range rules (new Fib, EQ touch) on the higher TF."
        )
        return False
    return True


def phase_final_entry(inputs: TCT5BInputs, result: TCT5BEvaluation):
    """Set final trade bias, SL, and target."""
    if inputs.direction == SchematicDirection.ACCUMULATION:
        result.trade_bias = TradeBias.LONG
        result.stop_loss_note = "Below Tap 3 low"
        result.primary_target = "Wyckoff High (= Range High unless re-accumulation context)"
    else:
        result.trade_bias = TradeBias.SHORT
        result.stop_loss_note = "Above Tap 3 high"
        result.primary_target = "Wyckoff Low (= Range Low unless re-distribution context)"

    result.status = SchematicStatus.VALID_ENTRY
    result.passed_phases.append(
        f"Entry: {result.trade_bias.value} on {result.entry_timeframe.value}. "
        f"SL: {result.stop_loss_note}. Target: {result.primary_target}."
    )


# ──────────────────────────────────────────────────────────
# Master evaluation entry point
# ──────────────────────────────────────────────────────────

def evaluate_5b_schematic(inputs: TCT5BInputs) -> TCT5BEvaluation:
    """
    Run the full TCT Schematic 5B decision tree.

    Assumes 5A prerequisites (range confirmed, Tap 2 and Tap 3 validated by 5A rules)
    have already been evaluated and are passed in via inputs.tap2_valid / inputs.tap3_valid.

    Returns a TCT5BEvaluation describing:
      - Status (VALID_ENTRY / WAIT / INVALID / RANGE_FORMATION / etc.)
      - Trade bias (long / short / wait)
      - Entry timeframe (black primary / red / blue / reconfirmation)
      - Stop loss and target
    """
    result = TCT5BEvaluation(
        direction=inputs.direction,
        model_type=inputs.model_type,
    )

    # Gate: 5A prerequisites
    if not inputs.tap2_valid or not inputs.tap3_valid:
        result.status = SchematicStatus.WAIT_FOR_BOS
        result.failed_at_phase = (
            "5A prerequisite: Tap 2 or Tap 3 not yet validated by 5A rules. "
            "Run the 5A decision tree first."
        )
        return result

    # Gate: range formation signal (check before anything else)
    if not phase_range_formation_signal(inputs, result):
        return result

    # Step 0: Rationality
    if not phase_rationality(inputs, result):
        return result

    # Step 1: Highest TF
    phase_highest_tf(inputs, result)

    # Step 2: Tap spacing
    if not phase_tap_spacing(inputs, result):
        return result

    # Step 3: Model 2 S/D priority (only for M2)
    if inputs.model_type in (ModelType.MODEL_2, ModelType.MODEL_2_FAILED_TO_M1):
        phase_model2_sd_priority(inputs, result)

    # Step 4: R:R check
    if not phase_rr_check(inputs, result):
        return result

    # Step 5: Overlapping structure validation
    if not phase_overlapping_structure(inputs, result):
        return result

    # Step 6: Reconfirmation tool
    if not phase_reconfirmation(inputs, result):
        return result

    # Check BOS
    if not inputs.bos_confirmed:
        result.status = SchematicStatus.WAIT_FOR_BOS
        result.failed_at_phase = (
            "BOS not yet confirmed. All structural prerequisites met. "
            "Wait for the break of structure in the schematic direction."
        )
        return result

    # Final entry
    phase_final_entry(inputs, result)
    return result


# ──────────────────────────────────────────────────────────
# Pretty-print helper
# ──────────────────────────────────────────────────────────

def print_5b_evaluation(result: TCT5BEvaluation, label: str = ""):
    """Print a human-readable 5B evaluation summary."""
    print("\n" + "=" * 66)
    if label:
        print(f"  {label}")
    print("  TCT SCHEMATIC (5B) DECISION TREE — EVALUATION RESULT")
    print("=" * 66)
    print(f"  Direction:       {result.direction.value if result.direction else 'N/A'}")
    print(f"  Model Type:      {result.model_type.value if result.model_type else 'N/A'}")
    print(f"  Highest TF:      {result.highest_valid_tf or 'N/A'}")
    print(f"  Status:          {result.status.value}")
    print(f"  Trade Bias:      {result.trade_bias.value}")
    print(f"  Entry TF:        {result.entry_timeframe.value}")
    print(f"  Primary R:R:     {result.primary_bos_rr:.2f}")
    print(f"  Stop Loss:       {result.stop_loss_note or 'N/A'}")
    print(f"  Primary Target:  {result.primary_target or 'N/A'}")
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
    print("=" * 66)


# ──────────────────────────────────────────────────────────
# Example usage
# ──────────────────────────────────────────────────────────

if __name__ == "__main__":

    # Example 1: SOLANA 4H M2 Accumulation — primary BOS 0.4R, blue BOS 6R
    sol_4h_m2_acc = TCT5BInputs(
        direction=SchematicDirection.ACCUMULATION,
        model_type=ModelType.MODEL_2,
        tap2_valid=True,
        tap3_valid=True,
        range_looks_horizontal=True,
        deviations_are_wicks_or_bad_breaks=True,
        highest_valid_tf="4H",
        tap23_gap_reasonable=True,
        extreme_liq_obvious=True,
        extreme_sd_obvious=True,       # 3H OB very obvious
        sd_tf_proportional_to_range=True,
        trendline_liquidity_present=False,
        primary_bos_rr=0.4,
        lower_tf_bos_available=True,
        lower_tf_bos_entry=EntryTimeframe.BLUE_LOW,
        lower_tf_bos_inside_range=True,
        bos_inside_sd_zone=False,
        retest_occurred=False,
        retest_blue_confirmed=False,
        bos_confirmed=True,
        ltf_bullish_break_then_new_low=False,
    )
    print_5b_evaluation(evaluate_5b_schematic(sol_4h_m2_acc), "SOLANA 4H M2 Accumulation")

    # Example 2: PEPE 2H M2 Distribution — 1min MSH extreme liq only, 2min BOS, 2.19R
    pepe_2h_m2_dist = TCT5BInputs(
        direction=SchematicDirection.DISTRIBUTION,
        model_type=ModelType.MODEL_2,
        tap2_valid=True,
        tap3_valid=True,
        range_looks_horizontal=True,
        deviations_are_wicks_or_bad_breaks=True,
        highest_valid_tf="2H",
        tap23_gap_reasonable=True,
        extreme_liq_obvious=True,      # 1min MSH was the only valid structure
        extreme_sd_obvious=False,      # OBs already mitigated
        sd_tf_proportional_to_range=True,
        trendline_liquidity_present=False,
        primary_bos_rr=0.7,            # BOS near EQ — poor
        lower_tf_bos_available=True,
        lower_tf_bos_entry=EntryTimeframe.BLUE_LOW,
        lower_tf_bos_inside_range=True,
        bos_inside_sd_zone=False,
        retest_occurred=False,
        retest_blue_confirmed=False,
        bos_confirmed=True,
        ltf_bullish_break_then_new_low=False,
    )
    print_5b_evaluation(evaluate_5b_schematic(pepe_2h_m2_dist), "PEPE 2H M2 Distribution")

    # Example 3: 45min M2→M1 Accumulation — BOS in supply zone, reconfirmation failed, watch M1
    m2_fail_reconfirm = TCT5BInputs(
        direction=SchematicDirection.ACCUMULATION,
        model_type=ModelType.MODEL_2_FAILED_TO_M1,
        tap2_valid=True,
        tap3_valid=True,
        range_looks_horizontal=True,
        deviations_are_wicks_or_bad_breaks=True,
        highest_valid_tf="45min",
        tap23_gap_reasonable=True,
        extreme_liq_obvious=True,
        extreme_sd_obvious=True,
        sd_tf_proportional_to_range=True,
        trendline_liquidity_present=False,
        primary_bos_rr=1.2,
        lower_tf_bos_available=True,
        lower_tf_bos_entry=EntryTimeframe.BLUE_LOW,
        lower_tf_bos_inside_range=True,
        bos_inside_sd_zone=True,       # BOS was in a supply zone
        retest_occurred=True,          # Retest happened
        retest_blue_confirmed=False,   # Blue never confirmed — MODEL 2 FAILED
        bos_confirmed=True,
        ltf_bullish_break_then_new_low=False,
    )
    print_5b_evaluation(evaluate_5b_schematic(m2_fail_reconfirm), "45min M2→M1 Reconfirmation Failed")

    # Example 4: SOLANA 30min M2 Distribution — trendline liquidity dominant, decent BOS
    sol_30m_m2_dist = TCT5BInputs(
        direction=SchematicDirection.DISTRIBUTION,
        model_type=ModelType.MODEL_2,
        tap2_valid=True,
        tap3_valid=True,
        range_looks_horizontal=True,
        deviations_are_wicks_or_bad_breaks=True,
        highest_valid_tf="45min",
        tap23_gap_reasonable=True,
        extreme_liq_obvious=True,
        extreme_sd_obvious=False,
        sd_tf_proportional_to_range=True,
        trendline_liquidity_present=True,   # Beautiful trendline liquidity trail
        primary_bos_rr=1.1,
        lower_tf_bos_available=False,       # 1.1R is borderline, no better refinement needed
        lower_tf_bos_entry=EntryTimeframe.BLACK_PRIMARY,
        lower_tf_bos_inside_range=True,
        bos_inside_sd_zone=False,
        retest_occurred=False,
        retest_blue_confirmed=False,
        bos_confirmed=True,
        ltf_bullish_break_then_new_low=False,
    )
    print_5b_evaluation(evaluate_5b_schematic(sol_30m_m2_dist), "SOLANA 30min M2 Distribution (Trendline Liq)")

    # Example 5: Range formation signal detected
    range_form_signal = TCT5BInputs(
        direction=SchematicDirection.ACCUMULATION,
        model_type=ModelType.MODEL_1,
        tap2_valid=True,
        tap3_valid=True,
        range_looks_horizontal=True,
        deviations_are_wicks_or_bad_breaks=True,
        highest_valid_tf="30min",
        tap23_gap_reasonable=True,
        extreme_liq_obvious=False,
        extreme_sd_obvious=False,
        sd_tf_proportional_to_range=True,
        trendline_liquidity_present=False,
        primary_bos_rr=2.0,
        lower_tf_bos_available=False,
        lower_tf_bos_entry=EntryTimeframe.NONE,
        lower_tf_bos_inside_range=False,
        bos_inside_sd_zone=False,
        retest_occurred=False,
        retest_blue_confirmed=False,
        bos_confirmed=False,
        ltf_bullish_break_then_new_low=True,  # LTF break up → immediate new low = range forming
    )
    print_5b_evaluation(evaluate_5b_schematic(range_form_signal), "Range Formation Signal")

    # Example 6: GOLD 15min M2 Accumulation — 0.62R primary, blue BOS 2.94R
    gold_15m_m2_acc = TCT5BInputs(
        direction=SchematicDirection.ACCUMULATION,
        model_type=ModelType.MODEL_2,
        tap2_valid=True,
        tap3_valid=True,
        range_looks_horizontal=True,
        deviations_are_wicks_or_bad_breaks=True,
        highest_valid_tf="15min",
        tap23_gap_reasonable=True,
        extreme_liq_obvious=True,      # 3min MSL
        extreme_sd_obvious=True,       # 5min OB in extreme discount
        sd_tf_proportional_to_range=True,
        trendline_liquidity_present=False,
        primary_bos_rr=0.62,
        lower_tf_bos_available=True,
        lower_tf_bos_entry=EntryTimeframe.BLUE_LOW,
        lower_tf_bos_inside_range=True,
        bos_inside_sd_zone=False,
        retest_occurred=False,
        retest_blue_confirmed=False,
        bos_confirmed=True,
        ltf_bullish_break_then_new_low=False,
    )
    print_5b_evaluation(evaluate_5b_schematic(gold_15m_m2_acc), "GOLD 15min M2 Accumulation")
