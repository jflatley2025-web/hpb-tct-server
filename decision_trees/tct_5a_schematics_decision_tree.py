"""
TCT Lecture 5A — TCT Schematics: Decision Tree (Python)

Encodes the full decision logic from TCT 2024 Mentorship Lecture 5A | TCT Schematics.

Four schematics — all three-tap models:
  Model 1 Accumulation  — two lower deviations → bullish BOS → long
  Model 2 Accumulation  — one deviation + higher low → bullish BOS → long
  Model 1 Distribution  — two higher deviations → bearish BOS → short
  Model 2 Distribution  — one deviation + lower high → bearish BOS → short

Usage:
    from decision_trees.tct_5a_schematics_decision_tree import evaluate_tct_schematic
    result = evaluate_tct_schematic(inputs)
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional


# ──────────────────────────────────────────────────────────
# Enumerations
# ──────────────────────────────────────────────────────────

class SchematicDirection(Enum):
    ACCUMULATION = "Accumulation / Re-accumulation — Range Low being deviated → expect break up"
    DISTRIBUTION = "Distribution / Re-distribution — Range High being deviated → expect break down"


class ModelType(Enum):
    MODEL_1 = "Model 1 — two successive deeper/higher deviations (T1 → T2 → T3 all more extreme)"
    MODEL_2 = "Model 2 — one deviation then higher low / lower high meeting extreme liq or S/D"
    UNKNOWN = "Unknown — not yet determined (wait for Tap 3 to form)"


class BOSLocation(Enum):
    INSIDE_RANGE = "Inside original range values — preferred, highest confidence"
    OUTSIDE_RANGE = "Outside original range values — valid but riskier"
    NOT_YET = "BOS has not occurred yet"


class BOSTimeframe(Enum):
    BLACK_HIGH_TF = "Black (higher TF) BOS"
    RED_MID_TF = "Red (mid TF) BOS — used when black gives poor R:R"
    NOT_YET = "BOS not yet confirmed"


class SchematicStatus(Enum):
    FORMING = "Schematic still forming — do not enter"
    VALID_ENTRY = "Valid entry — BOS confirmed, all conditions met"
    INVALID = "Invalid — schematic conditions not met"
    MODEL2_FAILED_WATCH_M1 = "Model 2 failed — watch for Model 1 to form"


class TradeBias(Enum):
    LONG = "Long — Accumulation schematic, bullish BOS confirmed"
    SHORT = "Short — Distribution schematic, bearish BOS confirmed"
    WAIT = "Wait — conditions not yet met"


# ──────────────────────────────────────────────────────────
# Input / Output dataclasses
# ──────────────────────────────────────────────────────────

@dataclass
class TCTSchematicInputs:
    """All observable conditions needed to evaluate a TCT schematic."""

    # Phase 2 — Range prerequisite
    range_confirmed: bool                  # Six-candle rule + EQ touch complete
    dl2_drawn: bool                        # DL2 (30% extension) levels drawn

    # Phase 3 — Direction
    direction: SchematicDirection          # ACCUMULATION or DISTRIBUTION

    # Phase 4 — Tap 2 validation
    tap2_price_exceeded_extreme: bool      # Price moved beyond the range extreme
    tap2_close_beyond_dl2: bool            # True = invalid (true break); False = valid deviation
    tap2_accepted_back_inside: bool        # Price closed back inside range after deviation

    # Tap 2 context for model anticipation
    tap2_grabbed_major_liquidity: bool     # Large liquidity pool taken at Tap 2
    tap2_mitigated_strong_sd: bool         # Strong supply/demand mitigated at Tap 2

    # Phase 6 — Model determination & Tap 3
    model_type: ModelType                  # MODEL_1, MODEL_2, or UNKNOWN

    # Model 1 Tap 3
    tap3_m1_exceeded_tap2: bool            # Price went deeper/higher than Tap 2 (M1 only)
    tap3_m1_close_beyond_dl2: bool         # True = invalid; False = valid (M1 only)

    # Model 2 Tap 3 — requirements
    tap3_m2_is_higher_low_or_lower_high: bool   # T3 stopped short of T2 (M2 only)
    tap3_m2_req_a_extreme_liq: bool             # Grabbed extreme liquidity (first MSL/MSH)
    tap3_m2_req_b_extreme_sd: bool              # Mitigated extreme demand/supply zone
    tap3_m2_sd_in_extreme_zone: bool            # Demand in extreme discount / supply in extreme premium
    tap3_m2_sd_tf_proportional: bool            # Zone TF proportional to range size

    # Phase 7 — BOS
    bos_confirmed: bool                    # BOS from Tap 2–3 structure in expected direction
    bos_wrong_direction: bool              # BOS broke in the wrong direction (M2 failure signal)

    # Phase 8 — BOS location & TF
    bos_location: BOSLocation
    bos_timeframe: BOSTimeframe

    # Black BOS gives poor R:R
    black_bos_poor_rr: bool                # True if black BOS is near EQ (poor R:R)
    red_bos_inside_range: bool             # Red mid-TF BOS is also inside range values


@dataclass
class TCTSchematicEvaluation:
    """Result of the full TCT schematic decision tree."""
    direction: Optional[SchematicDirection] = None
    model_type: Optional[ModelType] = None
    status: SchematicStatus = SchematicStatus.FORMING
    trade_bias: TradeBias = TradeBias.WAIT
    entry_timeframe: str = ""
    stop_loss_note: str = ""
    primary_target: str = ""
    secondary_note: str = ""
    passed_phases: list = field(default_factory=list)
    failed_at_phase: Optional[str] = None
    warnings: list = field(default_factory=list)


# ──────────────────────────────────────────────────────────
# Phase functions
# ──────────────────────────────────────────────────────────

def phase2_range_prereq(inputs: TCTSchematicInputs, result: TCTSchematicEvaluation) -> bool:
    """Phase 2: Confirm the range is valid before any schematic can form."""
    if not inputs.range_confirmed:
        result.failed_at_phase = (
            "Phase 2: Range not confirmed — six-candle rule or EQ touch not satisfied. "
            "No schematic can form until the range is valid."
        )
        result.status = SchematicStatus.INVALID
        return False
    if not inputs.dl2_drawn:
        result.failed_at_phase = (
            "Phase 2: DL2 not drawn — draw 30% extension levels (−0.3 / 1.3 Fib) "
            "before evaluating deviations."
        )
        result.status = SchematicStatus.INVALID
        return False
    result.direction = inputs.direction
    result.passed_phases.append(
        f"Phase 2: Range confirmed. DL2 drawn. "
        f"Direction: {inputs.direction.value}"
    )
    return True


def phase3_direction(inputs: TCTSchematicInputs, result: TCTSchematicEvaluation):
    """Phase 3: Record schematic direction and implied target."""
    if inputs.direction == SchematicDirection.ACCUMULATION:
        result.primary_target = "Range High / Wyckoff High"
        result.stop_loss_note = "Below Tap 3 low"
    else:
        result.primary_target = "Range Low / Wyckoff Low"
        result.stop_loss_note = "Above Tap 3 high"
    result.passed_phases.append(
        f"Phase 3: Schematic direction = {inputs.direction.name}. "
        f"Target: {result.primary_target}. SL: {result.stop_loss_note}."
    )


def phase4_validate_tap2(inputs: TCTSchematicInputs, result: TCTSchematicEvaluation) -> bool:
    """Phase 4: Validate Tap 2 — first deviation of the range extreme."""
    if not inputs.tap2_price_exceeded_extreme:
        result.failed_at_phase = (
            "Phase 4: Tap 2 not yet formed — price has not exceeded the range extreme. Wait."
        )
        return False

    if inputs.tap2_close_beyond_dl2:
        result.failed_at_phase = (
            "Phase 4: Tap 2 INVALID — candle closed beyond DL2. "
            "This is a true range break, not a deviation. Reassess macro structure."
        )
        result.status = SchematicStatus.INVALID
        return False

    if not inputs.tap2_accepted_back_inside:
        result.failed_at_phase = (
            "Phase 4: Tap 2 not yet confirmed — price has not accepted back inside the range. "
            "Wait for acceptance before proceeding."
        )
        return False

    # Model anticipation note
    if inputs.tap2_grabbed_major_liquidity or inputs.tap2_mitigated_strong_sd:
        result.warnings.append(
            "Phase 4: Tap 2 grabbed major liquidity or mitigated strong S/D — "
            "Model 2 is likely. Anticipate, but wait for confirmation. "
            "If structure does not break, watch for Model 1."
        )

    result.passed_phases.append(
        "Phase 4: Tap 2 VALID — deviation within DL2, accepted back inside. "
        "Range extreme extended to Tap 2. DL2 recalculated."
    )
    return True


def phase6_validate_tap3(inputs: TCTSchematicInputs, result: TCTSchematicEvaluation) -> bool:
    """Phase 6: Validate Tap 3 for the determined model type."""
    result.model_type = inputs.model_type

    if inputs.model_type == ModelType.MODEL_1:
        if not inputs.tap3_m1_exceeded_tap2:
            result.failed_at_phase = (
                "Phase 6 (Model 1): Tap 3 did not exceed Tap 2 extreme. "
                "Model 1 Tap 3 requires a new deeper/higher deviation. Wait."
            )
            return False
        if inputs.tap3_m1_close_beyond_dl2:
            result.failed_at_phase = (
                "Phase 6 (Model 1): Tap 3 closed beyond DL2 — true break. "
                "Schematic INVALID. Reassess direction."
            )
            result.status = SchematicStatus.INVALID
            return False
        result.passed_phases.append(
            "Phase 6: Model 1 Tap 3 VALID — new deeper/higher deviation within DL2."
        )

    elif inputs.model_type == ModelType.MODEL_2:
        if not inputs.tap3_m2_is_higher_low_or_lower_high:
            result.failed_at_phase = (
                "Phase 6 (Model 2): Tap 3 is not a higher low / lower high — "
                "it exceeded Tap 2 extreme. This is a Model 1 pattern, not Model 2."
            )
            return False

        req_a = inputs.tap3_m2_req_a_extreme_liq
        req_b = inputs.tap3_m2_req_b_extreme_sd

        if not req_a and not req_b:
            result.failed_at_phase = (
                "Phase 6 (Model 2): Tap 3 is INVALID — meets neither Requirement A "
                "(extreme liquidity) nor Requirement B (extreme demand/supply mitigation). "
                "Do not trade as Model 2. Continue watching for Model 1."
            )
            return False

        notes = []
        if req_a:
            notes.append("Req A (extreme liquidity) met")
        if req_b:
            notes.append("Req B (extreme demand/supply) met")
            if not inputs.tap3_m2_sd_in_extreme_zone:
                result.warnings.append(
                    "Phase 6: Model 2 Req B — demand/supply zone is not in the "
                    "extreme discount/premium (bottom/top 25% of range). "
                    "Still valid but preferred zone location is extreme discount/premium."
                )
            if not inputs.tap3_m2_sd_tf_proportional:
                result.warnings.append(
                    "Phase 6: Model 2 Req B — demand/supply zone timeframe may not be "
                    "proportional to the range size. Do not use 1m OBs for a 1h range schematic."
                )

        result.passed_phases.append(
            f"Phase 6: Model 2 Tap 3 VALID — {' + '.join(notes)}."
        )

    else:  # UNKNOWN
        result.failed_at_phase = (
            "Phase 6: Model type not yet determined. "
            "Observe whether Tap 3 exceeds Tap 2 (Model 1) or forms a higher/lower extreme (Model 2)."
        )
        return False

    return True


def phase7_bos(inputs: TCTSchematicInputs, result: TCTSchematicEvaluation) -> bool:
    """Phase 7: Draw Tap 2–3 structure and wait for BOS."""
    if inputs.bos_wrong_direction:
        result.status = SchematicStatus.MODEL2_FAILED_WATCH_M1
        result.failed_at_phase = (
            "Phase 7: BOS broke in the WRONG direction. "
            "If in Model 2 position → take the loss. "
            "Watch for price to take out Tap 2 extreme — if so, trade as Model 1."
        )
        return False

    if not inputs.bos_confirmed:
        result.failed_at_phase = (
            "Phase 7: BOS not yet confirmed from the Tap 2–3 structure. "
            "Do not enter. 'You trade confirmations, not expectations.'"
        )
        return False

    direction_label = "bullish" if inputs.direction == SchematicDirection.ACCUMULATION else "bearish"
    result.passed_phases.append(
        f"Phase 7: {direction_label.upper()} BOS confirmed from Tap 2–3 structure."
    )
    return True


def phase8_bos_location(inputs: TCTSchematicInputs, result: TCTSchematicEvaluation):
    """Phase 8: Evaluate BOS location and determine optimal entry timeframe."""
    if inputs.bos_location == BOSLocation.INSIDE_RANGE:
        result.passed_phases.append(
            "Phase 8: BOS INSIDE original range values — highest confidence entry."
        )
    else:
        result.warnings.append(
            "Phase 8: BOS OUTSIDE original range values — valid but riskier. "
            "Consider smaller position size or wait for price to return inside the range."
        )

    # Multi-TF refinement
    if inputs.black_bos_poor_rr:
        if inputs.red_bos_inside_range:
            result.entry_timeframe = "Red (mid-TF) BOS — inside original range values, better R:R"
            result.passed_phases.append(
                "Phase 8: Multi-TF refinement — black BOS gives poor R:R (near EQ). "
                "Red (mid-TF) BOS is inside range values → using red BOS for entry. "
                "Monitor: if red structure breaks in wrong direction before taking black-level low, "
                "be cautious (often a temporary supply mitigation bounce before resuming)."
            )
        else:
            result.entry_timeframe = "Black (high-TF) BOS — red BOS not inside range values"
            result.warnings.append(
                "Phase 8: Black BOS gives poor R:R but red BOS is also outside range values. "
                "Use black BOS as primary — accept the lower R:R."
            )
    else:
        result.entry_timeframe = "Black (high-TF) BOS — primary entry"
        result.passed_phases.append(
            f"Phase 8: Entry on {result.entry_timeframe}."
        )


def phase9_entry(inputs: TCTSchematicInputs, result: TCTSchematicEvaluation):
    """Phase 9: Set trade bias, SL, and target."""
    if inputs.direction == SchematicDirection.ACCUMULATION:
        result.trade_bias = TradeBias.LONG
    else:
        result.trade_bias = TradeBias.SHORT

    result.status = SchematicStatus.VALID_ENTRY
    result.secondary_note = (
        "Anything beyond the Wyckoff High/Low is extra — hold if broader "
        "market structure (HPB, main trend) supports continuation."
    )
    result.passed_phases.append(
        f"Phase 9: ENTRY — {result.trade_bias.value}. "
        f"SL: {result.stop_loss_note}. Target: {result.primary_target}."
    )


# ──────────────────────────────────────────────────────────
# Master evaluation entry point
# ──────────────────────────────────────────────────────────

def evaluate_tct_schematic(inputs: TCTSchematicInputs) -> TCTSchematicEvaluation:
    """
    Run the full TCT Schematic decision tree (5A).

    Returns a TCTSchematicEvaluation describing:
      - Schematic direction and model type
      - Status (FORMING / VALID_ENTRY / INVALID / MODEL2_FAILED_WATCH_M1)
      - Trade bias (long/short/wait)
      - Entry timeframe (black / red BOS)
      - Stop loss placement and primary target
      - All phases passed and failure point
    """
    result = TCTSchematicEvaluation()

    if not phase2_range_prereq(inputs, result):
        return result

    phase3_direction(inputs, result)

    if not phase4_validate_tap2(inputs, result):
        return result

    if not phase6_validate_tap3(inputs, result):
        return result

    if not phase7_bos(inputs, result):
        return result

    phase8_bos_location(inputs, result)
    phase9_entry(inputs, result)

    return result


# ──────────────────────────────────────────────────────────
# Pretty-print helper
# ──────────────────────────────────────────────────────────

def print_evaluation(result: TCTSchematicEvaluation):
    """Print a human-readable summary of the schematic evaluation."""
    print("\n" + "=" * 66)
    print("  TCT SCHEMATIC (5A) DECISION TREE — EVALUATION RESULT")
    print("=" * 66)
    print(f"  Direction:       {result.direction.value if result.direction else 'N/A'}")
    print(f"  Model Type:      {result.model_type.value if result.model_type else 'N/A'}")
    print(f"  Status:          {result.status.value}")
    print(f"  Trade Bias:      {result.trade_bias.value}")
    print(f"  Entry TF:        {result.entry_timeframe or 'N/A'}")
    print(f"  Stop Loss:       {result.stop_loss_note or 'N/A'}")
    print(f"  Primary Target:  {result.primary_target or 'N/A'}")
    if result.secondary_note:
        print(f"  Note:            {result.secondary_note}")
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
    # Example 1: Model 1 Accumulation — full valid entry
    m1_acc = TCTSchematicInputs(
        range_confirmed=True,
        dl2_drawn=True,
        direction=SchematicDirection.ACCUMULATION,
        tap2_price_exceeded_extreme=True,
        tap2_close_beyond_dl2=False,
        tap2_accepted_back_inside=True,
        tap2_grabbed_major_liquidity=False,
        tap2_mitigated_strong_sd=False,
        model_type=ModelType.MODEL_1,
        tap3_m1_exceeded_tap2=True,
        tap3_m1_close_beyond_dl2=False,
        tap3_m2_is_higher_low_or_lower_high=False,
        tap3_m2_req_a_extreme_liq=False,
        tap3_m2_req_b_extreme_sd=False,
        tap3_m2_sd_in_extreme_zone=False,
        tap3_m2_sd_tf_proportional=True,
        bos_confirmed=True,
        bos_wrong_direction=False,
        bos_location=BOSLocation.INSIDE_RANGE,
        bos_timeframe=BOSTimeframe.BLACK_HIGH_TF,
        black_bos_poor_rr=False,
        red_bos_inside_range=False,
    )
    print_evaluation(evaluate_tct_schematic(m1_acc))

    # Example 2: Model 2 Distribution — black BOS poor R:R, using red BOS
    m2_dist = TCTSchematicInputs(
        range_confirmed=True,
        dl2_drawn=True,
        direction=SchematicDirection.DISTRIBUTION,
        tap2_price_exceeded_extreme=True,
        tap2_close_beyond_dl2=False,
        tap2_accepted_back_inside=True,
        tap2_grabbed_major_liquidity=True,
        tap2_mitigated_strong_sd=True,
        model_type=ModelType.MODEL_2,
        tap3_m1_exceeded_tap2=False,
        tap3_m1_close_beyond_dl2=False,
        tap3_m2_is_higher_low_or_lower_high=True,
        tap3_m2_req_a_extreme_liq=True,
        tap3_m2_req_b_extreme_sd=True,
        tap3_m2_sd_in_extreme_zone=True,
        tap3_m2_sd_tf_proportional=True,
        bos_confirmed=True,
        bos_wrong_direction=False,
        bos_location=BOSLocation.INSIDE_RANGE,
        bos_timeframe=BOSTimeframe.RED_MID_TF,
        black_bos_poor_rr=True,
        red_bos_inside_range=True,
    )
    print_evaluation(evaluate_tct_schematic(m2_dist))

    # Example 3: Model 2 failing → watch for Model 1
    m2_fail = TCTSchematicInputs(
        range_confirmed=True,
        dl2_drawn=True,
        direction=SchematicDirection.ACCUMULATION,
        tap2_price_exceeded_extreme=True,
        tap2_close_beyond_dl2=False,
        tap2_accepted_back_inside=True,
        tap2_grabbed_major_liquidity=True,
        tap2_mitigated_strong_sd=False,
        model_type=ModelType.MODEL_2,
        tap3_m1_exceeded_tap2=False,
        tap3_m1_close_beyond_dl2=False,
        tap3_m2_is_higher_low_or_lower_high=True,
        tap3_m2_req_a_extreme_liq=True,
        tap3_m2_req_b_extreme_sd=False,
        tap3_m2_sd_in_extreme_zone=False,
        tap3_m2_sd_tf_proportional=True,
        bos_confirmed=False,
        bos_wrong_direction=True,       # BOS broke bearish — wrong for accumulation
        bos_location=BOSLocation.NOT_YET,
        bos_timeframe=BOSTimeframe.NOT_YET,
        black_bos_poor_rr=False,
        red_bos_inside_range=False,
    )
    print_evaluation(evaluate_tct_schematic(m2_fail))

    # Example 4: Model 2 Accumulation — Req B only, zone not in extreme discount (warning)
    m2_acc_reqb = TCTSchematicInputs(
        range_confirmed=True,
        dl2_drawn=True,
        direction=SchematicDirection.ACCUMULATION,
        tap2_price_exceeded_extreme=True,
        tap2_close_beyond_dl2=False,
        tap2_accepted_back_inside=True,
        tap2_grabbed_major_liquidity=False,
        tap2_mitigated_strong_sd=False,
        model_type=ModelType.MODEL_2,
        tap3_m1_exceeded_tap2=False,
        tap3_m1_close_beyond_dl2=False,
        tap3_m2_is_higher_low_or_lower_high=True,
        tap3_m2_req_a_extreme_liq=False,
        tap3_m2_req_b_extreme_sd=True,
        tap3_m2_sd_in_extreme_zone=False,   # Not in extreme discount — warning
        tap3_m2_sd_tf_proportional=True,
        bos_confirmed=True,
        bos_wrong_direction=False,
        bos_location=BOSLocation.INSIDE_RANGE,
        bos_timeframe=BOSTimeframe.BLACK_HIGH_TF,
        black_bos_poor_rr=False,
        red_bos_inside_range=False,
    )
    print_evaluation(evaluate_tct_schematic(m2_acc_reqb))
