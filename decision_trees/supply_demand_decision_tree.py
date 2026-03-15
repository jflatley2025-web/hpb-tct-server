"""
TCT Lecture 3 — Supply & Demand: Decision Tree (Python)
Fully aligned with the hybrid HTML logic for HPM v19.
Optimized for deterministic evaluation in a trading engine.

Usage:
    from decision_trees.supply_demand_decision_tree import evaluate_sd_zone
    result = evaluate_sd_zone(inputs)
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional


# ──────────────────────────────
# Enumerations
# ──────────────────────────────

class MarketContext(Enum):
    RANGE = auto()
    UPTREND = auto()
    DOWNTREND = auto()
    UNCLEAR = auto()


class ZoneType(Enum):
    ORDER_BLOCK = "Order Block (OBIF) — single candle + FVG"
    STRUCTURE_ZONE = "Structure Supply/Demand — multiple candles + FVG"


class ZoneDirection(Enum):
    SUPPLY = "Supply (bearish zone — distribution)"
    DEMAND = "Demand (bullish zone — accumulation)"


class MitigationStatus(Enum):
    UNMITIGATED = "Unmitigated — zone has not been entered"
    PARTIAL_WICK = "Wick through only — zone still valid (liquidity grab)"
    MINOR_MITIGATION = "Minor mitigation only — sole-zone exception may apply"
    FULL_MITIGATION = "Fully mitigated + expansion away — RETIRE this zone"


class ZonePriority(Enum):
    EXTREME = "Extreme zone — last remaining before range High/Low (highest priority)"
    MULTI_TF_CONFLUENCE = "Multi-TF confluence — higher TF + lower TF OB both unmitigated"
    STRUCTURE_PLUS_OB = "Structure zone + refined OB (broad + precise)"
    SINGLE_OB = "Single-layer OB — valid but lower confluence"
    SUPPLY_CHAIN_SECOND = "Supply chain 2nd OB — strong reaction expected"


class TradeBias(Enum):
    LONG = "Long / Buy — demand zone confirmed"
    SHORT = "Short / Sell — supply zone confirmed"
    WAIT = "Wait — zone not yet reached or conditions not met"


# ──────────────────────────────
# Input / Output Dataclasses
# ──────────────────────────────

@dataclass
class SDZoneInputs:
    # Phase 1
    market_context: MarketContext
    zone_direction: ZoneDirection
    context_reason_exists: bool

    # Phase 2
    zone_type: ZoneType

    # Phase 3
    fvg_confirmed: bool
    fvg_tapped_from_top_down: bool

    # Phase 4
    adjacent_candle_has_more_extreme_wick: bool

    # Phase 5
    mitigation_status: MitigationStatus
    is_only_zone_in_area: bool  # Only relevant for MINOR_MITIGATION

    # Phase 6
    refined_ob_found_on_lower_tf: bool
    higher_tf_ob_unmitigated: bool
    higher_tf_ob_mitigated_on_lower: bool
    is_supply_chain_second_ob: bool

    # Phase 7
    is_extreme_zone: bool

    # Phase 8
    price_inside_zone: bool
    tct_schematic_confirmed: bool


@dataclass
class SDZoneEvaluation:
    zone_direction: Optional[ZoneDirection] = None
    zone_type: Optional[ZoneType] = None
    fvg_valid: bool = False
    mitigation_status: Optional[MitigationStatus] = None
    priority: Optional[ZonePriority] = None
    trade_bias: TradeBias = TradeBias.WAIT
    primary_target: str = ""
    draw_note: str = ""
    entry_note: str = ""
    warnings: list[str] = field(default_factory=list)
    passed_phases: list[str] = field(default_factory=list)
    failed_at_phase: Optional[str] = None
    # Populated instead of failed_at_phase when the zone is valid but entry
    # conditions are not yet met — distinguishes "waiting" from "invalid".
    wait_reason: Optional[str] = None


# ──────────────────────────────
# Phase Functions
# ──────────────────────────────

def phase1_build_context(inputs: SDZoneInputs, result: SDZoneEvaluation) -> bool:
    if inputs.market_context == MarketContext.UNCLEAR:
        result.failed_at_phase = "Phase 1: Market context unclear — wait for structure."
        return False

    if not inputs.context_reason_exists:
        result.failed_at_phase = "Phase 1: No structural reason — do not place zones."
        return False

    context_map = {
        MarketContext.RANGE: "Range: supply upper / demand lower",
        MarketContext.UPTREND: "Uptrend: demand zones at higher-low pivots",
        MarketContext.DOWNTREND: "Downtrend: supply zones at lower-high pivots",
    }
    result.passed_phases.append(
        f"Phase 1: Context confirmed — {inputs.market_context.name}. "
        f"{context_map.get(inputs.market_context)}"
    )
    result.zone_direction = inputs.zone_direction
    result.zone_type = inputs.zone_type
    return True


def phase2_identify_zone_type(inputs: SDZoneInputs, result: SDZoneEvaluation) -> bool:
    result.passed_phases.append(f"Phase 2: Zone type identified — {inputs.zone_type.value}")
    return True


def phase3_confirm_fvg(inputs: SDZoneInputs, result: SDZoneEvaluation) -> bool:
    if not inputs.fvg_confirmed:
        result.fvg_valid = False
        result.failed_at_phase = "Phase 3: No FVG detected — invalid zone."
        return False

    result.fvg_valid = True
    fvg_note = " (FVG tapped from top down)" if inputs.fvg_tapped_from_top_down else ""
    result.passed_phases.append(f"Phase 3: FVG confirmed — valid inefficiency{fvg_note}.")
    return True


def phase4_draw_zone(inputs: SDZoneInputs, result: SDZoneEvaluation):
    if inputs.zone_direction == ZoneDirection.SUPPLY:
        if inputs.adjacent_candle_has_more_extreme_wick:
            result.draw_note = "SUPPLY OB: Extend top to adjacent candle wick high (most extreme)."
        else:
            result.draw_note = "SUPPLY OB: Box from wick low to wick high of last bullish candle."
    else:
        if inputs.adjacent_candle_has_more_extreme_wick:
            result.draw_note = "DEMAND OB: Extend bottom to adjacent candle wick low (most extreme)."
        else:
            result.draw_note = "DEMAND OB: Box from wick low to wick high of last bearish candle."

    result.passed_phases.append(f"Phase 4: Drawing note — {result.draw_note}")


def phase5_check_mitigation(inputs: SDZoneInputs, result: SDZoneEvaluation) -> bool:
    result.mitigation_status = inputs.mitigation_status

    if inputs.mitigation_status == MitigationStatus.FULL_MITIGATION:
        result.failed_at_phase = "Phase 5: FULLY MITIGATED — remove zone."
        return False

    if inputs.mitigation_status == MitigationStatus.MINOR_MITIGATION:
        if not inputs.is_only_zone_in_area:
            result.failed_at_phase = "Phase 5: Minor mitigation but other zones exist — skip this zone."
            return False
        result.warnings.append(
            "Phase 5: Sole-zone exception — redraw and still treat as unmitigated."
        )
        # Machine state must reflect the exception so downstream sees UNMITIGATED.
        result.mitigation_status = MitigationStatus.UNMITIGATED

    if inputs.mitigation_status == MitigationStatus.PARTIAL_WICK:
        result.warnings.append("Phase 5: Wick only — liquidity grab, zone remains valid.")

    result.passed_phases.append(
        f"Phase 5: Mitigation check passed — {inputs.mitigation_status.value}"
    )
    return True


def phase6_refine_timeframe(inputs: SDZoneInputs, result: SDZoneEvaluation):
    if inputs.is_supply_chain_second_ob:
        result.priority = ZonePriority.SUPPLY_CHAIN_SECOND
        result.passed_phases.append("Phase 6: Supply chain 2nd OB — high priority.")
        return

    if (
        inputs.higher_tf_ob_unmitigated
        and not inputs.higher_tf_ob_mitigated_on_lower
        and inputs.refined_ob_found_on_lower_tf
    ):
        result.priority = ZonePriority.MULTI_TF_CONFLUENCE
        result.passed_phases.append("Phase 6: Multi-TF confluence — best case.")
    elif inputs.refined_ob_found_on_lower_tf:
        result.priority = ZonePriority.STRUCTURE_PLUS_OB
        result.passed_phases.append("Phase 6: Refined OB found — place both zones.")
    else:
        result.priority = ZonePriority.SINGLE_OB
        result.passed_phases.append("Phase 6: Single-layer OB — valid.")


def phase7_assess_extreme(inputs: SDZoneInputs, result: SDZoneEvaluation):
    if inputs.is_extreme_zone:
        result.priority = ZonePriority.EXTREME
        result.passed_phases.append("Phase 7: EXTREME ZONE — highest priority.")
    else:
        result.passed_phases.append(
            f"Phase 7: Standard zone. Priority: {result.priority.value if result.priority else 'N/A'}"
        )


def phase8_entry(inputs: SDZoneInputs, result: SDZoneEvaluation):
    if not inputs.price_inside_zone:
        result.trade_bias = TradeBias.WAIT
        result.wait_reason = "Phase 8: Price not yet inside zone — zone is your POI, wait for arrival."
        return

    if not inputs.tct_schematic_confirmed:
        result.trade_bias = TradeBias.WAIT
        result.wait_reason = (
            "Phase 8: Price inside zone but TCT schematic not confirmed. "
            "Do NOT enter on zone touch alone. "
            "Wait for TCT Model 1 or Model 2 confirmation inside the zone."
        )
        return

    # Set trade bias and primary targets — extreme zone takes highest precedence.
    if inputs.zone_direction == ZoneDirection.DEMAND:
        result.trade_bias = TradeBias.LONG
        if inputs.is_extreme_zone:
            result.primary_target = "Body of the range / upper supply zone"
        elif inputs.market_context == MarketContext.RANGE:
            result.primary_target = "Range High (or upper range supply zone)"
        elif inputs.market_context == MarketContext.UPTREND:
            result.primary_target = "Next higher high (trend continuation)"
        else:
            result.primary_target = "Range High or next structural high"
    else:
        result.trade_bias = TradeBias.SHORT
        if inputs.is_extreme_zone:
            result.primary_target = "Body of the range / lower demand zone"
        elif inputs.market_context == MarketContext.RANGE:
            result.primary_target = "Range Low (or lower range demand zone)"
        elif inputs.market_context == MarketContext.DOWNTREND:
            result.primary_target = "Next lower low (trend continuation)"
        else:
            result.primary_target = "Range Low or next structural low"

    result.entry_note = (
        f"{result.trade_bias.value} — TCT schematic confirmed inside "
        f"{inputs.zone_direction.value}. Target: {result.primary_target}."
    )
    result.passed_phases.append(f"Phase 8: Entry conditions met — {result.trade_bias.value}")


# ──────────────────────────────────────────────────────────
# Master evaluation entry point
# ──────────────────────────────────────────────────────────

def evaluate_sd_zone(inputs: SDZoneInputs) -> SDZoneEvaluation:
    """
    Run the full TCT Supply & Demand decision tree.

    Returns an SDZoneEvaluation describing:
      - Whether the zone is valid (FVG confirmed, not mitigated)
      - Zone priority (extreme / multi-TF / structure+OB / single)
      - Trade bias (long/short/wait) and primary target
      - Drawing notes and any warnings
      - Which phases passed and where evaluation stopped
    """
    result = SDZoneEvaluation()

    if not phase1_build_context(inputs, result):
        return result

    if not phase2_identify_zone_type(inputs, result):
        return result

    if not phase3_confirm_fvg(inputs, result):
        return result

    phase4_draw_zone(inputs, result)

    if not phase5_check_mitigation(inputs, result):
        return result

    phase6_refine_timeframe(inputs, result)
    phase7_assess_extreme(inputs, result)
    phase8_entry(inputs, result)

    return result


# ──────────────────────────────────────────────────────────
# Pretty-print helper
# ──────────────────────────────────────────────────────────

def print_evaluation(result: SDZoneEvaluation):
    """Print a human-readable summary of the S&D evaluation."""
    print("\n" + "=" * 62)
    print("  TCT SUPPLY & DEMAND DECISION TREE — EVALUATION RESULT")
    print("=" * 62)
    print(f"  Zone Direction:  {result.zone_direction.value if result.zone_direction else 'N/A'}")
    print(f"  Zone Type:       {result.zone_type.value if result.zone_type else 'N/A'}")
    print(f"  FVG Valid:       {'YES' if result.fvg_valid else 'NO'}")
    print(f"  Mitigation:      {result.mitigation_status.value if result.mitigation_status else 'N/A'}")
    print(f"  Priority:        {result.priority.value if result.priority else 'N/A'}")
    print(f"  Trade Bias:      {result.trade_bias.value}")
    print(f"  Primary Target:  {result.primary_target or 'N/A'}")
    if result.draw_note:
        print(f"  Draw Note:       {result.draw_note}")
    if result.entry_note:
        print(f"  Entry Note:      {result.entry_note}")
    print()
    print("  Phases Passed:")
    for p in result.passed_phases:
        print(f"    ✓ {p}")
    if result.failed_at_phase:
        print(f"  Stopped At: ✗ {result.failed_at_phase}")
    if result.wait_reason:
        print(f"  Waiting:    ⏳ {result.wait_reason}")
    if result.warnings:
        print("  Warnings:")
        for w in result.warnings:
            print(f"    ⚠ {w}")
    print("=" * 62)


# ──────────────────────────────────────────────────────────
# Example usage
# ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Example 1: Valid demand OB in a range — price inside, TCT confirmed
    example_demand = SDZoneInputs(
        market_context=MarketContext.RANGE,
        zone_direction=ZoneDirection.DEMAND,
        context_reason_exists=True,
        zone_type=ZoneType.ORDER_BLOCK,
        fvg_confirmed=True,
        fvg_tapped_from_top_down=False,
        adjacent_candle_has_more_extreme_wick=True,
        mitigation_status=MitigationStatus.UNMITIGATED,
        is_only_zone_in_area=False,
        refined_ob_found_on_lower_tf=True,
        higher_tf_ob_unmitigated=True,
        higher_tf_ob_mitigated_on_lower=False,
        is_supply_chain_second_ob=False,
        is_extreme_zone=True,
        price_inside_zone=True,
        tct_schematic_confirmed=True,
    )
    print_evaluation(evaluate_sd_zone(example_demand))

    # Example 2: Bearish supply zone in downtrend — not yet entered
    example_supply = SDZoneInputs(
        market_context=MarketContext.DOWNTREND,
        zone_direction=ZoneDirection.SUPPLY,
        context_reason_exists=True,
        zone_type=ZoneType.STRUCTURE_ZONE,
        fvg_confirmed=True,
        fvg_tapped_from_top_down=False,
        adjacent_candle_has_more_extreme_wick=False,
        mitigation_status=MitigationStatus.UNMITIGATED,
        is_only_zone_in_area=False,
        refined_ob_found_on_lower_tf=True,
        higher_tf_ob_unmitigated=False,
        higher_tf_ob_mitigated_on_lower=False,
        is_supply_chain_second_ob=False,
        is_extreme_zone=False,
        price_inside_zone=False,
        tct_schematic_confirmed=False,
    )
    print_evaluation(evaluate_sd_zone(example_supply))

    # Example 3: Invalid — no FVG
    example_invalid = SDZoneInputs(
        market_context=MarketContext.RANGE,
        zone_direction=ZoneDirection.SUPPLY,
        context_reason_exists=True,
        zone_type=ZoneType.ORDER_BLOCK,
        fvg_confirmed=False,
        fvg_tapped_from_top_down=False,
        adjacent_candle_has_more_extreme_wick=False,
        mitigation_status=MitigationStatus.UNMITIGATED,
        is_only_zone_in_area=False,
        refined_ob_found_on_lower_tf=False,
        higher_tf_ob_unmitigated=False,
        higher_tf_ob_mitigated_on_lower=False,
        is_supply_chain_second_ob=False,
        is_extreme_zone=False,
        price_inside_zone=False,
        tct_schematic_confirmed=False,
    )
    print_evaluation(evaluate_sd_zone(example_invalid))
