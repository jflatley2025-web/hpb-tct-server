"""
TCT Lecture 3 — Supply & Demand: Decision Tree (Python)

Encodes the full decision logic from TCT 2024 Mentorship Lecture 3 | Supply & Demand.
Each function corresponds to one phase of the decision tree.

Usage:
    from decision_trees.supply_demand_decision_tree import evaluate_sd_zone
    result = evaluate_sd_zone(inputs)
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional


# ──────────────────────────────────────────────────────────
# Enumerations
# ──────────────────────────────────────────────────────────

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


# ──────────────────────────────────────────────────────────
# Input / Output dataclasses
# ──────────────────────────────────────────────────────────

@dataclass
class SDZoneInputs:
    """All observable conditions needed to evaluate a supply/demand zone."""

    # Phase 1 — Context
    market_context: MarketContext           # RANGE, UPTREND, DOWNTREND, UNCLEAR
    zone_direction: ZoneDirection           # SUPPLY or DEMAND
    context_reason_exists: bool            # True if there is a structural reason for this zone

    # Phase 2 — Zone type
    zone_type: ZoneType                    # ORDER_BLOCK or STRUCTURE_ZONE

    # Phase 3 — FVG validation
    fvg_confirmed: bool                    # Wick C1 and wick C3 do NOT connect → True = FVG exists
    fvg_tapped_from_top_down: bool         # True if FVG was entered from the "better" direction

    # Phase 4 — Drawing inputs
    # (Drawing is done on the chart — these flags capture edge cases)
    adjacent_candle_has_more_extreme_wick: bool  # True if the wick must be extended to adjacent candle

    # Phase 5 — Mitigation status
    mitigation_status: MitigationStatus

    # Only relevant if mitigation_status == MINOR_MITIGATION:
    is_only_zone_in_area: bool             # True if no other supply/demand zones exist nearby

    # Phase 6 — Timeframe refinement
    refined_ob_found_on_lower_tf: bool     # True if a more precise OB was found on a lower TF
    higher_tf_ob_unmitigated: bool         # True if the same zone is also unmitigated on a higher TF
    higher_tf_ob_mitigated_on_lower: bool  # Higher TF OB but mitigated on lower TF (still usable)
    is_supply_chain_second_ob: bool        # True if this is the 2nd OB in a supply chain

    # Phase 7 — Extreme zone
    is_extreme_zone: bool                  # Last remaining supply before range high / demand before range low

    # Phase 8 — Entry
    price_inside_zone: bool                # True if price has entered the zone
    tct_schematic_confirmed: bool          # True if TCT Model 1 or Model 2 confirmation received


@dataclass
class SDZoneEvaluation:
    """Result of the full S&D decision tree evaluation."""
    zone_direction: Optional[ZoneDirection] = None
    zone_type: Optional[ZoneType] = None
    fvg_valid: bool = False
    mitigation_status: Optional[MitigationStatus] = None
    priority: Optional[ZonePriority] = None
    trade_bias: TradeBias = TradeBias.WAIT
    primary_target: str = ""
    draw_note: str = ""
    entry_note: str = ""
    warnings: list = field(default_factory=list)
    passed_phases: list = field(default_factory=list)
    failed_at_phase: Optional[str] = None


# ──────────────────────────────────────────────────────────
# Phase functions
# ──────────────────────────────────────────────────────────

def phase1_build_context(inputs: SDZoneInputs, result: SDZoneEvaluation) -> bool:
    """Phase 1: Confirm structural context exists before placing any zone."""
    if inputs.market_context == MarketContext.UNCLEAR:
        result.failed_at_phase = (
            "Phase 1: Market context unclear — do not place zones. "
            "Wait for structure to become defined."
        )
        return False

    if not inputs.context_reason_exists:
        result.failed_at_phase = (
            "Phase 1: No structural reason for this zone. "
            "Do not randomly place OBs — context is always key."
        )
        return False

    context_map = {
        MarketContext.RANGE: (
            "Range context: supply in upper section / above Range High; "
            "demand in lower section / below Range Low"
        ),
        MarketContext.UPTREND: "Uptrend: demand zones at higher low pivots",
        MarketContext.DOWNTREND: "Downtrend: supply zones at lower high pivots",
    }
    result.passed_phases.append(
        f"Phase 1: Context confirmed — {inputs.market_context.name}. "
        f"{context_map.get(inputs.market_context, '')}"
    )
    result.zone_direction = inputs.zone_direction
    result.zone_type = inputs.zone_type
    return True


def phase2_identify_zone_type(inputs: SDZoneInputs, result: SDZoneEvaluation) -> bool:
    """Phase 2: Identify and record zone type (always passes if context passed)."""
    result.passed_phases.append(
        f"Phase 2: Zone type identified — {inputs.zone_type.value}"
    )
    return True


def phase3_confirm_fvg(inputs: SDZoneInputs, result: SDZoneEvaluation) -> bool:
    """Phase 3: Validate the Fair Value Gap — non-negotiable requirement."""
    if not inputs.fvg_confirmed:
        result.fvg_valid = False
        result.failed_at_phase = (
            "Phase 3: No FVG detected — wick of candle 1 connects to wick of candle 3. "
            "This is NOT a valid order block or structure zone. Disqualified."
        )
        return False

    result.fvg_valid = True
    fvg_note = ""
    if inputs.fvg_tapped_from_top_down:
        fvg_note = " (FVG tapped from top down — retains more validity if previously entered)"
    result.passed_phases.append(
        f"Phase 3: FVG confirmed — valid inefficiency exists.{fvg_note}"
    )
    return True


def phase4_draw_zone(inputs: SDZoneInputs, result: SDZoneEvaluation):
    """Phase 4: Record drawing notes (always passes — drawing is done on chart)."""
    if inputs.zone_direction == ZoneDirection.SUPPLY:
        if inputs.adjacent_candle_has_more_extreme_wick:
            result.draw_note = (
                "SUPPLY OB: Extend the top of the box to include the adjacent candle's wick high "
                "(most extreme wick across candles 1–2 defines the boundary)."
            )
        else:
            result.draw_note = (
                "SUPPLY OB: Box from wick low to wick high of the last bullish candle before expansion."
            )
    else:
        if inputs.adjacent_candle_has_more_extreme_wick:
            result.draw_note = (
                "DEMAND OB: Extend the bottom of the box to include the adjacent candle's wick low "
                "(most extreme wick across candles 1–2 defines the boundary)."
            )
        else:
            result.draw_note = (
                "DEMAND OB: Box from wick low to wick high of the last bearish candle before expansion."
            )

    result.passed_phases.append(f"Phase 4: Drawing note — {result.draw_note}")


def phase5_check_mitigation(inputs: SDZoneInputs, result: SDZoneEvaluation) -> bool:
    """Phase 5: Check mitigation status. Full mitigation = zone retired."""
    result.mitigation_status = inputs.mitigation_status

    if inputs.mitigation_status == MitigationStatus.FULL_MITIGATION:
        result.failed_at_phase = (
            "Phase 5: Zone is FULLY MITIGATED — price entered, reacted, and expanded away. "
            "Remove this zone from the chart. Do not reuse."
        )
        return False

    if inputs.mitigation_status == MitigationStatus.MINOR_MITIGATION:
        if not inputs.is_only_zone_in_area:
            result.failed_at_phase = (
                "Phase 5: Zone was mitigated (not just a wick) and other zones exist nearby. "
                "Use an adjacent unmitigated zone instead."
            )
            return False
        result.warnings.append(
            "Phase 5 (sole-zone exception): Minor mitigation but this is the only zone in the area. "
            "Redraw above the mitigation point (supply) or below it (demand). "
            "Still treat as unmitigated — watch for TCT confirmation."
        )

    if inputs.mitigation_status == MitigationStatus.PARTIAL_WICK:
        result.warnings.append(
            "Phase 5: Price wicked through the zone — this is a liquidity grab, not invalidation. "
            "Zone remains valid."
        )

    result.passed_phases.append(
        f"Phase 5: Mitigation check passed — status: {inputs.mitigation_status.value}"
    )
    return True


def phase6_refine_timeframe(inputs: SDZoneInputs, result: SDZoneEvaluation):
    """Phase 6: Determine zone priority based on timeframe confluence."""
    if inputs.is_supply_chain_second_ob:
        result.priority = ZonePriority.SUPPLY_CHAIN_SECOND
        result.passed_phases.append(
            "Phase 6: Supply chain 2nd OB detected — strong reaction expected. "
            "Mark as high priority."
        )
        return

    if inputs.higher_tf_ob_unmitigated and not inputs.higher_tf_ob_mitigated_on_lower:
        result.priority = ZonePriority.MULTI_TF_CONFLUENCE
        result.passed_phases.append(
            "Phase 6: Multi-TF confluence — higher TF OB also unmitigated on lower TF. Best case."
        )
    elif inputs.refined_ob_found_on_lower_tf:
        result.priority = ZonePriority.STRUCTURE_PLUS_OB
        result.passed_phases.append(
            "Phase 6: Refined OB found on lower TF inside structure zone. "
            "Place both on chart — watch whichever price enters."
        )
    elif inputs.higher_tf_ob_mitigated_on_lower:
        result.priority = ZonePriority.SINGLE_OB
        result.warnings.append(
            "Phase 6: Higher TF OB is mitigated on lower TF — still usable but less ideal. "
            "Higher TF structure takes precedence."
        )
        result.passed_phases.append(
            "Phase 6: Higher TF OB (mitigated on lower TF) — valid, lower confluence."
        )
    else:
        result.priority = ZonePriority.SINGLE_OB
        result.passed_phases.append("Phase 6: Single-layer OB — valid, standard confluence.")


def phase7_assess_extreme(inputs: SDZoneInputs, result: SDZoneEvaluation):
    """Phase 7: Upgrade priority if this is an extreme zone."""
    if inputs.is_extreme_zone:
        result.priority = ZonePriority.EXTREME
        result.passed_phases.append(
            "Phase 7: EXTREME ZONE — last remaining unmitigated zone before range extreme. "
            "Highest priority. Expect strong reaction. New ranges often form here."
        )
    else:
        result.passed_phases.append(
            f"Phase 7: Standard zone. Priority: {result.priority.value if result.priority else 'N/A'}"
        )


def phase8_entry(inputs: SDZoneInputs, result: SDZoneEvaluation):
    """Phase 8: Determine if entry conditions are met."""
    if not inputs.price_inside_zone:
        result.trade_bias = TradeBias.WAIT
        result.failed_at_phase = (
            "Phase 8: Price has not yet entered the zone. "
            "This zone is your POI — wait for price to arrive."
        )
        return

    if not inputs.tct_schematic_confirmed:
        result.trade_bias = TradeBias.WAIT
        result.failed_at_phase = (
            "Phase 8: Price is inside the zone but no TCT schematic confirmed yet. "
            "Do NOT enter on zone touch alone. "
            "Wait for TCT Model 1 or Model 2 confirmation inside the zone."
        )
        return

    # Set bias and targets
    if inputs.zone_direction == ZoneDirection.DEMAND:
        result.trade_bias = TradeBias.LONG
        if inputs.market_context == MarketContext.RANGE:
            result.primary_target = "Range High (or upper range supply zone)"
        elif inputs.market_context == MarketContext.UPTREND:
            result.primary_target = "Next higher high (trend continuation)"
        elif inputs.is_extreme_zone:
            result.primary_target = "Body of the range / upper supply zone"
        else:
            result.primary_target = "Range High or next structural high"
    else:
        result.trade_bias = TradeBias.SHORT
        if inputs.market_context == MarketContext.RANGE:
            result.primary_target = "Range Low (or lower range demand zone)"
        elif inputs.market_context == MarketContext.DOWNTREND:
            result.primary_target = "Next lower low (trend continuation)"
        elif inputs.is_extreme_zone:
            result.primary_target = "Body of the range / lower demand zone"
        else:
            result.primary_target = "Range Low or next structural low"

    result.entry_note = (
        f"{result.trade_bias.value} — TCT schematic confirmed inside "
        f"{inputs.zone_direction.value}. Target: {result.primary_target}."
    )
    result.passed_phases.append(
        f"Phase 8: Entry conditions met — {result.trade_bias.value}"
    )


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
