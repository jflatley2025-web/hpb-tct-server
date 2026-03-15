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


class FillState(Enum):
    UNFILLED = "Unfilled — gap intact"
    PARTIAL = "Partially filled — some price action inside gap"
    FILLED = "Fully filled — gap completely closed"


class ZonePriority(Enum):
    EXTREME = "Extreme zone — last remaining before range High/Low (highest priority)"
    MULTI_TF_CONFLUENCE = "Multi-TF confluence — higher TF + lower TF OB both unmitigated"
    STRUCTURE_PLUS_OB = "Structure zone + refined OB (broad + precise)"
    SINGLE_OB = "Single-layer OB — valid but lower confluence"
    SUPPLY_CHAIN_SECOND = "Supply chain 2nd OB — strong reaction expected"


class ZoneLifecycle(Enum):
    """Machine-readable terminal state of the S&D zone evaluation.

    evaluate_sd_zone is a one-shot pipeline — these are terminal exit states,
    not a stateful lifecycle.  Use alongside failed_at_phase / wait_reason for
    the human-readable explanation of *why* the zone reached this state.
    """
    EVALUATING = "Evaluation in progress"
    FAILED = "Zone structurally invalid (context, FVG, or competing zones)"
    RETIRED = "Zone fully mitigated — remove from chart"
    WAITING = "Zone valid but entry conditions not yet met"
    TRIGGERED = "Zone valid and entry confirmed"


class TradeBias(Enum):
    LONG = "Long / Buy — demand zone confirmed"
    SHORT = "Short / Sell — supply zone confirmed"
    WAIT = "Wait — zone not yet reached or conditions not met"


# ──────────────────────────────
# FVG Descriptor
# ──────────────────────────────

@dataclass
class FVGInfo:
    """Pre-computed FVG attributes for Phase 3 validation.

    Analogous to MitigationStatus: the caller computes these from candle data,
    the decision tree validates them.  Convenience constructors are provided for
    callers that only have boolean-level information.
    """
    gap_exists: bool
    gap_size: Optional[float] = None        # price units; None = caller lacks price data
    fill_state: FillState = FillState.UNFILLED
    overlap_ratio: Optional[float] = None   # 0.0–1.0; None = unknown
    candle_span: int = 3                    # standard FVG = 3 candles
    tapped_from_top_down: bool = False

    def __post_init__(self):
        if self.gap_size is not None and self.gap_size < 0:
            raise ValueError(f"gap_size cannot be negative: {self.gap_size}")
        if self.overlap_ratio is not None and not (0.0 <= self.overlap_ratio <= 1.0):
            raise ValueError(f"overlap_ratio must be in [0, 1]: {self.overlap_ratio}")
        if self.candle_span < 1:
            raise ValueError(f"candle_span must be >= 1: {self.candle_span}")

    @classmethod
    def confirmed(cls, tapped_from_top_down: bool = False, **kwargs) -> "FVGInfo":
        """Convenience: FVG confirmed with boolean-only info (sensible defaults)."""
        return cls(gap_exists=True, tapped_from_top_down=tapped_from_top_down, **kwargs)

    @classmethod
    def absent(cls) -> "FVGInfo":
        """Convenience: no FVG detected."""
        return cls(gap_exists=False)


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

    # Phase 3 — FVG descriptor (replaces former fvg_confirmed / fvg_tapped booleans).
    # Callers that only have boolean-level info can use FVGInfo.confirmed() or
    # FVGInfo.absent().  Callers with OHLC data should populate gap_size,
    # fill_state, overlap_ratio, and candle_span for full Phase 3 validation.
    fvg_info: FVGInfo

    # Phase 4
    # NOTE: Numeric zone boundaries (zone_top, zone_bottom, or a zone_bounds
    # dataclass) are intentionally deferred.  SDZoneInputs does not yet carry
    # OHLC candle arrays or price data, so any numeric boundary fields would
    # always be None today.  Phase 4 uses draw_note (prose) for boundary
    # descriptions.  When candle/price data is added to SDZoneInputs, introduce
    # a ZoneBounds dataclass here (top: float, bottom: float, candle_indices,
    # boundary_source) and have phase4_draw_zone populate it deterministically.
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
    lifecycle_state: ZoneLifecycle = ZoneLifecycle.EVALUATING
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
    # Human-readable explanations — lifecycle_state is the machine-readable
    # companion.  Check lifecycle_state for programmatic branching; use these
    # strings for logging / display.
    failed_at_phase: Optional[str] = None
    wait_reason: Optional[str] = None


# ──────────────────────────────
# Phase Functions
# ──────────────────────────────

def phase1_build_context(inputs: SDZoneInputs, result: SDZoneEvaluation) -> bool:
    if inputs.market_context == MarketContext.UNCLEAR:
        result.lifecycle_state = ZoneLifecycle.FAILED
        result.failed_at_phase = "Phase 1: Market context unclear — wait for structure."
        return False

    if not inputs.context_reason_exists:
        result.lifecycle_state = ZoneLifecycle.FAILED
        result.failed_at_phase = "Phase 1: No structural reason — do not place zones."
        return False

    context_map = {
        MarketContext.RANGE: "Range: supply upper / demand lower",
        MarketContext.UPTREND: "Uptrend: demand zones at higher-low pivots",
        MarketContext.DOWNTREND: "Downtrend: supply zones at lower-high pivots",
    }
    # Flag counter-trend pairs so downstream can handle them explicitly.
    # Counter-trend zones are not rejected (they exist at structure) but should
    # not be treated identically to with-trend zones.
    counter_trend = (
        (inputs.market_context == MarketContext.UPTREND
         and inputs.zone_direction == ZoneDirection.SUPPLY)
        or
        (inputs.market_context == MarketContext.DOWNTREND
         and inputs.zone_direction == ZoneDirection.DEMAND)
    )
    if counter_trend:
        result.warnings.append(
            f"Phase 1: Counter-trend zone — {inputs.zone_direction.name} "
            f"in {inputs.market_context.name}. Valid at structure but use "
            "reduced size or tighter targets."
        )

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
    fvg = inputs.fvg_info

    if not fvg.gap_exists:
        result.fvg_valid = False
        result.lifecycle_state = ZoneLifecycle.FAILED
        result.failed_at_phase = "Phase 3: No FVG detected — invalid zone."
        return False

    # Reject micro-gaps when gap_size is provided.
    if fvg.gap_size is not None and fvg.gap_size <= 0:
        result.fvg_valid = False
        result.lifecycle_state = ZoneLifecycle.FAILED
        result.failed_at_phase = (
            f"Phase 3: Micro FVG rejected — gap_size {fvg.gap_size} is not positive."
        )
        return False

    # Reject fully filled gaps — no remaining inefficiency.
    if fvg.fill_state == FillState.FILLED:
        result.fvg_valid = False
        result.lifecycle_state = ZoneLifecycle.FAILED
        result.failed_at_phase = "Phase 3: FVG fully filled — gap no longer valid."
        return False

    # Reject complete overlap (wicks connect, no real gap).
    if fvg.overlap_ratio is not None and fvg.overlap_ratio >= 1.0:
        result.fvg_valid = False
        result.lifecycle_state = ZoneLifecycle.FAILED
        result.failed_at_phase = (
            f"Phase 3: FVG overlap ratio {fvg.overlap_ratio:.0%} — "
            "wicks connect, no effective gap."
        )
        return False

    # Standard FVG requires at least 3 candles.
    if fvg.candle_span < 3:
        result.fvg_valid = False
        result.lifecycle_state = ZoneLifecycle.FAILED
        result.failed_at_phase = (
            f"Phase 3: FVG candle span {fvg.candle_span} too short — minimum 3 required."
        )
        return False

    # All checks passed — FVG is valid.
    result.fvg_valid = True
    detail_parts: list[str] = []
    if fvg.tapped_from_top_down:
        detail_parts.append("tapped from top down")
    if fvg.fill_state == FillState.PARTIAL:
        detail_parts.append("partially filled")
    if fvg.gap_size is not None:
        detail_parts.append(f"gap size: {fvg.gap_size}")
    detail = f" ({', '.join(detail_parts)})" if detail_parts else ""
    result.passed_phases.append(f"Phase 3: FVG confirmed — valid inefficiency{detail}.")
    return True


def phase4_draw_zone(inputs: SDZoneInputs, result: SDZoneEvaluation):
    # Zone boundaries are expressed as human-readable prose in draw_note,
    # branched by zone_type (ORDER_BLOCK vs STRUCTURE_ZONE) and direction.
    # A machine-readable zone_bounds field (top/bottom floats, candle indices,
    # timestamps) should be added once SDZoneInputs carries OHLC candle arrays
    # — until then there is no numeric data to populate such a field.
    if inputs.zone_type == ZoneType.ORDER_BLOCK:
        if inputs.zone_direction == ZoneDirection.SUPPLY:
            if inputs.adjacent_candle_has_more_extreme_wick:
                result.draw_note = (
                    "SUPPLY OB: Extend top to adjacent candle wick high (most extreme). "
                    "Single-candle OB boundary."
                )
            else:
                result.draw_note = (
                    "SUPPLY OB: Box from wick low to wick high of last bullish candle "
                    "before bearish expansion."
                )
        else:
            if inputs.adjacent_candle_has_more_extreme_wick:
                result.draw_note = (
                    "DEMAND OB: Extend bottom to adjacent candle wick low (most extreme). "
                    "Single-candle OB boundary."
                )
            else:
                result.draw_note = (
                    "DEMAND OB: Box from wick low to wick high of last bearish candle "
                    "before bullish expansion."
                )
    else:
        # STRUCTURE_ZONE: multi-candle zone — boundaries span the full
        # structure leg, not a single candle.
        if inputs.zone_direction == ZoneDirection.SUPPLY:
            if inputs.adjacent_candle_has_more_extreme_wick:
                result.draw_note = (
                    "SUPPLY STRUCTURE ZONE: Box spans multiple candles in the structure leg. "
                    "Extend top to the most extreme wick high across the structure candles."
                )
            else:
                result.draw_note = (
                    "SUPPLY STRUCTURE ZONE: Box from the lowest wick low to the highest "
                    "wick high across all candles in the structure leg before expansion."
                )
        else:
            if inputs.adjacent_candle_has_more_extreme_wick:
                result.draw_note = (
                    "DEMAND STRUCTURE ZONE: Box spans multiple candles in the structure leg. "
                    "Extend bottom to the most extreme wick low across the structure candles."
                )
            else:
                result.draw_note = (
                    "DEMAND STRUCTURE ZONE: Box from the lowest wick low to the highest "
                    "wick high across all candles in the structure leg before expansion."
                )

    result.passed_phases.append(f"Phase 4: Drawing note — {result.draw_note}")


def phase5_check_mitigation(inputs: SDZoneInputs, result: SDZoneEvaluation) -> bool:
    result.mitigation_status = inputs.mitigation_status

    if inputs.mitigation_status == MitigationStatus.FULL_MITIGATION:
        result.lifecycle_state = ZoneLifecycle.RETIRED
        result.failed_at_phase = "Phase 5: FULLY MITIGATED — remove zone."
        return False

    if inputs.mitigation_status == MitigationStatus.MINOR_MITIGATION:
        if not inputs.is_only_zone_in_area:
            result.lifecycle_state = ZoneLifecycle.FAILED
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
        f"Phase 5: Mitigation check passed — {result.mitigation_status.value}"
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
    elif (
        inputs.refined_ob_found_on_lower_tf
        and inputs.zone_type == ZoneType.STRUCTURE_ZONE
    ):
        result.priority = ZonePriority.STRUCTURE_PLUS_OB
        result.passed_phases.append(
            "Phase 6: Structure zone with refined OB on lower TF — place both zones."
        )
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
        result.lifecycle_state = ZoneLifecycle.WAITING
        result.trade_bias = TradeBias.WAIT
        result.wait_reason = "Phase 8: Price not yet inside zone — zone is your POI, wait for arrival."
        return

    if not inputs.tct_schematic_confirmed:
        result.lifecycle_state = ZoneLifecycle.WAITING
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

    result.lifecycle_state = ZoneLifecycle.TRIGGERED
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
    print(f"  Lifecycle:       {result.lifecycle_state.value}")
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
        fvg_info=FVGInfo.confirmed(),
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
        fvg_info=FVGInfo.confirmed(),
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
        fvg_info=FVGInfo.absent(),
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
