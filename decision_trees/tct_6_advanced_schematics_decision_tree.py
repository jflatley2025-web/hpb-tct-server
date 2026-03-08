"""
TCT Lecture 6 — Advanced TCT Schematics: Decision Tree (Python)

Four advanced concepts layered on top of Lectures 5A/5B:

  Concept 1: Schematic Flip — distributions turning into accumulations (and vice versa).
             Flip-aware targeting when both range sides deviated before confirmation.

  Concept 2: LTF → HTF Escalation — low-TF ranges growing into high-TF ranges.
             When compressed tap spacing or 4+ taps signal you are on the wrong TF.

  Concept 3: Wyckoff-in-Wyckoff — a local TCT schematic forming inside Tap 3 of the
             main schematic. Dramatically improves R:R (2R → 9–21R in real examples).

  Concept 4: Model 1 → Model 2 Upgrade — price curves back after M1 confirmation,
             meets M2 Tap 3 requirements. Stop trail + add-on protocol doubles R:R.
             High-quality M1→M2 = follow-through beyond Range High/Low.

Usage:
    from decision_trees.tct_6_advanced_schematics_decision_tree import (
        evaluate_schematic_flip,
        evaluate_ltf_htf_escalation,
        evaluate_wyckoff_in_wyckoff,
        evaluate_m1_to_m2_upgrade,
    )
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ──────────────────────────────────────────────────────────
# Shared enumerations
# ──────────────────────────────────────────────────────────

class SchematicDirection(Enum):
    ACCUMULATION = "Accumulation / Re-accumulation"
    DISTRIBUTION = "Distribution / Re-distribution"


class ModelType(Enum):
    MODEL_1 = "Model 1"
    MODEL_2 = "Model 2"


class OutcomeStatus(Enum):
    HOLD_TRADE          = "Hold — conditions not yet met for action"
    VALID_ENTRY         = "Valid entry — all conditions met"
    FLIP_EXIT_AND_ENTER = "Flip — exit current position, enter opposing schematic"
    WAIT                = "Wait — monitoring, no action yet"
    NOT_APPLICABLE      = "Not applicable — concept does not apply to this setup"
    SKIP                = "Skip — setup invalid for this concept"


# ══════════════════════════════════════════════════════════
# CONCEPT 1 — Schematic Flip
# ══════════════════════════════════════════════════════════

@dataclass
class FlipInputs:
    """Inputs for Concept 1: Schematic Flip evaluation."""
    active_schematic_direction: SchematicDirection   # The schematic you are already in
    opposite_extreme_deviated_before_confirm: bool   # Was the OPPOSITE range extreme deviated before your BOS?
    opposing_bos_confirmed: bool                     # Has structure broken in the OPPOSITE direction?
    supply_demand_at_opposite_extreme: bool          # Is there visible S/D at the opposite extreme?


@dataclass
class FlipEvaluation:
    status: OutcomeStatus = OutcomeStatus.HOLD_TRADE
    opposing_direction: Optional[SchematicDirection] = None
    notes: list = field(default_factory=list)
    warnings: list = field(default_factory=list)


def evaluate_schematic_flip(inputs: FlipInputs) -> FlipEvaluation:
    """
    Concept 1: Evaluate whether the active schematic is flipping into an opposing one.

    Called DURING an active trade to decide whether to exit + reverse.
    """
    result = FlipEvaluation()
    result.opposing_direction = (
        SchematicDirection.ACCUMULATION
        if inputs.active_schematic_direction == SchematicDirection.DISTRIBUTION
        else SchematicDirection.DISTRIBUTION
    )

    if not inputs.opposite_extreme_deviated_before_confirm:
        result.status = OutcomeStatus.HOLD_TRADE
        result.notes.append(
            "Flip-aware mode OFF: Opposite extreme was NOT deviated before your schematic confirmed. "
            "Target the full Wyckoff High/Low freely. No flip risk during this trade."
        )
        return result

    # Flip-aware mode: opposite was already deviated
    result.warnings.append(
        "FLIP-AWARE MODE: Opposite range extreme was deviated before your schematic confirmed. "
        "Do NOT target beyond the opposite range extreme blindly. "
        "Watch for opposing BOS as price approaches the opposite extreme."
    )

    if inputs.supply_demand_at_opposite_extreme:
        result.notes.append(
            "Supply/demand zone identified at the opposite extreme — pre-marked as potential "
            f"Model 2 Tap 3 zone for the opposing {result.opposing_direction.value} schematic."
        )

    if inputs.opposing_bos_confirmed:
        result.status = OutcomeStatus.FLIP_EXIT_AND_ENTER
        result.notes.append(
            f"OPPOSING BOS CONFIRMED → Risk-Off on {inputs.active_schematic_direction.value}. "
            f"Risk-On for new {result.opposing_direction.value} schematic. "
            "Exit current position. Enter the opposing schematic on this BOS."
        )
    else:
        result.status = OutcomeStatus.HOLD_TRADE
        result.notes.append(
            "No opposing BOS yet. Continue holding current trade toward target. "
            "Keep monitoring for opposing structure break."
        )

    return result


# ══════════════════════════════════════════════════════════
# CONCEPT 2 — LTF → HTF Escalation
# ══════════════════════════════════════════════════════════

@dataclass
class EscalationInputs:
    """Inputs for Concept 2: LTF → HTF Escalation."""
    tap12_distance: float            # Price distance between Tap 1 and Tap 2
    tap23_distance: float            # Price distance between Tap 2 and Tap 3
    tap_count: int                   # How many taps are you counting? (3 is normal; 4+ = escalation signal)
    taps_merge_on_higher_tf: bool    # Do Tap 2 & Tap 3 disappear/merge on the next higher TF?
    htf_supply_demand_zone_found: bool   # Is a proportionally-sized HTF S/D zone found for HTF Tap 3?
    current_tf_label: str            # e.g. "5min", "15min", "1H"
    next_tf_label: str               # e.g. "15min", "1H", "4H"
    unmitigated_supply_above: bool   # (Distribution) Is there unmitigated supply above the LTF range?


@dataclass
class EscalationEvaluation:
    status: OutcomeStatus = OutcomeStatus.HOLD_TRADE
    action: str = ""
    notes: list = field(default_factory=list)
    warnings: list = field(default_factory=list)


def evaluate_ltf_htf_escalation(inputs: EscalationInputs) -> EscalationEvaluation:
    """
    Concept 2: Determine whether a LTF schematic is nested inside a forming HTF range.
    """
    result = EscalationEvaluation()

    # Triggers
    compressed = (
        inputs.tap23_distance < inputs.tap12_distance * 0.5
    )
    too_many_taps = inputs.tap_count >= 4

    if not compressed and not too_many_taps:
        result.status = OutcomeStatus.NOT_APPLICABLE
        result.action = "No escalation signal. Tap spacing is normal. Proceed with standard 5A/5B evaluation."
        return result

    # Escalation signal detected
    if compressed:
        result.warnings.append(
            f"Tap 2–3 gap ({inputs.tap23_distance:.4f}) is less than 50% of Tap 1–2 gap "
            f"({inputs.tap12_distance:.4f}). Compressed tap spacing — LTF/HTF escalation check needed."
        )
    if too_many_taps:
        result.warnings.append(
            f"Counting {inputs.tap_count} taps — more than the 3-tap limit. "
            "Recheck on a higher TF: this is almost certainly a 3-tap schematic on the higher TF."
        )

    if not inputs.taps_merge_on_higher_tf:
        result.status = OutcomeStatus.HOLD_TRADE
        result.action = (
            f"Taps do NOT merge on {inputs.next_tf_label}. "
            "This is a legitimate LTF schematic — no HTF escalation needed. "
            "Apply 5A/5B rules normally on the current TF."
        )
        result.notes.append(
            "Even though tap spacing is compressed, the taps remain distinct on the next TF. "
            "Trade the LTF schematic, but be aware it may be part of a forming HTF range."
        )
        return result

    # Taps merge → HTF range
    result.notes.append(
        f"Tap 2 and Tap 3 merge into a single deviation on {inputs.next_tf_label}. "
        f"LTF ({inputs.current_tf_label}) schematic is nested inside a forming {inputs.next_tf_label} range. "
        "The LTF schematic is just one leg of the HTF range."
    )

    if inputs.unmitigated_supply_above:
        result.notes.append(
            "Unmitigated supply zone above the LTF range — further confirms that price will return "
            "to create the HTF Tap 3. Do not expect strong LTF follow-through downward."
        )

    if inputs.htf_supply_demand_zone_found:
        result.status = OutcomeStatus.VALID_ENTRY
        result.action = (
            f"HTF ({inputs.next_tf_label}) schematic identified. "
            "A proportionally-sized HTF S/D zone exists as the HTF Tap 3 validator. "
            "Apply standard 5A/5B evaluation on the HTF range. "
            "Enter on the HTF BOS (or use overlapping structure for better R:R)."
        )
    else:
        result.status = OutcomeStatus.WAIT
        result.action = (
            f"HTF ({inputs.next_tf_label}) range identified but no proportional HTF S/D zone "
            "found yet for Tap 3 validation. Continue scanning for the HTF Tap 3 zone. "
            "Do not force a Tap 3 without the proportional supply/demand."
        )

    return result


# ══════════════════════════════════════════════════════════
# CONCEPT 3 — Wyckoff-in-Wyckoff
# ══════════════════════════════════════════════════════════

@dataclass
class WyckoffInWyckoffInputs:
    """Inputs for Concept 3: Wyckoff-in-Wyckoff evaluation."""
    main_schematic_direction: SchematicDirection
    price_approaching_tap3: bool             # Is price currently moving toward Tap 3?
    local_range_visible: bool                # Is a local range forming in the Tap 3 leg?
    local_tap2_valid: bool                   # Local Tap 2: within local DL2, accepted back inside
    local_tap3_valid: bool                   # Local Tap 3: meets local extreme liq or S/D
    local_bos_confirmed: bool                # Has the local schematic's BOS been confirmed?
    local_bos_inside_main_range: bool        # Is the local BOS inside the main schematic's range values?
    main_tap3_sl_level: float                # Price level of the main Tap 3 extreme (for SL placement)
    main_target_level: float                 # Price level of the main Wyckoff High/Low (target)


@dataclass
class WyckoffInWyckoffEvaluation:
    status: OutcomeStatus = OutcomeStatus.HOLD_TRADE
    entry_note: str = ""
    stop_loss_note: str = ""
    target_note: str = ""
    estimated_rr_note: str = ""
    notes: list = field(default_factory=list)
    warnings: list = field(default_factory=list)


def evaluate_wyckoff_in_wyckoff(inputs: WyckoffInWyckoffInputs) -> WyckoffInWyckoffEvaluation:
    """
    Concept 3: Evaluate whether a Wyckoff-in-Wyckoff entry is available.

    Returns entry details if local schematic is confirmed and valid.
    """
    result = WyckoffInWyckoffEvaluation()

    if not inputs.price_approaching_tap3:
        result.status = OutcomeStatus.NOT_APPLICABLE
        result.notes.append("Price is not yet approaching Tap 3. Wyckoff-in-Wyckoff not yet applicable.")
        return result

    if not inputs.local_range_visible:
        result.status = OutcomeStatus.NOT_APPLICABLE
        result.notes.append(
            "No local range visible in the Tap 3 leg. "
            "Wyckoff-in-Wyckoff not available. Fall back to standard main BOS entry (5A/5B)."
        )
        return result

    if not inputs.local_tap2_valid:
        result.status = OutcomeStatus.WAIT
        result.notes.append("Local range visible but Tap 2 not yet validated. Wait.")
        return result

    if not inputs.local_tap3_valid:
        result.status = OutcomeStatus.WAIT
        result.notes.append(
            "Local Tap 2 valid but Tap 3 does not meet local extreme liq or S/D requirements. "
            "Wait for a valid local Tap 3 before looking for the local BOS."
        )
        return result

    if not inputs.local_bos_confirmed:
        result.status = OutcomeStatus.WAIT
        result.notes.append(
            "Local schematic Tap 3 valid. Waiting for local BOS confirmation. "
            "Do NOT enter on anticipation. 'Trade confirmations, not expectations.'"
        )
        return result

    # Local BOS confirmed
    direction_adj = "Long" if inputs.main_schematic_direction == SchematicDirection.ACCUMULATION else "Short"

    result.status = OutcomeStatus.VALID_ENTRY
    result.entry_note = (
        f"Enter {direction_adj} on local BOS confirmation. "
        "This local BOS simultaneously confirms the main Tap 3 has been exhausted."
    )
    result.stop_loss_note = (
        f"SL at MAIN schematic Tap 3 extreme: {inputs.main_tap3_sl_level:.4f}. "
        "Do NOT use the local schematic's stop — the main Tap 3 is the invalidation."
    )
    result.target_note = (
        f"Target: MAIN Wyckoff High/Low at {inputs.main_target_level:.4f}. "
        "Same target as if you had waited for the main BOS — just a much better R:R."
    )

    if not inputs.local_bos_inside_main_range:
        result.warnings.append(
            "Local BOS is outside the main range values — riskier entry. "
            "Try stepping down one more TF to find a BOS inside the main range values."
        )
    else:
        result.notes.append("Local BOS is inside the main range values — highest confidence Wyckoff-in-Wyckoff entry.")

    result.estimated_rr_note = (
        "R:R will be dramatically higher than the main BOS R:R. "
        "Real examples: 2.2R → 9R, 1.46R → 13R, 2R → 21R (same SL and target)."
    )

    return result


# ══════════════════════════════════════════════════════════
# CONCEPT 4 — Model 1 → Model 2 Upgrade
# ══════════════════════════════════════════════════════════

@dataclass
class M1ToM2Inputs:
    """Inputs for Concept 4: Model 1 → Model 2 upgrade evaluation."""
    active_direction: SchematicDirection     # Direction of the active M1 trade
    in_m1_trade: bool                        # Currently in a confirmed M1 position
    original_entry_price: float              # Price at which M1 trade was entered
    m1_tap3_extreme: float                   # M1 Tap 3 extreme (stop loss level)
    price_curving_back: bool                 # Is price reversing back toward range extreme?
    pullback_meets_extreme_liq: bool         # Pullback met extreme liquidity requirement
    pullback_meets_extreme_sd: bool          # Pullback met extreme demand/supply requirement
    m2_bos_confirmed: bool                   # Has the Model 2 BOS been confirmed?
    m2_tap3_level: float                     # Price level of the new M2 Tap 3 extreme
    htf_also_shows_m2: bool                  # On higher TF, M1+M2 taps merge into one clean HTF M2
    high_quality_setup: bool                 # Obvious S/D + liq + clean structure = extend target?


@dataclass
class M1ToM2Evaluation:
    status: OutcomeStatus = OutcomeStatus.HOLD_TRADE
    trade_action: str = ""
    position1_management: str = ""
    position2_action: str = ""
    target_extension: str = ""
    notes: list = field(default_factory=list)
    warnings: list = field(default_factory=list)


def evaluate_m1_to_m2_upgrade(inputs: M1ToM2Inputs) -> M1ToM2Evaluation:
    """
    Concept 4: Evaluate whether an active Model 1 trade is upgrading to a Model 2,
    and determine the stop trail + add-on position management protocol.
    """
    result = M1ToM2Evaluation()

    if not inputs.in_m1_trade:
        result.status = OutcomeStatus.NOT_APPLICABLE
        result.notes.append("Not currently in a Model 1 trade. Concept 4 not applicable.")
        return result

    if not inputs.price_curving_back:
        result.status = OutcomeStatus.HOLD_TRADE
        result.trade_action = "Price moving cleanly toward target. Hold M1 position normally. No upgrade occurring."
        return result

    # Price curving back — check M2 Tap 3 requirements
    req_met = inputs.pullback_meets_extreme_liq or inputs.pullback_meets_extreme_sd
    if not req_met:
        result.status = OutcomeStatus.HOLD_TRADE
        result.trade_action = (
            "Pullback does NOT meet extreme liq or extreme S/D requirements. "
            "This is NOT a valid M2 Tap 3. Do not adjust the trade. "
            "Hold with original stop below M1 Tap 3 extreme."
        )
        result.warnings.append(
            "If price takes out the M1 Tap 3 extreme, the trade is invalidated. "
            "Do not average down without valid M2 Tap 3 requirements."
        )
        return result

    # Requirements met — potential M2 Tap 3
    result.notes.append(
        "M2 Tap 3 requirements met: "
        + ("extreme liquidity grabbed" if inputs.pullback_meets_extreme_liq else "")
        + (" + " if inputs.pullback_meets_extreme_liq and inputs.pullback_meets_extreme_sd else "")
        + ("extreme demand/supply mitigated" if inputs.pullback_meets_extreme_sd else "")
        + ". Watching for M2 BOS confirmation."
    )

    if not inputs.m2_bos_confirmed:
        result.status = OutcomeStatus.WAIT
        result.trade_action = (
            "M2 Tap 3 requirements met but BOS not yet confirmed. "
            "Hold M1 position. Do NOT add or trail stop yet. Wait for M2 BOS."
        )
        return result

    # M2 BOS confirmed — execute management protocol
    is_acc = inputs.active_direction == SchematicDirection.ACCUMULATION
    new_tap3_above_entry = (
        (is_acc and inputs.m2_tap3_level > inputs.original_entry_price) or
        (not is_acc and inputs.m2_tap3_level < inputs.original_entry_price)
    )

    result.status = OutcomeStatus.VALID_ENTRY
    result.trade_action = (
        f"Model 2 BOS confirmed. M1 trade is upgrading to Model 2. "
        f"New Wyckoff Point (new Tap 3): {inputs.m2_tap3_level:.4f}."
    )

    if new_tap3_above_entry:
        # New Tap 3 is above original entry (for Acc) → stop goes to entry
        result.position1_management = (
            f"Position 1 (existing M1): Trail stop loss to entry ({inputs.original_entry_price:.4f}). "
            "New Tap 3 is above your entry — you are now de-risked (stop = entry = breakeven). "
            "Freed-up risk = original % risk - 0 (if stop at entry = no remaining risk)."
        )
    else:
        result.position1_management = (
            f"Position 1 (existing M1): Tighten stop to below new M2 Tap 3 ({inputs.m2_tap3_level:.4f}). "
            "New Tap 3 is NOT above/below original entry — cannot fully de-risk, "
            "but stop is now tighter. Freed-up risk = reduction in stop distance."
        )

    result.position2_action = (
        f"Position 2 (new add-on): Enter {'Long' if is_acc else 'Short'} on M2 BOS confirmation. "
        f"Risk = the freed-up % from the stop trail on Position 1. "
        f"SL below new M2 Tap 3 extreme ({inputs.m2_tap3_level:.4f}). "
        "Target: same Wyckoff High/Low as Position 1 (or extended if high quality)."
    )

    if inputs.htf_also_shows_m2:
        result.notes.append(
            "HTF perspective: on the higher TF, M1 Tap 2 + Tap 3 merge into a single deviation "
            "→ the HTF view is a clean Model 2. This confirms the upgrade and further supports "
            "follow-through expectations."
        )

    if inputs.high_quality_setup:
        result.target_extension = (
            "HIGH QUALITY M1 → M2 UPGRADE: Expect follow-through BEYOND the standard Range High/Low. "
            "MM has fully positioned (grabbed liq on M1 twice + M2 Tap 3 liq/demand). "
            "Extended target options: liquidity curve, unfilled inefficiency above/below Range High/Low, "
            "next significant HTF supply/demand zone. "
            "Scale out position partially at Range High/Low, trail remainder toward extended target."
        )
    else:
        result.target_extension = (
            "Standard quality M1 → M2: Take profit at Range High/Low as normal. "
            "Do not force an extended target without clear confluence beyond the range extreme."
        )

    return result


# ──────────────────────────────────────────────────────────
# Pretty-print helpers
# ──────────────────────────────────────────────────────────

def _print_section(label: str, items: list, prefix: str = ""):
    if items:
        print(f"\n  {label}:")
        for item in items:
            print(f"    {prefix} {item}")


def print_flip(result: FlipEvaluation, label: str = ""):
    print(f"\n{'='*66}")
    if label: print(f"  {label}")
    print("  CONCEPT 1 — SCHEMATIC FLIP")
    print(f"  Status: {result.status.value}")
    if result.opposing_direction:
        print(f"  Opposing direction if flip: {result.opposing_direction.value}")
    _print_section("Notes", result.notes, "→")
    _print_section("Warnings", result.warnings, "⚠")
    print(f"{'='*66}")


def print_escalation(result: EscalationEvaluation, label: str = ""):
    print(f"\n{'='*66}")
    if label: print(f"  {label}")
    print("  CONCEPT 2 — LTF → HTF ESCALATION")
    print(f"  Status: {result.status.value}")
    if result.action: print(f"  Action: {result.action}")
    _print_section("Notes", result.notes, "→")
    _print_section("Warnings", result.warnings, "⚠")
    print(f"{'='*66}")


def print_wiw(result: WyckoffInWyckoffEvaluation, label: str = ""):
    print(f"\n{'='*66}")
    if label: print(f"  {label}")
    print("  CONCEPT 3 — WYCKOFF-IN-WYCKOFF")
    print(f"  Status: {result.status.value}")
    if result.entry_note:     print(f"  Entry:  {result.entry_note}")
    if result.stop_loss_note: print(f"  SL:     {result.stop_loss_note}")
    if result.target_note:    print(f"  Target: {result.target_note}")
    if result.estimated_rr_note: print(f"  R:R:    {result.estimated_rr_note}")
    _print_section("Notes", result.notes, "→")
    _print_section("Warnings", result.warnings, "⚠")
    print(f"{'='*66}")


def print_m1m2(result: M1ToM2Evaluation, label: str = ""):
    print(f"\n{'='*66}")
    if label: print(f"  {label}")
    print("  CONCEPT 4 — MODEL 1 → MODEL 2 UPGRADE")
    print(f"  Status: {result.status.value}")
    if result.trade_action:         print(f"  Trade action: {result.trade_action}")
    if result.position1_management: print(f"  Position 1:   {result.position1_management}")
    if result.position2_action:     print(f"  Position 2:   {result.position2_action}")
    if result.target_extension:     print(f"  Target:       {result.target_extension}")
    _print_section("Notes", result.notes, "→")
    _print_section("Warnings", result.warnings, "⚠")
    print(f"{'='*66}")


# ──────────────────────────────────────────────────────────
# Example usage
# ──────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── Concept 1: S&P 500 — both sides deviated, dist confirmed, acc flips it ──
    print_flip(evaluate_schematic_flip(FlipInputs(
        active_schematic_direction=SchematicDirection.DISTRIBUTION,
        opposite_extreme_deviated_before_confirm=True,   # Range low already deviated
        opposing_bos_confirmed=True,                     # Acc BOS triggered
        supply_demand_at_opposite_extreme=True,          # 6H demand at range low area
    )), "S&P 500 — Model 2 Distribution flips to Accumulation")

    # ── Concept 1: Clean distribution, opposite NOT deviated — hold ──
    print_flip(evaluate_schematic_flip(FlipInputs(
        active_schematic_direction=SchematicDirection.DISTRIBUTION,
        opposite_extreme_deviated_before_confirm=False,
        opposing_bos_confirmed=False,
        supply_demand_at_opposite_extreme=False,
    )), "EUR/USD — Clean Distribution, no flip risk")

    # ── Concept 2: 5min schematic, Tap 2-3 compressed, taps merge on 1H ──
    print_escalation(evaluate_ltf_htf_escalation(EscalationInputs(
        tap12_distance=0.80,
        tap23_distance=0.12,   # Very compressed
        tap_count=3,
        taps_merge_on_higher_tf=True,
        htf_supply_demand_zone_found=True,    # 45min OB found as HTF Tap 3
        current_tf_label="5min",
        next_tf_label="1H",
        unmitigated_supply_above=True,
    )), "ETH — 5min schematic nested inside 1H range")

    # ── Concept 2: Normal tap spacing — no escalation ──
    print_escalation(evaluate_ltf_htf_escalation(EscalationInputs(
        tap12_distance=1.20,
        tap23_distance=0.95,   # Normal spacing
        tap_count=3,
        taps_merge_on_higher_tf=False,
        htf_supply_demand_zone_found=False,
        current_tf_label="45min",
        next_tf_label="1H",
        unmitigated_supply_above=False,
    )), "Normal schematic — no escalation needed")

    # ── Concept 3: EUR/USD Wyckoff-in-Wyckoff, 2.2R → 9R ──
    print_wiw(evaluate_wyckoff_in_wyckoff(WyckoffInWyckoffInputs(
        main_schematic_direction=SchematicDirection.DISTRIBUTION,
        price_approaching_tap3=True,
        local_range_visible=True,
        local_tap2_valid=True,
        local_tap3_valid=True,
        local_bos_confirmed=True,
        local_bos_inside_main_range=True,
        main_tap3_sl_level=1.0950,
        main_target_level=1.0750,
    )), "EUR/USD — Wyckoff-in-Wyckoff Distribution (9R entry)")

    # ── Concept 3: No local range — fall back to main BOS ──
    print_wiw(evaluate_wyckoff_in_wyckoff(WyckoffInWyckoffInputs(
        main_schematic_direction=SchematicDirection.ACCUMULATION,
        price_approaching_tap3=True,
        local_range_visible=False,
        local_tap2_valid=False,
        local_tap3_valid=False,
        local_bos_confirmed=False,
        local_bos_inside_main_range=False,
        main_tap3_sl_level=1.2000,
        main_target_level=1.2300,
    )), "No local range — fall back to main BOS")

    # ── Concept 4: EUR/USD M1 → M2 upgrade (3.6R → 5R+) ──
    print_m1m2(evaluate_m1_to_m2_upgrade(M1ToM2Inputs(
        active_direction=SchematicDirection.ACCUMULATION,
        in_m1_trade=True,
        original_entry_price=1.0820,
        m1_tap3_extreme=1.0780,
        price_curving_back=True,
        pullback_meets_extreme_liq=True,
        pullback_meets_extreme_sd=True,
        m2_bos_confirmed=True,
        m2_tap3_level=1.0835,           # Above original entry → full de-risk
        htf_also_shows_m2=True,
        high_quality_setup=True,
    )), "EUR/USD M1 Accumulation → M2 Upgrade (high quality, extended target)")

    # ── Concept 4: Pullback but no M2 requirements met ──
    print_m1m2(evaluate_m1_to_m2_upgrade(M1ToM2Inputs(
        active_direction=SchematicDirection.ACCUMULATION,
        in_m1_trade=True,
        original_entry_price=1.0820,
        m1_tap3_extreme=1.0780,
        price_curving_back=True,
        pullback_meets_extreme_liq=False,
        pullback_meets_extreme_sd=False,
        m2_bos_confirmed=False,
        m2_tap3_level=1.0800,
        htf_also_shows_m2=False,
        high_quality_setup=False,
    )), "M1 trade — pullback but no M2 requirements met")
