"""
tct_schematics.py — TCT Schematics Detection (Lecture 5A + 5B + 6 Advanced Methodology)
Author: HPB-TCT Dev Team
Date: 2026-01-27

Detects TCT Schematics (Accumulation & Distribution Models 1 and 2) based on:
- TCT 2024 mentorship - Lecture 5A | TCT schematics (core methodology)
- TCT 2024 mentorship - Lecture 5B | TCT schematics (advanced enhancements)
- TCT 2024 mentorship - Lecture 6 | Advanced TCT schematics
- Wyckoff methodology simplified with TCT rules

TCT Schematic Key Concepts (Lecture 5A):
- Model 1: Range → Deviation 1 → Deviation 2 (each lower/higher than previous)
- Model 2: Range → Deviation 1 → Higher Low/Lower High (grabs extreme liquidity OR extreme S/D)
- Three-tap model: Tap1 (range), Tap2 (first deviation), Tap3 (second deviation or HL/LH)
- Entry: Break of structure from highest/lowest point between Tap2 and Tap3
- Target: Opposite range extreme (Wyckoff High/Low)
- Stop Loss: Below/Above Tap3

Advanced Concepts (Lecture 5B):
- Highest timeframe validation: Schematic valid only on TF where 6-candle rule applies to all taps
- Overlapping structure (domino effect): Blue → Red → Black timeframe nesting for R:R optimization
- Supply/demand zone awareness: Never enter inside opposing S/D zones
- R:R calculation and optimization: Minimum 1:2 R:R requirement
- Trendline liquidity detection: Identify trendline sweeps before entry
- Tap spacing validation: Equal distribution between taps indicates healthy range
- Model 2 failure → Model 1 transition: Detect when M2 fails and converts to M1
- Range quality/rationality: Horizontal price action with logical tap distances

Advanced Concepts (Lecture 6):
- Distribution-to-Accumulation conversion: Distribution can turn into accumulation when price
  deviates the low instead of breaking it (and vice versa)
- Dual-side deviation awareness: When both sides already deviated, watch structure for confirmation
  of which direction (risk-on for long = bullish BOS, risk-off for short = bearish BOS)
- LTF-to-HTF range transition: Low timeframe ranges can grow into high timeframe ranges
  (LTF break → MTF structure → HTF range)
- Multi-timeframe schematic validity: Same range can have different schematics on different TFs
  (LTF tap2/tap3 may disappear on HTF, can trade both sequentially)
- WOV-in-WOV (schematic within schematic): Third tap of larger schematic contains its own
  TCT schematic, dramatically improving R:R (e.g., from 2.2 to 9 to 21)
- Model 1 → Model 2 flow: Model 1 confirms, but creates 4th tap (HL/LH) that becomes valid
  3rd tap from Model 2 perspective. Allows trailing stop and adding position.
  Usually indicates strong follow-through beyond range extreme.
- Context-based follow-through prediction: Premium pricing = expect distribution,
  Discount pricing = expect accumulation. Bigger trend affects follow-through.

Terminology:
- Accumulation: Trending down → range forms → breaks upside (reversal)
- Re-accumulation: Trending up → range forms → breaks upside (continuation)
- Distribution: Trending up → range forms → breaks downside (reversal)
- Re-distribution: Trending down → range forms → breaks downside (continuation)
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import logging

from pivot_cache import PivotCache
from range_engine_controller import RangeEngineController
from range_utils import check_equilibrium_touch
from session_manipulation import get_active_session, apply_session_multiplier

logger = logging.getLogger("TCT-Schematics")


class TCTSchematicDetector:
    """
    Main TCT Schematic Detection Engine implementing Lecture 5A + 5B + 6 methodology.

    Pure TCT Methodology (Lecture 5A):
    - Model 1 Accumulation: Tap1 (range low) → Tap2 (deviation lower) → Tap3 (deviation even lower)
    - Model 2 Accumulation: Tap1 (range low) → Tap2 (deviation) → Tap3 (higher low at extreme liq/demand)
    - Model 1 Distribution: Tap1 (range high) → Tap2 (deviation higher) → Tap3 (deviation even higher)
    - Model 2 Distribution: Tap1 (range high) → Tap2 (deviation) → Tap3 (lower high at extreme liq/supply)

    Advanced Features (Lecture 5B):
    - Highest timeframe validation: Validate schematic on correct TF using 6-candle rule
    - Overlapping structure: Detect blue/red/black domino effect for better R:R entries
    - Supply/demand zone awareness: Avoid entries inside opposing zones
    - Trendline liquidity detection: Identify trendline sweeps for additional confluence
    - Tap spacing validation: Ensure equal distribution for healthy ranges
    - Model 2 → Model 1 failure transition detection
    - Range quality scoring based on horizontal price action

    Advanced Features (Lecture 6):
    - Distribution-to-Accumulation conversion: Detect when distribution turns into accumulation
    - Dual-side deviation awareness: Handle ranges with deviations on both sides
    - LTF-to-HTF range transition: Detect low timeframe ranges growing into higher timeframes
    - Multi-TF schematic validity: Same range can have different schematics on different TFs
    - WOV-in-WOV: Third tap contains its own schematic for dramatically improved R:R
    - Model 1 → Model 2 flow: Detect when M1 flows into M2 for position management
    - Context-based follow-through prediction: Use premium/discount to predict direction

    Entry Confirmation:
    - Watch structure from highest/lowest point between Tap2 and Tap3
    - When structure breaks back to bullish/bearish = entry confirmation
    - Preferably BOS inside original range values (safer entry)
    - Never enter inside opposing supply/demand zones
    """

    DEVIATION_LIMIT_PERCENT = 0.30  # TCT: 30% of range size for DL
    SIX_CANDLE_LOOKBACK = 6  # Minimum candles for pivot validation
    MIN_RR_RATIO = 2.0  # Lecture 5B: Minimum 1:2 R:R requirement
    TAP_SPACING_TOLERANCE = 0.25  # 25% tolerance for equal tap spacing
    RANGE_QUALITY_MIN = 0.6  # Minimum range quality score for valid schematic

    # Lecture 6 constants
    PREMIUM_THRESHOLD = 0.75  # Above 75% of range = premium (expect distribution)
    DISCOUNT_THRESHOLD = 0.25  # Below 25% of range = discount (expect accumulation)
    LTF_HTF_SIZE_RATIO = 0.4  # LTF range should be max 40% of HTF range to be nested
    WOV_IN_WOV_MIN_RR_IMPROVEMENT = 2.0  # WOV entry should at least double R:R
    M1_TO_M2_FOLLOW_THROUGH_BONUS = 1.5  # M1→M2 typically extends target by 50%

    def __init__(self, candles: pd.DataFrame, pivot_cache: PivotCache = None,
                 range_engine_mode: str = None):
        """Initialize with candle data and optional centralized pivot cache."""
        self.candles = candles.reset_index(drop=True)
        # Centralized pivot cache — eliminates structural drift
        # Validate injected cache matches our candles; rebuild if stale
        if pivot_cache is not None and len(pivot_cache.candles) != len(self.candles):
            logger.debug("Injected PivotCache length mismatch — rebuilding")
            pivot_cache = None
        self._pivot_cache = pivot_cache or PivotCache(self.candles, lookback=self.SIX_CANDLE_LOOKBACK // 2)
        # Range engine controller with feature flag
        self._range_controller = RangeEngineController(
            mode=range_engine_mode, pivot_cache=self._pivot_cache
        )

    def detect_all_schematics(self, detected_ranges: List[Dict] = None) -> Dict:
        """
        Detect all TCT schematics (accumulation and distribution).

        Args:
            detected_ranges: Optional list of pre-detected ranges to use

        Returns: Dict with accumulation_schematics and distribution_schematics
        """
        if len(self.candles) < 50:
            return {
                "accumulation_schematics": [],
                "distribution_schematics": [],
                "total_schematics": 0,
                "error": "Insufficient data (need 50+ candles)"
            }

        # Detect schematics
        accumulation_schematics = self._detect_accumulation_schematics(detected_ranges)
        distribution_schematics = self._detect_distribution_schematics(detected_ranges)

        # Model 3: Continuation schematics (re-accumulation / re-distribution)
        continuation_schematics = self._detect_continuation_schematics()
        for s in continuation_schematics:
            if s.get("direction") == "bullish":
                accumulation_schematics.append(s)
            else:
                distribution_schematics.append(s)

        return {
            "accumulation_schematics": accumulation_schematics,
            "distribution_schematics": distribution_schematics,
            "total_schematics": len(accumulation_schematics) + len(distribution_schematics),
            "candles_analyzed": len(self.candles),
            "timestamp": datetime.utcnow().isoformat()
        }

    # ================================================================
    # ACCUMULATION SCHEMATIC DETECTION
    # ================================================================

    def _detect_accumulation_schematics(self, detected_ranges: List[Dict] = None) -> List[Dict]:
        """
        Detect TCT accumulation schematics (Model 1 and Model 2).

        TCT Methodology:
        - Range formation with equilibrium touch confirmation
        - Tap1 = Range Low
        - Tap2 = First deviation of range low (must come back inside)
        - Tap3 Model 1 = Second deviation (lower than Tap2)
        - Tap3 Model 2 = Higher low that grabs extreme liquidity OR mitigates extreme demand
        """
        schematics = []

        # Find potential range formations
        if detected_ranges:
            ranges = detected_ranges
        else:
            try:
                ranges = self._range_controller.detect_ranges(
                    self.candles, "accumulation", htf_bias="bearish",
                    pivot_cache=self._pivot_cache,
                )
            except Exception as e:
                logger.debug(f"Range controller failed, falling back to legacy: {e}")
                ranges = self._find_accumulation_ranges()

        # Rank ranges by Tap1 (range_low) path quality — best candidates first
        # Each candidate is scored against its own range context, not a shared ref
        if len(ranges) > 1:
            scored_ranges = []
            for r in ranges:
                candidates = [{"idx": r["range_low_idx"], "price": r["range_low"]}]
                ranked = self._rank_ms_lows_by_path_quality(candidates, r)
                # Ranker returns a reordered list; use relative position as score proxy
                # A single-element list always returns the same item, so we compute
                # the score directly using the same logic as the ranker
                range_size = r.get("range_size", 0)
                if range_size > 0:
                    distance = r["range_low"] - r.get("range_low", 0)
                    demand_zones = self._find_demand_zones_below(r["range_low"], r)
                    significant = [z for z in demand_zones
                                   if (z["top"] - z["bottom"]) > range_size * 0.1]
                    score = 30 if not significant else -len(significant) * 15
                else:
                    score = 0
                scored_ranges.append((score, r))
            scored_ranges.sort(key=lambda x: x[0], reverse=True)
            ranges = [r for _, r in scored_ranges]

        for range_data in ranges:
            try:
                # TCT: Tap1 is the range low
                tap1 = self._create_tab(range_data, "range_low", "tap1_acc")
                if not tap1:
                    continue

                # TCT: Find Tap2 (first deviation of range low)
                tap2 = self._find_accumulation_tap2(range_data, tap1)
                if not tap2:
                    continue

                # TCT: Validate Tap2 came back inside range (deviation rule)
                if not self._validate_deviation_came_back_inside(tap2, range_data, "low"):
                    continue

                # TCT: Try to find Model 1 Tap3 (deviation lower than Tap2)
                tap3_m1 = self._find_accumulation_tap3_model1(range_data, tap1, tap2)

                # TCT: Try to find Model 2 Tap3 (higher low at extreme liquidity/demand)
                tap3_m2 = self._find_accumulation_tap3_model2(range_data, tap1, tap2)

                # Build Model 1 schematic if valid
                if tap3_m1:
                    schematic = self._build_accumulation_schematic(
                        range_data, tap1, tap2, tap3_m1, model_type="Model_1"
                    )
                    if schematic:
                        schematics.append(schematic)

                # Build Model 2 schematic if valid
                if tap3_m2:
                    schematic = self._build_accumulation_schematic(
                        range_data, tap1, tap2, tap3_m2, model_type="Model_2"
                    )
                    if schematic:
                        schematics.append(schematic)

                    # TCT 5B: Check for Model 2 → Model 1 failure transition
                    # "A model two when you have a higher low, and when that fails
                    # to hold, it turns into a model one"
                    tap3_m2_failure = self._detect_model2_to_model1_failure(
                        range_data, tap1, tap2, tap3_m2
                    )
                    if tap3_m2_failure:
                        schematic_m2_to_m1 = self._build_accumulation_schematic(
                            range_data, tap1, tap2, tap3_m2_failure,
                            model_type="Model_1_from_M2_failure"
                        )
                        if schematic_m2_to_m1:
                            # Add metadata about the transition
                            schematic_m2_to_m1["lecture_5b_enhancements"]["m2_to_m1_transition"] = {
                                "original_m2_tap3": tap3_m2,
                                "failure_price": tap3_m2_failure.get("model2_failure_price"),
                                "transition_detected": True
                            }
                            schematics.append(schematic_m2_to_m1)

            except Exception as e:
                logger.warning(f"Error detecting accumulation schematic for range {range_data}: {e}", exc_info=True)
                continue

        # Sort by quality and recency
        schematics.sort(key=lambda x: (x.get("quality_score", 0), x.get("tap3", {}).get("idx", 0)), reverse=True)
        return schematics[:10]

    def _find_accumulation_ranges(self) -> List[Dict]:
        """
        Find potential accumulation ranges (trending down, pull from bottom to top).

        TCT: "When we're trending down, we pull our range from bottom to top"
        """
        ranges = []

        for i in range(10, len(self.candles) - 8):
            # Find potential range low (significant swing low)
            # Tail of -8 (down from -15) gives enough room for Tap2→Tap3→BOS to form
            # while still detecting consolidation zones from the last ~1 week on 1D.
            if not self._is_swing_low(i):
                continue

            range_low = float(self.candles.iloc[i]["low"])
            range_low_idx = i

            # Find potential range high after range low
            for j in range(i + 5, min(i + 50, len(self.candles) - 5)):
                if not self._is_swing_high(j):
                    continue

                range_high = float(self.candles.iloc[j]["high"])
                range_high_idx = j

                if range_high <= range_low * 1.005:  # Need meaningful range
                    continue

                # Calculate range properties
                range_size = range_high - range_low
                equilibrium = (range_high + range_low) / 2
                dl_low = range_low - (range_size * self.DEVIATION_LIMIT_PERCENT)
                dl_high = range_high + (range_size * self.DEVIATION_LIMIT_PERCENT)

                # Check for equilibrium touch (range confirmation)
                eq_touched = self._check_equilibrium_touch(range_low_idx, range_high_idx, equilibrium)

                if eq_touched:
                    ranges.append({
                        "range_high": range_high,
                        "range_low": range_low,
                        "range_high_idx": range_high_idx,
                        "range_low_idx": range_low_idx,
                        "equilibrium": equilibrium,
                        "range_size": range_size,
                        "dl_high": dl_high,
                        "dl_low": dl_low,
                        "direction": "accumulation"
                    })

        return ranges

    def _find_accumulation_tap2(self, range_data: Dict, tap1: Dict) -> Optional[Dict]:
        """
        Find Tap2 for accumulation (first deviation of range low).

        TCT: "After confirming your range what you want to see is that first deviation
        of that range low coming back inside your range again"
        TCT: "You do not want to see price close below your DL2"
        """
        range_low = range_data["range_low"]
        range_high_idx = range_data["range_high_idx"]
        dl_low = range_data["dl_low"]

        # Search for deviation after range formation
        start_idx = range_high_idx + 1

        for i in range(start_idx, min(start_idx + 40, len(self.candles) - 3)):
            candle = self.candles.iloc[i]

            # Check if this candle goes below range low (potential deviation)
            if candle["low"] < range_low:
                # TCT: Check if it's a valid deviation (not exceeding DL with close)
                if candle["close"] < dl_low:
                    continue  # Exceeded DL, not a valid deviation

                # Find the low point of this deviation move
                deviation_low = candle["low"]
                deviation_idx = i

                # Look for the actual lowest point of the deviation
                for j in range(i, min(i + 10, len(self.candles))):
                    if self.candles.iloc[j]["low"] < deviation_low:
                        deviation_low = self.candles.iloc[j]["low"]
                        deviation_idx = j
                    # Stop if price starts going back up significantly
                    if self.candles.iloc[j]["close"] > range_low:
                        break

                return {
                    "idx": deviation_idx,
                    "price": float(deviation_low),
                    "time": str(self.candles.iloc[deviation_idx]["open_time"]) if "open_time" in self.candles.columns else str(deviation_idx),
                    "type": "tap2_deviation",
                    "is_deviation": True,
                    "deviation_from_range_low": float(range_low - deviation_low)
                }

        return None

    def _find_accumulation_tap3_model1(self, range_data: Dict, tap1: Dict, tap2: Dict) -> Optional[Dict]:
        """
        Find Tap3 for Model 1 accumulation (second deviation, lower than Tap2).

        TCT: "For a model one is that we take out the low one more time"
        TCT: "After deviation one we extend our range low to the new deviation low"
        TCT: "Each deviation is lower than the previous one"
        """
        tap2_price = tap2["price"]
        tap2_idx = tap2["idx"]
        dl_low = range_data["dl_low"]

        # Search for second deviation after Tap2
        start_idx = tap2_idx + 3

        for i in range(start_idx, min(start_idx + 35, len(self.candles) - 3)):
            candle = self.candles.iloc[i]

            # TCT: Tap3 must go lower than Tap2 (deviation of deviation)
            if candle["low"] < tap2_price:
                # Check if it's a valid deviation (not exceeding DL with close)
                if candle["close"] < dl_low:
                    continue  # Exceeded DL, likely a real break

                # Find the actual lowest point
                deviation_low = candle["low"]
                deviation_idx = i

                for j in range(i, min(i + 8, len(self.candles))):
                    if self.candles.iloc[j]["low"] < deviation_low:
                        deviation_low = self.candles.iloc[j]["low"]
                        deviation_idx = j
                    if self.candles.iloc[j]["close"] > tap2_price:
                        break

                # Validate it came back inside
                came_back = self._validate_deviation_came_back_inside_from_idx(
                    deviation_idx, tap2_price, "low"
                )

                if came_back:
                    return {
                        "idx": deviation_idx,
                        "price": float(deviation_low),
                        "time": str(self.candles.iloc[deviation_idx]["open_time"]) if "open_time" in self.candles.columns else str(deviation_idx),
                        "type": "tap3_model1",
                        "is_deviation": True,
                        "is_lower_than_tap2": True,
                        "deviation_from_tap2": float(tap2_price - deviation_low)
                    }

        return None

    def _find_accumulation_tap3_model2(self, range_data: Dict, tap1: Dict, tap2: Dict) -> Optional[Dict]:
        """
        Find Tap3 for Model 2 accumulation (higher low at extreme liquidity/demand).

        TCT: "For a model two, our third tap is a higher low"
        TCT: "In order for your third tap to be valid it needs to grab extreme liquidity
        and/or it needs to mitigate extreme demand"
        TCT: "Only one is needed to meet the requirement"
        """
        tap2_price = tap2["price"]
        tap2_idx = tap2["idx"]
        range_low = range_data["range_low"]
        range_high = range_data["range_high"]

        # Find extreme liquidity level (first market structure low after Tap2)
        extreme_liquidity = self._find_extreme_liquidity_for_accumulation(tap2_idx, tap2_price)

        # Find extreme demand zone (last demand zone before range low)
        extreme_demand = self._find_extreme_demand(range_data, tap2_idx)

        # Search for higher low that meets Model 2 requirements
        start_idx = tap2_idx + 3

        for i in range(start_idx, min(start_idx + 35, len(self.candles) - 3)):
            if not self._is_swing_low(i):
                continue

            candle = self.candles.iloc[i]
            potential_tap3_price = float(candle["low"])

            # TCT: Must be a higher low (above Tap2)
            if potential_tap3_price <= tap2_price:
                continue

            # TCT: But still below range low (in the deviation area)
            # Allow some tolerance - can be slightly above range low if meeting requirements
            if potential_tap3_price > range_low + (range_data["range_size"] * 0.25):
                continue

            # Check if it meets Model 2 requirements
            grabs_extreme_liquidity = False
            mitigates_extreme_demand = False

            # TCT: Check if it grabs extreme liquidity
            if extreme_liquidity and potential_tap3_price <= extreme_liquidity["price"]:
                grabs_extreme_liquidity = True

            # TCT: Check if it mitigates extreme demand
            if extreme_demand:
                if extreme_demand["bottom"] <= potential_tap3_price <= extreme_demand["top"]:
                    mitigates_extreme_demand = True

            # TCT: "Only one is needed to meet the requirement"
            if grabs_extreme_liquidity or mitigates_extreme_demand:
                return {
                    "idx": i,
                    "price": potential_tap3_price,
                    "time": str(self.candles.iloc[i]["open_time"]) if "open_time" in self.candles.columns else str(i),
                    "type": "tap3_model2",
                    "is_higher_low": True,
                    "grabs_extreme_liquidity": grabs_extreme_liquidity,
                    "mitigates_extreme_demand": mitigates_extreme_demand,
                    "extreme_liquidity": extreme_liquidity,
                    "extreme_demand": extreme_demand
                }

        return None

    # ================================================================
    # DISTRIBUTION SCHEMATIC DETECTION
    # ================================================================

    def _detect_distribution_schematics(self, detected_ranges: List[Dict] = None) -> List[Dict]:
        """
        Detect TCT distribution schematics (Model 1 and Model 2).

        TCT Methodology:
        - Range formation with equilibrium touch confirmation
        - Tap1 = Range High
        - Tap2 = First deviation of range high (must come back inside)
        - Tap3 Model 1 = Second deviation (higher than Tap2)
        - Tap3 Model 2 = Lower high that grabs extreme liquidity OR mitigates extreme supply
        """
        schematics = []

        # Find potential range formations — use L2 range engine when available
        if detected_ranges:
            ranges = detected_ranges
        else:
            try:
                ranges = self._range_controller.detect_ranges(
                    self.candles, "distribution", htf_bias="bullish",
                    pivot_cache=self._pivot_cache,
                )
            except Exception as e:
                logger.debug(f"Range controller failed, falling back to legacy: {e}")
                ranges = self._find_distribution_ranges()

        for range_data in ranges:
            try:
                # TCT: Tap1 is the range high
                tap1 = self._create_tab(range_data, "range_high", "tap1_dist")
                if not tap1:
                    continue

                # TCT: Find Tap2 (first deviation of range high)
                tap2 = self._find_distribution_tap2(range_data, tap1)
                if not tap2:
                    continue

                # TCT: Validate Tap2 came back inside range (deviation rule)
                if not self._validate_deviation_came_back_inside(tap2, range_data, "high"):
                    continue

                # TCT: Try to find Model 1 Tap3 (deviation higher than Tap2)
                tap3_m1 = self._find_distribution_tap3_model1(range_data, tap1, tap2)

                # TCT: Try to find Model 2 Tap3 (lower high at extreme liquidity/supply)
                tap3_m2 = self._find_distribution_tap3_model2(range_data, tap1, tap2)

                # Build Model 1 schematic if valid (sweep gate)
                if tap3_m1:
                    # Liquidity sweep validation: deviation must be a liquidity
                    # grab, not a true break. Must validate BEFORE schematic
                    # construction per correct pipeline ordering.
                    sweep_m1 = self._validate_distribution_sweep(
                        range_data, tap2, tap3_m1
                    )
                    if sweep_m1["has_sweep"] and sweep_m1["classification"] == "true_break":
                        logger.debug(
                            f"Distribution M1 aborted: deviation classified as true_break "
                            f"(swept={sweep_m1['pools_swept']})"
                        )
                    else:
                        schematic = self._build_distribution_schematic(
                            range_data, tap1, tap2, tap3_m1, model_type="Model_1"
                        )
                        if schematic:
                            schematic["sweep_validation"] = sweep_m1
                            schematics.append(schematic)

                # Build Model 2 schematic if valid (sweep gate)
                if tap3_m2:
                    sweep_m2 = self._validate_distribution_sweep(
                        range_data, tap2, tap3_m2
                    )
                    if sweep_m2["has_sweep"] and sweep_m2["classification"] == "true_break":
                        logger.debug(
                            f"Distribution M2 aborted: deviation classified as true_break"
                        )
                    else:
                        schematic = self._build_distribution_schematic(
                            range_data, tap1, tap2, tap3_m2, model_type="Model_2"
                        )
                        if schematic:
                            schematic["sweep_validation"] = sweep_m2
                            schematics.append(schematic)

                    # TCT 5B: Check for Model 2 → Model 1 failure transition
                    # "A model two when you have a lower high, and when that fails
                    # to hold, it turns into a model one"
                    tap3_m2_failure = self._detect_model2_to_model1_failure(
                        range_data, tap1, tap2, tap3_m2, schematic_type="distribution"
                    )
                    if tap3_m2_failure:
                        schematic_m2_to_m1 = self._build_distribution_schematic(
                            range_data, tap1, tap2, tap3_m2_failure,
                            model_type="Model_1_from_M2_failure"
                        )
                        if schematic_m2_to_m1:
                            # Add metadata about the transition
                            schematic_m2_to_m1["lecture_5b_enhancements"]["m2_to_m1_transition"] = {
                                "original_m2_tap3": tap3_m2,
                                "failure_price": tap3_m2_failure.get("model2_failure_price"),
                                "transition_detected": True
                            }
                            schematics.append(schematic_m2_to_m1)

            except Exception as e:
                logger.warning(f"Error detecting distribution schematic for range {range_data}: {e}", exc_info=True)
                continue

        # Sort by quality and recency
        schematics.sort(key=lambda x: (x.get("quality_score", 0), x.get("tap3", {}).get("idx", 0)), reverse=True)
        return schematics[:10]

    # ================================================================
    # MODEL 3: CONTINUATION SCHEMATIC DETECTION
    # Re-accumulation (bullish trend -> consolidation -> break UP)
    # Re-distribution (bearish trend -> consolidation -> break DOWN)
    # ================================================================

    # Continuation ranges are typically tighter and shorter-lived than
    # reversal ranges.  We relax the lookback window slightly so we
    # don't miss valid pullback consolidations within a strong trend.
    CONTINUATION_IMPULSE_MIN_CANDLES = 10   # min candles for impulse leg
    CONTINUATION_IMPULSE_MIN_PCT = 0.02     # 2% min impulse move
    CONTINUATION_RANGE_MAX_CANDLES = 40     # max range duration (candles)

    def _detect_continuation_schematics(self) -> List[Dict]:
        """
        Detect Model 3 continuation schematics.

        Re-accumulation: bullish impulse -> consolidation range ->
                         deviation of range low (demand test) ->
                         break UP (continuation)

        Re-distribution: bearish impulse -> consolidation range ->
                         deviation of range high (supply test) ->
                         break DOWN (continuation)

        Uses the SAME tap structure, BOS confirmation, quality scoring,
        and schematic output format as Model 1/2.  The only difference
        is the required pre-range impulse direction.
        """
        schematics = []

        # --- Re-accumulation (bullish continuation) ---
        re_acc_ranges = self._find_continuation_ranges("bullish")
        logger.debug("[M3] bullish continuation: %d candidate ranges found", len(re_acc_ranges))
        for range_data in re_acc_ranges:
            try:
                tap1 = self._create_tab(range_data, "range_low", "tap1_acc")
                if not tap1:
                    continue

                tap2 = self._find_accumulation_tap2(range_data, tap1)
                if not tap2:
                    continue

                if not self._validate_deviation_came_back_inside(tap2, range_data, "low"):
                    continue

                tap3_m1 = self._find_accumulation_tap3_model1(range_data, tap1, tap2)
                tap3_m2 = self._find_accumulation_tap3_model2(range_data, tap1, tap2)

                for tap3, sub in [(tap3_m1, "a"), (tap3_m2, "b")]:
                    if tap3 is None:
                        continue
                    schematic = self._build_accumulation_schematic(
                        range_data, tap1, tap2, tap3,
                        model_type="Model_3"
                    )
                    if schematic:
                        bos_info = schematic.get("bos_confirmation") or {}
                        logger.debug(
                            "[M3] re_acc schematic built: bos_idx=%s, confirmed=%s, n=%d",
                            bos_info.get("bos_idx"), schematic.get("is_confirmed"), len(self.candles)
                        )
                        schematic["continuation_context"] = {
                            "type": "re_accumulation",
                            "impulse_direction": "bullish",
                            "impulse_pct": range_data.get("impulse_pct", 0),
                        }
                        schematics.append(schematic)

            except Exception as e:
                logger.debug(f"Model 3 re-accumulation error: {e}")
                continue

        # --- Re-distribution (bearish continuation) ---
        re_dist_ranges = self._find_continuation_ranges("bearish")
        logger.debug("[M3] bearish continuation: %d candidate ranges found", len(re_dist_ranges))
        for range_data in re_dist_ranges:
            try:
                tap1 = self._create_tab(range_data, "range_high", "tap1_dist")
                if not tap1:
                    continue

                tap2 = self._find_distribution_tap2(range_data, tap1)
                if not tap2:
                    continue

                if not self._validate_deviation_came_back_inside(tap2, range_data, "high"):
                    continue

                tap3_m1 = self._find_distribution_tap3_model1(range_data, tap1, tap2)
                tap3_m2 = self._find_distribution_tap3_model2(range_data, tap1, tap2)

                for tap3, sub in [(tap3_m1, "a"), (tap3_m2, "b")]:
                    if tap3 is None:
                        continue
                    # Sweep gate: reject true breaks (same as Model_1/2 distribution)
                    sweep = self._validate_distribution_sweep(
                        range_data, tap2, tap3
                    )
                    if sweep["has_sweep"] and sweep["classification"] == "true_break":
                        logger.debug(
                            "Model_3 re-distribution aborted: true_break "
                            "(swept=%s)", sweep.get("pools_swept")
                        )
                        continue
                    schematic = self._build_distribution_schematic(
                        range_data, tap1, tap2, tap3,
                        model_type="Model_3"
                    )
                    if schematic:
                        bos_info = schematic.get("bos_confirmation") or {}
                        logger.debug(
                            "[M3] re_dist schematic built: bos_idx=%s, confirmed=%s, n=%d",
                            bos_info.get("bos_idx"), schematic.get("is_confirmed"), len(self.candles)
                        )
                        schematic["sweep_validation"] = sweep
                        schematic["continuation_context"] = {
                            "type": "re_distribution",
                            "impulse_direction": "bearish",
                            "impulse_pct": range_data.get("impulse_pct", 0),
                        }
                        schematics.append(schematic)

            except Exception as e:
                logger.debug(f"Model 3 re-distribution error: {e}")
                continue

        # Partition by direction, sort/cap each side independently
        sort_key = lambda x: (x.get("quality_score", 0), x.get("tap3", {}).get("idx", 0))
        bullish = sorted([s for s in schematics if s.get("direction") == "bullish"],
                         key=sort_key, reverse=True)[:10]
        bearish = sorted([s for s in schematics if s.get("direction") != "bullish"],
                         key=sort_key, reverse=True)[:10]
        return bullish + bearish

    def _find_continuation_ranges(self, impulse_direction: str,
                                   lookback_limit: int = 80) -> List[Dict]:
        """
        Find consolidation ranges that form AFTER an impulse move.

        For bullish continuation: find ranges preceded by a bullish impulse
        (price moved UP significantly before the range formed).

        For bearish continuation: find ranges preceded by a bearish impulse
        (price moved DOWN significantly before the range formed).

        Returns ranges in the same format as _find_accumulation_ranges /
        _find_distribution_ranges so they can be processed identically.

        lookback_limit: only scan the most recent N candles to avoid
        detecting stale ranges deep in the history window.
        """
        ranges = []
        seen_pairs = set()
        candles = self.candles
        n = len(candles)

        if n < self.CONTINUATION_IMPULSE_MIN_CANDLES + 20:
            return ranges

        # TASK 1: restrict search to recent candles only
        # 200-candle window is ~8 days on 1h — too wide for continuation.
        # 80 candles = ~3.3 days — aligns with active schematic formation.
        scan_start = max(self.CONTINUATION_IMPULSE_MIN_CANDLES + 5,
                         n - lookback_limit)

        # TASK 2: range must complete recently (last 20 candles)
        recent_threshold = n - 20

        for i in range(scan_start, n - 8):
            # --- 1. Detect impulse leg ending near index i ---
            # Look back CONTINUATION_IMPULSE_MIN_CANDLES candles for a
            # directional move of at least CONTINUATION_IMPULSE_MIN_PCT.
            impulse_start = max(0, i - self.CONTINUATION_IMPULSE_MIN_CANDLES - 10)
            impulse_end = i

            if impulse_direction == "bullish":
                low_before = float(candles.iloc[impulse_start:impulse_end]["low"].min())
                high_at = float(candles.iloc[impulse_end]["high"])
                impulse_pct = (high_at - low_before) / low_before if low_before > 0 else 0

                if impulse_pct < self.CONTINUATION_IMPULSE_MIN_PCT:
                    continue

                # Confirm impulse: closes should trend up
                closes = candles.iloc[impulse_start:impulse_end]["close"].values
                if len(closes) >= 4 and float(closes[-1]) <= float(closes[0]):
                    continue
            else:
                high_before = float(candles.iloc[impulse_start:impulse_end]["high"].max())
                low_at = float(candles.iloc[impulse_end]["low"])
                impulse_pct = (high_before - low_at) / high_before if high_before > 0 else 0

                if impulse_pct < self.CONTINUATION_IMPULSE_MIN_PCT:
                    continue

                closes = candles.iloc[impulse_start:impulse_end]["close"].values
                if len(closes) >= 4 and float(closes[-1]) >= float(closes[0]):
                    continue

            # --- 2. Find consolidation range starting near impulse end ---
            # The range should form right after the impulse (within a few candles)
            range_search_start = i
            range_search_end = min(i + self.CONTINUATION_RANGE_MAX_CANDLES, n - 5)

            # For bullish continuation: range forms at the top of the impulse
            # Look for swing high then swing low (range high -> range low)
            if impulse_direction == "bullish":
                # Find range high (near impulse top)
                best_high_idx = None
                best_high = 0
                for j in range(range_search_start, min(range_search_start + 10, range_search_end)):
                    if self._is_swing_high(j):
                        h = float(candles.iloc[j]["high"])
                        if h > best_high:
                            best_high = h
                            best_high_idx = j

                if best_high_idx is None:
                    continue

                # Find range low after the high
                for j in range(best_high_idx + 3, range_search_end):
                    if not self._is_swing_low(j):
                        continue

                    range_low = float(candles.iloc[j]["low"])
                    range_high = best_high
                    if range_high <= range_low * 1.003:
                        continue

                    range_size = range_high - range_low
                    equilibrium = (range_high + range_low) / 2

                    # Check equilibrium touch
                    if not self._check_equilibrium_touch(best_high_idx, j, equilibrium):
                        continue

                    # TASK 2: range must complete recently
                    if j < recent_threshold:
                        logger.debug("[M3] rejected bullish range: range_low_idx=%d < threshold=%d", j, recent_threshold)
                        continue

                    pair = (best_high_idx, j)
                    if pair in seen_pairs:
                        continue
                    seen_pairs.add(pair)

                    logger.debug("[M3] accepted bullish range: high_idx=%d, low_idx=%d, n=%d", best_high_idx, j, n)
                    ranges.append({
                        "range_high": range_high,
                        "range_low": range_low,
                        "range_high_idx": best_high_idx,
                        "range_low_idx": j,
                        "equilibrium": equilibrium,
                        "range_size": range_size,
                        "dl_high": range_high + (range_size * self.DEVIATION_LIMIT_PERCENT),
                        "dl_low": range_low - (range_size * self.DEVIATION_LIMIT_PERCENT),
                        "direction": "accumulation",
                        "impulse_pct": impulse_pct,
                        "is_continuation": True,
                    })
                    break  # one range per impulse

            else:
                # bearish: range forms at the bottom of the impulse
                # Find range low (near impulse bottom)
                best_low_idx = None
                best_low = float("inf")
                for j in range(range_search_start, min(range_search_start + 10, range_search_end)):
                    if self._is_swing_low(j):
                        lo = float(candles.iloc[j]["low"])
                        if lo < best_low:
                            best_low = lo
                            best_low_idx = j

                if best_low_idx is None:
                    continue

                # Find range high after the low
                for j in range(best_low_idx + 3, range_search_end):
                    if not self._is_swing_high(j):
                        continue

                    range_high = float(candles.iloc[j]["high"])
                    range_low = best_low
                    if range_high <= range_low * 1.003:
                        continue

                    range_size = range_high - range_low
                    equilibrium = (range_high + range_low) / 2

                    if not self._check_equilibrium_touch(best_low_idx, j, equilibrium):
                        continue

                    # TASK 2: range must complete recently
                    if j < recent_threshold:
                        logger.debug("[M3] rejected bearish range: range_high_idx=%d < threshold=%d", j, recent_threshold)
                        continue

                    pair = (j, best_low_idx)
                    if pair in seen_pairs:
                        continue
                    seen_pairs.add(pair)

                    logger.debug("[M3] accepted bearish range: low_idx=%d, high_idx=%d, n=%d", best_low_idx, j, n)
                    ranges.append({
                        "range_high": range_high,
                        "range_low": range_low,
                        "range_high_idx": j,
                        "range_low_idx": best_low_idx,
                        "equilibrium": equilibrium,
                        "range_size": range_size,
                        "dl_high": range_high + (range_size * self.DEVIATION_LIMIT_PERCENT),
                        "dl_low": range_low - (range_size * self.DEVIATION_LIMIT_PERCENT),
                        "direction": "distribution",
                        "impulse_pct": impulse_pct,
                        "is_continuation": True,
                    })
                    break

        return ranges

    def _find_distribution_ranges(self) -> List[Dict]:
        """
        Find potential distribution ranges (trending up, pull from top to bottom).

        TCT: "When we're trending up, we pull our range from top to bottom"
        """
        ranges = []

        for i in range(10, len(self.candles) - 8):
            # Find potential range high (significant swing high)
            # Tail of -8 (down from -15) gives enough room for Tap2→Tap3→BOS to form
            # while still detecting consolidation zones from the last ~1 week on 1D.
            if not self._is_swing_high(i):
                continue

            range_high = float(self.candles.iloc[i]["high"])
            range_high_idx = i

            # Find potential range low after range high
            for j in range(i + 5, min(i + 50, len(self.candles) - 5)):
                if not self._is_swing_low(j):
                    continue

                range_low = float(self.candles.iloc[j]["low"])
                range_low_idx = j

                if range_high <= range_low * 1.005:  # Need meaningful range
                    continue

                # Calculate range properties
                range_size = range_high - range_low
                equilibrium = (range_high + range_low) / 2
                dl_low = range_low - (range_size * self.DEVIATION_LIMIT_PERCENT)
                dl_high = range_high + (range_size * self.DEVIATION_LIMIT_PERCENT)

                # Check for equilibrium touch (range confirmation)
                eq_touched = self._check_equilibrium_touch(range_high_idx, range_low_idx, equilibrium)

                if eq_touched:
                    ranges.append({
                        "range_high": range_high,
                        "range_low": range_low,
                        "range_high_idx": range_high_idx,
                        "range_low_idx": range_low_idx,
                        "equilibrium": equilibrium,
                        "range_size": range_size,
                        "dl_high": dl_high,
                        "dl_low": dl_low,
                        "direction": "distribution"
                    })

        return ranges

    def _find_distribution_tap2(self, range_data: Dict, tap1: Dict) -> Optional[Dict]:
        """
        Find Tap2 for distribution (first deviation of range high).

        TCT: "We come back towards that equilibrium of the range confirming the range
        followed by a valid deviation one using our DL2"
        """
        range_high = range_data["range_high"]
        range_low_idx = range_data["range_low_idx"]
        dl_high = range_data["dl_high"]

        # Search for deviation after range formation
        start_idx = range_low_idx + 1

        for i in range(start_idx, min(start_idx + 40, len(self.candles) - 3)):
            candle = self.candles.iloc[i]

            # Check if this candle goes above range high (potential deviation)
            if candle["high"] > range_high:
                # TCT: Check if it's a valid deviation (not exceeding DL with close)
                if candle["close"] > dl_high:
                    continue  # Exceeded DL, not a valid deviation

                # Find the high point of this deviation move
                deviation_high = candle["high"]
                deviation_idx = i

                # Look for the actual highest point of the deviation
                for j in range(i, min(i + 10, len(self.candles))):
                    if self.candles.iloc[j]["high"] > deviation_high:
                        deviation_high = self.candles.iloc[j]["high"]
                        deviation_idx = j
                    # Stop if price starts going back down significantly
                    if self.candles.iloc[j]["close"] < range_high:
                        break

                return {
                    "idx": deviation_idx,
                    "price": float(deviation_high),
                    "time": str(self.candles.iloc[deviation_idx]["open_time"]) if "open_time" in self.candles.columns else str(deviation_idx),
                    "type": "tap2_deviation",
                    "is_deviation": True,
                    "deviation_from_range_high": float(deviation_high - range_high)
                }

        return None

    def _find_distribution_tap3_model1(self, range_data: Dict, tap1: Dict, tap2: Dict) -> Optional[Dict]:
        """
        Find Tap3 for Model 1 distribution (second deviation, higher than Tap2).

        TCT: "Range High with two deviations where each deviation is higher than the previous one"
        """
        tap2_price = tap2["price"]
        tap2_idx = tap2["idx"]
        dl_high = range_data["dl_high"]

        # Search for second deviation after Tap2
        start_idx = tap2_idx + 3

        for i in range(start_idx, min(start_idx + 35, len(self.candles) - 3)):
            candle = self.candles.iloc[i]

            # TCT: Tap3 must go higher than Tap2 (deviation of deviation)
            if candle["high"] > tap2_price:
                # Check if it's a valid deviation (not exceeding DL with close)
                if candle["close"] > dl_high:
                    continue  # Exceeded DL, likely a real break

                # Find the actual highest point
                deviation_high = candle["high"]
                deviation_idx = i

                for j in range(i, min(i + 8, len(self.candles))):
                    if self.candles.iloc[j]["high"] > deviation_high:
                        deviation_high = self.candles.iloc[j]["high"]
                        deviation_idx = j
                    if self.candles.iloc[j]["close"] < tap2_price:
                        break

                # Validate it came back inside
                came_back = self._validate_deviation_came_back_inside_from_idx(
                    deviation_idx, tap2_price, "high"
                )

                if came_back:
                    return {
                        "idx": deviation_idx,
                        "price": float(deviation_high),
                        "time": str(self.candles.iloc[deviation_idx]["open_time"]) if "open_time" in self.candles.columns else str(deviation_idx),
                        "type": "tap3_model1",
                        "is_deviation": True,
                        "is_higher_than_tap2": True,
                        "deviation_from_tap2": float(deviation_high - tap2_price)
                    }

        return None

    def _find_distribution_tap3_model2(self, range_data: Dict, tap1: Dict, tap2: Dict) -> Optional[Dict]:
        """
        Find Tap3 for Model 2 distribution (lower high at extreme liquidity/supply).

        TCT: "For a model 2 TCT distribution schematic our third tap is a lower high"
        TCT: "That third tap is going to be a lower high that grabs either one or both"
        """
        tap2_price = tap2["price"]
        tap2_idx = tap2["idx"]
        range_high = range_data["range_high"]
        range_low = range_data["range_low"]

        # Find extreme liquidity level (first market structure high after Tap2)
        extreme_liquidity = self._find_extreme_liquidity_for_distribution(tap2_idx, tap2_price)

        # Find extreme supply zone (last supply zone before range high)
        extreme_supply = self._find_extreme_supply(range_data, tap2_idx)

        # Search for lower high that meets Model 2 requirements
        start_idx = tap2_idx + 3

        for i in range(start_idx, min(start_idx + 35, len(self.candles) - 3)):
            if not self._is_swing_high(i):
                continue

            candle = self.candles.iloc[i]
            potential_tap3_price = float(candle["high"])

            # TCT: Must be a lower high (below Tap2)
            if potential_tap3_price >= tap2_price:
                continue

            # TCT: But still above range high (in the deviation area)
            # Allow some tolerance - can be slightly below if meeting requirements
            if potential_tap3_price < range_high - (range_data["range_size"] * 0.25):
                continue

            # Check if it meets Model 2 requirements
            grabs_extreme_liquidity = False
            mitigates_extreme_supply = False

            # TCT: Check if it grabs extreme liquidity
            if extreme_liquidity and potential_tap3_price >= extreme_liquidity["price"]:
                grabs_extreme_liquidity = True

            # TCT: Check if it mitigates extreme supply
            if extreme_supply:
                if extreme_supply["bottom"] <= potential_tap3_price <= extreme_supply["top"]:
                    mitigates_extreme_supply = True

            # TCT: "Only one is needed to meet the requirement"
            if grabs_extreme_liquidity or mitigates_extreme_supply:
                return {
                    "idx": i,
                    "price": potential_tap3_price,
                    "time": str(self.candles.iloc[i]["open_time"]) if "open_time" in self.candles.columns else str(i),
                    "type": "tap3_model2",
                    "is_lower_high": True,
                    "grabs_extreme_liquidity": grabs_extreme_liquidity,
                    "mitigates_extreme_supply": mitigates_extreme_supply,
                    "extreme_liquidity": extreme_liquidity,
                    "extreme_supply": extreme_supply
                }

        return None

    # ================================================================
    # BREAK OF STRUCTURE CONFIRMATION
    # ================================================================

    def _detect_bos_confirmation(self, tap2: Dict, tap3: Dict, schematic_type: str,
                                  range_data: Optional[Dict] = None) -> Optional[Dict]:
        """
        Detect break of structure confirmation for entry.

        TCT: "To confirm a TCT model one accumulation schematic we need to watch our
        Market structure from the highest point between tap two and tap three down
        towards our third tap low"

        TCT: "When that downwards Market structure breaks back to bullish after
        deviating that second Tap low that is when we confirm our TCT model"

        Uses LTF (lookback=1) internal market structure for BOS detection so that
        the entry confirms in the discount/premium zone rather than near EQ.
        """
        tap2_idx = tap2["idx"]
        tap3_idx = tap3["idx"]
        tap3_price = tap3["price"]

        if tap3_idx >= len(self.candles) - 3:
            return None

        # Extract equilibrium for LTF filtering (keeps BOS away from EQ)
        equilibrium = range_data.get("equilibrium") if range_data else None

        if "Accumulation" in schematic_type:
            # Find highest point between Tap2 and Tap3
            range_candles = self.candles.iloc[tap2_idx:tap3_idx + 1]
            highest_point_idx = range_candles["high"].idxmax()
            highest_point_price = float(self.candles.iloc[highest_point_idx]["high"])

            # Watch LTF structure from highest point to Tap3 low
            # TCT: Look for break back to bullish on internal structure
            # EQ filter preserved: BOS must confirm below EQ (in the discount zone)
            bos = self._find_bullish_bos(tap3_idx, highest_point_price, tap3_price,
                                          equilibrium=equilibrium,
                                          range_data=range_data)

            if bos:
                return {
                    "type": "bullish_bos",
                    "highest_point_between_tabs": {
                        "idx": int(highest_point_idx),
                        "price": highest_point_price
                    },
                    "bos_idx": bos["idx"],
                    "bos_price": bos["price"],
                    "is_inside_range": bos.get("is_inside_range", False),
                    "confirmed": True
                }

        elif "Distribution" in schematic_type:
            # Find lowest point between Tap2 and Tap3
            range_candles = self.candles.iloc[tap2_idx:tap3_idx + 1]
            lowest_point_idx = range_candles["low"].idxmin()
            lowest_point_price = float(self.candles.iloc[lowest_point_idx]["low"])

            # Watch LTF structure from lowest point to Tap3 high
            # TCT: Look for break back to bearish on internal structure
            bos = self._find_bearish_bos(tap3_idx, lowest_point_price, tap3_price,
                                          equilibrium=equilibrium,
                                          range_data=range_data)

            if bos:
                return {
                    "type": "bearish_bos",
                    "lowest_point_between_tabs": {
                        "idx": int(lowest_point_idx),
                        "price": lowest_point_price
                    },
                    "bos_idx": bos["idx"],
                    "bos_price": bos["price"],
                    "is_inside_range": bos.get("is_inside_range", False),
                    "confirmed": True
                }

        return None

    def _find_bullish_bos(self, start_idx: int, high_price: float, low_price: float,
                           equilibrium: float = None, window: int = 25,
                           range_data: Optional[Dict] = None) -> Optional[Dict]:
        """
        Find bullish break of structure after Tap3 using LTF internal structure.

        TCT Lecture 1: BOS is confirmed when candle CLOSE breaks above the
        previous MS high (not wick). Preferably inside original range values.

        Uses lookback=1 for LTF swing detection so the BOS reference point
        is a small internal swing near Tap3 (discount zone), not a larger
        HTF swing near EQ that eats into profit.
        """
        # Collect swing highs after Tap3 using lookback=1 for LTF
        # internal market structure (finer-grained pivots)
        swing_highs = []
        for i in range(start_idx + 1, min(start_idx + window, len(self.candles) - 1)):
            if self._is_swing_high(i, lookback=1):
                sh_price = float(self.candles.iloc[i]["high"])
                # Filter: for accumulation BOS, prefer swing highs below EQ
                # so the entry confirms in the discount zone
                if equilibrium is not None and sh_price > equilibrium:
                    continue
                swing_highs.append({
                    "idx": i,
                    "price": sh_price
                })

        # Sort by price ascending — prefer the lowest swing high first
        # so BOS confirms at a lower price (more room to target)
        swing_highs.sort(key=lambda s: s["price"])

        if not swing_highs:
            # Fallback: use lookback=2 if no LTF swings found
            for i in range(start_idx + 1, min(start_idx + window, len(self.candles) - 2)):
                if self._is_swing_high(i, lookback=2):
                    sh_price = float(self.candles.iloc[i]["high"])
                    if equilibrium is not None and sh_price > equilibrium:
                        continue
                    swing_highs.append({
                        "idx": i,
                        "price": sh_price
                    })
            swing_highs.sort(key=lambda s: s["price"])

        # Issue 4: No last-resort fallback. If no valid LTF swing highs passed the EQ
        # filter, there is no confirmed BOS — return None rather than guess with a
        # distant historical swing that produces entries far from the discount zone.

        # Supply-path ranking: rank MS high candidates by path quality to range high.
        # Prefer MS highs with a clean path (no significant supply above).
        if range_data and swing_highs:
            swing_highs = self._rank_ms_highs_by_path_quality(swing_highs, range_data)

        # Try each swing high (best path quality first) — BOS = first broken MS level
        for sh in swing_highs:
            # BOS must confirm AFTER tap3, never before the deviation completes
            search_start = max(sh["idx"] + 1, start_idx + 1)
            for i in range(search_start, min(start_idx + window + 10, len(self.candles))):
                close_price = float(self.candles.iloc[i]["close"])
                # BOS close must break the swing high AND the entry level
                # (swing high) must be above tap3 low — otherwise stop > entry.
                # TCT: "enter on the break" = the MS level that was broken.
                if close_price > sh["price"] and sh["price"] > low_price:
                    is_inside_range = sh["price"] < high_price
                    return {
                        "idx": i,
                        "price": sh["price"],
                        "confirmation_close": close_price,
                        "is_inside_range": is_inside_range,
                        "prev_swing_high": sh,
                        "bos_method": "candle_close"
                    }

        return None

    def _find_bearish_bos(self, start_idx: int, low_price: float, high_price: float,
                           equilibrium: float = None, window: int = 25,
                           range_data: Optional[Dict] = None) -> Optional[Dict]:
        """
        Find bearish break of structure after Tap3 using LTF internal structure.

        TCT Lecture 1: BOS is confirmed when candle CLOSE breaks below the
        previous MS low (not wick). Preferably inside original range values.

        Uses lookback=1 for LTF swing detection so the BOS reference point
        is a small internal swing near Tap3 (premium zone), not a larger
        HTF swing near EQ that eats into profit.
        """
        # Collect swing lows after Tap3 using lookback=1 for LTF
        # internal market structure (finer-grained pivots)
        swing_lows = []
        for i in range(start_idx + 1, min(start_idx + window, len(self.candles) - 1)):
            if self._is_swing_low(i, lookback=1):
                sl_price = float(self.candles.iloc[i]["low"])
                # Filter: for distribution BOS, prefer swing lows above EQ
                # so the entry confirms in the premium zone
                if equilibrium is not None and sl_price < equilibrium:
                    continue
                swing_lows.append({
                    "idx": i,
                    "price": sl_price
                })

        # Sort by price descending — prefer the highest swing low first
        # so BOS confirms at a higher price (more room to target)
        swing_lows.sort(key=lambda s: s["price"], reverse=True)

        if not swing_lows:
            # Fallback: use lookback=2 if no LTF swings found
            for i in range(start_idx + 1, min(start_idx + window, len(self.candles) - 2)):
                if self._is_swing_low(i, lookback=2):
                    sl_price = float(self.candles.iloc[i]["low"])
                    if equilibrium is not None and sl_price < equilibrium:
                        continue
                    swing_lows.append({
                        "idx": i,
                        "price": sl_price
                    })
            swing_lows.sort(key=lambda s: s["price"], reverse=True)

        # Issue 4: No last-resort fallback. If no valid LTF swing lows passed the EQ
        # filter, there is no confirmed BOS — return None rather than guess with a
        # distant historical swing that produces entries far from the premium zone.

        # Demand-path ranking: rank MS low candidates by path quality to range low.
        # Prefer MS lows with a clean path (no significant demand underneath).
        if range_data and swing_lows:
            swing_lows = self._rank_ms_lows_by_path_quality(swing_lows, range_data)

        # Try each swing low (best path quality first) — BOS = first broken MS level
        for sl in swing_lows:
            # BOS must confirm AFTER tap3, never before the deviation completes
            search_start = max(sl["idx"] + 1, start_idx + 1)
            for i in range(search_start, min(start_idx + window + 10, len(self.candles))):
                close_price = float(self.candles.iloc[i]["close"])
                # BOS close must break the swing low AND the entry level
                # (swing low) must be below tap3 high — otherwise stop < entry.
                # TCT: "enter on the break" = the MS level that was broken.
                if close_price < sl["price"] and sl["price"] < high_price and sl["price"] > low_price:
                    is_inside_range = sl["price"] > low_price and sl["price"] < high_price
                    return {
                        "idx": i,
                        "price": sl["price"],
                        "confirmation_close": close_price,
                        "is_inside_range": is_inside_range,
                        "prev_swing_low": sl,
                        "bos_method": "candle_close"
                    }

        return None

    def _rank_ms_lows_by_path_quality(self, swing_lows: List[Dict],
                                       range_data: Dict) -> List[Dict]:
        """
        Rank MS low candidates by path quality to range low.

        TCT: "The correct market structure low must be the one that leads
        to a smooth path toward the range low (no significant demand underneath)."

        Scoring factors:
        - distance_to_range_low: More room = higher score
        - demand_zone_strength: Fewer/smaller demand zones = higher score
        - clean_path_score: No significant demand = bonus
        """
        range_low = range_data["range_low"]
        range_size = range_data["range_size"]

        if range_size <= 0:
            return swing_lows

        scored = []
        for sl in swing_lows:
            score = 0.0

            # 1. Distance to range low (more room to target = higher score)
            distance = sl["price"] - range_low
            if distance > 0:
                score += (distance / range_size) * 40  # 0-40 pts

            # 2. Demand zone strength below this MS low
            demand_zones = self._find_demand_zones_below(sl["price"], range_data)
            significant = [z for z in demand_zones
                           if (z["top"] - z["bottom"]) > range_size * 0.1]

            # Penalize per significant demand zone (-15 each)
            score -= len(significant) * 15

            # 3. Clean path bonus (no significant demand = +30)
            if not significant:
                score += 30

            # 4. Proximity to premium zone (higher MS low = better for shorts)
            if sl["price"] > range_data.get("equilibrium", 0):
                score += 10  # Premium zone bonus

            scored.append((sl, score))

        # Sort by score descending (best path quality first)
        scored.sort(key=lambda x: x[1], reverse=True)
        return [s[0] for s in scored]

    def _rank_ms_highs_by_path_quality(self, swing_highs: List[Dict],
                                        range_data: Dict) -> List[Dict]:
        """
        Rank MS high candidates by path quality to range high.

        TCT: "The correct market structure high must be the one that leads
        to a smooth path toward the range high (no significant supply above)."

        Scoring factors:
        - distance_to_range_high: More room = higher score
        - supply_zone_strength: Fewer/smaller supply zones = higher score
        - clean_path_score: No significant supply = bonus
        """
        range_high = range_data["range_high"]
        range_size = range_data["range_size"]

        if range_size <= 0:
            return swing_highs

        scored = []
        for sh in swing_highs:
            score = 0.0

            # 1. Distance to range high (more room to target = higher score)
            distance = range_high - sh["price"]
            if distance > 0:
                score += (distance / range_size) * 40  # 0-40 pts

            # 2. Supply zone strength above this MS high
            supply_zones = self._find_supply_zones_above(sh["price"], range_data)
            significant = [z for z in supply_zones
                           if (z["top"] - z["bottom"]) > range_size * 0.1]

            # Penalize per significant supply zone (-15 each)
            score -= len(significant) * 15

            # 3. Clean path bonus (no significant supply = +30)
            if not significant:
                score += 30

            # 4. Proximity to discount zone (lower MS high = better for longs)
            if sh["price"] < range_data.get("equilibrium", float("inf")):
                score += 10  # Discount zone bonus

            scored.append((sh, score))

        # Sort by score descending (best path quality first)
        scored.sort(key=lambda x: x[1], reverse=True)
        return [s[0] for s in scored]

    # ================================================================
    # EXTREME LIQUIDITY & DEMAND/SUPPLY DETECTION
    # ================================================================

    def _find_extreme_liquidity_for_accumulation(self, tap2_idx: int, tap2_price: float) -> Optional[Dict]:
        """
        Find extreme liquidity for Model 2 accumulation.

        TCT Lecture 1: "Extreme liquidity is the last liquidity point remaining
        before taking your second tap low which is your range low."
        "Often your extreme liquidity will simply be your first confirmed MS low
        if you pull your market structure from the second Tap low up."

        Uses 6-candle rule with inside bar exclusion for swing detection.
        """
        # Find first confirmed MS low after Tap2 (using proper 6CR with inside bars)
        for i in range(tap2_idx + 2, min(tap2_idx + 20, len(self.candles) - 2)):
            if self._is_swing_low(i, lookback=2):
                return {
                    "idx": i,
                    "price": float(self.candles.iloc[i]["low"]),
                    "type": "extreme_liquidity",
                    "description": "First confirmed MS low after Tap2 (6CR validated)"
                }

        return None

    def _find_extreme_liquidity_for_distribution(self, tap2_idx: int, tap2_price: float) -> Optional[Dict]:
        """
        Find extreme liquidity for Model 2 distribution.

        TCT Lecture 1: "First MS high by drawing market structure from the top down
        from your second tap high towards the lowest point between tap two and tap three."

        Uses 6-candle rule with inside bar exclusion for swing detection.
        """
        # Find first confirmed MS high after Tap2 (using proper 6CR with inside bars)
        for i in range(tap2_idx + 2, min(tap2_idx + 20, len(self.candles) - 2)):
            if self._is_swing_high(i, lookback=2):
                return {
                    "idx": i,
                    "price": float(self.candles.iloc[i]["high"]),
                    "type": "extreme_liquidity",
                    "description": "First confirmed MS high after Tap2 (6CR validated)"
                }

        return None

    def _find_extreme_demand(self, range_data: Dict, tap2_idx: int) -> Optional[Dict]:
        """
        Find extreme demand zone for Model 2 accumulation.

        TCT: "Extreme demand is our last demand Zone protecting us from taking
        our deviation low our second tap low"

        TCT: "Preferably you find demand zones that are hovering around that
        extreme discount Zone that last 25% of your range low"
        """
        range_low = range_data["range_low"]
        range_high = range_data["range_high"]
        range_size = range_data["range_size"]

        # Extreme discount is the bottom 25% of the range
        extreme_discount_top = range_low + (range_size * 0.25)

        # Look for order blocks or structure demand in this area
        for i in range(tap2_idx, max(tap2_idx - 30, 0), -1):
            candle = self.candles.iloc[i]

            # Look for bearish candle followed by bullish expansion (potential demand)
            if candle["close"] < candle["open"]:  # Bearish candle
                # Check if in extreme discount zone
                candle_low = float(candle["low"])
                candle_high = float(candle["high"])

                if candle_low < extreme_discount_top and candle_low >= range_low - range_size * 0.3:
                    # Check for expansion after
                    if i + 1 < len(self.candles):
                        next_candle = self.candles.iloc[i + 1]
                        expansion = next_candle["high"] - next_candle["low"]
                        ob_size = candle_high - candle_low

                        if expansion > ob_size * 1.2:  # Some expansion
                            return {
                                "idx": i,
                                "top": candle_high,
                                "bottom": candle_low,
                                "type": "extreme_demand",
                                "description": "Demand zone in extreme discount"
                            }

        return None

    def _find_extreme_supply(self, range_data: Dict, tap2_idx: int) -> Optional[Dict]:
        """
        Find extreme supply zone for Model 2 distribution.

        TCT: "We check where is our extreme order block or our extreme structure
        Supply drawing structure Supply"
        """
        range_low = range_data["range_low"]
        range_high = range_data["range_high"]
        range_size = range_data["range_size"]

        # Extreme premium is the top 25% of the range
        extreme_premium_bottom = range_high - (range_size * 0.25)

        # Look for order blocks or structure supply in this area
        for i in range(tap2_idx, max(tap2_idx - 30, 0), -1):
            candle = self.candles.iloc[i]

            # Look for bullish candle followed by bearish expansion (potential supply)
            if candle["close"] > candle["open"]:  # Bullish candle
                # Check if in extreme premium zone
                candle_low = float(candle["low"])
                candle_high = float(candle["high"])

                if candle_high > extreme_premium_bottom and candle_high <= range_high + range_size * 0.3:
                    # Check for expansion after
                    if i + 1 < len(self.candles):
                        next_candle = self.candles.iloc[i + 1]
                        expansion = next_candle["high"] - next_candle["low"]
                        ob_size = candle_high - candle_low

                        if expansion > ob_size * 1.2:  # Some expansion
                            return {
                                "idx": i,
                                "top": candle_high,
                                "bottom": candle_low,
                                "type": "extreme_supply",
                                "description": "Supply zone in extreme premium"
                            }

        return None

    # ================================================================
    # SCHEMATIC BUILDERS
    # ================================================================

    def _build_accumulation_schematic(self, range_data: Dict, tap1: Dict, tap2: Dict,
                                       tap3: Dict, model_type: str) -> Optional[Dict]:
        """
        Build complete accumulation schematic with entry, stop loss, and target.

        TCT 5A: "You enter on the break you put your stop loss below your third tap low
        and you target the Range High"

        TCT 5B: Enhanced with overlapping structure, S/D awareness, R:R optimization,
        trendline liquidity, tap spacing, and range quality validation.
        """
        schematic_type = f"{model_type}_Accumulation"

        # Detect BOS confirmation (LTF internal structure, filtered by EQ)
        bos = self._detect_bos_confirmation(tap2, tap3, schematic_type, range_data=range_data)

        # Calculate entry, stop loss, target
        entry_price = bos["bos_price"] if bos else None
        # TCT: "Stop loss below your third tap low" — add 2% of range as buffer
        stop_buffer = range_data["range_size"] * 0.02
        stop_loss = tap3["price"] - stop_buffer
        target = range_data["range_high"]  # TCT: "Target the Range High"

        # Safety: for a long (accumulation), entry must be above stop.
        # If not, the BOS is invalid — discard it.
        if entry_price is not None and entry_price <= stop_loss:
            bos = None
            entry_price = None

        # TCT: Entry must be inside the range (≥ range_low).
        # A BOS swing high below range_low means price hasn't re-entered the range yet.
        if entry_price is not None and entry_price < range_data["range_low"]:
            bos = None
            entry_price = None

        # ============================================================
        # LECTURE 5B ENHANCEMENTS
        # ============================================================

        # 1. Highest timeframe validation
        htf_validation = self._validate_highest_timeframe(tap1, tap2, tap3)

        # 2. Overlapping structure (domino effect) for R:R optimization
        overlapping_structure = self._detect_overlapping_structure(
            range_data, tap3, schematic_type
        )

        # 3. Supply/demand zone awareness
        sd_zone_check = None
        if entry_price:
            sd_zone_check = self._check_supply_demand_zone_conflict(
                entry_price, schematic_type, range_data, tap3["idx"]
            )

        # 4. R:R calculation and optimization
        rr_analysis = None
        if entry_price and stop_loss and target:
            rr_analysis = self._calculate_optimized_rr(
                entry_price, stop_loss, target, schematic_type
            )

        # 5. Trendline liquidity detection
        trendline_liq = self._detect_trendline_liquidity(tap3, range_data, schematic_type)

        # 6. Tap spacing validation
        tap_spacing = self._validate_tap_spacing(tap1, tap2, tap3, range_data)

        # 7. Range quality calculation
        range_quality = self._calculate_range_quality(range_data, tap1, tap2, tap3)

        # Calculate risk/reward (standard)
        risk_reward = rr_analysis["risk_reward_ratio"] if rr_analysis else None

        # Use optimized entry if available from overlapping structure
        optimized_entry = None
        if overlapping_structure["has_overlapping_structure"]:
            optimized_entry = {
                "entry": overlapping_structure["optimized_entry"],
                "stop_loss": overlapping_structure["optimized_stop_loss"],
                "target": overlapping_structure["optimized_target"],
                "risk_reward": overlapping_structure["optimized_rr"],
                "domino_levels": overlapping_structure["domino_levels"]
            }

        # Calculate overall quality score (enhanced with 5B factors)
        quality_score = self._calculate_schematic_quality_enhanced(
            range_data, tap1, tap2, tap3, bos, model_type,
            htf_validation, sd_zone_check, rr_analysis, tap_spacing, range_quality
        )

        # Validate six candle rule
        six_candle_valid = htf_validation["all_taps_valid_6cr"]

        # Determine if entry is safe (no S/D conflicts)
        entry_is_safe = not (sd_zone_check and sd_zone_check.get("has_conflict", False))

        # ============================================================
        # LECTURE 6 ENHANCEMENTS: ADVANCED TCT SCHEMATICS
        # ============================================================

        # Build preliminary schematic for Lecture 6 methods that need schematic input
        preliminary_schematic = {
            "schematic_type": schematic_type,
            "direction": "bullish",
            "model": model_type,
            "tap1": tap1,
            "tap2": tap2,
            "tap3": tap3,
            "bos_confirmation": bos,
            "entry": {"price": entry_price},
            "stop_loss": {"price": stop_loss},
            "target": {"price": target}
        }

        # 1. Schematic conversion detection (Distribution converting to Accumulation)
        schematic_conversion = self._detect_schematic_conversion(preliminary_schematic, range_data)

        # 2. Dual-side deviation awareness
        dual_side_deviation = self._detect_dual_side_deviation(
            range_data, tap2, tap3, schematic_type
        )

        # 3. LTF-to-HTF range transition detection
        ltf_htf_transition = self._detect_ltf_to_htf_range_transition(range_data, tap3["idx"])

        # 4. Multi-timeframe schematic validity
        multi_tf_validity = self._detect_multi_tf_schematic_validity(tap1, tap2, tap3, range_data)

        # 5. Enhanced WOV-in-WOV (schematic within schematic)
        wov_in_wov = self._detect_enhanced_wov_in_wov(tap3, range_data, schematic_type)

        # 6. Model 1 to Model 2 flow detection
        m1_to_m2_flow = self._detect_model1_to_model2_flow(preliminary_schematic, range_data)

        # 7. Context-based follow-through prediction
        context_follow_through = self._calculate_context_based_follow_through(
            range_data, schematic_type
        )

        # Session context: prefer BOS/entry timestamp, fall back to Tap3
        chosen_timestamp = None
        if bos and bos.get("bos_idx") is not None:
            try:
                chosen_timestamp = str(self.candles.iloc[bos["bos_idx"]].get("open_time", ""))
            except (IndexError, KeyError):
                pass
        if not chosen_timestamp:
            chosen_timestamp = tap3.get("time")
        session_context = self._get_session_context(chosen_timestamp)

        return {
            "schematic_type": schematic_type,
            "direction": "bullish",
            "model": model_type,
            "tap1": tap1,
            "tap2": tap2,
            "tap3": tap3,
            "range": {
                "high": range_data["range_high"],
                "low": range_data["range_low"],
                "equilibrium": range_data["equilibrium"],
                "size": range_data["range_size"],
                "dl_high": range_data["dl_high"],
                "dl_low": range_data["dl_low"]
            },
            "wyckoff_high": range_data["range_high"],  # TCT: Target
            "wyckoff_low": tap3["price"] if tap3.get("is_deviation") else tap2["price"],
            "bos_confirmation": bos,
            "session_context": session_context,
            "entry": {
                "type": "BOS_confirmation",
                "price": entry_price,
                "description": "Enter on break of structure back to bullish",
                "is_safe": entry_is_safe
            },
            "stop_loss": {
                "price": stop_loss,
                "description": "Below Tap3 low"
            },
            "target": {
                "price": target,
                "description": "Range High (Wyckoff High)"
            },
            "risk_reward": risk_reward,
            "quality_score": quality_score,
            "six_candle_valid": six_candle_valid,
            "is_confirmed": bos is not None and bos.get("confirmed", False),
            # Lecture 5B enhanced fields
            "lecture_5b_enhancements": {
                "htf_validation": htf_validation,
                "overlapping_structure": overlapping_structure,
                "optimized_entry": optimized_entry,
                "supply_demand_check": sd_zone_check,
                "rr_analysis": rr_analysis,
                "trendline_liquidity": trendline_liq,
                "tap_spacing": tap_spacing,
                "range_quality": range_quality,
                "meets_minimum_rr": rr_analysis["meets_minimum_rr"] if rr_analysis else False,
                "has_trendline_confluence": trendline_liq.get("provides_confluence", False)
            },
            # Lecture 6 enhanced fields: Advanced TCT Schematics
            "lecture_6_enhancements": {
                "schematic_conversion": schematic_conversion,
                "dual_side_deviation": dual_side_deviation,
                "ltf_htf_transition": ltf_htf_transition,
                "multi_tf_validity": multi_tf_validity,
                "wov_in_wov": wov_in_wov,
                "model1_to_model2_flow": m1_to_m2_flow,
                "context_follow_through": context_follow_through,
                # Summary flags for quick reference
                "has_conversion": schematic_conversion is not None,
                "has_dual_deviation": dual_side_deviation.get("has_dual_deviation", False),
                "is_nested_in_htf": ltf_htf_transition.get("is_nested_in_htf_range", False),
                "valid_on_htf": multi_tf_validity.get("would_be_valid_on_htf", False),
                "has_wov_opportunity": wov_in_wov.get("has_inner_schematic", False),
                "has_m1_to_m2_opportunity": m1_to_m2_flow.get("m1_to_m2_detected", False),
                "follow_through_bias": context_follow_through.get("bias", "neutral"),
                "enhanced_target": context_follow_through.get("enhanced_target")
            },
            # Session manipulation context (MSCE integration)
            "session_context": session_context,
            "timestamp": datetime.utcnow().isoformat()
        }

    def _build_distribution_schematic(self, range_data: Dict, tap1: Dict, tap2: Dict,
                                       tap3: Dict, model_type: str) -> Optional[Dict]:
        """
        Build complete distribution schematic with entry, stop loss, and target.

        TCT 5A: "Once we do break that we put our stop loss above our Tap high
        and we target the range low"

        TCT 5B: Enhanced with overlapping structure, S/D awareness, R:R optimization,
        trendline liquidity, tap spacing, and range quality validation.
        """
        schematic_type = f"{model_type}_Distribution"

        # Detect BOS confirmation (LTF internal structure, filtered by EQ)
        bos = self._detect_bos_confirmation(tap2, tap3, schematic_type, range_data=range_data)

        # Calculate entry, stop loss, target
        entry_price = bos["bos_price"] if bos else None
        # TCT: "Stop loss above your third tap high" — add 2% of range as buffer
        stop_buffer = range_data["range_size"] * 0.02
        stop_loss = tap3["price"] + stop_buffer
        target = range_data["range_low"]  # TCT: "Target the Range Low"

        # Safety: for a short (distribution), entry must be below stop.
        # If not, the BOS is invalid — discard it.
        if entry_price is not None and entry_price >= stop_loss:
            bos = None
            entry_price = None

        # TCT: Entry must be inside the range (≤ range_high).
        # A BOS swing low above range_high means price hasn't re-entered the range yet.
        if entry_price is not None and entry_price > range_data["range_high"]:
            bos = None
            entry_price = None

        # ============================================================
        # LECTURE 5B ENHANCEMENTS
        # ============================================================

        # 1. Highest timeframe validation
        htf_validation = self._validate_highest_timeframe(tap1, tap2, tap3)

        # 2. Overlapping structure (domino effect) for R:R optimization
        overlapping_structure = self._detect_overlapping_structure(
            range_data, tap3, schematic_type
        )

        # 3. Supply/demand zone awareness
        sd_zone_check = None
        if entry_price:
            sd_zone_check = self._check_supply_demand_zone_conflict(
                entry_price, schematic_type, range_data, tap3["idx"]
            )

        # 4. R:R calculation and optimization
        rr_analysis = None
        if entry_price and stop_loss and target:
            rr_analysis = self._calculate_optimized_rr(
                entry_price, stop_loss, target, schematic_type
            )

        # 5. Trendline liquidity detection
        trendline_liq = self._detect_trendline_liquidity(tap3, range_data, schematic_type)

        # 6. Tap spacing validation
        tap_spacing = self._validate_tap_spacing(tap1, tap2, tap3, range_data)

        # 7. Range quality calculation
        range_quality = self._calculate_range_quality(range_data, tap1, tap2, tap3)

        # Calculate risk/reward (standard)
        risk_reward = rr_analysis["risk_reward_ratio"] if rr_analysis else None

        # Use optimized entry if available from overlapping structure
        optimized_entry = None
        if overlapping_structure["has_overlapping_structure"]:
            optimized_entry = {
                "entry": overlapping_structure["optimized_entry"],
                "stop_loss": overlapping_structure["optimized_stop_loss"],
                "target": overlapping_structure["optimized_target"],
                "risk_reward": overlapping_structure["optimized_rr"],
                "domino_levels": overlapping_structure["domino_levels"]
            }

        # Calculate overall quality score (enhanced with 5B factors)
        quality_score = self._calculate_schematic_quality_enhanced(
            range_data, tap1, tap2, tap3, bos, model_type,
            htf_validation, sd_zone_check, rr_analysis, tap_spacing, range_quality
        )

        # Validate six candle rule
        six_candle_valid = htf_validation["all_taps_valid_6cr"]

        # Determine if entry is safe (no S/D conflicts)
        entry_is_safe = not (sd_zone_check and sd_zone_check.get("has_conflict", False))

        # ============================================================
        # LECTURE 6 ENHANCEMENTS: ADVANCED TCT SCHEMATICS
        # ============================================================

        # Build preliminary schematic for Lecture 6 methods that need schematic input
        preliminary_schematic = {
            "schematic_type": schematic_type,
            "direction": "bearish",
            "model": model_type,
            "tap1": tap1,
            "tap2": tap2,
            "tap3": tap3,
            "bos_confirmation": bos,
            "entry": {"price": entry_price},
            "stop_loss": {"price": stop_loss},
            "target": {"price": target}
        }

        # 1. Schematic conversion detection (Accumulation converting to Distribution)
        schematic_conversion = self._detect_schematic_conversion(preliminary_schematic, range_data)

        # 2. Dual-side deviation awareness
        dual_side_deviation = self._detect_dual_side_deviation(
            range_data, tap2, tap3, schematic_type
        )

        # 3. LTF-to-HTF range transition detection
        ltf_htf_transition = self._detect_ltf_to_htf_range_transition(range_data, tap3["idx"])

        # 4. Multi-timeframe schematic validity
        multi_tf_validity = self._detect_multi_tf_schematic_validity(tap1, tap2, tap3, range_data)

        # 5. Enhanced WOV-in-WOV (schematic within schematic)
        wov_in_wov = self._detect_enhanced_wov_in_wov(tap3, range_data, schematic_type)

        # 6. Model 1 to Model 2 flow detection
        m1_to_m2_flow = self._detect_model1_to_model2_flow(preliminary_schematic, range_data)

        # 7. Context-based follow-through prediction
        context_follow_through = self._calculate_context_based_follow_through(
            range_data, schematic_type
        )

        # Session context: prefer BOS/entry timestamp, fall back to Tap3
        chosen_timestamp = None
        if bos and bos.get("bos_idx") is not None:
            try:
                chosen_timestamp = str(self.candles.iloc[bos["bos_idx"]].get("open_time", ""))
            except (IndexError, KeyError):
                pass
        if not chosen_timestamp:
            chosen_timestamp = tap3.get("time")
        session_context = self._get_session_context(chosen_timestamp)

        return {
            "schematic_type": schematic_type,
            "direction": "bearish",
            "model": model_type,
            "tap1": tap1,
            "tap2": tap2,
            "tap3": tap3,
            "range": {
                "high": range_data["range_high"],
                "low": range_data["range_low"],
                "equilibrium": range_data["equilibrium"],
                "size": range_data["range_size"],
                "dl_high": range_data["dl_high"],
                "dl_low": range_data["dl_low"]
            },
            "wyckoff_high": tap3["price"] if tap3.get("is_deviation") else tap2["price"],
            "wyckoff_low": range_data["range_low"],  # TCT: Target
            "bos_confirmation": bos,
            "session_context": session_context,
            "entry": {
                "type": "BOS_confirmation",
                "price": entry_price,
                "description": "Enter on break of structure back to bearish",
                "is_safe": entry_is_safe
            },
            "stop_loss": {
                "price": stop_loss,
                "description": "Above Tap3 high"
            },
            "target": {
                "price": target,
                "description": "Range Low (Wyckoff Low)"
            },
            "risk_reward": risk_reward,
            "quality_score": quality_score,
            "six_candle_valid": six_candle_valid,
            "is_confirmed": bos is not None and bos.get("confirmed", False),
            # Lecture 5B enhanced fields
            "lecture_5b_enhancements": {
                "htf_validation": htf_validation,
                "overlapping_structure": overlapping_structure,
                "optimized_entry": optimized_entry,
                "supply_demand_check": sd_zone_check,
                "rr_analysis": rr_analysis,
                "trendline_liquidity": trendline_liq,
                "tap_spacing": tap_spacing,
                "range_quality": range_quality,
                "meets_minimum_rr": rr_analysis["meets_minimum_rr"] if rr_analysis else False,
                "has_trendline_confluence": trendline_liq.get("provides_confluence", False)
            },
            # Lecture 6 enhanced fields: Advanced TCT Schematics
            "lecture_6_enhancements": {
                "schematic_conversion": schematic_conversion,
                "dual_side_deviation": dual_side_deviation,
                "ltf_htf_transition": ltf_htf_transition,
                "multi_tf_validity": multi_tf_validity,
                "wov_in_wov": wov_in_wov,
                "model1_to_model2_flow": m1_to_m2_flow,
                "context_follow_through": context_follow_through,
                # Summary flags for quick reference
                "has_conversion": schematic_conversion is not None,
                "has_dual_deviation": dual_side_deviation.get("has_dual_deviation", False),
                "is_nested_in_htf": ltf_htf_transition.get("is_nested_in_htf_range", False),
                "valid_on_htf": multi_tf_validity.get("would_be_valid_on_htf", False),
                "has_wov_opportunity": wov_in_wov.get("has_inner_schematic", False),
                "has_m1_to_m2_opportunity": m1_to_m2_flow.get("m1_to_m2_detected", False),
                "follow_through_bias": context_follow_through.get("bias", "neutral"),
                "enhanced_target": context_follow_through.get("enhanced_target")
            },
            # Session manipulation context (MSCE integration)
            "session_context": session_context,
            "timestamp": datetime.utcnow().isoformat()
        }

    # ================================================================
    # LECTURE 5B: ADVANCED ENHANCEMENTS
    # ================================================================

    def _validate_highest_timeframe(self, tap1: Dict, tap2: Dict, tap3: Dict,
                                     timeframe: str = None) -> Dict:
        """
        Lecture 5B: Validate which timeframe the schematic is valid on.

        TCT 5B: "Your TCT schematic is only valid on a certain time frame if you can
        draw your mark structure on each tap applying the six candle rule"

        TCT 5B: "When your schematic is playing out on one time frame if that schematic
        does not have all taps applying the 6 candle rule on your highest time frame,
        we need to zoom out to find the time frame where all three taps do apply"

        Returns:
            Dict with validity info and recommended timeframe
        """
        result = {
            "all_taps_valid_6cr": False,
            "tap1_valid_6cr": False,
            "tap2_valid_6cr": False,
            "tap3_valid_6cr": False,
            "recommended_tf": None,
            "current_tf": timeframe,
            "validity_explanation": ""
        }

        try:
            # Check each tap applies 6-candle rule
            tap1_valid = self._is_swing_low(tap1["idx"]) or self._is_swing_high(tap1["idx"])
            tap2_valid = self._is_swing_low(tap2["idx"]) or self._is_swing_high(tap2["idx"])
            tap3_valid = self._is_swing_low(tap3["idx"]) or self._is_swing_high(tap3["idx"])

            result["tap1_valid_6cr"] = tap1_valid
            result["tap2_valid_6cr"] = tap2_valid
            result["tap3_valid_6cr"] = tap3_valid
            result["all_taps_valid_6cr"] = tap1_valid and tap2_valid and tap3_valid

            if result["all_taps_valid_6cr"]:
                result["validity_explanation"] = "All taps apply 6-candle rule on current TF"
                result["recommended_tf"] = timeframe
            else:
                invalid_taps = []
                if not tap1_valid:
                    invalid_taps.append("Tap1")
                if not tap2_valid:
                    invalid_taps.append("Tap2")
                if not tap3_valid:
                    invalid_taps.append("Tap3")
                result["validity_explanation"] = f"{', '.join(invalid_taps)} fail 6CR - zoom out to higher TF"
                # Suggest next higher TF
                tf_hierarchy = ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"]
                if timeframe and timeframe in tf_hierarchy:
                    current_idx = tf_hierarchy.index(timeframe)
                    if current_idx < len(tf_hierarchy) - 1:
                        result["recommended_tf"] = tf_hierarchy[current_idx + 1]

        except Exception as e:
            result["validity_explanation"] = f"Validation error: {e}"

        return result

    def _detect_overlapping_structure(self, range_data: Dict, tap3: Dict,
                                       schematic_type: str) -> Dict:
        """
        Lecture 5B: Detect overlapping structure (domino effect) for R:R optimization.

        TCT 5B: "What is overlapping structure? Basically this is a schematic inside
        of a schematic... the blue inside of the red inside of the black"

        TCT 5B: "The black range plays out and we zoom into the red and then inside
        the red we have the blue"

        This creates optimized entries where:
        - Entry happens at blue schematic BOS
        - Stop loss is below blue Tap3
        - Target is the black range high/low (much larger move)

        Returns:
            Dict with overlapping structure info and optimized R:R
        """
        result = {
            "has_overlapping_structure": False,
            "nested_schematics": [],
            "optimized_entry": None,
            "optimized_stop_loss": None,
            "optimized_target": None,
            "optimized_rr": None,
            "domino_levels": 1  # 1 = current only, 2 = has nested, 3 = double nested
        }

        try:
            tap3_idx = tap3["idx"]
            range_low = range_data["range_low"]
            range_high = range_data["range_high"]

            # Look for nested schematic structure after Tap3
            # This is a smaller range that forms within the deviation zone
            nested_range = self._find_nested_schematic_range(tap3_idx, range_data, schematic_type)

            if nested_range:
                result["has_overlapping_structure"] = True
                result["domino_levels"] = 2
                result["nested_schematics"].append({
                    "level": "inner",
                    "range_high": nested_range["range_high"],
                    "range_low": nested_range["range_low"],
                    "is_accumulation": "accumulation" in schematic_type.lower()
                })

                # Calculate optimized R:R using nested schematic entry with outer target
                if "accumulation" in schematic_type.lower():
                    # Entry at nested BOS, stop below nested Tap3, target outer range high
                    optimized_entry = nested_range.get("entry_price", nested_range["range_low"])
                    optimized_stop = nested_range.get("tap3_price", nested_range["range_low"] * 0.995)
                    optimized_target = range_high  # Outer range target

                    risk = optimized_entry - optimized_stop
                    reward = optimized_target - optimized_entry
                    if risk > 0:
                        result["optimized_rr"] = round(reward / risk, 2)
                        result["optimized_entry"] = optimized_entry
                        result["optimized_stop_loss"] = optimized_stop
                        result["optimized_target"] = optimized_target

                else:  # Distribution
                    optimized_entry = nested_range.get("entry_price", nested_range["range_high"])
                    optimized_stop = nested_range.get("tap3_price", nested_range["range_high"] * 1.005)
                    optimized_target = range_low  # Outer range target

                    risk = optimized_stop - optimized_entry
                    reward = optimized_entry - optimized_target
                    if risk > 0:
                        result["optimized_rr"] = round(reward / risk, 2)
                        result["optimized_entry"] = optimized_entry
                        result["optimized_stop_loss"] = optimized_stop
                        result["optimized_target"] = optimized_target

        except Exception as e:
            logger.debug(f"Error detecting overlapping structure: {e}")

        return result

    def _find_nested_schematic_range(self, start_idx: int, outer_range: Dict,
                                      schematic_type: str) -> Optional[Dict]:
        """
        Find a nested schematic range (smaller range forming within outer deviation).

        Lecture 5B: Looking for a range that forms within the deviation zone
        that can provide a more optimized entry point.
        """
        if start_idx >= len(self.candles) - 20:
            return None

        # Define search area based on schematic type
        if "accumulation" in schematic_type.lower():
            # Look for small range forming below outer range low
            search_start = start_idx
            search_end = min(start_idx + 30, len(self.candles) - 5)

            for i in range(search_start, search_end - 10):
                if self._is_swing_low(i):
                    potential_range_low = float(self.candles.iloc[i]["low"])

                    # Look for range high
                    for j in range(i + 3, min(i + 15, search_end)):
                        if self._is_swing_high(j):
                            potential_range_high = float(self.candles.iloc[j]["high"])

                            # Validate it's a smaller nested range
                            nested_size = potential_range_high - potential_range_low
                            outer_size = outer_range["range_size"]

                            # Nested range should be smaller (20-60% of outer)
                            if 0.2 * outer_size <= nested_size <= 0.6 * outer_size:
                                return {
                                    "range_high": potential_range_high,
                                    "range_low": potential_range_low,
                                    "range_size": nested_size,
                                    "range_high_idx": j,
                                    "range_low_idx": i,
                                    "is_nested": True
                                }

        else:  # Distribution
            search_start = start_idx
            search_end = min(start_idx + 30, len(self.candles) - 5)

            for i in range(search_start, search_end - 10):
                if self._is_swing_high(i):
                    potential_range_high = float(self.candles.iloc[i]["high"])

                    for j in range(i + 3, min(i + 15, search_end)):
                        if self._is_swing_low(j):
                            potential_range_low = float(self.candles.iloc[j]["low"])

                            nested_size = potential_range_high - potential_range_low
                            outer_size = outer_range["range_size"]

                            if 0.2 * outer_size <= nested_size <= 0.6 * outer_size:
                                return {
                                    "range_high": potential_range_high,
                                    "range_low": potential_range_low,
                                    "range_size": nested_size,
                                    "range_high_idx": i,
                                    "range_low_idx": j,
                                    "is_nested": True
                                }

        return None

    def _check_supply_demand_zone_conflict(self, entry_price: float, schematic_type: str,
                                            range_data: Dict, tap3_idx: int) -> Dict:
        """
        Lecture 5B: Check for supply/demand zone conflicts before entry.

        TCT 5B: "What we don't want to see on entry is inside the structure demand
        that is in the way of our Target and that we are inside the zone it's a
        no-go basically"

        TCT 5B: "If there's structure demand in the way of our Target we shouldn't
        take the trade"

        Returns:
            Dict with zone conflict info
        """
        result = {
            "has_conflict": False,
            "entry_inside_opposing_zone": False,
            "opposing_zone_blocks_target": False,
            "conflicting_zones": [],
            "recommendation": "clear_entry"
        }

        try:
            if "accumulation" in schematic_type.lower():
                # For long entries, check for supply zones above entry blocking target
                supply_zones = self._find_supply_zones_above(entry_price, range_data)

                for zone in supply_zones:
                    # Check if entry is inside a supply zone (bad)
                    if zone["bottom"] <= entry_price <= zone["top"]:
                        result["has_conflict"] = True
                        result["entry_inside_opposing_zone"] = True
                        result["conflicting_zones"].append(zone)
                        result["recommendation"] = "no_entry_inside_supply"

                    # Check if supply zone blocks path to target
                    if zone["bottom"] < range_data["range_high"]:
                        # Zone is between entry and target
                        result["opposing_zone_blocks_target"] = True
                        result["conflicting_zones"].append(zone)
                        result["recommendation"] = "caution_supply_in_way"

            else:  # Distribution
                # For short entries, check for demand zones below entry blocking target
                demand_zones = self._find_demand_zones_below(entry_price, range_data)

                for zone in demand_zones:
                    # Check if entry is inside a demand zone (bad)
                    if zone["bottom"] <= entry_price <= zone["top"]:
                        result["has_conflict"] = True
                        result["entry_inside_opposing_zone"] = True
                        result["conflicting_zones"].append(zone)
                        result["recommendation"] = "no_entry_inside_demand"

                    # Check if demand zone blocks path to target
                    if zone["top"] > range_data["range_low"]:
                        result["opposing_zone_blocks_target"] = True
                        result["conflicting_zones"].append(zone)
                        result["recommendation"] = "caution_demand_in_way"

        except Exception as e:
            logger.debug(f"Error checking S/D zone conflict: {e}")

        return result

    def _find_supply_zones_above(self, price: float, range_data: Dict) -> List[Dict]:
        """Find supply zones above a given price that could block target."""
        supply_zones = []
        range_high = range_data["range_high"]

        # Look for bearish order blocks above entry price
        for i in range(max(0, range_data.get("range_low_idx", 0) - 30),
                       range_data.get("range_high_idx", len(self.candles)) + 30):
            if i >= len(self.candles) - 1:
                break

            candle = self.candles.iloc[i]

            # Bullish candle followed by bearish move = potential supply
            if candle["close"] > candle["open"]:
                next_candle = self.candles.iloc[i + 1] if i + 1 < len(self.candles) else None
                if next_candle is not None and next_candle["close"] < next_candle["open"]:
                    zone_top = float(candle["high"])
                    zone_bottom = float(candle["low"])

                    # Only include zones above entry price
                    if zone_bottom > price and zone_bottom < range_high:
                        supply_zones.append({
                            "top": zone_top,
                            "bottom": zone_bottom,
                            "idx": i,
                            "type": "supply"
                        })

        return supply_zones[:3]  # Return top 3 most relevant

    def _find_demand_zones_below(self, price: float, range_data: Dict) -> List[Dict]:
        """Find demand zones below a given price that could block target."""
        demand_zones = []
        range_low = range_data["range_low"]

        # Look for bullish order blocks below entry price
        for i in range(max(0, range_data.get("range_low_idx", 0) - 30),
                       range_data.get("range_high_idx", len(self.candles)) + 30):
            if i >= len(self.candles) - 1:
                break

            candle = self.candles.iloc[i]

            # Bearish candle followed by bullish move = potential demand
            if candle["close"] < candle["open"]:
                next_candle = self.candles.iloc[i + 1] if i + 1 < len(self.candles) else None
                if next_candle is not None and next_candle["close"] > next_candle["open"]:
                    zone_top = float(candle["high"])
                    zone_bottom = float(candle["low"])

                    # Only include zones below entry price
                    if zone_top < price and zone_top > range_low:
                        demand_zones.append({
                            "top": zone_top,
                            "bottom": zone_bottom,
                            "idx": i,
                            "type": "demand"
                        })

        return demand_zones[:3]

    def _calculate_optimized_rr(self, entry_price: float, stop_loss: float,
                                 target: float, schematic_type: str) -> Dict:
        """
        Lecture 5B: Calculate R:R and determine if trade meets minimum requirements.

        TCT 5B: "We calculate our R to R ratio before we enter the trade to make
        sure we are getting enough R out of it"

        TCT 5B: Minimum 1:2 R:R is generally required.

        Returns:
            Dict with R:R calculation and optimization suggestions
        """
        result = {
            "risk_reward_ratio": None,
            "meets_minimum_rr": False,
            "risk_amount": None,
            "reward_amount": None,
            "optimization_suggestions": []
        }

        try:
            if "accumulation" in schematic_type.lower():
                risk = entry_price - stop_loss
                reward = target - entry_price
            else:
                risk = stop_loss - entry_price
                reward = entry_price - target

            if risk > 0:
                rr = reward / risk
                result["risk_reward_ratio"] = round(rr, 2)
                result["risk_amount"] = round(risk, 8)
                result["reward_amount"] = round(reward, 8)
                result["meets_minimum_rr"] = rr >= self.MIN_RR_RATIO

                if not result["meets_minimum_rr"]:
                    result["optimization_suggestions"].append(
                        f"R:R of {rr:.2f} below minimum {self.MIN_RR_RATIO}:1"
                    )
                    result["optimization_suggestions"].append(
                        "Look for overlapping structure for better entry"
                    )
                    result["optimization_suggestions"].append(
                        "Wait for nested schematic to improve R:R"
                    )

        except Exception as e:
            logger.debug(f"Error calculating R:R: {e}")

        return result

    def _detect_trendline_liquidity(self, tap3: Dict, range_data: Dict,
                                     schematic_type: str) -> Dict:
        """
        Lecture 5B: Detect trendline liquidity sweeps.

        TCT 5B: "When we have your range drawn what other concepts are happening
        in the background so for example your trendlines"

        TCT 5B: "We're likely going to take out trendline liquidity at a minimum"

        Returns:
            Dict with trendline liquidity info
        """
        result = {
            "has_trendline": False,
            "trendline_swept": False,
            "trendline_price": None,
            "provides_confluence": False
        }

        try:
            tap3_idx = tap3["idx"]
            tap3_price = tap3["price"]

            if "accumulation" in schematic_type.lower():
                # Look for descending trendline that gets swept at Tap3
                trendline = self._find_descending_trendline(tap3_idx, range_data)
                if trendline:
                    result["has_trendline"] = True
                    result["trendline_price"] = trendline["price_at_tap3"]

                    # Check if Tap3 swept the trendline
                    if tap3_price <= trendline["price_at_tap3"]:
                        result["trendline_swept"] = True
                        result["provides_confluence"] = True

            else:  # Distribution
                # Look for ascending trendline that gets swept at Tap3
                trendline = self._find_ascending_trendline(tap3_idx, range_data)
                if trendline:
                    result["has_trendline"] = True
                    result["trendline_price"] = trendline["price_at_tap3"]

                    # Check if Tap3 swept the trendline
                    if tap3_price >= trendline["price_at_tap3"]:
                        result["trendline_swept"] = True
                        result["provides_confluence"] = True

        except Exception as e:
            logger.debug(f"Error detecting trendline liquidity: {e}")

        return result

    def _find_descending_trendline(self, tap3_idx: int, range_data: Dict) -> Optional[Dict]:
        """Find descending trendline from swing highs before the range."""
        swing_highs = []

        # Find swing highs before range formation
        start_idx = max(0, range_data.get("range_low_idx", 0) - 50)
        end_idx = range_data.get("range_low_idx", tap3_idx)

        for i in range(start_idx, end_idx):
            if self._is_swing_high(i, lookback=3):
                swing_highs.append({
                    "idx": i,
                    "price": float(self.candles.iloc[i]["high"])
                })

        if len(swing_highs) >= 2:
            # Use last two swing highs to draw trendline
            sh1, sh2 = swing_highs[-2], swing_highs[-1]

            # Calculate slope
            if sh2["idx"] != sh1["idx"]:
                slope = (sh2["price"] - sh1["price"]) / (sh2["idx"] - sh1["idx"])

                # Project trendline to Tap3 index
                price_at_tap3 = sh2["price"] + slope * (tap3_idx - sh2["idx"])

                return {
                    "start_idx": sh1["idx"],
                    "end_idx": sh2["idx"],
                    "slope": slope,
                    "price_at_tap3": price_at_tap3,
                    "is_descending": slope < 0
                }

        return None

    def _find_ascending_trendline(self, tap3_idx: int, range_data: Dict) -> Optional[Dict]:
        """Find ascending trendline from swing lows before the range."""
        swing_lows = []

        # Find swing lows before range formation
        start_idx = max(0, range_data.get("range_high_idx", 0) - 50)
        end_idx = range_data.get("range_high_idx", tap3_idx)

        for i in range(start_idx, end_idx):
            if self._is_swing_low(i, lookback=3):
                swing_lows.append({
                    "idx": i,
                    "price": float(self.candles.iloc[i]["low"])
                })

        if len(swing_lows) >= 2:
            # Use last two swing lows to draw trendline
            sl1, sl2 = swing_lows[-2], swing_lows[-1]

            if sl2["idx"] != sl1["idx"]:
                slope = (sl2["price"] - sl1["price"]) / (sl2["idx"] - sl1["idx"])

                price_at_tap3 = sl2["price"] + slope * (tap3_idx - sl2["idx"])

                return {
                    "start_idx": sl1["idx"],
                    "end_idx": sl2["idx"],
                    "slope": slope,
                    "price_at_tap3": price_at_tap3,
                    "is_ascending": slope > 0
                }

        return None

    def _validate_tap_spacing(self, tap1: Dict, tap2: Dict, tap3: Dict,
                               range_data: Dict) -> Dict:
        """
        Lecture 5B: Validate equal spacing between taps.

        TCT 5B: "This is also one thing you want to watch as well is that
        there's some equal distribution between your taps"

        TCT 5B: "Range quality, rational and horizontal range with logical
        tap distances"

        Returns:
            Dict with tap spacing validation info
        """
        result = {
            "spacing_valid": False,
            "tap1_to_tap2_candles": None,
            "tap2_to_tap3_candles": None,
            "spacing_ratio": None,
            "is_horizontal": False,
            "spacing_quality": 0.0
        }

        try:
            tap1_idx = tap1["idx"]
            tap2_idx = tap2["idx"]
            tap3_idx = tap3["idx"]

            # Calculate candle distances
            tap1_to_tap2 = tap2_idx - tap1_idx
            tap2_to_tap3 = tap3_idx - tap2_idx

            result["tap1_to_tap2_candles"] = tap1_to_tap2
            result["tap2_to_tap3_candles"] = tap2_to_tap3

            if tap1_to_tap2 > 0 and tap2_to_tap3 > 0:
                # Calculate spacing ratio (ideal is 1.0 = equal spacing)
                ratio = min(tap1_to_tap2, tap2_to_tap3) / max(tap1_to_tap2, tap2_to_tap3)
                result["spacing_ratio"] = round(ratio, 2)

                # Check if within tolerance for equal spacing
                result["spacing_valid"] = ratio >= (1.0 - self.TAP_SPACING_TOLERANCE)

                # Calculate quality score
                result["spacing_quality"] = round(ratio, 2)

            # Check horizontal range quality
            range_size = range_data["range_size"]
            tap_price_range = max(tap1["price"], tap2["price"], tap3["price"]) - \
                             min(tap1["price"], tap2["price"], tap3["price"])

            # Horizontal range means taps are within reasonable price distance
            # relative to range size
            if tap_price_range <= range_size * 0.6:
                result["is_horizontal"] = True

        except Exception as e:
            logger.debug(f"Error validating tap spacing: {e}")

        return result

    def _detect_model2_to_model1_failure(self, range_data: Dict, tap1: Dict, tap2: Dict,
                                          existing_tap3_m2: Optional[Dict],
                                          schematic_type: str = "accumulation") -> Optional[Dict]:
        """
        Lecture 5B: Detect when Model 2 fails and transitions to Model 1.

        TCT 5B: "A model two when you have a higher low, and you would expect
        that third tap to hold being a higher low there when that fails to hold,
        it goes through, it turns into a model one"

        TCT 5B: "This happens all the time, the model two fails and converts
        into a model one at which point you'd take out your stops"

        Returns:
            New Tap3 for Model 1 if Model 2 failed, None otherwise
        """
        if not existing_tap3_m2:
            return None

        try:
            tap3_m2_idx = existing_tap3_m2["idx"]
            tap3_m2_price = existing_tap3_m2["price"]
            tap2_price = tap2["price"]

            if "accumulation" in schematic_type.lower():
                # For accumulation: Look for price breaking below the Model 2 Tap3 (higher low fails)
                for i in range(tap3_m2_idx + 1, min(tap3_m2_idx + 20, len(self.candles) - 2)):
                    candle = self.candles.iloc[i]

                    # Price broke below Model 2 Tap3 - it failed
                    if candle["low"] < tap3_m2_price:
                        # Now find the new Model 1 Tap3 (needs to be lower than Tap2)
                        if candle["low"] < tap2_price:
                            # This is now a Model 1 with successive lower lows
                            deviation_low = candle["low"]
                            deviation_idx = i

                            # Find actual lowest point
                            for j in range(i, min(i + 10, len(self.candles))):
                                if self.candles.iloc[j]["low"] < deviation_low:
                                    deviation_low = self.candles.iloc[j]["low"]
                                    deviation_idx = j

                            # Validate it came back inside
                            came_back = self._validate_deviation_came_back_inside_from_idx(
                                deviation_idx, tap2_price, "low"
                            )

                            if came_back:
                                return {
                                    "idx": deviation_idx,
                                    "price": float(deviation_low),
                                    "time": str(self.candles.iloc[deviation_idx]["open_time"]) if "open_time" in self.candles.columns else str(deviation_idx),
                                    "type": "tap3_model1_from_m2_failure",
                                    "is_deviation": True,
                                    "is_lower_than_tap2": True,
                                    "model2_failure_price": tap3_m2_price,
                                    "original_m2_idx": tap3_m2_idx
                                }

            else:  # Distribution
                # For distribution: Look for price breaking above the Model 2 Tap3 (lower high fails)
                for i in range(tap3_m2_idx + 1, min(tap3_m2_idx + 20, len(self.candles) - 2)):
                    candle = self.candles.iloc[i]

                    # Price broke above Model 2 Tap3 - it failed
                    if candle["high"] > tap3_m2_price:
                        # Now find the new Model 1 Tap3 (needs to be higher than Tap2)
                        if candle["high"] > tap2_price:
                            # This is now a Model 1 with successive higher highs
                            deviation_high = candle["high"]
                            deviation_idx = i

                            # Find actual highest point
                            for j in range(i, min(i + 10, len(self.candles))):
                                if self.candles.iloc[j]["high"] > deviation_high:
                                    deviation_high = self.candles.iloc[j]["high"]
                                    deviation_idx = j

                            # Validate it came back inside
                            came_back = self._validate_deviation_came_back_inside_from_idx(
                                deviation_idx, tap2_price, "high"
                            )

                            if came_back:
                                return {
                                    "idx": deviation_idx,
                                    "price": float(deviation_high),
                                    "time": str(self.candles.iloc[deviation_idx]["open_time"]) if "open_time" in self.candles.columns else str(deviation_idx),
                                    "type": "tap3_model1_from_m2_failure",
                                    "is_deviation": True,
                                    "is_higher_than_tap2": True,
                                    "model2_failure_price": tap3_m2_price,
                                    "original_m2_idx": tap3_m2_idx
                                }

        except Exception as e:
            logger.debug(f"Error detecting M2 to M1 failure: {e}")

        return None

    def _calculate_range_quality(self, range_data: Dict, tap1: Dict, tap2: Dict,
                                  tap3: Dict) -> Dict:
        """
        Lecture 5B: Calculate range quality/rationality score.

        TCT 5B: "Range quality, rational and horizontal range with logical
        tap distances"

        Factors:
        - Horizontal price action (not too much slope)
        - Equal tap spacing
        - Clean pivot structure at each tap
        - Range size relative to overall price

        Returns:
            Dict with range quality metrics
        """
        result = {
            "quality_score": 0.0,
            "is_horizontal": False,
            "has_clean_pivots": False,
            "has_equal_spacing": False,
            "quality_factors": []
        }

        try:
            score = 0.0

            # Check horizontal range (25% weight)
            range_size = range_data["range_size"]
            range_midpoint = (range_data["range_high"] + range_data["range_low"]) / 2
            horizontal_tolerance = range_size * 0.3

            # Check if taps are roughly horizontal (within tolerance of each other)
            tap_prices = [tap1["price"], tap2["price"], tap3["price"]]
            tap_variance = max(tap_prices) - min(tap_prices)

            if tap_variance <= range_size * 0.5:
                score += 0.25
                result["is_horizontal"] = True
                result["quality_factors"].append("Horizontal range structure")

            # Check clean pivots (25% weight)
            tap1_clean = self._is_swing_low(tap1["idx"]) or self._is_swing_high(tap1["idx"])
            tap2_clean = self._is_swing_low(tap2["idx"]) or self._is_swing_high(tap2["idx"])
            tap3_clean = self._is_swing_low(tap3["idx"]) or self._is_swing_high(tap3["idx"])

            clean_count = sum([tap1_clean, tap2_clean, tap3_clean])
            score += (clean_count / 3) * 0.25

            if clean_count == 3:
                result["has_clean_pivots"] = True
                result["quality_factors"].append("All taps have clean pivot structure")

            # Check tap spacing (25% weight)
            spacing = self._validate_tap_spacing(tap1, tap2, tap3, range_data)
            if spacing["spacing_valid"]:
                score += 0.25
                result["has_equal_spacing"] = True
                result["quality_factors"].append("Equal tap spacing")
            else:
                score += spacing["spacing_quality"] * 0.25

            # Check range size rationality (25% weight)
            # Range should not be too small or too large relative to price
            equilibrium = range_data["equilibrium"]
            range_percent = (range_size / equilibrium) * 100

            # Good range is typically 1-5% of price
            if 0.5 <= range_percent <= 8:
                score += 0.25
                result["quality_factors"].append(f"Range size {range_percent:.2f}% is rational")
            elif range_percent < 0.5:
                score += 0.1
                result["quality_factors"].append("Range may be too small")
            else:
                score += 0.1
                result["quality_factors"].append("Range may be too large")

            result["quality_score"] = round(min(score, 1.0), 3)

        except Exception as e:
            logger.debug(f"Error calculating range quality: {e}")

        return result

    # ================================================================
    # LECTURE 6: ADVANCED TCT SCHEMATICS
    # ================================================================

    def _detect_schematic_conversion(self, schematic: Dict, range_data: Dict) -> Optional[Dict]:
        """
        Lecture 6: Detect when a schematic converts to the opposite type.

        TCT 6: "The same way if price starts deviating the low confirming a potential
        TCT accumulation schematic, the same way price could be confirming a TCT
        distribution once that range high target of the accumulation gets hit but
        starts to deviate"

        TCT 6: "A distribution can always go over into an accumulation just like
        the accumulation can go over into a distribution"

        Args:
            schematic: The original schematic that may be converting
            range_data: Range data for the schematic

        Returns:
            Dict with conversion info if detected, None otherwise
        """
        result = {
            "conversion_detected": False,
            "original_type": schematic.get("schematic_type"),
            "converted_type": None,
            "conversion_trigger": None,
            "new_tap3_price": None,
            "new_tap3_idx": None,
            "follow_through_expectation": None
        }

        try:
            original_type = schematic.get("schematic_type", "")
            target_price = schematic.get("target", {}).get("price")
            tap3_idx = schematic.get("tap3", {}).get("idx", 0)
            range_high = range_data["range_high"]
            range_low = range_data["range_low"]

            if not target_price or tap3_idx >= len(self.candles) - 10:
                return result

            # Search for conversion after schematic confirmation
            search_start = tap3_idx + 5
            search_end = min(tap3_idx + 50, len(self.candles) - 5)

            if "Accumulation" in original_type:
                # Accumulation targets range high
                # Conversion happens if price reaches target then deviates it,
                # potentially starting a distribution

                for i in range(search_start, search_end):
                    candle = self.candles.iloc[i]

                    # Check if target (range high) was reached
                    if candle["high"] >= target_price:
                        # Now look for deviation above range high (start of distribution)
                        deviation_high = candle["high"]
                        deviation_idx = i

                        # Find the highest point of this deviation
                        for j in range(i, min(i + 15, len(self.candles))):
                            if self.candles.iloc[j]["high"] > deviation_high:
                                deviation_high = self.candles.iloc[j]["high"]
                                deviation_idx = j
                            # Check if it came back inside (valid deviation)
                            if self.candles.iloc[j]["close"] < range_high:
                                # This is a valid deviation - potential distribution starting
                                result["conversion_detected"] = True
                                result["converted_type"] = "Model_2_Distribution"
                                result["conversion_trigger"] = "target_hit_then_deviated"
                                result["new_tap3_price"] = float(deviation_high)
                                result["new_tap3_idx"] = deviation_idx
                                result["follow_through_expectation"] = "expect_rotation_to_low"
                                return result

            elif "Distribution" in original_type:
                # Distribution targets range low
                # Conversion happens if price reaches target then deviates it,
                # potentially starting an accumulation

                for i in range(search_start, search_end):
                    candle = self.candles.iloc[i]

                    # Check if target (range low) was reached
                    if candle["low"] <= target_price:
                        # Now look for deviation below range low (start of accumulation)
                        deviation_low = candle["low"]
                        deviation_idx = i

                        # Find the lowest point of this deviation
                        for j in range(i, min(i + 15, len(self.candles))):
                            if self.candles.iloc[j]["low"] < deviation_low:
                                deviation_low = self.candles.iloc[j]["low"]
                                deviation_idx = j
                            # Check if it came back inside (valid deviation)
                            if self.candles.iloc[j]["close"] > range_low:
                                # This is a valid deviation - potential accumulation starting
                                result["conversion_detected"] = True
                                result["converted_type"] = "Model_2_Accumulation"
                                result["conversion_trigger"] = "target_hit_then_deviated"
                                result["new_tap3_price"] = float(deviation_low)
                                result["new_tap3_idx"] = deviation_idx
                                result["follow_through_expectation"] = "expect_rotation_to_high"
                                return result

        except Exception as e:
            logger.debug(f"Error detecting schematic conversion: {e}")

        return result

    def _detect_dual_side_deviation(self, range_data: Dict, tap2: Dict, tap3: Dict,
                                     schematic_type: str) -> Dict:
        """
        Lecture 6: Detect when range has deviations on both sides prior to confirmation.

        TCT 6: "It happens quite often that prior to confirming either the accumulation
        or the distribution you have deviations on both sides"

        TCT 6: "In the situations where you do have deviations on both ends, you can't
        just target the opposite range extreme point once your confirmations have been
        met because there's always the possibility of price turning into an opposite
        model 2 schematic"

        Args:
            range_data: Range data
            tap2: Current Tap2
            tap3: Current Tap3
            schematic_type: Type of schematic being evaluated

        Returns:
            Dict with dual-side deviation info and risk triggers
        """
        result = {
            "has_dual_side_deviation": False,
            "high_side_deviated": False,
            "low_side_deviated": False,
            "high_deviation_price": None,
            "low_deviation_price": None,
            "extreme_demand_zone": None,
            "extreme_supply_zone": None,
            "extreme_liquidity_high": None,
            "extreme_liquidity_low": None,
            "risk_on_trigger": None,  # When to go risk-on for direction
            "risk_off_trigger": None,  # When to take profit/exit
            "watch_structure_from": None,
            "can_convert_to_opposite": False
        }

        try:
            range_high = range_data["range_high"]
            range_low = range_data["range_low"]
            dl_high = range_data["dl_high"]
            dl_low = range_data["dl_low"]

            # Determine the search window
            range_high_idx = range_data.get("range_high_idx", 0)
            range_low_idx = range_data.get("range_low_idx", 0)
            tap3_idx = tap3.get("idx", 0)

            search_start = min(range_high_idx, range_low_idx)
            search_end = tap3_idx + 5

            # Check for deviation on both sides
            for i in range(search_start, min(search_end, len(self.candles))):
                candle = self.candles.iloc[i]

                # Check high side deviation
                if candle["high"] > range_high and candle["close"] <= dl_high:
                    if not result["high_side_deviated"]:
                        result["high_side_deviated"] = True
                        result["high_deviation_price"] = float(candle["high"])

                # Check low side deviation
                if candle["low"] < range_low and candle["close"] >= dl_low:
                    if not result["low_side_deviated"]:
                        result["low_side_deviated"] = True
                        result["low_deviation_price"] = float(candle["low"])

            # If both sides are deviated
            result["has_dual_side_deviation"] = result["high_side_deviated"] and result["low_side_deviated"]

            if result["has_dual_side_deviation"]:
                result["can_convert_to_opposite"] = True

                # Find extreme S/D zones
                result["extreme_demand_zone"] = self._find_extreme_demand(range_data, tap3_idx)
                result["extreme_supply_zone"] = self._find_extreme_supply(range_data, tap3_idx)

                # Find extreme liquidity levels
                if result["low_deviation_price"]:
                    result["extreme_liquidity_low"] = {
                        "price": result["low_deviation_price"],
                        "type": "extreme_sell_side_liquidity"
                    }
                if result["high_deviation_price"]:
                    result["extreme_liquidity_high"] = {
                        "price": result["high_deviation_price"],
                        "type": "extreme_buy_side_liquidity"
                    }

                # Set risk triggers based on current schematic type
                if "Accumulation" in schematic_type:
                    # For accumulation: BOS bullish = risk-on for long, BOS bearish = risk-off
                    result["risk_on_trigger"] = {
                        "type": "bullish_bos",
                        "description": "Break of structure bullish confirms accumulation",
                        "watch_level": result["low_deviation_price"]
                    }
                    result["risk_off_trigger"] = {
                        "type": "bearish_bos",
                        "description": "Break of structure bearish - potential distribution forming",
                        "watch_level": result["high_deviation_price"]
                    }
                    result["watch_structure_from"] = "low_up"

                else:  # Distribution
                    # For distribution: BOS bearish = risk-on for short, BOS bullish = risk-off
                    result["risk_on_trigger"] = {
                        "type": "bearish_bos",
                        "description": "Break of structure bearish confirms distribution",
                        "watch_level": result["high_deviation_price"]
                    }
                    result["risk_off_trigger"] = {
                        "type": "bullish_bos",
                        "description": "Break of structure bullish - potential accumulation forming",
                        "watch_level": result["low_deviation_price"]
                    }
                    result["watch_structure_from"] = "high_down"

        except Exception as e:
            logger.debug(f"Error detecting dual-side deviation: {e}")

        return result

    def _detect_ltf_to_htf_range_transition(self, range_data: Dict, tap3_idx: int) -> Dict:
        """
        Lecture 6: Detect when a low timeframe range is growing into a high timeframe range.

        TCT 6: "What you'll see after an aggressive expansion is Market structure trending
        up and we know after an aggressive expansion price loves to create a range and
        that range will always start with a low time frame break"

        TCT 6: "You have those low time frame breaks, they will make up for a higher
        time frame Market structure that will have its own little range"

        TCT 6: "This is exactly how a low time frame range goes over into a high time
        frame range it happens all the time"

        Args:
            range_data: Current range data
            tap3_idx: Index of Tap3

        Returns:
            Dict with LTF-to-HTF transition info
        """
        result = {
            "transition_detected": False,
            "original_range": range_data,
            "expanded_range": None,
            "transition_type": None,  # "upside_expansion" or "downside_expansion"
            "new_range_high": None,
            "new_range_low": None,
            "expansion_factor": None,
            "recommendation": None
        }

        try:
            range_high = range_data["range_high"]
            range_low = range_data["range_low"]
            range_size = range_data["range_size"]

            # Look for expansion beyond the current range after Tap3
            search_start = tap3_idx + 1
            search_end = min(tap3_idx + 40, len(self.candles) - 5)

            highest_high = range_high
            lowest_low = range_low
            highest_idx = None
            lowest_idx = None

            for i in range(search_start, search_end):
                candle = self.candles.iloc[i]

                if candle["high"] > highest_high:
                    highest_high = float(candle["high"])
                    highest_idx = i

                if candle["low"] < lowest_low:
                    lowest_low = float(candle["low"])
                    lowest_idx = i

            # Check if range expanded significantly
            new_range_size = highest_high - lowest_low
            expansion = new_range_size / range_size if range_size > 0 else 1

            # If range expanded by more than 50%, it's transitioning to HTF
            if expansion > 1.5:
                result["transition_detected"] = True
                result["new_range_high"] = highest_high
                result["new_range_low"] = lowest_low
                result["expansion_factor"] = round(expansion, 2)

                if highest_idx and highest_idx > tap3_idx + 5:
                    result["transition_type"] = "upside_expansion"
                    result["recommendation"] = "Zoom out - LTF range became HTF accumulation zone"
                elif lowest_idx and lowest_idx > tap3_idx + 5:
                    result["transition_type"] = "downside_expansion"
                    result["recommendation"] = "Zoom out - LTF range became HTF distribution zone"
                else:
                    result["transition_type"] = "both_sides_expansion"
                    result["recommendation"] = "Zoom out - LTF range became larger HTF range"

                result["expanded_range"] = {
                    "range_high": highest_high,
                    "range_low": lowest_low,
                    "range_size": new_range_size,
                    "equilibrium": (highest_high + lowest_low) / 2,
                    "dl_high": highest_high + (new_range_size * self.DEVIATION_LIMIT_PERCENT),
                    "dl_low": lowest_low - (new_range_size * self.DEVIATION_LIMIT_PERCENT)
                }

        except Exception as e:
            logger.debug(f"Error detecting LTF-to-HTF transition: {e}")

        return result

    def _detect_multi_tf_schematic_validity(self, tap1: Dict, tap2: Dict, tap3: Dict,
                                            range_data: Dict) -> Dict:
        """
        Lecture 6: Check if schematic taps are valid on multiple timeframes.

        TCT 6: "Your tab two and your tab three on the five just look like one deviation
        on the 30"

        TCT 6: "So you can kind of trade a model two twice okay you can trade it twice
        one time on the 5 to 10 minute that already hit its Target and then we came back
        for a higher time frame model two"

        TCT 6: "Always when you're tap two and your tap three are way more narrow and
        close by than your tab one and your tab two right... if I zoom out this will
        probably look like one deviation"

        Args:
            tap1, tap2, tap3: Tap dictionaries
            range_data: Range data

        Returns:
            Dict with multi-TF validity info and potential HTF schematic
        """
        result = {
            "has_multi_tf_opportunity": False,
            "tap_distance_ratio": None,  # tap2-tap3 distance vs tap1-tap2 distance
            "ltf_schematic_valid": True,
            "htf_schematic_potential": False,
            "htf_tap2_price": None,  # Tap2 if viewed from HTF
            "htf_extreme_sd": None,  # Extreme S/D zone for HTF tap3
            "recommendation": None
        }

        try:
            tap1_idx = tap1.get("idx", 0)
            tap2_idx = tap2.get("idx", 0)
            tap3_idx = tap3.get("idx", 0)
            tap2_price = tap2.get("price", 0)
            tap3_price = tap3.get("price", 0)

            # Calculate distances
            tap1_to_tap2 = tap2_idx - tap1_idx
            tap2_to_tap3 = tap3_idx - tap2_idx

            if tap1_to_tap2 > 0:
                result["tap_distance_ratio"] = round(tap2_to_tap3 / tap1_to_tap2, 2)

                # If tap2-tap3 is much closer than tap1-tap2, HTF sees them as one deviation
                if result["tap_distance_ratio"] < 0.5:
                    result["has_multi_tf_opportunity"] = True
                    result["htf_schematic_potential"] = True

                    # On HTF, tap2 and tap3 merge into one deviation
                    # The "tap2" on HTF would be the more extreme of the two
                    if "Accumulation" in range_data.get("direction", ""):
                        result["htf_tap2_price"] = min(tap2_price, tap3_price)
                    else:
                        result["htf_tap2_price"] = max(tap2_price, tap3_price)

                    # Find extreme S/D for potential HTF tap3
                    result["htf_extreme_sd"] = self._find_extreme_supply(range_data, tap3_idx)
                    if not result["htf_extreme_sd"]:
                        result["htf_extreme_sd"] = self._find_extreme_demand(range_data, tap3_idx)

                    result["recommendation"] = (
                        "Trade LTF schematic first. After target hit, watch for price "
                        "return to extreme S/D for HTF schematic entry with larger R:R"
                    )

        except Exception as e:
            logger.debug(f"Error detecting multi-TF validity: {e}")

        return result

    def _detect_enhanced_wov_in_wov(self, tap3: Dict, range_data: Dict,
                                    schematic_type: str) -> Dict:
        """
        Lecture 6: Enhanced WOV-in-WOV detection - schematic within schematic.

        TCT 6: "What I mean when I'm talking about WOV and WOV is that our third tap
        okay our third tap of our initial TCT schematic has its own little TCT schematic"

        TCT 6: "This way we can have an entry confirmation way earlier than the original
        break of structure level causing our risk to reward to improve drastically"

        TCT 6: "Where we had that 2.2 R reward now we have nine... which is an absolute
        Banger of a trade"

        This is an enhanced version that specifically looks for complete schematics
        within the third tap, not just nested ranges.

        Args:
            tap3: Tap3 dictionary
            range_data: Range data for outer schematic
            schematic_type: Type of outer schematic

        Returns:
            Dict with enhanced WOV-in-WOV info
        """
        result = {
            "has_wov_in_wov": False,
            "inner_schematic_type": None,
            "inner_tap1": None,
            "inner_tap2": None,
            "inner_tap3": None,
            "inner_bos": None,
            "inner_entry_price": None,
            "inner_stop_loss": None,
            "outer_target": None,
            "standard_rr": None,
            "wov_optimized_rr": None,
            "rr_improvement_factor": None,
            "entry_improvement_pips": None
        }

        try:
            tap3_idx = tap3.get("idx", 0)
            tap3_price = tap3.get("price", 0)

            if tap3_idx >= len(self.candles) - 20:
                return result

            # Search for inner schematic structure within the tap3 area
            search_start = max(0, tap3_idx - 15)
            search_end = min(tap3_idx + 20, len(self.candles) - 5)

            inner_candles = self.candles.iloc[search_start:search_end].reset_index(drop=True)

            if len(inner_candles) < 15:
                return result

            # Look for inner schematic based on outer schematic type
            if "Distribution" in schematic_type:
                # For distribution tap3 (which is a high), look for inner distribution
                inner_schematic = self._find_inner_distribution_schematic(
                    inner_candles, tap3_price, range_data
                )
            else:  # Accumulation
                # For accumulation tap3 (which is a low), look for inner accumulation
                inner_schematic = self._find_inner_accumulation_schematic(
                    inner_candles, tap3_price, range_data
                )

            if inner_schematic:
                result["has_wov_in_wov"] = True
                result["inner_schematic_type"] = inner_schematic.get("type")
                result["inner_tap1"] = inner_schematic.get("tap1")
                result["inner_tap2"] = inner_schematic.get("tap2")
                result["inner_tap3"] = inner_schematic.get("tap3")
                result["inner_bos"] = inner_schematic.get("bos")
                result["inner_entry_price"] = inner_schematic.get("entry_price")
                result["inner_stop_loss"] = inner_schematic.get("stop_loss")

                # Calculate R:R comparison
                outer_entry = range_data.get("equilibrium")  # Typical standard entry area

                if "Distribution" in schematic_type:
                    result["outer_target"] = range_data["range_low"]

                    # Standard R:R (entering at BOS level near range)
                    if outer_entry and tap3_price:
                        standard_risk = tap3_price - outer_entry
                        standard_reward = outer_entry - range_data["range_low"]
                        if standard_risk > 0:
                            result["standard_rr"] = round(standard_reward / standard_risk, 2)

                    # WOV optimized R:R (entering at inner schematic BOS)
                    if result["inner_entry_price"] and result["inner_stop_loss"]:
                        wov_risk = result["inner_stop_loss"] - result["inner_entry_price"]
                        wov_reward = result["inner_entry_price"] - range_data["range_low"]
                        if wov_risk > 0:
                            result["wov_optimized_rr"] = round(wov_reward / wov_risk, 2)

                else:  # Accumulation
                    result["outer_target"] = range_data["range_high"]

                    # Standard R:R
                    if outer_entry and tap3_price:
                        standard_risk = outer_entry - tap3_price
                        standard_reward = range_data["range_high"] - outer_entry
                        if standard_risk > 0:
                            result["standard_rr"] = round(standard_reward / standard_risk, 2)

                    # WOV optimized R:R
                    if result["inner_entry_price"] and result["inner_stop_loss"]:
                        wov_risk = result["inner_entry_price"] - result["inner_stop_loss"]
                        wov_reward = range_data["range_high"] - result["inner_entry_price"]
                        if wov_risk > 0:
                            result["wov_optimized_rr"] = round(wov_reward / wov_risk, 2)

                # Calculate improvement factor
                if result["standard_rr"] and result["wov_optimized_rr"]:
                    result["rr_improvement_factor"] = round(
                        result["wov_optimized_rr"] / result["standard_rr"], 2
                    )

                # Calculate entry improvement in price units
                if result["inner_entry_price"] and outer_entry:
                    result["entry_improvement_pips"] = abs(
                        result["inner_entry_price"] - outer_entry
                    )

        except Exception as e:
            logger.debug(f"Error detecting enhanced WOV-in-WOV: {e}")

        return result

    def _find_inner_distribution_schematic(self, candles: pd.DataFrame, tap3_price: float,
                                           outer_range: Dict) -> Optional[Dict]:
        """Find an inner distribution schematic within the tap3 area."""
        try:
            # Look for structure: higher highs forming a range, then deviation
            swing_highs = []
            swing_lows = []

            for i in range(3, len(candles) - 3):
                if self._is_local_swing_high(candles, i):
                    swing_highs.append({"idx": i, "price": float(candles.iloc[i]["high"])})
                if self._is_local_swing_low(candles, i):
                    swing_lows.append({"idx": i, "price": float(candles.iloc[i]["low"])})

            if len(swing_highs) < 2 or len(swing_lows) < 1:
                return None

            # Find potential inner range high and low
            inner_range_high = max(swing_highs, key=lambda x: x["price"])
            inner_range_low = min(swing_lows, key=lambda x: x["price"])

            inner_range_size = inner_range_high["price"] - inner_range_low["price"]

            # Inner range should be smaller than outer range
            if inner_range_size > outer_range["range_size"] * self.LTF_HTF_SIZE_RATIO:
                return None

            # Look for deviation pattern
            # Find tap1 (range high), tap2 (first deviation higher), tap3 (lower high)
            tap1 = inner_range_high

            # Look for tap2 (higher than tap1)
            tap2 = None
            for sh in swing_highs:
                if sh["idx"] > tap1["idx"] and sh["price"] > tap1["price"]:
                    tap2 = sh
                    break

            if not tap2:
                return None

            # Look for tap3 (lower high, lower than tap2)
            tap3 = None
            for sh in swing_highs:
                if sh["idx"] > tap2["idx"] and sh["price"] < tap2["price"]:
                    tap3 = sh
                    break

            if not tap3:
                return None

            # Look for BOS (break of structure bearish)
            bos = None
            for i in range(tap3["idx"] + 1, len(candles) - 1):
                current_low = float(candles.iloc[i]["low"])
                # Find previous swing low to break
                for sl in swing_lows:
                    if sl["idx"] < i and current_low < sl["price"]:
                        bos = {
                            "idx": i,
                            "price": current_low,
                            "broke_level": sl["price"]
                        }
                        break
                if bos:
                    break

            if bos:
                return {
                    "type": "Inner_Model_2_Distribution",
                    "tap1": tap1,
                    "tap2": tap2,
                    "tap3": tap3,
                    "bos": bos,
                    "entry_price": bos["price"],
                    "stop_loss": tap3["price"]
                }

        except Exception as e:
            logger.debug(f"Error finding inner distribution: {e}")

        return None

    def _find_inner_accumulation_schematic(self, candles: pd.DataFrame, tap3_price: float,
                                           outer_range: Dict) -> Optional[Dict]:
        """Find an inner accumulation schematic within the tap3 area."""
        try:
            # Look for structure: lower lows forming a range, then deviation
            swing_highs = []
            swing_lows = []

            for i in range(3, len(candles) - 3):
                if self._is_local_swing_high(candles, i):
                    swing_highs.append({"idx": i, "price": float(candles.iloc[i]["high"])})
                if self._is_local_swing_low(candles, i):
                    swing_lows.append({"idx": i, "price": float(candles.iloc[i]["low"])})

            if len(swing_lows) < 2 or len(swing_highs) < 1:
                return None

            # Find potential inner range high and low
            inner_range_high = max(swing_highs, key=lambda x: x["price"])
            inner_range_low = min(swing_lows, key=lambda x: x["price"])

            inner_range_size = inner_range_high["price"] - inner_range_low["price"]

            # Inner range should be smaller than outer range
            if inner_range_size > outer_range["range_size"] * self.LTF_HTF_SIZE_RATIO:
                return None

            # Look for deviation pattern
            # Find tap1 (range low), tap2 (first deviation lower), tap3 (higher low)
            tap1 = inner_range_low

            # Look for tap2 (lower than tap1)
            tap2 = None
            for sl in swing_lows:
                if sl["idx"] > tap1["idx"] and sl["price"] < tap1["price"]:
                    tap2 = sl
                    break

            if not tap2:
                return None

            # Look for tap3 (higher low, higher than tap2)
            tap3 = None
            for sl in swing_lows:
                if sl["idx"] > tap2["idx"] and sl["price"] > tap2["price"]:
                    tap3 = sl
                    break

            if not tap3:
                return None

            # Look for BOS (break of structure bullish)
            bos = None
            for i in range(tap3["idx"] + 1, len(candles) - 1):
                current_high = float(candles.iloc[i]["high"])
                # Find previous swing high to break
                for sh in swing_highs:
                    if sh["idx"] < i and current_high > sh["price"]:
                        bos = {
                            "idx": i,
                            "price": current_high,
                            "broke_level": sh["price"]
                        }
                        break
                if bos:
                    break

            if bos:
                return {
                    "type": "Inner_Model_2_Accumulation",
                    "tap1": tap1,
                    "tap2": tap2,
                    "tap3": tap3,
                    "bos": bos,
                    "entry_price": bos["price"],
                    "stop_loss": tap3["price"]
                }

        except Exception as e:
            logger.debug(f"Error finding inner accumulation: {e}")

        return None

    def _is_local_swing_high(self, candles: pd.DataFrame, idx: int, lookback: int = 2) -> bool:
        """Check for local swing high in a candle subset."""
        if idx < lookback or idx >= len(candles) - lookback:
            return False

        current = candles.iloc[idx]["high"]

        for i in range(idx - lookback, idx):
            if candles.iloc[i]["high"] >= current:
                return False

        for i in range(idx + 1, idx + lookback + 1):
            if candles.iloc[i]["high"] >= current:
                return False

        return True

    def _is_local_swing_low(self, candles: pd.DataFrame, idx: int, lookback: int = 2) -> bool:
        """Check for local swing low in a candle subset."""
        if idx < lookback or idx >= len(candles) - lookback:
            return False

        current = candles.iloc[idx]["low"]

        for i in range(idx - lookback, idx):
            if candles.iloc[i]["low"] <= current:
                return False

        for i in range(idx + 1, idx + lookback + 1):
            if candles.iloc[i]["low"] <= current:
                return False

        return True

    def _detect_model1_to_model2_flow(self, schematic: Dict, range_data: Dict) -> Dict:
        """
        Lecture 6: Detect when a Model 1 schematic flows into a Model 2.

        TCT 6: "What could happen is that price follows up with a higher low for a
        higher high and then maybe comes back down sweeping that liquidity before
        moving higher"

        TCT 6: "So it's kind of a model one that flows over into a model two and
        this happens quite often"

        TCT 6: "Usually schematics where the model one flows over into a model two
        they have follow through towards the upside... meaning he's fully positioning
        himself to push price higher"

        Args:
            schematic: The original Model 1 schematic
            range_data: Range data

        Returns:
            Dict with M1-to-M2 flow info and position management suggestions
        """
        result = {
            "m1_to_m2_flow_detected": False,
            "original_tap3": schematic.get("tap3"),
            "new_tap3_higher_low": None,  # For accumulation
            "new_tap3_lower_high": None,  # For distribution
            "new_wov_point": None,
            "trail_stop_to": None,
            "add_position_trigger": None,
            "expected_follow_through": None,
            "extended_target": None,
            "position_management": {}
        }

        try:
            model_type = schematic.get("model", "")
            if "Model_1" not in model_type:
                return result

            schematic_type = schematic.get("schematic_type", "")
            tap3 = schematic.get("tap3", {})
            tap3_idx = tap3.get("idx", 0)
            tap3_price = tap3.get("price", 0)

            if tap3_idx >= len(self.candles) - 15:
                return result

            # Search for 4th tap forming after the original Model 1 tap3
            search_start = tap3_idx + 3
            search_end = min(tap3_idx + 30, len(self.candles) - 5)

            if "Accumulation" in schematic_type:
                # Look for price coming back to form a higher low (Model 2 tap3)
                # First, price should move up (confirming M1), then come back down

                # Check if price moved up first (M1 working)
                highest_after_tap3 = tap3_price
                for i in range(search_start, search_end):
                    if self.candles.iloc[i]["high"] > highest_after_tap3:
                        highest_after_tap3 = float(self.candles.iloc[i]["high"])

                # If price moved up, now look for a higher low coming back
                if highest_after_tap3 > tap3_price:
                    for i in range(search_start, search_end):
                        if self._is_swing_low(i):
                            potential_hl_price = float(self.candles.iloc[i]["low"])

                            # Must be higher than tap3 (Model 2 requirement)
                            if potential_hl_price > tap3_price:
                                # Check if it sweeps extreme liquidity or mitigates extreme demand
                                extreme_liq = self._find_extreme_liquidity_for_accumulation(
                                    tap3_idx, tap3_price
                                )
                                extreme_demand = self._find_extreme_demand(range_data, i)

                                grabs_liq = extreme_liq and potential_hl_price <= extreme_liq.get("price", float('inf'))
                                mitigates_demand = extreme_demand and (
                                    extreme_demand["bottom"] <= potential_hl_price <= extreme_demand["top"]
                                )

                                if grabs_liq or mitigates_demand:
                                    result["m1_to_m2_flow_detected"] = True
                                    result["new_tap3_higher_low"] = {
                                        "idx": i,
                                        "price": potential_hl_price,
                                        "grabs_extreme_liquidity": grabs_liq,
                                        "mitigates_extreme_demand": mitigates_demand
                                    }
                                    result["new_wov_point"] = potential_hl_price

                                    # Position management
                                    result["trail_stop_to"] = potential_hl_price * 0.998  # Just below new HL
                                    result["add_position_trigger"] = {
                                        "type": "bullish_bos_from_hl",
                                        "description": "Add position on BOS bullish from higher low"
                                    }

                                    # Extended target (usually 50% beyond range high)
                                    range_high = range_data["range_high"]
                                    range_size = range_data["range_size"]
                                    result["extended_target"] = range_high + (range_size * self.M1_TO_M2_FOLLOW_THROUGH_BONUS - 1)
                                    result["expected_follow_through"] = "strong_upside_beyond_range_high"

                                    result["position_management"] = {
                                        "step_1": "Trail stop to new higher low (new WOV point)",
                                        "step_2": "Add position on BOS confirmation from higher low",
                                        "step_3": "Target extended beyond range high due to M1→M2 confluence",
                                        "risk_freed": "Original risk reduced, can re-deploy freed risk"
                                    }
                                    break

            else:  # Distribution
                # Look for price coming back to form a lower high (Model 2 tap3)
                # First, price should move down (confirming M1), then come back up

                lowest_after_tap3 = tap3_price
                for i in range(search_start, search_end):
                    if self.candles.iloc[i]["low"] < lowest_after_tap3:
                        lowest_after_tap3 = float(self.candles.iloc[i]["low"])

                # If price moved down, now look for a lower high coming back
                if lowest_after_tap3 < tap3_price:
                    for i in range(search_start, search_end):
                        if self._is_swing_high(i):
                            potential_lh_price = float(self.candles.iloc[i]["high"])

                            # Must be lower than tap3 (Model 2 requirement)
                            if potential_lh_price < tap3_price:
                                # Check if it grabs extreme liquidity or mitigates extreme supply
                                extreme_liq = self._find_extreme_liquidity_for_distribution(
                                    tap3_idx, tap3_price
                                )
                                extreme_supply = self._find_extreme_supply(range_data, i)

                                grabs_liq = extreme_liq and potential_lh_price >= extreme_liq.get("price", 0)
                                mitigates_supply = extreme_supply and (
                                    extreme_supply["bottom"] <= potential_lh_price <= extreme_supply["top"]
                                )

                                if grabs_liq or mitigates_supply:
                                    result["m1_to_m2_flow_detected"] = True
                                    result["new_tap3_lower_high"] = {
                                        "idx": i,
                                        "price": potential_lh_price,
                                        "grabs_extreme_liquidity": grabs_liq,
                                        "mitigates_extreme_supply": mitigates_supply
                                    }
                                    result["new_wov_point"] = potential_lh_price

                                    # Position management
                                    result["trail_stop_to"] = potential_lh_price * 1.002  # Just above new LH
                                    result["add_position_trigger"] = {
                                        "type": "bearish_bos_from_lh",
                                        "description": "Add position on BOS bearish from lower high"
                                    }

                                    # Extended target
                                    range_low = range_data["range_low"]
                                    range_size = range_data["range_size"]
                                    result["extended_target"] = range_low - (range_size * (self.M1_TO_M2_FOLLOW_THROUGH_BONUS - 1))
                                    result["expected_follow_through"] = "strong_downside_beyond_range_low"

                                    result["position_management"] = {
                                        "step_1": "Trail stop to new lower high (new WOV point)",
                                        "step_2": "Add position on BOS confirmation from lower high",
                                        "step_3": "Target extended beyond range low due to M1→M2 confluence",
                                        "risk_freed": "Original risk reduced, can re-deploy freed risk"
                                    }
                                    break

        except Exception as e:
            logger.debug(f"Error detecting M1-to-M2 flow: {e}")

        return result

    def _calculate_context_based_follow_through(self, range_data: Dict,
                                                 schematic_type: str) -> Dict:
        """
        Lecture 6: Calculate follow-through expectation based on context.

        TCT 6: "What do you think makes more sense an accumulation or a distribution
        obviously it will be a distribution because prices in ranges they rotate you
        just had a rotation towards the Range High now we can come back down again"

        TCT 6: "You don't want to try and catch a crazy long in the extreme premium
        pricing of the range"

        TCT 6: "We want to be looking for bigger shorts in the premium section of
        the range and bigger Longs in the discount section"

        Args:
            range_data: Range data
            schematic_type: Type of schematic

        Returns:
            Dict with context-based follow-through prediction
        """
        result = {
            "current_position_in_range": None,  # "premium", "discount", "equilibrium"
            "position_percentage": None,
            "expected_follow_through": None,
            "confidence_level": None,  # "high", "medium", "low"
            "reasoning": None,
            "recommendation": None
        }

        try:
            range_high = range_data["range_high"]
            range_low = range_data["range_low"]
            range_size = range_data["range_size"]
            equilibrium = range_data["equilibrium"]

            # Get current price (last candle close)
            current_price = float(self.candles.iloc[-1]["close"])

            # Calculate position within range
            if range_size > 0:
                position_pct = (current_price - range_low) / range_size
                result["position_percentage"] = round(position_pct * 100, 1)

                # Classify position
                if position_pct >= self.PREMIUM_THRESHOLD:
                    result["current_position_in_range"] = "premium"
                elif position_pct <= self.DISCOUNT_THRESHOLD:
                    result["current_position_in_range"] = "discount"
                else:
                    result["current_position_in_range"] = "equilibrium"

            # Determine follow-through expectation
            if "Accumulation" in schematic_type:
                if result["current_position_in_range"] == "discount":
                    result["expected_follow_through"] = "strong"
                    result["confidence_level"] = "high"
                    result["reasoning"] = (
                        "Accumulation in discount zone aligns with range rotation "
                        "dynamics - expect strong follow-through to range high"
                    )
                    result["recommendation"] = "Trade with full position size"
                elif result["current_position_in_range"] == "equilibrium":
                    result["expected_follow_through"] = "moderate"
                    result["confidence_level"] = "medium"
                    result["reasoning"] = (
                        "Accumulation near equilibrium - decent follow-through "
                        "expected but watch for distribution forming at highs"
                    )
                    result["recommendation"] = "Trade with standard position size"
                else:  # Premium
                    result["expected_follow_through"] = "weak"
                    result["confidence_level"] = "low"
                    result["reasoning"] = (
                        "Accumulation in premium zone contradicts range rotation - "
                        "price may struggle, distribution more likely"
                    )
                    result["recommendation"] = "Consider skipping or reduced size"

            else:  # Distribution
                if result["current_position_in_range"] == "premium":
                    result["expected_follow_through"] = "strong"
                    result["confidence_level"] = "high"
                    result["reasoning"] = (
                        "Distribution in premium zone aligns with range rotation "
                        "dynamics - expect strong follow-through to range low"
                    )
                    result["recommendation"] = "Trade with full position size"
                elif result["current_position_in_range"] == "equilibrium":
                    result["expected_follow_through"] = "moderate"
                    result["confidence_level"] = "medium"
                    result["reasoning"] = (
                        "Distribution near equilibrium - decent follow-through "
                        "expected but watch for accumulation forming at lows"
                    )
                    result["recommendation"] = "Trade with standard position size"
                else:  # Discount
                    result["expected_follow_through"] = "weak"
                    result["confidence_level"] = "low"
                    result["reasoning"] = (
                        "Distribution in discount zone contradicts range rotation - "
                        "price may struggle, accumulation more likely"
                    )
                    result["recommendation"] = "Consider skipping or reduced size"

        except Exception as e:
            logger.debug(f"Error calculating context follow-through: {e}")

        return result

    # ================================================================
    # UTILITY METHODS
    # ================================================================


    def _is_inside_bar(self, idx: int) -> bool:
        """
        Check if candle at idx is an inside bar (TCT Lecture 1).
        Inside bar: high and low are inside the previous bar's high and low.
        Inside bars do NOT count for the 6-candle rule.
        """
        if idx < 1 or idx >= len(self.candles):
            return False
        curr = self.candles.iloc[idx]
        prev = self.candles.iloc[idx - 1]
        return float(curr["high"]) <= float(prev["high"]) and float(curr["low"]) >= float(prev["low"])

    def _is_swing_high(self, idx: int, lookback: int = None) -> bool:
        """
        Check if index is a swing high. Delegates to centralized PivotCache
        to eliminate structural drift across modules.
        """
        lb = lookback or self.SIX_CANDLE_LOOKBACK // 2
        return self._pivot_cache.get_swing_high(idx, lookback=lb)

    def _is_swing_low(self, idx: int, lookback: int = None) -> bool:
        """
        Check if index is a swing low. Delegates to centralized PivotCache
        to eliminate structural drift across modules.
        """
        lb = lookback or self.SIX_CANDLE_LOOKBACK // 2
        return self._pivot_cache.get_swing_low(idx, lookback=lb)

    def _find_previous_swing_high(self, current_idx: int) -> Optional[Dict]:
        """Find the previous swing high before current index."""
        for i in range(current_idx - 3, max(current_idx - 15, 2), -1):
            if self._is_swing_high(i, lookback=2):
                return {
                    "idx": i,
                    "price": float(self.candles.iloc[i]["high"])
                }
        return None

    def _find_previous_swing_low(self, current_idx: int) -> Optional[Dict]:
        """Find the previous swing low before current index."""
        for i in range(current_idx - 3, max(current_idx - 15, 2), -1):
            if self._is_swing_low(i, lookback=2):
                return {
                    "idx": i,
                    "price": float(self.candles.iloc[i]["low"])
                }
        return None

    def _check_equilibrium_touch(self, idx1: int, idx2: int, equilibrium: float) -> bool:
        """
        Check if price touched equilibrium to confirm range.

        TCT: "When we have a move back to the equilibrium, that's when the range is confirmed"

        Delegates to the shared check_equilibrium_touch utility in range_utils.
        """
        return check_equilibrium_touch(
            self.candles, idx1, idx2, equilibrium,
            check_between=True, post_range_candles=30,
        )

    def _get_session_context(self, timestamp: Optional[str] = None) -> Dict:
        """
        Return trading session context for a given timestamp.

        Prefers the BOS/entry confirmation timestamp when available;
        falls back to Tap3 timestamp otherwise.
        """
        default = {"session": None, "boost_applied": False, "multiplier": 1.0}
        if not timestamp:
            return default
        try:
            ts = pd.Timestamp(timestamp)
            if ts.tzinfo is None:
                from datetime import timezone
                ts = ts.replace(tzinfo=timezone.utc)
            return apply_session_multiplier(50.0, ts.to_pydatetime())
        except (TypeError, ValueError, Exception):
            return default

    def _validate_distribution_sweep(
        self,
        tap2: Dict,
        tap3: Dict,
        range_data: Dict,
    ) -> Dict:
        """
        Validate a distribution sweep using a window that always
        includes the tap2 and tap3 pivots.

        Uses the shared MarketStructureEngine to detect the sweep
        over a window anchored to the original pivots so they remain
        in scope for accurate classification.
        """
        from decision_trees.market_structure_engine import MarketStructureEngine

        buffer = 10
        start_idx = max(0, tap2["idx"] - buffer)
        end_idx = min(len(self.candles), tap3["idx"] + buffer)
        window_df = self.candles.iloc[start_idx:end_idx].reset_index(drop=True)

        ms = MarketStructureEngine()
        sweep = ms.detect_sweep(
            window_df,
            range_high=range_data["range_high"],
            range_low=range_data["range_low"],
            direction="bearish",
        )
        return {
            "classification": sweep.classification,
            "swept": sweep.swept,
            "returned_inside": sweep.returned_inside,
            "sweep_count": sweep.sweep_count,
        }

    def _validate_distribution_sweep(self, range_data: Dict, tap2: Dict,
                                      tap3: Dict) -> Dict:
        """
        Validate that the distribution deviation actually sweeps built-up
        liquidity above the range high, rather than being a true breakout.

        Uses MarketStructureEngine.detect_liquidity_pools() and detect_sweep()
        to classify the deviation.

        Must be called BEFORE schematic construction.
        """
        result = {
            "has_sweep": False,
            "classification": "no_sweep",
            "pools_swept": 0,
            "returned_inside": False,
        }

        try:
            from decision_trees.market_structure_engine import MarketStructureEngine
            ms_engine = MarketStructureEngine()

            range_high = range_data["range_high"]
            range_low = range_data["range_low"]

            # Get the candle window around the deviation
            start_idx = max(0, range_data.get("range_high_idx", 0) - 5)
            end_idx = min(len(self.candles), tap3["idx"] + 10)
            window_df = self.candles.iloc[start_idx:end_idx].copy()

            if len(window_df) < 5:
                return result

            # Detect liquidity pools (stacked highs, equal highs)
            pools = ms_engine.detect_liquidity_pools(window_df)
            stacked_highs = pools.get("equal_highs", []) + pools.get("swing_highs", [])

            # Count pools near or below range high (these get swept by deviation)
            pools_near_range_high = [
                p for p in stacked_highs
                if abs(p - range_high) <= range_data["range_size"] * 0.15
            ]

            # Detect sweep classification
            sweep = ms_engine.detect_sweep(window_df, range_high, range_low, "bearish")

            result["has_sweep"] = sweep.swept
            result["classification"] = sweep.classification
            result["pools_swept"] = len(pools_near_range_high)
            result["returned_inside"] = sweep.returned_inside

        except Exception as e:
            logger.debug(f"Sweep validation error: {e}")

        return result

    def _validate_deviation_came_back_inside(self, tab: Dict, range_data: Dict, direction: str) -> bool:
        """
        Validate that deviation came back inside range.

        TCT: "We come back inside okay making it a valid deviation"
        """
        tab_idx = tab["idx"]
        range_level = range_data["range_low"] if direction == "low" else range_data["range_high"]

        # Check next few candles for price coming back inside
        for i in range(tab_idx + 1, min(tab_idx + 15, len(self.candles))):
            candle = self.candles.iloc[i]
            if direction == "low" and candle["close"] > range_level:
                return True
            elif direction == "high" and candle["close"] < range_level:
                return True

        return False

    def _validate_deviation_came_back_inside_from_idx(self, idx: int, level: float, direction: str) -> bool:
        """Validate deviation came back inside from a specific index."""
        for i in range(idx + 1, min(idx + 15, len(self.candles))):
            candle = self.candles.iloc[i]
            if direction == "low" and candle["close"] > level:
                return True
            elif direction == "high" and candle["close"] < level:
                return True

        return False

    def _validate_six_candle_rule_on_tabs(self, tap1: Dict, tap2: Dict, tap3: Dict) -> bool:
        """
        Validate six candle rule applies on each tap pivot.

        TCT: "Your TCT schematic is only valid on a certain time frame if you can
        draw your marker structure on each tap applying the six candle rule"
        """
        try:
            # Validate each tap has proper pivot structure
            tap1_valid = self._is_swing_low(tap1["idx"]) or self._is_swing_high(tap1["idx"])
            tap2_valid = self._is_swing_low(tap2["idx"]) or self._is_swing_high(tap2["idx"])
            tap3_valid = self._is_swing_low(tap3["idx"]) or self._is_swing_high(tap3["idx"])

            return tap1_valid and tap2_valid and tap3_valid
        except:
            return False

    def _create_tab(self, range_data: Dict, key: str, tab_type: str) -> Optional[Dict]:
        """Create a tap dict from range data."""
        idx_key = f"{key}_idx"
        if key not in range_data or idx_key not in range_data:
            return None

        idx = range_data[idx_key]
        price = range_data[key]

        return {
            "idx": idx,
            "price": float(price),
            "time": str(self.candles.iloc[idx]["open_time"]) if "open_time" in self.candles.columns and idx < len(self.candles) else str(idx),
            "type": tab_type
        }

    def _calculate_schematic_quality(self, range_data: Dict, tap1: Dict, tap2: Dict,
                                      tap3: Dict, bos: Optional[Dict], model_type: str) -> float:
        """
        Calculate schematic quality score (Lecture 5A basic scoring).

        Factors:
        - BOS confirmation (especially inside range)
        - Deviation validity
        - Six candle rule compliance
        - Model 2 extreme requirements
        """
        score = 0.0

        # BOS confirmation (40% weight)
        if bos:
            score += 0.25
            if bos.get("is_inside_range"):
                score += 0.15  # TCT: "Always safer inside original range values"

        # Tap structure validity (30% weight)
        if self._validate_six_candle_rule_on_tabs(tap1, tap2, tap3):
            score += 0.3
        else:
            score += 0.15  # Partial credit

        # Model-specific requirements (30% weight)
        if model_type == "Model_2":
            if tap3.get("grabs_extreme_liquidity") and tap3.get("mitigates_extreme_demand"):
                score += 0.3  # TCT: "It can do both"
            elif tap3.get("grabs_extreme_liquidity") or tap3.get("mitigates_extreme_demand"):
                score += 0.25  # TCT: "Only one is needed"
            if tap3.get("grabs_extreme_liquidity") or tap3.get("mitigates_extreme_supply"):
                score += 0.25
            elif tap3.get("grabs_extreme_liquidity") and tap3.get("mitigates_extreme_supply"):
                score += 0.3
        else:  # Model 1
            # Valid lower lows / higher highs
            score += 0.3

        return round(min(score, 1.0), 3)

    def _calculate_schematic_quality_enhanced(self, range_data: Dict, tap1: Dict, tap2: Dict,
                                               tap3: Dict, bos: Optional[Dict], model_type: str,
                                               htf_validation: Dict, sd_zone_check: Optional[Dict],
                                               rr_analysis: Optional[Dict], tap_spacing: Dict,
                                               range_quality: Dict) -> float:
        """
        Lecture 5B: Enhanced schematic quality scoring.

        Additional factors from Lecture 5B:
        - Highest timeframe validation (all taps apply 6CR)
        - No S/D zone conflicts
        - Meets minimum R:R requirement
        - Equal tap spacing
        - Range quality/rationality
        - Trendline confluence (bonus)
        """
        # Start with base score from 5A methodology
        base_score = self._calculate_schematic_quality(range_data, tap1, tap2, tap3, bos, model_type)

        # Lecture 5B enhancements can add up to 0.3 more (capped at 1.0)
        enhanced_score = 0.0

        # HTF validation (10% bonus)
        if htf_validation and htf_validation.get("all_taps_valid_6cr"):
            enhanced_score += 0.10

        # No S/D conflicts (10% bonus)
        if sd_zone_check is None or not sd_zone_check.get("has_conflict"):
            enhanced_score += 0.05
            if sd_zone_check and not sd_zone_check.get("opposing_zone_blocks_target"):
                enhanced_score += 0.05  # Clear path to target

        # Meets minimum R:R (5% bonus)
        if rr_analysis and rr_analysis.get("meets_minimum_rr"):
            enhanced_score += 0.05

        # Equal tap spacing (5% bonus)
        if tap_spacing and tap_spacing.get("spacing_valid"):
            enhanced_score += 0.05

        # Range quality (5% bonus)
        if range_quality and range_quality.get("quality_score", 0) >= self.RANGE_QUALITY_MIN:
            enhanced_score += 0.05

        # Combine scores (cap at 1.0)
        total_score = base_score + enhanced_score
        return round(min(total_score, 1.0), 3)


def detect_tct_schematics(candles: pd.DataFrame, detected_ranges: List[Dict] = None,
                           pivot_cache: "PivotCache" = None,
                           range_engine_mode: str = None) -> Dict:
    """
    Main entry point for TCT schematic detection.

    Args:
        candles: DataFrame with OHLC data
        detected_ranges: Optional pre-detected ranges
        pivot_cache: Optional centralized pivot cache (avoids recomputation)
        range_engine_mode: Optional range engine mode override

    Returns: Dict with accumulation and distribution schematics
    """
    if len(candles) < 50:
        return {
            "accumulation_schematics": [],
            "distribution_schematics": [],
            "total_schematics": 0,
            "error": "Insufficient data (need 50+ candles)",
            "timestamp": datetime.utcnow().isoformat()
        }

    try:
        detector = TCTSchematicDetector(
            candles, pivot_cache=pivot_cache,
            range_engine_mode=range_engine_mode,
        )
        return detector.detect_all_schematics(detected_ranges)
    except Exception as e:
        logger.error(f"[ERROR] TCT Schematic detection failed: {e}")
        return {
            "accumulation_schematics": [],
            "distribution_schematics": [],
            "total_schematics": 0,
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("TCT SCHEMATIC DETECTOR TEST (Lecture 5A + 5B Methodology)")
    print("=" * 60)

    # Generate test data simulating accumulation pattern
    np.random.seed(42)
    dates = pd.date_range('2026-01-01', periods=200, freq='1h')

    # Create accumulation pattern: downtrend → range → deviation → higher low → breakout
    prices = []
    base = 100000

    for i in range(200):
        if i < 30:
            # Downtrend
            prices.append(base - i * 100 + np.random.uniform(-200, 200))
        elif i < 60:
            # Range low formed
            prices.append(base - 3000 + np.random.uniform(-500, 500))
        elif i < 80:
            # Move to range high
            prices.append(base - 3000 + (i - 60) * 75 + np.random.uniform(-300, 300))
        elif i < 100:
            # First deviation below range low
            prices.append(base - 3200 + np.random.uniform(-600, 400))
        elif i < 120:
            # Come back inside, higher low
            prices.append(base - 2800 + np.random.uniform(-400, 400))
        elif i < 140:
            # Second move down (Tap3 for Model 1 or higher low for Model 2)
            prices.append(base - 3100 + np.random.uniform(-500, 300))
        else:
            # Breakout
            prices.append(base - 2500 + (i - 140) * 50 + np.random.uniform(-300, 300))

    df = pd.DataFrame({
        'open_time': dates,
        'open': prices,
        'high': [p + np.random.uniform(100, 400) for p in prices],
        'low': [p - np.random.uniform(100, 400) for p in prices],
        'close': [p + np.random.uniform(-200, 200) for p in prices],
        'volume': np.random.uniform(100, 1000, 200)
    })

    print(f"\nAnalyzing {len(df)} candles for TCT schematics...")
    result = detect_tct_schematics(df)

    print(f"\n{'=' * 60}")
    print(f"Total Schematics Found: {result['total_schematics']}")
    print(f"Accumulation Schematics: {len(result['accumulation_schematics'])}")
    print(f"Distribution Schematics: {len(result['distribution_schematics'])}")

    for i, schematic in enumerate(result['accumulation_schematics'][:3], 1):
        print(f"\n  Accumulation #{i}: {schematic['schematic_type']}")
        print(f"    Quality Score: {schematic['quality_score']}")
        print(f"    Confirmed: {schematic['is_confirmed']}")
        print(f"    R:R = {schematic['risk_reward']}")
        if schematic['entry']['price']:
            print(f"    Entry: ${schematic['entry']['price']:.2f}")
            print(f"    Stop: ${schematic['stop_loss']['price']:.2f}")
            print(f"    Target: ${schematic['target']['price']:.2f}")

        # Lecture 5B enhancements output
        if 'lecture_5b_enhancements' in schematic:
            enhancements = schematic['lecture_5b_enhancements']
            print(f"    --- Lecture 5B Enhancements ---")
            if enhancements.get('htf_validation'):
                print(f"    6CR Valid: {enhancements['htf_validation'].get('all_taps_valid_6cr')}")
            if enhancements.get('rr_analysis'):
                print(f"    Meets Min R:R: {enhancements['rr_analysis'].get('meets_minimum_rr')}")
            if enhancements.get('tap_spacing'):
                print(f"    Tap Spacing Valid: {enhancements['tap_spacing'].get('spacing_valid')}")
            if enhancements.get('overlapping_structure', {}).get('has_overlapping_structure'):
                print(f"    Has Overlapping Structure (Domino Effect)")
                print(f"    Optimized R:R: {enhancements['overlapping_structure'].get('optimized_rr')}")
            if enhancements.get('trendline_liquidity', {}).get('provides_confluence'):
                print(f"    Trendline Confluence: YES")

    for i, schematic in enumerate(result['distribution_schematics'][:3], 1):
        print(f"\n  Distribution #{i}: {schematic['schematic_type']}")
        print(f"    Quality Score: {schematic['quality_score']}")
        print(f"    Confirmed: {schematic['is_confirmed']}")
        print(f"    R:R = {schematic['risk_reward']}")
        # Lecture 5B enhancements output
        if 'lecture_5b_enhancements' in schematic:
            enhancements = schematic['lecture_5b_enhancements']
            print(f"    --- Lecture 5B Enhancements ---")
            if enhancements.get('htf_validation'):
                print(f"    6CR Valid: {enhancements['htf_validation'].get('all_taps_valid_6cr')}")
            if enhancements.get('rr_analysis'):
                print(f"    Meets Min R:R: {enhancements['rr_analysis'].get('meets_minimum_rr')}")

    print(f"\n{'=' * 60}")
    print("TEST COMPLETE")
    print(f"{'=' * 60}\n")
