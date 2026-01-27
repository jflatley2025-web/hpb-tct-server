"""
tct_schematics.py — TCT Schematics Detection (Lecture 5A + 5B Advanced Methodology)
Author: HPB-TCT Dev Team
Date: 2026-01-26

Detects TCT Schematics (Accumulation & Distribution Models 1 and 2) based on:
- TCT 2024 mentorship - Lecture 5A | TCT schematics (core methodology)
- TCT 2024 mentorship - Lecture 5B | TCT schematics (advanced enhancements)
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

logger = logging.getLogger("TCT-Schematics")


class TCTSchematicDetector:
    """
    Main TCT Schematic Detection Engine implementing Lecture 5A + 5B methodology.

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

    def __init__(self, candles: pd.DataFrame):
        """Initialize with candle data."""
        self.candles = candles.copy()
        self.candles.reset_index(drop=True, inplace=True)

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
        ranges = detected_ranges if detected_ranges else self._find_accumulation_ranges()

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
                logger.debug(f"Error detecting accumulation schematic: {e}")
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

        for i in range(10, len(self.candles) - 30):
            # Find potential range low (significant swing low)
            if not self._is_swing_low(i):
                continue

            range_low = float(self.candles.iloc[i]["low"])
            range_low_idx = i

            # Find potential range high after range low
            for j in range(i + 5, min(i + 40, len(self.candles) - 10)):
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

        for i in range(start_idx, min(start_idx + 30, len(self.candles) - 5)):
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

        for i in range(start_idx, min(start_idx + 25, len(self.candles) - 5)):
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

        for i in range(start_idx, min(start_idx + 25, len(self.candles) - 5)):
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

        # Find potential range formations
        ranges = detected_ranges if detected_ranges else self._find_distribution_ranges()

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

                # Build Model 1 schematic if valid
                if tap3_m1:
                    schematic = self._build_distribution_schematic(
                        range_data, tap1, tap2, tap3_m1, model_type="Model_1"
                    )
                    if schematic:
                        schematics.append(schematic)

                # Build Model 2 schematic if valid
                if tap3_m2:
                    schematic = self._build_distribution_schematic(
                        range_data, tap1, tap2, tap3_m2, model_type="Model_2"
                    )
                    if schematic:
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
                logger.debug(f"Error detecting distribution schematic: {e}")
                continue

        # Sort by quality and recency
        schematics.sort(key=lambda x: (x.get("quality_score", 0), x.get("tap3", {}).get("idx", 0)), reverse=True)
        return schematics[:10]

    def _find_distribution_ranges(self) -> List[Dict]:
        """
        Find potential distribution ranges (trending up, pull from top to bottom).

        TCT: "When we're trending up, we pull our range from top to bottom"
        """
        ranges = []

        for i in range(10, len(self.candles) - 30):
            # Find potential range high (significant swing high)
            if not self._is_swing_high(i):
                continue

            range_high = float(self.candles.iloc[i]["high"])
            range_high_idx = i

            # Find potential range low after range high
            for j in range(i + 5, min(i + 40, len(self.candles) - 10)):
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

        for i in range(start_idx, min(start_idx + 30, len(self.candles) - 5)):
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

        for i in range(start_idx, min(start_idx + 25, len(self.candles) - 5)):
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

        for i in range(start_idx, min(start_idx + 25, len(self.candles) - 5)):
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

    def _detect_bos_confirmation(self, tap2: Dict, tap3: Dict, schematic_type: str) -> Optional[Dict]:
        """
        Detect break of structure confirmation for entry.

        TCT: "To confirm a TCT model one accumulation schematic we need to watch our
        Market structure from the highest point between tap two and tap three down
        towards our third tap low"

        TCT: "When that downwards Market structure breaks back to bullish after
        deviating that second Tap low that is when we confirm our TCT model"
        """
        tap2_idx = tap2["idx"]
        tap3_idx = tap3["idx"]
        tap3_price = tap3["price"]

        if tap3_idx >= len(self.candles) - 3:
            return None

        if schematic_type in ["Model_1_Accumulation", "Model_2_Accumulation"]:
            # Find highest point between Tap2 and Tap3
            range_candles = self.candles.iloc[tap2_idx:tap3_idx + 1]
            highest_point_idx = range_candles["high"].idxmax()
            highest_point_price = float(self.candles.iloc[highest_point_idx]["high"])

            # Watch structure from highest point to Tap3 low
            # TCT: Look for break back to bullish
            bos = self._find_bullish_bos(tap3_idx, highest_point_price, tap3_price)

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

        elif schematic_type in ["Model_1_Distribution", "Model_2_Distribution"]:
            # Find lowest point between Tap2 and Tap3
            range_candles = self.candles.iloc[tap2_idx:tap3_idx + 1]
            lowest_point_idx = range_candles["low"].idxmin()
            lowest_point_price = float(self.candles.iloc[lowest_point_idx]["low"])

            # Watch structure from lowest point to Tap3 high
            # TCT: Look for break back to bearish
            bos = self._find_bearish_bos(tap3_idx, lowest_point_price, tap3_price)

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

    def _find_bullish_bos(self, start_idx: int, high_price: float, low_price: float) -> Optional[Dict]:
        """
        Find bullish break of structure after Tap3.

        TCT: "When that breaks back to bullish and preferably that break of structure
        is back inside your original range values that's when we go long"
        """
        # Find swing lows and highs after Tap3 to detect structure
        for i in range(start_idx + 1, min(start_idx + 20, len(self.candles) - 2)):
            # Look for a higher high that breaks previous structure
            if self._is_swing_high(i):
                current_high = float(self.candles.iloc[i]["high"])

                # Check previous swing high
                prev_high = self._find_previous_swing_high(i)
                if prev_high and current_high > prev_high["price"]:
                    # BOS confirmed
                    is_inside_range = current_high < high_price  # Inside original range

                    return {
                        "idx": i,
                        "price": current_high,
                        "is_inside_range": is_inside_range,
                        "prev_swing_high": prev_high
                    }

        return None

    def _find_bearish_bos(self, start_idx: int, low_price: float, high_price: float) -> Optional[Dict]:
        """
        Find bearish break of structure after Tap3.

        TCT: "To confirm a TCT model one distribution schematic again we need to break
        structure from the lowest point between tap two and tap three up towards
        our third tap High"
        """
        # Find swing lows and highs after Tap3 to detect structure
        for i in range(start_idx + 1, min(start_idx + 20, len(self.candles) - 2)):
            # Look for a lower low that breaks previous structure
            if self._is_swing_low(i):
                current_low = float(self.candles.iloc[i]["low"])

                # Check previous swing low
                prev_low = self._find_previous_swing_low(i)
                if prev_low and current_low < prev_low["price"]:
                    # BOS confirmed
                    is_inside_range = current_low > low_price  # Inside original range

                    return {
                        "idx": i,
                        "price": current_low,
                        "is_inside_range": is_inside_range,
                        "prev_swing_low": prev_low
                    }

        return None

    # ================================================================
    # EXTREME LIQUIDITY & DEMAND/SUPPLY DETECTION
    # ================================================================

    def _find_extreme_liquidity_for_accumulation(self, tap2_idx: int, tap2_price: float) -> Optional[Dict]:
        """
        Find extreme liquidity for Model 2 accumulation.

        TCT: "Extreme liquidity is the last liquidity Point remaining before taking
        your second tap low which is your range low"

        TCT: "Often times your extreme liquidity will simply just be your first
        Mark structure low if you pull your mark structure from the second Tap low up"
        """
        # Find first market structure low after Tap2
        for i in range(tap2_idx + 2, min(tap2_idx + 20, len(self.candles) - 2)):
            if self._is_swing_low(i, lookback=3):  # Use smaller lookback for internal structure
                return {
                    "idx": i,
                    "price": float(self.candles.iloc[i]["low"]),
                    "type": "extreme_liquidity",
                    "description": "First market structure low after Tap2"
                }

        return None

    def _find_extreme_liquidity_for_distribution(self, tap2_idx: int, tap2_price: float) -> Optional[Dict]:
        """
        Find extreme liquidity for Model 2 distribution.

        TCT: "What is the first Mark structure High by drawing mark from the top down
        from your second tap High towards the lowest point between tap two and tap three"
        """
        # Find first market structure high after Tap2
        for i in range(tap2_idx + 2, min(tap2_idx + 20, len(self.candles) - 2)):
            if self._is_swing_high(i, lookback=3):  # Use smaller lookback for internal structure
                return {
                    "idx": i,
                    "price": float(self.candles.iloc[i]["high"]),
                    "type": "extreme_liquidity",
                    "description": "First market structure high after Tap2"
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

        # Detect BOS confirmation
        bos = self._detect_bos_confirmation(tap2, tap3, schematic_type)

        # Calculate entry, stop loss, target
        entry_price = bos["bos_price"] if bos else None
        stop_loss = tap3["price"]  # TCT: "Stop loss below your third tap low"
        target = range_data["range_high"]  # TCT: "Target the Range High"

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

        # Detect BOS confirmation
        bos = self._detect_bos_confirmation(tap2, tap3, schematic_type)

        # Calculate entry, stop loss, target
        entry_price = bos["bos_price"] if bos else None
        stop_loss = tap3["price"]  # TCT: "Stop loss above your third tap high"
        target = range_data["range_low"]  # TCT: "Target the Range Low"

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
    # UTILITY METHODS
    # ================================================================

    def _is_swing_high(self, idx: int, lookback: int = None) -> bool:
        """Check if index is a swing high using 6-candle rule."""
        lookback = lookback or self.SIX_CANDLE_LOOKBACK // 2

        if idx < lookback or idx >= len(self.candles) - lookback:
            return False

        current = self.candles.iloc[idx]["high"]

        # Check candles before
        for i in range(idx - lookback, idx):
            if self.candles.iloc[i]["high"] >= current:
                return False

        # Check candles after
        for i in range(idx + 1, idx + lookback + 1):
            if self.candles.iloc[i]["high"] >= current:
                return False

        return True

    def _is_swing_low(self, idx: int, lookback: int = None) -> bool:
        """Check if index is a swing low using 6-candle rule."""
        lookback = lookback or self.SIX_CANDLE_LOOKBACK // 2

        if idx < lookback or idx >= len(self.candles) - lookback:
            return False

        current = self.candles.iloc[idx]["low"]

        # Check candles before
        for i in range(idx - lookback, idx):
            if self.candles.iloc[i]["low"] <= current:
                return False

        # Check candles after
        for i in range(idx + 1, idx + lookback + 1):
            if self.candles.iloc[i]["low"] <= current:
                return False

        return True

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
        """
        start = min(idx1, idx2)
        end = max(idx1, idx2)

        # Check candles after the range formation
        check_start = end + 1
        check_end = min(check_start + 20, len(self.candles))

        for i in range(check_start, check_end):
            candle = self.candles.iloc[i]
            if candle["low"] <= equilibrium <= candle["high"]:
                return True

        return False

    def _validate_deviation_came_back_inside(self, tab: Dict, range_data: Dict, direction: str) -> bool:
        """
        Validate that deviation came back inside range.

        TCT: "We come back inside okay making it a valid deviation"
        """
        tab_idx = tab["idx"]
        range_level = range_data["range_low"] if direction == "low" else range_data["range_high"]

        # Check next few candles for price coming back inside
        for i in range(tab_idx + 1, min(tab_idx + 10, len(self.candles))):
            candle = self.candles.iloc[i]
            if direction == "low" and candle["close"] > range_level:
                return True
            elif direction == "high" and candle["close"] < range_level:
                return True

        return False

    def _validate_deviation_came_back_inside_from_idx(self, idx: int, level: float, direction: str) -> bool:
        """Validate deviation came back inside from a specific index."""
        for i in range(idx + 1, min(idx + 10, len(self.candles))):
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


def detect_tct_schematics(candles: pd.DataFrame, detected_ranges: List[Dict] = None) -> Dict:
    """
    Main entry point for TCT schematic detection.

    Args:
        candles: DataFrame with OHLC data
        detected_ranges: Optional pre-detected ranges

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
        detector = TCTSchematicDetector(candles)
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
