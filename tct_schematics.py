"""
tct_schematics.py — TCT Schematics Detection (Lecture 5A Pure TCT Methodology)
Author: HPB-TCT Dev Team
Date: 2026-01-26

Detects TCT Schematics (Accumulation & Distribution Models 1 and 2) based on:
- TCT 2024 mentorship - Lecture 5A | TCT schematics
- Wyckoff methodology simplified with TCT rules

TCT Schematic Key Concepts:
- Model 1: Range → Deviation 1 → Deviation 2 (each lower/higher than previous)
- Model 2: Range → Deviation 1 → Higher Low/Lower High (grabs extreme liquidity OR extreme S/D)
- Three-tap model: Tap1 (range), Tap2 (first deviation), Tap3 (second deviation or HL/LH)
- Entry: Break of structure from highest/lowest point between Tap2 and Tap3
- Target: Opposite range extreme (Wyckoff High/Low)
- Stop Loss: Below/Above Tap3

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
    Main TCT Schematic Detection Engine implementing Lecture 5A methodology.

    Pure TCT Methodology:
    - Model 1 Accumulation: Tap1 (range low) → Tap2 (deviation lower) → Tap3 (deviation even lower)
    - Model 2 Accumulation: Tap1 (range low) → Tap2 (deviation) → Tap3 (higher low at extreme liq/demand)
    - Model 1 Distribution: Tap1 (range high) → Tap2 (deviation higher) → Tap3 (deviation even higher)
    - Model 2 Distribution: Tap1 (range high) → Tap2 (deviation) → Tap3 (lower high at extreme liq/supply)

    Entry Confirmation:
    - Watch structure from highest/lowest point between Tap2 and Tap3
    - When structure breaks back to bullish/bearish = entry confirmation
    - Preferably BOS inside original range values (safer entry)
    """

    DEVIATION_LIMIT_PERCENT = 0.30  # TCT: 30% of range size for DL
    SIX_CANDLE_LOOKBACK = 6  # Minimum candles for pivot validation

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

        TCT: "You enter on the break you put your stop loss below your third tap low
        and you target the Range High"
        """
        schematic_type = f"{model_type}_Accumulation"

        # Detect BOS confirmation
        bos = self._detect_bos_confirmation(tap2, tap3, schematic_type)

        # Calculate entry, stop loss, target
        entry_price = bos["bos_price"] if bos else None
        stop_loss = tap3["price"]  # TCT: "Stop loss below your third tap low"
        target = range_data["range_high"]  # TCT: "Target the Range High"

        # Calculate risk/reward
        risk_reward = None
        if entry_price and stop_loss and target:
            risk = entry_price - stop_loss
            reward = target - entry_price
            if risk > 0:
                risk_reward = round(reward / risk, 2)

        # Calculate quality score
        quality_score = self._calculate_schematic_quality(
            range_data, tap1, tap2, tap3, bos, model_type
        )

        # Validate six candle rule
        six_candle_valid = self._validate_six_candle_rule_on_tabs(tap1, tap2, tap3)

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
                "description": "Enter on break of structure back to bullish"
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
            "timestamp": datetime.utcnow().isoformat()
        }

    def _build_distribution_schematic(self, range_data: Dict, tap1: Dict, tap2: Dict,
                                       tap3: Dict, model_type: str) -> Optional[Dict]:
        """
        Build complete distribution schematic with entry, stop loss, and target.

        TCT: "Once we do break that we put our stop loss above our Tap high
        and we target the range low"
        """
        schematic_type = f"{model_type}_Distribution"

        # Detect BOS confirmation
        bos = self._detect_bos_confirmation(tap2, tap3, schematic_type)

        # Calculate entry, stop loss, target
        entry_price = bos["bos_price"] if bos else None
        stop_loss = tap3["price"]  # TCT: "Stop loss above your third tap high"
        target = range_data["range_low"]  # TCT: "Target the Range Low"

        # Calculate risk/reward
        risk_reward = None
        if entry_price and stop_loss and target:
            risk = stop_loss - entry_price
            reward = entry_price - target
            if risk > 0:
                risk_reward = round(reward / risk, 2)

        # Calculate quality score
        quality_score = self._calculate_schematic_quality(
            range_data, tap1, tap2, tap3, bos, model_type
        )

        # Validate six candle rule
        six_candle_valid = self._validate_six_candle_rule_on_tabs(tap1, tap2, tap3)

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
                "description": "Enter on break of structure back to bearish"
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
            "timestamp": datetime.utcnow().isoformat()
        }

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
        Calculate schematic quality score.

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
    print("TCT SCHEMATIC DETECTOR TEST (Lecture 5A Methodology)")
    print("=" * 60)

    # Generate test data simulating accumulation pattern
    np.random.seed(42)
    dates = pd.date_range('2026-01-01', periods=200, freq='1H')

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

    for i, schematic in enumerate(result['distribution_schematics'][:3], 1):
        print(f"\n  Distribution #{i}: {schematic['schematic_type']}")
        print(f"    Quality Score: {schematic['quality_score']}")
        print(f"    Confirmed: {schematic['is_confirmed']}")
        print(f"    R:R = {schematic['risk_reward']}")

    print(f"\n{'=' * 60}")
    print("TEST COMPLETE")
    print(f"{'=' * 60}\n")
