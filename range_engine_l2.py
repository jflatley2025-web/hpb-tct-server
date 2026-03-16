"""
range_engine_l2.py — TCT-Correct L2 Range Engine
==================================================
Pulls ranges from L2 structure pools per TCT methodology. L2 = counter-trend
pullback structure within the primary L1 trend.

For distribution (L1 bullish): L2 is bearish counter-structure.
Range high = last L2 lower-high pivot, range low = last L2 lower-low pivot.

Critical rule: minimum range duration is 24 hours. Ranges shorter than 24h
are classified as micro-ranges and excluded from HTF detection.
"""

import logging
from typing import Dict, List, Optional, Tuple
import pandas as pd

from pivot_cache import PivotCache
from decision_trees.market_structure_engine import MarketStructureEngine

logger = logging.getLogger("RangeEngineL2")

DEVIATION_LIMIT_PERCENT = 0.30  # TCT: 30% of range size for DL
MINIMUM_RANGE_DURATION_HOURS = 24  # HTF ranges must be >= 24h


class RangeEngineL2:
    """
    L2 range detection using counter-trend structure pools.
    TCT: "Range pulls must use Level 2 structure pools, not Level 1."
    """

    def __init__(self, pivot_cache: PivotCache):
        self._pivot_cache = pivot_cache
        self._ms_engine = MarketStructureEngine()

    def detect_distribution_ranges(self, candles: pd.DataFrame,
                                    htf_bias: str = "bullish") -> List[Dict]:
        """
        Find distribution ranges using L2 counter-structure.

        Algorithm:
        1. Confirm L1 trend is bullish
        2. Detect L2 bearish counter-structure
        3. Find L2 lower-high and lower-low pivots
        4. Validate duration >= 24h
        5. Confirm equilibrium touch
        """
        ranges = []

        # Step 1: Confirm L1 trend
        l1_result = self._ms_engine.detect_l1_structure(candles)
        if l1_result.trend != "bullish" and htf_bias != "bullish":
            logger.debug("L2 distribution: L1 trend not bullish, skipping")
            return ranges

        # Step 2: Detect L2 counter-structure
        l2_result = self._ms_engine.detect_l2_structure(candles, "bullish")
        if not l2_result["exists"]:
            logger.debug("L2 distribution: no bearish counter-structure found")
            return ranges

        # Step 3: Find L2 pivots using the centralized pivot cache
        # For distribution (bullish L1), L2 counter-structure means
        # bearish moves within the bullish trend — lower-highs and lower-lows
        pivot_highs = self._pivot_cache.get_pivot_highs()
        pivot_lows = self._pivot_cache.get_pivot_lows()

        # Find L2 lower-high sequences (counter-trend pullback highs)
        l2_lower_highs = self._find_l2_lower_highs(pivot_highs)
        l2_lower_lows = self._find_l2_lower_lows(pivot_lows)

        if not l2_lower_highs or not l2_lower_lows:
            # Fall back to regular pivots with L2 counter-structure validation
            l2_lower_highs = pivot_highs[-5:] if len(pivot_highs) >= 5 else pivot_highs
            l2_lower_lows = pivot_lows[-5:] if len(pivot_lows) >= 5 else pivot_lows

        # Step 4: Build ranges from L2 pivot pairs
        has_time = "open_time" in candles.columns

        for ph in reversed(l2_lower_highs):
            ph_idx = ph["idx"]
            ph_price = ph["price"]

            for pl in l2_lower_lows:
                pl_idx = pl["idx"]
                pl_price = pl["price"]

                # Range low must come after range high for distribution
                if pl_idx <= ph_idx + 3:
                    continue

                if ph_price <= pl_price * 1.005:
                    continue

                # Duration validation (24h minimum)
                if has_time:
                    duration_ok = self._validate_duration(
                        candles, ph_idx, pl_idx
                    )
                    if not duration_ok:
                        continue
                else:
                    # Without timestamps, require minimum candle gap
                    if pl_idx - ph_idx < 5:
                        continue

                range_size = ph_price - pl_price
                equilibrium = (ph_price + pl_price) / 2
                dl_low = pl_price - (range_size * DEVIATION_LIMIT_PERCENT)
                dl_high = ph_price + (range_size * DEVIATION_LIMIT_PERCENT)

                # Equilibrium touch confirmation
                eq_touched = self._check_equilibrium_touch(
                    candles, ph_idx, pl_idx, equilibrium
                )
                if not eq_touched:
                    continue

                ranges.append({
                    "range_high": ph_price,
                    "range_low": pl_price,
                    "range_high_idx": ph_idx,
                    "range_low_idx": pl_idx,
                    "equilibrium": equilibrium,
                    "range_size": range_size,
                    "dl_high": dl_high,
                    "dl_low": dl_low,
                    "direction": "distribution",
                    "engine": "L2",
                    "l1_trend": l1_result.trend,
                    "l2_counter": l2_result["exists"],
                })

        return ranges

    def detect_accumulation_ranges(self, candles: pd.DataFrame,
                                    htf_bias: str = "bearish") -> List[Dict]:
        """
        Find accumulation ranges using L2 counter-structure.
        For accumulation (L1 bearish): L2 is bullish counter-structure.
        """
        ranges = []

        l1_result = self._ms_engine.detect_l1_structure(candles)
        if l1_result.trend != "bearish" and htf_bias != "bearish":
            return ranges

        l2_result = self._ms_engine.detect_l2_structure(candles, "bearish")
        if not l2_result["exists"]:
            return ranges

        pivot_highs = self._pivot_cache.get_pivot_highs()
        pivot_lows = self._pivot_cache.get_pivot_lows()

        l2_higher_lows = self._find_l2_higher_lows(pivot_lows)
        l2_higher_highs = self._find_l2_higher_highs(pivot_highs)

        if not l2_higher_lows or not l2_higher_highs:
            l2_higher_lows = pivot_lows[-5:] if len(pivot_lows) >= 5 else pivot_lows
            l2_higher_highs = pivot_highs[-5:] if len(pivot_highs) >= 5 else pivot_highs

        has_time = "open_time" in candles.columns

        for pl in reversed(l2_higher_lows):
            pl_idx = pl["idx"]
            pl_price = pl["price"]

            for ph in l2_higher_highs:
                ph_idx = ph["idx"]
                ph_price = ph["price"]

                if ph_idx <= pl_idx + 3:
                    continue

                if ph_price <= pl_price * 1.005:
                    continue

                if has_time:
                    if not self._validate_duration(candles, pl_idx, ph_idx):
                        continue
                else:
                    if ph_idx - pl_idx < 5:
                        continue

                range_size = ph_price - pl_price
                equilibrium = (ph_price + pl_price) / 2
                dl_low = pl_price - (range_size * DEVIATION_LIMIT_PERCENT)
                dl_high = ph_price + (range_size * DEVIATION_LIMIT_PERCENT)

                eq_touched = self._check_equilibrium_touch(
                    candles, pl_idx, ph_idx, equilibrium
                )
                if not eq_touched:
                    continue

                ranges.append({
                    "range_high": ph_price,
                    "range_low": pl_price,
                    "range_high_idx": ph_idx,
                    "range_low_idx": pl_idx,
                    "equilibrium": equilibrium,
                    "range_size": range_size,
                    "dl_high": dl_high,
                    "dl_low": dl_low,
                    "direction": "accumulation",
                    "engine": "L2",
                    "l1_trend": self._ms_engine.detect_l1_structure(candles).trend,
                    "l2_counter": True,
                })

        return ranges

    # ================================================================
    # L2 PIVOT IDENTIFICATION
    # ================================================================

    def _find_l2_lower_highs(self, pivot_highs: List[Dict]) -> List[Dict]:
        """Find consecutive lower-high pivots (bearish counter-structure)."""
        if len(pivot_highs) < 2:
            return pivot_highs

        result = []
        for i in range(1, len(pivot_highs)):
            if pivot_highs[i]["price"] < pivot_highs[i - 1]["price"]:
                if not result:
                    result.append(pivot_highs[i - 1])
                result.append(pivot_highs[i])

        return result if len(result) >= 2 else []

    def _find_l2_lower_lows(self, pivot_lows: List[Dict]) -> List[Dict]:
        """Find consecutive lower-low pivots (bearish counter-structure)."""
        if len(pivot_lows) < 2:
            return pivot_lows

        result = []
        for i in range(1, len(pivot_lows)):
            if pivot_lows[i]["price"] < pivot_lows[i - 1]["price"]:
                if not result:
                    result.append(pivot_lows[i - 1])
                result.append(pivot_lows[i])

        return result if len(result) >= 2 else []

    def _find_l2_higher_lows(self, pivot_lows: List[Dict]) -> List[Dict]:
        """Find consecutive higher-low pivots (bullish counter-structure)."""
        if len(pivot_lows) < 2:
            return pivot_lows

        result = []
        for i in range(1, len(pivot_lows)):
            if pivot_lows[i]["price"] > pivot_lows[i - 1]["price"]:
                if not result:
                    result.append(pivot_lows[i - 1])
                result.append(pivot_lows[i])

        return result if len(result) >= 2 else []

    def _find_l2_higher_highs(self, pivot_highs: List[Dict]) -> List[Dict]:
        """Find consecutive higher-high pivots (bullish counter-structure)."""
        if len(pivot_highs) < 2:
            return pivot_highs

        result = []
        for i in range(1, len(pivot_highs)):
            if pivot_highs[i]["price"] > pivot_highs[i - 1]["price"]:
                if not result:
                    result.append(pivot_highs[i - 1])
                result.append(pivot_highs[i])

        return result if len(result) >= 2 else []

    # ================================================================
    # VALIDATION
    # ================================================================

    def _validate_duration(self, candles: pd.DataFrame,
                           start_idx: int, end_idx: int) -> bool:
        """
        Validate range duration is >= 24 hours.
        Ranges < 24h are micro-ranges and excluded from HTF detection.
        """
        try:
            start_time = pd.Timestamp(candles.iloc[start_idx]["open_time"])
            end_time = pd.Timestamp(candles.iloc[end_idx]["open_time"])
            duration_hours = (end_time - start_time).total_seconds() / 3600
            return duration_hours >= MINIMUM_RANGE_DURATION_HOURS
        except (KeyError, TypeError, ValueError):
            # If timestamps unavailable, fall back to candle count heuristic
            return (end_idx - start_idx) >= 5

    @staticmethod
    def _check_equilibrium_touch(candles: pd.DataFrame, idx1: int, idx2: int,
                                  equilibrium: float) -> bool:
        """Check if price touched equilibrium to confirm range."""
        start = min(idx1, idx2)
        end = max(idx1, idx2)

        for i in range(start + 1, end):
            candle = candles.iloc[i]
            if candle["low"] <= equilibrium <= candle["high"]:
                return True

        check_start = end + 1
        check_end = min(check_start + 30, len(candles))
        for i in range(check_start, check_end):
            candle = candles.iloc[i]
            if candle["low"] <= equilibrium <= candle["high"]:
                return True

        return False
