"""
range_engine_l2.py — L2 Range Detection Engine.

Builds on L1 structure detection to find higher-resolution distribution
and accumulation ranges using L2 pivot sequences (lower highs / lower
lows for distribution, higher lows / higher highs for accumulation).

Enforces L1 trend gate: ranges are only emitted when the L1 trend
(AND htf_bias) agrees with the range direction.
"""

import logging
import pandas as pd
from typing import Dict, List, Optional

from pivot_cache import PivotCache
from decision_trees.market_structure_engine import MarketStructureEngine
from range_utils import check_equilibrium_touch

logger = logging.getLogger("RangeEngineL2")

DEVIATION_LIMIT_PERCENT = 0.30
MIN_L2_PIVOTS = 2


class RangeEngineL2:
    """
    L2 range detection engine.

    Uses L1 structure results as a directional gate and finds L2-level
    ranges from lower-high / lower-low (distribution) or higher-low /
    higher-high (accumulation) sequences.

    Parameters
    ----------
    pivot_cache : PivotCache
        Pre-computed pivot cache for the candle data.
    """

    def __init__(self, pivot_cache: PivotCache) -> None:
        self._pc = pivot_cache
        self._ms_engine = MarketStructureEngine()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_distribution_ranges(
        self,
        candles: pd.DataFrame,
        htf_bias: str = "bearish",
    ) -> List[Dict]:
        """
        Detect L2 distribution ranges.

        Returns early if L1 trend is not bearish OR htf_bias is not
        bearish — both must agree for distribution ranges.
        """
        l1_result = self._ms_engine.detect_l1_structure(candles)

        # L1 trend gate: require BOTH L1 trend and htf_bias to be bearish
        if l1_result.trend != "bearish" or htf_bias != "bearish":
            logger.debug(
                "L2 distribution skipped: L1 trend=%s, htf_bias=%s",
                l1_result.trend, htf_bias,
            )
            return []

        highs = self._pc.swing_highs
        lows = self._pc.swing_lows

        lower_highs = _find_l2_lower_highs(highs)
        lower_lows = _find_l2_lower_lows(lows)

        if not lower_highs or not lower_lows:
            return []

        return self._build_ranges(
            candles, lower_highs, lower_lows, direction="distribution",
        )

    def detect_accumulation_ranges(
        self,
        candles: pd.DataFrame,
        htf_bias: str = "bullish",
    ) -> List[Dict]:
        """
        Detect L2 accumulation ranges.

        Returns early if L1 trend is not bullish OR htf_bias is not
        bullish — both must agree for accumulation ranges.
        """
        l1_result = self._ms_engine.detect_l1_structure(candles)

        if l1_result.trend != "bullish" or htf_bias != "bullish":
            logger.debug(
                "L2 accumulation skipped: L1 trend=%s, htf_bias=%s",
                l1_result.trend, htf_bias,
            )
            return []

        highs = self._pc.swing_highs
        lows = self._pc.swing_lows

        higher_lows = _find_l2_higher_lows(lows)
        higher_highs = _find_l2_higher_highs(highs)

        if not higher_lows or not higher_highs:
            return []

        return self._build_ranges(
            candles, higher_highs, higher_lows, direction="accumulation",
        )

    # ------------------------------------------------------------------
    # Range builder
    # ------------------------------------------------------------------

    def _build_ranges(
        self,
        candles: pd.DataFrame,
        high_pivots: List[Dict],
        low_pivots: List[Dict],
        direction: str,
    ) -> List[Dict]:
        ranges: List[Dict] = []

        for hp in high_pivots:
            for lp in low_pivots:
                range_high = hp["price"]
                range_low = lp["price"]

                if range_high <= range_low:
                    continue

                range_high_idx = hp["idx"]
                range_low_idx = lp["idx"]
                range_size = range_high - range_low
                equilibrium = (range_high + range_low) / 2

                eq_touched = check_equilibrium_touch(
                    candles, range_high_idx, range_low_idx, equilibrium,
                    check_between=True, post_range_candles=30,
                )
                if not eq_touched:
                    continue

                ranges.append({
                    "range_high": range_high,
                    "range_low": range_low,
                    "range_high_idx": range_high_idx,
                    "range_low_idx": range_low_idx,
                    "equilibrium": equilibrium,
                    "range_size": range_size,
                    "dl_high": range_high + range_size * DEVIATION_LIMIT_PERCENT,
                    "dl_low": range_low - range_size * DEVIATION_LIMIT_PERCENT,
                    "direction": direction,
                    "engine": "L2",
                })
                break  # first qualifying pair

        return ranges


# ======================================================================
# L2 pivot-sequence helpers
# ======================================================================

def _find_l2_lower_highs(pivots: List[Dict]) -> List[Dict]:
    """Return sub-sequence of consecutively lower highs."""
    if len(pivots) < MIN_L2_PIVOTS:
        return []
    result = [pivots[0]]
    for p in pivots[1:]:
        if p["price"] < result[-1]["price"]:
            result.append(p)
    return result if len(result) >= MIN_L2_PIVOTS else []


def _find_l2_lower_lows(pivots: List[Dict]) -> List[Dict]:
    """Return sub-sequence of consecutively lower lows."""
    if len(pivots) < MIN_L2_PIVOTS:
        return []
    result = [pivots[0]]
    for p in pivots[1:]:
        if p["price"] < result[-1]["price"]:
            result.append(p)
    return result if len(result) >= MIN_L2_PIVOTS else []


def _find_l2_higher_lows(pivots: List[Dict]) -> List[Dict]:
    """Return sub-sequence of consecutively higher lows."""
    if len(pivots) < MIN_L2_PIVOTS:
        return []
    result = [pivots[0]]
    for p in pivots[1:]:
        if p["price"] > result[-1]["price"]:
            result.append(p)
    return result if len(result) >= MIN_L2_PIVOTS else []


def _find_l2_higher_highs(pivots: List[Dict]) -> List[Dict]:
    """Return sub-sequence of consecutively higher highs."""
    if len(pivots) < MIN_L2_PIVOTS:
        return []
    result = [pivots[0]]
    for p in pivots[1:]:
        if p["price"] > result[-1]["price"]:
            result.append(p)
    return result if len(result) >= MIN_L2_PIVOTS else []
