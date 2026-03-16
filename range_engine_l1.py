"""
range_engine_l1.py — L1 Range Detection Engine.

Detects distribution and accumulation ranges using the TCT methodology
with pivot-cache-aware swing detection and shared equilibrium-touch
validation from range_utils.
"""

import logging
import pandas as pd
from typing import Dict, List, Optional

from pivot_cache import PivotCache
from range_utils import check_equilibrium_touch

logger = logging.getLogger("RangeEngineL1")

# TCT deviation limit: 30 % of range size
DEVIATION_LIMIT_PERCENT = 0.30


class RangeEngineL1:
    """
    L1 (primary) range detection engine.

    Uses a shared PivotCache to find swing highs/lows and detects
    accumulation and distribution ranges with equilibrium-touch
    confirmation.

    Parameters
    ----------
    pivot_cache : PivotCache
        Pre-computed pivot cache for the candle data.
    """

    def __init__(self, pivot_cache: PivotCache) -> None:
        self._pc = pivot_cache

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_accumulation_ranges(self) -> List[Dict]:
        """
        Detect accumulation ranges (downtrend → consolidation).

        TCT: "When we're trending down, we pull our range from bottom to top."
        """
        candles = self._pc.candles
        swing_lows = self._pc.swing_lows
        swing_highs = self._pc.swing_highs
        ranges: List[Dict] = []

        for sl in swing_lows:
            range_low = sl["price"]
            range_low_idx = sl["idx"]

            # Find next swing high after this low
            for sh in swing_highs:
                if sh["idx"] <= range_low_idx:
                    continue

                range_high = sh["price"]
                range_high_idx = sh["idx"]

                if range_high <= range_low:
                    continue

                range_size = range_high - range_low
                equilibrium = (range_high + range_low) / 2

                # Confirm with equilibrium touch
                eq_touched = check_equilibrium_touch(
                    candles, range_low_idx, range_high_idx, equilibrium,
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
                    "direction": "accumulation",
                })
                break  # first qualifying high per low

        return ranges

    def detect_distribution_ranges(self) -> List[Dict]:
        """
        Detect distribution ranges (uptrend → consolidation).

        TCT: "When we're trending up, we pull our range from top to bottom."
        """
        candles = self._pc.candles
        swing_highs = self._pc.swing_highs
        swing_lows = self._pc.swing_lows
        ranges: List[Dict] = []

        for sh in swing_highs:
            range_high = sh["price"]
            range_high_idx = sh["idx"]

            # Find next swing low after this high
            for sl in swing_lows:
                if sl["idx"] <= range_high_idx:
                    continue

                range_low = sl["price"]
                range_low_idx = sl["idx"]

                if range_high <= range_low:
                    continue

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
                    "direction": "distribution",
                })
                break  # first qualifying low per high

        return ranges
