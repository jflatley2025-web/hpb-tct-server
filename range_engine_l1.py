"""
range_engine_l1.py — Legacy L1 Range Engine
Extracts the existing L1 swing-based range detection logic from
tct_schematics.py for use in the dual range engine architecture.

Uses simple swing high/low pivots (L1 primary trend structure) to
identify range boundaries. This is the legacy behavior.
"""

import logging
from typing import Dict, List, Optional
import pandas as pd

from pivot_cache import PivotCache
from range_utils import check_equilibrium_touch

logger = logging.getLogger("RangeEngineL1")

DEVIATION_LIMIT_PERCENT = 0.30  # TCT: 30% of range size for DL


class RangeEngineL1:
    """
    L1 range detection using simple swing high/low pivots.
    This is the legacy range detection extracted from tct_schematics.py.
    """

    def __init__(self, pivot_cache: PivotCache):
        self._pivot_cache = pivot_cache

    def detect_distribution_ranges(self, candles: pd.DataFrame) -> List[Dict]:
        """
        Find potential distribution ranges (trending up, pull from top to bottom).
        TCT: "When we're trending up, we pull our range from top to bottom"
        """
        ranges = []

        for i in range(10, len(candles) - 8):
            if not self._pivot_cache.get_swing_high(i):
                continue

            range_high = float(candles.iloc[i]["high"])
            range_high_idx = i

            for j in range(i + 5, min(i + 50, len(candles) - 5)):
                if not self._pivot_cache.get_swing_low(j):
                    continue

                range_low = float(candles.iloc[j]["low"])
                range_low_idx = j

                if range_high <= range_low * 1.005:
                    continue

                range_size = range_high - range_low
                equilibrium = (range_high + range_low) / 2
                dl_low = range_low - (range_size * DEVIATION_LIMIT_PERCENT)
                dl_high = range_high + (range_size * DEVIATION_LIMIT_PERCENT)

                eq_touched = check_equilibrium_touch(
                    candles, range_high_idx, range_low_idx, equilibrium,
                    check_between=True, post_range_candles=30,
                )

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
                        "direction": "distribution",
                        "engine": "L1",
                    })

        return ranges

    def detect_accumulation_ranges(self, candles: pd.DataFrame) -> List[Dict]:
        """
        Find potential accumulation ranges (trending down, pull from bottom to top).
        TCT: "When we're trending down, we pull our range from bottom to top"
        """
        ranges = []

        for i in range(10, len(candles) - 8):
            if not self._pivot_cache.get_swing_low(i):
                continue

            range_low = float(candles.iloc[i]["low"])
            range_low_idx = i

            for j in range(i + 5, min(i + 50, len(candles) - 5)):
                if not self._pivot_cache.get_swing_high(j):
                    continue

                range_high = float(candles.iloc[j]["high"])
                range_high_idx = j

                if range_high <= range_low * 1.005:
                    continue

                range_size = range_high - range_low
                equilibrium = (range_high + range_low) / 2
                dl_low = range_low - (range_size * DEVIATION_LIMIT_PERCENT)
                dl_high = range_high + (range_size * DEVIATION_LIMIT_PERCENT)

                eq_touched = check_equilibrium_touch(
                    candles, range_low_idx, range_high_idx, equilibrium,
                    check_between=True, post_range_candles=30,
                )

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
                        "direction": "accumulation",
                        "engine": "L1",
                    })

        return ranges
