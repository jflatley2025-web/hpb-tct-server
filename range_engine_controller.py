"""
range_engine_controller.py — Controller that manages L1 and L2 range engines.

Lazily initialises engines on first use and detects pivot-cache mismatches
so engines always operate on the current cache.
"""

import logging
from typing import Dict, List, Optional

import pandas as pd

from pivot_cache import PivotCache
from range_engine_l1 import RangeEngineL1
from range_engine_l2 import RangeEngineL2

logger = logging.getLogger("RangeEngineController")


class RangeEngineController:
    """
    Manages L1 and L2 range engines with lazy initialisation.

    Parameters
    ----------
    pivot_cache : PivotCache, optional
        Initial pivot cache.  If ``None`` the engines are created on
        the first call to :meth:`detect_ranges` using the cache passed
        at that time.
    """

    def __init__(self, pivot_cache: Optional[PivotCache] = None) -> None:
        self._pivot_cache: Optional[PivotCache] = pivot_cache
        self._l1: Optional[RangeEngineL1] = None
        self._l2: Optional[RangeEngineL2] = None

        if pivot_cache is not None:
            self._l1 = RangeEngineL1(pivot_cache)
            self._l2 = RangeEngineL2(pivot_cache)

    # ------------------------------------------------------------------
    # Engine lifecycle
    # ------------------------------------------------------------------

    def _ensure_engines(self, pivot_cache: PivotCache) -> None:
        """
        Lazily create or recreate engines.

        If *pivot_cache* differs from the currently stored cache the
        engines are rebuilt so they always use the current data.
        """
        if self._pivot_cache is not pivot_cache:
            logger.debug("Pivot cache changed — recreating L1/L2 engines")
            self._pivot_cache = pivot_cache
            self._l1 = RangeEngineL1(pivot_cache)
            self._l2 = RangeEngineL2(pivot_cache)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_ranges(
        self,
        candles: pd.DataFrame,
        pivot_cache: PivotCache,
        htf_bias: str = "neutral",
    ) -> Dict[str, List[Dict]]:
        """
        Run both L1 and L2 detection and return combined results.

        Parameters
        ----------
        candles : pd.DataFrame
            OHLC candle data.
        pivot_cache : PivotCache
            Current pivot cache (triggers engine rebuild on mismatch).
        htf_bias : str
            Higher-timeframe bias ("bullish", "bearish", "neutral").

        Returns
        -------
        dict
            ``{"l1_accumulation": [...], "l1_distribution": [...],
              "l2_accumulation": [...], "l2_distribution": [...]}``
        """
        self._ensure_engines(pivot_cache)
        assert self._l1 is not None and self._l2 is not None

        l1_acc = self._l1.detect_accumulation_ranges()
        l1_dist = self._l1.detect_distribution_ranges()

        l2_acc = self._l2.detect_accumulation_ranges(candles, htf_bias=htf_bias)
        l2_dist = self._l2.detect_distribution_ranges(candles, htf_bias=htf_bias)

        return {
            "l1_accumulation": l1_acc,
            "l1_distribution": l1_dist,
            "l2_accumulation": l2_acc,
            "l2_distribution": l2_dist,
        }
