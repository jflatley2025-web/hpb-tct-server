"""
range_engine_controller.py — Range Engine Feature Flag Controller
Controls which range engine (L1 legacy vs L2 TCT-correct) is used.

Modes:
- L1:        L1 engine only (legacy behavior)
- L2:        L2 engine primary, L1 fallback when L2 returns zero ranges
- compare:   Both engines run, L2 used for trading, diffs logged
- strict_L2: L2 only, NO silent fallback, warnings on zero ranges
"""

import os
import logging
from typing import Dict, List, Optional
import pandas as pd

from pivot_cache import PivotCache
from range_engine_l1 import RangeEngineL1
from range_engine_l2 import RangeEngineL2
from range_comparison_logger import RangeComparisonLogger
from session_manipulation import get_active_session

logger = logging.getLogger("RangeEngineController")

VALID_MODES = {"L1", "L2", "compare", "strict_L2"}


class RangeEngineController:
    """
    Feature flag controller for dual range engine architecture.
    Read mode from RANGE_ENGINE_MODE env var (default: "compare").
    """

    def __init__(self, mode: str = None, pivot_cache: PivotCache = None):
        self._mode = mode or os.getenv("RANGE_ENGINE_MODE", "compare")
        if self._mode not in VALID_MODES:
            logger.warning(f"Invalid mode '{self._mode}', defaulting to 'compare'")
            self._mode = "compare"

        self._pivot_cache = pivot_cache
        self._l1: Optional[RangeEngineL1] = None
        self._l2: Optional[RangeEngineL2] = None
        self._logger = RangeComparisonLogger()

    def _ensure_engines(self, pivot_cache: PivotCache):
        """Lazily initialize engines when pivot_cache is available.
        Reinitialize if a different pivot_cache is provided."""
        if pivot_cache is not self._pivot_cache:
            self._pivot_cache = pivot_cache
            self._l1 = RangeEngineL1(self._pivot_cache)
            self._l2 = RangeEngineL2(self._pivot_cache)
            return
        if self._l1 is None:
            self._l1 = RangeEngineL1(self._pivot_cache)
        if self._l2 is None:
            self._l2 = RangeEngineL2(self._pivot_cache)

    def set_mode(self, mode: str):
        """Runtime mode switching."""
        if mode not in VALID_MODES:
            raise ValueError(f"Invalid mode: {mode}. Must be one of {VALID_MODES}")
        self._mode = mode
        logger.info(f"Range engine mode set to: {mode}")

    @property
    def mode(self) -> str:
        return self._mode

    def detect_ranges(
        self,
        candles: pd.DataFrame,
        direction: str,
        htf_bias: str = "bullish",
        pivot_cache: PivotCache = None,
        symbol: str = "BTCUSDT",
    ) -> List[Dict]:
        """
        Detect ranges using the configured engine mode.

        Args:
            candles: OHLC candle data
            direction: "distribution" or "accumulation"
            htf_bias: HTF directional context
            pivot_cache: Centralized pivot cache (required if not set in __init__)
            symbol: Trading symbol for logging
        """
        pc = pivot_cache or self._pivot_cache
        if pc is None:
            raise ValueError("PivotCache required — pass via constructor or detect_ranges()")
        self._ensure_engines(pc)

        if self._mode == "L1":
            return self._detect_l1(candles, direction)

        elif self._mode == "L2":
            return self._detect_l2_with_fallback(candles, direction, htf_bias, symbol)

        elif self._mode == "compare":
            return self._detect_compare(candles, direction, htf_bias, symbol)

        elif self._mode == "strict_L2":
            return self._detect_strict_l2(candles, direction, htf_bias, symbol)

        return []

    def _detect_l1(self, candles: pd.DataFrame, direction: str) -> List[Dict]:
        """L1 only (legacy)."""
        if direction == "distribution":
            return self._l1.detect_distribution_ranges(candles)
        return self._l1.detect_accumulation_ranges(candles)

    def _detect_l2_with_fallback(self, candles: pd.DataFrame, direction: str,
                                  htf_bias: str, symbol: str) -> List[Dict]:
        """L2 primary, L1 fallback when L2 finds zero ranges."""
        if direction == "distribution":
            l2_ranges = self._l2.detect_distribution_ranges(candles, htf_bias)
        else:
            l2_ranges = self._l2.detect_accumulation_ranges(candles, htf_bias)

        if l2_ranges:
            return l2_ranges

        # Fallback to L1
        logger.info(f"[L2] No ranges found, falling back to L1 | {symbol} {direction}")
        return self._detect_l1(candles, direction)

    def _detect_compare(self, candles: pd.DataFrame, direction: str,
                         htf_bias: str, symbol: str) -> List[Dict]:
        """Both engines run. L2 used for trading. Diffs logged."""
        l1_ranges = self._detect_l1(candles, direction)

        if direction == "distribution":
            l2_ranges = self._l2.detect_distribution_ranges(candles, htf_bias)
        else:
            l2_ranges = self._l2.detect_accumulation_ranges(candles, htf_bias)

        # Log comparison
        session = get_active_session()
        self._logger.log_comparison(
            symbol=symbol,
            session=session,
            engine_used="L2" if l2_ranges else "L1_fallback",
            l1_ranges=l1_ranges,
            l2_ranges=l2_ranges,
        )

        # Use L2 results for trading; fall back to L1 if L2 empty
        if l2_ranges:
            return l2_ranges
        return l1_ranges

    def _detect_strict_l2(self, candles: pd.DataFrame, direction: str,
                           htf_bias: str, symbol: str) -> List[Dict]:
        """
        L2 only. NO silent fallback. Warns when L2 finds zero ranges.
        """
        if direction == "distribution":
            l2_ranges = self._l2.detect_distribution_ranges(candles, htf_bias)
        else:
            l2_ranges = self._l2.detect_accumulation_ranges(candles, htf_bias)

        if not l2_ranges:
            # Log structural context for debugging — no fallback
            from decision_trees.market_structure_engine import MarketStructureEngine
            ms = MarketStructureEngine()
            l1_result = ms.detect_l1_structure(candles)
            l2_result = ms.detect_l2_structure(candles, htf_bias)

            logger.warning(
                f"[strict_L2] No L2 ranges detected | symbol={symbol} | "
                f"direction={direction} | l1_trend={l1_result.trend} | "
                f"l2_exists={l2_result['exists']} | candles={len(candles)} | "
                f"htf_bias={htf_bias}"
            )
            return []

        return l2_ranges
