"""
pivot_cache.py — Centralized Pivot Engine
Computes pivots ONCE per candle batch. All modules (range detection, BOS,
sweep detection, tap detection) consume from this cache to eliminate
structural drift where each module independently recomputes pivots and
arrives at different sets.

Uses TCT 6-candle rule with inside bar exclusion for swing detection.
"""

import logging
from typing import Dict, List, Optional
import pandas as pd

logger = logging.getLogger("PivotCache")


class PivotCache:
    """
    Centralized pivot cache. Computed once, shared by all consumers.

    Consumers:
    - RangeEngineL1 / RangeEngineL2 (range boundary detection)
    - _find_bearish_bos / _find_bullish_bos (BOS swing detection)
    - _find_distribution_tap2/tap3 (tap identification)
    - _validate_distribution_sweep (liquidity pool detection)
    """

    def __init__(self, candles: pd.DataFrame, lookback: int = 3):
        self._candles = candles.reset_index(drop=True)
        self._default_lookback = lookback
        self._pivot_highs: Dict[int, List[Dict]] = {}  # lookback -> [{idx, price}]
        self._pivot_lows: Dict[int, List[Dict]] = {}   # lookback -> [{idx, price}]
        self._inside_bars: Optional[set] = None
        self._computed = False

        # Eagerly compute for default lookback
        self._compute_inside_bars()
        self._compute_pivots(lookback)
        self._computed = True

    def _compute_inside_bars(self):
        """Pre-compute inside bar set for the entire candle array."""
        self._inside_bars = set()
        for idx in range(1, len(self._candles)):
            curr = self._candles.iloc[idx]
            prev = self._candles.iloc[idx - 1]
            if float(curr["high"]) <= float(prev["high"]) and float(curr["low"]) >= float(prev["low"]):
                self._inside_bars.add(idx)

    def _is_inside_bar(self, idx: int) -> bool:
        if self._inside_bars is None:
            self._compute_inside_bars()
        return idx in self._inside_bars

    def _compute_pivots(self, lookback: int):
        """
        Compute all pivot highs and lows for a given lookback using TCT 6-candle rule.
        Inside bars are excluded from the count per TCT Lecture 1.
        """
        if lookback in self._pivot_highs:
            return  # Already computed

        highs = []
        lows = []

        for idx in range(lookback, len(self._candles) - lookback):
            if self._check_swing_high(idx, lookback):
                highs.append({
                    "idx": idx,
                    "price": float(self._candles.iloc[idx]["high"])
                })
            if self._check_swing_low(idx, lookback):
                lows.append({
                    "idx": idx,
                    "price": float(self._candles.iloc[idx]["low"])
                })

        self._pivot_highs[lookback] = highs
        self._pivot_lows[lookback] = lows
        logger.debug(
            f"PivotCache: computed lookback={lookback} — "
            f"{len(highs)} pivot highs, {len(lows)} pivot lows"
        )

    def _check_swing_high(self, idx: int, lookback: int) -> bool:
        """
        TCT 6-candle rule swing high: requires `lookback` non-inside-bar candles
        on each side with lower highs. Inside bars are excluded from count.
        """
        if idx < lookback or idx >= len(self._candles) - lookback:
            return False

        current = float(self._candles.iloc[idx]["high"])

        # Collect non-inside-bar candles before idx
        before = []
        for i in range(idx - 1, max(idx - lookback - 3, -1), -1):
            if not self._is_inside_bar(i):
                before.append(i)
            if len(before) >= lookback:
                break

        # Collect non-inside-bar candles after idx
        after = []
        for i in range(idx + 1, min(idx + lookback + 3, len(self._candles))):
            if not self._is_inside_bar(i):
                after.append(i)
            if len(after) >= lookback:
                break

        if len(before) < lookback or len(after) < lookback:
            return False

        for i in before:
            if float(self._candles.iloc[i]["high"]) >= current:
                return False
        for i in after:
            if float(self._candles.iloc[i]["high"]) >= current:
                return False

        return True

    def _check_swing_low(self, idx: int, lookback: int) -> bool:
        """
        TCT 6-candle rule swing low: requires `lookback` non-inside-bar candles
        on each side with higher lows. Inside bars are excluded from count.
        """
        if idx < lookback or idx >= len(self._candles) - lookback:
            return False

        current = float(self._candles.iloc[idx]["low"])

        before = []
        for i in range(idx - 1, max(idx - lookback - 3, -1), -1):
            if not self._is_inside_bar(i):
                before.append(i)
            if len(before) >= lookback:
                break

        after = []
        for i in range(idx + 1, min(idx + lookback + 3, len(self._candles))):
            if not self._is_inside_bar(i):
                after.append(i)
            if len(after) >= lookback:
                break

        if len(before) < lookback or len(after) < lookback:
            return False

        for i in before:
            if float(self._candles.iloc[i]["low"]) <= current:
                return False
        for i in after:
            if float(self._candles.iloc[i]["low"]) <= current:
                return False

        return True

    # ================================================================
    # PUBLIC API
    # ================================================================

    def get_pivot_highs(self, lookback: int = None) -> List[Dict]:
        """Return all pivot highs for the given lookback. Cached."""
        lb = lookback or self._default_lookback
        if lb not in self._pivot_highs:
            self._compute_pivots(lb)
        return self._pivot_highs[lb]

    def get_pivot_lows(self, lookback: int = None) -> List[Dict]:
        """Return all pivot lows for the given lookback. Cached."""
        lb = lookback or self._default_lookback
        if lb not in self._pivot_lows:
            self._compute_pivots(lb)
        return self._pivot_lows[lb]

    def get_swing_high(self, idx: int, lookback: int = None) -> bool:
        """Check if a specific index is a swing high. Uses cached pivots."""
        lb = lookback or self._default_lookback
        if lb not in self._pivot_highs:
            self._compute_pivots(lb)
        return any(p["idx"] == idx for p in self._pivot_highs[lb])

    def get_swing_low(self, idx: int, lookback: int = None) -> bool:
        """Check if a specific index is a swing low. Uses cached pivots."""
        lb = lookback or self._default_lookback
        if lb not in self._pivot_lows:
            self._compute_pivots(lb)
        return any(p["idx"] == idx for p in self._pivot_lows[lb])

    def get_swing_lows_in_range(self, start_idx: int, end_idx: int,
                                 lookback: int = None) -> List[Dict]:
        """Return all swing lows between start_idx and end_idx (inclusive)."""
        lb = lookback or self._default_lookback
        if lb not in self._pivot_lows:
            self._compute_pivots(lb)
        return [p for p in self._pivot_lows[lb]
                if start_idx <= p["idx"] <= end_idx]

    def get_swing_highs_in_range(self, start_idx: int, end_idx: int,
                                  lookback: int = None) -> List[Dict]:
        """Return all swing highs between start_idx and end_idx (inclusive)."""
        lb = lookback or self._default_lookback
        if lb not in self._pivot_highs:
            self._compute_pivots(lb)
        return [p for p in self._pivot_highs[lb]
                if start_idx <= p["idx"] <= end_idx]

    def invalidate(self):
        """Force recompute on next access (e.g. after new candles arrive)."""
        self._pivot_highs.clear()
        self._pivot_lows.clear()
        self._inside_bars = None
        self._computed = False

    @property
    def candles(self) -> pd.DataFrame:
        return self._candles

    def __repr__(self):
        total_highs = sum(len(v) for v in self._pivot_highs.values())
        total_lows = sum(len(v) for v in self._pivot_lows.values())
        return (
            f"PivotCache(candles={len(self._candles)}, "
            f"highs={total_highs}, lows={total_lows}, "
            f"lookbacks={list(self._pivot_highs.keys())})"
        )
