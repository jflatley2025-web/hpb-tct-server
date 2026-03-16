"""
pivot_cache.py — Cached pivot (swing high/low) detection with inside-bar awareness.

Provides a PivotCache that pre-computes swing highs and swing lows for a
candle DataFrame using the TCT 6-candle rule, skipping inside bars.

The "before" and "after" searches dynamically extend until either
``lookback`` non-inside-bar indices are collected or the sequence bounds
are reached, so consecutive inside bars never cause false negatives.
"""

import pandas as pd
from typing import Dict, List, Optional

import logging

logger = logging.getLogger("PivotCache")


class PivotCache:
    """
    Pre-computed pivot cache for a candle DataFrame.

    Parameters
    ----------
    candles : pd.DataFrame
        OHLC data with 'high' and 'low' columns.
    lookback : int
        Number of non-inside-bar candles required on each side of a
        pivot candidate (default 3, i.e. the TCT 6-candle rule with
        3 candles before + 3 candles after).
    """

    def __init__(self, candles: pd.DataFrame, lookback: int = 3) -> None:
        self._candles = candles.reset_index(drop=True)
        self._lookback = lookback
        self._swing_highs: Optional[List[Dict]] = None
        self._swing_lows: Optional[List[Dict]] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def swing_highs(self) -> List[Dict]:
        """List of ``{"idx": int, "price": float}`` for every swing high."""
        if self._swing_highs is None:
            self._compute()
        return self._swing_highs  # type: ignore[return-value]

    @property
    def swing_lows(self) -> List[Dict]:
        """List of ``{"idx": int, "price": float}`` for every swing low."""
        if self._swing_lows is None:
            self._compute()
        return self._swing_lows  # type: ignore[return-value]

    @property
    def candles(self) -> pd.DataFrame:
        return self._candles

    # ------------------------------------------------------------------
    # Inside-bar detection
    # ------------------------------------------------------------------

    def _is_inside_bar(self, idx: int) -> bool:
        """
        Check if candle at *idx* is an inside bar.

        Inside bar: high ≤ previous high AND low ≥ previous low.
        Inside bars do NOT count for the 6-candle rule.
        """
        if idx < 1 or idx >= len(self._candles):
            return False
        curr = self._candles.iloc[idx]
        prev = self._candles.iloc[idx - 1]
        return float(curr["high"]) <= float(prev["high"]) and float(curr["low"]) >= float(prev["low"])

    # ------------------------------------------------------------------
    # Swing high / low checks (dynamic inside-bar skipping)
    # ------------------------------------------------------------------

    def _check_swing_high(self, idx: int) -> bool:
        """
        Return True if *idx* is a swing high per the TCT 6-candle rule.

        Collects ``lookback`` non-inside-bar candles before and after
        *idx* using while-style iteration so that arbitrarily many
        consecutive inside bars are skipped correctly.
        """
        lookback = self._lookback
        n = len(self._candles)

        if idx < 1 or idx >= n - 1:
            return False

        current = float(self._candles.iloc[idx]["high"])

        # Collect non-inside-bar candles BEFORE idx
        before: List[int] = []
        j = idx - 1
        while j >= 0 and len(before) < lookback:
            if not self._is_inside_bar(j):
                before.append(j)
            j -= 1

        if len(before) < lookback:
            return False

        # Collect non-inside-bar candles AFTER idx
        after: List[int] = []
        j = idx + 1
        while j < n and len(after) < lookback:
            if not self._is_inside_bar(j):
                after.append(j)
            j += 1

        if len(after) < lookback:
            return False

        # All collected candles must have strictly lower highs
        for i in before:
            if float(self._candles.iloc[i]["high"]) >= current:
                return False
        for i in after:
            if float(self._candles.iloc[i]["high"]) >= current:
                return False

        return True

    def _check_swing_low(self, idx: int) -> bool:
        """
        Return True if *idx* is a swing low per the TCT 6-candle rule.

        Collects ``lookback`` non-inside-bar candles before and after
        *idx* using while-style iteration so that arbitrarily many
        consecutive inside bars are skipped correctly.
        """
        lookback = self._lookback
        n = len(self._candles)

        if idx < 1 or idx >= n - 1:
            return False

        current = float(self._candles.iloc[idx]["low"])

        # Collect non-inside-bar candles BEFORE idx
        before: List[int] = []
        j = idx - 1
        while j >= 0 and len(before) < lookback:
            if not self._is_inside_bar(j):
                before.append(j)
            j -= 1

        if len(before) < lookback:
            return False

        # Collect non-inside-bar candles AFTER idx
        after: List[int] = []
        j = idx + 1
        while j < n and len(after) < lookback:
            if not self._is_inside_bar(j):
                after.append(j)
            j += 1

        if len(after) < lookback:
            return False

        # All collected candles must have strictly higher lows
        for i in before:
            if float(self._candles.iloc[i]["low"]) <= current:
                return False
        for i in after:
            if float(self._candles.iloc[i]["low"]) <= current:
                return False

        return True

    # ------------------------------------------------------------------
    # Full computation
    # ------------------------------------------------------------------

    def _compute(self) -> None:
        """Scan all candles and cache swing highs / lows."""
        highs: List[Dict] = []
        lows: List[Dict] = []

        for idx in range(len(self._candles)):
            if self._check_swing_high(idx):
                highs.append({"idx": idx, "price": float(self._candles.iloc[idx]["high"])})
            if self._check_swing_low(idx):
                lows.append({"idx": idx, "price": float(self._candles.iloc[idx]["low"])})

        self._swing_highs = highs
        self._swing_lows = lows
        logger.debug(
            "PivotCache computed: %d swing highs, %d swing lows from %d candles",
            len(highs), len(lows), len(self._candles),
        )
