"""Pivot detector — finds swing highs and lows in OHLCV data.

This is the foundation for all schematic suggestions.
A pivot high is a local maximum surrounded by lower highs.
A pivot low is a local minimum surrounded by higher lows.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import pandas as pd


@dataclass
class Pivot:
    """A detected swing point in the price series."""
    time: datetime
    price: float
    bar_index: int          # index within the DataFrame
    pivot_type: str         # "high" or "low"
    strength: int           # how many bars on each side confirm this pivot
    prominence: float       # price distance from nearest opposing pivot (0 if unknown)

    @property
    def is_high(self) -> bool:
        return self.pivot_type == "high"

    @property
    def is_low(self) -> bool:
        return self.pivot_type == "low"


class PivotDetector:
    """Detects swing highs and lows at multiple lookback strengths.

    Uses a simple n-bar pivot rule: a bar is a pivot high if its high
    is greater than the highs of the n bars before and after it.
    Similarly for pivot lows with the low price.
    """

    def __init__(self, min_strength: int = 2, max_strength: int = 10):
        """
        Args:
            min_strength: minimum bars on each side to qualify as pivot
            max_strength: maximum lookback to test (higher = stronger pivots)
        """
        self.min_strength = min_strength
        self.max_strength = max_strength

    def detect(self, df: pd.DataFrame) -> list[Pivot]:
        """Detect all pivots in the OHLCV DataFrame.

        Returns pivots sorted by time, each annotated with its maximum
        confirmed strength (the largest n for which it qualifies).
        """
        if df.empty or len(df) < (self.min_strength * 2 + 1):
            return []

        highs = df["high"].values
        lows = df["low"].values
        times = df["open_time"].values
        n = len(df)

        pivot_high_strength = [0] * n
        pivot_low_strength = [0] * n

        # For each strength level, mark qualifying pivots
        for strength in range(self.min_strength, self.max_strength + 1):
            for i in range(strength, n - strength):
                # Check pivot high
                is_ph = True
                for j in range(1, strength + 1):
                    if highs[i] <= highs[i - j] or highs[i] <= highs[i + j]:
                        is_ph = False
                        break
                if is_ph:
                    pivot_high_strength[i] = strength

                # Check pivot low
                is_pl = True
                for j in range(1, strength + 1):
                    if lows[i] >= lows[i - j] or lows[i] >= lows[i + j]:
                        is_pl = False
                        break
                if is_pl:
                    pivot_low_strength[i] = strength

        # Build pivot list
        pivots: list[Pivot] = []
        for i in range(n):
            if pivot_high_strength[i] >= self.min_strength:
                dt = pd.Timestamp(times[i]).to_pydatetime()
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                pivots.append(Pivot(
                    time=dt,
                    price=float(highs[i]),
                    bar_index=i,
                    pivot_type="high",
                    strength=pivot_high_strength[i],
                    prominence=0.0,
                ))
            if pivot_low_strength[i] >= self.min_strength:
                dt = pd.Timestamp(times[i]).to_pydatetime()
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                pivots.append(Pivot(
                    time=dt,
                    price=float(lows[i]),
                    bar_index=i,
                    pivot_type="low",
                    strength=pivot_low_strength[i],
                    prominence=0.0,
                ))

        # Compute prominence (distance from nearest opposing pivot)
        self._compute_prominence(pivots)

        return sorted(pivots, key=lambda p: p.time)

    def detect_in_range(
        self,
        df: pd.DataFrame,
        start_time: datetime,
        end_time: datetime,
    ) -> list[Pivot]:
        """Detect pivots only within a time range."""
        all_pivots = self.detect(df)
        return [
            p for p in all_pivots
            if start_time <= p.time <= end_time
        ]

    def _compute_prominence(self, pivots: list[Pivot]):
        """Annotate each pivot with its prominence.

        Prominence = absolute price distance to the nearest pivot of
        the opposite type (high vs low). Higher prominence means more
        significant swing.
        """
        highs = [p for p in pivots if p.is_high]
        lows = [p for p in pivots if p.is_low]

        for ph in highs:
            nearest_dist = float("inf")
            for pl in lows:
                dist = abs(ph.price - pl.price)
                if dist < nearest_dist:
                    nearest_dist = dist
            ph.prominence = nearest_dist if nearest_dist != float("inf") else 0.0

        for pl in lows:
            nearest_dist = float("inf")
            for ph in highs:
                dist = abs(ph.price - pl.price)
                if dist < nearest_dist:
                    nearest_dist = dist
            pl.prominence = nearest_dist if nearest_dist != float("inf") else 0.0

    def get_significant_pivots(
        self,
        pivots: list[Pivot],
        min_strength: int = 3,
        top_n: Optional[int] = None,
    ) -> list[Pivot]:
        """Filter to the most significant pivots.

        Args:
            min_strength: minimum strength to include
            top_n: if set, return only the top N by prominence
        """
        filtered = [p for p in pivots if p.strength >= min_strength]
        filtered.sort(key=lambda p: p.prominence, reverse=True)
        if top_n:
            filtered = filtered[:top_n]
        return sorted(filtered, key=lambda p: p.time)
