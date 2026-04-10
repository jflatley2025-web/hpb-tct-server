"""Range suggester — proposes range high/low/EQ from pivot data.

A range is a consolidation zone where price oscillates between
swing highs and lows before breaking out. This module finds candidate
ranges that formed BEFORE the trade entry.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

from local_chart_overlay.analysis.pivot_detector import Pivot, PivotDetector


@dataclass
class RangeCandidate:
    """A proposed consolidation range."""
    high_price: float
    high_time: datetime
    low_price: float
    low_time: datetime
    confidence: float           # 0.0 – 1.0
    reason_tags: list[str] = field(default_factory=list)
    num_touches_high: int = 0   # how many pivot highs near this level
    num_touches_low: int = 0    # how many pivot lows near this level
    duration_bars: int = 0      # how many bars the range spans

    @property
    def eq_price(self) -> float:
        return (self.high_price + self.low_price) / 2.0

    @property
    def size(self) -> float:
        return self.high_price - self.low_price

    @property
    def size_pct(self) -> float:
        if self.low_price == 0:
            return 0.0
        return (self.size / self.low_price) * 100.0


class RangeSuggester:
    """Finds consolidation ranges from pivot clusters before entry.

    Strategy:
    1. Find pivot highs and lows in the pre-entry window
    2. Cluster nearby pivot highs → candidate range high
    3. Cluster nearby pivot lows → candidate range low
    4. Score based on:
       - number of touches (more = stronger range)
       - duration (longer = more developed)
       - proximity to entry (closer = more relevant)
       - horizontality (tight cluster = better range)
    """

    def __init__(
        self,
        cluster_tolerance_pct: float = 0.3,
        min_touches: int = 2,
        min_duration_bars: int = 6,
    ):
        """
        Args:
            cluster_tolerance_pct: max % distance to consider pivots as
                                   touching the same level
            min_touches: minimum pivot touches to qualify as range boundary
            min_duration_bars: minimum bars between first and last touch
        """
        self.cluster_tolerance_pct = cluster_tolerance_pct
        self.min_touches = min_touches
        self.min_duration_bars = min_duration_bars

    def suggest(
        self,
        pivots: list[Pivot],
        entry_time: datetime,
        entry_price: float,
        direction: str,
        lookback_bars: int = 150,
    ) -> list[RangeCandidate]:
        """Find candidate ranges from pivots before entry.

        Args:
            pivots: all detected pivots (will be filtered to pre-entry)
            entry_time: trade entry time
            entry_price: trade entry price
            direction: "bullish" or "bearish"
            lookback_bars: how far back to search (in pivot count)

        Returns:
            List of RangeCandidate sorted by confidence (highest first).
        """
        # Filter to pivots before entry
        pre_entry = [p for p in pivots if p.time < entry_time]
        if len(pre_entry) < 4:
            return []

        # Limit lookback
        pre_entry = pre_entry[-lookback_bars:]

        # Separate highs and lows
        pivot_highs = [p for p in pre_entry if p.is_high]
        pivot_lows = [p for p in pre_entry if p.is_low]

        if not pivot_highs or not pivot_lows:
            return []

        # Cluster pivot highs into levels
        high_clusters = self._cluster_pivots(pivot_highs)
        low_clusters = self._cluster_pivots(pivot_lows)

        # Build range candidates from high+low cluster pairs
        candidates = []
        for hc in high_clusters:
            for lc in low_clusters:
                cand = self._evaluate_range(
                    hc, lc, entry_time, entry_price, direction
                )
                if cand:
                    candidates.append(cand)

        candidates.sort(key=lambda c: c.confidence, reverse=True)
        return candidates[:5]  # top 5

    def _cluster_pivots(self, pivots: list[Pivot]) -> list[dict]:
        """Cluster pivots at similar price levels.

        Returns list of clusters, each containing:
          - level: average price
          - pivots: list of Pivot objects in this cluster
          - first_time / last_time
        """
        if not pivots:
            return []

        # Sort by price
        sorted_p = sorted(pivots, key=lambda p: p.price)
        clusters = []
        current = [sorted_p[0]]

        for p in sorted_p[1:]:
            ref_price = sum(pp.price for pp in current) / len(current)
            pct_dist = abs(p.price - ref_price) / ref_price * 100
            if pct_dist <= self.cluster_tolerance_pct:
                current.append(p)
            else:
                clusters.append(self._finalize_cluster(current))
                current = [p]

        clusters.append(self._finalize_cluster(current))

        # Filter by minimum touches
        return [c for c in clusters if len(c["pivots"]) >= self.min_touches]

    def _finalize_cluster(self, pivots: list[Pivot]) -> dict:
        avg_price = sum(p.price for p in pivots) / len(pivots)
        times = [p.time for p in pivots]
        # Use the pivot with highest strength as the representative
        best = max(pivots, key=lambda p: p.strength)
        return {
            "level": avg_price,
            "representative_time": best.time,
            "representative_price": best.price,
            "pivots": pivots,
            "first_time": min(times),
            "last_time": max(times),
            "spread_pct": (
                (max(p.price for p in pivots) - min(p.price for p in pivots))
                / avg_price * 100
            ) if len(pivots) > 1 else 0.0,
        }

    def _evaluate_range(
        self,
        high_cluster: dict,
        low_cluster: dict,
        entry_time: datetime,
        entry_price: float,
        direction: str,
    ) -> Optional[RangeCandidate]:
        """Score a high+low cluster pair as a range candidate."""
        high_level = high_cluster["level"]
        low_level = low_cluster["level"]

        # Range must have high > low
        if high_level <= low_level:
            return None

        range_size_pct = (high_level - low_level) / low_level * 100

        # Skip ranges that are too tight (<0.1%) or too wide (>15%)
        if range_size_pct < 0.1 or range_size_pct > 15.0:
            return None

        # Duration: from earliest pivot to latest pivot across both clusters
        all_times = (
            [p.time for p in high_cluster["pivots"]] +
            [p.time for p in low_cluster["pivots"]]
        )
        first_time = min(all_times)
        last_time = max(all_times)
        duration_seconds = (last_time - first_time).total_seconds()

        # Entry must be after the range formed
        if last_time >= entry_time:
            return None

        # Score components
        confidence = 0.0
        tags = []

        # 1. Touch count (more = stronger range)
        total_touches = len(high_cluster["pivots"]) + len(low_cluster["pivots"])
        touch_score = min(total_touches / 8.0, 1.0)  # max out at 8 touches
        confidence += touch_score * 0.30
        if total_touches >= 4:
            tags.append("multi_touch")

        # 2. Horizontality (tight clusters = better range)
        avg_spread = (high_cluster["spread_pct"] + low_cluster["spread_pct"]) / 2
        horiz_score = max(0, 1.0 - avg_spread / self.cluster_tolerance_pct)
        confidence += horiz_score * 0.20
        if horiz_score > 0.7:
            tags.append("horizontal")

        # 3. Duration
        if duration_seconds > 3600 * 6:  # >6h
            confidence += 0.15
            tags.append("developed")
        elif duration_seconds > 3600 * 2:
            confidence += 0.10

        # 4. Proximity to entry (closer = more relevant)
        gap = (entry_time - last_time).total_seconds()
        if gap < 3600 * 4:  # within 4h
            confidence += 0.15
            tags.append("near_entry")
        elif gap < 3600 * 24:
            confidence += 0.10

        # 5. Entry price relative to range
        eq = (high_level + low_level) / 2
        if direction == "bearish" and entry_price > eq:
            confidence += 0.10
            tags.append("entry_above_eq")
        elif direction == "bullish" and entry_price < eq:
            confidence += 0.10
            tags.append("entry_below_eq")

        # 6. Pivot strength bonus
        avg_strength = sum(
            p.strength for p in high_cluster["pivots"] + low_cluster["pivots"]
        ) / total_touches
        if avg_strength >= 4:
            confidence += 0.10
            tags.append("strong_pivots")

        return RangeCandidate(
            high_price=high_cluster["representative_price"],
            high_time=high_cluster["representative_time"],
            low_price=low_cluster["representative_price"],
            low_time=low_cluster["representative_time"],
            confidence=round(min(confidence, 1.0), 3),
            reason_tags=tags,
            num_touches_high=len(high_cluster["pivots"]),
            num_touches_low=len(low_cluster["pivots"]),
            duration_bars=int(duration_seconds),  # stored as seconds for generality
        )
