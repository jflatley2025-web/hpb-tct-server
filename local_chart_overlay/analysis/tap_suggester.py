"""Tap suggester — proposes tap 1/2/3 anchors from pivots + range context.

TCT schematic taps follow a specific pattern:
  - Tap 1: First touch of range boundary (high for bearish, low for bullish)
  - Tap 2: Deviation beyond range boundary (liquidity sweep)
  - Tap 3: Second deviation or higher-low/lower-high confirming direction

For bearish (distribution):
  Tap 1 = pivot high at/near range high
  Tap 2 = higher pivot high (deviation above range)
  Tap 3 = lower high than tap 2 (confirmation of bearish intent)

For bullish (accumulation):
  Tap 1 = pivot low at/near range low
  Tap 2 = lower pivot low (deviation below range)
  Tap 3 = higher low than tap 2 (confirmation of bullish intent)

This module does NOT enforce strict TCT rules. It produces ranked
suggestions that a human confirms. The intelligence stays outside Pine.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from local_chart_overlay.analysis.pivot_detector import Pivot
from local_chart_overlay.analysis.range_suggester import RangeCandidate


@dataclass
class TapCandidate:
    """A proposed tap point."""
    tap_number: int         # 1, 2, or 3
    price: float
    time: datetime
    confidence: float       # 0.0 – 1.0
    reason_tags: list[str] = field(default_factory=list)
    score_breakdown: dict = field(default_factory=dict)  # component -> weight
    pivot_strength: int = 0
    source_pivot_index: int = 0  # bar index of the underlying pivot

    def explain(self) -> str:
        """Human-readable explanation of why this confidence score."""
        if not self.score_breakdown:
            return f"confidence={self.confidence:.0%} (no breakdown)"
        parts = []
        for component, weight in sorted(
            self.score_breakdown.items(), key=lambda x: x[1], reverse=True
        ):
            sign = "+" if weight > 0 else ""
            parts.append(f"  {sign}{weight:.0%} {component}")
        return f"confidence={self.confidence:.0%}\n" + "\n".join(parts)


@dataclass
class SchematicSuggestion:
    """Complete suggested schematic with ranked candidates for each component."""
    trade_id: int
    direction: str
    entry_price: float
    entry_time: datetime

    # Best range candidate
    range_candidate: Optional[RangeCandidate] = None

    # Top candidates for each tap (ranked by confidence)
    tap1_candidates: list[TapCandidate] = field(default_factory=list)
    tap2_candidates: list[TapCandidate] = field(default_factory=list)
    tap3_candidates: list[TapCandidate] = field(default_factory=list)

    # BOS candidate
    bos_price: Optional[float] = None
    bos_time: Optional[datetime] = None
    bos_confidence: float = 0.0

    @property
    def overall_confidence(self) -> float:
        """Aggregate confidence across all components."""
        scores = []
        if self.range_candidate:
            scores.append(self.range_candidate.confidence)
        for taps in (self.tap1_candidates, self.tap2_candidates, self.tap3_candidates):
            if taps:
                scores.append(taps[0].confidence)
        if self.bos_confidence > 0:
            scores.append(self.bos_confidence)
        return sum(scores) / len(scores) if scores else 0.0

    @property
    def best_tap1(self) -> Optional[TapCandidate]:
        return self.tap1_candidates[0] if self.tap1_candidates else None

    @property
    def best_tap2(self) -> Optional[TapCandidate]:
        return self.tap2_candidates[0] if self.tap2_candidates else None

    @property
    def best_tap3(self) -> Optional[TapCandidate]:
        return self.tap3_candidates[0] if self.tap3_candidates else None


class _Score:
    """Accumulates confidence score with named components for explanation."""

    def __init__(self):
        self.total = 0.0
        self.breakdown: dict[str, float] = {}
        self.tags: list[str] = []

    def add(self, component: str, value: float, tag: str | None = None):
        self.total += value
        self.breakdown[component] = value
        if tag:
            self.tags.append(tag)

    @property
    def confidence(self) -> float:
        return round(min(self.total, 1.0), 3)


class TapSuggester:
    """Proposes tap candidates given pivots and a range.

    This is a suggestion engine, not an auto-labeler.
    All outputs require human confirmation.
    """

    def __init__(
        self,
        range_tolerance_pct: float = 0.5,
        deviation_min_pct: float = 0.05,
    ):
        """
        Args:
            range_tolerance_pct: how close a pivot must be to range
                                  boundary to count as "at range"
            deviation_min_pct: minimum % beyond range to count as deviation
        """
        self.range_tolerance_pct = range_tolerance_pct
        self.deviation_min_pct = deviation_min_pct

    def suggest(
        self,
        pivots: list[Pivot],
        range_cand: Optional[RangeCandidate],
        entry_time: datetime,
        entry_price: float,
        stop_price: float,
        direction: str,
    ) -> SchematicSuggestion:
        """Generate a complete schematic suggestion.

        Args:
            pivots: all detected pivots in the OHLCV window
            range_cand: the best range candidate (or None)
            entry_time: trade entry time
            entry_price: trade entry price
            stop_price: trade stop loss price
            direction: "bullish" or "bearish"
        """
        suggestion = SchematicSuggestion(
            trade_id=0,  # set by caller
            direction=direction,
            entry_price=entry_price,
            entry_time=entry_time,
            range_candidate=range_cand,
        )

        # Filter pivots to pre-entry window
        pre_entry = [p for p in pivots if p.time < entry_time]
        if not pre_entry:
            return suggestion

        if direction == "bearish":
            self._suggest_bearish_taps(
                pre_entry, range_cand, entry_time, entry_price, stop_price, suggestion
            )
        else:
            self._suggest_bullish_taps(
                pre_entry, range_cand, entry_time, entry_price, stop_price, suggestion
            )

        # BOS detection
        self._suggest_bos(pre_entry, range_cand, entry_time, direction, suggestion)

        return suggestion

    def _suggest_bearish_taps(
        self,
        pivots: list[Pivot],
        rng: Optional[RangeCandidate],
        entry_time: datetime,
        entry_price: float,
        stop_price: float,
        suggestion: SchematicSuggestion,
    ):
        """For bearish/distribution: taps are at/above range high.

        Tap 1: pivot high at range high
        Tap 2: higher high (deviation above range)
        Tap 3: lower high (failing to make new high = bearish confirmation)
        """
        pivot_highs = [p for p in pivots if p.is_high]
        if not pivot_highs:
            return

        range_high = rng.high_price if rng else stop_price

        # ── Tap 1: pivot highs near range high ──
        for ph in pivot_highs:
            dist_pct = abs(ph.price - range_high) / range_high * 100
            s = _Score()

            if dist_pct <= self.range_tolerance_pct:
                s.add("range_proximity", 0.40, "at_range_high")
            elif dist_pct <= self.range_tolerance_pct * 2:
                s.add("range_proximity", 0.20, "near_range_high")
            else:
                continue

            strength_val = min(ph.strength / 10.0, 0.2)
            s.add("pivot_strength", strength_val, "strong_pivot" if ph.strength >= 3 else None)

            if ph.prominence > 0:
                s.add("prominence", min(ph.prominence / range_high * 50, 0.15))

            if rng and abs((ph.time - rng.high_time).total_seconds()) < 3600 * 12:
                s.add("time_alignment", 0.10, "time_aligned")

            suggestion.tap1_candidates.append(TapCandidate(
                tap_number=1, price=ph.price, time=ph.time,
                confidence=s.confidence,
                reason_tags=s.tags, score_breakdown=s.breakdown,
                pivot_strength=ph.strength,
                source_pivot_index=ph.bar_index,
            ))

        # ── Tap 2: higher highs (deviation above range) ──
        for ph in pivot_highs:
            if ph.price <= range_high:
                continue
            deviation_pct = (ph.price - range_high) / range_high * 100
            if deviation_pct < self.deviation_min_pct:
                continue

            s = _Score()
            s.tags.append("deviation_high")

            if 0.1 <= deviation_pct <= 2.0:
                s.add("deviation_magnitude", 0.40, "clean_deviation")
            elif deviation_pct < 0.1:
                s.add("deviation_magnitude", 0.15)
            else:
                s.add("deviation_magnitude", 0.25, "extended_deviation")

            s.add("pivot_strength", min(ph.strength / 10.0, 0.2),
                  "strong_pivot" if ph.strength >= 3 else None)

            if suggestion.tap1_candidates:
                best_t1 = min(suggestion.tap1_candidates, key=lambda t: t.time)
                if ph.time > best_t1.time:
                    s.add("sequence_after_tap1", 0.15, "after_tap1")

            suggestion.tap2_candidates.append(TapCandidate(
                tap_number=2, price=ph.price, time=ph.time,
                confidence=s.confidence,
                reason_tags=s.tags, score_breakdown=s.breakdown,
                pivot_strength=ph.strength,
                source_pivot_index=ph.bar_index,
            ))

        # ── Tap 3: lower highs (bearish confirmation) ──
        if suggestion.tap2_candidates:
            best_t2 = max(suggestion.tap2_candidates, key=lambda t: t.confidence)
            for ph in pivot_highs:
                if ph.time <= best_t2.time:
                    continue
                if ph.price >= best_t2.price:
                    continue  # not a lower high

                s = _Score()
                s.tags.append("lower_high")

                drop_pct = (best_t2.price - ph.price) / best_t2.price * 100
                if 0.1 <= drop_pct <= 3.0:
                    s.add("lower_high_structure", 0.40, "clean_lower_high")
                elif drop_pct > 3.0:
                    s.add("lower_high_structure", 0.25)
                else:
                    s.add("lower_high_structure", 0.15)

                if ph.price >= range_high * 0.995:
                    s.add("range_position", 0.15, "above_range")

                s.add("pivot_strength", min(ph.strength / 10.0, 0.15))

                gap_sec = (entry_time - ph.time).total_seconds()
                if 0 < gap_sec < 3600 * 6:
                    s.add("entry_proximity", 0.15, "near_entry")

                suggestion.tap3_candidates.append(TapCandidate(
                    tap_number=3, price=ph.price, time=ph.time,
                    confidence=s.confidence,
                    reason_tags=s.tags, score_breakdown=s.breakdown,
                    pivot_strength=ph.strength,
                    source_pivot_index=ph.bar_index,
                ))

        # Sort all by confidence
        suggestion.tap1_candidates.sort(key=lambda t: t.confidence, reverse=True)
        suggestion.tap2_candidates.sort(key=lambda t: t.confidence, reverse=True)
        suggestion.tap3_candidates.sort(key=lambda t: t.confidence, reverse=True)

    def _suggest_bullish_taps(
        self,
        pivots: list[Pivot],
        rng: Optional[RangeCandidate],
        entry_time: datetime,
        entry_price: float,
        stop_price: float,
        suggestion: SchematicSuggestion,
    ):
        """For bullish/accumulation: taps are at/below range low.

        Tap 1: pivot low at range low
        Tap 2: lower low (deviation below range)
        Tap 3: higher low (bullish confirmation)
        """
        pivot_lows = [p for p in pivots if p.is_low]
        if not pivot_lows:
            return

        range_low = rng.low_price if rng else stop_price

        # ── Tap 1: pivot lows near range low ──
        for pl in pivot_lows:
            dist_pct = abs(pl.price - range_low) / range_low * 100
            s = _Score()

            if dist_pct <= self.range_tolerance_pct:
                s.add("range_proximity", 0.40, "at_range_low")
            elif dist_pct <= self.range_tolerance_pct * 2:
                s.add("range_proximity", 0.20, "near_range_low")
            else:
                continue

            s.add("pivot_strength", min(pl.strength / 10.0, 0.2),
                  "strong_pivot" if pl.strength >= 3 else None)

            if pl.prominence > 0:
                s.add("prominence", min(pl.prominence / range_low * 50, 0.15))

            if rng and abs((pl.time - rng.low_time).total_seconds()) < 3600 * 12:
                s.add("time_alignment", 0.10, "time_aligned")

            suggestion.tap1_candidates.append(TapCandidate(
                tap_number=1, price=pl.price, time=pl.time,
                confidence=s.confidence,
                reason_tags=s.tags, score_breakdown=s.breakdown,
                pivot_strength=pl.strength,
                source_pivot_index=pl.bar_index,
            ))

        # ── Tap 2: lower lows (deviation below range) ──
        for pl in pivot_lows:
            if pl.price >= range_low:
                continue
            deviation_pct = (range_low - pl.price) / range_low * 100
            if deviation_pct < self.deviation_min_pct:
                continue

            s = _Score()
            s.tags.append("deviation_low")

            if 0.1 <= deviation_pct <= 2.0:
                s.add("deviation_magnitude", 0.40, "clean_deviation")
            elif deviation_pct < 0.1:
                s.add("deviation_magnitude", 0.15)
            else:
                s.add("deviation_magnitude", 0.25, "extended_deviation")

            s.add("pivot_strength", min(pl.strength / 10.0, 0.2),
                  "strong_pivot" if pl.strength >= 3 else None)

            if suggestion.tap1_candidates:
                best_t1 = min(suggestion.tap1_candidates, key=lambda t: t.time)
                if pl.time > best_t1.time:
                    s.add("sequence_after_tap1", 0.15, "after_tap1")

            suggestion.tap2_candidates.append(TapCandidate(
                tap_number=2, price=pl.price, time=pl.time,
                confidence=s.confidence,
                reason_tags=s.tags, score_breakdown=s.breakdown,
                pivot_strength=pl.strength,
                source_pivot_index=pl.bar_index,
            ))

        # ── Tap 3: higher lows (bullish confirmation) ──
        if suggestion.tap2_candidates:
            best_t2 = max(suggestion.tap2_candidates, key=lambda t: t.confidence)
            for pl in pivot_lows:
                if pl.time <= best_t2.time:
                    continue
                if pl.price <= best_t2.price:
                    continue

                s = _Score()
                s.tags.append("higher_low")

                rise_pct = (pl.price - best_t2.price) / best_t2.price * 100
                if 0.1 <= rise_pct <= 3.0:
                    s.add("higher_low_structure", 0.40, "clean_higher_low")
                elif rise_pct > 3.0:
                    s.add("higher_low_structure", 0.25)
                else:
                    s.add("higher_low_structure", 0.15)

                if pl.price <= range_low * 1.005:
                    s.add("range_position", 0.15, "below_range")

                s.add("pivot_strength", min(pl.strength / 10.0, 0.15))

                gap_sec = (entry_time - pl.time).total_seconds()
                if 0 < gap_sec < 3600 * 6:
                    s.add("entry_proximity", 0.15, "near_entry")

                suggestion.tap3_candidates.append(TapCandidate(
                    tap_number=3, price=pl.price, time=pl.time,
                    confidence=s.confidence,
                    reason_tags=s.tags, score_breakdown=s.breakdown,
                    pivot_strength=pl.strength,
                    source_pivot_index=pl.bar_index,
                ))

        suggestion.tap1_candidates.sort(key=lambda t: t.confidence, reverse=True)
        suggestion.tap2_candidates.sort(key=lambda t: t.confidence, reverse=True)
        suggestion.tap3_candidates.sort(key=lambda t: t.confidence, reverse=True)

    def _suggest_bos(
        self,
        pivots: list[Pivot],
        rng: Optional[RangeCandidate],
        entry_time: datetime,
        direction: str,
        suggestion: SchematicSuggestion,
    ):
        """Find the break of structure (BOS) — price breaking back into range.

        For bearish: BOS is a pivot low that breaks below a prior swing low
                     (ideally below range EQ or range low)
        For bullish: BOS is a pivot high that breaks above a prior swing high
                     (ideally above range EQ or range high)
        """
        if not rng:
            return

        eq = rng.eq_price

        if direction == "bearish":
            # Look for swing lows breaking below EQ or range low after tap3
            pivot_lows = [
                p for p in pivots
                if p.is_low and p.price < eq and p.time < entry_time
            ]
            # Best BOS: closest to entry, below EQ
            if pivot_lows:
                # Sort by time (latest first)
                pivot_lows.sort(key=lambda p: p.time, reverse=True)
                bos = pivot_lows[0]
                conf = 0.3
                if bos.price <= rng.low_price:
                    conf += 0.3
                gap = (entry_time - bos.time).total_seconds()
                if 0 < gap < 3600 * 8:
                    conf += 0.2
                if bos.strength >= 3:
                    conf += 0.1
                suggestion.bos_price = bos.price
                suggestion.bos_time = bos.time
                suggestion.bos_confidence = round(min(conf, 1.0), 3)
        else:
            pivot_highs = [
                p for p in pivots
                if p.is_high and p.price > eq and p.time < entry_time
            ]
            if pivot_highs:
                pivot_highs.sort(key=lambda p: p.time, reverse=True)
                bos = pivot_highs[0]
                conf = 0.3
                if bos.price >= rng.high_price:
                    conf += 0.3
                gap = (entry_time - bos.time).total_seconds()
                if 0 < gap < 3600 * 8:
                    conf += 0.2
                if bos.strength >= 3:
                    conf += 0.1
                suggestion.bos_price = bos.price
                suggestion.bos_time = bos.time
                suggestion.bos_confidence = round(min(conf, 1.0), 3)
