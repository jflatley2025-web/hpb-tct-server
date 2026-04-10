"""Accuracy scorer — compares derived suggestions against confirmed schematics.

Measures how well the suggestion engine performs by comparing its output
against manually confirmed (frozen) schematics. Tracks:
  - timestamp error per anchor (seconds)
  - price error per anchor (absolute + %)
  - per-tap hit rates
  - range match quality
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from local_chart_overlay.models.schematic import FrozenSchematic
from local_chart_overlay.analysis.tap_suggester import SchematicSuggestion


@dataclass
class AnchorError:
    """Error between a suggested anchor and the confirmed truth."""
    anchor_name: str        # "tap1", "tap2", "tap3", "range_high", etc.
    price_error: Optional[float] = None          # absolute price difference
    price_error_pct: Optional[float] = None      # percentage price difference
    time_error_seconds: Optional[float] = None   # seconds between timestamps
    hit: bool = False       # True if within acceptable tolerance
    note: str = ""


@dataclass
class AccuracyReport:
    """Full accuracy comparison for one trade."""
    trade_id: int
    anchor_errors: list[AnchorError] = field(default_factory=list)

    @property
    def hit_rate(self) -> float:
        """Fraction of anchors that matched within tolerance."""
        evaluated = [e for e in self.anchor_errors if e.price_error is not None]
        if not evaluated:
            return 0.0
        return sum(1 for e in evaluated if e.hit) / len(evaluated)

    @property
    def avg_price_error_pct(self) -> Optional[float]:
        """Average absolute price error as percentage."""
        vals = [e.price_error_pct for e in self.anchor_errors if e.price_error_pct is not None]
        return sum(vals) / len(vals) if vals else None

    @property
    def avg_time_error_seconds(self) -> Optional[float]:
        """Average absolute time error in seconds."""
        vals = [abs(e.time_error_seconds) for e in self.anchor_errors if e.time_error_seconds is not None]
        return sum(vals) / len(vals) if vals else None

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [f"Trade #{self.trade_id} — accuracy report"]
        lines.append(f"  Hit rate: {self.hit_rate:.0%} "
                      f"({sum(1 for e in self.anchor_errors if e.hit)}/"
                      f"{len([e for e in self.anchor_errors if e.price_error is not None])})")
        if self.avg_price_error_pct is not None:
            lines.append(f"  Avg price error: {self.avg_price_error_pct:.3f}%")
        if self.avg_time_error_seconds is not None:
            hrs = self.avg_time_error_seconds / 3600
            lines.append(f"  Avg time error: {hrs:.1f}h ({self.avg_time_error_seconds:.0f}s)")
        for e in self.anchor_errors:
            status = "HIT" if e.hit else "MISS"
            price_str = f"price_err={e.price_error_pct:.3f}%" if e.price_error_pct is not None else "n/a"
            time_str = f"time_err={e.time_error_seconds:.0f}s" if e.time_error_seconds is not None else "n/a"
            lines.append(f"  {e.anchor_name:>12}: [{status}] {price_str}  {time_str}  {e.note}")
        return "\n".join(lines)


class AccuracyScorer:
    """Compares suggestions against confirmed schematics."""

    def __init__(
        self,
        price_tolerance_pct: float = 0.5,
        time_tolerance_seconds: float = 14400,  # 4 hours
    ):
        """
        Args:
            price_tolerance_pct: max % price difference to count as hit
            time_tolerance_seconds: max seconds time difference to count as hit
        """
        self.price_tolerance_pct = price_tolerance_pct
        self.time_tolerance_seconds = time_tolerance_seconds

    def score(
        self,
        suggestion: SchematicSuggestion,
        confirmed: FrozenSchematic,
    ) -> AccuracyReport:
        """Compare a suggestion against a confirmed schematic."""
        report = AccuracyReport(trade_id=suggestion.trade_id)

        # Compare taps
        tap_pairs = [
            ("tap1", suggestion.best_tap1, confirmed.tap1_price, confirmed.tap1_time),
            ("tap2", suggestion.best_tap2, confirmed.tap2_price, confirmed.tap2_time),
            ("tap3", suggestion.best_tap3, confirmed.tap3_price, confirmed.tap3_time),
        ]
        for name, candidate, true_price, true_time in tap_pairs:
            if true_price is None:
                report.anchor_errors.append(AnchorError(
                    anchor_name=name, note="no confirmed anchor"
                ))
                continue
            if candidate is None:
                report.anchor_errors.append(AnchorError(
                    anchor_name=name, note="no suggestion generated"
                ))
                continue

            err = self._compare_anchor(
                name, candidate.price, candidate.time, true_price, true_time
            )
            report.anchor_errors.append(err)

        # Compare range
        if confirmed.range_high_price is not None and suggestion.range_candidate:
            rng = suggestion.range_candidate
            report.anchor_errors.append(
                self._compare_anchor(
                    "range_high", rng.high_price, rng.high_time,
                    confirmed.range_high_price, confirmed.range_high_time,
                )
            )
            report.anchor_errors.append(
                self._compare_anchor(
                    "range_low", rng.low_price, rng.low_time,
                    confirmed.range_low_price, confirmed.range_low_time,
                )
            )
        elif confirmed.range_high_price is not None:
            report.anchor_errors.append(AnchorError(
                anchor_name="range_high", note="no range suggestion"
            ))
            report.anchor_errors.append(AnchorError(
                anchor_name="range_low", note="no range suggestion"
            ))

        # Compare BOS
        if confirmed.bos_price is not None:
            if suggestion.bos_price is not None:
                report.anchor_errors.append(
                    self._compare_anchor(
                        "bos", suggestion.bos_price, suggestion.bos_time,
                        confirmed.bos_price, confirmed.bos_time,
                    )
                )
            else:
                report.anchor_errors.append(AnchorError(
                    anchor_name="bos", note="no BOS suggestion"
                ))

        return report

    def _compare_anchor(
        self,
        name: str,
        suggested_price: float,
        suggested_time: Optional[datetime],
        true_price: float,
        true_time: Optional[datetime],
    ) -> AnchorError:
        """Compare a single anchor point."""
        price_err = abs(suggested_price - true_price)
        price_err_pct = (price_err / true_price * 100) if true_price != 0 else 0

        time_err = None
        if suggested_time and true_time:
            time_err = abs((suggested_time - true_time).total_seconds())

        price_hit = price_err_pct <= self.price_tolerance_pct
        time_hit = time_err is None or time_err <= self.time_tolerance_seconds
        hit = price_hit and time_hit

        note_parts = []
        if not price_hit:
            note_parts.append(f"price off by {price_err_pct:.2f}%")
        if time_err is not None and not time_hit:
            note_parts.append(f"time off by {time_err/3600:.1f}h")

        return AnchorError(
            anchor_name=name,
            price_error=price_err,
            price_error_pct=price_err_pct,
            time_error_seconds=time_err,
            hit=hit,
            note="; ".join(note_parts) if note_parts else "within tolerance",
        )

    def score_batch(
        self,
        pairs: list[tuple[SchematicSuggestion, FrozenSchematic]],
    ) -> list[AccuracyReport]:
        """Score multiple suggestion-vs-confirmed pairs."""
        return [self.score(s, c) for s, c in pairs]

    @staticmethod
    def aggregate_reports(reports: list[AccuracyReport]) -> str:
        """Aggregate accuracy across multiple trades."""
        if not reports:
            return "No reports to aggregate."

        all_errors = []
        for r in reports:
            all_errors.extend(r.anchor_errors)

        evaluated = [e for e in all_errors if e.price_error is not None]
        hits = sum(1 for e in evaluated if e.hit)

        # Per-anchor breakdown
        anchor_names = ["tap1", "tap2", "tap3", "range_high", "range_low", "bos"]
        lines = [f"=== Aggregate accuracy ({len(reports)} trades) ==="]
        lines.append(f"Overall hit rate: {hits}/{len(evaluated)} = "
                      f"{hits/len(evaluated):.0%}" if evaluated else "n/a")

        for name in anchor_names:
            errs = [e for e in all_errors if e.anchor_name == name and e.price_error is not None]
            if not errs:
                continue
            hit_count = sum(1 for e in errs if e.hit)
            avg_pct = sum(e.price_error_pct for e in errs) / len(errs)
            lines.append(f"  {name:>12}: {hit_count}/{len(errs)} hit  "
                          f"avg_err={avg_pct:.3f}%")

        return "\n".join(lines)
