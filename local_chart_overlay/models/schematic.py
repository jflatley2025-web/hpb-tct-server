"""Frozen schematic — immutable time + price anchors for chart overlay."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass(frozen=True)
class FrozenSchematic:
    """Immutable schematic anchors.

    Once saved, these values NEVER recalculate.
    This is the single source of truth for rendering.
    All time fields are UTC datetime objects.
    """

    # ── Tap points ────────────────────────────────────────────────────
    tap1_price: Optional[float] = None
    tap1_time: Optional[datetime] = None
    tap2_price: Optional[float] = None
    tap2_time: Optional[datetime] = None
    tap3_price: Optional[float] = None
    tap3_time: Optional[datetime] = None

    # ── Range levels ──────────────────────────────────────────────────
    range_high_price: Optional[float] = None
    range_high_time: Optional[datetime] = None
    range_low_price: Optional[float] = None
    range_low_time: Optional[datetime] = None

    # ── BOS (break of structure) ──────────────────────────────────────
    bos_price: Optional[float] = None
    bos_time: Optional[datetime] = None

    # ── Sweep ─────────────────────────────────────────────────────────
    sweep_type: Optional[str] = None

    # ── Classification ────────────────────────────────────────────────
    model_label: Optional[str] = None
    timeframe: Optional[str] = None

    # ── Multi-timeframe context ───────────────────────────────────────
    context_timeframe: Optional[str] = None       # HTF that provided bias (e.g. "4h")
    execution_timeframe: Optional[str] = None     # LTF used for entry (e.g. "15m")
    parent_structure_id: Optional[int] = None     # FK to parent schematic (future use)

    # ── Versioning ────────────────────────────────────────────────────
    version: int = 1                              # incremented on each update
    source: str = "manual"                        # "manual" | "imported" | "derived"
    created_at: Optional[datetime] = None         # first creation time
    updated_at: Optional[datetime] = None         # last modification time

    # ── Edit tracking ─────────────────────────────────────────────────
    manually_edited: bool = False
    last_edited_at: Optional[datetime] = None     # deprecated: use updated_at

    # ── Metadata ──────────────────────────────────────────────────────
    notes: Optional[str] = None
    confidence: Optional[float] = None
    tags: Optional[str] = None  # comma-separated
    data_source: Optional[str] = None  # OHLCV provenance: "mexc_api", "cached_parquet", "local_csv"

    @property
    def range_eq_price(self) -> Optional[float]:
        """Equilibrium = midpoint of range."""
        if self.range_high_price is not None and self.range_low_price is not None:
            return (self.range_high_price + self.range_low_price) / 2.0
        return None

    @property
    def has_taps(self) -> bool:
        return any([self.tap1_price, self.tap2_price, self.tap3_price])

    @property
    def has_range(self) -> bool:
        return self.range_high_price is not None and self.range_low_price is not None

    @property
    def completeness(self) -> float:
        """0.0–1.0 score of how many anchor fields are populated."""
        fields = [
            self.tap1_price, self.tap1_time,
            self.tap2_price, self.tap2_time,
            self.tap3_price, self.tap3_time,
            self.range_high_price, self.range_high_time,
            self.range_low_price, self.range_low_time,
            self.bos_price, self.bos_time,
        ]
        filled = sum(1 for f in fields if f is not None)
        return filled / len(fields)
