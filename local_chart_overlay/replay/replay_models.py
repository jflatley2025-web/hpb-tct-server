"""Replay data models — structured types for the replay inspector."""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import IntEnum
from typing import Optional


class Stage(IntEnum):
    """Progressive reveal stages for the replay."""
    PRE_CONTEXT = 0
    TAP_1 = 1
    TAP_2 = 2
    TAP_3 = 3
    BOS = 4
    ENTRY = 5
    OUTCOME = 6


STAGE_LABELS = {
    Stage.PRE_CONTEXT: "Pre-Context",
    Stage.TAP_1: "Tap 1",
    Stage.TAP_2: "Tap 2",
    Stage.TAP_3: "Tap 3",
    Stage.BOS: "Break of Structure",
    Stage.ENTRY: "Entry / SL / TP",
    Stage.OUTCOME: "Outcome",
}


@dataclass
class ReplayChartPoint:
    """A single OHLCV candle for chart rendering."""
    time_ms: int    # Unix ms
    time_iso: str
    open: float
    high: float
    low: float
    close: float
    volume: float

    @staticmethod
    def from_row(row) -> ReplayChartPoint:
        dt = row["open_time"]
        if hasattr(dt, "timestamp"):
            time_ms = int(dt.timestamp() * 1000)
            time_iso = dt.isoformat()
        else:
            time_ms = 0
            time_iso = str(dt)
        return ReplayChartPoint(
            time_ms=time_ms,
            time_iso=time_iso,
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=float(row.get("volume", 0)),
        )


@dataclass
class AnchorPoint:
    """A single price+time anchor for rendering."""
    label: str
    price: Optional[float] = None
    time_ms: Optional[int] = None
    time_iso: Optional[str] = None
    visible_from_stage: int = 0  # Stage int at which this becomes visible

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "price": self.price,
            "time_ms": self.time_ms,
            "time_iso": self.time_iso,
            "visible_from_stage": self.visible_from_stage,
        }


@dataclass
class AnchorComparison:
    """Side-by-side comparison of confirmed vs suggested anchor."""
    anchor_name: str
    confirmed_price: Optional[float] = None
    confirmed_time_iso: Optional[str] = None
    suggested_price: Optional[float] = None
    suggested_time_iso: Optional[str] = None
    price_delta: Optional[float] = None
    price_delta_pct: Optional[float] = None
    time_delta_seconds: Optional[float] = None
    hit: Optional[bool] = None
    score_breakdown: Optional[dict] = None


@dataclass
class ReplaySummary:
    """Trade summary for the side panel."""
    trade_id: int
    symbol: str
    direction: str
    side: str
    timeframe: str
    model: str
    entry_price: float
    stop_price: float
    target_price: float
    tp1_price: Optional[float]
    rr: Optional[float]
    entry_score: Optional[int]
    opened_at: str
    closed_at: Optional[str]
    exit_price: Optional[float]
    exit_reason: Optional[str]
    pnl_pct: Optional[float]
    pnl_dollars: Optional[float]
    is_win: Optional[bool]
    htf_bias: Optional[str]
    schematic_completeness: Optional[float]
    schematic_source: Optional[str]
    schematic_version: Optional[int]


@dataclass
class AccuracySummary:
    """Condensed accuracy report for display."""
    hit_rate: float
    avg_price_error_pct: Optional[float]
    avg_time_error_seconds: Optional[float]
    anchor_results: list[dict] = field(default_factory=list)
    # Each entry: {name, hit, price_err_pct, time_err_sec, note}


@dataclass
class ReplayPayload:
    """Complete payload for rendering one trade replay.

    This is serialized to JSON and embedded in the HTML file.
    """
    # ── Identity ──────────────────────────────────────────────────────
    trade_id: int
    generated_at: str

    # ── Summary ───────────────────────────────────────────────────────
    summary: ReplaySummary

    # ── Chart data ────────────────────────────────────────────────────
    candles: list[ReplayChartPoint] = field(default_factory=list)

    # ── Confirmed anchors ─────────────────────────────────────────────
    confirmed_anchors: list[AnchorPoint] = field(default_factory=list)

    # ── Suggested anchors ─────────────────────────────────────────────
    suggested_anchors: list[AnchorPoint] = field(default_factory=list)
    has_suggestion: bool = False

    # ── Comparison ────────────────────────────────────────────────────
    comparisons: list[AnchorComparison] = field(default_factory=list)
    has_accuracy: bool = False
    accuracy: Optional[AccuracySummary] = None

    # ── Stage info ────────────────────────────────────────────────────
    stages: list[dict] = field(default_factory=list)  # [{id, label}]
    max_stage: int = 6

    # ── Annotations ────────────────────────────────────────────────────
    tags: list[str] = field(default_factory=list)
    notes: Optional[str] = None

    # ── Pine Script (excluded from JSON, injected separately) ─────────
    pine_script: str = ""

    def to_json(self) -> str:
        """Serialize to JSON for embedding in HTML.

        pine_script is excluded — it's injected separately into the
        template as a JS variable for safe escaping.
        """
        d = asdict(self)
        d.pop("pine_script", None)
        return json.dumps(d, default=str)
