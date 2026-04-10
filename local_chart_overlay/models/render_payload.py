"""Render payload — intermediate format consumed by Pine template."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from local_chart_overlay.models.trade import TradeRecord
from local_chart_overlay.models.schematic import FrozenSchematic


def _to_unix_ms(dt: Optional[datetime]) -> int:
    """Convert datetime to Unix milliseconds. Returns 0 if None."""
    if dt is None:
        return 0
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


@dataclass
class RenderPayload:
    """Pre-computed payload for a single trade, ready for Pine template."""

    trade_id: int               # internal SQLite ID
    trade: TradeRecord
    schematic: Optional[FrozenSchematic]

    # ── Derived at construction ───────────────────────────────────────
    pine_label: str = ""        # e.g. "BTCUSDT_1h_bearish_1"
    direction_int: int = 0      # 1 = long, -1 = short

    # ── Pre-computed timestamps (Unix ms) ─────────────────────────────
    entry_time_ms: int = 0
    exit_time_ms: int = 0
    tap1_time_ms: int = 0
    tap2_time_ms: int = 0
    tap3_time_ms: int = 0
    range_high_time_ms: int = 0
    range_low_time_ms: int = 0
    bos_time_ms: int = 0

    def __post_init__(self):
        t = self.trade
        s = self.schematic

        self.pine_label = (
            f"{t.symbol}_{t.timeframe or 'na'}_{t.direction}_{self.trade_id}"
        )
        self.direction_int = 1 if t.direction == "bullish" else -1

        self.entry_time_ms = _to_unix_ms(t.opened_at)
        self.exit_time_ms = _to_unix_ms(t.closed_at)

        if s:
            self.tap1_time_ms = _to_unix_ms(s.tap1_time)
            self.tap2_time_ms = _to_unix_ms(s.tap2_time)
            self.tap3_time_ms = _to_unix_ms(s.tap3_time)
            self.range_high_time_ms = _to_unix_ms(s.range_high_time)
            self.range_low_time_ms = _to_unix_ms(s.range_low_time)
            self.bos_time_ms = _to_unix_ms(s.bos_time)
