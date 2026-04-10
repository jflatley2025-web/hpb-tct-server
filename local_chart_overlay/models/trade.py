"""Normalized trade record — the universal format for all ingested trades."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass(frozen=True)
class TradeRecord:
    """Immutable normalized trade record.

    Every ingestion adapter must produce TradeRecord instances.
    Fields map to the union of live JSON, backtest DB, and CSV sources.
    """

    # ── Identity ──────────────────────────────────────────────────────
    source_id: str          # provenance key, e.g. "json:1", "csv:trades.csv:3"
    source_type: str        # "json" | "csv" | "postgres"

    # ── Core ──────────────────────────────────────────────────────────
    symbol: str             # e.g. "BTCUSDT"
    direction: str          # "bullish" | "bearish"
    entry_price: float
    stop_price: float
    target_price: float
    opened_at: datetime     # UTC, timezone-aware

    # ── Optional core ─────────────────────────────────────────────────
    timeframe: Optional[str] = None
    model: Optional[str] = None
    tp1_price: Optional[float] = None
    closed_at: Optional[datetime] = None

    # ── Outcome ───────────────────────────────────────────────────────
    pnl_pct: Optional[float] = None
    pnl_dollars: Optional[float] = None
    is_win: Optional[bool] = None
    exit_reason: Optional[str] = None
    exit_price: Optional[float] = None

    # ── Sizing / scoring ──────────────────────────────────────────────
    entry_score: Optional[int] = None
    rr: Optional[float] = None
    leverage: Optional[int] = None
    position_size: Optional[float] = None
    risk_amount: Optional[float] = None

    # ── Backtest-only ─────────────────────────────────────────────────
    mfe: Optional[float] = None
    mae: Optional[float] = None

    # ── Metadata ──────────────────────────────────────────────────────
    entry_reasons: Optional[list] = field(default=None, repr=False)
    htf_bias: Optional[str] = None
    metadata: Optional[dict] = field(default=None, repr=False)

    @property
    def side(self) -> str:
        """Long/short derived from direction."""
        return "long" if self.direction == "bullish" else "short"
