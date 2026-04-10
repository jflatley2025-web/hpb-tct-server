"""Replay builder — assembles all data needed for a trade replay."""
from __future__ import annotations

import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from local_chart_overlay.models.trade import TradeRecord
from local_chart_overlay.models.schematic import FrozenSchematic
from local_chart_overlay.models.render_payload import RenderPayload
from local_chart_overlay.rendering.pine_generator import PineGenerator
from local_chart_overlay.replay.replay_models import (
    Stage, STAGE_LABELS,
    ReplayPayload, ReplaySummary, ReplayChartPoint,
    AnchorPoint, AnchorComparison, AccuracySummary,
)


def _dt_to_ms(dt: Optional[datetime]) -> Optional[int]:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _dt_to_iso(dt: Optional[datetime]) -> Optional[str]:
    if dt is None:
        return None
    return dt.isoformat()


class ReplayBuilder:
    """Assembles a ReplayPayload from trade + schematic + OHLCV + suggestion data."""

    def build(
        self,
        trade_id: int,
        trade: TradeRecord,
        confirmed: Optional[FrozenSchematic],
        candles_df: Optional[pd.DataFrame],
        suggested_anchors: Optional[list[AnchorPoint]] = None,
        comparisons: Optional[list[AnchorComparison]] = None,
        accuracy: Optional[AccuracySummary] = None,
        tags: Optional[list[str]] = None,
        notes: Optional[str] = None,
    ) -> ReplayPayload:
        """Build the complete replay payload.

        Args:
            trade_id: Internal DB ID.
            trade: Normalized trade record.
            confirmed: Frozen schematic (may be None).
            candles_df: OHLCV DataFrame (may be None for no-chart mode).
            suggested_anchors: Pre-built suggested anchor list (optional).
            comparisons: Pre-built anchor comparisons (optional).
            accuracy: Pre-built accuracy summary (optional).
        """
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        # Summary
        summary = self._build_summary(trade_id, trade, confirmed)

        # Candles
        candles = []
        if candles_df is not None and not candles_df.empty:
            candles = [ReplayChartPoint.from_row(row) for _, row in candles_df.iterrows()]

        # Confirmed anchors
        conf_anchors = self._build_confirmed_anchors(trade, confirmed)

        # Stages
        stages = [{"id": s.value, "label": STAGE_LABELS[s]} for s in Stage]

        # Generate Pine script (reuses PineGenerator, writes to temp dir)
        pine_script = self._generate_pine_script(trade_id, trade, confirmed)

        payload = ReplayPayload(
            trade_id=trade_id,
            generated_at=now,
            summary=summary,
            candles=candles,
            confirmed_anchors=conf_anchors,
            suggested_anchors=suggested_anchors or [],
            has_suggestion=bool(suggested_anchors),
            comparisons=comparisons or [],
            has_accuracy=accuracy is not None,
            accuracy=accuracy,
            tags=tags or [],
            notes=notes,
            stages=stages,
            max_stage=Stage.OUTCOME.value,
            pine_script=pine_script,
        )
        return payload

    def _generate_pine_script(
        self, trade_id: int, trade: TradeRecord,
        confirmed: Optional[FrozenSchematic],
    ) -> str:
        """Generate Pine Script content using existing PineGenerator."""
        render_payload = RenderPayload(
            trade_id=trade_id, trade=trade, schematic=confirmed,
        )
        with tempfile.TemporaryDirectory() as tmp:
            pine_path = PineGenerator().generate_single(render_payload, Path(tmp))
            return pine_path.read_text(encoding="utf-8")

    def _build_summary(
        self, trade_id: int, trade: TradeRecord, sch: Optional[FrozenSchematic],
    ) -> ReplaySummary:
        return ReplaySummary(
            trade_id=trade_id,
            symbol=trade.symbol,
            direction=trade.direction,
            side=trade.side,
            timeframe=trade.timeframe or "n/a",
            model=trade.model or "n/a",
            entry_price=trade.entry_price,
            stop_price=trade.stop_price,
            target_price=trade.target_price,
            tp1_price=trade.tp1_price,
            rr=trade.rr,
            entry_score=trade.entry_score,
            opened_at=_dt_to_iso(trade.opened_at) or "",
            closed_at=_dt_to_iso(trade.closed_at),
            exit_price=trade.exit_price,
            exit_reason=trade.exit_reason,
            pnl_pct=trade.pnl_pct,
            pnl_dollars=trade.pnl_dollars,
            is_win=trade.is_win,
            htf_bias=trade.htf_bias,
            schematic_completeness=sch.completeness if sch else None,
            schematic_source=sch.source if sch else None,
            schematic_version=sch.version if sch else None,
        )

    def _build_confirmed_anchors(
        self, trade: TradeRecord, sch: Optional[FrozenSchematic],
    ) -> list[AnchorPoint]:
        """Build the confirmed anchor list with stage visibility."""
        anchors = []

        # Range levels — visible from pre-context (stage 0)
        if sch and sch.has_range:
            anchors.append(AnchorPoint(
                label="range_high", price=sch.range_high_price,
                time_ms=_dt_to_ms(sch.range_high_time),
                time_iso=_dt_to_iso(sch.range_high_time),
                visible_from_stage=Stage.PRE_CONTEXT,
            ))
            anchors.append(AnchorPoint(
                label="range_eq", price=sch.range_eq_price,
                visible_from_stage=Stage.PRE_CONTEXT,
            ))
            anchors.append(AnchorPoint(
                label="range_low", price=sch.range_low_price,
                time_ms=_dt_to_ms(sch.range_low_time),
                time_iso=_dt_to_iso(sch.range_low_time),
                visible_from_stage=Stage.PRE_CONTEXT,
            ))

        # Taps
        if sch and sch.tap1_price is not None:
            anchors.append(AnchorPoint(
                label="tap1", price=sch.tap1_price,
                time_ms=_dt_to_ms(sch.tap1_time),
                time_iso=_dt_to_iso(sch.tap1_time),
                visible_from_stage=Stage.TAP_1,
            ))
        if sch and sch.tap2_price is not None:
            anchors.append(AnchorPoint(
                label="tap2", price=sch.tap2_price,
                time_ms=_dt_to_ms(sch.tap2_time),
                time_iso=_dt_to_iso(sch.tap2_time),
                visible_from_stage=Stage.TAP_2,
            ))
        if sch and sch.tap3_price is not None:
            anchors.append(AnchorPoint(
                label="tap3", price=sch.tap3_price,
                time_ms=_dt_to_ms(sch.tap3_time),
                time_iso=_dt_to_iso(sch.tap3_time),
                visible_from_stage=Stage.TAP_3,
            ))

        # BOS
        if sch and sch.bos_price is not None:
            anchors.append(AnchorPoint(
                label="bos", price=sch.bos_price,
                time_ms=_dt_to_ms(sch.bos_time),
                time_iso=_dt_to_iso(sch.bos_time),
                visible_from_stage=Stage.BOS,
            ))

        # Entry / Stop / Target — visible from entry stage
        anchors.append(AnchorPoint(
            label="entry", price=trade.entry_price,
            time_ms=_dt_to_ms(trade.opened_at),
            time_iso=_dt_to_iso(trade.opened_at),
            visible_from_stage=Stage.ENTRY,
        ))
        anchors.append(AnchorPoint(
            label="stop_loss", price=trade.stop_price,
            time_ms=_dt_to_ms(trade.opened_at),
            time_iso=_dt_to_iso(trade.opened_at),
            visible_from_stage=Stage.ENTRY,
        ))
        anchors.append(AnchorPoint(
            label="target", price=trade.target_price,
            time_ms=_dt_to_ms(trade.opened_at),
            time_iso=_dt_to_iso(trade.opened_at),
            visible_from_stage=Stage.ENTRY,
        ))
        if trade.tp1_price:
            anchors.append(AnchorPoint(
                label="tp1", price=trade.tp1_price,
                time_ms=_dt_to_ms(trade.opened_at),
                time_iso=_dt_to_iso(trade.opened_at),
                visible_from_stage=Stage.ENTRY,
            ))

        # Exit — outcome stage
        if trade.exit_price and trade.closed_at:
            anchors.append(AnchorPoint(
                label="exit", price=trade.exit_price,
                time_ms=_dt_to_ms(trade.closed_at),
                time_iso=_dt_to_iso(trade.closed_at),
                visible_from_stage=Stage.OUTCOME,
            ))

        return anchors


def build_comparisons_from_schematics(
    confirmed: FrozenSchematic,
    suggested: FrozenSchematic,
    score_breakdowns: Optional[dict[str, dict]] = None,
) -> list[AnchorComparison]:
    """Build anchor-by-anchor comparisons between confirmed and suggested."""
    score_breakdowns = score_breakdowns or {}
    comps = []

    pairs = [
        ("tap1", confirmed.tap1_price, confirmed.tap1_time,
         suggested.tap1_price, suggested.tap1_time),
        ("tap2", confirmed.tap2_price, confirmed.tap2_time,
         suggested.tap2_price, suggested.tap2_time),
        ("tap3", confirmed.tap3_price, confirmed.tap3_time,
         suggested.tap3_price, suggested.tap3_time),
        ("range_high", confirmed.range_high_price, confirmed.range_high_time,
         suggested.range_high_price, suggested.range_high_time),
        ("range_low", confirmed.range_low_price, confirmed.range_low_time,
         suggested.range_low_price, suggested.range_low_time),
        ("bos", confirmed.bos_price, confirmed.bos_time,
         suggested.bos_price, suggested.bos_time),
    ]

    for name, c_price, c_time, s_price, s_time in pairs:
        price_delta = None
        price_delta_pct = None
        time_delta = None

        if c_price is not None and s_price is not None:
            price_delta = abs(s_price - c_price)
            price_delta_pct = (price_delta / c_price * 100) if c_price != 0 else 0

        if c_time is not None and s_time is not None:
            time_delta = abs((s_time - c_time).total_seconds())

        comps.append(AnchorComparison(
            anchor_name=name,
            confirmed_price=c_price,
            confirmed_time_iso=_dt_to_iso(c_time),
            suggested_price=s_price,
            suggested_time_iso=_dt_to_iso(s_time),
            price_delta=price_delta,
            price_delta_pct=price_delta_pct,
            time_delta_seconds=time_delta,
            score_breakdown=score_breakdowns.get(name),
        ))

    return comps


def build_suggested_anchors(sch: FrozenSchematic) -> list[AnchorPoint]:
    """Convert a suggested FrozenSchematic into AnchorPoint list."""
    anchors = []
    if sch.range_high_price is not None:
        anchors.append(AnchorPoint(
            label="range_high", price=sch.range_high_price,
            time_ms=_dt_to_ms(sch.range_high_time),
            time_iso=_dt_to_iso(sch.range_high_time),
            visible_from_stage=Stage.PRE_CONTEXT,
        ))
    if sch.range_low_price is not None:
        anchors.append(AnchorPoint(
            label="range_low", price=sch.range_low_price,
            time_ms=_dt_to_ms(sch.range_low_time),
            time_iso=_dt_to_iso(sch.range_low_time),
            visible_from_stage=Stage.PRE_CONTEXT,
        ))
    if sch.tap1_price is not None:
        anchors.append(AnchorPoint(
            label="tap1", price=sch.tap1_price,
            time_ms=_dt_to_ms(sch.tap1_time),
            time_iso=_dt_to_iso(sch.tap1_time),
            visible_from_stage=Stage.TAP_1,
        ))
    if sch.tap2_price is not None:
        anchors.append(AnchorPoint(
            label="tap2", price=sch.tap2_price,
            time_ms=_dt_to_ms(sch.tap2_time),
            time_iso=_dt_to_iso(sch.tap2_time),
            visible_from_stage=Stage.TAP_2,
        ))
    if sch.tap3_price is not None:
        anchors.append(AnchorPoint(
            label="tap3", price=sch.tap3_price,
            time_ms=_dt_to_ms(sch.tap3_time),
            time_iso=_dt_to_iso(sch.tap3_time),
            visible_from_stage=Stage.TAP_3,
        ))
    if sch.bos_price is not None:
        anchors.append(AnchorPoint(
            label="bos", price=sch.bos_price,
            time_ms=_dt_to_ms(sch.bos_time),
            time_iso=_dt_to_iso(sch.bos_time),
            visible_from_stage=Stage.BOS,
        ))
    return anchors
