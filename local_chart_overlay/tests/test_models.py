"""Tests for data models."""
from datetime import datetime, timezone

from local_chart_overlay.models.trade import TradeRecord
from local_chart_overlay.models.schematic import FrozenSchematic
from local_chart_overlay.models.render_payload import RenderPayload, _to_unix_ms


class TestTradeRecord:
    def test_frozen(self, sample_trade):
        """TradeRecord should be immutable."""
        import pytest
        with pytest.raises(AttributeError):
            sample_trade.symbol = "ETHUSDT"

    def test_side_long(self):
        t = TradeRecord(
            source_id="test:1", source_type="test", symbol="X",
            direction="bullish", entry_price=100, stop_price=95,
            target_price=110, opened_at=datetime.now(timezone.utc),
        )
        assert t.side == "long"

    def test_side_short(self, sample_trade):
        assert sample_trade.side == "short"
        assert sample_trade.direction == "bearish"


class TestFrozenSchematic:
    def test_range_eq(self, sample_schematic):
        assert sample_schematic.range_eq_price == (72500.0 + 70500.0) / 2.0

    def test_range_eq_none_when_missing(self):
        sch = FrozenSchematic(range_high_price=100.0)
        assert sch.range_eq_price is None

    def test_has_taps(self, sample_schematic):
        assert sample_schematic.has_taps is True

    def test_no_taps(self):
        sch = FrozenSchematic()
        assert sch.has_taps is False

    def test_has_range(self, sample_schematic):
        assert sample_schematic.has_range is True

    def test_completeness_full(self, sample_schematic):
        assert sample_schematic.completeness == 1.0

    def test_completeness_empty(self):
        sch = FrozenSchematic()
        assert sch.completeness == 0.0

    def test_completeness_partial(self):
        sch = FrozenSchematic(tap1_price=100.0, tap1_time=datetime.now(timezone.utc))
        assert 0.0 < sch.completeness < 1.0

    def test_versioning_defaults(self):
        sch = FrozenSchematic()
        assert sch.version == 1
        assert sch.source == "manual"
        assert sch.created_at is None
        assert sch.updated_at is None

    def test_multi_timeframe_fields(self):
        sch = FrozenSchematic(
            context_timeframe="4h",
            execution_timeframe="15m",
            parent_structure_id=42,
        )
        assert sch.context_timeframe == "4h"
        assert sch.execution_timeframe == "15m"
        assert sch.parent_structure_id == 42


class TestRenderPayload:
    def test_unix_ms_conversion(self):
        dt = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        ms = _to_unix_ms(dt)
        assert ms == int(dt.timestamp() * 1000)

    def test_unix_ms_none(self):
        assert _to_unix_ms(None) == 0

    def test_direction_int(self, sample_trade, sample_schematic):
        p = RenderPayload(trade_id=1, trade=sample_trade, schematic=sample_schematic)
        assert p.direction_int == -1  # bearish = short

    def test_pine_label(self, sample_trade):
        p = RenderPayload(trade_id=42, trade=sample_trade, schematic=None)
        assert "BTCUSDT" in p.pine_label
        assert "42" in p.pine_label

    def test_timestamps_populated(self, sample_trade, sample_schematic):
        p = RenderPayload(trade_id=1, trade=sample_trade, schematic=sample_schematic)
        assert p.entry_time_ms > 0
        assert p.exit_time_ms > 0
        assert p.tap1_time_ms > 0
        assert p.tap2_time_ms > 0
        assert p.tap3_time_ms > 0

    def test_timestamps_zero_without_schematic(self, sample_trade):
        p = RenderPayload(trade_id=1, trade=sample_trade, schematic=None)
        assert p.tap1_time_ms == 0
        assert p.range_high_time_ms == 0
