"""Tests for SQLite storage layer."""
from datetime import datetime, timezone

from local_chart_overlay.models.trade import TradeRecord
from local_chart_overlay.models.schematic import FrozenSchematic


class TestSqliteStore:
    def test_schema_creation(self, tmp_db):
        """Schema should be created on init."""
        assert tmp_db.trade_count() == 0

    def test_upsert_trade(self, tmp_db, sample_trade):
        tid = tmp_db.upsert_trade(sample_trade)
        assert tid == 1

    def test_get_trade(self, tmp_db, sample_trade):
        tid = tmp_db.upsert_trade(sample_trade)
        trade = tmp_db.get_trade(tid)
        assert trade is not None
        assert trade.symbol == "BTCUSDT"
        assert trade.entry_price == 71044.18
        assert trade.direction == "bearish"

    def test_upsert_idempotent(self, tmp_db, sample_trade):
        """Same source_id should update, not duplicate."""
        tid1 = tmp_db.upsert_trade(sample_trade)
        tid2 = tmp_db.upsert_trade(sample_trade)
        assert tid1 == tid2
        assert tmp_db.trade_count() == 1

    def test_list_trades(self, tmp_db, sample_trade, sample_trade_win):
        tmp_db.upsert_trade(sample_trade)
        tmp_db.upsert_trade(sample_trade_win)
        trades = tmp_db.list_trades()
        assert len(trades) == 2

    def test_list_trades_filter_symbol(self, tmp_db, sample_trade):
        tmp_db.upsert_trade(sample_trade)
        trades = tmp_db.list_trades(symbol="BTCUSDT")
        assert len(trades) == 1
        trades = tmp_db.list_trades(symbol="ETHUSDT")
        assert len(trades) == 0

    def test_list_trades_filter_direction(self, tmp_db, sample_trade):
        tmp_db.upsert_trade(sample_trade)
        trades = tmp_db.list_trades(direction="bearish")
        assert len(trades) == 1
        trades = tmp_db.list_trades(direction="bullish")
        assert len(trades) == 0

    def test_upsert_schematic(self, tmp_db, sample_trade, sample_schematic):
        tid = tmp_db.upsert_trade(sample_trade)
        tmp_db.upsert_schematic(tid, sample_schematic)
        sch = tmp_db.get_schematic(tid)
        assert sch is not None
        assert sch.tap1_price == 72000.0
        assert sch.range_high_price == 72500.0

    def test_schematic_roundtrip(self, tmp_db, sample_trade, sample_schematic):
        """All fields should survive save/load."""
        tid = tmp_db.upsert_trade(sample_trade)
        tmp_db.upsert_schematic(tid, sample_schematic)
        sch = tmp_db.get_schematic(tid)
        assert sch.tap1_price == sample_schematic.tap1_price
        assert sch.tap2_price == sample_schematic.tap2_price
        assert sch.tap3_price == sample_schematic.tap3_price
        assert sch.range_high_price == sample_schematic.range_high_price
        assert sch.range_low_price == sample_schematic.range_low_price
        assert sch.bos_price == sample_schematic.bos_price
        assert sch.sweep_type == sample_schematic.sweep_type

    def test_schematic_upsert_replaces(self, tmp_db, sample_trade, sample_schematic):
        tid = tmp_db.upsert_trade(sample_trade)
        tmp_db.upsert_schematic(tid, sample_schematic)

        updated = FrozenSchematic(tap1_price=99999.0, model_label="Updated")
        tmp_db.upsert_schematic(tid, updated)
        sch = tmp_db.get_schematic(tid)
        assert sch.tap1_price == 99999.0
        assert sch.tap2_price is None

    def test_get_schematic_missing(self, tmp_db, sample_trade):
        tid = tmp_db.upsert_trade(sample_trade)
        sch = tmp_db.get_schematic(tid)
        assert sch is None

    def test_trades_with_schematics(self, tmp_db, sample_trade, sample_trade_win, sample_schematic):
        tid1 = tmp_db.upsert_trade(sample_trade)
        tid2 = tmp_db.upsert_trade(sample_trade_win)
        tmp_db.upsert_schematic(tid1, sample_schematic)

        results = tmp_db.get_trades_with_schematics()
        assert len(results) == 2
        # One has schematic, one doesn't
        schematics = [sch for _, _, sch in results]
        assert any(s is not None for s in schematics)
        assert any(s is None for s in schematics)

    def test_trades_with_schematics_filtered(self, tmp_db, sample_trade, sample_trade_win):
        tid1 = tmp_db.upsert_trade(sample_trade)
        tid2 = tmp_db.upsert_trade(sample_trade_win)

        results = tmp_db.get_trades_with_schematics(trade_ids=[tid1])
        assert len(results) == 1

    def test_record_export(self, tmp_db, sample_trade):
        tid = tmp_db.upsert_trade(sample_trade)
        tmp_db.record_export([tid], "/tmp/test.pine")

    def test_boolean_roundtrip(self, tmp_db, sample_trade, sample_trade_win):
        """is_win should survive as proper bool."""
        tid1 = tmp_db.upsert_trade(sample_trade)
        tid2 = tmp_db.upsert_trade(sample_trade_win)
        t1 = tmp_db.get_trade(tid1)
        t2 = tmp_db.get_trade(tid2)
        assert t1.is_win is False
        assert t2.is_win is True

    # ── Versioning tests ─────────────────────────────────────────────

    def test_schematic_version_starts_at_1(self, tmp_db, sample_trade, sample_schematic):
        tid = tmp_db.upsert_trade(sample_trade)
        tmp_db.upsert_schematic(tid, sample_schematic)
        sch = tmp_db.get_schematic(tid)
        assert sch.version == 1

    def test_schematic_version_increments(self, tmp_db, sample_trade, sample_schematic):
        """Updating a schematic should increment version."""
        tid = tmp_db.upsert_trade(sample_trade)
        tmp_db.upsert_schematic(tid, sample_schematic)

        updated = FrozenSchematic(tap1_price=99999.0, source="manual")
        tmp_db.upsert_schematic(tid, updated)
        sch = tmp_db.get_schematic(tid)
        assert sch.version == 2

    def test_schematic_history_created(self, tmp_db, sample_trade, sample_schematic):
        """Updating should archive the old version."""
        tid = tmp_db.upsert_trade(sample_trade)
        tmp_db.upsert_schematic(tid, sample_schematic)

        updated = FrozenSchematic(tap1_price=99999.0, source="manual")
        tmp_db.upsert_schematic(tid, updated)

        history = tmp_db.get_schematic_history(tid)
        assert len(history) == 1
        assert history[0]["version"] == 1
        assert history[0]["snapshot"]["tap1_price"] == 72000.0

    def test_schematic_history_accumulates(self, tmp_db, sample_trade, sample_schematic):
        """Multiple updates should create multiple history entries."""
        tid = tmp_db.upsert_trade(sample_trade)
        tmp_db.upsert_schematic(tid, sample_schematic)

        for i in range(3):
            tmp_db.upsert_schematic(tid, FrozenSchematic(tap1_price=float(i)))

        sch = tmp_db.get_schematic(tid)
        assert sch.version == 4
        history = tmp_db.get_schematic_history(tid)
        assert len(history) == 3

    def test_schematic_source_field(self, tmp_db, sample_trade):
        tid = tmp_db.upsert_trade(sample_trade)
        sch = FrozenSchematic(tap1_price=100.0, source="imported")
        tmp_db.upsert_schematic(tid, sch)
        loaded = tmp_db.get_schematic(tid)
        assert loaded.source == "imported"

    def test_schematic_updated_at_set(self, tmp_db, sample_trade, sample_schematic):
        tid = tmp_db.upsert_trade(sample_trade)
        tmp_db.upsert_schematic(tid, sample_schematic)
        sch = tmp_db.get_schematic(tid)
        assert sch.updated_at is not None

    # ── Multi-timeframe fields ────────────────────────────────────────

    def test_multi_timeframe_roundtrip(self, tmp_db, sample_trade):
        tid = tmp_db.upsert_trade(sample_trade)
        sch = FrozenSchematic(
            tap1_price=100.0,
            context_timeframe="4h",
            execution_timeframe="15m",
            parent_structure_id=42,
        )
        tmp_db.upsert_schematic(tid, sch)
        loaded = tmp_db.get_schematic(tid)
        assert loaded.context_timeframe == "4h"
        assert loaded.execution_timeframe == "15m"
        assert loaded.parent_structure_id == 42
