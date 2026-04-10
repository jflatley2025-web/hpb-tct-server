"""Tests for ingestion adapters."""
import pytest

from local_chart_overlay.ingest.json_adapter import JsonAdapter
from local_chart_overlay.ingest.csv_adapter import CsvAdapter


class TestJsonAdapter:
    def test_extract_closed_trades(self, sample_json_file):
        adapter = JsonAdapter(sample_json_file)
        pairs = adapter.extract()
        # Only closed trades (2 of 3 in fixture)
        assert len(pairs) == 2

    def test_trade_fields(self, sample_json_file):
        adapter = JsonAdapter(sample_json_file)
        pairs = adapter.extract()
        trade, sch = pairs[0]
        assert trade.symbol == "BTCUSDT"
        assert trade.direction == "bearish"
        assert trade.entry_price == 71044.18
        assert trade.source_id == "json:1"
        assert trade.source_type == "json"

    def test_no_schematic_data(self, sample_json_file):
        adapter = JsonAdapter(sample_json_file)
        pairs = adapter.extract()
        for _, sch in pairs:
            assert sch is None

    def test_skips_open_trades(self, sample_json_file):
        adapter = JsonAdapter(sample_json_file)
        pairs = adapter.extract()
        source_ids = [t.source_id for t, _ in pairs]
        assert "json:99" not in source_ids

    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            JsonAdapter(tmp_path / "nonexistent.json")

    def test_source_type(self, sample_json_file):
        adapter = JsonAdapter(sample_json_file)
        assert adapter.source_type == "json"


class TestCsvAdapter:
    def test_extract(self, sample_csv_file):
        adapter = CsvAdapter(sample_csv_file)
        pairs = adapter.extract()
        assert len(pairs) == 2

    def test_trade_fields(self, sample_csv_file):
        adapter = CsvAdapter(sample_csv_file)
        pairs = adapter.extract()
        trade, _ = pairs[0]
        assert trade.symbol == "BTCUSDT"
        assert trade.direction == "bearish"
        assert trade.entry_price == 71000.0
        assert trade.source_type == "csv"

    def test_direction_normalization(self, tmp_path):
        csv = (
            "symbol,direction,entry_price,stop_price,target_price,opened_at\n"
            "BTC,long,100,95,110,2026-01-01T00:00:00+00:00\n"
            "BTC,short,100,105,90,2026-01-01T00:00:00+00:00\n"
            "BTC,buy,100,95,110,2026-01-01T00:00:00+00:00\n"
            "BTC,sell,100,105,90,2026-01-01T00:00:00+00:00\n"
        )
        path = tmp_path / "dir_test.csv"
        path.write_text(csv)
        adapter = CsvAdapter(path)
        pairs = adapter.extract()
        assert pairs[0][0].direction == "bullish"
        assert pairs[1][0].direction == "bearish"
        assert pairs[2][0].direction == "bullish"
        assert pairs[3][0].direction == "bearish"

    def test_missing_required_column(self, tmp_path):
        csv = "symbol,entry_price\nBTC,100\n"
        path = tmp_path / "bad.csv"
        path.write_text(csv)
        with pytest.raises(ValueError, match="missing required"):
            CsvAdapter(path).extract()

    def test_no_schematic_data(self, sample_csv_file):
        adapter = CsvAdapter(sample_csv_file)
        for _, sch in adapter.extract():
            assert sch is None

    def test_source_id_format(self, sample_csv_file):
        adapter = CsvAdapter(sample_csv_file)
        pairs = adapter.extract()
        assert pairs[0][0].source_id.startswith("csv:trades.csv:")

    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            CsvAdapter(tmp_path / "nonexistent.csv")

    def test_custom_mapping(self, tmp_path):
        csv = "sym,dir,ep,sp,tp,dt\nBTC,long,100,95,110,2026-01-01T00:00:00+00:00\n"
        path = tmp_path / "custom.csv"
        path.write_text(csv)
        mapping = {
            "symbol": "sym",
            "direction": "dir",
            "entry_price": "ep",
            "stop_price": "sp",
            "target_price": "tp",
            "opened_at": "dt",
        }
        adapter = CsvAdapter(path, column_mapping=mapping)
        pairs = adapter.extract()
        assert len(pairs) == 1
        assert pairs[0][0].symbol == "BTC"
