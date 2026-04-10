"""Tests for replay share package generation."""
import json
from datetime import datetime, timedelta, timezone

import pandas as pd

from local_chart_overlay.models.render_payload import RenderPayload
from local_chart_overlay.models.schematic import FrozenSchematic
from local_chart_overlay.replay.replay_models import AnchorPoint, AccuracySummary
from local_chart_overlay.replay_share.manifest_builder import build_manifest
from local_chart_overlay.replay_share.readme_builder import build_readme
from local_chart_overlay.replay_share.package_builder import ReplayShareBuilder


def _make_candles_df(n=20):
    start = datetime(2026, 3, 10, 0, 0, 0, tzinfo=timezone.utc)
    rows = []
    p = 100.0
    for i in range(n):
        o = p + (i % 3) * 0.5
        rows.append({
            "open_time": start + timedelta(hours=i),
            "open": o, "high": o + 2, "low": o - 1.5, "close": o + 0.5, "volume": 1000.0,
        })
        p = o + 0.5
    return pd.DataFrame(rows)


# ── Manifest Tests ────────────────────────────────────────────────────


class TestManifestBuilder:
    def test_valid_json(self):
        m = build_manifest(
            trade_ids=[1], symbol="BTCUSDT", timeframe="1h", side="short",
            files=["replay.html", "overlay.pine"],
        )
        data = json.loads(m)
        assert data["package_type"] == "replay_share"
        assert data["trade_ids"] == [1]
        assert data["symbol"] == "BTCUSDT"

    def test_all_fields_present(self):
        m = build_manifest(
            trade_ids=[1, 2], symbol="ETH", timeframe="4h", side="long",
            files=["a.html"], has_confirmed_schematic=True,
            has_suggested_schematic=True, has_accuracy_report=True,
            model="Model_1",
        )
        data = json.loads(m)
        assert data["trade_count"] == 2
        assert data["has_confirmed_schematic"] is True
        assert data["has_suggested_schematic"] is True
        assert data["has_accuracy_report"] is True
        assert data["model"] == "Model_1"
        assert "created_at" in data
        assert data["version"] == "1.0"

    def test_empty_trade_ids(self):
        m = build_manifest(
            trade_ids=[], symbol="X", timeframe="1m", side="long", files=[],
        )
        data = json.loads(m)
        assert data["trade_count"] == 0


# ── README Tests ──────────────────────────────────────────────────────


class TestReadmeBuilder:
    def test_contains_instructions(self):
        r = build_readme(
            trade_ids=[1], symbol="BTCUSDT", timeframe="1h", side="short",
        )
        assert "replay.html" in r
        assert "open_chart.html" in r
        assert "Copy Pine Script" in r
        assert "Add to chart" in r

    def test_contains_trade_details(self):
        r = build_readme(
            trade_ids=[1, 2], symbol="ETHUSDT", timeframe="4h", side="long",
            model="Model_2", has_confirmed=True, has_suggested=True,
            has_accuracy=True,
        )
        assert "ETHUSDT" in r
        assert "4h" in r
        assert "long" in r
        assert "Model_2" in r
        assert "Confirmed schematic: yes" in r
        assert "Suggested overlay:   yes" in r
        assert "Accuracy report:     yes" in r
        assert "1, 2" in r

    def test_no_schematic_flags(self):
        r = build_readme(
            trade_ids=[1], symbol="X", timeframe="1m", side="short",
        )
        assert "Confirmed schematic: no" in r
        assert "Suggested overlay:   no" in r
        assert "Accuracy report:     no" in r


# ── Package Builder Tests ─────────────────────────────────────────────


class TestReplayShareBuilder:
    def test_creates_all_files(self, tmp_path, sample_trade, sample_schematic):
        builder = ReplayShareBuilder(output_dir=tmp_path / "share")
        pkg = builder.build(
            trade_id=1, trade=sample_trade, confirmed=sample_schematic,
            candles_df=_make_candles_df(),
        )
        assert (pkg / "replay.html").exists()
        assert (pkg / "replay_data.json").exists()
        assert (pkg / "overlay.pine").exists()
        assert (pkg / "open_chart.html").exists()
        assert (pkg / "README.txt").exists()
        assert (pkg / "manifest.json").exists()

    def test_replay_html_contains_trade_data(self, tmp_path, sample_trade, sample_schematic):
        builder = ReplayShareBuilder(output_dir=tmp_path / "share")
        pkg = builder.build(
            trade_id=1, trade=sample_trade, confirmed=sample_schematic,
            candles_df=_make_candles_df(),
        )
        html = (pkg / "replay.html").read_text(encoding="utf-8")
        assert "BTCUSDT" in html
        assert "71044.18" in html
        assert "prevStage" in html

    def test_pine_file_valid(self, tmp_path, sample_trade):
        builder = ReplayShareBuilder(output_dir=tmp_path / "share")
        pkg = builder.build(
            trade_id=1, trade=sample_trade, confirmed=None,
            candles_df=None,
        )
        pine = (pkg / "overlay.pine").read_text(encoding="utf-8")
        assert "//@version=6" in pine
        assert "71044.18" in pine

    def test_pine_launcher_has_copy(self, tmp_path, sample_trade):
        builder = ReplayShareBuilder(output_dir=tmp_path / "share")
        pkg = builder.build(
            trade_id=1, trade=sample_trade, confirmed=None,
            candles_df=None,
        )
        html = (pkg / "open_chart.html").read_text(encoding="utf-8")
        assert "Copy Pine Script" in html
        assert "navigator.clipboard.writeText" in html

    def test_manifest_valid_json(self, tmp_path, sample_trade, sample_schematic):
        builder = ReplayShareBuilder(output_dir=tmp_path / "share")
        pkg = builder.build(
            trade_id=1, trade=sample_trade, confirmed=sample_schematic,
            candles_df=None,
        )
        m = json.loads((pkg / "manifest.json").read_text())
        assert m["trade_ids"] == [1]
        assert m["symbol"] == "BTCUSDT"
        assert m["has_confirmed_schematic"] is True
        assert m["side"] == "short"

    def test_manifest_without_schematic(self, tmp_path, sample_trade):
        builder = ReplayShareBuilder(output_dir=tmp_path / "share")
        pkg = builder.build(
            trade_id=1, trade=sample_trade, confirmed=None,
            candles_df=None,
        )
        m = json.loads((pkg / "manifest.json").read_text())
        assert m["has_confirmed_schematic"] is False

    def test_readme_content(self, tmp_path, sample_trade, sample_schematic):
        builder = ReplayShareBuilder(output_dir=tmp_path / "share")
        pkg = builder.build(
            trade_id=1, trade=sample_trade, confirmed=sample_schematic,
            candles_df=None,
        )
        readme = (pkg / "README.txt").read_text()
        assert "BTCUSDT" in readme
        assert "Confirmed schematic: yes" in readme

    def test_replay_data_json_valid(self, tmp_path, sample_trade):
        builder = ReplayShareBuilder(output_dir=tmp_path / "share")
        pkg = builder.build(
            trade_id=1, trade=sample_trade, confirmed=None,
            candles_df=_make_candles_df(),
        )
        data = json.loads((pkg / "replay_data.json").read_text())
        assert data["trade_id"] == 1
        assert len(data["candles"]) == 20

    def test_with_suggested_anchors(self, tmp_path, sample_trade, sample_schematic):
        sugg = [AnchorPoint("tap1", price=72050.0, visible_from_stage=1)]
        builder = ReplayShareBuilder(output_dir=tmp_path / "share")
        pkg = builder.build(
            trade_id=1, trade=sample_trade, confirmed=sample_schematic,
            candles_df=None, suggested_anchors=sugg,
        )
        m = json.loads((pkg / "manifest.json").read_text())
        assert m["has_suggested_schematic"] is True

    def test_with_accuracy(self, tmp_path, sample_trade, sample_schematic):
        acc = AccuracySummary(hit_rate=0.8, avg_price_error_pct=0.1,
                              avg_time_error_seconds=1800)
        builder = ReplayShareBuilder(output_dir=tmp_path / "share")
        pkg = builder.build(
            trade_id=1, trade=sample_trade, confirmed=sample_schematic,
            candles_df=None, accuracy=acc,
        )
        m = json.loads((pkg / "manifest.json").read_text())
        assert m["has_accuracy_report"] is True
        data = json.loads((pkg / "replay_data.json").read_text())
        assert data["has_accuracy"] is True

    def test_custom_label(self, tmp_path, sample_trade):
        builder = ReplayShareBuilder(output_dir=tmp_path / "share")
        pkg = builder.build(
            trade_id=1, trade=sample_trade, confirmed=None,
            candles_df=None, label="BTCUSDT_1h",
        )
        assert "BTCUSDT_1h" in pkg.name

    def test_safe_label(self, tmp_path, sample_trade):
        builder = ReplayShareBuilder(output_dir=tmp_path / "share")
        pkg = builder.build(
            trade_id=1, trade=sample_trade, confirmed=None,
            candles_df=None, label="BTC/USDT:1h test",
        )
        assert pkg.exists()
        assert "/" not in pkg.name

    def test_no_extra_pine_files(self, tmp_path, sample_trade):
        builder = ReplayShareBuilder(output_dir=tmp_path / "share")
        pkg = builder.build(
            trade_id=1, trade=sample_trade, confirmed=None,
            candles_df=None,
        )
        pine_files = list(pkg.glob("*.pine"))
        assert len(pine_files) == 1
        assert pine_files[0].name == "overlay.pine"

    def test_no_candles_still_works(self, tmp_path, sample_trade, sample_schematic):
        """Package should work without OHLCV data (chart says 'No candle data')."""
        builder = ReplayShareBuilder(output_dir=tmp_path / "share")
        pkg = builder.build(
            trade_id=1, trade=sample_trade, confirmed=sample_schematic,
            candles_df=None,
        )
        assert (pkg / "replay.html").exists()
        data = json.loads((pkg / "replay_data.json").read_text())
        assert data["candles"] == []
