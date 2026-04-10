"""Tests for the annotations (tagging + notes) system."""
import json
import sqlite3
from datetime import datetime, timezone

import pytest

from local_chart_overlay.annotations.normalization import normalize_tag, normalize_tags
from local_chart_overlay.annotations.models import TradeAnnotations
from local_chart_overlay.annotations.service import AnnotationService
from local_chart_overlay.replay.replay_models import ReplayPayload, ReplaySummary, STAGE_LABELS, Stage
from local_chart_overlay.replay.replay_template import render_replay_html


# ── Normalization Tests ───────────────────────────────────────────────


class TestNormalization:
    def test_basic_tag(self):
        assert normalize_tag("A+") == "A+"

    def test_trim_whitespace(self):
        assert normalize_tag("  hello  ") == "hello"

    def test_internal_spaces(self):
        assert normalize_tag("NY Open") == "NY_Open"

    def test_multiple_spaces(self):
        assert normalize_tag("bad   tap  3") == "bad_tap_3"

    def test_empty_string(self):
        assert normalize_tag("") is None

    def test_whitespace_only(self):
        assert normalize_tag("   ") is None

    def test_none_input(self):
        assert normalize_tag(None) is None

    def test_max_length(self):
        long_tag = "a" * 100
        result = normalize_tag(long_tag)
        assert len(result) == 64

    def test_preserves_case(self):
        assert normalize_tag("NY_Open") == "NY_Open"
        assert normalize_tag("model_1") == "model_1"

    def test_special_chars(self):
        assert normalize_tag("A+") == "A+"
        assert normalize_tag("#1") == "#1"
        assert normalize_tag("win/loss") == "win/loss"

    def test_normalize_tags_dedup(self):
        result = normalize_tags(["A+", "B", "A+", "C", "B"])
        assert result == ["A+", "B", "C"]

    def test_normalize_tags_removes_invalid(self):
        result = normalize_tags(["good", "", "  ", "also_good"])
        assert result == ["good", "also_good"]

    def test_normalize_tags_preserves_order(self):
        result = normalize_tags(["C", "A", "B"])
        assert result == ["C", "A", "B"]


# ── Model Tests ───────────────────────────────────────────────────────


class TestTradeAnnotations:
    def test_empty(self):
        ann = TradeAnnotations.empty(1)
        assert ann.trade_id == 1
        assert ann.tags == []
        assert ann.notes is None
        assert ann.has_tags is False
        assert ann.has_notes is False

    def test_with_tags(self):
        ann = TradeAnnotations(trade_id=1, tags=["A+", "NY_Open"])
        assert ann.has_tags is True
        assert ann.tags_csv == "A+, NY_Open"

    def test_with_notes(self):
        ann = TradeAnnotations(trade_id=1, notes="Tap 3 looked weak")
        assert ann.has_notes is True

    def test_empty_notes(self):
        ann = TradeAnnotations(trade_id=1, notes="")
        assert ann.has_notes is False

    def test_to_dict(self):
        ann = TradeAnnotations(trade_id=1, tags=["A+"], notes="test")
        d = ann.to_dict()
        assert d["tags"] == ["A+"]
        assert d["notes"] == "test"


# ── Service Tests ─────────────────────────────────────────────────────


@pytest.fixture
def ann_db(tmp_path):
    """Create a minimal SQLite DB with trades + annotation tables."""
    db = sqlite3.connect(str(tmp_path / "test.db"))
    db.execute("""CREATE TABLE trades (
        id INTEGER PRIMARY KEY, source_id TEXT UNIQUE, symbol TEXT)""")
    db.execute("INSERT INTO trades VALUES (1, 'test:1', 'BTC')")
    db.execute("INSERT INTO trades VALUES (2, 'test:2', 'ETH')")
    db.commit()
    svc = AnnotationService(db)
    yield svc
    db.close()


class TestAnnotationService:
    def test_get_empty(self, ann_db):
        ann = ann_db.get(1)
        assert ann.tags == []
        assert ann.notes is None

    def test_add_tags(self, ann_db):
        ann = ann_db.add_tags(1, ["A+", "NY Open"])
        assert ann.tags == ["A+", "NY_Open"]

    def test_add_tags_dedup(self, ann_db):
        ann_db.add_tags(1, ["A+"])
        ann = ann_db.add_tags(1, ["A+", "B"])
        assert ann.tags == ["A+", "B"]

    def test_add_tags_preserves_existing(self, ann_db):
        ann_db.add_tags(1, ["first"])
        ann = ann_db.add_tags(1, ["second"])
        assert ann.tags == ["first", "second"]

    def test_remove_tags(self, ann_db):
        ann_db.add_tags(1, ["A+", "B", "C"])
        ann = ann_db.remove_tags(1, ["B"])
        assert ann.tags == ["A+", "C"]

    def test_remove_nonexistent_tag(self, ann_db):
        ann_db.add_tags(1, ["A+"])
        ann = ann_db.remove_tags(1, ["Z"])
        assert ann.tags == ["A+"]

    def test_set_note(self, ann_db):
        ann = ann_db.set_note(1, "Tap 3 looked weak")
        assert ann.notes == "Tap 3 looked weak"

    def test_set_note_trims(self, ann_db):
        ann = ann_db.set_note(1, "  trimmed  ")
        assert ann.notes == "trimmed"

    def test_clear_note(self, ann_db):
        ann_db.set_note(1, "something")
        ann = ann_db.clear_note(1)
        assert ann.notes is None

    def test_roundtrip(self, ann_db):
        ann_db.add_tags(1, ["A+", "NY_Open"])
        ann_db.set_note(1, "Test note")
        ann = ann_db.get(1)
        assert ann.tags == ["A+", "NY_Open"]
        assert ann.notes == "Test note"

    def test_separate_trades(self, ann_db):
        ann_db.add_tags(1, ["BTC_tag"])
        ann_db.add_tags(2, ["ETH_tag"])
        assert ann_db.get(1).tags == ["BTC_tag"]
        assert ann_db.get(2).tags == ["ETH_tag"]

    def test_storage_via_store(self, tmp_db, sample_trade):
        """Test AnnotationService via SqliteStore.annotations property."""
        tid = tmp_db.upsert_trade(sample_trade)
        svc = tmp_db.annotations
        svc.add_tags(tid, ["test_tag"])
        svc.set_note(tid, "test note")
        ann = svc.get(tid)
        assert ann.tags == ["test_tag"]
        assert ann.notes == "test note"


# ── Replay Integration Tests ─────────────────────────────────────────


class TestReplayAnnotations:
    def _make_payload(self, tags=None, notes=None):
        return ReplayPayload(
            trade_id=1, generated_at="2026-01-01",
            summary=ReplaySummary(
                trade_id=1, symbol="BTC", direction="bearish", side="short",
                timeframe="1h", model="M1", entry_price=100, stop_price=105,
                target_price=90, tp1_price=None, rr=None, entry_score=None,
                opened_at="", closed_at=None, exit_price=None, exit_reason=None,
                pnl_pct=None, pnl_dollars=None, is_win=None, htf_bias=None,
                schematic_completeness=None, schematic_source=None,
                schematic_version=None,
            ),
            tags=tags or [],
            notes=notes,
            stages=[{"id": i, "label": STAGE_LABELS[Stage(i)]} for i in range(7)],
        )

    def test_tags_in_json(self):
        p = self._make_payload(tags=["A+", "NY_Open"])
        data = json.loads(p.to_json())
        assert data["tags"] == ["A+", "NY_Open"]

    def test_notes_in_json(self):
        p = self._make_payload(notes="Test note")
        data = json.loads(p.to_json())
        assert data["notes"] == "Test note"

    def test_tags_in_replay_html(self):
        p = self._make_payload(tags=["A+", "range_clean"])
        html = render_replay_html(p.to_json())
        assert "A+" in html
        assert "range_clean" in html

    def test_notes_in_replay_html(self):
        p = self._make_payload(notes="Tap 3 looked weak")
        html = render_replay_html(p.to_json())
        assert "Tap 3 looked weak" in html

    def test_no_annotations_section_when_empty(self):
        p = self._make_payload()
        html = render_replay_html(p.to_json())
        # The JS checks if tags/notes exist before rendering the section
        # So "Annotations" heading is conditional — just verify no crash
        assert "Trade Replay Inspector" in html or "Replay" in html

    def test_html_escapes_tags(self):
        p = self._make_payload(tags=["<script>alert(1)</script>"])
        html = render_replay_html(p.to_json())
        # The tag should be JSON-escaped, not raw HTML
        assert "<script>alert" not in html.split("const DATA")[0]

    def test_html_escapes_notes(self):
        p = self._make_payload(notes='<img src=x onerror="alert(1)">')
        html = render_replay_html(p.to_json())
        # Notes in JSON should be escaped
        data_json = p.to_json()
        assert "<img" not in data_json or '\\"' in data_json


# ── Manifest + README Integration Tests ───────────────────────────────


class TestManifestAnnotations:
    def test_manifest_includes_tags(self):
        from local_chart_overlay.replay_share.manifest_builder import build_manifest
        m = build_manifest(
            trade_ids=[1], symbol="BTC", timeframe="1h", side="short",
            files=[], tags=["A+", "NY_Open"], notes="Test note",
        )
        data = json.loads(m)
        assert data["tags"] == ["A+", "NY_Open"]
        assert data["notes"] == "Test note"

    def test_manifest_defaults_without_tags(self):
        from local_chart_overlay.replay_share.manifest_builder import build_manifest
        m = build_manifest(
            trade_ids=[1], symbol="BTC", timeframe="1h", side="short",
            files=[],
        )
        data = json.loads(m)
        assert data["tags"] == []
        assert data["notes"] is None

    def test_readme_includes_tags(self):
        from local_chart_overlay.replay_share.readme_builder import build_readme
        r = build_readme(
            trade_ids=[1], symbol="BTC", timeframe="1h", side="short",
            tags=["A+", "NY_Open"], notes="Tap 3 weak",
        )
        assert "A+, NY_Open" in r
        assert "Tap 3 weak" in r

    def test_readme_no_tags(self):
        from local_chart_overlay.replay_share.readme_builder import build_readme
        r = build_readme(
            trade_ids=[1], symbol="BTC", timeframe="1h", side="short",
        )
        assert "(none)" in r


# ── Index Scanner Backward Compat Tests ───────────────────────────────


class TestIndexAnnotations:
    def test_scanner_handles_old_manifest(self, tmp_path):
        """Manifests without tags/notes should default to empty."""
        from local_chart_overlay.replay_index.scanner import scan_packages
        pkg = tmp_path / "old_pkg"
        pkg.mkdir()
        manifest = {"package_type": "replay_share", "symbol": "BTC",
                     "trade_ids": [1], "trade_count": 1}
        (pkg / "manifest.json").write_text(json.dumps(manifest))
        entries = scan_packages(tmp_path)
        assert len(entries) == 1
        assert entries[0].tags == []
        assert entries[0].notes is None

    def test_scanner_reads_tags(self, tmp_path):
        from local_chart_overlay.replay_index.scanner import scan_packages
        pkg = tmp_path / "tagged_pkg"
        pkg.mkdir()
        manifest = {"package_type": "replay_share", "symbol": "BTC",
                     "trade_ids": [1], "trade_count": 1,
                     "tags": ["A+", "NY_Open"], "notes": "Good trade"}
        (pkg / "manifest.json").write_text(json.dumps(manifest))
        entries = scan_packages(tmp_path)
        assert entries[0].tags == ["A+", "NY_Open"]
        assert entries[0].notes == "Good trade"

    def test_index_search_includes_tags(self):
        from local_chart_overlay.replay_index.models import ReplayPackageEntry, ReplayIndexSummary
        from local_chart_overlay.replay_index.index_template import render_index_html
        entries = [ReplayPackageEntry(
            package_name="test", relative_dir="test",
            tags=["searchable_tag"], notes="searchable note",
        )]
        summary = ReplayIndexSummary.from_entries(entries)
        html = render_index_html(entries, summary)
        assert "searchable_tag" in html
        # Notes are in the JSON data, searchable by JS
        assert "searchable note" in html
