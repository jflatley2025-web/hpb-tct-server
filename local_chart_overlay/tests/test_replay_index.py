"""Tests for the replay index page generator."""
import json
from pathlib import Path

import pytest

from local_chart_overlay.replay_index.models import (
    ReplayPackageEntry, ReplayPackageFiles, ReplayIndexSummary,
)
from local_chart_overlay.replay_index.scanner import scan_packages
from local_chart_overlay.replay_index.index_template import render_index_html
from local_chart_overlay.replay_index.index_builder import IndexBuilder


# ── Helpers ───────────────────────────────────────────────────────────


def _create_package(
    root: Path,
    name: str,
    symbol: str = "BTCUSDT",
    timeframe: str = "1h",
    side: str = "short",
    trade_ids: list = None,
    has_confirmed: bool = True,
    has_suggested: bool = False,
    has_accuracy: bool = False,
    omit_files: list = None,
):
    """Create a synthetic replay package folder for testing."""
    trade_ids = trade_ids or [1]
    omit_files = omit_files or []
    pkg = root / name
    pkg.mkdir(parents=True, exist_ok=True)

    manifest = {
        "package_type": "replay_share",
        "version": "1.0",
        "created_at": "2026-04-10T12:00:00+00:00",
        "trade_ids": trade_ids,
        "trade_count": len(trade_ids),
        "symbol": symbol,
        "timeframe": timeframe,
        "side": side,
        "model": "Model_1",
        "files": [],
        "has_confirmed_schematic": has_confirmed,
        "has_suggested_schematic": has_suggested,
        "has_accuracy_report": has_accuracy,
    }
    (pkg / "manifest.json").write_text(json.dumps(manifest))

    standard_files = ["replay.html", "replay_data.json", "overlay.pine",
                      "open_chart.html", "README.txt"]
    for f in standard_files:
        if f not in omit_files:
            (pkg / f).write_text(f"stub: {f}")

    return pkg


# ── Model Tests ───────────────────────────────────────────────────────


class TestIndexModels:
    def test_package_files_complete(self):
        f = ReplayPackageFiles(
            has_replay_html=True, has_replay_data=True,
            has_overlay_pine=True, has_open_chart=True,
            has_readme=True, has_manifest=True,
        )
        assert f.complete is True
        assert f.file_count == 6

    def test_package_files_incomplete(self):
        f = ReplayPackageFiles(has_replay_html=True, has_manifest=True)
        assert f.complete is False
        assert f.file_count == 2

    def test_entry_links(self):
        e = ReplayPackageEntry(
            package_name="test",
            relative_dir="test",
            files=ReplayPackageFiles(
                has_replay_html=True, has_open_chart=True, has_readme=True,
            ),
        )
        assert e.replay_link == "test/replay.html"
        assert e.chart_link == "test/open_chart.html"
        assert e.readme_link == "test/README.txt"

    def test_entry_links_missing(self):
        e = ReplayPackageEntry(
            package_name="test", relative_dir="test",
            files=ReplayPackageFiles(),
        )
        assert e.replay_link is None
        assert e.chart_link is None
        assert e.readme_link is None

    def test_summary_from_entries(self):
        entries = [
            ReplayPackageEntry(
                package_name="a", relative_dir="a",
                symbol="BTC", timeframe="1h", side="short",
                trade_count=1, has_confirmed_schematic=True,
            ),
            ReplayPackageEntry(
                package_name="b", relative_dir="b",
                symbol="ETH", timeframe="4h", side="long",
                trade_count=2, has_suggested_schematic=True,
                has_accuracy_report=True,
            ),
        ]
        s = ReplayIndexSummary.from_entries(entries)
        assert s.total_packages == 2
        assert s.total_trades == 3
        assert "BTC" in s.unique_symbols
        assert "ETH" in s.unique_symbols
        assert s.with_confirmed == 1
        assert s.with_suggested == 1
        assert s.with_accuracy == 1


# ── Scanner Tests ─────────────────────────────────────────────────────


class TestScanner:
    def test_finds_packages(self, tmp_path):
        _create_package(tmp_path, "pkg_a")
        _create_package(tmp_path, "pkg_b", symbol="ETHUSDT")
        entries = scan_packages(tmp_path)
        assert len(entries) == 2

    def test_sorted_by_name(self, tmp_path):
        _create_package(tmp_path, "zzz")
        _create_package(tmp_path, "aaa")
        entries = scan_packages(tmp_path)
        assert entries[0].package_name == "aaa"
        assert entries[1].package_name == "zzz"

    def test_reads_manifest_fields(self, tmp_path):
        _create_package(tmp_path, "test", symbol="SOLUSDT", timeframe="4h",
                        side="long", trade_ids=[5, 6], has_confirmed=True,
                        has_suggested=True, has_accuracy=True)
        entries = scan_packages(tmp_path)
        assert len(entries) == 1
        e = entries[0]
        assert e.symbol == "SOLUSDT"
        assert e.timeframe == "4h"
        assert e.side == "long"
        assert e.trade_ids == [5, 6]
        assert e.trade_count == 2
        assert e.has_confirmed_schematic is True
        assert e.has_suggested_schematic is True
        assert e.has_accuracy_report is True

    def test_detects_file_presence(self, tmp_path):
        _create_package(tmp_path, "full")
        entries = scan_packages(tmp_path)
        f = entries[0].files
        assert f.has_replay_html is True
        assert f.has_overlay_pine is True
        assert f.has_open_chart is True
        assert f.has_readme is True
        assert f.has_manifest is True

    def test_detects_missing_files(self, tmp_path):
        _create_package(tmp_path, "partial", omit_files=["overlay.pine", "open_chart.html"])
        entries = scan_packages(tmp_path)
        f = entries[0].files
        assert f.has_overlay_pine is False
        assert f.has_open_chart is False
        assert f.has_replay_html is True

    def test_skips_no_manifest(self, tmp_path):
        """Folders without manifest.json are skipped."""
        (tmp_path / "no_manifest").mkdir()
        (tmp_path / "no_manifest" / "replay.html").write_text("stub")
        entries = scan_packages(tmp_path)
        assert len(entries) == 0

    def test_skips_invalid_manifest(self, tmp_path):
        pkg = tmp_path / "bad_json"
        pkg.mkdir()
        (pkg / "manifest.json").write_text("not json {{{")
        entries = scan_packages(tmp_path)
        assert len(entries) == 0

    def test_skips_non_object_manifest(self, tmp_path):
        pkg = tmp_path / "array_manifest"
        pkg.mkdir()
        (pkg / "manifest.json").write_text("[1, 2, 3]")
        entries = scan_packages(tmp_path)
        assert len(entries) == 0

    def test_relative_paths(self, tmp_path):
        _create_package(tmp_path, "pkg_a")
        entries = scan_packages(tmp_path, relative_to=tmp_path)
        assert entries[0].relative_dir == "pkg_a"
        assert entries[0].replay_link == "pkg_a/replay.html"

    def test_recursive_scan(self, tmp_path):
        _create_package(tmp_path / "group1", "trade_1")
        _create_package(tmp_path / "group2", "trade_2")
        # Non-recursive should find nothing (packages are nested)
        entries_flat = scan_packages(tmp_path, recursive=False)
        assert len(entries_flat) == 0
        # Recursive should find both
        entries_deep = scan_packages(tmp_path, recursive=True)
        assert len(entries_deep) == 2

    def test_empty_dir(self, tmp_path):
        entries = scan_packages(tmp_path)
        assert entries == []

    def test_nonexistent_dir(self, tmp_path):
        entries = scan_packages(tmp_path / "nope")
        assert entries == []


# ── Template Tests ────────────────────────────────────────────────────


class TestIndexTemplate:
    def _make_entries(self, n=2):
        return [
            ReplayPackageEntry(
                package_name=f"pkg_{i}", relative_dir=f"pkg_{i}",
                trade_ids=[i], trade_count=1,
                symbol="BTCUSDT" if i % 2 == 0 else "ETHUSDT",
                timeframe="1h", side="short",
                created_at="2026-04-10T12:00:00",
                has_confirmed_schematic=True,
                files=ReplayPackageFiles(
                    has_replay_html=True, has_overlay_pine=True,
                    has_open_chart=True, has_readme=True, has_manifest=True,
                ),
            )
            for i in range(n)
        ]

    def test_html_contains_entries(self):
        entries = self._make_entries(3)
        summary = ReplayIndexSummary.from_entries(entries)
        html = render_index_html(entries, summary)
        assert "pkg_0" in html
        assert "pkg_1" in html
        assert "pkg_2" in html

    def test_html_has_search(self):
        html = render_index_html([], ReplayIndexSummary())
        assert "searchBox" in html
        assert "applyFilters" in html

    def test_html_has_filters(self):
        html = render_index_html([], ReplayIndexSummary())
        assert "filterSymbol" in html
        assert "filterTf" in html
        assert "filterSide" in html
        assert "filterFlag" in html

    def test_html_has_sort(self):
        html = render_index_html([], ReplayIndexSummary())
        assert "sortBy" in html

    def test_html_has_summary_metrics(self):
        entries = self._make_entries(5)
        summary = ReplayIndexSummary.from_entries(entries)
        html = render_index_html(entries, summary)
        assert "5" in html  # total packages
        assert "Packages" in html

    def test_html_self_contained(self):
        html = render_index_html([], ReplayIndexSummary())
        assert "https://cdn" not in html
        assert "https://unpkg" not in html

    def test_html_has_replay_links(self):
        entries = self._make_entries(1)
        summary = ReplayIndexSummary.from_entries(entries)
        html = render_index_html(entries, summary)
        assert "pkg_0/replay.html" in html

    def test_html_escapes_package_names(self):
        entries = [
            ReplayPackageEntry(
                package_name='<script>alert(1)</script>',
                relative_dir="safe_dir",
                files=ReplayPackageFiles(has_manifest=True),
            ),
        ]
        summary = ReplayIndexSummary.from_entries(entries)
        html = render_index_html(entries, summary)
        # The script tag should be escaped in the JSON
        assert "<script>alert" not in html.split("const ENTRIES")[0]

    def test_empty_state(self):
        html = render_index_html([], ReplayIndexSummary())
        assert "No packages match" in html
        assert "0" in html


# ── Builder Tests ─────────────────────────────────────────────────────


class TestIndexBuilder:
    def test_builds_index_file(self, tmp_path):
        _create_package(tmp_path, "pkg_a")
        _create_package(tmp_path, "pkg_b", symbol="ETHUSDT")
        builder = IndexBuilder()
        out = builder.build(input_dir=tmp_path)
        assert out.exists()
        assert out.name == "index.html"
        html = out.read_text(encoding="utf-8")
        assert "pkg_a" in html
        assert "pkg_b" in html

    def test_custom_output_path(self, tmp_path):
        _create_package(tmp_path / "packages", "pkg_a")
        builder = IndexBuilder()
        custom = tmp_path / "my_index.html"
        out = builder.build(
            input_dir=tmp_path / "packages",
            output_path=custom,
        )
        assert out == custom
        assert custom.exists()

    def test_recursive_build(self, tmp_path):
        _create_package(tmp_path / "group", "pkg_a")
        builder = IndexBuilder()
        out = builder.build(input_dir=tmp_path, recursive=True)
        html = out.read_text(encoding="utf-8")
        assert "pkg_a" in html

    def test_empty_dir_still_works(self, tmp_path):
        builder = IndexBuilder()
        out = builder.build(input_dir=tmp_path)
        assert out.exists()
        html = out.read_text(encoding="utf-8")
        assert "0" in html  # 0 packages
