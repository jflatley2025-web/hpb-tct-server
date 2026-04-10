"""Tests for share package generation."""
from datetime import datetime, timezone

from local_chart_overlay.models.render_payload import RenderPayload
from local_chart_overlay.share.html_template import render_html
from local_chart_overlay.share.package_builder import PackageBuilder


class TestHtmlTemplate:
    def test_contains_pine_script(self):
        pine = '//@version=6\nindicator("test", overlay=true)'
        html = render_html(pine)
        assert "//@version=6" in html
        assert "indicator" in html

    def test_contains_copy_button(self):
        html = render_html("// test")
        assert "Copy Pine Script" in html

    def test_contains_tradingview_link(self):
        html = render_html("// test")
        assert "https://www.tradingview.com/chart/" in html

    def test_contains_instructions(self):
        html = render_html("// test")
        assert "Pine Editor" in html
        assert "Add to chart" in html

    def test_clipboard_api(self):
        html = render_html("// test")
        assert "navigator.clipboard.writeText" in html

    def test_fallback_textarea(self):
        html = render_html("// test")
        assert "scriptArea" in html
        assert "<textarea" in html

    def test_escapes_backticks(self):
        pine = 'label.new(x, y, "test `value`")'
        html = render_html(pine)
        # Backticks should be escaped in the JS template literal
        assert "\\`value\\`" in html

    def test_escapes_template_literals(self):
        pine = 'str.tostring(x, "${value}")'
        html = render_html(pine)
        assert "\\${value}" in html

    def test_escapes_html_in_textarea(self):
        pine = '// <script>alert("xss")</script>'
        result = render_html(pine)
        # The textarea content should have HTML-escaped tags
        assert "&lt;script&gt;" in result
        # The JS template literal will contain the raw text but it's inside
        # a JS string, not interpreted as HTML. Verify the textarea is safe.
        # Find the textarea content specifically
        import re
        textarea_match = re.search(r'<textarea[^>]*>(.*?)</textarea>', result, re.DOTALL)
        assert textarea_match
        assert "<script>" not in textarea_match.group(1)

    def test_subtitle_info(self):
        html = render_html("// test", symbol="BTCUSDT", timeframe="1h", trade_count=3)
        assert "BTCUSDT" in html
        assert "1h" in html
        assert "3 trades" in html

    def test_singular_trade(self):
        html = render_html("// test", trade_count=1)
        assert "1 trade" in html
        assert "1 trades" not in html

    def test_generated_at(self):
        html = render_html("// test", generated_at="2026-04-10 12:00 UTC")
        assert "2026-04-10 12:00 UTC" in html

    def test_self_contained(self):
        """No external CSS/JS references."""
        html = render_html("// test")
        assert "https://cdn" not in html
        assert "https://unpkg" not in html
        assert "<link rel=\"stylesheet\" href=\"http" not in html


class TestPackageBuilder:
    def test_single_trade_creates_folder(self, tmp_path, sample_trade, sample_schematic):
        payload = RenderPayload(trade_id=1, trade=sample_trade, schematic=sample_schematic)
        builder = PackageBuilder(output_dir=tmp_path / "share")
        pkg = builder.build_single(payload, label="test_trade")
        assert pkg.exists()
        assert (pkg / "overlay.pine").exists()
        assert (pkg / "open_chart.html").exists()
        assert (pkg / "README.txt").exists()

    def test_pine_file_valid(self, tmp_path, sample_trade, sample_schematic):
        payload = RenderPayload(trade_id=1, trade=sample_trade, schematic=sample_schematic)
        builder = PackageBuilder(output_dir=tmp_path / "share")
        pkg = builder.build_single(payload, label="test")
        pine = (pkg / "overlay.pine").read_text()
        assert "//@version=6" in pine

    def test_html_contains_pine(self, tmp_path, sample_trade, sample_schematic):
        payload = RenderPayload(trade_id=1, trade=sample_trade, schematic=sample_schematic)
        builder = PackageBuilder(output_dir=tmp_path / "share")
        pkg = builder.build_single(payload, label="test")
        html = (pkg / "open_chart.html").read_text()
        assert "71044.18" in html  # entry price baked into Pine
        assert "Copy Pine Script" in html

    def test_readme_content(self, tmp_path, sample_trade, sample_schematic):
        payload = RenderPayload(trade_id=1, trade=sample_trade, schematic=sample_schematic)
        builder = PackageBuilder(output_dir=tmp_path / "share")
        pkg = builder.build_single(payload, label="test")
        readme = (pkg / "README.txt").read_text()
        assert "BTCUSDT" in readme
        assert "1h" in readme
        assert "Trades:" in readme
        assert "Copy Pine Script" in readme

    def test_batch_creates_folder(self, tmp_path, sample_trade, sample_trade_win):
        p1 = RenderPayload(trade_id=1, trade=sample_trade, schematic=None)
        p2 = RenderPayload(trade_id=2, trade=sample_trade_win, schematic=None)
        builder = PackageBuilder(output_dir=tmp_path / "share")
        pkg = builder.build_batch([p1, p2], label="batch_test")
        assert pkg.exists()
        assert (pkg / "overlay.pine").exists()
        assert (pkg / "open_chart.html").exists()

    def test_batch_readme_trade_count(self, tmp_path, sample_trade, sample_trade_win):
        p1 = RenderPayload(trade_id=1, trade=sample_trade, schematic=None)
        p2 = RenderPayload(trade_id=2, trade=sample_trade_win, schematic=None)
        builder = PackageBuilder(output_dir=tmp_path / "share")
        pkg = builder.build_batch([p1, p2], label="batch_test")
        readme = (pkg / "README.txt").read_text()
        assert "2" in readme  # 2 trades

    def test_no_extra_pine_files(self, tmp_path, sample_trade):
        """Only overlay.pine should remain, no tct_overlay_*.pine leftovers."""
        payload = RenderPayload(trade_id=1, trade=sample_trade, schematic=None)
        builder = PackageBuilder(output_dir=tmp_path / "share")
        pkg = builder.build_single(payload, label="clean_test")
        pine_files = list(pkg.glob("*.pine"))
        assert len(pine_files) == 1
        assert pine_files[0].name == "overlay.pine"

    def test_safe_label_characters(self, tmp_path, sample_trade):
        """Labels with special chars should produce valid folder names."""
        payload = RenderPayload(trade_id=1, trade=sample_trade, schematic=None)
        builder = PackageBuilder(output_dir=tmp_path / "share")
        pkg = builder.build_single(payload, label="BTC/USDT:1h test")
        assert pkg.exists()
        assert "/" not in pkg.name
