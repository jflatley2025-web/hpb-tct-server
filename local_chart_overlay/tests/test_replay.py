"""Tests for the trade replay inspector."""
import json
from datetime import datetime, timedelta, timezone

import pandas as pd

from local_chart_overlay.models.trade import TradeRecord
from local_chart_overlay.models.schematic import FrozenSchematic
from local_chart_overlay.replay.replay_models import (
    Stage, STAGE_LABELS, ReplayPayload, ReplayChartPoint,
    AnchorPoint, AnchorComparison, AccuracySummary, ReplaySummary,
)
from local_chart_overlay.replay.replay_builder import (
    ReplayBuilder, build_comparisons_from_schematics, build_suggested_anchors,
)
from local_chart_overlay.replay.replay_template import render_replay_html


def _make_candles_df(n=20):
    """Build a small synthetic OHLCV DataFrame."""
    start = datetime(2026, 3, 10, 0, 0, 0, tzinfo=timezone.utc)
    rows = []
    price = 100.0
    for i in range(n):
        o = price + (i % 3) * 0.5
        h = o + 2
        l = o - 1.5
        c = o + 0.5
        rows.append({
            "open_time": start + timedelta(hours=i),
            "open": o, "high": h, "low": l, "close": c, "volume": 1000.0,
        })
        price = c
    return pd.DataFrame(rows)


# ── Model Tests ───────────────────────────────────────────────────────


class TestReplayModels:
    def test_stage_enum(self):
        assert Stage.PRE_CONTEXT == 0
        assert Stage.OUTCOME == 6
        assert len(Stage) == 7

    def test_stage_labels(self):
        assert STAGE_LABELS[Stage.TAP_1] == "Tap 1"
        assert STAGE_LABELS[Stage.BOS] == "Break of Structure"

    def test_chart_point_from_row(self):
        row = {
            "open_time": datetime(2026, 3, 1, tzinfo=timezone.utc),
            "open": 100.0, "high": 105.0, "low": 95.0,
            "close": 102.0, "volume": 500.0,
        }
        cp = ReplayChartPoint.from_row(row)
        assert cp.open == 100.0
        assert cp.high == 105.0
        assert cp.time_ms > 0
        assert "2026" in cp.time_iso

    def test_anchor_point_to_dict(self):
        a = AnchorPoint(label="tap1", price=100.0, visible_from_stage=1)
        d = a.to_dict()
        assert d["label"] == "tap1"
        assert d["visible_from_stage"] == 1

    def test_replay_payload_to_json(self):
        payload = ReplayPayload(
            trade_id=1,
            generated_at="2026-01-01",
            summary=ReplaySummary(
                trade_id=1, symbol="BTC", direction="bearish", side="short",
                timeframe="1h", model="M1", entry_price=100, stop_price=105,
                target_price=90, tp1_price=95, rr=2.0, entry_score=90,
                opened_at="2026-01-01T00:00:00", closed_at=None,
                exit_price=None, exit_reason=None, pnl_pct=None,
                pnl_dollars=None, is_win=None, htf_bias=None,
                schematic_completeness=None, schematic_source=None,
                schematic_version=None,
            ),
        )
        j = payload.to_json()
        data = json.loads(j)
        assert data["trade_id"] == 1
        assert data["summary"]["symbol"] == "BTC"

    def test_accuracy_summary(self):
        a = AccuracySummary(
            hit_rate=0.8,
            avg_price_error_pct=0.1,
            avg_time_error_seconds=3600,
            anchor_results=[{"name": "tap1", "hit": True, "price_err_pct": 0.05}],
        )
        assert a.hit_rate == 0.8
        assert len(a.anchor_results) == 1


# ── Builder Tests ─────────────────────────────────────────────────────


class TestReplayBuilder:
    def test_build_minimal(self, sample_trade):
        builder = ReplayBuilder()
        payload = builder.build(
            trade_id=1, trade=sample_trade,
            confirmed=None, candles_df=None,
        )
        assert payload.trade_id == 1
        assert payload.summary.symbol == "BTCUSDT"
        assert len(payload.candles) == 0
        assert payload.max_stage == 6

    def test_build_with_candles(self, sample_trade):
        df = _make_candles_df(20)
        builder = ReplayBuilder()
        payload = builder.build(
            trade_id=1, trade=sample_trade,
            confirmed=None, candles_df=df,
        )
        assert len(payload.candles) == 20
        assert payload.candles[0].open > 0

    def test_build_with_schematic(self, sample_trade, sample_schematic):
        builder = ReplayBuilder()
        payload = builder.build(
            trade_id=1, trade=sample_trade,
            confirmed=sample_schematic, candles_df=None,
        )
        # Should have confirmed anchors
        labels = [a.label for a in payload.confirmed_anchors]
        assert "tap1" in labels
        assert "tap2" in labels
        assert "tap3" in labels
        assert "range_high" in labels
        assert "range_low" in labels
        assert "range_eq" in labels
        assert "entry" in labels
        assert "stop_loss" in labels

    def test_anchor_stage_visibility(self, sample_trade, sample_schematic):
        builder = ReplayBuilder()
        payload = builder.build(
            trade_id=1, trade=sample_trade,
            confirmed=sample_schematic, candles_df=None,
        )
        anchors = {a.label: a for a in payload.confirmed_anchors}
        assert anchors["range_high"].visible_from_stage == Stage.PRE_CONTEXT
        assert anchors["tap1"].visible_from_stage == Stage.TAP_1
        assert anchors["tap2"].visible_from_stage == Stage.TAP_2
        assert anchors["tap3"].visible_from_stage == Stage.TAP_3
        assert anchors["bos"].visible_from_stage == Stage.BOS
        assert anchors["entry"].visible_from_stage == Stage.ENTRY

    def test_exit_anchor_at_outcome(self, sample_trade, sample_schematic):
        """Exit anchor should only appear at outcome stage."""
        builder = ReplayBuilder()
        payload = builder.build(
            trade_id=1, trade=sample_trade,
            confirmed=sample_schematic, candles_df=None,
        )
        exits = [a for a in payload.confirmed_anchors if a.label == "exit"]
        assert len(exits) == 1
        assert exits[0].visible_from_stage == Stage.OUTCOME

    def test_stages_list(self, sample_trade):
        builder = ReplayBuilder()
        payload = builder.build(
            trade_id=1, trade=sample_trade,
            confirmed=None, candles_df=None,
        )
        assert len(payload.stages) == 7
        assert payload.stages[0]["label"] == "Pre-Context"
        assert payload.stages[6]["label"] == "Outcome"

    def test_build_with_suggestion(self, sample_trade, sample_schematic):
        sugg_anchors = [
            AnchorPoint("tap1", price=72050.0, visible_from_stage=1),
            AnchorPoint("tap2", price=72250.0, visible_from_stage=2),
        ]
        builder = ReplayBuilder()
        payload = builder.build(
            trade_id=1, trade=sample_trade,
            confirmed=sample_schematic, candles_df=None,
            suggested_anchors=sugg_anchors,
        )
        assert payload.has_suggestion is True
        assert len(payload.suggested_anchors) == 2

    def test_build_with_accuracy(self, sample_trade, sample_schematic):
        acc = AccuracySummary(hit_rate=0.75, avg_price_error_pct=0.2,
                              avg_time_error_seconds=1800)
        builder = ReplayBuilder()
        payload = builder.build(
            trade_id=1, trade=sample_trade,
            confirmed=sample_schematic, candles_df=None,
            accuracy=acc,
        )
        assert payload.has_accuracy is True
        assert payload.accuracy.hit_rate == 0.75

    def test_summary_pnl(self, sample_trade):
        builder = ReplayBuilder()
        payload = builder.build(
            trade_id=1, trade=sample_trade,
            confirmed=None, candles_df=None,
        )
        assert payload.summary.pnl_pct == -1.61
        assert payload.summary.is_win is False


# ── Comparison Tests ──────────────────────────────────────────────────


class TestComparisons:
    def test_build_comparisons(self, sample_schematic):
        suggested = FrozenSchematic(
            tap1_price=72050.0,
            tap1_time=datetime(2026, 3, 12, 10, 30, 0, tzinfo=timezone.utc),
            tap2_price=72200.0,
            range_high_price=72600.0,
            range_low_price=70400.0,
        )
        comps = build_comparisons_from_schematics(sample_schematic, suggested)
        assert len(comps) == 6  # tap1,2,3 + range_high,low + bos

        tap1 = [c for c in comps if c.anchor_name == "tap1"][0]
        assert tap1.confirmed_price == 72000.0
        assert tap1.suggested_price == 72050.0
        assert tap1.price_delta == 50.0
        assert tap1.price_delta_pct is not None

    def test_comparison_with_missing(self, sample_schematic):
        suggested = FrozenSchematic()  # all None
        comps = build_comparisons_from_schematics(sample_schematic, suggested)
        # Confirmed has values, suggested doesn't — deltas should be None
        tap1 = [c for c in comps if c.anchor_name == "tap1"][0]
        assert tap1.price_delta is None

    def test_build_suggested_anchors(self):
        sch = FrozenSchematic(
            tap1_price=100.0,
            tap1_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
            range_high_price=110.0,
            bos_price=95.0,
        )
        anchors = build_suggested_anchors(sch)
        labels = [a.label for a in anchors]
        assert "tap1" in labels
        assert "range_high" in labels
        assert "bos" in labels


# ── Template Tests ────────────────────────────────────────────────────


class TestReplayTemplate:
    def _make_payload_json(self) -> str:
        payload = ReplayPayload(
            trade_id=1,
            generated_at="2026-01-01",
            summary=ReplaySummary(
                trade_id=1, symbol="BTCUSDT", direction="bearish", side="short",
                timeframe="1h", model="Model_1", entry_price=71044.18,
                stop_price=72036.35, target_price=68979.18, tp1_price=70011.68,
                rr=2.08, entry_score=93,
                opened_at="2026-03-13T19:01:28", closed_at="2026-03-15T21:38:24",
                exit_price=72187.18, exit_reason="stop_hit",
                pnl_pct=-1.61, pnl_dollars=-57.6, is_win=False,
                htf_bias="bullish",
                schematic_completeness=1.0, schematic_source="manual",
                schematic_version=1,
            ),
            confirmed_anchors=[
                AnchorPoint("tap1", price=72000.0, visible_from_stage=1),
                AnchorPoint("entry", price=71044.18, visible_from_stage=5),
            ],
            stages=[{"id": i, "label": STAGE_LABELS[Stage(i)]} for i in range(7)],
        )
        return payload.to_json()

    def test_html_contains_data(self):
        html = render_replay_html(self._make_payload_json())
        assert "BTCUSDT" in html
        assert "71044.18" in html

    def test_html_has_stage_controls(self):
        html = render_replay_html(self._make_payload_json())
        assert "prevStage" in html
        assert "nextStage" in html
        assert "stage-btn" in html

    def test_html_has_toggles(self):
        html = render_replay_html(self._make_payload_json())
        assert "togConfirmed" in html
        assert "togSuggested" in html
        assert "togLabels" in html

    def test_html_has_svg_chart(self):
        html = render_replay_html(self._make_payload_json())
        assert "<svg" in html
        assert "chart" in html

    def test_html_has_keyboard_nav(self):
        html = render_replay_html(self._make_payload_json())
        assert "ArrowRight" in html
        assert "ArrowLeft" in html

    def test_html_self_contained(self):
        html = render_replay_html(self._make_payload_json())
        assert "https://cdn" not in html
        assert "https://unpkg" not in html

    def test_html_no_script_injection(self):
        """JSON with </script> should be safely escaped."""
        payload = ReplayPayload(
            trade_id=1,
            generated_at="</script><script>alert(1)</script>",
            summary=ReplaySummary(
                trade_id=1, symbol="X", direction="bullish", side="long",
                timeframe="1h", model="M", entry_price=1, stop_price=0.9,
                target_price=1.1, tp1_price=None, rr=None, entry_score=None,
                opened_at="", closed_at=None, exit_price=None, exit_reason=None,
                pnl_pct=None, pnl_dollars=None, is_win=None, htf_bias=None,
                schematic_completeness=None, schematic_source=None,
                schematic_version=None,
            ),
        )
        html = render_replay_html(payload.to_json())
        assert "</script><script>" not in html

    def test_title_override(self):
        html = render_replay_html(self._make_payload_json(), title="Custom Title")
        assert "Custom Title" in html

    def test_payload_json_embeddable(self):
        """The payload JSON should parse correctly when extracted."""
        pj = self._make_payload_json()
        data = json.loads(pj)
        assert data["trade_id"] == 1
        assert data["summary"]["entry_price"] == 71044.18


# ── TradingView Launcher Tests ────────────────────────────────────────


class TestTradingViewLauncher:
    SAMPLE_PINE = '//@version=6\nindicator("test", overlay=true)\nplot(close)'

    def test_launcher_button_present(self):
        html = render_replay_html("{}", pine_script=self.SAMPLE_PINE)
        assert "tvLaunchBtn" in html
        assert "View on TradingView" in html

    def test_copy_button_present(self):
        html = render_replay_html("{}", pine_script=self.SAMPLE_PINE)
        assert "tvCopyBtn" in html
        assert "Copy Pine" in html

    def test_buttons_hidden_without_pine(self):
        html = render_replay_html("{}", pine_script="")
        assert "HAS_PINE = false" in html

    def test_buttons_shown_with_pine(self):
        html = render_replay_html("{}", pine_script=self.SAMPLE_PINE)
        assert "HAS_PINE = true" in html

    def test_pine_embedded_in_js(self):
        html = render_replay_html("{}", pine_script=self.SAMPLE_PINE)
        assert "PINE_SCRIPT" in html
        assert "@version=6" in html

    def test_overlay_exists(self):
        html = render_replay_html("{}", pine_script=self.SAMPLE_PINE)
        assert "tvOverlay" in html
        assert "TradingView Ready" in html
        assert "Got it" in html

    def test_fallback_textarea_exists(self):
        html = render_replay_html("{}", pine_script=self.SAMPLE_PINE)
        assert "fallbackTa" in html
        assert "fallback-ta" in html

    def test_clipboard_api_used(self):
        html = render_replay_html("{}", pine_script=self.SAMPLE_PINE)
        assert "navigator.clipboard.writeText" in html

    def test_tradingview_link(self):
        html = render_replay_html("{}", pine_script=self.SAMPLE_PINE)
        assert "https://www.tradingview.com/chart/" in html

    def test_escapes_backticks(self):
        pine = 'label.new(x, y, "`value`")'
        html = render_replay_html("{}", pine_script=pine)
        assert "\\`value\\`" in html

    def test_escapes_template_literals(self):
        pine = 'str.tostring(x, "${val}")'
        html = render_replay_html("{}", pine_script=pine)
        assert "\\${val}" in html

    def test_escapes_script_tag(self):
        pine = '// </script><script>alert(1)</script>'
        html = render_replay_html("{}", pine_script=pine)
        # Should not contain raw closing script tag inside JS
        assert "</script><script>" not in html

    def test_html_escapes_in_textarea(self):
        pine = '// <b>bold</b> & "quotes"'
        html = render_replay_html("{}", pine_script=pine)
        # The fallback textarea should have HTML-escaped content
        assert "&lt;b&gt;" in html
        assert "&amp;" in html

    def test_large_pine_script(self):
        """Large Pine scripts should not break the template."""
        pine = '//@version=6\n' + '\n'.join(
            f'var float x{i} = {i}.0' for i in range(500)
        )
        html = render_replay_html("{}", pine_script=pine)
        assert "x499" in html
        assert "HAS_PINE = true" in html

    def test_pine_excluded_from_json(self):
        """pine_script should NOT appear in the DATA JSON blob."""
        payload = ReplayPayload(
            trade_id=1, generated_at="2026-01-01",
            summary=ReplaySummary(
                trade_id=1, symbol="X", direction="bullish", side="long",
                timeframe="1h", model="M", entry_price=1, stop_price=0.9,
                target_price=1.1, tp1_price=None, rr=None, entry_score=None,
                opened_at="", closed_at=None, exit_price=None, exit_reason=None,
                pnl_pct=None, pnl_dollars=None, is_win=None, htf_bias=None,
                schematic_completeness=None, schematic_source=None,
                schematic_version=None,
            ),
            pine_script="should_not_be_in_json",
        )
        j = payload.to_json()
        data = json.loads(j)
        assert "pine_script" not in data
        assert "should_not_be_in_json" not in j


# ── Pine Generation in Builder ────────────────────────────────────────


class TestBuilderPineGeneration:
    def test_payload_has_pine_script(self, sample_trade, sample_schematic):
        builder = ReplayBuilder()
        payload = builder.build(
            trade_id=1, trade=sample_trade,
            confirmed=sample_schematic, candles_df=None,
        )
        assert payload.pine_script != ""
        assert "//@version=6" in payload.pine_script
        assert "71044.18" in payload.pine_script  # entry price

    def test_payload_pine_without_schematic(self, sample_trade):
        builder = ReplayBuilder()
        payload = builder.build(
            trade_id=1, trade=sample_trade,
            confirmed=None, candles_df=None,
        )
        assert "//@version=6" in payload.pine_script


# ── File Output Tests ─────────────────────────────────────────────────


class TestReplayFileOutput:
    def test_creates_output_files(self, tmp_path, sample_trade, sample_schematic):
        builder = ReplayBuilder()
        payload = builder.build(
            trade_id=1, trade=sample_trade,
            confirmed=sample_schematic, candles_df=_make_candles_df(),
        )
        out_dir = tmp_path / "trade_1"
        out_dir.mkdir()

        title = "Test Replay"
        html = render_replay_html(payload.to_json(), title=title)
        (out_dir / "replay.html").write_text(html, encoding="utf-8")
        (out_dir / "replay_data.json").write_text(payload.to_json(), encoding="utf-8")

        assert (out_dir / "replay.html").exists()
        assert (out_dir / "replay_data.json").exists()
        html_content = (out_dir / "replay.html").read_text()
        assert "BTCUSDT" in html_content

    def test_json_debug_file_valid(self, tmp_path, sample_trade):
        builder = ReplayBuilder()
        payload = builder.build(
            trade_id=1, trade=sample_trade,
            confirmed=None, candles_df=None,
        )
        json_path = tmp_path / "replay_data.json"
        json_path.write_text(payload.to_json())
        data = json.loads(json_path.read_text())
        assert data["trade_id"] == 1
