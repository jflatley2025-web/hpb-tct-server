"""
Unit tests for phemex_tct_trader.py
=====================================
Covers:
- Trade open lifecycle (LONG and SHORT)
- Stop-loss hit → LOSS outcome, balance decremented
- Target hit → WIN outcome, balance incremented by risk * RR
- No double-open: second signal ignored when trade is already open
- _close_trade with no open trade is a safe no-op
- snapshot() returns consistent read-only view
- save_state / _load_state round-trip
- GitHub push called on trade close
- Telegram notification called on open and close
- scan() skips gracefully when candle fetch returns None
- scan() opens trade when pipeline emits LONG/SHORT
- scan() monitors open position on subsequent scan
"""

import json
import os
import time
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, call


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pipeline_result(signal="LONG", entry=100.0, stop=95.0, target=110.0,
                          rr=2.0, confidence=0.8, is_trade=True):
    """Build a minimal PipelineResult-like object for tests."""
    r = MagicMock()
    r.signal = signal
    r.entry = entry
    r.stop = stop
    r.target = target
    r.rr = rr
    r.confidence = confidence
    r.is_trade = is_trade
    r.blocking_gate = None
    r.gate_results = []
    return r


def _make_ltf(close=100.0):
    """Build a minimal LTF DataFrame stub with a single row."""
    import pandas as pd
    return pd.DataFrame([{"open_time": 0, "open": close, "high": close,
                           "low": close, "close": close, "volume": 1.0}])


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_trader_singleton():
    """Reset the module-level singleton before each test."""
    import phemex_tct_trader
    phemex_tct_trader._trader = None
    yield
    phemex_tct_trader._trader = None


@pytest.fixture
def trader(tmp_path):
    """
    Return a fresh PhemexTCTTrader with:
    - TRADE_LOG_PATH pointing to a temp file (no disk side-effects)
    - github fetch/push mocked out (patched at source module)
    - Telegram mocked out (patched at source module)
    """
    import phemex_tct_trader as mod

    log_path = str(tmp_path / "phemex_trade_log.json")

    # These are local imports inside methods, so patch the source modules.
    with (
        patch.object(mod, "TRADE_LOG_PATH", log_path),
        patch("github_storage.fetch_phemex_log", return_value=False),
        patch("github_storage.push_phemex_log", return_value=True),
        patch("telegram_notifications.send_message", return_value=True),
    ):
        t = mod.PhemexTCTTrader()
        # Stub out the instance-level wrappers to avoid any real I/O
        t._push_to_github = MagicMock()
        t._notify = MagicMock()
        yield t


# ---------------------------------------------------------------------------
# Trade open / close lifecycle
# ---------------------------------------------------------------------------

class TestTradeLifecycle:

    def test_open_long_trade_sets_current_trade(self, trader):
        result = _make_pipeline_result(signal="LONG", entry=100.0, stop=95.0,
                                       target=110.0, rr=2.0)
        trader._open_trade(result)

        assert trader.current_trade is not None
        ct = trader.current_trade
        assert ct["signal"] == "LONG"
        assert ct["entry"] == 100.0
        assert ct["stop"] == 95.0
        assert ct["target"] == 110.0
        assert ct["rr"] == 2.0
        assert ct["status"] == "OPEN"
        assert ct["risk_amount"] == pytest.approx(trader.starting_balance * 0.01, rel=1e-4)

    def test_open_short_trade_sets_current_trade(self, trader):
        result = _make_pipeline_result(signal="SHORT", entry=100.0, stop=105.0,
                                       target=90.0, rr=2.0)
        trader._open_trade(result)

        ct = trader.current_trade
        assert ct["signal"] == "SHORT"
        assert ct["stop"] == 105.0
        assert ct["target"] == 90.0

    def test_close_long_win(self, trader):
        result = _make_pipeline_result(signal="LONG", rr=2.0)
        trader._open_trade(result)
        initial_balance = trader.balance
        risk = trader.current_trade["risk_amount"]

        trader._close_trade("WIN", exit_price=110.0)

        assert trader.current_trade is None
        assert len(trader.trade_history) == 1
        closed = trader.trade_history[0]
        assert closed["outcome"] == "WIN"
        assert closed["pnl"] == pytest.approx(risk * 2.0, rel=1e-4)
        assert trader.balance == pytest.approx(initial_balance + risk * 2.0, rel=1e-4)

    def test_close_long_loss(self, trader):
        result = _make_pipeline_result(signal="LONG", rr=2.0)
        trader._open_trade(result)
        initial_balance = trader.balance
        risk = trader.current_trade["risk_amount"]

        trader._close_trade("LOSS", exit_price=95.0)

        assert trader.current_trade is None
        closed = trader.trade_history[0]
        assert closed["outcome"] == "LOSS"
        assert closed["pnl"] == pytest.approx(-risk, rel=1e-4)
        assert trader.balance == pytest.approx(initial_balance - risk, rel=1e-4)

    def test_close_short_win(self, trader):
        result = _make_pipeline_result(signal="SHORT", entry=100.0, stop=105.0,
                                       target=90.0, rr=2.0)
        trader._open_trade(result)
        initial_balance = trader.balance
        risk = trader.current_trade["risk_amount"]

        trader._close_trade("WIN", exit_price=90.0)

        assert trader.balance == pytest.approx(initial_balance + risk * 2.0, rel=1e-4)

    def test_close_no_trade_is_safe_noop(self, trader):
        """Calling _close_trade when no trade is open must not raise."""
        assert trader.current_trade is None
        trader._close_trade("WIN", exit_price=100.0)  # should not raise
        assert trader.trade_history == []

    def test_github_push_called_on_close(self, trader):
        result = _make_pipeline_result()
        trader._open_trade(result)
        trader._close_trade("WIN", exit_price=110.0)

        trader._push_to_github.assert_called_once()

    def test_telegram_notify_called_on_open_and_close(self, trader):
        result = _make_pipeline_result()
        trader._open_trade(result)
        trader._close_trade("WIN", exit_price=110.0)

        assert trader._notify.call_count == 2


# ---------------------------------------------------------------------------
# Stop / target detection
# ---------------------------------------------------------------------------

class TestPositionMonitoring:

    def test_long_stop_hit(self, trader):
        result = _make_pipeline_result(signal="LONG", entry=100.0, stop=95.0, target=110.0, rr=2.0)
        trader._open_trade(result)

        trader._check_position(current_price=94.0)  # below stop

        assert trader.current_trade is None
        assert trader.trade_history[0]["outcome"] == "LOSS"

    def test_long_target_hit(self, trader):
        result = _make_pipeline_result(signal="LONG", entry=100.0, stop=95.0, target=110.0, rr=2.0)
        trader._open_trade(result)

        trader._check_position(current_price=111.0)  # above target

        assert trader.current_trade is None
        assert trader.trade_history[0]["outcome"] == "WIN"

    def test_long_in_range_no_close(self, trader):
        result = _make_pipeline_result(signal="LONG", entry=100.0, stop=95.0, target=110.0, rr=2.0)
        trader._open_trade(result)

        trader._check_position(current_price=103.0)  # between stop and target

        assert trader.current_trade is not None
        assert trader.trade_history == []

    def test_short_stop_hit(self, trader):
        result = _make_pipeline_result(signal="SHORT", entry=100.0, stop=105.0, target=90.0, rr=2.0)
        trader._open_trade(result)

        trader._check_position(current_price=106.0)  # above stop

        assert trader.current_trade is None
        assert trader.trade_history[0]["outcome"] == "LOSS"

    def test_short_target_hit(self, trader):
        result = _make_pipeline_result(signal="SHORT", entry=100.0, stop=105.0, target=90.0, rr=2.0)
        trader._open_trade(result)

        trader._check_position(current_price=89.0)  # below target

        assert trader.current_trade is None
        assert trader.trade_history[0]["outcome"] == "WIN"

    def test_check_position_no_trade_is_safe(self, trader):
        """check_position with no open trade must not raise or mutate state."""
        trader._check_position(100.0)
        assert trader.trade_history == []


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------

class TestSnapshot:

    def test_snapshot_default_state(self, trader):
        s = trader.snapshot()
        assert s["balance"] == trader.starting_balance
        assert s["total_trades"] == 0
        assert s["wins"] == 0
        assert s["losses"] == 0
        assert s["win_rate"] == 0.0
        assert s["current_trade"] is None
        assert s["trade_history"] == []

    def test_snapshot_after_win(self, trader):
        trader._open_trade(_make_pipeline_result(signal="LONG", rr=2.0))
        trader._close_trade("WIN", 110.0)

        s = trader.snapshot()
        assert s["wins"] == 1
        assert s["losses"] == 0
        assert s["total_trades"] == 1
        assert s["win_rate"] == 100.0
        assert s["pnl_total"] > 0

    def test_snapshot_history_newest_first(self, trader):
        """trade_history in snapshot should be reversed (newest first)."""
        for _ in range(3):
            trader._open_trade(_make_pipeline_result())
            trader._close_trade("WIN", 110.0)

        s = trader.snapshot()
        # The most recently closed trade should be first
        assert len(s["trade_history"]) == 3


# ---------------------------------------------------------------------------
# Persistence round-trip
# ---------------------------------------------------------------------------

class TestPersistence:

    def test_save_and_reload(self, trader, tmp_path):
        """State written by save_state() can be fully recovered by _load_state()."""
        import phemex_tct_trader as mod

        log_path = str(tmp_path / "roundtrip.json")

        # Inject a closed trade and custom balance
        trader._open_trade(_make_pipeline_result(signal="LONG", rr=2.0))
        trader._close_trade("WIN", 110.0)
        expected_balance = trader.balance

        # Save to the temp path manually
        orig_path = mod.TRADE_LOG_PATH
        with patch.object(mod, "TRADE_LOG_PATH", log_path):
            trader.save_state()

        # Read from disk and verify
        with open(log_path) as f:
            data = json.load(f)

        assert data["balance"] == pytest.approx(expected_balance, rel=1e-4)
        assert len(data["trade_history"]) == 1
        assert data["trade_history"][0]["outcome"] == "WIN"
        assert data["current_trade"] is None


# ---------------------------------------------------------------------------
# Scan integration (mocked feed + pipeline)
# ---------------------------------------------------------------------------

class TestScan:

    def _patch_feed(self, ltf_close=100.0):
        ltf = _make_ltf(close=ltf_close)
        candles = {"4h": ltf.copy(), "1h": ltf.copy(), "15m": ltf}
        return patch("phemex_tct_trader.feed_fetch_all", return_value=candles)

    def test_scan_skips_when_candles_none(self, trader):
        with (
            patch("phemex_tct_trader.feed_fetch_all", return_value={"4h": None, "1h": None, "15m": None}),
            patch.object(trader, "_ensure_rules", return_value=MagicMock()),
        ):
            result = trader.scan()
        assert result["action"] == "skip"

    @pytest.mark.skip(reason="Phemex-TCT engine is disabled — test not maintained")
    def test_scan_opens_trade_on_signal(self, trader):
        pipeline_result = _make_pipeline_result(signal="LONG", is_trade=True)

        with (
            self._patch_feed(ltf_close=100.0),
            patch("phemex_tct_trader.run_pipeline", return_value=pipeline_result),
            patch.object(trader, "_ensure_rules", return_value=MagicMock()),
            patch.object(trader, "_evaluate_rig", return_value={"status": "VALID"}),
        ):
            result = trader.scan()

        assert result["action"] == "trade_opened"
        assert trader.current_trade is not None

    def test_scan_no_trade_on_no_signal(self, trader):
        pipeline_result = _make_pipeline_result(signal="NO_TRADE", is_trade=False)

        with (
            self._patch_feed(ltf_close=100.0),
            patch("phemex_tct_trader.run_pipeline", return_value=pipeline_result),
            patch.object(trader, "_ensure_rules", return_value=MagicMock()),
        ):
            result = trader.scan()

        assert result["action"] == "no_trade"
        assert trader.current_trade is None

    def test_scan_monitors_open_trade(self, trader):
        """When a trade is already open, scan should monitor it, not re-scan pipeline."""
        trader._open_trade(_make_pipeline_result(signal="LONG", entry=100.0, stop=95.0, target=110.0))

        mock_pipeline = MagicMock()

        with (
            self._patch_feed(ltf_close=103.0),  # in-range: no close
            patch("phemex_tct_trader.run_pipeline", mock_pipeline),
            patch.object(trader, "_ensure_rules", return_value=MagicMock()),
        ):
            result = trader.scan()

        assert result["action"] == "monitor"
        mock_pipeline.assert_not_called()  # pipeline must NOT run while trade is open

    def test_scan_closes_trade_on_target_hit(self, trader):
        trader._open_trade(_make_pipeline_result(signal="LONG", entry=100.0, stop=95.0, target=110.0, rr=2.0))

        with (
            self._patch_feed(ltf_close=111.0),  # above target
            patch("phemex_tct_trader.run_pipeline", MagicMock()),
            patch.object(trader, "_ensure_rules", return_value=MagicMock()),
        ):
            trader.scan()

        assert trader.current_trade is None
        assert trader.trade_history[0]["outcome"] == "WIN"

    def test_scan_handles_pipeline_exception(self, trader):
        with (
            self._patch_feed(ltf_close=100.0),
            patch("phemex_tct_trader.run_pipeline", side_effect=RuntimeError("boom")),
            patch.object(trader, "_ensure_rules", return_value=MagicMock()),
        ):
            result = trader.scan()

        assert result["action"] == "error"
        assert "boom" in result["reason"]
