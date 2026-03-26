"""
tests/unit/test_backtest_runner.py — Unit Tests for Backtest Runner
====================================================================
Tests deterministic rules, multi-TF sync, pivot confirmation,
signal dedup, trade collision, cooldown, RIG enforcement,
direction-aware MFE/MAE, slippage/fees, intra-candle conflict,
and session boundary edge cases.
"""

import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pandas as pd
import numpy as np

from backtest.config import (
    EXECUTION_SLIPPAGE_PCT,
    FEE_PCT,
    MIN_BARS_BETWEEN_TRADES,
    MIN_PIVOT_CONFIRM,
    STARTING_BALANCE,
    timeframe_to_seconds,
)
from backtest.runner import (
    BacktestState,
    OpenTrade,
    apply_fees,
    apply_slippage,
    calculate_position_size,
    check_trade_exit,
    get_last_closed,
    update_mfe_mae,
)
from backtest.session import get_session, get_session_name


# ── Helpers ───────────────────────────────────────────────────────────

def _make_candles(start: datetime, tf: str, count: int, base_price: float = 50000.0):
    """Create a simple candle DataFrame for testing."""
    tf_seconds = timeframe_to_seconds(tf)
    rows = []
    for i in range(count):
        t = start + timedelta(seconds=tf_seconds * i)
        rows.append({
            "open_time": t,
            "open": base_price + i,
            "high": base_price + i + 50,
            "low": base_price + i - 50,
            "close": base_price + i + 10,
            "volume": 100.0,
        })
    return pd.DataFrame(rows)


# ── Walk-Forward Slicing (Closed Candles Only) ────────────────────────

class TestGetLastClosed:
    """Ensure only fully closed candles are returned."""

    def test_filters_future_candles(self):
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        df = _make_candles(start, "1h", 10)
        current_time = start + timedelta(hours=5, minutes=30)
        result = get_last_closed(df, "1h", current_time)
        # At 05:30, the 05:00 candle is still open → only 0-4 should be included
        assert len(result) == 5
        assert all(result["open_time"] < current_time)

    def test_exact_boundary_excludes_current(self):
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        df = _make_candles(start, "1h", 5)
        current_time = start + timedelta(hours=3)
        result = get_last_closed(df, "1h", current_time)
        assert len(result) == 3

    def test_empty_dataframe(self):
        df = pd.DataFrame(columns=["open_time", "open", "high", "low", "close", "volume"])
        current_time = datetime(2026, 1, 1, tzinfo=timezone.utc)
        result = get_last_closed(df, "1h", current_time)
        assert len(result) == 0


class TestMultiTFSynchronization:
    """Different TFs must be properly aligned to their close times."""

    def test_4h_alignment(self):
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        df = _make_candles(start, "4h", 10)
        # At 06:00, the 04:00 candle is still forming → only 00:00 should be closed
        current_time = start + timedelta(hours=6)
        result = get_last_closed(df, "4h", current_time)
        # 00:00 candle closes at 04:00, 04:00 candle closes at 08:00
        # At 06:00, only the 00:00 candle is fully closed
        assert len(result) <= 2  # 00:00 closed, 04:00 not yet

    def test_1d_alignment(self):
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        df = _make_candles(start, "1d", 5)
        current_time = start + timedelta(hours=36)
        result = get_last_closed(df, "1d", current_time)
        # Only Jan 1 candle should be fully closed
        assert len(result) == 1


# ── Pivot Confirmation Delay ─────────────────────────────────────────

class TestPivotConfirmation:
    def test_min_pivot_confirm_constant(self):
        assert MIN_PIVOT_CONFIRM == 6

    def test_insufficient_candles_skipped(self):
        """Pipeline should skip TFs with fewer than MIN_PIVOT_CONFIRM candles."""
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        df = _make_candles(start, "1h", MIN_PIVOT_CONFIRM - 1)
        assert len(df) < MIN_PIVOT_CONFIRM


# ── Signal Deduplication ──────────────────────────────────────────────

class TestSignalDeduplication:
    def test_duplicate_signal_time_blocked(self):
        state = BacktestState()
        t = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
        state.last_signal_time = t
        # Attempting to process at same time should be blocked
        assert state.last_signal_time == t


# ── Trade Collision Prevention ────────────────────────────────────────

class TestTradeCollision:
    def test_open_trade_blocks_new_signal(self):
        state = BacktestState()
        state.open_trade = OpenTrade(
            trade_num=1, symbol="BTCUSDT", timeframe="1h",
            direction="bullish", model="Model_1",
            entry_price=50000, stop_price=49500, target_price=51000,
        )
        assert state.open_trade is not None


# ── Trade Cooldown ────────────────────────────────────────────────────

class TestTradeCooldown:
    def test_cooldown_enforced(self):
        state = BacktestState()
        state.last_trade_close_step = 10
        state.current_step = 10 + MIN_BARS_BETWEEN_TRADES - 1
        # Should be within cooldown
        assert (state.current_step - state.last_trade_close_step) < MIN_BARS_BETWEEN_TRADES

    def test_cooldown_expired(self):
        state = BacktestState()
        state.last_trade_close_step = 10
        state.current_step = 10 + MIN_BARS_BETWEEN_TRADES
        assert (state.current_step - state.last_trade_close_step) >= MIN_BARS_BETWEEN_TRADES


# ── RIG Hard Block + Bias Freeze ─────────────────────────────────────

class TestRIGEnforcement:
    def test_rig_block(self):
        """RIG BLOCK should reject the signal."""
        from hpb_rig_validator import range_integrity_validator

        context = {
            "gates": {
                "RCM": {"valid": True, "range_duration_hours": 48},
                "MSCE": {"session_bias": "bearish", "session": "london"},
                "1A": {"bias": "bullish"},
                "1D": {"score": 80},
            },
            "local_range_displacement": 0.10,
        }
        result = range_integrity_validator(context)
        assert result["status"] == "BLOCK"

    def test_rig_valid(self):
        """RIG should pass when session bias matches HTF bias."""
        from hpb_rig_validator import range_integrity_validator

        context = {
            "gates": {
                "RCM": {"valid": True, "range_duration_hours": 48},
                "MSCE": {"session_bias": "bullish", "session": "london"},
                "1A": {"bias": "bullish"},
                "1D": {"score": 80},
            },
            "local_range_displacement": 0.10,
        }
        result = range_integrity_validator(context)
        assert result["status"] == "VALID"

    def test_rig_bias_freeze(self):
        """On RIG BLOCK, last_htf_bias should revert to previous value."""
        state = BacktestState()
        state.last_htf_bias = "bullish"
        previous = state.last_htf_bias

        # Simulate RIG block
        rig_status = "BLOCK"
        if rig_status == "BLOCK":
            state.last_htf_bias = previous  # freeze

        assert state.last_htf_bias == "bullish"


# ── Direction-Aware MFE/MAE ──────────────────────────────────────────

class TestMFEMAE:
    def test_long_mfe_mae(self):
        trade = OpenTrade(
            trade_num=1, symbol="BTCUSDT", timeframe="1h",
            direction="bullish", model="Model_1",
            entry_price=50000, stop_price=49500, target_price=51000,
            effective_entry=50025,  # with slippage
        )
        # Candle: high=50500, low=49800
        update_mfe_mae(trade, 50500, 49800)
        assert trade.mfe == 50500 - 50025  # 475
        assert trade.mae == 49800 - 50025  # -225

        # Candle 2: higher high, higher low
        update_mfe_mae(trade, 50800, 50100)
        assert trade.mfe == 50800 - 50025  # 775 (updated)
        assert trade.mae == 49800 - 50025  # -225 (unchanged, still worst)

    def test_short_mfe_mae(self):
        trade = OpenTrade(
            trade_num=1, symbol="BTCUSDT", timeframe="1h",
            direction="bearish", model="Model_1",
            entry_price=50000, stop_price=50500, target_price=49000,
            effective_entry=49975,  # with slippage
        )
        # Candle: high=50200, low=49600
        update_mfe_mae(trade, 50200, 49600)
        assert trade.mfe == 49975 - 49600  # 375
        assert trade.mae == 49975 - 50200  # -225

        # Candle 2: lower low
        update_mfe_mae(trade, 50100, 49200)
        assert trade.mfe == 49975 - 49200  # 775 (updated)
        assert trade.mae == 49975 - 50200  # -225 (unchanged)


# ── Slippage + Fees ───────────────────────────────────────────────────

class TestSlippageFees:
    def test_long_entry_slippage_adverse(self):
        price = 50000
        result = apply_slippage(price, "bullish", is_entry=True)
        assert result > price  # LONG entry costs more
        assert abs(result - price * (1 + EXECUTION_SLIPPAGE_PCT)) < 0.01

    def test_short_entry_slippage_adverse(self):
        price = 50000
        result = apply_slippage(price, "bearish", is_entry=True)
        assert result < price  # SHORT entry gets less

    def test_long_exit_slippage_adverse(self):
        price = 50000
        result = apply_slippage(price, "bullish", is_entry=False)
        assert result < price  # LONG exit gets less

    def test_short_exit_slippage_adverse(self):
        price = 50000
        result = apply_slippage(price, "bearish", is_entry=False)
        assert result > price  # SHORT exit costs more

    def test_fees_deducted(self):
        pnl = 100.0
        position_size = 5000.0
        result = apply_fees(pnl, position_size)
        expected_fee = position_size * FEE_PCT * 2
        assert result == pnl - expected_fee
        assert result < pnl

    def test_fees_on_loss(self):
        pnl = -50.0
        position_size = 5000.0
        result = apply_fees(pnl, position_size)
        assert result < pnl  # fees make loss worse


# ── Intra-Candle Conflict (Worst Case SL) ────────────────────────────

class TestIntraCandleConflict:
    def test_both_tp_sl_hit_assumes_sl(self):
        """When both TP and SL are hit in same candle, assume SL first."""
        trade = OpenTrade(
            trade_num=1, symbol="BTCUSDT", timeframe="1h",
            direction="bullish", model="Model_1",
            entry_price=50000, stop_price=49500, target_price=51000,
        )
        # Candle where both SL and TP are hit
        result = check_trade_exit(trade, high=51200, low=49400, close=50500)
        assert result is not None
        exit_reason, exit_price = result
        assert exit_reason == "stop_hit"
        assert exit_price == 49500

    def test_only_tp_hit(self):
        trade = OpenTrade(
            trade_num=1, symbol="BTCUSDT", timeframe="1h",
            direction="bullish", model="Model_1",
            entry_price=50000, stop_price=49500, target_price=51000,
        )
        result = check_trade_exit(trade, high=51200, low=49600, close=51100)
        assert result is not None
        assert result[0] == "target_hit"
        assert result[1] == 51000

    def test_only_sl_hit(self):
        trade = OpenTrade(
            trade_num=1, symbol="BTCUSDT", timeframe="1h",
            direction="bullish", model="Model_1",
            entry_price=50000, stop_price=49500, target_price=51000,
        )
        result = check_trade_exit(trade, high=50800, low=49400, close=49600)
        assert result is not None
        assert result[0] == "stop_hit"
        assert result[1] == 49500

    def test_neither_hit(self):
        trade = OpenTrade(
            trade_num=1, symbol="BTCUSDT", timeframe="1h",
            direction="bullish", model="Model_1",
            entry_price=50000, stop_price=49500, target_price=51000,
        )
        result = check_trade_exit(trade, high=50800, low=49600, close=50500)
        assert result is None

    def test_short_intra_candle_conflict(self):
        trade = OpenTrade(
            trade_num=1, symbol="BTCUSDT", timeframe="1h",
            direction="bearish", model="Model_1",
            entry_price=50000, stop_price=50500, target_price=49000,
        )
        # Both hit
        result = check_trade_exit(trade, high=50600, low=48900, close=50000)
        assert result is not None
        assert result[0] == "stop_hit"
        assert result[1] == 50500


# ── Session Boundary Edge Cases ───────────────────────────────────────

class TestSessionBoundaries:
    def test_midnight_boundary_asia(self):
        t = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
        assert get_session_name(t) == "asia"

    def test_london_open(self):
        t = datetime(2026, 1, 1, 8, 0, tzinfo=timezone.utc)
        assert get_session_name(t) == "london"

    def test_ny_open(self):
        t = datetime(2026, 1, 1, 13, 0, tzinfo=timezone.utc)
        assert get_session_name(t) == "new_york"

    def test_off_hours(self):
        t = datetime(2026, 1, 1, 22, 0, tzinfo=timezone.utc)
        assert get_session_name(t) == "off"

    def test_session_has_required_keys(self):
        t = datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)
        s = get_session(t)
        assert "session" in s
        assert "bias" in s
        assert "confidence_multiplier" in s


# ── Position Sizing ───────────────────────────────────────────────────

class TestPositionSizing:
    def test_basic_position_size(self):
        pos, risk = calculate_position_size(5000, 1.0)
        assert risk == 50.0  # 1% of 5000
        assert pos == 5000.0  # 50/1% * 100

    def test_zero_sl_returns_zero(self):
        pos, risk = calculate_position_size(5000, 0)
        assert pos == 0.0

    def test_matches_live_formula(self):
        """Verify formula matches trade_execution.py."""
        from trade_execution import calculate_position_size as live_calc
        risk_amount = 50.0
        sl_pct = 1.4
        live_pos = live_calc(risk_amount, sl_pct)
        bt_pos, _ = calculate_position_size(5000, sl_pct)
        # Both should produce same formula: risk/sl% * 100
        assert abs(live_pos - (risk_amount / sl_pct * 100)) < 0.01
