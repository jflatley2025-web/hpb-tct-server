"""
Unit tests for tensor_tct_trader.py
====================================
Covers all 15 approved fixes from the BIG CHANGE review session:

Issue 1  — Stale BOS gate (bos_idx recency check)
Issue 2  — Live R:R calculation at current market price
Issue 3  — HTF neutral bias blocks entry
Issue 4  — No BOS fallback (tested via tct_schematics; see test_bos_no_fallback)
Issue 5  — consecutive_losses derived from trade history (survives restarts)
Issue 6  — Dead code deleted (compute_reward_bias, HPB_TensorTrade_Env)
Issue 7  — fetch_error flag set on schematic eval exceptions
Issue 8  — quality_score gate (floor 0.70) + bonus points
Issue 9  — Trade lifecycle: enter, win, loss, duplicate suppression
Issue 10 — Parametrized pass/fail pairs for every gate
Issue 11 — TradeState persistence (save/load round-trip, model weights)
Issue 12 — BOS no-fallback contract (via tct_schematics._find_bullish_bos)
"""
import json
import math
import time
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_schematic(
    direction="bullish",
    model="Model_1_Accumulation",
    is_confirmed=True,
    bos_idx=195,
    quality_score=0.80,
    stop_price=95_000.0,
    target_price=105_000.0,
    tap3_type="tap3_model1",
    is_higher_low=True,
    is_lower_high=True,
):
    """Build a minimal schematic dict for evaluator tests."""
    return {
        "direction": direction,
        "model": model,
        "schematic_type": model,
        "is_confirmed": is_confirmed,
        "quality_score": quality_score,
        "bos_confirmation": {"bos_idx": bos_idx, "confirmed": True, "bos_price": 98_000.0},
        "stop_loss": {"price": stop_price},
        "target": {"price": target_price},
        "entry": {"price": 98_000.0},
        "tap3": {
            "type": tap3_type,
            "is_higher_low": is_higher_low,
            "is_lower_high": is_lower_high,
            "price": stop_price + 100,
        },
        "risk_reward": 2.5,
    }


def _make_closed_trade(is_win: bool, entry_price: float = 100_000.0,
                       direction: str = "bullish", model: str = "Model_1_Accumulation",
                       balance_after: float = 5_000.0) -> dict:
    return {
        "id": 1,
        "direction": direction,
        "model": model,
        "entry_price": entry_price,
        "exit_price": entry_price + (500 if is_win else -500),
        "is_win": is_win,
        "status": "closed",
        "pnl_pct": 0.5 if is_win else -0.5,
        "pnl_dollars": 25.0 if is_win else -25.0,
        "balance_after": balance_after,
        "closed_at": datetime.now(timezone.utc).isoformat(),
        "rr": 2.0,
        "entry_score": 65,
        "htf_bias": "bullish",
    }


# ---------------------------------------------------------------------------
# Issue 6: dead-code deletion check
# ---------------------------------------------------------------------------

class TestDeadCodeDeleted:
    """Issue 6 — compute_reward_bias and HPB_TensorTrade_Env must not exist."""

    def test_compute_reward_bias_removed(self):
        from tensor_tct_trader import TCTTradeEvaluator
        assert not hasattr(TCTTradeEvaluator, "compute_reward_bias"), (
            "compute_reward_bias was supposed to be deleted (Issue 6)"
        )

    def test_hpb_tensor_env_not_instantiated(self):
        """self.env should not exist on the evaluator after Issue 6."""
        from tensor_tct_trader import TCTTradeEvaluator
        ev = TCTTradeEvaluator()
        assert not hasattr(ev, "env"), (
            "TCTTradeEvaluator.env (HPB_TensorTrade_Env) should have been removed (Issue 6)"
        )


# ---------------------------------------------------------------------------
# Issue 5: consecutive_losses derived from trade history
# ---------------------------------------------------------------------------

class TestDeriveConsecutiveLosses:
    """Issue 5 — consecutive_losses is derived from tail of trade history."""

    def _make_state_with_history(self, tmp_path, trades):
        from tensor_tct_trader import TradeState, STARTING_BALANCE
        log = tmp_path / "tl.json"
        data = {
            "balance": STARTING_BALANCE,
            "starting_balance": STARTING_BALANCE,
            "current_trade": None,
            "trade_history": trades,
            "reward_history": [],
            "total_wins": sum(1 for t in trades if t.get("is_win")),
            "total_losses": sum(1 for t in trades if not t.get("is_win") and t.get("status") == "closed"),
            "solutions_applied": [],
            "last_scan_time": None,
            "last_scan_action": None,
            "last_error": None,
        }
        log.write_text(json.dumps(data))
        with patch("tensor_tct_trader.TRADE_LOG_PATH", str(log)), \
             patch("github_storage.fetch_trade_log", return_value=None):
            state = TradeState()
        return state

    def test_three_consecutive_losses_at_tail(self, tmp_path):
        # Use unique entry prices so _deduplicate_history doesn't collapse them
        trades = [
            _make_closed_trade(is_win=True,  entry_price=100_000.0),
            _make_closed_trade(is_win=False, entry_price=100_100.0),
            _make_closed_trade(is_win=False, entry_price=100_200.0),
            _make_closed_trade(is_win=False, entry_price=100_300.0),
        ]
        state = self._make_state_with_history(tmp_path, trades)
        assert state.derive_consecutive_losses() == 3

    def test_win_resets_streak(self, tmp_path):
        trades = [
            _make_closed_trade(is_win=False, entry_price=100_000.0),
            _make_closed_trade(is_win=False, entry_price=100_100.0),
            _make_closed_trade(is_win=True,  entry_price=100_200.0),
        ]
        state = self._make_state_with_history(tmp_path, trades)
        assert state.derive_consecutive_losses() == 0

    def test_empty_history_returns_zero(self, tmp_path):
        state = self._make_state_with_history(tmp_path, [])
        assert state.derive_consecutive_losses() == 0

    def test_single_loss_returns_one(self, tmp_path):
        trades = [_make_closed_trade(is_win=False, entry_price=100_000.0)]
        state = self._make_state_with_history(tmp_path, trades)
        assert state.derive_consecutive_losses() == 1

    def test_consecutive_losses_restored_on_trader_init(self, tmp_path):
        """Issue 5: TensorTCTTrader must set evaluator.consecutive_losses from history."""
        trades = [
            _make_closed_trade(is_win=False, entry_price=100_000.0),
            _make_closed_trade(is_win=False, entry_price=100_100.0),
        ]
        from tensor_tct_trader import TensorTCTTrader, STARTING_BALANCE
        log = tmp_path / "tl.json"
        data = {
            "balance": STARTING_BALANCE, "starting_balance": STARTING_BALANCE,
            "current_trade": None, "trade_history": trades, "reward_history": [],
            "total_wins": 0, "total_losses": 2, "solutions_applied": [],
            "last_scan_time": None, "last_scan_action": None, "last_error": None,
        }
        log.write_text(json.dumps(data))
        with patch("tensor_tct_trader.TRADE_LOG_PATH", str(log)), \
             patch("github_storage.fetch_trade_log", return_value=None):
            trader = TensorTCTTrader()
        assert trader.evaluator.consecutive_losses == 2


# ---------------------------------------------------------------------------
# Issue 11: TradeState persistence and compute_model_weights
# ---------------------------------------------------------------------------

class TestTradeStatePersistence:
    """Issue 11 — save/load round-trip and model weights formula."""

    def _make_and_save(self, tmp_path, extra_trades=None):
        from tensor_tct_trader import TradeState, STARTING_BALANCE
        log = tmp_path / "tl.json"
        trades = extra_trades or []
        data = {
            "balance": 5100.0,
            "starting_balance": STARTING_BALANCE,
            "current_trade": None,
            "trade_history": trades,
            "reward_history": [0.01, -0.005],
            "total_wins": sum(1 for t in trades if t.get("is_win")),
            "total_losses": sum(1 for t in trades if not t.get("is_win") and t.get("status") == "closed"),
            "solutions_applied": ["test solution"],
            "last_scan_time": "2026-02-19T00:00:00+00:00",
            "last_scan_action": "no_qualifying_setups",
            "last_error": None,
        }
        log.write_text(json.dumps(data))
        return log

    def test_save_load_roundtrip(self, tmp_path):
        from tensor_tct_trader import TradeState
        log = self._make_and_save(tmp_path)
        with patch("tensor_tct_trader.TRADE_LOG_PATH", str(log)), \
             patch("github_storage.fetch_trade_log", return_value=None):
            state = TradeState()
        assert state.balance == 5100.0
        assert state.last_scan_action == "no_qualifying_setups"

    def test_model_weights_empty_below_min_trades(self, tmp_path):
        from tensor_tct_trader import TradeState
        # Only 5 trades — below the 10-trade minimum
        trades = [_make_closed_trade(is_win=(i % 2 == 0)) for i in range(5)]
        log = self._make_and_save(tmp_path, extra_trades=trades)
        with patch("tensor_tct_trader.TRADE_LOG_PATH", str(log)), \
             patch("github_storage.fetch_trade_log", return_value=None):
            state = TradeState()
        weights = state.compute_model_weights(min_trades=10)
        assert weights == {}, "Should return {} when fewer than min_trades closed trades"

    def test_model_weights_correct_formula(self, tmp_path):
        from tensor_tct_trader import TradeState
        # 10 trades: 8 wins for Model_1, 2 losses
        # win_rate = 0.8 → bonus = round((0.8 - 0.5) * 20) = round(6.0) = 6
        trades = []
        for i in range(10):
            t = _make_closed_trade(is_win=(i < 8), model="Model_1_Accumulation",
                                   entry_price=100_000.0 + i * 100)
            t["id"] = i + 1
            trades.append(t)
        log = self._make_and_save(tmp_path, extra_trades=trades)
        with patch("tensor_tct_trader.TRADE_LOG_PATH", str(log)), \
             patch("github_storage.fetch_trade_log", return_value=None):
            state = TradeState()
        weights = state.compute_model_weights(min_trades=10)
        assert "Model_1_Accumulation" in weights
        assert weights["Model_1_Accumulation"] == 6

    def test_model_weights_negative_for_poor_performer(self, tmp_path):
        from tensor_tct_trader import TradeState
        # 10 trades: 2 wins → win_rate=0.2 → round((0.2-0.5)*20) = round(-6) = -6
        trades = []
        for i in range(10):
            t = _make_closed_trade(is_win=(i < 2), model="Model_2_Accumulation",
                                   entry_price=100_000.0 + i * 100)
            t["id"] = i + 1
            trades.append(t)
        log = self._make_and_save(tmp_path, extra_trades=trades)
        with patch("tensor_tct_trader.TRADE_LOG_PATH", str(log)), \
             patch("github_storage.fetch_trade_log", return_value=None):
            state = TradeState()
        weights = state.compute_model_weights(min_trades=10)
        assert weights.get("Model_2_Accumulation") == -6


# ---------------------------------------------------------------------------
# Issues 1, 2, 3, 8, 10: evaluate_schematic gate pass/fail pairs
# ---------------------------------------------------------------------------

class TestEvaluateSchematics:
    """Issue 10 — parametrized pass/fail pairs for every gate in evaluate_schematic."""

    @pytest.fixture(autouse=True)
    def evaluator(self):
        from tensor_tct_trader import TCTTradeEvaluator
        ev = TCTTradeEvaluator()
        ev.consecutive_losses = 0
        ev.model_weights = {}
        return ev

    # --- Issue 1: Stale BOS ---

    def test_stale_bos_rejected(self, evaluator):
        """bos_idx far from end of candle series → rejected."""
        s = _make_schematic(bos_idx=50)
        result = evaluator.evaluate_schematic(
            s, htf_bias="bullish", current_price=100_000.0,
            total_candles=200, max_stale_candles=5
        )
        assert result["pass"] is False
        assert any("Stale BOS" in r for r in result["reasons"])

    def test_fresh_bos_passes_staleness_gate(self, evaluator):
        """bos_idx within max_stale_candles of end → passes staleness gate."""
        s = _make_schematic(bos_idx=197)
        result = evaluator.evaluate_schematic(
            s, htf_bias="bullish", current_price=100_000.0,
            total_candles=200, max_stale_candles=5
        )
        # May still fail other gates, but not the staleness gate
        assert not any("Stale BOS" in r for r in result["reasons"])

    # --- Issue 2: Live R:R ---

    def test_live_rr_below_minimum_rejected(self, evaluator):
        """If current_price has moved past target reducing live R:R below 1.5 → rejected."""
        # entry zone around 98k, stop=95k, target=105k
        # If current_price = 104_000, live_reward = 1000, live_risk = 9000 → R:R ≈ 0.11
        s = _make_schematic(stop_price=95_000.0, target_price=105_000.0, bos_idx=197)
        result = evaluator.evaluate_schematic(
            s, htf_bias="bullish", current_price=104_000.0,
            total_candles=200, max_stale_candles=5
        )
        assert result["pass"] is False
        assert any("R:R" in r for r in result["reasons"])

    def test_live_rr_sufficient_passes(self, evaluator):
        """current_price close to ideal entry → live R:R is healthy → passes R:R gate."""
        # stop=95k, target=110k, current=100k → risk=5k, reward=10k → R:R=2.0
        s = _make_schematic(stop_price=95_000.0, target_price=110_000.0, bos_idx=197)
        result = evaluator.evaluate_schematic(
            s, htf_bias="bullish", current_price=100_000.0,
            total_candles=200, max_stale_candles=5
        )
        assert not any("R:R too low" in r for r in result["reasons"])

    # --- Issue 3: HTF neutral blocks entry ---

    def test_htf_neutral_blocks_entry(self, evaluator):
        """Issue 3: neutral HTF bias must reject the schematic."""
        # Use stop=95k, target=115k so live R:R=4.0 at entry=100k — reaches the HTF gate
        s = _make_schematic(bos_idx=197, stop_price=95_000.0, target_price=115_000.0)
        result = evaluator.evaluate_schematic(
            s, htf_bias="neutral", current_price=100_000.0,
            total_candles=200, max_stale_candles=5
        )
        assert result["pass"] is False
        assert any("HTF" in r and "neutral" in r.lower() for r in result["reasons"])

    def test_htf_aligned_bullish_passes(self, evaluator):
        """Aligned HTF bias (bullish schematic + bullish HTF) passes the HTF gate."""
        s = _make_schematic(direction="bullish", bos_idx=197,
                            stop_price=95_000.0, target_price=110_000.0)
        result = evaluator.evaluate_schematic(
            s, htf_bias="bullish", current_price=100_000.0,
            total_candles=200, max_stale_candles=5
        )
        assert any("HTF bias aligned" in r for r in result["reasons"])

    def test_htf_aligned_bearish_passes(self, evaluator):
        """Aligned HTF bias (bearish schematic + bearish HTF) passes the HTF gate."""
        # stop=105k (above entry), target=90k → risk=5k, reward=10k, R:R=2.0
        s = _make_schematic(direction="bearish", model="Model_1_Distribution",
                            bos_idx=197, stop_price=105_000.0, target_price=90_000.0,
                            tap3_type="tap3_model1")
        result = evaluator.evaluate_schematic(
            s, htf_bias="bearish", current_price=100_000.0,
            total_candles=200, max_stale_candles=5
        )
        assert any("HTF bias aligned" in r for r in result["reasons"])

    # --- Issue 8: quality_score gate + bonus ---

    def test_quality_below_floor_rejected(self, evaluator):
        """quality_score < 0.70 → rejected before scoring."""
        s = _make_schematic(quality_score=0.65, bos_idx=197)
        result = evaluator.evaluate_schematic(
            s, htf_bias="bullish", current_price=100_000.0,
            total_candles=200, max_stale_candles=5
        )
        assert result["pass"] is False
        assert any("quality" in r.lower() for r in result["reasons"])

    def test_quality_at_floor_passes_gate(self, evaluator):
        """quality_score == 0.70 → passes quality gate (not rejected)."""
        s = _make_schematic(quality_score=0.70, bos_idx=197,
                            stop_price=95_000.0, target_price=110_000.0)
        result = evaluator.evaluate_schematic(
            s, htf_bias="bullish", current_price=100_000.0,
            total_candles=200, max_stale_candles=5
        )
        assert not any("quality too low" in r.lower() for r in result["reasons"])

    def test_quality_bonus_applied_to_score(self, evaluator):
        """quality_score of 0.80 adds round(0.80 * 15) = 12 pts to score."""
        s = _make_schematic(quality_score=0.80, bos_idx=197,
                            stop_price=95_000.0, target_price=110_000.0)
        result = evaluator.evaluate_schematic(
            s, htf_bias="bullish", current_price=100_000.0,
            total_candles=200, max_stale_candles=5
        )
        expected_bonus = round(0.80 * 15)  # 12
        assert any(f"+{expected_bonus}" in r for r in result["reasons"]), (
            f"Expected quality bonus of +{expected_bonus} in reasons: {result['reasons']}"
        )

    def test_higher_quality_yields_higher_score(self, evaluator):
        """quality_score 0.95 must produce a higher final score than 0.75."""
        base_kwargs = dict(
            bos_idx=197, stop_price=95_000.0, target_price=110_000.0
        )
        s_high = _make_schematic(quality_score=0.95, **base_kwargs)
        s_low = _make_schematic(quality_score=0.75, **base_kwargs)
        r_high = evaluator.evaluate_schematic(
            s_high, "bullish", 100_000.0, total_candles=200, max_stale_candles=5
        )
        r_low = evaluator.evaluate_schematic(
            s_low, "bullish", 100_000.0, total_candles=200, max_stale_candles=5
        )
        assert r_high["score"] > r_low["score"]

    # --- Unconfirmed schematic always blocked ---

    def test_unconfirmed_schematic_blocked(self, evaluator):
        s = _make_schematic(is_confirmed=False, bos_idx=197)
        result = evaluator.evaluate_schematic(
            s, htf_bias="bullish", current_price=100_000.0,
            total_candles=200, max_stale_candles=5
        )
        assert result["pass"] is False
        assert result["reasons"] == ["No BOS confirmation"]

    # --- Required score is fixed (reward learning disabled) ---

    def test_required_score_fixed_regardless_of_loss_streak(self, evaluator):
        """Reward learning is disabled — required score is always 50, even after many losses."""
        evaluator.consecutive_losses = 3
        s = _make_schematic(quality_score=0.80, bos_idx=197,
                            stop_price=95_000.0, target_price=110_000.0)
        result = evaluator.evaluate_schematic(
            s, htf_bias="bullish", current_price=100_000.0,
            total_candles=200, max_stale_candles=5
        )
        assert result["required_score"] == 50, (
            "Reward learning is disabled — required_score must always be 50, not adaptive"
        )


# ---------------------------------------------------------------------------
# Issue 9: Trade lifecycle tests
# ---------------------------------------------------------------------------

class TestTradeLifecycle:
    """Issue 9 — enter, win, loss, balance arithmetic, duplicate suppression."""

    @pytest.fixture
    def trader(self, tmp_path):
        from tensor_tct_trader import TensorTCTTrader, STARTING_BALANCE
        log = tmp_path / "tl.json"
        data = {
            "balance": STARTING_BALANCE, "starting_balance": STARTING_BALANCE,
            "current_trade": None, "trade_history": [], "reward_history": [],
            "total_wins": 0, "total_losses": 0, "solutions_applied": [],
            "last_scan_time": None, "last_scan_action": None, "last_error": None,
        }
        log.write_text(json.dumps(data))
        with patch("tensor_tct_trader.TRADE_LOG_PATH", str(log)), \
             patch("tensor_tct_trader.TRADE_LOG_BACKUP_PATH", str(tmp_path / "backup.json")), \
             patch("github_storage.fetch_trade_log", return_value=None), \
             patch("tensor_tct_trader.notify_trade_entered"), \
             patch("tensor_tct_trader.notify_trade_closed"), \
             patch("tensor_tct_trader.notify_trade_force_closed"):
            t = TensorTCTTrader()
        return t

    def _open_trade(self, trader, entry=100_000.0, stop=95_000.0,
                    target=110_000.0, direction="bullish"):
        """Helper: directly open a simulated trade on the trader."""
        schematic = _make_schematic(
            direction=direction,
            stop_price=stop,
            target_price=target,
            bos_idx=197,
            quality_score=0.80,
        )
        evaluation = {
            "direction": direction,
            "model": "Model_1_Accumulation",
            "score": 70,
            "reasons": ["test"],
        }
        with patch("tensor_tct_trader.notify_trade_entered"):
            return trader._enter_trade(schematic, evaluation, entry, "bullish")

    def test_enter_trade_sets_current_trade(self, trader):
        self._open_trade(trader)
        assert trader.state.current_trade is not None
        assert trader.state.current_trade["status"] == "open"

    def test_win_increases_balance(self, trader):
        start_balance = trader.state.balance
        self._open_trade(trader, entry=100_000.0, stop=95_000.0, target=110_000.0)
        with patch("tensor_tct_trader.notify_trade_closed"):
            trader._close_trade(110_000.0, "target_hit")
        assert trader.state.balance > start_balance
        assert trader.state.total_wins == 1
        assert trader.state.total_losses == 0
        assert trader.state.current_trade is None

    def test_loss_decreases_balance(self, trader):
        start_balance = trader.state.balance
        self._open_trade(trader, entry=100_000.0, stop=95_000.0, target=110_000.0)
        with patch("tensor_tct_trader.notify_trade_closed"):
            trader._close_trade(95_000.0, "stop_hit")
        assert trader.state.balance < start_balance
        assert trader.state.total_wins == 0
        assert trader.state.total_losses == 1

    def test_is_win_set_correctly_on_close(self, trader):
        self._open_trade(trader)
        with patch("tensor_tct_trader.notify_trade_closed"):
            result = trader._close_trade(110_000.0, "target_hit")
        assert result["trade"]["is_win"] is True

    def test_enter_trade_rejected_target_below_entry_long(self, trader):
        """Long trade where target < entry must be rejected."""
        schematic = _make_schematic(
            direction="bullish", stop_price=95_000.0,
            target_price=99_000.0, bos_idx=197
        )
        evaluation = {"direction": "bullish", "model": "Model_1_Accumulation",
                      "score": 70, "reasons": []}
        result = trader._enter_trade(schematic, evaluation, 100_000.0, "bullish")
        assert "error" in result
        assert "target" in result["error"].lower()

    def test_enter_trade_rejected_stop_above_entry_long(self, trader):
        """Long trade where stop > entry must be rejected."""
        schematic = _make_schematic(
            direction="bullish", stop_price=101_000.0,
            target_price=110_000.0, bos_idx=197
        )
        evaluation = {"direction": "bullish", "model": "Model_1_Accumulation",
                      "score": 70, "reasons": []}
        result = trader._enter_trade(schematic, evaluation, 100_000.0, "bullish")
        assert "error" in result

    def test_enter_trade_rejected_rr_below_1(self, trader):
        """actual_rr < 1.0 at market price must be rejected."""
        # stop=99500, target=100100 at entry=100000 → risk=500, reward=100 → R:R=0.2
        schematic = _make_schematic(
            direction="bullish", stop_price=99_500.0,
            target_price=100_100.0, bos_idx=197
        )
        evaluation = {"direction": "bullish", "model": "Model_1_Accumulation",
                      "score": 70, "reasons": []}
        result = trader._enter_trade(schematic, evaluation, 100_000.0, "bullish")
        assert "error" in result
        assert "R:R" in result["error"]

    def test_enter_trade_rejected_rr_between_1_and_1_5(self, trader):
        """actual_rr >= 1.0 but < 1.5 must also be rejected (minimum is now 1.5:1)."""
        # stop=99000, target=101200 at entry=100000 → risk=1000, reward=1200 → R:R=1.2
        schematic = _make_schematic(
            direction="bullish", stop_price=99_000.0,
            target_price=101_200.0, bos_idx=197
        )
        evaluation = {"direction": "bullish", "model": "Model_1_Accumulation",
                      "score": 70, "reasons": []}
        result = trader._enter_trade(schematic, evaluation, 100_000.0, "bullish")
        assert "error" in result
        assert "1.5" in result["error"], "Error message must reference the 1.5 minimum"

    # --- Duplicate suppression ---

    def test_duplicate_setup_blocked_within_cooldown(self, trader):
        """Same entry price + direction within DUPLICATE_COOLDOWN_SECONDS → blocked."""
        from tensor_tct_trader import DUPLICATE_COOLDOWN_SECONDS
        # Inject a recently closed trade with matching price/direction
        trader.state.trade_history.append({
            **_make_closed_trade(is_win=False, entry_price=100_000.0),
            "direction": "bullish",
            "closed_at": datetime.now(timezone.utc).isoformat(),
        })
        # Same price within 0.1% tolerance
        assert trader._is_duplicate_setup(100_010.0, "bullish") is True

    def test_duplicate_setup_allowed_after_cooldown(self, trader):
        """Same entry price + direction AFTER cooldown expires → allowed."""
        from tensor_tct_trader import DUPLICATE_COOLDOWN_SECONDS
        stale_time = datetime.now(timezone.utc) - timedelta(seconds=DUPLICATE_COOLDOWN_SECONDS + 60)
        trader.state.trade_history.append({
            **_make_closed_trade(is_win=False, entry_price=100_000.0),
            "direction": "bullish",
            "closed_at": stale_time.isoformat(),
        })
        assert trader._is_duplicate_setup(100_010.0, "bullish") is False

    def test_different_direction_not_duplicate(self, trader):
        """Opposite direction at same price is not a duplicate."""
        trader.state.trade_history.append({
            **_make_closed_trade(is_win=False, entry_price=100_000.0),
            "direction": "bullish",
            "closed_at": datetime.now(timezone.utc).isoformat(),
        })
        assert trader._is_duplicate_setup(100_000.0, "bearish") is False


# ---------------------------------------------------------------------------
# Issue 7: fetch_error flag in exception handler
# ---------------------------------------------------------------------------

class TestFetchErrorFlag:
    """Issue 7 — runtime exceptions in schematic evaluation set fetch_error: True."""

    def test_exception_sets_fetch_error_true(self, tmp_path):
        """When detect_tct_schematics raises, fetch_error must be True so the
        all-MTF-failed guard in scan_and_trade fires correctly."""
        from tensor_tct_trader import TensorTCTTrader, STARTING_BALANCE
        log = tmp_path / "tl.json"
        data = {
            "balance": STARTING_BALANCE, "starting_balance": STARTING_BALANCE,
            "current_trade": None, "trade_history": [], "reward_history": [],
            "total_wins": 0, "total_losses": 0, "solutions_applied": [],
            "last_scan_time": None, "last_scan_action": None, "last_error": None,
        }
        log.write_text(json.dumps(data))

        import pandas as pd
        import numpy as np
        dummy_df = pd.DataFrame({
            "open_time": pd.date_range("2026-01-01", periods=60, freq="1h"),
            "open": np.ones(60) * 100_000,
            "high": np.ones(60) * 101_000,
            "low": np.ones(60) * 99_000,
            "close": np.ones(60) * 100_000,
            "volume": np.ones(60) * 500,
        })

        with patch("tensor_tct_trader.TRADE_LOG_PATH", str(log)), \
             patch("tensor_tct_trader.TRADE_LOG_BACKUP_PATH", str(tmp_path / "bak.json")), \
             patch("github_storage.fetch_trade_log", return_value=None), \
             patch("tensor_tct_trader.fetch_candles_sync", return_value=dummy_df), \
             patch("tensor_tct_trader.detect_tct_schematics",
                   side_effect=RuntimeError("boom")), \
             patch("tensor_tct_trader.notify_trade_entered"), \
             patch("tensor_tct_trader.notify_trade_closed"):
            trader = TensorTCTTrader()
            # Force cache expiry so HTF scan runs
            trader._htf_bias_expiry = 0.0
            result = trader.scan_and_trade()

        # The MTF timeframe entries that raised must have fetch_error: True
        tf_results = result.get("details", {}).get("timeframes_scanned", {})
        for tf, tf_data in tf_results.items():
            if tf_data.get("status") == "error":
                assert tf_data.get("fetch_error") is True, (
                    f"Issue 7: {tf} error result missing fetch_error=True"
                )


# ---------------------------------------------------------------------------
# Issue 12: BOS no-fallback contract (via tct_schematics)
# ---------------------------------------------------------------------------

class TestBosNoFallback:
    """Issue 12 — _find_bullish_bos returns None when all swing highs are above EQ."""

    def _make_candles(self, n=60, base=100_000.0):
        """Flat candle series — no meaningful swing highs below any reasonable EQ."""
        return pd.DataFrame({
            "open_time": pd.date_range("2026-01-01", periods=n, freq="1h"),
            "open":  np.full(n, base),
            "high":  np.full(n, base + 2_000),   # all highs well above any EQ we'd set
            "low":   np.full(n, base - 500),
            "close": np.full(n, base),
            "volume": np.full(n, 500.0),
        })

    def test_bullish_bos_returns_none_without_valid_swings(self):
        """When all swing highs are above equilibrium the fallback must not fire."""
        from tct_schematics import TCTSchematicDetector
        df = self._make_candles()
        detector = TCTSchematicDetector(df)

        # Set a very high equilibrium so every swing high is filtered out
        very_high_eq = 200_000.0
        result = detector._find_bullish_bos(
            start_idx=10,
            high_price=110_000.0,
            low_price=99_000.0,
            equilibrium=very_high_eq,
        )
        assert result is None, (
            "Issue 12: _find_bullish_bos must return None when no valid LTF swing highs "
            "pass the EQ filter — the last-resort fallback was removed (Issue 4)"
        )

    def test_bearish_bos_returns_none_without_valid_swings(self):
        """Mirror test for the bearish side."""
        from tct_schematics import TCTSchematicDetector
        df = self._make_candles()
        detector = TCTSchematicDetector(df)

        # Set a very low equilibrium so every swing low is filtered out
        very_low_eq = 0.0
        result = detector._find_bearish_bos(
            start_idx=10,
            low_price=90_000.0,
            high_price=101_000.0,
            equilibrium=very_low_eq,
        )
        assert result is None, (
            "Issue 12: _find_bearish_bos must return None when no valid LTF swing lows "
            "pass the EQ filter — the last-resort fallback was removed (Issue 4)"
        )


# ---------------------------------------------------------------------------
# Half Take Profit — entry, trigger, break-even stop, P&L accounting
# ---------------------------------------------------------------------------

class TestHalfTakeProfit:
    """Tests for the 1/2 take profit feature:
    - half_tp_price set at 49% of technical target distance on entry
    - half TP fires when price crosses that level
    - stop_price moved to entry (break even) after half TP
    - position_size halved after half TP
    - balance credited immediately; not double-counted on close
    - trade is_win=True when stopped at break even after half TP
    - bearish direction handled correctly
    """

    @pytest.fixture
    def trader(self, tmp_path):
        from tensor_tct_trader import TensorTCTTrader, STARTING_BALANCE
        log = tmp_path / "tl.json"
        data = {
            "balance": STARTING_BALANCE, "starting_balance": STARTING_BALANCE,
            "current_trade": None, "trade_history": [], "reward_history": [],
            "total_wins": 0, "total_losses": 0, "solutions_applied": [],
            "last_scan_time": None, "last_scan_action": None, "last_error": None,
        }
        log.write_text(json.dumps(data))
        with patch("tensor_tct_trader.TRADE_LOG_PATH", str(log)), \
             patch("tensor_tct_trader.TRADE_LOG_BACKUP_PATH", str(tmp_path / "backup.json")), \
             patch("github_storage.fetch_trade_log", return_value=None), \
             patch("tensor_tct_trader.notify_trade_entered"), \
             patch("tensor_tct_trader.notify_trade_closed"), \
             patch("tensor_tct_trader.notify_trade_force_closed"), \
             patch("tensor_tct_trader.notify_half_tp_taken"):
            t = TensorTCTTrader()
        return t

    def _open_trade(self, trader, entry=100_000.0, stop=95_000.0,
                    target=110_000.0, direction="bullish"):
        schematic = _make_schematic(
            direction=direction, stop_price=stop, target_price=target,
            bos_idx=197, quality_score=0.80,
        )
        evaluation = {
            "direction": direction, "model": "Model_1_Accumulation",
            "score": 70, "reasons": ["test"],
        }
        with patch("tensor_tct_trader.notify_trade_entered"):
            return trader._enter_trade(schematic, evaluation, entry, "bullish")

    def test_enter_trade_sets_half_tp_price_bullish(self, trader):
        """half_tp_price = entry + 49% of (target - entry) for bullish trades."""
        # entry=100k, target=110k → distance=10k → 49% = 4900 → half_tp=104900
        self._open_trade(trader, entry=100_000.0, stop=95_000.0, target=110_000.0)
        trade = trader.state.current_trade
        expected = round(100_000.0 + (110_000.0 - 100_000.0) * 0.49, 2)
        assert trade["half_tp_price"] == expected
        assert trade["half_tp_taken"] is False
        assert trade["half_tp_pnl_dollars"] == 0.0

    def test_enter_trade_sets_original_position_size(self, trader):
        """original_position_size must equal position_size at entry."""
        self._open_trade(trader)
        trade = trader.state.current_trade
        assert trade["original_position_size"] == trade["position_size"]

    def test_half_tp_not_triggered_below_level(self, trader):
        """No half TP event when price is between entry and half_tp_price."""
        self._open_trade(trader, entry=100_000.0, stop=95_000.0, target=110_000.0)
        # half_tp_price = 104900; 102000 is below it
        with patch("tensor_tct_trader.notify_half_tp_taken") as mock_notify:
            result = trader._manage_open_trade(102_000.0)
        assert result["action"] == "holding"
        assert trader.state.current_trade["half_tp_taken"] is False
        mock_notify.assert_not_called()

    def test_half_tp_triggered_at_level(self, trader):
        """half_tp_taken event fires when price reaches half_tp_price."""
        self._open_trade(trader, entry=100_000.0, stop=95_000.0, target=110_000.0)
        with patch("tensor_tct_trader.notify_half_tp_taken"):
            result = trader._manage_open_trade(104_900.0)
        assert result["action"] == "half_tp_taken"
        assert trader.state.current_trade["half_tp_taken"] is True

    def test_half_tp_moves_stop_to_break_even(self, trader):
        """After half TP, stop_price must equal entry_price."""
        self._open_trade(trader, entry=100_000.0, stop=95_000.0, target=110_000.0)
        with patch("tensor_tct_trader.notify_half_tp_taken"):
            trader._manage_open_trade(105_000.0)
        trade = trader.state.current_trade
        assert trade["stop_price"] == 100_000.0
        assert trade["stop_is_breakeven"] is True

    def test_half_tp_reduces_position_size_by_half(self, trader):
        """Position size must be halved after half TP is taken."""
        self._open_trade(trader, entry=100_000.0, stop=95_000.0, target=110_000.0)
        original_size = trader.state.current_trade["position_size"]
        with patch("tensor_tct_trader.notify_half_tp_taken"):
            trader._manage_open_trade(105_000.0)
        trade = trader.state.current_trade
        assert abs(trade["position_size"] - original_size / 2) < 0.01

    def test_half_tp_credits_balance_immediately(self, trader):
        """Balance must increase by half TP P&L as soon as it fires."""
        self._open_trade(trader, entry=100_000.0, stop=95_000.0, target=110_000.0)
        balance_before = trader.state.balance
        with patch("tensor_tct_trader.notify_half_tp_taken"):
            trader._manage_open_trade(105_000.0)
        assert trader.state.balance > balance_before

    def test_half_tp_sends_telegram_notification(self, trader):
        """notify_half_tp_taken must be called exactly once when half TP fires."""
        self._open_trade(trader, entry=100_000.0, stop=95_000.0, target=110_000.0)
        with patch("tensor_tct_trader.notify_half_tp_taken") as mock_notify:
            trader._manage_open_trade(105_000.0)
        mock_notify.assert_called_once()

    def test_half_tp_only_fires_once(self, trader):
        """Half TP must not fire again if price stays at the same level."""
        self._open_trade(trader, entry=100_000.0, stop=95_000.0, target=110_000.0)
        with patch("tensor_tct_trader.notify_half_tp_taken"):
            trader._manage_open_trade(105_000.0)  # first pass — fires
        balance_after_half_tp = trader.state.balance
        with patch("tensor_tct_trader.notify_half_tp_taken") as mock_notify:
            result = trader._manage_open_trade(105_000.0)  # second pass — must NOT re-fire
        assert mock_notify.call_count == 0
        assert result["action"] == "holding"
        assert trader.state.balance == balance_after_half_tp

    def test_close_at_target_after_half_tp_includes_combined_pnl(self, trader):
        """pnl_dollars on close reflects total P&L: half TP profit + remaining position profit."""
        self._open_trade(trader, entry=100_000.0, stop=95_000.0, target=110_000.0)
        with patch("tensor_tct_trader.notify_half_tp_taken"):
            trader._manage_open_trade(105_000.0)
        half_tp_pnl = trader.state.current_trade["half_tp_pnl_dollars"]
        with patch("tensor_tct_trader.notify_trade_closed"):
            result = trader._close_trade(110_000.0, "target_hit")
        closed = result["trade"]
        # Total P&L must exceed just the half TP portion
        assert closed["pnl_dollars"] > half_tp_pnl
        assert closed["is_win"] is True

    def test_balance_not_double_counted_on_close(self, trader):
        """Balance after half TP + close must equal half_pnl + remaining_pnl, not double the half."""
        self._open_trade(trader, entry=100_000.0, stop=95_000.0, target=110_000.0)
        start_balance = trader.state.balance
        original_size = trader.state.current_trade["position_size"]

        with patch("tensor_tct_trader.notify_half_tp_taken"):
            trader._manage_open_trade(105_000.0)
        balance_after_half_tp = trader.state.balance

        with patch("tensor_tct_trader.notify_trade_closed"):
            trader._close_trade(110_000.0, "target_hit")
        final_balance = trader.state.balance

        # Half TP P&L = (half_size) * pnl_pct_at_105k
        # Remaining P&L = (half_size) * pnl_pct_at_110k
        half_size = original_size / 2
        half_pnl = half_size * ((105_000.0 - 100_000.0) / 100_000.0)
        remaining_pnl = half_size * ((110_000.0 - 100_000.0) / 100_000.0)
        expected_final = start_balance + half_pnl + remaining_pnl

        assert abs(final_balance - expected_final) < 0.01

    def test_stop_at_break_even_after_half_tp_is_win(self, trader):
        """Stop hit at break even after half TP → is_win=True (positive total P&L)."""
        self._open_trade(trader, entry=100_000.0, stop=95_000.0, target=110_000.0)
        with patch("tensor_tct_trader.notify_half_tp_taken"):
            trader._manage_open_trade(105_000.0)
        with patch("tensor_tct_trader.notify_trade_closed"):
            result = trader._close_trade(100_000.0, "stop_hit")
        closed = result["trade"]
        # Half TP booked positive dollars; remaining closed at break even (0 P&L)
        assert closed["is_win"] is True
        assert closed["pnl_dollars"] > 0

    def test_stop_at_break_even_counts_as_total_win(self, trader):
        """total_wins increments when stopped at break even after half TP."""
        self._open_trade(trader, entry=100_000.0, stop=95_000.0, target=110_000.0)
        with patch("tensor_tct_trader.notify_half_tp_taken"):
            trader._manage_open_trade(105_000.0)
        with patch("tensor_tct_trader.notify_trade_closed"):
            trader._close_trade(100_000.0, "stop_hit")
        assert trader.state.total_wins == 1
        assert trader.state.total_losses == 0

    def test_bearish_half_tp_price_calculated_correctly(self, trader):
        """For bearish trades, half_tp_price = entry - 49% of (entry - target)."""
        # entry=100k, stop=105k, target=90k → distance=10k → 49%=4900 → half_tp=95100
        self._open_trade(trader, entry=100_000.0, stop=105_000.0,
                         target=90_000.0, direction="bearish")
        trade = trader.state.current_trade
        expected = round(100_000.0 - (100_000.0 - 90_000.0) * 0.49, 2)
        assert trade["half_tp_price"] == expected

    def test_bearish_half_tp_triggers_on_downward_move(self, trader):
        """For bearish trades, half TP fires when price drops to the half_tp level."""
        self._open_trade(trader, entry=100_000.0, stop=105_000.0,
                         target=90_000.0, direction="bearish")
        half_tp = trader.state.current_trade["half_tp_price"]
        with patch("tensor_tct_trader.notify_half_tp_taken"):
            result = trader._manage_open_trade(half_tp)
        assert result["action"] == "half_tp_taken"
        assert trader.state.current_trade["half_tp_taken"] is True

    def test_bearish_stop_moved_to_break_even(self, trader):
        """Bearish: stop_price moved to entry after half TP."""
        self._open_trade(trader, entry=100_000.0, stop=105_000.0,
                         target=90_000.0, direction="bearish")
        with patch("tensor_tct_trader.notify_half_tp_taken"):
            trader._manage_open_trade(95_000.0)
        assert trader.state.current_trade["stop_price"] == 100_000.0
