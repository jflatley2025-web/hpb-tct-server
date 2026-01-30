"""
Unit tests for trade_execution.py — TCT Lecture 9 Trade Execution Engine

Tests cover:
- Position sizing formula
- Margin calculation
- Liquidation price estimation (long/short)
- Liquidation safety checks
- Leverage comparison
- Partial take profits
- Trailing SL scenario
- Capital allocation
- Full execution plan generation
- Max safe leverage finder
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from trade_execution import (
    calculate_position_size,
    calculate_margin,
    calculate_liquidation_price,
    check_liquidation_safety,
    calculate_leverage_comparison,
    calculate_partial_take_profits,
    calculate_trailing_sl_scenario,
    calculate_capital_allocation,
    generate_execution_plan,
    find_max_safe_leverage,
)


# ===================== Position Sizing =====================

class TestPositionSizing:
    def test_basic_position_size(self):
        """$100 risk, 0.5% SL -> $20,000 position"""
        result = calculate_position_size(100, 0.5)
        assert result == 20000.0

    def test_small_sl_large_position(self):
        """$100 risk, 0.1% SL -> $100,000 position"""
        result = calculate_position_size(100, 0.1)
        assert result == 100000.0

    def test_large_sl_small_position(self):
        """$100 risk, 5% SL -> $2,000 position"""
        result = calculate_position_size(100, 5.0)
        assert result == 2000.0

    def test_zero_sl_raises(self):
        with pytest.raises(ValueError, match="greater than 0"):
            calculate_position_size(100, 0)

    def test_negative_sl_raises(self):
        with pytest.raises(ValueError, match="greater than 0"):
            calculate_position_size(100, -1)

    def test_tct_example(self):
        """TCT example: $10K balance, 1% risk = $100, 0.26% SL"""
        risk = 10000 * 0.01  # $100
        pos = calculate_position_size(risk, 0.26)
        assert round(pos, 2) == 38461.54


# ===================== Margin =====================

class TestMargin:
    def test_basic_margin(self):
        """$20K pos / 10x leverage = $2K margin"""
        result = calculate_margin(20000, 10)
        assert result == 2000.0

    def test_high_leverage(self):
        """$25K pos / 200x = $125 margin"""
        result = calculate_margin(25000, 200)
        assert result == 125.0

    def test_low_leverage(self):
        """$25K pos / 1x = $25K margin"""
        result = calculate_margin(25000, 1)
        assert result == 25000.0

    def test_zero_leverage_raises(self):
        with pytest.raises(ValueError, match="greater than 0"):
            calculate_margin(10000, 0)


# ===================== Liquidation Price =====================

class TestLiquidationPrice:
    def test_long_10x(self):
        """Long at $100K with 10x: liq ~$90,500"""
        liq = calculate_liquidation_price(100000, 10, "long")
        assert liq < 100000  # Must be below entry
        assert round(liq, 2) == 90500.0

    def test_short_10x(self):
        """Short at $100K with 10x: liq ~$109,500"""
        liq = calculate_liquidation_price(100000, 10, "short")
        assert liq > 100000  # Must be above entry
        assert round(liq, 2) == 109500.0

    def test_higher_leverage_closer_liquidation_long(self):
        """Higher leverage = liquidation closer to entry for longs"""
        liq_10x = calculate_liquidation_price(100000, 10, "long")
        liq_50x = calculate_liquidation_price(100000, 50, "long")
        assert liq_50x > liq_10x  # 50x liq is closer to entry

    def test_higher_leverage_closer_liquidation_short(self):
        """Higher leverage = liquidation closer to entry for shorts"""
        liq_10x = calculate_liquidation_price(100000, 10, "short")
        liq_50x = calculate_liquidation_price(100000, 50, "short")
        assert liq_50x < liq_10x  # 50x liq is closer to entry

    def test_invalid_direction(self):
        with pytest.raises(ValueError, match="long.*short"):
            calculate_liquidation_price(100000, 10, "sideways")

    def test_zero_leverage_raises(self):
        with pytest.raises(ValueError, match="greater than 0"):
            calculate_liquidation_price(100000, 0, "long")


# ===================== Liquidation Safety =====================

class TestLiquidationSafety:
    def test_safe_long(self):
        """Long: liq at $90K, SL at $99K -> safe"""
        result = check_liquidation_safety(90000, 99000, 100000, "long")
        assert result["is_safe"] is True
        assert result["gap"] == 9000.0

    def test_unsafe_long(self):
        """Long: liq at $99.5K, SL at $99K -> DANGER"""
        result = check_liquidation_safety(99500, 99000, 100000, "long")
        assert result["is_safe"] is False

    def test_safe_short(self):
        """Short: liq at $110K, SL at $101K -> safe"""
        result = check_liquidation_safety(110000, 101000, 100000, "short")
        assert result["is_safe"] is True
        assert result["gap"] == 9000.0

    def test_unsafe_short(self):
        """Short: liq at $100.5K, SL at $101K -> DANGER"""
        result = check_liquidation_safety(100500, 101000, 100000, "short")
        assert result["is_safe"] is False

    def test_large_buffer_recommendation(self):
        result = check_liquidation_safety(85000, 99000, 100000, "long")
        assert "large buffer" in result["recommendation"].lower()

    def test_danger_recommendation(self):
        result = check_liquidation_safety(99500, 99000, 100000, "long")
        assert "DANGER" in result["recommendation"]

    def test_invalid_direction(self):
        with pytest.raises(ValueError):
            check_liquidation_safety(90000, 99000, 100000, "invalid")


# ===================== Leverage Comparison =====================

class TestLeverageComparison:
    def test_returns_list(self):
        result = calculate_leverage_comparison(20000, 100000, 99500, "long")
        assert isinstance(result, list)
        assert len(result) == 6  # Default: 5, 10, 25, 50, 100, 200

    def test_custom_leverage_options(self):
        result = calculate_leverage_comparison(20000, 100000, 99500, "long", [10, 50])
        assert len(result) == 2

    def test_margin_decreases_with_leverage(self):
        result = calculate_leverage_comparison(20000, 100000, 99500, "long")
        margins = [r["margin"] for r in result]
        assert margins == sorted(margins, reverse=True)

    def test_low_leverage_safe_high_danger(self):
        """Low leverage should be safe, very high leverage may be dangerous"""
        result = calculate_leverage_comparison(20000, 100000, 99500, "long")
        # 5x should be safe
        assert result[0]["is_safe"] is True
        # 200x should be dangerous for 0.5% SL
        assert result[-1]["is_safe"] is False


# ===================== Partial Take Profits =====================

class TestPartialTakeProfits:
    def test_default_long_tps(self):
        result = calculate_partial_take_profits(100000, 20000, "long")
        assert "tp_levels" in result
        assert len(result["tp_levels"]) == 2
        assert result["total_profit"] > 0

    def test_default_short_tps(self):
        result = calculate_partial_take_profits(100000, 20000, "short")
        assert result["total_profit"] > 0

    def test_custom_tp_levels(self):
        tps = [{"price": 101000, "pct": 30}, {"price": 102000, "pct": 70}]
        result = calculate_partial_take_profits(100000, 20000, "long", tps)
        assert len(result["tp_levels"]) == 2
        assert result["tp_levels"][0]["close_pct"] == 30
        assert result["tp_levels"][1]["close_pct"] == 70

    def test_tp_pct_exceeds_100_raises(self):
        tps = [{"price": 101000, "pct": 60}, {"price": 102000, "pct": 60}]
        with pytest.raises(ValueError, match="100%"):
            calculate_partial_take_profits(100000, 20000, "long", tps)

    def test_remaining_position(self):
        tps = [{"price": 101000, "pct": 50}]
        result = calculate_partial_take_profits(100000, 20000, "long", tps)
        assert result["remaining_position"] == 10000.0


# ===================== Trailing SL Scenario =====================

class TestTrailingSL:
    def test_trailing_reduces_risk_long(self):
        result = calculate_trailing_sl_scenario(
            entry_price=100000, position_size=20000,
            original_sl_price=99500, new_sl_price=99800,
            direction="long", risk_amount=100, account_balance=10000
        )
        assert result["new_risk"] < result["original_risk"]
        assert result["risk_freed"] > 0

    def test_breakeven_long(self):
        result = calculate_trailing_sl_scenario(
            entry_price=100000, position_size=20000,
            original_sl_price=99500, new_sl_price=100000,
            direction="long", risk_amount=100, account_balance=10000
        )
        assert result["is_breakeven"] is True
        assert result["margin_can_reduce"] is True

    def test_trailing_reduces_risk_short(self):
        result = calculate_trailing_sl_scenario(
            entry_price=100000, position_size=20000,
            original_sl_price=100500, new_sl_price=100200,
            direction="short", risk_amount=100, account_balance=10000
        )
        assert result["new_risk"] < result["original_risk"]
        assert result["risk_freed"] > 0

    def test_breakeven_short(self):
        result = calculate_trailing_sl_scenario(
            entry_price=100000, position_size=20000,
            original_sl_price=100500, new_sl_price=100000,
            direction="short", risk_amount=100, account_balance=10000
        )
        assert result["is_breakeven"] is True


# ===================== Capital Allocation =====================

class TestCapitalAllocation:
    def test_50_50_split(self):
        result = calculate_capital_allocation(100000, 50)
        assert result["on_exchange"] == 50000
        assert result["off_exchange"] == 50000
        assert "conservative" in result["recommendation"].lower()

    def test_30_70_split(self):
        result = calculate_capital_allocation(100000, 30)
        assert result["on_exchange"] == 30000
        assert result["off_exchange"] == 70000

    def test_high_pct_warning(self):
        result = calculate_capital_allocation(100000, 80)
        assert "Caution" in result["recommendation"]

    def test_clamp_min(self):
        result = calculate_capital_allocation(100000, 5)
        assert result["exchange_pct"] == 10  # Clamped to minimum

    def test_clamp_max(self):
        result = calculate_capital_allocation(100000, 150)
        assert result["exchange_pct"] == 100  # Clamped to maximum


# ===================== Generate Execution Plan =====================

class TestExecutionPlan:
    def test_basic_long_plan(self):
        plan = generate_execution_plan(
            account_balance=10000, risk_pct=1.0,
            entry_price=100000, stop_loss_price=99500,
            take_profit_price=101500, direction="long", leverage=10
        )
        assert "error" not in plan
        assert "execution_plan" in plan
        assert "position_sizing" in plan
        assert "leverage_analysis" in plan
        assert "trade_outcome" in plan
        assert "partial_take_profits" in plan
        assert "capital_management" in plan
        assert "execution_checklist" in plan

    def test_basic_short_plan(self):
        plan = generate_execution_plan(
            account_balance=10000, risk_pct=1.0,
            entry_price=100000, stop_loss_price=100500,
            take_profit_price=98500, direction="short", leverage=10
        )
        assert "error" not in plan
        assert plan["execution_plan"]["direction"] == "SHORT"

    def test_invalid_sl_direction(self):
        """SL on wrong side of entry should error"""
        plan = generate_execution_plan(
            account_balance=10000, risk_pct=1.0,
            entry_price=100000, stop_loss_price=100500,
            take_profit_price=101500, direction="long", leverage=10
        )
        assert "error" in plan

    def test_plan_has_rr_ratio(self):
        plan = generate_execution_plan(
            account_balance=10000, risk_pct=1.0,
            entry_price=100000, stop_loss_price=99500,
            take_profit_price=101500, direction="long", leverage=10
        )
        assert plan["trade_outcome"]["risk_reward"] == 3.0

    def test_plan_with_tp2(self):
        plan = generate_execution_plan(
            account_balance=10000, risk_pct=1.0,
            entry_price=100000, stop_loss_price=99500,
            take_profit_price=101000, direction="long", leverage=10,
            tp2_price=102000
        )
        tps = plan["partial_take_profits"]["tp_levels"]
        assert len(tps) == 2
        assert tps[0]["price"] == 101000.0
        assert tps[1]["price"] == 102000.0

    def test_plan_isolated_margin(self):
        plan = generate_execution_plan(
            account_balance=10000, risk_pct=1.0,
            entry_price=100000, stop_loss_price=99500,
            take_profit_price=101500, direction="long"
        )
        assert "ISOLATED" in plan["execution_plan"]["margin_mode"]

    def test_plan_market_order(self):
        plan = generate_execution_plan(
            account_balance=10000, risk_pct=1.0,
            entry_price=100000, stop_loss_price=99500,
            take_profit_price=101500, direction="long"
        )
        assert "MARKET" in plan["execution_plan"]["order_type"]


# ===================== Max Safe Leverage =====================

class TestMaxSafeLeverage:
    def test_wide_sl_allows_high_leverage(self):
        """Wide SL (5%) allows very high leverage"""
        max_lev = find_max_safe_leverage(20000, 100000, 95000, "long")
        assert max_lev >= 10

    def test_tight_sl_limits_leverage(self):
        """Tight SL (0.1%) limits leverage"""
        max_lev = find_max_safe_leverage(20000, 100000, 99900, "long")
        assert max_lev < 200

    def test_short_direction(self):
        max_lev = find_max_safe_leverage(20000, 100000, 105000, "short")
        assert max_lev >= 10

    def test_returns_at_least_1(self):
        """Even very tight SL should return at least 1"""
        max_lev = find_max_safe_leverage(20000, 100000, 99999, "long")
        assert max_lev >= 1
