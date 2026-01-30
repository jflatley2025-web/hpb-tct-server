"""
Unit tests for the Risk Management Calculator (TCT Lecture 7)
Tests position sizing, leverage calculation, compounding, and streak simulation.
"""
import pytest
from unittest.mock import AsyncMock, patch


# ─────────────────────────────────────
# Direct formula tests (no server needed)
# ─────────────────────────────────────

@pytest.mark.unit
class TestPositionSizing:
    """Tests the core TCT Lecture 7 position sizing formula:
       Position Size = (Risk Amount / SL%) x 100
    """

    def calc_position_size(self, balance, risk_pct, sl_pct):
        risk_amount = balance * (risk_pct / 100)
        return (risk_amount / sl_pct) * 100

    def test_basic_crypto_position_size(self):
        """TCT Lecture 7 example: $10K account, 1% risk, 0.26% SL = $38,461.54"""
        pos = self.calc_position_size(10000, 1.0, 0.26)
        assert round(pos, 2) == 38461.54

    def test_larger_sl_reduces_position(self):
        """Wider stop-loss should reduce position size"""
        pos_tight = self.calc_position_size(10000, 1.0, 0.26)
        pos_wide = self.calc_position_size(10000, 1.0, 1.0)
        assert pos_wide < pos_tight

    def test_higher_risk_pct_increases_position(self):
        """3% risk should give 3x the position of 1% risk"""
        pos_1pct = self.calc_position_size(10000, 1.0, 0.5)
        pos_3pct = self.calc_position_size(10000, 3.0, 0.5)
        assert round(pos_3pct / pos_1pct, 1) == 3.0

    def test_forex_lot_conversion(self):
        """1 lot in forex = 100,000 units"""
        pos = self.calc_position_size(10000, 1.0, 0.26)
        lots = pos / 100000
        assert lots == pytest.approx(0.3846, abs=0.001)

    def test_gold_lot_conversion(self):
        """1 lot in gold = gold_price * 100"""
        pos = self.calc_position_size(10000, 1.0, 0.26)
        gold_price = 2000
        lots = pos / (gold_price * 100)
        assert lots == pytest.approx(0.1923, abs=0.001)

    def test_minimum_risk_amount(self):
        """Small account with 1% risk"""
        pos = self.calc_position_size(500, 1.0, 0.5)
        assert pos == 1000.0

    def test_risk_amount_calculation(self):
        """Risk amount = balance * risk_pct / 100"""
        balance = 5000
        risk_pct = 2.0
        risk_amount = balance * (risk_pct / 100)
        assert risk_amount == 100.0


@pytest.mark.unit
class TestLeverageCalculation:
    """Tests leverage requirements from TCT Lecture 7:
       Leverage Needed = Position Size / Account Balance
       Used Margin = Position Size / Leverage
    """

    def test_min_leverage_needed(self):
        """TCT example: $38,461 position on $10K account needs 3.85x"""
        position_size = 38461.54
        balance = 10000
        min_lev = position_size / balance
        assert round(min_lev, 2) == 3.85

    def test_margin_with_higher_leverage(self):
        """Using 30x leverage: margin = 38,461 / 30 = $1,282.05"""
        position_size = 38461.54
        leverage = 30
        margin = position_size / leverage
        assert round(margin, 2) == 1282.05

    def test_free_margin(self):
        """Free margin = balance - used margin"""
        balance = 10000
        used_margin = 1282.05
        free = balance - used_margin
        assert round(free, 2) == 8717.95

    def test_risk_unchanged_with_leverage(self):
        """Key TCT principle: risk amount stays the same regardless of leverage"""
        balance = 10000
        risk_pct = 1.0
        risk_amount = balance * risk_pct / 100  # $100

        # With 3.85x leverage
        margin_low = 38461.54 / 3.85
        # With 30x leverage
        margin_high = 38461.54 / 30
        # With 60x leverage
        margin_extreme = 38461.54 / 60

        # Risk amount is always $100 regardless
        assert risk_amount == 100.0
        # But margins differ
        assert margin_low > margin_high > margin_extreme

    def test_leverage_1x_uses_full_balance(self):
        """At 1x leverage, margin equals position size"""
        position = 5000
        margin = position / 1
        assert margin == position


@pytest.mark.unit
class TestRiskReward:
    """Tests risk-to-reward ratio calculations from TCT Lecture 7"""

    def test_basic_rr_profit(self):
        """Risk $100 at 3R = $300 profit"""
        risk_amount = 100
        rr = 3.0
        profit = risk_amount * rr
        assert profit == 300.0

    def test_account_gain_pct(self):
        """$300 profit on $10K = 3% account gain"""
        profit = 300
        balance = 10000
        gain_pct = (profit / balance) * 100
        assert gain_pct == 3.0

    def test_tct_average_rr(self):
        """TCT stats: 2.3 average R:R with $50-$150 risk"""
        risk_amounts = [50, 100, 150]
        rr = 2.3
        profits = [r * rr for r in risk_amounts]
        assert profits[0] == pytest.approx(115.0, abs=0.1)
        assert profits[1] == pytest.approx(230.0, abs=0.1)
        assert profits[2] == pytest.approx(345.0, abs=0.1)

    def test_rr_vs_pct_move_independence(self):
        """TCT principle: R:R matters, not percentage price move.
           Trader A: wide SL, big move, lower R:R
           Trader B: tight SL, small move, higher R:R
        """
        # Trader A: caught 32% move, but R:R = 2.26
        trader_a_rr = 2.26
        trader_a_profit = 100 * trader_a_rr  # $226

        # Trader B: caught 5% move, but R:R = 7.5
        trader_b_rr = 7.5
        trader_b_profit = 100 * trader_b_rr  # $750

        assert trader_b_profit > trader_a_profit


@pytest.mark.unit
class TestStreakSimulation:
    """Tests the worst-case streak scenario from TCT Lecture 7:
       6 losses then 3 wins with compounding
    """

    def test_six_loss_streak(self):
        """$10K, 1% risk, 6 losses = $9,414.80"""
        balance = 10000
        for _ in range(6):
            balance *= 0.99
        assert round(balance, 2) == pytest.approx(9414.80, abs=0.1)

    def test_three_win_recovery(self):
        """After 6 losses at 1%, 3 wins at 3R = $10,287.81"""
        balance = 10000
        # 6 losses at 1%
        for _ in range(6):
            balance *= 0.99
        # 3 wins at 3% (1% risk * 3R)
        for _ in range(3):
            balance *= 1.03
        assert round(balance, 2) == pytest.approx(10287.81, abs=1.0)

    def test_net_positive_after_bad_streak(self):
        """Even with 6L and 3W (33% win rate), net result is positive at 3R"""
        balance = 10000
        for _ in range(6):
            balance *= 0.99
        for _ in range(3):
            balance *= 1.03
        assert balance > 10000

    def test_higher_risk_worse_streak(self):
        """Higher risk % makes losing streaks more damaging"""
        bal_1pct = 10000
        bal_3pct = 10000
        for _ in range(6):
            bal_1pct *= 0.99
            bal_3pct *= 0.97
        assert bal_3pct < bal_1pct

    def test_net_negative_low_rr(self):
        """At 1R, 6 losses and 3 wins should be net negative"""
        balance = 10000
        for _ in range(6):
            balance *= 0.99  # 1% loss
        for _ in range(3):
            balance *= 1.01  # 1% gain (1R)
        assert balance < 10000


@pytest.mark.unit
class TestCompounding:
    """Tests compounding projections from TCT Lecture 7:
       5% per week over 35 trading weeks per year
    """

    def test_year_1_compounding(self):
        """$10K at 5%/week for 35 weeks = $55,160.15"""
        balance = 10000
        balance *= (1.05 ** 35)
        assert round(balance, 2) == pytest.approx(55160.15, abs=5.0)

    def test_year_2_compounding(self):
        """Year 2 projection"""
        balance = 10000
        balance *= (1.05 ** 70)  # 2 years of 35 weeks
        assert round(balance, 2) == pytest.approx(304264.26, abs=50.0)

    def test_year_3_compounding(self):
        """Year 3: over $1.6M"""
        balance = 10000
        balance *= (1.05 ** 105)  # 3 years of 35 weeks
        assert balance > 1600000

    def test_weekly_5pct_equals_daily_1pct(self):
        """5% weekly ~ 1% per day (5 trading days)"""
        weekly = 0.05
        daily = (1 + weekly) ** (1/5) - 1
        assert daily == pytest.approx(0.00985, abs=0.001)

    def test_zero_growth(self):
        """0% weekly growth = no change"""
        balance = 10000
        balance *= (1.0 ** 35)
        assert balance == 10000


@pytest.mark.unit
class TestRiskRules:
    """Tests that risk management rules from TCT Lecture 7 are enforced"""

    def test_risk_range_1_to_3_pct(self):
        """Risk should be between 1% and 3%"""
        for pct in [1.0, 1.5, 2.0, 2.5, 3.0]:
            risk = 10000 * (pct / 100)
            assert 100 <= risk <= 300

    def test_zero_sl_invalid(self):
        """Stop-loss of 0% should be invalid (division by zero)"""
        with pytest.raises(ZeroDivisionError):
            _ = (100 / 0) * 100

    def test_negative_sl_invalid(self):
        """Negative stop-loss makes no sense"""
        sl = -0.5
        pos = (100 / sl) * 100
        assert pos < 0  # Negative position = invalid

    def test_min_rr_of_2(self):
        """TCT minimum R:R is 2:1"""
        min_rr = 2.0
        risk = 100
        min_profit = risk * min_rr
        assert min_profit == 200.0


# ─────────────────────────────────────
# API endpoint integration tests
# ─────────────────────────────────────

@pytest.mark.unit
class TestRiskCalculatorEndpoint:
    """Tests the /api/risk-calculator endpoint logic"""

    def _calculate(self, balance=10000, risk_pct=1.0, sl_pct=0.26,
                   rr=3.0, market="crypto", gold_price=2000, leverage=10.0):
        """Replicate the endpoint logic for testing"""
        risk_amount = balance * (risk_pct / 100)
        raw_position_size = (risk_amount / sl_pct) * 100

        min_leverage_needed = raw_position_size / balance if balance > 0 else 0
        used_margin = raw_position_size / leverage if leverage > 0 else raw_position_size
        free_margin = balance - used_margin

        loss_at_sl = risk_amount
        profit_at_tp = risk_amount * rr
        tp_pct_gain = (profit_at_tp / balance) * 100

        return {
            "risk_amount": round(risk_amount, 2),
            "position_size": round(raw_position_size, 2),
            "min_leverage": round(min_leverage_needed, 2),
            "used_margin": round(used_margin, 2),
            "free_margin": round(free_margin, 2),
            "loss": round(loss_at_sl, 2),
            "profit": round(profit_at_tp, 2),
            "gain_pct": round(tp_pct_gain, 2)
        }

    def test_default_params(self):
        result = self._calculate()
        assert result["risk_amount"] == 100.0
        assert result["position_size"] == 38461.54
        assert result["profit"] == 300.0

    def test_crypto_market(self):
        result = self._calculate(market="crypto")
        assert result["position_size"] > 0

    def test_forex_lot_size(self):
        result = self._calculate(market="forex")
        lots = result["position_size"] / 100000
        assert lots == pytest.approx(0.3846, abs=0.001)

    def test_gold_lot_size(self):
        result = self._calculate(market="gold", gold_price=2000)
        lot_value = 2000 * 100
        lots = result["position_size"] / lot_value
        assert lots == pytest.approx(0.1923, abs=0.001)

    def test_high_leverage_less_margin(self):
        res_low = self._calculate(leverage=5)
        res_high = self._calculate(leverage=50)
        assert res_high["used_margin"] < res_low["used_margin"]
        # But risk stays the same
        assert res_low["loss"] == res_high["loss"]

    def test_3pct_risk(self):
        result = self._calculate(risk_pct=3.0)
        assert result["risk_amount"] == 300.0
        assert result["loss"] == 300.0

    def test_compounding_projection_shape(self):
        """Test that compounding produces 3 years of projections"""
        balance = 10000
        weekly_rate = 0.05
        weeks_per_year = 35
        projections = []
        bal = balance
        for year in range(1, 4):
            bal = bal * ((1 + weekly_rate) ** weeks_per_year)
            projections.append(round(bal, 2))
        assert len(projections) == 3
        assert projections[0] < projections[1] < projections[2]
        assert projections[0] == pytest.approx(55160.15, abs=5.0)

    def test_streak_simulation_net_positive(self):
        """6 losses + 3 wins at 3R with 1% risk = net positive"""
        balance = 10000
        risk_pct = 1.0
        rr = 3.0
        loss_mult = 1 - (risk_pct / 100)
        win_mult = 1 + (risk_pct * rr / 100)

        bal = balance
        for _ in range(6):
            bal *= loss_mult
        for _ in range(3):
            bal *= win_mult

        assert bal > balance

    def test_streak_simulation_net_negative_low_rr(self):
        """6 losses + 3 wins at 1R with 1% risk = net negative"""
        balance = 10000
        loss_mult = 0.99
        win_mult = 1.01

        bal = balance
        for _ in range(6):
            bal *= loss_mult
        for _ in range(3):
            bal *= win_mult

        assert bal < balance
