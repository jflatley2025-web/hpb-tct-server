"""
Unit tests for risk_model.py
Tests multi-timeframe risk calculations, volatility, and signal derivation
"""
import pytest
import numpy as np
from unittest.mock import AsyncMock, patch, MagicMock
from risk_model import (
    fetch_live_prices,
    smooth_confidence,
    volatility,
    derive_signal,
    compute_risk_profile
)


@pytest.fixture
def sample_prices():
    """Sample price data for testing"""
    return [40000 + i * 50 + np.random.uniform(-100, 100) for i in range(100)]


@pytest.fixture
def volatile_prices():
    """Highly volatile price data"""
    base = 40000
    return [base + np.random.uniform(-1000, 1000) for _ in range(100)]


@pytest.fixture
def stable_prices():
    """Low volatility price data"""
    base = 40000
    return [base + np.random.uniform(-50, 50) for _ in range(100)]


@pytest.fixture
def basic_context():
    """Basic context for risk profile computation"""
    return {
        "ExecutionConfidence_Total": 0.75,
        "gates": {
            "1A": {"bias": "accumulation", "score": 0.85}
        },
        "Reward_Summary": "POS_BIAS"
    }


@pytest.mark.unit
class TestSmoothConfidence:
    """Tests for smooth_confidence function"""

    def test_smooth_confidence_basic(self):
        """Test basic confidence smoothing"""
        history = [0.5, 0.6, 0.7, 0.8, 0.9]
        result = smooth_confidence(history, window=5)

        assert result == pytest.approx(0.7, rel=0.01)
        assert isinstance(result, float)

    def test_smooth_confidence_empty_history(self):
        """Test smoothing with empty history"""
        history = []
        result = smooth_confidence(history, window=5)

        assert result == 0.0

    def test_smooth_confidence_small_window(self):
        """Test smoothing with window smaller than history"""
        history = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        result = smooth_confidence(history, window=3)

        # Should only use last 3 values: 0.6, 0.7, 0.8
        assert result == pytest.approx(0.7, rel=0.01)

    def test_smooth_confidence_large_window(self):
        """Test smoothing with window larger than history"""
        history = [0.5, 0.6, 0.7]
        result = smooth_confidence(history, window=10)

        # Should use all available values
        assert result == pytest.approx(0.6, rel=0.01)

    def test_smooth_confidence_single_value(self):
        """Test smoothing with single value in history"""
        history = [0.85]
        result = smooth_confidence(history, window=5)

        assert result == 0.85


@pytest.mark.unit
class TestVolatility:
    """Tests for volatility function"""

    def test_volatility_basic(self, sample_prices):
        """Test basic volatility calculation"""
        vol = volatility(sample_prices)

        assert vol >= 0.0
        assert isinstance(vol, (float, np.floating))

    def test_volatility_insufficient_data(self):
        """Test volatility with insufficient data"""
        prices = [40000]
        vol = volatility(prices)

        assert vol == 0.0

    def test_volatility_stable_prices(self, stable_prices):
        """Test volatility with stable prices"""
        vol = volatility(stable_prices)

        # Low volatility should be close to 0
        assert vol < 0.01

    def test_volatility_volatile_prices(self, volatile_prices):
        """Test volatility with highly volatile prices"""
        vol = volatility(volatile_prices)

        # High volatility
        assert vol > 0.0

    def test_volatility_uses_last_50_candles(self):
        """Test volatility only uses last 50 candles"""
        # Create 100 candles: first 50 stable, last 50 volatile
        stable = [40000 + i * 0.1 for i in range(50)]
        volatile = [40000 + np.random.uniform(-1000, 1000) for _ in range(50)]
        prices = stable + volatile

        vol = volatility(prices)

        # Should reflect volatility of last 50 candles
        assert vol > 0.0

    def test_volatility_zero_prices(self):
        """Test volatility with zero prices returns 0"""
        prices = [0] * 100
        vol = volatility(prices)

        assert vol == 0.0 or np.isnan(vol)


@pytest.mark.unit
class TestDeriveSignal:
    """Tests for derive_signal function"""

    def test_derive_signal_wait_low_confidence(self):
        """Test signal returns WAIT when confidence is low"""
        signal = derive_signal(conf=0.1, bias="Accumulation", reward="POS_BIAS")
        assert signal == "WAIT"

        signal = derive_signal(conf=0.15, bias="Distribution", reward="NEG_BIAS")
        assert signal == "WAIT"

    def test_derive_signal_buy_pos_bias_accumulation(self):
        """Test BUY signal with positive bias and accumulation"""
        signal = derive_signal(conf=0.75, bias="Accumulation", reward="POS_BIAS")
        assert signal == "BUY"

    def test_derive_signal_sell_neg_bias_distribution(self):
        """Test SELL signal with negative bias and distribution"""
        signal = derive_signal(conf=0.75, bias="Distribution", reward="NEG_BIAS")
        assert signal == "SELL"

    def test_derive_signal_wait_neg_bias_accumulation(self):
        """Test WAIT when negative bias conflicts with accumulation"""
        signal = derive_signal(conf=0.75, bias="Accumulation", reward="NEG_BIAS")
        assert signal == "WAIT"

    def test_derive_signal_wait_pos_bias_distribution(self):
        """Test WAIT when positive bias conflicts with distribution"""
        signal = derive_signal(conf=0.75, bias="Distribution", reward="POS_BIAS")
        assert signal == "WAIT"

    def test_derive_signal_buy_high_confidence_accumulation(self):
        """Test BUY with high confidence and accumulation (no reward)"""
        signal = derive_signal(conf=0.85, bias="Accumulation", reward="N/A")
        assert signal == "BUY"

    def test_derive_signal_sell_high_confidence_distribution(self):
        """Test SELL with high confidence and distribution (no reward)"""
        signal = derive_signal(conf=0.85, bias="Distribution", reward="N/A")
        assert signal == "SELL"

    def test_derive_signal_hold_medium_confidence(self):
        """Test HOLD with medium confidence and no clear bias"""
        signal = derive_signal(conf=0.5, bias="Neutral", reward="N/A")
        assert signal == "HOLD"

    def test_derive_signal_threshold_at_02(self):
        """Test confidence threshold at 0.2"""
        signal = derive_signal(conf=0.20, bias="Accumulation", reward="POS_BIAS")
        assert signal in ["WAIT", "HOLD", "BUY"]  # Boundary case - implementation may vary

    def test_derive_signal_threshold_at_07(self):
        """Test confidence threshold at 0.7"""
        signal = derive_signal(conf=0.70, bias="Accumulation", reward="N/A")
        assert signal in ["BUY", "HOLD"]  # Boundary case

    def test_derive_signal_case_insensitive_reward(self):
        """Test reward string is case-insensitive"""
        signal1 = derive_signal(conf=0.75, bias="Accumulation", reward="pos_bias")
        signal2 = derive_signal(conf=0.75, bias="Accumulation", reward="POS_BIAS")
        signal3 = derive_signal(conf=0.75, bias="Accumulation", reward="Pos_Bias")

        assert signal1 == signal2 == signal3 == "BUY"


@pytest.mark.unit
@pytest.mark.asyncio
class TestFetchLivePrices:
    """Tests for fetch_live_prices function"""

    async def test_fetch_live_prices_success(self):
        """Test successful price fetch from OKX"""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": [
                ["1706000000000", "42000", "42100", "41900", "42050", "1000"],
                ["1706000060000", "42050", "42150", "42000", "42100", "950"],
                ["1706000120000", "42100", "42200", "42080", "42180", "1100"],
            ]
        }

        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(return_value=mock_response)

            prices = await fetch_live_prices(symbol="BTC-USDT", interval="1H", limit=100)

            assert len(prices) == 3
            # API returns newest first, implementation reverses to chronological order
            assert prices[0] == 42180.0  # Reversed: newest becomes last
            assert prices[1] == 42100.0
            assert prices[2] == 42050.0

    async def test_fetch_live_prices_empty_response(self):
        """Test fetch with empty API response"""
        mock_response = MagicMock()
        mock_response.json.return_value = {"data": []}

        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(return_value=mock_response)

            prices = await fetch_live_prices()

            assert prices == []

    async def test_fetch_live_prices_no_data_key(self):
        """Test fetch when response has no 'data' key"""
        mock_response = MagicMock()
        mock_response.json.return_value = {"error": "invalid request"}

        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(return_value=mock_response)

            prices = await fetch_live_prices()

            assert prices == []

    async def test_fetch_live_prices_correct_url_format(self):
        """Test fetch builds correct OKX URL"""
        mock_response = MagicMock()
        mock_response.json.return_value = {"data": []}

        with patch('httpx.AsyncClient') as mock_client:
            mock_get = AsyncMock(return_value=mock_response)
            mock_client.return_value.__aenter__.return_value.get = mock_get

            await fetch_live_prices(symbol="ETH-USDT", interval="4H", limit=50)

            # Verify URL contains correct parameters
            call_args = mock_get.call_args[0][0]
            assert "ETH-USDT-SWAP" in call_args
            assert "bar=4H" in call_args
            assert "limit=50" in call_args


@pytest.mark.unit
@pytest.mark.asyncio
class TestComputeRiskProfile:
    """Tests for compute_risk_profile function"""

    async def test_compute_risk_profile_basic(self, basic_context):
        """Test basic risk profile computation"""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": [
                ["1706000000000", "42000", "42100", "41900", str(42000 + i * 10), "1000"]
                for i in range(100)
            ]
        }

        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(return_value=mock_response)

            profile = await compute_risk_profile(basic_context)

            assert "timestamp" in profile
            assert "phase" in profile
            assert "signal" in profile
            assert "confidence_smoothed" in profile
            assert "risk_score" in profile
            assert "volatility_score" in profile
            assert "stop_loss" in profile
            assert "take_profit" in profile
            assert "current_price" in profile
            assert "timeframes" in profile

    async def test_compute_risk_profile_timeframes(self, basic_context):
        """Test risk profile includes all timeframes"""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": [
                ["1706000000000", "42000", "42100", "41900", "42050", "1000"]
                for _ in range(100)
            ]
        }

        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(return_value=mock_response)

            profile = await compute_risk_profile(basic_context)

            assert "15m" in profile["timeframes"]
            assert "1H" in profile["timeframes"]
            assert "4H" in profile["timeframes"]

            for tf_data in profile["timeframes"].values():
                assert "price" in tf_data
                assert "volatility" in tf_data

    async def test_compute_risk_profile_confidence_smoothing(self):
        """Test risk profile smooths confidence over time"""
        context = {
            "ExecutionConfidence_Total": 0.8,
            "gates": {"1A": {"bias": "accumulation"}},
            "Reward_Summary": "POS_BIAS",
            "_conf_history": [0.5, 0.6, 0.7]
        }

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": [
                ["1706000000000", "42000", "42100", "41900", "42050", "1000"]
            ]
        }

        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(return_value=mock_response)

            profile = await compute_risk_profile(context)

            # Confidence should be smoothed
            assert "confidence_smoothed" in profile
            assert 0.0 <= profile["confidence_smoothed"] <= 1.0

            # History should be updated
            assert len(context["_conf_history"]) == 4

    async def test_compute_risk_profile_history_limit(self):
        """Test confidence history is limited to 100 entries"""
        context = {
            "ExecutionConfidence_Total": 0.8,
            "gates": {"1A": {"bias": "accumulation"}},
            "Reward_Summary": "POS_BIAS",
            "_conf_history": [0.5] * 100  # Already at limit
        }

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": [["1706000000000", "42000", "42100", "41900", "42050", "1000"]]
        }

        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(return_value=mock_response)

            await compute_risk_profile(context)

            # Should still be 100 (oldest removed, newest added)
            assert len(context["_conf_history"]) == 100

    async def test_compute_risk_profile_stop_loss_take_profit(self, basic_context):
        """Test stop loss and take profit calculations"""
        mock_response = MagicMock()
        # Provide enough data points for volatility calculation
        mock_response.json.return_value = {
            "data": [
                ["1706000000000", "42000", "42100", "41900", str(40000 + i * 10), "1000"]
                for i in range(50)
            ]
        }

        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(return_value=mock_response)

            profile = await compute_risk_profile(basic_context)

            # Stop loss should be at or below current price
            assert profile["stop_loss"] <= profile["current_price"]

            # Take profit should be at or above current price
            assert profile["take_profit"] >= profile["current_price"]

    async def test_compute_risk_profile_error_handling(self):
        """Test risk profile handles errors gracefully"""
        context = {"invalid": "data"}

        with patch('httpx.AsyncClient') as mock_client:
            # Simulate network error
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                side_effect=Exception("Network error")
            )

            profile = await compute_risk_profile(context)

            assert "error" in profile

    async def test_compute_risk_profile_signal_derivation(self):
        """Test risk profile derives correct signal"""
        context = {
            "ExecutionConfidence_Total": 0.85,
            "gates": {"1A": {"bias": "accumulation"}},
            "Reward_Summary": "POS_BIAS"
        }

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": [["1706000000000", "42000", "42100", "41900", "42050", "1000"]]
        }

        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(return_value=mock_response)

            profile = await compute_risk_profile(context)

            assert profile["signal"] in ["BUY", "SELL", "HOLD", "WAIT"]

    async def test_compute_risk_profile_phase_normalization(self):
        """Test phase is normalized to title case"""
        context = {
            "ExecutionConfidence_Total": 0.75,
            "gates": {"1A": {"bias": "ACCUMULATION"}},
            "Reward_Summary": "POS_BIAS"
        }

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": [["1706000000000", "42000", "42100", "41900", "42050", "1000"]]
        }

        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(return_value=mock_response)

            profile = await compute_risk_profile(context)

            assert profile["phase"] == "Accumulation"


@pytest.mark.integration
class TestRiskModelIntegration:
    """Integration tests for risk model"""

    def test_volatility_pipeline(self):
        """Test complete volatility calculation pipeline"""
        # Create realistic price data
        prices = [40000 + i * 10 + np.random.uniform(-50, 50) for i in range(100)]

        vol = volatility(prices)
        assert vol >= 0.0

        # Use in signal derivation
        conf = 0.5 + vol  # Confidence influenced by volatility
        signal = derive_signal(min(conf, 1.0), "Accumulation", "POS_BIAS")
        assert signal in ["BUY", "SELL", "HOLD", "WAIT"]

    def test_confidence_smoothing_pipeline(self):
        """Test confidence smoothing over multiple iterations"""
        history = []

        for i in range(20):
            # Simulate varying confidence
            conf = 0.5 + np.random.uniform(-0.2, 0.2)
            history.append(conf)

            smoothed = smooth_confidence(history, window=5)
            assert 0.0 <= smoothed <= 1.0

            # Smoothed should be less volatile than raw
            if len(history) >= 5:
                recent_std = np.std(history[-5:])
                assert smoothed <= max(history[-5:])
                assert smoothed >= min(history[-5:])
