"""
Shared pytest fixtures and configuration for HPB-TCT tests
"""
import json
import pytest
from pathlib import Path


@pytest.fixture
def mock_candles():
    """Load mock candle data from fixtures"""
    fixtures_path = Path(__file__).parent / "fixtures" / "mock_candles.json"
    with open(fixtures_path) as f:
        return json.load(f)


@pytest.fixture
def simple_range_candles(mock_candles):
    """Simple range candles for basic range detection tests"""
    return mock_candles["simple_range_candles"]


@pytest.fixture
def uptrend_candles(mock_candles):
    """Candles showing clear uptrend"""
    return mock_candles["uptrend_candles"]


@pytest.fixture
def downtrend_candles(mock_candles):
    """Candles showing clear downtrend"""
    return mock_candles["downtrend_candles"]


@pytest.fixture
def mock_mexc_response(mock_candles):
    """Mock MEXC API response"""
    return mock_candles["mexc_klines_btcusdt_1m"]


@pytest.fixture
def mock_context_basic():
    """Basic HPB context for gate validation tests"""
    return {
        "gates": {
            "1A": {"bias": "bullish", "score": 0.85},
            "1B": {"valid": True, "score": 0.90},
            "1C": {"valid": True, "score": 0.88},
            "RCM": {"valid": True, "range_duration_hours": 48},
            "MSCE": {"session_bias": "bullish", "session": "London"},
            "RIG": {"valid": True, "score": 0.92},
            "1D": {"score": 0.87}
        },
        "ExecutionConfidence_Total": 0.88,
        "Reward_Summary": "POS_BIAS",
        "local_range_displacement": 0.15
    }


@pytest.fixture
def mock_context_counter_bias():
    """Context with counter-bias conditions for RIG blocking"""
    return {
        "gates": {
            "1A": {"bias": "bullish", "score": 0.85},
            "RCM": {"valid": True, "range_duration_hours": 48},
            "MSCE": {"session_bias": "bearish", "session": "New York"},
            "1D": {"score": 0.75}
        },
        "local_range_displacement": 0.15,
        "ExecutionConfidence_Total": 0.70
    }


@pytest.fixture
def mock_context_invalid_range():
    """Context with invalid/broken range"""
    return {
        "gates": {
            "1A": {"bias": "bullish", "score": 0.85},
            "RCM": {"valid": False, "range_duration_hours": 12},
            "MSCE": {"session_bias": "bullish", "session": "London"},
            "1D": {"score": 0.80}
        },
        "local_range_displacement": 0.45,
        "ExecutionConfidence_Total": 0.75
    }
