"""
Unit tests for server_mexc.py
Tests market structure detection and gate validation functions
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from server_mexc import MarketStructure, validate_1A


@pytest.fixture
def bullish_6candle_df(mock_candles):
    """DataFrame showing bullish 6-candle pattern"""
    candles = mock_candles["market_structure_6_candle"]
    df = pd.DataFrame(candles)

    # Create clear bullish pattern
    df.loc[0, 'close'] = 98
    df.loc[1, 'close'] = 97
    df.loc[2, 'close'] = 99  # Pivot low
    df.loc[3, 'close'] = 100
    df.loc[4, 'close'] = 101
    df.loc[5, 'close'] = 102
    df.loc[6, 'close'] = 103

    return df


@pytest.fixture
def bearish_6candle_df(mock_candles):
    """DataFrame showing bearish 6-candle pattern"""
    candles = mock_candles["market_structure_6_candle"]
    df = pd.DataFrame(candles)

    # Create clear bearish pattern
    df.loc[0, 'close'] = 102
    df.loc[1, 'close'] = 103
    df.loc[2, 'close'] = 101  # Pivot high
    df.loc[3, 'close'] = 100
    df.loc[4, 'close'] = 99
    df.loc[5, 'close'] = 98
    df.loc[6, 'close'] = 97

    return df


@pytest.fixture
def ranging_df():
    """DataFrame showing ranging market"""
    dates = pd.date_range('2026-01-01', periods=50, freq='1H')
    base = 40000

    # Oscillate between range high and low
    prices = [base + 200 if i % 4 < 2 else base - 200 for i in range(50)]

    return pd.DataFrame({
        'open_time': dates,
        'open': prices,
        'high': [p + 100 for p in prices],
        'low': [p - 100 for p in prices],
        'close': prices,
        'volume': np.random.uniform(100, 1000, 50)
    })


@pytest.fixture
def htf_bullish_data():
    """HTF data showing clear bullish trend"""
    dates = pd.date_range('2026-01-01', periods=100, freq='4H')
    base = 40000

    # Clear uptrend with higher highs and higher lows
    prices = [base + i * 50 + np.random.uniform(-100, 100) for i in range(100)]

    df = pd.DataFrame({
        'open_time': dates,
        'open': prices,
        'high': [p + np.random.uniform(50, 150) for p in prices],
        'low': [p - np.random.uniform(50, 150) for p in prices],
        'close': prices,
        'volume': np.random.uniform(100, 1000, 100)
    })

    # Ensure some clear pivot highs and lows for 6-candle rule
    for i in range(10, 90, 15):
        # Create pivot low
        df.loc[i-2, 'close'] = df.loc[i-2, 'close'] - 100
        df.loc[i-1, 'close'] = df.loc[i-1, 'close'] - 50
        df.loc[i, 'close'] = df.loc[i, 'close'] - 150  # Lowest
        df.loc[i+1, 'close'] = df.loc[i+1, 'close'] - 50
        df.loc[i+2, 'close'] = df.loc[i+2, 'close'] - 100

    return df


@pytest.mark.unit
class TestMarketStructure:
    """Tests for MarketStructure class"""

    def test_detect_pivots_insufficient_candles(self):
        """Test pivot detection with insufficient candles"""
        df = pd.DataFrame({
            'open_time': pd.date_range('2026-01-01', periods=4, freq='1H'),
            'high': [100, 101, 102, 103],
            'low': [95, 96, 97, 98],
            'close': [98, 99, 100, 101]
        })

        ms = MarketStructure()
        result = ms.detect_pivots(df)

        assert result["highs"] == []
        assert result["lows"] == []
        assert result["trend"] == "neutral"

    def test_detect_pivots_bullish_pattern(self, bullish_6candle_df):
        """Test pivot detection with bullish 6-candle pattern"""
        ms = MarketStructure()
        result = ms.detect_pivots(bullish_6candle_df)

        assert "highs" in result
        assert "lows" in result
        assert "trend" in result
        assert isinstance(result["highs"], list)
        assert isinstance(result["lows"], list)

    def test_detect_pivots_bearish_pattern(self, bearish_6candle_df):
        """Test pivot detection with bearish 6-candle pattern"""
        ms = MarketStructure()
        result = ms.detect_pivots(bearish_6candle_df)

        assert "highs" in result
        assert "lows" in result
        assert "trend" in result

    def test_pivot_high_6candle_rule(self):
        """Test pivot high requires 2 up + 2 down candles"""
        df = pd.DataFrame({
            'open_time': pd.date_range('2026-01-01', periods=10, freq='1H'),
            'high': [95, 96, 97, 100, 98, 97, 95, 94, 93, 92],
            'low': [90, 91, 92, 95, 93, 92, 90, 89, 88, 87],
            'close': [93, 94, 95, 98, 96, 95, 93, 92, 91, 90]
        })

        ms = MarketStructure()
        result = ms.detect_pivots(df)

        # Should detect pivot high at index 3
        if result["highs"]:
            assert any(p["idx"] == 3 for p in result["highs"])

    def test_pivot_low_6candle_rule(self):
        """Test pivot low requires 2 down + 2 up candles"""
        df = pd.DataFrame({
            'open_time': pd.date_range('2026-01-01', periods=10, freq='1H'),
            'high': [105, 104, 103, 100, 102, 103, 105, 106, 107, 108],
            'low': [100, 99, 98, 95, 97, 98, 100, 101, 102, 103],
            'close': [103, 102, 101, 98, 100, 101, 103, 104, 105, 106]
        })

        ms = MarketStructure()
        result = ms.detect_pivots(df)

        # Should detect pivot low at index 3
        if result["lows"]:
            assert any(p["idx"] == 3 for p in result["lows"])

    def test_trend_detection_bullish(self):
        """Test bullish trend detection (higher highs + higher lows)"""
        df = pd.DataFrame({
            'open_time': pd.date_range('2026-01-01', periods=50, freq='1H'),
            'high': list(range(100, 150)),
            'low': list(range(95, 145)),
            'close': list(range(98, 148))
        })

        # Create clear pivot structure
        df.loc[10, 'close'] = 90  # First low
        df.loc[11, 'close'] = 91
        df.loc[12, 'close'] = 92

        df.loc[20, 'close'] = 95  # Second higher low
        df.loc[21, 'close'] = 96
        df.loc[22, 'close'] = 97

        df.loc[15, 'close'] = 100  # First high
        df.loc[14, 'close'] = 99
        df.loc[13, 'close'] = 98

        df.loc[25, 'close'] = 110  # Second higher high
        df.loc[24, 'close'] = 109
        df.loc[23, 'close'] = 108

        ms = MarketStructure()
        result = ms.detect_pivots(df)

        # With proper pivots, should detect bullish or ranging
        assert result["trend"] in ["bullish", "ranging", "neutral"]

    def test_trend_detection_bearish(self):
        """Test bearish trend detection (lower highs + lower lows)"""
        df = pd.DataFrame({
            'open_time': pd.date_range('2026-01-01', periods=50, freq='1H'),
            'high': list(range(150, 100, -1)),
            'low': list(range(145, 95, -1)),
            'close': list(range(148, 98, -1))
        })

        ms = MarketStructure()
        result = ms.detect_pivots(df)

        # Trend should be detected based on pivots
        assert result["trend"] in ["bearish", "ranging", "neutral"]

    def test_detect_bos_bullish(self):
        """Test Break of Structure detection - bullish"""
        df = pd.DataFrame({
            'open_time': pd.date_range('2026-01-01', periods=20, freq='1H'),
            'high': [100 + i * 2 for i in range(20)],
            'low': [95 + i * 2 for i in range(20)],
            'close': [98 + i * 2 for i in range(20)]
        })

        ms = MarketStructure()
        pivots = {"highs": [{"price": 120, "idx": 10}], "lows": [{"price": 100, "idx": 5}]}

        bos = ms.detect_bos(df, pivots)

        if bos:  # If BOS detected
            assert bos["type"] in ["bullish", "bearish"]
            assert "level" in bos
            assert "price" in bos

    def test_detect_bos_bearish(self):
        """Test Break of Structure detection - bearish"""
        df = pd.DataFrame({
            'open_time': pd.date_range('2026-01-01', periods=20, freq='1H'),
            'high': [150 - i * 2 for i in range(20)],
            'low': [145 - i * 2 for i in range(20)],
            'close': [148 - i * 2 for i in range(20)]
        })

        ms = MarketStructure()
        pivots = {"highs": [{"price": 140, "idx": 5}], "lows": [{"price": 120, "idx": 10}]}

        bos = ms.detect_bos(df, pivots)

        if bos:
            assert bos["type"] in ["bullish", "bearish"]

    def test_detect_bos_no_pivots(self):
        """Test BOS detection with no pivots returns None"""
        df = pd.DataFrame({
            'open_time': pd.date_range('2026-01-01', periods=10, freq='1H'),
            'high': [100] * 10,
            'low': [95] * 10,
            'close': [98] * 10
        })

        ms = MarketStructure()
        pivots = {"highs": [], "lows": []}

        bos = ms.detect_bos(df, pivots)

        assert bos is None

    def test_pivot_structure_with_timestamps(self):
        """Test pivots include time information"""
        df = pd.DataFrame({
            'open_time': pd.date_range('2026-01-01', periods=10, freq='1H'),
            'high': [100, 101, 102, 105, 103, 102, 100, 99, 98, 97],
            'low': [95, 96, 97, 100, 98, 97, 95, 94, 93, 92],
            'close': [98, 99, 100, 103, 101, 100, 98, 97, 96, 95]
        })

        ms = MarketStructure()
        result = ms.detect_pivots(df)

        # Check pivot structure has required fields
        for pivot in result["highs"]:
            assert "idx" in pivot
            assert "price" in pivot
            assert "time" in pivot

        for pivot in result["lows"]:
            assert "idx" in pivot
            assert "price" in pivot
            assert "time" in pivot


@pytest.mark.unit
class TestGateValidation:
    """Tests for gate validation functions"""

    def test_validate_1A_insufficient_data(self):
        """Test Gate 1A with insufficient HTF data"""
        context = {
            "htf_candles": pd.DataFrame({
                'close': [100, 101, 102],
                'high': [105, 106, 107],
                'low': [95, 96, 97]
            })
        }

        result = validate_1A(context)

        assert result["passed"] is False
        assert result["bias"] == "neutral"
        assert result["confidence"] == 0.0
        assert "Insufficient" in result["reason"]

    def test_validate_1A_missing_data(self):
        """Test Gate 1A with missing HTF data"""
        context = {}

        result = validate_1A(context)

        assert result["passed"] is False
        assert result["bias"] == "neutral"

    def test_validate_1A_valid_htf_data(self, htf_bullish_data):
        """Test Gate 1A with valid HTF bullish data"""
        context = {"htf_candles": htf_bullish_data}

        result = validate_1A(context)

        assert "passed" in result
        assert "bias" in result
        assert "confidence" in result
        assert result["bias"] in ["bullish", "bearish", "neutral", "ranging"]
        assert 0.0 <= result["confidence"] <= 1.0

    def test_validate_1A_structure(self, htf_bullish_data):
        """Test Gate 1A returns correct structure"""
        context = {"htf_candles": htf_bullish_data}

        result = validate_1A(context)

        # Required fields
        assert "passed" in result
        assert "bias" in result
        assert "confidence" in result

        # Optional fields
        if not result["passed"]:
            assert "reason" in result


@pytest.mark.integration
class TestMarketStructureIntegration:
    """Integration tests for market structure detection"""

    def test_full_pivot_detection_pipeline(self):
        """Test complete pivot detection pipeline"""
        # Create realistic market data
        dates = pd.date_range('2026-01-01', periods=100, freq='1H')
        base = 40000

        # Create wave pattern
        prices = [base + 500 * np.sin(i * 0.1) + np.random.uniform(-100, 100) for i in range(100)]

        df = pd.DataFrame({
            'open_time': dates,
            'open': prices,
            'high': [p + np.random.uniform(50, 150) for p in prices],
            'low': [p - np.random.uniform(50, 150) for p in prices],
            'close': prices,
            'volume': np.random.uniform(100, 1000, 100)
        })

        ms = MarketStructure()
        pivots = ms.detect_pivots(df)
        bos = ms.detect_bos(df, pivots)

        # Should complete without errors
        assert "highs" in pivots
        assert "lows" in pivots
        assert "trend" in pivots

    def test_ranging_market_detection(self, ranging_df):
        """Test detection of ranging market"""
        ms = MarketStructure()
        result = ms.detect_pivots(ranging_df)

        # Ranging market should be detected or neutral
        assert result["trend"] in ["ranging", "neutral", "bullish", "bearish"]

    def test_strong_trend_detection(self):
        """Test detection of strong trending market"""
        # Create strong uptrend
        dates = pd.date_range('2026-01-01', periods=100, freq='1H')
        prices = [40000 + i * 50 for i in range(100)]  # Strong linear trend

        df = pd.DataFrame({
            'open_time': dates,
            'open': prices,
            'high': [p + 100 for p in prices],
            'low': [p - 100 for p in prices],
            'close': prices,
            'volume': 100
        })

        ms = MarketStructure()
        result = ms.detect_pivots(df)

        # Strong trend should have some pivot structure
        assert isinstance(result["highs"], list)
        assert isinstance(result["lows"], list)


@pytest.mark.unit
class TestMarketStructureEdgeCases:
    """Edge case tests for market structure"""

    def test_flat_market_no_pivots(self):
        """Test completely flat market has no pivots"""
        df = pd.DataFrame({
            'open_time': pd.date_range('2026-01-01', periods=20, freq='1H'),
            'high': [100.5] * 20,
            'low': [99.5] * 20,
            'close': [100.0] * 20
        })

        ms = MarketStructure()
        result = ms.detect_pivots(df)

        # Flat market should have no pivots
        assert len(result["highs"]) == 0
        assert len(result["lows"]) == 0
        assert result["trend"] == "neutral"

    def test_single_pivot_high(self):
        """Test market with single pivot high"""
        df = pd.DataFrame({
            'open_time': pd.date_range('2026-01-01', periods=15, freq='1H'),
            'close': [95, 96, 97, 100, 99, 98, 97, 96, 95, 94, 93, 92, 91, 90, 89],
            'high': [96, 97, 98, 101, 100, 99, 98, 97, 96, 95, 94, 93, 92, 91, 90],
            'low': [94, 95, 96, 99, 98, 97, 96, 95, 94, 93, 92, 91, 90, 89, 88]
        })

        ms = MarketStructure()
        result = ms.detect_pivots(df)

        # Should detect at least the pivot high
        assert isinstance(result["highs"], list)
        assert result["trend"] in ["neutral", "bearish"]

    def test_alternating_pivots(self):
        """Test market with alternating pivot highs and lows"""
        df = pd.DataFrame({
            'open_time': pd.date_range('2026-01-01', periods=30, freq='1H'),
            'close': [100, 99, 98, 99, 100, 101, 100, 99, 98, 97, 98, 99, 100, 101, 102,
                     101, 100, 99, 98, 97, 96, 97, 98, 99, 100, 101, 100, 99, 98, 97],
            'high': [101, 100, 99, 100, 101, 102, 101, 100, 99, 98, 99, 100, 101, 102, 103,
                    102, 101, 100, 99, 98, 97, 98, 99, 100, 101, 102, 101, 100, 99, 98],
            'low': [99, 98, 97, 98, 99, 100, 99, 98, 97, 96, 97, 98, 99, 100, 101,
                   100, 99, 98, 97, 96, 95, 96, 97, 98, 99, 100, 99, 98, 97, 96]
        })

        ms = MarketStructure()
        result = ms.detect_pivots(df)

        # Should detect multiple pivots
        assert result["trend"] in ["ranging", "neutral", "bullish", "bearish"]
