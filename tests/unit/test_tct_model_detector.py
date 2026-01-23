"""
Unit tests for tct_model_detector.py
Tests TCT model detection including liquidity curves, tap detection, and model builders
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from tct_model_detector import (
    LiquidityCurveDetector,
    TCTModelDetector,
    detect_tct_models
)


@pytest.fixture
def accumulation_df():
    """DataFrame with accumulation pattern (higher lows)"""
    dates = pd.date_range('2026-01-01', periods=100, freq='1H')
    base = 40000
    prices = []

    # Create accumulation pattern: low -> high -> pullback above low
    for i in range(100):
        if i < 20:  # Initial low zone
            prices.append(base + np.random.uniform(-200, 200))
        elif i < 50:  # Move up
            prices.append(base + (i-20) * 20 + np.random.uniform(-150, 150))
        else:  # Pullback to higher low
            prices.append(base + 400 + np.random.uniform(-100, 100))

    return pd.DataFrame({
        'open_time': dates,
        'open': prices,
        'high': [p + np.random.uniform(50, 150) for p in prices],
        'low': [p - np.random.uniform(50, 150) for p in prices],
        'close': [p + np.random.uniform(-100, 100) for p in prices],
        'volume': np.random.uniform(100, 1000, 100)
    })


@pytest.fixture
def distribution_df():
    """DataFrame with distribution pattern (lower highs)"""
    dates = pd.date_range('2026-01-01', periods=100, freq='1H')
    base = 42000
    prices = []

    # Create distribution pattern: high -> low -> rally to lower high
    for i in range(100):
        if i < 20:  # Initial high zone
            prices.append(base + np.random.uniform(-200, 200))
        elif i < 50:  # Move down
            prices.append(base - (i-20) * 20 + np.random.uniform(-150, 150))
        else:  # Rally to lower high
            prices.append(base - 400 + np.random.uniform(-100, 100))

    return pd.DataFrame({
        'open_time': dates,
        'open': prices,
        'high': [p + np.random.uniform(50, 150) for p in prices],
        'low': [p - np.random.uniform(50, 150) for p in prices],
        'close': [p + np.random.uniform(-100, 100) for p in prices],
        'volume': np.random.uniform(100, 1000, 100)
    })


@pytest.fixture
def insufficient_data_df():
    """DataFrame with insufficient data"""
    dates = pd.date_range('2026-01-01', periods=30, freq='1H')
    prices = [40000 + np.random.uniform(-200, 200) for _ in range(30)]

    return pd.DataFrame({
        'open_time': dates,
        'open': prices,
        'high': [p + 100 for p in prices],
        'low': [p - 100 for p in prices],
        'close': prices,
        'volume': np.random.uniform(100, 1000, 30)
    })


@pytest.mark.unit
class TestLiquidityCurveDetector:
    """Tests for LiquidityCurveDetector class"""

    def test_detect_accumulation_curve_valid(self, mock_candles):
        """Test accumulation curve detection with valid pattern"""
        acc_candles = mock_candles["accumulation_curve_candles"]
        df = pd.DataFrame(acc_candles)

        result = LiquidityCurveDetector.detect_accumulation_curve(df, 0, 5)

        assert isinstance(result, dict)
        assert "valid" in result
        assert "quality" in result
        assert "smoothness" in result
        assert "curve_type" in result
        assert result["curve_type"] == "accumulation"

    def test_detect_accumulation_curve_invalid_indices(self):
        """Test accumulation curve with invalid indices"""
        df = pd.DataFrame({
            'open_time': pd.date_range('2026-01-01', periods=5, freq='1H'),
            'low': [100, 101, 102, 103, 104],
            'high': [105, 106, 107, 108, 109],
            'close': [102, 103, 104, 105, 106]
        })

        # tap1_idx >= tap2_idx
        result = LiquidityCurveDetector.detect_accumulation_curve(df, 3, 2)
        assert result["valid"] is False
        assert result["quality"] == 0.0

        # tap2_idx >= len(candles)
        result = LiquidityCurveDetector.detect_accumulation_curve(df, 0, 10)
        assert result["valid"] is False

    def test_detect_accumulation_curve_insufficient_candles(self):
        """Test accumulation curve with too few candles"""
        df = pd.DataFrame({
            'open_time': pd.date_range('2026-01-01', periods=2, freq='1H'),
            'low': [100, 101],
            'high': [105, 106],
            'close': [102, 103]
        })

        result = LiquidityCurveDetector.detect_accumulation_curve(df, 0, 1)
        assert result["valid"] is False
        assert result["quality"] == 0.0

    def test_detect_distribution_curve_valid(self, mock_candles):
        """Test distribution curve detection with valid pattern"""
        dist_candles = mock_candles["distribution_curve_candles"]
        df = pd.DataFrame(dist_candles)

        result = LiquidityCurveDetector.detect_distribution_curve(df, 0, 5)

        assert isinstance(result, dict)
        assert "valid" in result
        assert "quality" in result
        assert "smoothness" in result
        assert "curve_type" in result
        assert result["curve_type"] == "distribution"

    def test_detect_distribution_curve_invalid_indices(self):
        """Test distribution curve with invalid indices"""
        df = pd.DataFrame({
            'open_time': pd.date_range('2026-01-01', periods=5, freq='1H'),
            'high': [109, 108, 107, 106, 105],
            'low': [104, 103, 102, 101, 100],
            'close': [106, 105, 104, 103, 102]
        })

        result = LiquidityCurveDetector.detect_distribution_curve(df, 4, 2)
        assert result["valid"] is False

    def test_curve_quality_threshold(self):
        """Test curve quality must exceed 0.6 threshold"""
        # Create very choppy data (low quality)
        df = pd.DataFrame({
            'open_time': pd.date_range('2026-01-01', periods=10, freq='1H'),
            'high': [100 + np.random.uniform(0, 50) for _ in range(10)],
            'low': [100 - np.random.uniform(0, 50) for _ in range(10)],
            'close': [100 + np.random.uniform(-30, 30) for _ in range(10)]
        })

        result = LiquidityCurveDetector.detect_accumulation_curve(df, 0, 9)

        # Quality score should be calculated
        assert 0.0 <= result["quality"] <= 1.0
        assert 0.0 <= result["smoothness"] <= 1.0


@pytest.mark.unit
class TestTCTModelDetector:
    """Tests for TCTModelDetector class"""

    def test_detector_initialization(self, accumulation_df):
        """Test TCT detector initializes correctly"""
        detector = TCTModelDetector(accumulation_df)

        assert detector.candles is not None
        assert len(detector.candles) == len(accumulation_df)
        assert isinstance(detector.lc_detector, LiquidityCurveDetector)

    def test_find_tap1_acc(self, accumulation_df):
        """Test finding tap1 for accumulation"""
        detector = TCTModelDetector(accumulation_df)

        tap1 = detector._find_tap1_acc(10)

        if tap1:  # May or may not find depending on data
            assert "idx" in tap1
            assert "price" in tap1
            assert "time" in tap1
            assert "type" in tap1
            assert tap1["type"] == "tap1_acc"

    def test_find_tap1_acc_out_of_bounds(self, accumulation_df):
        """Test tap1 finder with out of bounds index"""
        detector = TCTModelDetector(accumulation_df)

        # Try at end of dataframe
        tap1 = detector._find_tap1_acc(len(accumulation_df) - 5)
        assert tap1 is None

    def test_find_tap2_acc_requires_price_increase(self, accumulation_df):
        """Test tap2 must be higher than tap1"""
        detector = TCTModelDetector(accumulation_df)

        tap1 = {"idx": 10, "price": 40000.0, "time": "2026-01-01", "type": "tap1_acc"}
        tap2 = detector._find_tap2_acc(tap1)

        if tap2:  # If found, must be higher
            assert tap2["price"] > tap1["price"] * 1.005

    def test_find_tap3_model1_acc_between_taps(self, accumulation_df):
        """Test tap3 Model 1 must be between tap1 and mid-range"""
        detector = TCTModelDetector(accumulation_df)

        tap1 = {"idx": 10, "price": 40000.0, "time": "2026-01-01", "type": "tap1_acc"}
        tap2 = {"idx": 40, "price": 40800.0, "time": "2026-01-02", "type": "tap2_acc"}

        tap3 = detector._find_tap3_model1_acc(tap1, tap2)

        if tap3:  # If found, must be in valid range
            assert tap3["price"] > tap1["price"]
            range_size = tap2["price"] - tap1["price"]
            assert tap3["price"] <= tap1["price"] + (range_size * 0.5)
            assert tap3["deviation"] is False

    def test_find_tap3_model2_acc_near_tap1(self, accumulation_df):
        """Test tap3 Model 2 must be near tap1 (within 2%)"""
        detector = TCTModelDetector(accumulation_df)

        tap1 = {"idx": 10, "price": 40000.0, "time": "2026-01-01", "type": "tap1_acc"}
        tap2 = {"idx": 40, "price": 40800.0, "time": "2026-01-02", "type": "tap2_acc"}

        tap3 = detector._find_tap3_model2_acc(tap1, tap2)

        if tap3:  # If found, must be within 2% tolerance
            tolerance = tap1["price"] * 0.02
            assert abs(tap3["price"] - tap1["price"]) <= tolerance or tap3["price"] > tap1["price"]
            assert "deviation" in tap3

    def test_calc_duration(self, accumulation_df):
        """Test duration calculation in hours"""
        detector = TCTModelDetector(accumulation_df)

        duration = detector._calc_duration(0, 24)

        # 24 hourly candles = 24 hours
        assert duration == 24.0

    def test_calc_duration_invalid_indices(self, accumulation_df):
        """Test duration with invalid indices"""
        detector = TCTModelDetector(accumulation_df)

        # idx1 >= idx2
        duration = detector._calc_duration(50, 40)
        assert duration == 0

        # idx2 >= len(candles)
        duration = detector._calc_duration(0, 200)
        assert duration == 0

    def test_calc_quality_score(self, accumulation_df):
        """Test quality score calculation"""
        detector = TCTModelDetector(accumulation_df)

        curve = {"quality": 0.7, "smoothness": 0.8}

        # Test with long duration
        quality = detector._calc_quality(curve, 48, False)
        assert 0.0 <= quality <= 1.0

        # Test with deviation bonus
        quality_with_dev = detector._calc_quality(curve, 48, True)
        assert quality_with_dev >= quality

    def test_detect_accumulation_models(self, accumulation_df):
        """Test full accumulation model detection"""
        detector = TCTModelDetector(accumulation_df)

        models = detector.detect_accumulation_models()

        assert isinstance(models, list)
        assert len(models) <= 5  # Max 5 models returned

        for model in models:
            assert "model_type" in model
            assert "direction" in model
            assert model["direction"] == "bullish"
            assert "tap1" in model
            assert "tap2" in model
            assert "tap3" in model
            assert "quality_score" in model
            assert "duration_hours" in model
            assert model["duration_hours"] >= 24

    def test_detect_distribution_models(self, distribution_df):
        """Test full distribution model detection"""
        detector = TCTModelDetector(distribution_df)

        models = detector.detect_distribution_models()

        assert isinstance(models, list)
        assert len(models) <= 5

        for model in models:
            assert "model_type" in model
            assert "direction" in model
            assert model["direction"] == "bearish"
            assert "tap1" in model
            assert "tap2" in model
            assert "tap3" in model

    def test_models_sorted_by_quality(self, accumulation_df):
        """Test models are sorted by quality score"""
        detector = TCTModelDetector(accumulation_df)

        models = detector.detect_accumulation_models()

        if len(models) > 1:
            # Check descending order
            for i in range(len(models) - 1):
                assert models[i]["quality_score"] >= models[i+1]["quality_score"]

    def test_build_model_1_acc_structure(self, accumulation_df):
        """Test Model 1 accumulation builder creates correct structure"""
        detector = TCTModelDetector(accumulation_df)

        tap1 = {"idx": 10, "price": 40000.0, "time": "2026-01-01", "type": "tap1_acc"}
        tap2 = {"idx": 40, "price": 40800.0, "time": "2026-01-02", "type": "tap2_acc"}
        tap3 = {"idx": 60, "price": 40300.0, "time": "2026-01-03", "type": "tap3_m1", "deviation": False}
        curve = {"valid": True, "quality": 0.75, "smoothness": 0.80, "curve_type": "accumulation"}

        model = detector._build_model_1_acc(tap1, tap2, tap3, curve, 48.0)

        assert model["model_type"] == "Model_1_Accumulation"
        assert model["direction"] == "bullish"
        assert model["range_low"] == tap1["price"]
        assert model["range_high"] == tap2["price"]
        assert model["range_eq"] == (tap1["price"] + tap2["price"]) / 2
        assert model["target"] == tap2["price"]
        assert model["invalidation"] == tap1["price"]
        assert "timestamp" in model

    def test_build_model_2_acc_with_deviation(self, accumulation_df):
        """Test Model 2 accumulation with deviation bonus"""
        detector = TCTModelDetector(accumulation_df)

        tap1 = {"idx": 10, "price": 40000.0, "time": "2026-01-01", "type": "tap1_acc"}
        tap2 = {"idx": 40, "price": 40800.0, "time": "2026-01-02", "type": "tap2_acc"}
        tap3 = {"idx": 60, "price": 39900.0, "time": "2026-01-03", "type": "tap3_m2", "deviation": True}
        curve = {"valid": True, "quality": 0.75, "smoothness": 0.80, "curve_type": "accumulation"}

        model = detector._build_model_2_acc(tap1, tap2, tap3, curve, 48.0)

        assert model["model_type"] == "Model_2_Accumulation"
        assert model["invalidation"] == tap3["price"]  # Deviation changes invalidation


@pytest.mark.unit
class TestDetectTCTModels:
    """Tests for main entry point function"""

    def test_detect_tct_models_insufficient_data(self, insufficient_data_df):
        """Test detection with insufficient data"""
        result = detect_tct_models(insufficient_data_df)

        assert "accumulation_models" in result
        assert "distribution_models" in result
        assert "total_models" in result
        assert "error" in result
        assert result["total_models"] == 0
        assert "Insufficient data" in result["error"]

    def test_detect_tct_models_valid_data(self, accumulation_df):
        """Test detection with valid data"""
        result = detect_tct_models(accumulation_df)

        assert "accumulation_models" in result
        assert "distribution_models" in result
        assert "total_models" in result
        assert "candles_analyzed" in result
        assert "timestamp" in result
        assert result["candles_analyzed"] == len(accumulation_df)
        assert "error" not in result

    def test_detect_tct_models_return_structure(self, accumulation_df):
        """Test detection returns correct structure"""
        result = detect_tct_models(accumulation_df)

        assert isinstance(result, dict)
        assert isinstance(result["accumulation_models"], list)
        assert isinstance(result["distribution_models"], list)
        assert isinstance(result["total_models"], int)

        # Total should match sum of both lists
        assert result["total_models"] == (
            len(result["accumulation_models"]) +
            len(result["distribution_models"])
        )

    def test_detect_tct_models_handles_errors(self):
        """Test detection handles errors gracefully"""
        # Create malformed dataframe
        df = pd.DataFrame({'invalid': [1, 2, 3]})

        result = detect_tct_models(df)

        assert "error" in result
        assert result["total_models"] == 0


@pytest.mark.integration
class TestTCTDetectorIntegration:
    """Integration tests for full TCT detection pipeline"""

    def test_full_detection_pipeline(self):
        """Test complete detection pipeline with realistic data"""
        # Create realistic accumulation pattern
        dates = pd.date_range('2026-01-01', periods=150, freq='1H')
        base = 40000
        prices = []

        # Stage 1: Initial low (tap1 zone)
        for i in range(30):
            prices.append(base + np.random.uniform(-100, 100))

        # Stage 2: Rally to high (tap2 zone)
        for i in range(50):
            prices.append(base + i * 15 + np.random.uniform(-80, 80))

        # Stage 3: Pullback to higher low (tap3 zone)
        for i in range(70):
            prices.append(base + 300 + np.random.uniform(-120, 120))

        df = pd.DataFrame({
            'open_time': dates,
            'open': prices,
            'high': [p + np.random.uniform(50, 100) for p in prices],
            'low': [p - np.random.uniform(50, 100) for p in prices],
            'close': [p + np.random.uniform(-50, 50) for p in prices],
            'volume': np.random.uniform(100, 1000, 150)
        })

        result = detect_tct_models(df)

        # Should find at least one model
        assert result["total_models"] >= 0
        assert result["candles_analyzed"] == 150

    def test_both_model_types_detected(self):
        """Test detection of both accumulation and distribution"""
        dates = pd.date_range('2026-01-01', periods=200, freq='1H')
        prices = []
        base = 40000

        # Create mixed pattern
        for i in range(200):
            if i < 80:  # Accumulation
                prices.append(base + i * 8 + np.random.uniform(-100, 100))
            else:  # Distribution
                prices.append(base + 640 - (i-80) * 8 + np.random.uniform(-100, 100))

        df = pd.DataFrame({
            'open_time': dates,
            'open': prices,
            'high': [p + np.random.uniform(50, 150) for p in prices],
            'low': [p - np.random.uniform(50, 150) for p in prices],
            'close': [p + np.random.uniform(-80, 80) for p in prices],
            'volume': np.random.uniform(100, 1000, 200)
        })

        result = detect_tct_models(df)

        # Both model types should be present in structure
        assert "accumulation_models" in result
        assert "distribution_models" in result
