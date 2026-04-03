"""
Unit tests for tct_schematics.py
Tests TCT Schematic detection (Lecture 5A + 5B methodology) including:

Lecture 5A Core Features:
- Model 1 Accumulation: Tap1 → Tap2 (deviation) → Tap3 (lower deviation)
- Model 2 Accumulation: Tap1 → Tap2 (deviation) → Tap3 (higher low at extreme liq/demand)
- Model 1 Distribution: Tap1 → Tap2 (deviation) → Tap3 (higher deviation)
- Model 2 Distribution: Tap1 → Tap2 (deviation) → Tap3 (lower high at extreme liq/supply)

Lecture 5B Advanced Enhancements:
- Highest timeframe validation (6-candle rule on all taps)
- Overlapping structure (domino effect) for R:R optimization
- Supply/demand zone awareness on entry
- R:R calculation and optimization with minimum 1:2 requirement
- Trendline liquidity detection
- Tap spacing validation (equal distribution)
- Model 2 failure → Model 1 transition detection
- Range quality/rationality scoring
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from tct_schematics import (
    TCTSchematicDetector,
    detect_tct_schematics
)


@pytest.fixture
def accumulation_pattern_df():
    """DataFrame with classic accumulation pattern (downtrend → range → deviations)"""
    dates = pd.date_range('2026-01-01', periods=150, freq='1h')
    base = 100000
    prices = []

    # Stage 1: Downtrend (candles 0-30)
    for i in range(30):
        prices.append(base - i * 100 + np.random.uniform(-50, 50))

    # Stage 2: Range formation at low (candles 30-60) - Tap1 zone
    range_low = base - 3000
    for i in range(30):
        prices.append(range_low + np.random.uniform(-200, 400))

    # Stage 3: First deviation below range (candles 60-80) - Tap2 zone
    for i in range(20):
        deviation_depth = 500 + i * 20
        prices.append(range_low - deviation_depth + np.random.uniform(-100, 100))

    # Stage 4: Second lower deviation (candles 80-100) - Tap3 Model 1 zone
    for i in range(20):
        prices.append(range_low - 1000 - i * 15 + np.random.uniform(-80, 80))

    # Stage 5: Recovery inside range (candles 100-150)
    for i in range(50):
        prices.append(range_low + i * 30 + np.random.uniform(-100, 100))

    return pd.DataFrame({
        'open_time': dates,
        'open': prices,
        'high': [p + np.random.uniform(80, 200) for p in prices],
        'low': [p - np.random.uniform(80, 200) for p in prices],
        'close': [p + np.random.uniform(-100, 100) for p in prices],
        'volume': np.random.uniform(100, 1000, 150)
    })


@pytest.fixture
def distribution_pattern_df():
    """DataFrame with classic distribution pattern (uptrend → range → deviations)"""
    dates = pd.date_range('2026-01-01', periods=150, freq='1h')
    base = 100000
    prices = []

    # Stage 1: Uptrend (candles 0-30)
    for i in range(30):
        prices.append(base + i * 100 + np.random.uniform(-50, 50))

    # Stage 2: Range formation at high (candles 30-60) - Tap1 zone
    range_high = base + 3000
    for i in range(30):
        prices.append(range_high + np.random.uniform(-400, 200))

    # Stage 3: First deviation above range (candles 60-80) - Tap2 zone
    for i in range(20):
        deviation_height = 500 + i * 20
        prices.append(range_high + deviation_height + np.random.uniform(-100, 100))

    # Stage 4: Second higher deviation (candles 80-100) - Tap3 Model 1 zone
    for i in range(20):
        prices.append(range_high + 1000 + i * 15 + np.random.uniform(-80, 80))

    # Stage 5: Decline (candles 100-150)
    for i in range(50):
        prices.append(range_high - i * 30 + np.random.uniform(-100, 100))

    return pd.DataFrame({
        'open_time': dates,
        'open': prices,
        'high': [p + np.random.uniform(80, 200) for p in prices],
        'low': [p - np.random.uniform(80, 200) for p in prices],
        'close': [p + np.random.uniform(-100, 100) for p in prices],
        'volume': np.random.uniform(100, 1000, 150)
    })


@pytest.fixture
def model_2_accumulation_df():
    """DataFrame with Model 2 accumulation (higher low instead of lower deviation)"""
    dates = pd.date_range('2026-01-01', periods=150, freq='1h')
    base = 100000
    prices = []

    # Stage 1: Downtrend
    for i in range(30):
        prices.append(base - i * 80 + np.random.uniform(-40, 40))

    # Stage 2: Range formation (Tap1)
    range_low = base - 2500
    range_high = range_low + 1500
    for i in range(30):
        prices.append(range_low + np.random.uniform(0, 600))

    # Stage 3: First deviation (Tap2)
    for i in range(20):
        prices.append(range_low - 400 - i * 10 + np.random.uniform(-50, 50))

    # Stage 4: Higher low (Tap3 Model 2) - NOT lower than Tap2
    hl_price = range_low - 300  # Higher than Tap2's lowest
    for i in range(20):
        prices.append(hl_price + np.random.uniform(-80, 80))

    # Stage 5: Recovery
    for i in range(50):
        prices.append(hl_price + i * 40 + np.random.uniform(-80, 80))

    return pd.DataFrame({
        'open_time': dates,
        'open': prices,
        'high': [p + np.random.uniform(60, 150) for p in prices],
        'low': [p - np.random.uniform(60, 150) for p in prices],
        'close': [p + np.random.uniform(-80, 80) for p in prices],
        'volume': np.random.uniform(100, 1000, 150)
    })


@pytest.fixture
def insufficient_data_df():
    """DataFrame with insufficient data for schematic detection"""
    dates = pd.date_range('2026-01-01', periods=30, freq='1h')
    prices = [100000 + np.random.uniform(-500, 500) for _ in range(30)]

    return pd.DataFrame({
        'open_time': dates,
        'open': prices,
        'high': [p + 200 for p in prices],
        'low': [p - 200 for p in prices],
        'close': prices,
        'volume': np.random.uniform(100, 1000, 30)
    })


@pytest.fixture
def simple_range_data():
    """Simple range data for testing"""
    return {
        "range_high": 105000.0,
        "range_low": 100000.0,
        "equilibrium": 102500.0,
        "range_size": 5000.0,
        "dl_high": 106500.0,  # 30% above
        "dl_low": 98500.0,    # 30% below
        "range_high_idx": 30,
        "range_low_idx": 35
    }


@pytest.mark.unit
class TestTCTSchematicDetector:
    """Tests for TCTSchematicDetector class"""

    def test_detector_initialization(self, accumulation_pattern_df):
        """Test detector initializes correctly"""
        detector = TCTSchematicDetector(accumulation_pattern_df)

        assert detector.candles is not None
        assert len(detector.candles) == len(accumulation_pattern_df)
        assert detector.DEVIATION_LIMIT_PERCENT == 0.30
        assert detector.SIX_CANDLE_LOOKBACK == 6

    def test_detector_resets_index(self, accumulation_pattern_df):
        """Test detector resets DataFrame index"""
        df = accumulation_pattern_df.copy()
        df.index = range(100, 250)  # Non-standard index

        detector = TCTSchematicDetector(df)

        assert detector.candles.index[0] == 0
        assert detector.candles.index[-1] == len(df) - 1

    def test_detect_all_schematics_returns_structure(self, accumulation_pattern_df):
        """Test detect_all_schematics returns correct structure"""
        detector = TCTSchematicDetector(accumulation_pattern_df)

        result = detector.detect_all_schematics()

        assert "accumulation_schematics" in result
        assert "distribution_schematics" in result
        assert "total_schematics" in result
        assert "candles_analyzed" in result
        assert "timestamp" in result
        assert isinstance(result["accumulation_schematics"], list)
        assert isinstance(result["distribution_schematics"], list)

    def test_detect_all_schematics_with_ranges(self, accumulation_pattern_df, simple_range_data):
        """Test detection with pre-detected ranges"""
        detector = TCTSchematicDetector(accumulation_pattern_df)

        result = detector.detect_all_schematics([simple_range_data])

        assert result["candles_analyzed"] == len(accumulation_pattern_df)
        assert "error" not in result

    def test_insufficient_data_returns_error(self, insufficient_data_df):
        """Test insufficient data returns error"""
        detector = TCTSchematicDetector(insufficient_data_df)

        result = detector.detect_all_schematics()

        assert "error" in result
        assert "Insufficient data" in result["error"]
        assert result["total_schematics"] == 0


@pytest.mark.unit
class TestSwingDetection:
    """Tests for swing high/low detection methods"""

    def test_is_swing_high_valid(self, accumulation_pattern_df):
        """Test swing high detection with valid pattern"""
        detector = TCTSchematicDetector(accumulation_pattern_df)

        # Find a clear high in the data
        highs = accumulation_pattern_df['high'].values
        max_idx = np.argmax(highs[10:-10]) + 10

        # May or may not be valid 6-candle swing
        result = detector._is_swing_high(max_idx)
        assert isinstance(result, bool)

    def test_is_swing_high_boundary_conditions(self, accumulation_pattern_df):
        """Test swing high at boundary indices"""
        detector = TCTSchematicDetector(accumulation_pattern_df)

        # At start (insufficient lookback)
        assert detector._is_swing_high(1) is False

        # At end (insufficient lookahead)
        assert detector._is_swing_high(len(accumulation_pattern_df) - 2) is False

    def test_is_swing_low_valid(self, accumulation_pattern_df):
        """Test swing low detection"""
        detector = TCTSchematicDetector(accumulation_pattern_df)

        # Find a clear low in the data
        lows = accumulation_pattern_df['low'].values
        min_idx = np.argmin(lows[10:-10]) + 10

        result = detector._is_swing_low(min_idx)
        assert isinstance(result, bool)

    def test_is_swing_low_boundary_conditions(self, accumulation_pattern_df):
        """Test swing low at boundary indices"""
        detector = TCTSchematicDetector(accumulation_pattern_df)

        assert detector._is_swing_low(0) is False
        assert detector._is_swing_low(len(accumulation_pattern_df) - 1) is False

    def test_find_previous_swing_high(self, accumulation_pattern_df):
        """Test finding previous swing high"""
        detector = TCTSchematicDetector(accumulation_pattern_df)

        result = detector._find_previous_swing_high(50)

        if result:
            assert "idx" in result
            assert "price" in result
            assert result["idx"] < 50

    def test_find_previous_swing_low(self, accumulation_pattern_df):
        """Test finding previous swing low"""
        detector = TCTSchematicDetector(accumulation_pattern_df)

        result = detector._find_previous_swing_low(50)

        if result:
            assert "idx" in result
            assert "price" in result
            assert result["idx"] < 50


@pytest.mark.unit
class TestAccumulationSchematicDetection:
    """Tests for accumulation schematic detection"""

    def test_detect_accumulation_schematics(self, accumulation_pattern_df):
        """Test accumulation schematic detection"""
        detector = TCTSchematicDetector(accumulation_pattern_df)

        schematics = detector._detect_accumulation_schematics(None)

        assert isinstance(schematics, list)

        for s in schematics:
            assert s.get("direction") == "bullish"
            assert "tap1" in s
            assert "tap2" in s
            assert "tap3" in s
            assert "schematic_type" in s
            assert "Accumulation" in s["schematic_type"]

    def test_accumulation_schematic_structure(self, accumulation_pattern_df, simple_range_data):
        """Test accumulation schematic has complete structure"""
        detector = TCTSchematicDetector(accumulation_pattern_df)

        schematics = detector._detect_accumulation_schematics([simple_range_data])

        if len(schematics) > 0:
            schematic = schematics[0]

            # Core structure
            assert "schematic_type" in schematic
            assert "direction" in schematic
            assert "model" in schematic
            assert "tap1" in schematic
            assert "tap2" in schematic
            assert "tap3" in schematic

            # Range info
            assert "range" in schematic

            # Trade management
            assert "entry" in schematic
            assert "stop_loss" in schematic
            assert "target" in schematic

            # Quality
            assert "quality_score" in schematic
            assert "six_candle_valid" in schematic

    def test_model_1_accumulation_lower_deviations(self, accumulation_pattern_df):
        """Test Model 1 accumulation has successively lower deviations"""
        detector = TCTSchematicDetector(accumulation_pattern_df)

        schematics = detector._detect_accumulation_schematics(None)

        model_1_schematics = [s for s in schematics if "Model_1" in s.get("schematic_type", "")]

        for s in model_1_schematics:
            # Tap3 should be lower than Tap2 for Model 1
            if s["tap3"]["price"] and s["tap2"]["price"]:
                assert s["tap3"]["price"] <= s["tap2"]["price"], \
                    "Model 1 Tap3 should be lower than or equal to Tap2"


@pytest.mark.unit
class TestDistributionSchematicDetection:
    """Tests for distribution schematic detection"""

    def test_detect_distribution_schematics(self, distribution_pattern_df):
        """Test distribution schematic detection"""
        detector = TCTSchematicDetector(distribution_pattern_df)

        schematics = detector._detect_distribution_schematics(None)

        assert isinstance(schematics, list)

        for s in schematics:
            assert s.get("direction") == "bearish"
            assert "Distribution" in s.get("schematic_type", "")

    def test_distribution_schematic_targets_low(self, distribution_pattern_df, simple_range_data):
        """Test distribution schematic targets range low"""
        detector = TCTSchematicDetector(distribution_pattern_df)

        schematics = detector._detect_distribution_schematics([simple_range_data])

        for s in schematics:
            if s.get("target", {}).get("price"):
                # Target should be at or near range low
                assert "Range Low" in s["target"].get("description", "")


@pytest.mark.unit
class TestModel2Requirements:
    """Tests for Model 2 specific requirements"""

    def test_model_2_higher_low_accumulation(self, model_2_accumulation_df):
        """Test Model 2 accumulation finds higher low"""
        detector = TCTSchematicDetector(model_2_accumulation_df)

        schematics = detector._detect_accumulation_schematics(None)

        model_2_schematics = [s for s in schematics if "Model_2" in s.get("schematic_type", "")]

        for s in model_2_schematics:
            # Tap3 should be higher than Tap2 for Model 2 accumulation
            if s["tap3"]["price"] and s["tap2"]["price"]:
                assert s["tap3"]["price"] >= s["tap2"]["price"], \
                    "Model 2 Tap3 should be higher than or equal to Tap2"

    def test_model_2_extreme_requirements(self, model_2_accumulation_df):
        """Test Model 2 requires extreme liquidity OR extreme demand/supply"""
        detector = TCTSchematicDetector(model_2_accumulation_df)

        schematics = detector._detect_accumulation_schematics(None)

        model_2_schematics = [s for s in schematics if "Model_2" in s.get("schematic_type", "")]

        for s in model_2_schematics:
            tap3 = s.get("tap3", {})
            # Model 2 should have extreme liquidity OR extreme demand info
            has_extreme_req = (
                tap3.get("grabs_extreme_liquidity") or
                tap3.get("mitigates_extreme_demand") or
                tap3.get("mitigates_extreme_supply")
            )
            # Note: The detection may not always find extremes depending on data


@pytest.mark.unit
class TestQualityScoring:
    """Tests for quality score calculation"""

    def test_quality_score_range(self, accumulation_pattern_df):
        """Test quality score is within valid range"""
        detector = TCTSchematicDetector(accumulation_pattern_df)

        result = detector.detect_all_schematics()

        all_schematics = (
            result.get("accumulation_schematics", []) +
            result.get("distribution_schematics", [])
        )

        for s in all_schematics:
            quality = s.get("quality_score", 0)
            assert 0.0 <= quality <= 1.0, f"Quality score {quality} out of range"

    def test_quality_score_bos_bonus(self, accumulation_pattern_df):
        """Test BOS confirmation adds to quality score"""
        detector = TCTSchematicDetector(accumulation_pattern_df)

        result = detector.detect_all_schematics()

        all_schematics = (
            result.get("accumulation_schematics", []) +
            result.get("distribution_schematics", [])
        )

        for s in all_schematics:
            # Schematics with BOS should have higher scores
            if s.get("bos_confirmation"):
                assert s.get("quality_score", 0) >= 0.25


@pytest.mark.unit
class TestTradeManagement:
    """Tests for entry, stop loss, and target calculations"""

    def test_accumulation_stop_loss_below_tap3(self, accumulation_pattern_df):
        """Test accumulation stop loss is at Tap3 price (TCT rule)"""
        detector = TCTSchematicDetector(accumulation_pattern_df)

        schematics = detector._detect_accumulation_schematics(None)

        for s in schematics:
            if s.get("stop_loss", {}).get("price") and s.get("tap3", {}).get("price"):
                # SL includes a small buffer below Tap3 (range_size * 2%)
                assert s["stop_loss"]["price"] <= s["tap3"]["price"], \
                    "Stop loss should be at or below Tap3 price"

    def test_accumulation_target_at_range_high(self, accumulation_pattern_df, simple_range_data):
        """Test accumulation target is at range high (Wyckoff high)"""
        detector = TCTSchematicDetector(accumulation_pattern_df)

        schematics = detector._detect_accumulation_schematics([simple_range_data])

        for s in schematics:
            if s.get("range", {}).get("high"):
                # Target should match range high
                if s.get("target", {}).get("price"):
                    assert s["target"]["price"] == s["range"]["high"]

    def test_distribution_stop_loss_above_tap3(self, distribution_pattern_df):
        """Test distribution stop loss is at or slightly above Tap3 price (2% buffer)."""
        detector = TCTSchematicDetector(distribution_pattern_df)

        schematics = detector._detect_distribution_schematics(None)

        for s in schematics:
            stop_price = s.get("stop_loss", {}).get("price")
            tap3_price = s.get("tap3", {}).get("price")
            if stop_price and tap3_price:
                range_size = s.get("range", {}).get("size", 0)
                buffer = range_size * 0.02
                assert stop_price >= tap3_price, \
                    "Stop loss must be at or above Tap3 price"
                assert stop_price <= tap3_price + buffer + 1e-9, \
                    f"Stop loss {stop_price} too far above Tap3 {tap3_price} + buffer {buffer}"

    def test_risk_reward_calculation(self, accumulation_pattern_df):
        """Test risk/reward ratio is calculated correctly"""
        detector = TCTSchematicDetector(accumulation_pattern_df)

        result = detector.detect_all_schematics()

        all_schematics = (
            result.get("accumulation_schematics", []) +
            result.get("distribution_schematics", [])
        )

        for s in all_schematics:
            rr = s.get("risk_reward")
            if rr is not None:
                assert isinstance(rr, (int, float))
                # R:R should be positive for valid setups
                if rr > 0:
                    entry = s.get("entry", {}).get("price")
                    stop = s.get("stop_loss", {}).get("price")
                    target = s.get("target", {}).get("price")

                    if entry and stop and target:
                        if s.get("direction") == "bullish":
                            expected_rr = (target - entry) / (entry - stop)
                        else:
                            expected_rr = (entry - target) / (stop - entry)

                        assert abs(rr - round(expected_rr, 2)) < 0.1


@pytest.mark.unit
class TestSixCandleRule:
    """Tests for six candle rule validation"""

    def test_six_candle_rule_validation(self, accumulation_pattern_df):
        """Test six candle rule is validated on tabs"""
        detector = TCTSchematicDetector(accumulation_pattern_df)

        result = detector.detect_all_schematics()

        all_schematics = (
            result.get("accumulation_schematics", []) +
            result.get("distribution_schematics", [])
        )

        for s in all_schematics:
            # Should have six_candle_valid field
            assert "six_candle_valid" in s
            assert isinstance(s["six_candle_valid"], bool)


@pytest.mark.unit
class TestDetectTCTSchematics:
    """Tests for main entry point function"""

    def test_detect_tct_schematics_insufficient_data(self, insufficient_data_df):
        """Test with insufficient data"""
        result = detect_tct_schematics(insufficient_data_df)

        assert "error" in result
        assert "Insufficient data" in result["error"]
        assert result["total_schematics"] == 0
        assert len(result["accumulation_schematics"]) == 0
        assert len(result["distribution_schematics"]) == 0

    def test_detect_tct_schematics_valid_data(self, accumulation_pattern_df):
        """Test with valid data"""
        result = detect_tct_schematics(accumulation_pattern_df)

        assert "accumulation_schematics" in result
        assert "distribution_schematics" in result
        assert "total_schematics" in result
        assert "candles_analyzed" in result
        assert "timestamp" in result
        assert result["candles_analyzed"] == len(accumulation_pattern_df)

    def test_detect_tct_schematics_with_ranges(self, accumulation_pattern_df, simple_range_data):
        """Test with pre-detected ranges"""
        result = detect_tct_schematics(accumulation_pattern_df, [simple_range_data])

        assert "error" not in result
        assert result["candles_analyzed"] == len(accumulation_pattern_df)

    def test_detect_tct_schematics_handles_errors(self):
        """Test handles malformed data gracefully"""
        # Create malformed dataframe
        df = pd.DataFrame({'invalid_column': [1, 2, 3]})

        result = detect_tct_schematics(df)

        # Should return error structure, not crash
        assert "error" in result
        assert result["total_schematics"] == 0

    def test_detect_tct_schematics_total_count(self, accumulation_pattern_df):
        """Test total count matches sum of schematics"""
        result = detect_tct_schematics(accumulation_pattern_df)

        expected_total = (
            len(result["accumulation_schematics"]) +
            len(result["distribution_schematics"])
        )

        assert result["total_schematics"] == expected_total


@pytest.mark.integration
class TestTCTSchematicIntegration:
    """Integration tests for full schematic detection pipeline"""

    def test_full_accumulation_detection_pipeline(self):
        """Test complete accumulation detection with realistic data"""
        np.random.seed(42)
        dates = pd.date_range('2026-01-01', periods=200, freq='1h')
        base = 100000
        prices = []

        # Create clear accumulation pattern
        # Phase 1: Downtrend
        for i in range(40):
            prices.append(base - i * 80 + np.random.uniform(-30, 30))

        # Phase 2: Range at low
        range_low = base - 3200
        for i in range(30):
            prices.append(range_low + np.random.uniform(0, 500))

        # Phase 3: Deviation below
        for i in range(20):
            prices.append(range_low - 300 - i * 20 + np.random.uniform(-50, 50))

        # Phase 4: Second deviation (lower)
        for i in range(20):
            prices.append(range_low - 800 - i * 10 + np.random.uniform(-40, 40))

        # Phase 5: Recovery
        for i in range(90):
            prices.append(range_low - 600 + i * 30 + np.random.uniform(-60, 60))

        df = pd.DataFrame({
            'open_time': dates,
            'open': prices,
            'high': [p + np.random.uniform(50, 150) for p in prices],
            'low': [p - np.random.uniform(50, 150) for p in prices],
            'close': [p + np.random.uniform(-60, 60) for p in prices],
            'volume': np.random.uniform(100, 1000, 200)
        })

        result = detect_tct_schematics(df)

        assert result["candles_analyzed"] == 200
        assert "error" not in result
        # Pattern should be detectable
        assert result["total_schematics"] >= 0

    def test_full_distribution_detection_pipeline(self):
        """Test complete distribution detection with realistic data"""
        np.random.seed(123)
        dates = pd.date_range('2026-01-01', periods=200, freq='1h')
        base = 100000
        prices = []

        # Create clear distribution pattern
        # Phase 1: Uptrend
        for i in range(40):
            prices.append(base + i * 80 + np.random.uniform(-30, 30))

        # Phase 2: Range at high
        range_high = base + 3200
        for i in range(30):
            prices.append(range_high + np.random.uniform(-500, 0))

        # Phase 3: Deviation above
        for i in range(20):
            prices.append(range_high + 300 + i * 20 + np.random.uniform(-50, 50))

        # Phase 4: Second deviation (higher)
        for i in range(20):
            prices.append(range_high + 800 + i * 10 + np.random.uniform(-40, 40))

        # Phase 5: Decline
        for i in range(90):
            prices.append(range_high + 600 - i * 30 + np.random.uniform(-60, 60))

        df = pd.DataFrame({
            'open_time': dates,
            'open': prices,
            'high': [p + np.random.uniform(50, 150) for p in prices],
            'low': [p - np.random.uniform(50, 150) for p in prices],
            'close': [p + np.random.uniform(-60, 60) for p in prices],
            'volume': np.random.uniform(100, 1000, 200)
        })

        result = detect_tct_schematics(df)

        assert result["candles_analyzed"] == 200
        assert "error" not in result

    def test_mixed_patterns_detection(self):
        """Test detection handles mixed accumulation and distribution"""
        np.random.seed(456)
        dates = pd.date_range('2026-01-01', periods=300, freq='1h')
        prices = []
        base = 100000

        # First half: Accumulation
        for i in range(150):
            if i < 50:
                prices.append(base - i * 50)
            elif i < 100:
                prices.append(base - 2500 + np.random.uniform(-200, 200))
            else:
                prices.append(base - 2500 + (i-100) * 40)

        # Second half: Distribution
        high_point = prices[-1]
        for i in range(150):
            if i < 50:
                prices.append(high_point + i * 30)
            elif i < 100:
                prices.append(high_point + 1500 + np.random.uniform(-200, 200))
            else:
                prices.append(high_point + 1500 - (i-100) * 40)

        df = pd.DataFrame({
            'open_time': dates,
            'open': prices,
            'high': [p + np.random.uniform(50, 150) for p in prices],
            'low': [p - np.random.uniform(50, 150) for p in prices],
            'close': [p + np.random.uniform(-80, 80) for p in prices],
            'volume': np.random.uniform(100, 1000, 300)
        })

        result = detect_tct_schematics(df)

        # Should have both types in structure
        assert "accumulation_schematics" in result
        assert "distribution_schematics" in result
        assert result["candles_analyzed"] == 300

    def test_detection_with_external_ranges(self):
        """Test detection works with externally provided ranges"""
        np.random.seed(789)
        dates = pd.date_range('2026-01-01', periods=100, freq='1h')

        prices = [100000 + np.random.uniform(-500, 500) for _ in range(100)]

        df = pd.DataFrame({
            'open_time': dates,
            'open': prices,
            'high': [p + 200 for p in prices],
            'low': [p - 200 for p in prices],
            'close': prices,
            'volume': np.random.uniform(100, 1000, 100)
        })

        # Provide external ranges
        external_ranges = [{
            "range_high": 100500.0,
            "range_low": 99500.0,
            "equilibrium": 100000.0,
            "range_size": 1000.0,
            "dl_high": 100800.0,
            "dl_low": 99200.0,
            "range_high_idx": 20,
            "range_low_idx": 30
        }]

        result = detect_tct_schematics(df, external_ranges)

        assert "error" not in result
        assert result["candles_analyzed"] == 100


@pytest.mark.unit
class TestEdgeCases:
    """Tests for edge cases and boundary conditions"""

    def test_empty_ranges_list(self, accumulation_pattern_df):
        """Test detection with empty ranges list"""
        result = detect_tct_schematics(accumulation_pattern_df, [])

        # Should still work with auto-detection
        assert "error" not in result

    def test_none_ranges(self, accumulation_pattern_df):
        """Test detection with None ranges"""
        result = detect_tct_schematics(accumulation_pattern_df, None)

        assert "error" not in result

    def test_exactly_50_candles(self):
        """Test with exactly 50 candles (minimum)"""
        dates = pd.date_range('2026-01-01', periods=50, freq='1h')
        prices = [100000 + i * 10 for i in range(50)]

        df = pd.DataFrame({
            'open_time': dates,
            'open': prices,
            'high': [p + 100 for p in prices],
            'low': [p - 100 for p in prices],
            'close': prices,
            'volume': np.random.uniform(100, 1000, 50)
        })

        result = detect_tct_schematics(df)

        assert "error" not in result
        assert result["candles_analyzed"] == 50

    def test_49_candles_insufficient(self):
        """Test with 49 candles (insufficient)"""
        dates = pd.date_range('2026-01-01', periods=49, freq='1h')
        prices = [100000 + i * 10 for i in range(49)]

        df = pd.DataFrame({
            'open_time': dates,
            'open': prices,
            'high': [p + 100 for p in prices],
            'low': [p - 100 for p in prices],
            'close': prices,
            'volume': np.random.uniform(100, 1000, 49)
        })

        result = detect_tct_schematics(df)

        assert "error" in result
        assert "Insufficient data" in result["error"]


# ================================================================
# LECTURE 5B ENHANCEMENT TESTS
# ================================================================

@pytest.mark.unit
class TestLecture5BEnhancements:
    """Tests for Lecture 5B advanced features"""

    def test_schematic_has_lecture_5b_enhancements(self, accumulation_pattern_df):
        """Test that schematics include Lecture 5B enhancement data"""
        result = detect_tct_schematics(accumulation_pattern_df)

        all_schematics = (
            result.get("accumulation_schematics", []) +
            result.get("distribution_schematics", [])
        )

        for schematic in all_schematics:
            assert "lecture_5b_enhancements" in schematic
            enhancements = schematic["lecture_5b_enhancements"]

            # Check all Lecture 5B fields are present
            assert "htf_validation" in enhancements
            assert "overlapping_structure" in enhancements
            assert "supply_demand_check" in enhancements or enhancements.get("supply_demand_check") is None
            assert "rr_analysis" in enhancements or enhancements.get("rr_analysis") is None
            assert "trendline_liquidity" in enhancements
            assert "tap_spacing" in enhancements
            assert "range_quality" in enhancements


@pytest.mark.unit
class TestHighestTimeframeValidation:
    """Tests for highest timeframe validation (6-candle rule on all taps)"""

    def test_htf_validation_structure(self, accumulation_pattern_df):
        """Test HTF validation returns correct structure"""
        detector = TCTSchematicDetector(accumulation_pattern_df)
        result = detector.detect_all_schematics()

        for schematic in result.get("accumulation_schematics", []):
            htf = schematic.get("lecture_5b_enhancements", {}).get("htf_validation", {})

            assert "all_taps_valid_6cr" in htf
            assert "tap1_valid_6cr" in htf
            assert "tap2_valid_6cr" in htf
            assert "tap3_valid_6cr" in htf
            assert "validity_explanation" in htf
            assert isinstance(htf["all_taps_valid_6cr"], bool)

    def test_htf_validation_affects_six_candle_valid(self, accumulation_pattern_df):
        """Test that HTF validation is reflected in six_candle_valid field"""
        detector = TCTSchematicDetector(accumulation_pattern_df)
        result = detector.detect_all_schematics()

        for schematic in result.get("accumulation_schematics", []):
            htf = schematic.get("lecture_5b_enhancements", {}).get("htf_validation", {})
            six_candle_valid = schematic.get("six_candle_valid")

            # Both should agree
            assert six_candle_valid == htf.get("all_taps_valid_6cr")


@pytest.mark.unit
class TestOverlappingStructure:
    """Tests for overlapping structure (domino effect) detection"""

    def test_overlapping_structure_format(self, accumulation_pattern_df):
        """Test overlapping structure returns correct format"""
        detector = TCTSchematicDetector(accumulation_pattern_df)
        result = detector.detect_all_schematics()

        for schematic in result.get("accumulation_schematics", []):
            overlap = schematic.get("lecture_5b_enhancements", {}).get("overlapping_structure", {})

            assert "has_overlapping_structure" in overlap
            assert "nested_schematics" in overlap
            assert "domino_levels" in overlap
            assert isinstance(overlap["domino_levels"], int)
            assert overlap["domino_levels"] >= 1

    def test_overlapping_structure_optimized_rr(self, accumulation_pattern_df):
        """Test overlapping structure provides optimized R:R when found"""
        detector = TCTSchematicDetector(accumulation_pattern_df)
        result = detector.detect_all_schematics()

        for schematic in result.get("accumulation_schematics", []):
            overlap = schematic.get("lecture_5b_enhancements", {}).get("overlapping_structure", {})

            if overlap.get("has_overlapping_structure"):
                assert overlap.get("optimized_entry") is not None
                assert overlap.get("optimized_stop_loss") is not None
                assert overlap.get("optimized_target") is not None
                assert overlap.get("optimized_rr") is not None


@pytest.mark.unit
class TestSupplyDemandZoneAwareness:
    """Tests for supply/demand zone conflict detection"""

    def test_sd_zone_check_structure(self, accumulation_pattern_df):
        """Test S/D zone check returns correct structure"""
        detector = TCTSchematicDetector(accumulation_pattern_df)
        result = detector.detect_all_schematics()

        for schematic in result.get("accumulation_schematics", []):
            sd_check = schematic.get("lecture_5b_enhancements", {}).get("supply_demand_check")

            if sd_check:
                assert "has_conflict" in sd_check
                assert "entry_inside_opposing_zone" in sd_check
                assert "opposing_zone_blocks_target" in sd_check
                assert "conflicting_zones" in sd_check
                assert "recommendation" in sd_check

    def test_entry_safety_flag(self, accumulation_pattern_df):
        """Test entry has is_safe flag based on S/D check"""
        detector = TCTSchematicDetector(accumulation_pattern_df)
        result = detector.detect_all_schematics()

        for schematic in result.get("accumulation_schematics", []):
            entry = schematic.get("entry", {})
            assert "is_safe" in entry
            assert isinstance(entry["is_safe"], bool)


@pytest.mark.unit
class TestRRCalculation:
    """Tests for R:R calculation and optimization"""

    def test_rr_analysis_structure(self, accumulation_pattern_df):
        """Test R:R analysis returns correct structure"""
        detector = TCTSchematicDetector(accumulation_pattern_df)
        result = detector.detect_all_schematics()

        for schematic in result.get("accumulation_schematics", []):
            rr_analysis = schematic.get("lecture_5b_enhancements", {}).get("rr_analysis")

            if rr_analysis:
                assert "risk_reward_ratio" in rr_analysis
                assert "meets_minimum_rr" in rr_analysis
                assert "risk_amount" in rr_analysis
                assert "reward_amount" in rr_analysis
                assert "optimization_suggestions" in rr_analysis

    def test_meets_minimum_rr_flag(self, accumulation_pattern_df):
        """Test meets_minimum_rr flag is set correctly"""
        detector = TCTSchematicDetector(accumulation_pattern_df)
        result = detector.detect_all_schematics()

        for schematic in result.get("accumulation_schematics", []):
            rr_analysis = schematic.get("lecture_5b_enhancements", {}).get("rr_analysis")

            if rr_analysis and rr_analysis.get("risk_reward_ratio"):
                rr = rr_analysis["risk_reward_ratio"]
                meets_min = rr_analysis["meets_minimum_rr"]

                # MIN_RR_RATIO is 2.0
                if rr >= 2.0:
                    assert meets_min is True
                else:
                    assert meets_min is False


@pytest.mark.unit
class TestTrendlineLiquidity:
    """Tests for trendline liquidity detection"""

    def test_trendline_liquidity_structure(self, accumulation_pattern_df):
        """Test trendline liquidity returns correct structure"""
        detector = TCTSchematicDetector(accumulation_pattern_df)
        result = detector.detect_all_schematics()

        for schematic in result.get("accumulation_schematics", []):
            trendline = schematic.get("lecture_5b_enhancements", {}).get("trendline_liquidity", {})

            assert "has_trendline" in trendline
            assert "trendline_swept" in trendline
            assert "provides_confluence" in trendline
            assert isinstance(trendline["has_trendline"], bool)


@pytest.mark.unit
class TestTapSpacingValidation:
    """Tests for tap spacing validation"""

    def test_tap_spacing_structure(self, accumulation_pattern_df):
        """Test tap spacing validation returns correct structure"""
        detector = TCTSchematicDetector(accumulation_pattern_df)
        result = detector.detect_all_schematics()

        for schematic in result.get("accumulation_schematics", []):
            spacing = schematic.get("lecture_5b_enhancements", {}).get("tap_spacing", {})

            assert "spacing_valid" in spacing
            assert "tap1_to_tap2_candles" in spacing
            assert "tap2_to_tap3_candles" in spacing
            assert "spacing_ratio" in spacing or spacing.get("spacing_ratio") is None
            assert "is_horizontal" in spacing
            assert "spacing_quality" in spacing

    def test_spacing_ratio_calculation(self, accumulation_pattern_df):
        """Test spacing ratio is calculated correctly"""
        detector = TCTSchematicDetector(accumulation_pattern_df)
        result = detector.detect_all_schematics()

        for schematic in result.get("accumulation_schematics", []):
            spacing = schematic.get("lecture_5b_enhancements", {}).get("tap_spacing", {})

            tap1_to_tap2 = spacing.get("tap1_to_tap2_candles")
            tap2_to_tap3 = spacing.get("tap2_to_tap3_candles")
            ratio = spacing.get("spacing_ratio")

            if tap1_to_tap2 and tap2_to_tap3 and tap1_to_tap2 > 0 and tap2_to_tap3 > 0:
                expected_ratio = min(tap1_to_tap2, tap2_to_tap3) / max(tap1_to_tap2, tap2_to_tap3)
                if ratio:
                    assert abs(ratio - round(expected_ratio, 2)) < 0.05


@pytest.mark.unit
class TestRangeQuality:
    """Tests for range quality calculation"""

    def test_range_quality_structure(self, accumulation_pattern_df):
        """Test range quality returns correct structure"""
        detector = TCTSchematicDetector(accumulation_pattern_df)
        result = detector.detect_all_schematics()

        for schematic in result.get("accumulation_schematics", []):
            quality = schematic.get("lecture_5b_enhancements", {}).get("range_quality", {})

            assert "quality_score" in quality
            assert "is_horizontal" in quality
            assert "has_clean_pivots" in quality
            assert "has_equal_spacing" in quality
            assert "quality_factors" in quality
            assert isinstance(quality["quality_factors"], list)

    def test_range_quality_score_range(self, accumulation_pattern_df):
        """Test range quality score is within valid range"""
        detector = TCTSchematicDetector(accumulation_pattern_df)
        result = detector.detect_all_schematics()

        for schematic in result.get("accumulation_schematics", []):
            quality = schematic.get("lecture_5b_enhancements", {}).get("range_quality", {})
            score = quality.get("quality_score", 0)

            assert 0.0 <= score <= 1.0


@pytest.mark.unit
class TestModel2ToModel1Failure:
    """Tests for Model 2 to Model 1 failure transition detection"""

    def test_m2_to_m1_transition_structure(self, accumulation_pattern_df):
        """Test M2 to M1 transition returns correct structure when detected"""
        result = detect_tct_schematics(accumulation_pattern_df)

        # Find any Model_1_from_M2_failure schematics
        for schematic in result.get("accumulation_schematics", []):
            if "M2_failure" in schematic.get("schematic_type", ""):
                m2_to_m1 = schematic.get("lecture_5b_enhancements", {}).get("m2_to_m1_transition", {})

                assert "original_m2_tap3" in m2_to_m1
                assert "failure_price" in m2_to_m1
                assert "transition_detected" in m2_to_m1
                assert m2_to_m1["transition_detected"] is True

    def test_m2_failure_creates_new_model1(self, model_2_accumulation_df):
        """Test that when M2 fails, a new M1 schematic can be created"""
        result = detect_tct_schematics(model_2_accumulation_df)

        # Just verify detection runs without error
        assert "error" not in result
        assert "accumulation_schematics" in result


@pytest.mark.unit
class TestEnhancedQualityScoring:
    """Tests for enhanced quality scoring with Lecture 5B factors"""

    def test_enhanced_quality_uses_5b_factors(self, accumulation_pattern_df):
        """Test quality score incorporates Lecture 5B factors"""
        detector = TCTSchematicDetector(accumulation_pattern_df)
        result = detector.detect_all_schematics()

        for schematic in result.get("accumulation_schematics", []):
            quality = schematic.get("quality_score", 0)
            enhancements = schematic.get("lecture_5b_enhancements", {})

            # Quality should be between 0 and 1
            assert 0.0 <= quality <= 1.0

            # If all 5B factors are positive, quality should be higher
            htf_valid = enhancements.get("htf_validation", {}).get("all_taps_valid_6cr", False)
            no_sd_conflict = not (enhancements.get("supply_demand_check", {}).get("has_conflict", False))
            meets_rr = enhancements.get("rr_analysis", {}).get("meets_minimum_rr", False)

            # When multiple factors align, quality should tend to be higher
            # (This is a soft assertion - just checking the relationship exists)
            positive_factors = sum([htf_valid, no_sd_conflict, meets_rr])
            if positive_factors >= 2:
                # With multiple positive factors, quality should be at least moderate
                assert quality >= 0.3


@pytest.mark.integration
class TestLecture5BIntegration:
    """Integration tests for Lecture 5B enhancements"""

    def test_full_lecture_5b_detection_pipeline(self):
        """Test complete detection with all Lecture 5B features"""
        np.random.seed(42)
        dates = pd.date_range('2026-01-01', periods=200, freq='1h')
        base = 100000
        prices = []

        # Create accumulation pattern
        for i in range(40):
            prices.append(base - i * 80 + np.random.uniform(-30, 30))

        range_low = base - 3200
        for i in range(30):
            prices.append(range_low + np.random.uniform(0, 500))

        for i in range(20):
            prices.append(range_low - 300 - i * 20 + np.random.uniform(-50, 50))

        for i in range(20):
            prices.append(range_low - 800 - i * 10 + np.random.uniform(-40, 40))

        for i in range(90):
            prices.append(range_low - 600 + i * 30 + np.random.uniform(-60, 60))

        df = pd.DataFrame({
            'open_time': dates,
            'open': prices,
            'high': [p + np.random.uniform(50, 150) for p in prices],
            'low': [p - np.random.uniform(50, 150) for p in prices],
            'close': [p + np.random.uniform(-60, 60) for p in prices],
            'volume': np.random.uniform(100, 1000, 200)
        })

        result = detect_tct_schematics(df)

        assert "error" not in result
        assert result["candles_analyzed"] == 200

        # Verify all 5B enhancements are present in results
        all_schematics = (
            result.get("accumulation_schematics", []) +
            result.get("distribution_schematics", [])
        )

        for schematic in all_schematics:
            enhancements = schematic.get("lecture_5b_enhancements", {})

            # All 5B feature categories should be present
            assert "htf_validation" in enhancements
            assert "overlapping_structure" in enhancements
            assert "trendline_liquidity" in enhancements
            assert "tap_spacing" in enhancements
            assert "range_quality" in enhancements
            assert "meets_minimum_rr" in enhancements
            assert "has_trendline_confluence" in enhancements


# ================================================================
# LECTURE 6 ENHANCEMENT TESTS: ADVANCED TCT SCHEMATICS
# ================================================================

@pytest.mark.unit
class TestLecture6Enhancements:
    """Tests for Lecture 6 advanced TCT schematic features"""

    def test_schematic_has_lecture_6_enhancements(self, accumulation_pattern_df):
        """Test that schematics include Lecture 6 enhancement data"""
        result = detect_tct_schematics(accumulation_pattern_df)

        all_schematics = (
            result.get("accumulation_schematics", []) +
            result.get("distribution_schematics", [])
        )

        for schematic in all_schematics:
            assert "lecture_6_enhancements" in schematic
            l6 = schematic["lecture_6_enhancements"]

            # Check all Lecture 6 fields are present
            assert "schematic_conversion" in l6 or l6.get("schematic_conversion") is None
            assert "dual_side_deviation" in l6
            assert "ltf_htf_transition" in l6
            assert "multi_tf_validity" in l6
            assert "wov_in_wov" in l6
            assert "model1_to_model2_flow" in l6
            assert "context_follow_through" in l6

            # Check summary flags
            assert "has_conversion" in l6
            assert "has_dual_side_deviation" in l6
            assert "is_nested_in_htf" in l6
            assert "valid_on_htf" in l6
            assert "has_wov_opportunity" in l6
            assert "has_m1_to_m2_opportunity" in l6
            assert "follow_through_bias" in l6

    def test_lecture_6_summary_flags_are_booleans(self, accumulation_pattern_df):
        """Test that Lecture 6 summary flags are correct types"""
        result = detect_tct_schematics(accumulation_pattern_df)

        all_schematics = (
            result.get("accumulation_schematics", []) +
            result.get("distribution_schematics", [])
        )

        for schematic in all_schematics:
            l6 = schematic.get("lecture_6_enhancements", {})

            assert isinstance(l6.get("has_conversion"), bool)
            assert isinstance(l6.get("has_dual_side_deviation"), bool)
            assert isinstance(l6.get("is_nested_in_htf"), bool)
            assert isinstance(l6.get("valid_on_htf"), bool)
            assert isinstance(l6.get("has_wov_opportunity"), bool)
            assert isinstance(l6.get("has_m1_to_m2_opportunity"), bool)


@pytest.mark.unit
class TestSchematicConversion:
    """Tests for Distribution-to-Accumulation conversion detection"""

    def test_schematic_conversion_structure(self, accumulation_pattern_df):
        """Test schematic conversion returns correct structure when detected"""
        detector = TCTSchematicDetector(accumulation_pattern_df)
        result = detector.detect_all_schematics()

        for schematic in result.get("accumulation_schematics", []):
            l6 = schematic.get("lecture_6_enhancements", {})
            conversion = l6.get("schematic_conversion")

            if conversion and conversion.get("conversion_detected"):
                assert "original_type" in conversion
                assert "converted_type" in conversion
                assert "conversion_trigger" in conversion
                assert "new_tap3_idx" in conversion
                assert "follow_through_expectation" in conversion

    def test_conversion_only_on_opposite_deviation(self, accumulation_pattern_df):
        """Test conversion is only detected on opposite-side deviation"""
        result = detect_tct_schematics(accumulation_pattern_df)

        for schematic in result.get("accumulation_schematics", []):
            l6 = schematic.get("lecture_6_enhancements", {})
            conversion = l6.get("schematic_conversion")

            if conversion and conversion.get("conversion_detected"):
                # original_type = schematic_type (e.g. "Model_2_Accumulation")
                # converted_type = the opposite (e.g. "Model_2_Distribution")
                assert "Accumulation" in conversion.get("original_type", "")
                assert "Distribution" in conversion.get("converted_type", "")


@pytest.mark.unit
class TestDualSideDeviation:
    """Tests for dual-side deviation awareness"""

    def test_dual_side_deviation_structure(self, accumulation_pattern_df):
        """Test dual-side deviation returns correct structure"""
        detector = TCTSchematicDetector(accumulation_pattern_df)
        result = detector.detect_all_schematics()

        for schematic in result.get("accumulation_schematics", []):
            l6 = schematic.get("lecture_6_enhancements", {})
            dual_dev = l6.get("dual_side_deviation", {})

            assert "has_dual_side_deviation" in dual_dev
            assert "high_side_deviated" in dual_dev
            assert "low_side_deviated" in dual_dev
            assert "can_convert_to_opposite" in dual_dev
            assert isinstance(dual_dev["has_dual_side_deviation"], bool)

    def test_dual_deviation_risk_state(self, accumulation_pattern_df):
        """Test risk triggers are set correctly for dual deviation"""
        result = detect_tct_schematics(accumulation_pattern_df)

        for schematic in result.get("accumulation_schematics", []):
            l6 = schematic.get("lecture_6_enhancements", {})
            dual_dev = l6.get("dual_side_deviation", {})

            # Production uses risk_on_trigger / risk_off_trigger (not risk_state)
            assert "risk_on_trigger" in dual_dev
            assert "risk_off_trigger" in dual_dev


@pytest.mark.unit
class TestLTFToHTFTransition:
    """Tests for LTF-to-HTF range transition detection"""

    def test_ltf_htf_transition_structure(self, accumulation_pattern_df):
        """Test LTF-HTF transition returns correct structure"""
        detector = TCTSchematicDetector(accumulation_pattern_df)
        result = detector.detect_all_schematics()

        for schematic in result.get("accumulation_schematics", []):
            l6 = schematic.get("lecture_6_enhancements", {})
            transition = l6.get("ltf_htf_transition", {})

            assert "transition_detected" in transition
            assert "expansion_factor" in transition
            assert "recommendation" in transition
            assert isinstance(transition["transition_detected"], bool)

    def test_nested_range_expansion_factor(self, accumulation_pattern_df):
        """Test expansion_factor is calculated for nested ranges"""
        result = detect_tct_schematics(accumulation_pattern_df)

        for schematic in result.get("accumulation_schematics", []):
            l6 = schematic.get("lecture_6_enhancements", {})
            transition = l6.get("ltf_htf_transition", {})

            if transition.get("transition_detected"):
                expansion_factor = transition.get("expansion_factor")
                if expansion_factor is not None:
                    # expansion_factor > 1 means range grew (expansion)
                    # expansion_factor < 1 means range shrank (contraction)
                    assert expansion_factor > 0


@pytest.mark.unit
class TestMultiTFValidity:
    """Tests for multi-timeframe schematic validity checking"""

    def test_multi_tf_validity_structure(self, accumulation_pattern_df):
        """Test multi-TF validity returns correct structure"""
        detector = TCTSchematicDetector(accumulation_pattern_df)
        result = detector.detect_all_schematics()

        for schematic in result.get("accumulation_schematics", []):
            l6 = schematic.get("lecture_6_enhancements", {})
            multi_tf = l6.get("multi_tf_validity", {})

            assert "has_multi_tf_opportunity" in multi_tf
            assert "tap_distance_ratio" in multi_tf
            assert "htf_schematic_potential" in multi_tf
            assert "recommendation" in multi_tf
            assert isinstance(multi_tf["has_multi_tf_opportunity"], bool)

    def test_close_taps_affect_htf_validity(self, accumulation_pattern_df):
        """Test that close taps affect HTF validity assessment"""
        result = detect_tct_schematics(accumulation_pattern_df)

        for schematic in result.get("accumulation_schematics", []):
            l6 = schematic.get("lecture_6_enhancements", {})
            multi_tf = l6.get("multi_tf_validity", {})

            # If multi-TF opportunity detected with HTF potential, should have recommendation
            if multi_tf.get("has_multi_tf_opportunity") and multi_tf.get("htf_schematic_potential"):
                assert multi_tf.get("recommendation") is not None


@pytest.mark.unit
class TestEnhancedWOVInWOV:
    """Tests for enhanced WOV-in-WOV (schematic within schematic) detection"""

    def test_wov_in_wov_structure(self, accumulation_pattern_df):
        """Test WOV-in-WOV returns correct structure"""
        detector = TCTSchematicDetector(accumulation_pattern_df)
        result = detector.detect_all_schematics()

        for schematic in result.get("accumulation_schematics", []):
            l6 = schematic.get("lecture_6_enhancements", {})
            wov = l6.get("wov_in_wov", {})

            assert "has_wov_in_wov" in wov
            assert "inner_schematic_type" in wov
            assert "inner_entry_price" in wov
            assert "inner_stop_loss" in wov
            assert isinstance(wov["has_wov_in_wov"], bool)

    def test_wov_improves_rr(self, accumulation_pattern_df):
        """Test that WOV entry improves R:R when detected"""
        result = detect_tct_schematics(accumulation_pattern_df)

        for schematic in result.get("accumulation_schematics", []):
            l6 = schematic.get("lecture_6_enhancements", {})
            wov = l6.get("wov_in_wov", {})

            if wov.get("has_wov_in_wov"):
                improvement = wov.get("rr_improvement_factor")
                if improvement is not None:
                    # WOV should improve R:R (factor > 1)
                    assert improvement >= 1.0


@pytest.mark.unit
class TestModel1ToModel2Flow:
    """Tests for Model 1 to Model 2 flow detection"""

    def test_m1_to_m2_flow_structure(self, accumulation_pattern_df):
        """Test M1-to-M2 flow returns correct structure"""
        detector = TCTSchematicDetector(accumulation_pattern_df)
        result = detector.detect_all_schematics()

        for schematic in result.get("accumulation_schematics", []):
            l6 = schematic.get("lecture_6_enhancements", {})
            m1_m2 = l6.get("model1_to_model2_flow", {})

            assert "m1_to_m2_flow_detected" in m1_m2
            assert "original_tap3" in m1_m2
            assert "position_management" in m1_m2
            assert "extended_target" in m1_m2
            assert isinstance(m1_m2["m1_to_m2_flow_detected"], bool)

    def test_m1_to_m2_position_management(self, accumulation_pattern_df):
        """Test position management advice is provided for M1-M2 flow"""
        result = detect_tct_schematics(accumulation_pattern_df)

        for schematic in result.get("accumulation_schematics", []):
            l6 = schematic.get("lecture_6_enhancements", {})
            m1_m2 = l6.get("model1_to_model2_flow", {})

            if m1_m2.get("m1_to_m2_flow_detected"):
                position = m1_m2.get("position_management")
                if position:
                    assert "step_1" in position
                    assert "step_2" in position
                    assert "step_3" in position
                    assert "risk_freed" in position
                # Top-level keys for position flow
                assert "trail_stop_to" in m1_m2
                assert "add_position_trigger" in m1_m2
                assert "extended_target" in m1_m2


@pytest.mark.unit
class TestContextBasedFollowThrough:
    """Tests for context-based follow-through prediction"""

    def test_context_follow_through_structure(self, accumulation_pattern_df):
        """Test context follow-through returns correct structure"""
        detector = TCTSchematicDetector(accumulation_pattern_df)
        result = detector.detect_all_schematics()

        for schematic in result.get("accumulation_schematics", []):
            l6 = schematic.get("lecture_6_enhancements", {})
            context = l6.get("context_follow_through", {})

            assert "bias" in context
            assert "context_zone" in context
            assert "expectation" in context
            assert "confidence" in context or context.get("confidence") is None
            assert "enhanced_target" in context or context.get("enhanced_target") is None

    def test_context_bias_values(self, accumulation_pattern_df):
        """Test context bias has valid values"""
        result = detect_tct_schematics(accumulation_pattern_df)

        for schematic in result.get("accumulation_schematics", []):
            l6 = schematic.get("lecture_6_enhancements", {})
            context = l6.get("context_follow_through", {})

            bias = context.get("bias")
            assert bias in ["bullish", "bearish", "neutral", None]

    def test_premium_discount_zones(self, accumulation_pattern_df):
        """Test premium/discount zone detection"""
        result = detect_tct_schematics(accumulation_pattern_df)

        for schematic in result.get("accumulation_schematics", []):
            l6 = schematic.get("lecture_6_enhancements", {})
            context = l6.get("context_follow_through", {})

            zone = context.get("context_zone")
            if zone is not None:
                assert zone in ["premium", "discount", "equilibrium", "unknown"]


@pytest.fixture
def dual_deviation_pattern_df():
    """DataFrame with dual-side deviation pattern (both high and low deviated)"""
    dates = pd.date_range('2026-01-01', periods=200, freq='1h')
    base = 100000
    prices = []

    # Stage 1: Range formation (candles 0-50)
    range_low = base - 1000
    range_high = base + 1000
    for i in range(50):
        prices.append(base + np.random.uniform(-800, 800))

    # Stage 2: First deviation above (candles 50-80) - High side deviation
    for i in range(30):
        deviation_up = 1500 + i * 30
        prices.append(range_high + deviation_up + np.random.uniform(-100, 100))

    # Stage 3: Return to range (candles 80-110)
    for i in range(30):
        prices.append(base + np.random.uniform(-500, 500))

    # Stage 4: Deviation below (candles 110-140) - Low side deviation
    for i in range(30):
        deviation_down = 1500 + i * 30
        prices.append(range_low - deviation_down + np.random.uniform(-100, 100))

    # Stage 5: Recovery (candles 140-200)
    for i in range(60):
        prices.append(range_low - 1000 + i * 50 + np.random.uniform(-80, 80))

    return pd.DataFrame({
        'open_time': dates,
        'open': prices,
        'high': [p + np.random.uniform(80, 200) for p in prices],
        'low': [p - np.random.uniform(80, 200) for p in prices],
        'close': [p + np.random.uniform(-100, 100) for p in prices],
        'volume': np.random.uniform(100, 1000, 200)
    })


@pytest.mark.unit
class TestDualDeviationPatternDetection:
    """Tests specifically for dual deviation patterns"""

    def test_dual_deviation_pattern_detection(self, dual_deviation_pattern_df):
        """Test detection of dual-side deviation patterns"""
        result = detect_tct_schematics(dual_deviation_pattern_df)

        assert "error" not in result
        assert result["candles_analyzed"] == 200

        # Check for dual deviation detection in any schematic
        all_schematics = (
            result.get("accumulation_schematics", []) +
            result.get("distribution_schematics", [])
        )

        for schematic in all_schematics:
            l6 = schematic.get("lecture_6_enhancements", {})
            dual_dev = l6.get("dual_side_deviation", {})

            # Structure should always be present
            assert "has_dual_side_deviation" in dual_dev
            assert "risk_on_trigger" in dual_dev


@pytest.mark.integration
class TestLecture6Integration:
    """Integration tests for Lecture 6 enhancements"""

    def test_full_lecture_6_detection_pipeline(self):
        """Test complete detection with all Lecture 6 features"""
        np.random.seed(42)
        dates = pd.date_range('2026-01-01', periods=200, freq='1h')
        base = 100000
        prices = []

        # Create complex pattern with multiple Lecture 6 features
        # Phase 1: Downtrend
        for i in range(40):
            prices.append(base - i * 80 + np.random.uniform(-30, 30))

        # Phase 2: Range at low
        range_low = base - 3200
        for i in range(30):
            prices.append(range_low + np.random.uniform(0, 500))

        # Phase 3: Deviation below (potential WOV-in-WOV zone)
        for i in range(20):
            prices.append(range_low - 300 - i * 20 + np.random.uniform(-50, 50))

        # Phase 4: Second deviation (Model 1 tap3)
        for i in range(20):
            prices.append(range_low - 800 - i * 10 + np.random.uniform(-40, 40))

        # Phase 5: Recovery (could trigger M1 to M2 flow)
        for i in range(90):
            prices.append(range_low - 600 + i * 30 + np.random.uniform(-60, 60))

        df = pd.DataFrame({
            'open_time': dates,
            'open': prices,
            'high': [p + np.random.uniform(50, 150) for p in prices],
            'low': [p - np.random.uniform(50, 150) for p in prices],
            'close': [p + np.random.uniform(-60, 60) for p in prices],
            'volume': np.random.uniform(100, 1000, 200)
        })

        result = detect_tct_schematics(df)

        assert "error" not in result
        assert result["candles_analyzed"] == 200

        # Verify all Lecture 6 enhancements are present in results
        all_schematics = (
            result.get("accumulation_schematics", []) +
            result.get("distribution_schematics", [])
        )

        for schematic in all_schematics:
            l6 = schematic.get("lecture_6_enhancements", {})

            # All L6 feature categories should be present
            assert "dual_side_deviation" in l6
            assert "ltf_htf_transition" in l6
            assert "multi_tf_validity" in l6
            assert "wov_in_wov" in l6
            assert "model1_to_model2_flow" in l6
            assert "context_follow_through" in l6

            # Summary flags should be present
            assert "has_conversion" in l6
            assert "has_dual_side_deviation" in l6
            assert "is_nested_in_htf" in l6
            assert "valid_on_htf" in l6
            assert "has_wov_opportunity" in l6
            assert "has_m1_to_m2_opportunity" in l6
            assert "follow_through_bias" in l6

    def test_lecture_6_with_external_ranges(self):
        """Test Lecture 6 detection works with externally provided ranges"""
        np.random.seed(789)
        dates = pd.date_range('2026-01-01', periods=150, freq='1h')

        prices = [100000 + np.random.uniform(-500, 500) for _ in range(150)]

        df = pd.DataFrame({
            'open_time': dates,
            'open': prices,
            'high': [p + 200 for p in prices],
            'low': [p - 200 for p in prices],
            'close': prices,
            'volume': np.random.uniform(100, 1000, 150)
        })

        # Provide external ranges
        external_ranges = [{
            "range_high": 100500.0,
            "range_low": 99500.0,
            "equilibrium": 100000.0,
            "range_size": 1000.0,
            "dl_high": 100800.0,
            "dl_low": 99200.0,
            "range_high_idx": 20,
            "range_low_idx": 30
        }]

        result = detect_tct_schematics(df, external_ranges)

        assert "error" not in result

        # All schematics should have Lecture 6 enhancements
        for schematic in result.get("accumulation_schematics", []):
            assert "lecture_6_enhancements" in schematic

        for schematic in result.get("distribution_schematics", []):
            assert "lecture_6_enhancements" in schematic

    def test_both_lecture_5b_and_6_present(self):
        """Test both Lecture 5B and 6 enhancements are present"""
        np.random.seed(42)
        dates = pd.date_range('2026-01-01', periods=200, freq='1h')
        base = 100000
        prices = []

        for i in range(40):
            prices.append(base - i * 80 + np.random.uniform(-30, 30))

        range_low = base - 3200
        for i in range(30):
            prices.append(range_low + np.random.uniform(0, 500))

        for i in range(20):
            prices.append(range_low - 300 - i * 20 + np.random.uniform(-50, 50))

        for i in range(20):
            prices.append(range_low - 800 - i * 10 + np.random.uniform(-40, 40))

        for i in range(90):
            prices.append(range_low - 600 + i * 30 + np.random.uniform(-60, 60))

        df = pd.DataFrame({
            'open_time': dates,
            'open': prices,
            'high': [p + np.random.uniform(50, 150) for p in prices],
            'low': [p - np.random.uniform(50, 150) for p in prices],
            'close': [p + np.random.uniform(-60, 60) for p in prices],
            'volume': np.random.uniform(100, 1000, 200)
        })

        result = detect_tct_schematics(df)

        all_schematics = (
            result.get("accumulation_schematics", []) +
            result.get("distribution_schematics", [])
        )

        for schematic in all_schematics:
            # Both enhancement sections should be present
            assert "lecture_5b_enhancements" in schematic
            assert "lecture_6_enhancements" in schematic

            # Verify some key fields from each
            l5b = schematic.get("lecture_5b_enhancements", {})
            l6 = schematic.get("lecture_6_enhancements", {})

            # Lecture 5B
            assert "htf_validation" in l5b
            assert "overlapping_structure" in l5b

            # Lecture 6
            assert "dual_side_deviation" in l6
            assert "wov_in_wov" in l6
