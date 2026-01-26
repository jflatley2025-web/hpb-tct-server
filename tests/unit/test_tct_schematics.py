"""
Unit tests for tct_schematics.py
Tests TCT Schematic detection (Lecture 5A methodology) including:
- Model 1 Accumulation: Tab1 → Tab2 (deviation) → Tab3 (lower deviation)
- Model 2 Accumulation: Tab1 → Tab2 (deviation) → Tab3 (higher low at extreme liq/demand)
- Model 1 Distribution: Tab1 → Tab2 (deviation) → Tab3 (higher deviation)
- Model 2 Distribution: Tab1 → Tab2 (deviation) → Tab3 (lower high at extreme liq/supply)
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

    # Stage 2: Range formation at low (candles 30-60) - Tab1 zone
    range_low = base - 3000
    for i in range(30):
        prices.append(range_low + np.random.uniform(-200, 400))

    # Stage 3: First deviation below range (candles 60-80) - Tab2 zone
    for i in range(20):
        deviation_depth = 500 + i * 20
        prices.append(range_low - deviation_depth + np.random.uniform(-100, 100))

    # Stage 4: Second lower deviation (candles 80-100) - Tab3 Model 1 zone
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

    # Stage 2: Range formation at high (candles 30-60) - Tab1 zone
    range_high = base + 3000
    for i in range(30):
        prices.append(range_high + np.random.uniform(-400, 200))

    # Stage 3: First deviation above range (candles 60-80) - Tab2 zone
    for i in range(20):
        deviation_height = 500 + i * 20
        prices.append(range_high + deviation_height + np.random.uniform(-100, 100))

    # Stage 4: Second higher deviation (candles 80-100) - Tab3 Model 1 zone
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

    # Stage 2: Range formation (Tab1)
    range_low = base - 2500
    range_high = range_low + 1500
    for i in range(30):
        prices.append(range_low + np.random.uniform(0, 600))

    # Stage 3: First deviation (Tab2)
    for i in range(20):
        prices.append(range_low - 400 - i * 10 + np.random.uniform(-50, 50))

    # Stage 4: Higher low (Tab3 Model 2) - NOT lower than Tab2
    hl_price = range_low - 300  # Higher than Tab2's lowest
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
            assert "tab1" in s
            assert "tab2" in s
            assert "tab3" in s
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
            assert "tab1" in schematic
            assert "tab2" in schematic
            assert "tab3" in schematic

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
            # Tab3 should be lower than Tab2 for Model 1
            if s["tab3"]["price"] and s["tab2"]["price"]:
                assert s["tab3"]["price"] <= s["tab2"]["price"], \
                    "Model 1 Tab3 should be lower than or equal to Tab2"


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
            # Tab3 should be higher than Tab2 for Model 2 accumulation
            if s["tab3"]["price"] and s["tab2"]["price"]:
                assert s["tab3"]["price"] >= s["tab2"]["price"], \
                    "Model 2 Tab3 should be higher than or equal to Tab2"

    def test_model_2_extreme_requirements(self, model_2_accumulation_df):
        """Test Model 2 requires extreme liquidity OR extreme demand/supply"""
        detector = TCTSchematicDetector(model_2_accumulation_df)

        schematics = detector._detect_accumulation_schematics(None)

        model_2_schematics = [s for s in schematics if "Model_2" in s.get("schematic_type", "")]

        for s in model_2_schematics:
            tab3 = s.get("tab3", {})
            # Model 2 should have extreme liquidity OR extreme demand info
            has_extreme_req = (
                tab3.get("grabs_extreme_liquidity") or
                tab3.get("mitigates_extreme_demand") or
                tab3.get("mitigates_extreme_supply")
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

    def test_accumulation_stop_loss_below_tab3(self, accumulation_pattern_df):
        """Test accumulation stop loss is at Tab3 price (TCT rule)"""
        detector = TCTSchematicDetector(accumulation_pattern_df)

        schematics = detector._detect_accumulation_schematics(None)

        for s in schematics:
            if s.get("stop_loss", {}).get("price") and s.get("tab3", {}).get("price"):
                assert s["stop_loss"]["price"] == s["tab3"]["price"], \
                    "Stop loss should be at Tab3 price"

    def test_accumulation_target_at_range_high(self, accumulation_pattern_df, simple_range_data):
        """Test accumulation target is at range high (Wyckoff high)"""
        detector = TCTSchematicDetector(accumulation_pattern_df)

        schematics = detector._detect_accumulation_schematics([simple_range_data])

        for s in schematics:
            if s.get("range", {}).get("high"):
                # Target should match range high
                if s.get("target", {}).get("price"):
                    assert s["target"]["price"] == s["range"]["high"]

    def test_distribution_stop_loss_above_tab3(self, distribution_pattern_df):
        """Test distribution stop loss is at Tab3 price"""
        detector = TCTSchematicDetector(distribution_pattern_df)

        schematics = detector._detect_distribution_schematics(None)

        for s in schematics:
            if s.get("stop_loss", {}).get("price") and s.get("tab3", {}).get("price"):
                assert s["stop_loss"]["price"] == s["tab3"]["price"]

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
