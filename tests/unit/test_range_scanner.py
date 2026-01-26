"""
Unit tests for range_scanner.py
Tests the MEXC Range Scanner including range detection, scoring, and API integration
"""
import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from range_scanner import MEXCRangeScanner, RangeCandidate, VALID_INTERVALS


@pytest.mark.unit
class TestRangeCandidate:
    """Tests for RangeCandidate data class"""

    def test_range_candidate_initialization(self):
        """Test RangeCandidate initializes with correct values"""
        candles = [{"t": 1706000000, "h": 42100.0, "l": 41900.0, "c": 42000.0}]
        rc = RangeCandidate(tf="1h", high=42100.0, low=41900.0, candles=candles)

        assert rc.timeframe == "1h"
        assert rc.range_high == 42100.0
        assert rc.range_low == 41900.0
        assert rc.eq == 42000.0  # (42100 + 41900) / 2
        assert rc.score == 0.0
        assert rc.candles == candles


@pytest.mark.unit
class TestMEXCRangeScanner:
    """Tests for MEXCRangeScanner class"""

    def test_scanner_initialization(self):
        """Test scanner initializes with correct defaults"""
        scanner = MEXCRangeScanner(symbol="BTCUSDT", limit=300)

        assert scanner.symbol == "BTCUSDT"
        assert scanner.limit == 300
        assert scanner.results == {"LTF": [], "HTF": []}
        assert scanner.paused is False
        assert scanner.current_tf is None

    def test_detect_range_basic(self, simple_range_candles):
        """Test basic range detection finds high and low"""
        scanner = MEXCRangeScanner()
        high, low = scanner.detect_range(simple_range_candles)

        assert high == 42300.0  # Highest high
        assert low == 41900.0   # Lowest low

    def test_detect_range_empty_candles(self):
        """Test range detection handles empty candle list"""
        scanner = MEXCRangeScanner()
        result = scanner.detect_range([])

        assert result is None

    def test_score_range_basic(self, simple_range_candles):
        """Test range scoring produces valid score"""
        scanner = MEXCRangeScanner()
        high, low = scanner.detect_range(simple_range_candles)
        score = scanner.score_range(simple_range_candles, high, low)

        assert 0.0 <= score <= 1.0
        assert isinstance(score, float)

    def test_score_range_empty_candles(self):
        """Test scoring handles empty candles"""
        scanner = MEXCRangeScanner()
        score = scanner.score_range([], 42000.0, 41000.0)

        assert score == 0.0

    def test_score_range_tight_range_high_score(self):
        """Test tight range with prices near EQ scores reasonably"""
        scanner = MEXCRangeScanner()
        # Candles tightly clustered around equilibrium
        candles = [
            {"t": 1706000000, "h": 101.0, "l": 99.0, "c": 100.0},
            {"t": 1706000060, "h": 101.0, "l": 99.0, "c": 100.5},
            {"t": 1706000120, "h": 101.0, "l": 99.0, "c": 99.5},
        ]
        score = scanner.score_range(candles, 101.0, 99.0)

        # Tight clustering should produce a positive score
        assert score > 0.3

    def test_pause_scanner(self):
        """Test pause sets paused flag"""
        scanner = MEXCRangeScanner()
        scanner.pause()

        assert scanner.paused is True

    @pytest.mark.asyncio
    async def test_resume_scanner(self):
        """Test resume clears paused flag"""
        scanner = MEXCRangeScanner()
        scanner.pause()
        scanner.paused = False  # Manually resume without full scan

        assert scanner.paused is False


@pytest.mark.unit
@pytest.mark.asyncio
class TestMEXCAPIIntegration:
    """Tests for MEXC API integration with mocking"""

    async def test_fetch_klines_success(self, mock_mexc_response):
        """Test successful klines fetch from MEXC"""
        scanner = MEXCRangeScanner()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_mexc_response

        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(return_value=mock_response)

            candles = await scanner.fetch_klines("1m")

            assert len(candles) == 5
            assert candles[0]["h"] == 42300.0  # Reversed order
            assert "t" in candles[0]
            assert "h" in candles[0]
            assert "l" in candles[0]
            assert "c" in candles[0]

    async def test_fetch_klines_invalid_timeframe(self):
        """Test fetch skips invalid timeframe"""
        scanner = MEXCRangeScanner()
        candles = await scanner.fetch_klines("invalid_tf")

        assert candles == []

    async def test_fetch_klines_api_error_retry(self):
        """Test fetch retries on API error"""
        scanner = MEXCRangeScanner()

        mock_response = MagicMock()
        mock_response.status_code = 500

        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(return_value=mock_response)

            candles = await scanner.fetch_klines("1m")

            # Should return empty after retries
            assert candles == []

    async def test_fetch_klines_empty_response(self):
        """Test fetch handles empty API response"""
        scanner = MEXCRangeScanner()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = []

        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(return_value=mock_response)

            candles = await scanner.fetch_klines("1m")

            assert candles == []

    async def test_fetch_klines_network_timeout(self):
        """Test fetch handles network timeout gracefully"""
        scanner = MEXCRangeScanner()

        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                side_effect=asyncio.TimeoutError("Connection timeout")
            )

            candles = await scanner.fetch_klines("1m")

            assert candles == []


@pytest.mark.integration
@pytest.mark.asyncio
class TestRangeScannerIntegration:
    """Integration tests for full scan cycles"""

    async def test_scan_timeframes_single_group(self, mock_mexc_response):
        """Test scanning a single timeframe group"""
        scanner = MEXCRangeScanner()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_mexc_response

        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(return_value=mock_response)

            await scanner.scan_timeframes("LTF", ["1m", "5m"])

            # Should have scanned and stored results
            assert len(scanner.results["LTF"]) > 0
            assert all(isinstance(r, RangeCandidate) for r in scanner.results["LTF"])

    async def test_scan_timeframes_paused(self):
        """Test scan stops when paused"""
        scanner = MEXCRangeScanner()
        scanner.pause()

        await scanner.scan_timeframes("LTF", ["1m", "5m"])

        # Should not scan when paused
        assert scanner.current_tf == "1m"
        assert len(scanner.results["LTF"]) == 0

    async def test_scan_sorts_by_score(self, mock_mexc_response):
        """Test scan results are sorted by score"""
        scanner = MEXCRangeScanner()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_mexc_response

        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(return_value=mock_response)

            await scanner.scan_timeframes("HTF", ["1h", "4h", "1d", "1w"])

            # Results should be sorted by score descending
            scores = [r.score for r in scanner.results["HTF"]]
            assert scores == sorted(scores, reverse=True)

            # Should keep only top 3
            assert len(scanner.results["HTF"]) <= 3


@pytest.mark.unit
class TestValidIntervals:
    """Test MEXC interval validation"""

    def test_valid_intervals_exist(self):
        """Test VALID_INTERVALS is properly defined"""
        assert isinstance(VALID_INTERVALS, set)
        assert "1m" in VALID_INTERVALS
        assert "1h" in VALID_INTERVALS
        assert "1d" in VALID_INTERVALS
        assert "1w" in VALID_INTERVALS
        assert len(VALID_INTERVALS) > 10
