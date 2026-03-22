"""
Tests for msce_engine — real session-based MSCE context.
"""

import pytest
from unittest.mock import patch
from datetime import datetime, timezone
from msce_engine import get_msce_context, _derive_session_bias, _detect_session


class TestDeriveSessionBias:
    """Test time-based session bias derivation."""

    def test_accumulation_bullish(self):
        """Asia accumulation aligns with bullish HTF."""
        assert _derive_session_bias("accumulation", "bullish") == "bullish"

    def test_accumulation_bearish(self):
        """Asia accumulation aligns with bearish HTF."""
        assert _derive_session_bias("accumulation", "bearish") == "bearish"

    def test_expansion_bullish(self):
        """London expansion aligns with bullish HTF."""
        assert _derive_session_bias("expansion", "bullish") == "bullish"

    def test_expansion_bearish(self):
        """London expansion aligns with bearish HTF."""
        assert _derive_session_bias("expansion", "bearish") == "bearish"

    def test_distribution_bullish(self):
        """NY distribution opposes bullish HTF → bearish."""
        assert _derive_session_bias("distribution", "bullish") == "bearish"

    def test_distribution_bearish(self):
        """NY distribution opposes bearish HTF → bullish."""
        assert _derive_session_bias("distribution", "bearish") == "bullish"

    def test_neutral_htf_returns_none(self):
        """Neutral HTF bias → None (no valid session bias)."""
        assert _derive_session_bias("expansion", "neutral") is None

    def test_unknown_htf_returns_none(self):
        """Unknown HTF bias string → None."""
        assert _derive_session_bias("accumulation", "unknown") is None


class TestDetectSession:
    """Test session detection with mocked session_manipulation."""

    @patch("session_manipulation.get_active_session", return_value="asia")
    def test_asia_manipulation_window(self, _mock):
        """In Asia manipulation window → Asia accumulation."""
        name, stype, in_window = _detect_session()
        assert name == "Asia"
        assert stype == "accumulation"
        assert in_window is True

    @patch("session_manipulation.get_active_session", return_value="london")
    def test_london_manipulation_window(self, _mock):
        """In London manipulation window → London expansion."""
        name, stype, in_window = _detect_session()
        assert name == "London"
        assert stype == "expansion"
        assert in_window is True

    @patch("session_manipulation.get_active_session", return_value="new_york")
    def test_ny_manipulation_window(self, _mock):
        """In NY manipulation window → NY distribution."""
        name, stype, in_window = _detect_session()
        assert name == "New York"
        assert stype == "distribution"
        assert in_window is True

    @patch("session_manipulation.get_active_session", return_value=None)
    def test_off_session_uses_broad_hours(self, _mock):
        """Not in manipulation window → falls back to broad session hours."""
        name, stype, in_window = _detect_session()
        # Should return a valid session based on current UTC hour
        assert name in ("Asia", "London", "New York")
        assert in_window is False


class TestGetMSCEContext:
    """Test full MSCE context building."""

    @patch("msce_engine._detect_session")
    def test_london_bullish(self, mock_detect):
        """London + bullish HTF → bullish session_bias."""
        mock_detect.return_value = ("London", "expansion", True)
        ctx = get_msce_context("bullish")
        assert ctx["session"] == "London"
        assert ctx["session_bias"] == "bullish"
        assert ctx["session_type"] == "expansion"
        assert ctx["is_manipulation_window"] is True

    @patch("msce_engine._detect_session")
    def test_ny_bullish_opposes(self, mock_detect):
        """NY distribution + bullish HTF → bearish session_bias."""
        mock_detect.return_value = ("New York", "distribution", False)
        ctx = get_msce_context("bullish")
        assert ctx["session"] == "New York"
        assert ctx["session_bias"] == "bearish"

    @patch("msce_engine._detect_session")
    def test_neutral_htf(self, mock_detect):
        """Neutral HTF → None session_bias."""
        mock_detect.return_value = ("London", "expansion", True)
        ctx = get_msce_context("neutral")
        assert ctx["session_bias"] is None

    @patch("msce_engine._detect_session")
    def test_context_structure(self, mock_detect):
        """All required fields present."""
        mock_detect.return_value = ("Asia", "accumulation", False)
        ctx = get_msce_context("bearish")
        assert "session" in ctx
        assert "session_bias" in ctx
        assert "session_type" in ctx
        assert "is_manipulation_window" in ctx
