"""
tests/unit/test_decision_engine_v2.py — HTF bias ↔ model direction gate tests
===============================================================================
Validates the 1A-b gate (FAIL_HTF_MODEL_DIRECTION) that enforces:
  - Bullish HTF → only bullish (accumulation) setups allowed
  - Bearish HTF → only bearish (distribution) setups allowed
  - Neutral HTF → blocked by 1A (pre-existing gate)
"""

import os
import sys
from datetime import datetime, timezone
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

# Ensure the project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Stub moondev_feed before any transitive import pulls it in.
os.environ.setdefault("MOONDEV_API_KEY", "test")


def _make_candles(n: int = 100, base_price: float = 100.0) -> pd.DataFrame:
    """Generate synthetic OHLCV candles."""
    dates = pd.date_range("2025-06-01", periods=n, freq="4h", tz="UTC")
    rng = np.random.default_rng(42)
    closes = base_price + np.cumsum(rng.normal(0, 0.5, n))
    return pd.DataFrame({
        "open_time": dates,
        "open": closes - 0.2,
        "high": closes + 1.0,
        "low": closes - 1.0,
        "close": closes,
        "volume": rng.uniform(100, 1000, n),
    })


def _make_pre_evaluated(direction: str, model: str, tf: str = "4h",
                        score: int = 80) -> list:
    """Build a pre_evaluated schematic that will reach the gate chain."""
    return [{
        "tf": tf,
        "schematic": {
            "direction": direction,
            "model": model,
            "is_confirmed": True,
            "range": {
                "high": 110.0, "low": 90.0,
                "start_idx": 10, "end_idx": 50,
                "displacement": 0.7,
            },
            "entry": {"price": 105.0},
            "stop_loss": {"price": 89.0},
            "target": {"price": 115.0},
            "tap1": {"price": 90.5},
            "tap2": {"price": 109.5},
            "tap3": {"price": 91.0},
            "bos_confirmation": {"bos_idx": 60, "bos_price": 105.0},
            "quality_score": 0.8,
        },
        "eval_result": {
            "pass": True,
            "score": score,
            "direction": direction,
            "model": model,
        },
    }]


def _call_decide(htf_bias_direction: str, signal_direction: str,
                 model: str = "Model_1") -> dict:
    """Run decide() with controlled HTF bias and signal direction.

    Uses pre_evaluated schematics to bypass detection and evaluation,
    and patches the HTF bias computation to return the desired value.
    """
    from decision_engine_v2 import decide

    # Supply 1d candles so Gate 1A's HTF bias detection runs on them.
    candles = {"1d": _make_candles(100), "4h": _make_candles(100)}
    context = {
        "current_price": 105.0,
        "current_time": datetime(2025, 7, 1, 12, 0, tzinfo=timezone.utc),
        "entry_threshold": 60,
        "pre_evaluated": _make_pre_evaluated(signal_direction, model),
    }

    # Patch market_structure functions imported inside decide() for Gate 1A.
    mock_pivots = {"highs": [1, 2, 3], "lows": [1, 2, 3]}
    mock_ms = {"ms_highs": [1, 2], "ms_lows": [1, 2]}
    with (
        patch("market_structure.classify_trend", return_value=htf_bias_direction),
        patch("market_structure.find_6cr_pivots", return_value=mock_pivots),
        patch("market_structure.confirm_structure_points", return_value=mock_ms),
    ):
        result = decide(candles, context)

    return result


class TestHTFModelDirectionGate:
    """Tests for Gate 1A-b: HTF bias ↔ model direction enforcement."""

    def test_bullish_htf_allows_bullish_signal(self):
        """Bullish HTF + bullish direction (accumulation) → should not be blocked by 1A-b."""
        result = _call_decide("bullish", "bullish", "Model_1")
        # Should not fail with HTF_MODEL_DIRECTION
        assert result.get("failure_code") != "FAIL_HTF_MODEL_DIRECTION"

    def test_bullish_htf_blocks_bearish_signal(self):
        """Bullish HTF + bearish direction (distribution) → FAIL_HTF_MODEL_DIRECTION."""
        result = _call_decide("bullish", "bearish", "Model_2")
        assert result["decision"] == "PASS"
        assert result["failure_code"] == "FAIL_HTF_MODEL_DIRECTION"

    def test_bearish_htf_allows_bearish_signal(self):
        """Bearish HTF + bearish direction (distribution) → should not be blocked by 1A-b."""
        result = _call_decide("bearish", "bearish", "Model_2")
        assert result.get("failure_code") != "FAIL_HTF_MODEL_DIRECTION"

    def test_bearish_htf_blocks_bullish_signal(self):
        """Bearish HTF + bullish direction (accumulation) → FAIL_HTF_MODEL_DIRECTION."""
        result = _call_decide("bearish", "bullish", "Model_1")
        assert result["decision"] == "PASS"
        assert result["failure_code"] == "FAIL_HTF_MODEL_DIRECTION"

    def test_neutral_htf_blocked_by_1a_not_1ab(self):
        """Neutral HTF → blocked by Gate 1A (NO_HTF_BIAS), not 1A-b."""
        result = _call_decide("neutral", "bullish", "Model_1")
        assert result["decision"] == "PASS"
        assert result["failure_code"] == "FAIL_1A_BIAS"

    def test_ranging_htf_blocked_by_1a_not_1ab(self):
        """Ranging HTF → treated as non-directional, blocked by Gate 1A, not 1A-b."""
        result = _call_decide("ranging", "bullish", "Model_1")
        assert result["decision"] == "PASS"
        assert result["failure_code"] == "FAIL_1A_BIAS"

    def test_bullish_htf_blocks_bearish_continuation(self):
        """Bullish HTF + bearish CONTINUATION → FAIL_HTF_MODEL_DIRECTION."""
        result = _call_decide("bullish", "bearish", "CONTINUATION")
        assert result["decision"] == "PASS"
        assert result["failure_code"] == "FAIL_HTF_MODEL_DIRECTION"

    def test_bearish_htf_allows_bearish_continuation(self):
        """Bearish HTF + bearish CONTINUATION → not blocked by 1A-b."""
        result = _call_decide("bearish", "bearish", "CONTINUATION")
        assert result.get("failure_code") != "FAIL_HTF_MODEL_DIRECTION"

    def test_bullish_htf_blocks_m1_from_m2_failure_bearish(self):
        """Bullish HTF + bearish Model_1_from_M2_failure → FAIL_HTF_MODEL_DIRECTION."""
        result = _call_decide("bullish", "bearish", "Model_1_from_M2_failure")
        assert result["decision"] == "PASS"
        assert result["failure_code"] == "FAIL_HTF_MODEL_DIRECTION"

    def test_bullish_htf_allows_m1_from_m2_failure_bullish(self):
        """Bullish HTF + bullish Model_1_from_M2_failure → allowed."""
        result = _call_decide("bullish", "bullish", "Model_1_from_M2_failure")
        assert result.get("failure_code") != "FAIL_HTF_MODEL_DIRECTION"

    def test_accumulation_direction_treated_as_non_matching(self):
        """Direction='accumulation' (range-level label) would be blocked by 1A-b.

        Confirmed schematics always use 'bullish'/'bearish' — this test
        documents that non-binary direction strings do NOT match HTF bias
        values and would be correctly blocked if they ever leaked through.
        """
        result = _call_decide("bullish", "accumulation", "Model_1")
        assert result["decision"] == "PASS"
        assert result["failure_code"] == "FAIL_HTF_MODEL_DIRECTION"

    def test_distribution_direction_treated_as_non_matching(self):
        """Symmetric test: 'distribution' direction does not match bearish HTF."""
        result = _call_decide("bearish", "distribution", "Model_2")
        assert result["decision"] == "PASS"
        assert result["failure_code"] == "FAIL_HTF_MODEL_DIRECTION"


class TestRolloutFraction:
    """Verify ROLLOUT_FRACTION is set to 0.1 for canary validation."""

    def test_rollout_fraction_is_canary(self):
        from decision_engine_v2 import ROLLOUT_FRACTION
        assert ROLLOUT_FRACTION == 0.1, (
            f"ROLLOUT_FRACTION should be 0.1 for canary, got {ROLLOUT_FRACTION}"
        )


class TestTradingSymbols:
    """Verify all 12 pairs are in TRADING_SYMBOLS."""

    def test_all_pairs_present(self):
        from schematics_5b_trader import TRADING_SYMBOLS
        expected = [
            "BTCUSDT", "ETHUSDT", "SOLUSDT",
            "BCHUSDT", "WIFUSDT", "DOGEUSDT", "HBARUSDT", "FETUSDT",
            "XMRUSDT", "FARTCOINUSDT", "PEPEUSDT", "XRPUSDT",
        ]
        for sym in expected:
            assert sym in TRADING_SYMBOLS, f"{sym} missing from TRADING_SYMBOLS"


class TestMoonDevMappings:
    """Verify MoonDev feed maps include all pairs."""

    def test_symbol_map_complete(self):
        from moondev_feed import _SYMBOL_MAP
        for sym in ["BCHUSDT", "WIFUSDT", "DOGEUSDT", "HBARUSDT", "FETUSDT",
                     "XMRUSDT", "FARTCOINUSDT", "PEPEUSDT", "XRPUSDT"]:
            assert sym in _SYMBOL_MAP, f"{sym} missing from _SYMBOL_MAP"

    def test_price_ticker_map_complete(self):
        from moondev_feed import _PRICE_TICKER_MAP
        expected = {
            "BCHUSDT": "BCH", "WIFUSDT": "WIF", "DOGEUSDT": "DOGE",
            "HBARUSDT": "HBAR", "FETUSDT": "FET", "XMRUSDT": "XMR",
            "FARTCOINUSDT": "FARTCOIN", "PEPEUSDT": "PEPE", "XRPUSDT": "XRP",
        }
        for sym, ticker in expected.items():
            assert _PRICE_TICKER_MAP.get(sym) == ticker, (
                f"{sym} → expected {ticker}, got {_PRICE_TICKER_MAP.get(sym)}"
            )
