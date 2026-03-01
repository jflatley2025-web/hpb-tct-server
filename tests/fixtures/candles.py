"""
tests/fixtures/candles.py — Deterministic candle DataFrame factory functions.

All factories use a seed parameter so tests are reproducible.
All DataFrames use the standard column format used by phemex_feed.py:
  open_time (datetime64[ns, UTC]), open, high, low, close, volume (float64)

Factory functions:
  make_uptrend_candles        — higher highs and higher lows
  make_downtrend_candles      — lower highs and lower lows
  make_ranging_candles        — oscillating within a band
  make_demand_zone_candles    — contains a valid bearish OB + bullish impulse
  make_supply_zone_candles    — contains a valid bullish OB + bearish impulse
  make_tap_at_zone_candles    — candles that tap/wick into a zone
  make_no_tap_candles         — candles that stay above a zone (no tap)
  make_phemex_ohlcv_raw       — raw ccxt list format [[ts, o, h, l, c, v], ...]
  make_minimal_candles        — just enough candles to pass length checks
  make_insufficient_candles   — fewer candles than minimum thresholds
"""

from __future__ import annotations

from datetime import timezone
from typing import Optional

import numpy as np
import pandas as pd


# Standard column order matching phemex_feed.py
COLUMNS = ["open_time", "open", "high", "low", "close", "volume"]

# Base timestamp for all fixtures (2026-01-01 00:00 UTC)
_BASE_TS = pd.Timestamp("2026-01-01", tz="UTC")


def _make_timestamps(n: int, freq: str = "15min") -> pd.DatetimeIndex:
    return pd.date_range(_BASE_TS, periods=n, freq=freq, tz="UTC")


def _assemble(
    open_time: pd.DatetimeIndex,
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    volumes: np.ndarray,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open_time": open_time,
            "open": opens.astype(float),
            "high": highs.astype(float),
            "low": lows.astype(float),
            "close": closes.astype(float),
            "volume": volumes.astype(float),
        }
    )


# ---------------------------------------------------------------------------
# Trend candles
# ---------------------------------------------------------------------------

def make_uptrend_candles(
    n: int = 200,
    base_price: float = 40000.0,
    step: float = 50.0,
    candle_range: float = 200.0,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Return n candles forming a clear uptrend (higher highs, higher lows).

    Each candle advances base_price by step with a random body in ±candle_range/2.
    BOS events and RTZ are constructed so Gate 1 passes on this data.
    """
    rng = np.random.default_rng(seed)
    times = _make_timestamps(n)

    closes = np.array([base_price + i * step + rng.uniform(-20, 20) for i in range(n)])
    opens = closes - rng.uniform(10, candle_range / 4, size=n)
    highs = closes + rng.uniform(10, candle_range / 2, size=n)
    lows = opens - rng.uniform(10, candle_range / 2, size=n)

    # Enforce open < close (bullish candles) and lows < opens, highs > closes
    lows = np.minimum(lows, opens - 5)
    highs = np.maximum(highs, closes + 5)

    return _assemble(times, opens, highs, lows, closes,
                     rng.uniform(100, 1000, size=n))


def make_downtrend_candles(
    n: int = 200,
    base_price: float = 40000.0,
    step: float = 50.0,
    candle_range: float = 200.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Return n candles forming a clear downtrend (lower highs, lower lows)."""
    rng = np.random.default_rng(seed)
    times = _make_timestamps(n)

    closes = np.array([base_price - i * step + rng.uniform(-20, 20) for i in range(n)])
    opens = closes + rng.uniform(10, candle_range / 4, size=n)
    highs = opens + rng.uniform(10, candle_range / 2, size=n)
    lows = closes - rng.uniform(10, candle_range / 2, size=n)

    # Enforce open > close (bearish candles)
    highs = np.maximum(highs, opens + 5)
    lows = np.minimum(lows, closes - 5)

    return _assemble(times, opens, highs, lows, closes,
                     rng.uniform(100, 1000, size=n))


def make_ranging_candles(
    n: int = 100,
    center_price: float = 40000.0,
    band: float = 500.0,
    candle_range: float = 100.0,
    min_touches: int = 3,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Return n candles oscillating within a band around center_price.

    Ensures at least min_touches at each extreme for Gate 2 (range detection).
    """
    rng = np.random.default_rng(seed)
    times = _make_timestamps(n, freq="4h")

    rng_high = center_price + band / 2
    rng_low = center_price - band / 2

    closes = center_price + rng.uniform(-band / 2 * 0.8, band / 2 * 0.8, size=n)
    opens = closes + rng.uniform(-candle_range / 2, candle_range / 2, size=n)
    highs = np.maximum(opens, closes) + rng.uniform(5, candle_range / 2, size=n)
    lows = np.minimum(opens, closes) - rng.uniform(5, candle_range / 2, size=n)

    # Force explicit touches at range extremes
    touch_step = n // (min_touches + 1)
    for i in range(min_touches):
        idx = touch_step * (i + 1)
        if idx < n:
            highs[idx] = rng_high
            lows[idx] = rng_low

    # Clip to prevent exceeding the band by too much
    highs = np.minimum(highs, rng_high * 1.001)
    lows = np.maximum(lows, rng_low * 0.999)

    return _assemble(times, opens, highs, lows, closes,
                     rng.uniform(100, 1000, size=n))


# ---------------------------------------------------------------------------
# Supply & Demand zone candles
# ---------------------------------------------------------------------------

def make_demand_zone_candles(
    n: int = 150,
    base_price: float = 40000.0,
    zone_high: float = 39800.0,
    zone_low: float = 39600.0,
    impulse_candles: int = 4,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Return candles containing a valid demand zone (bearish OB + bullish impulse).

    Structure:
      - First 60% of candles: sideways noise around base_price
      - Then: one bearish candle at zone high/low level (the OB)
      - Then: impulse_candles bullish candles moving strongly upward
      - Remainder: continuation upward

    This satisfies Gate 3 pass conditions for bullish bias.
    """
    rng = np.random.default_rng(seed)
    times = _make_timestamps(n)

    ob_idx = int(n * 0.6)
    impulse_step = (base_price - zone_high) / impulse_candles * 2

    closes_list = []
    opens_list = []
    highs_list = []
    lows_list = []

    for i in range(n):
        if i < ob_idx:
            # Sideways noise
            c = base_price + rng.uniform(-200, 200)
            o = c + rng.uniform(-100, 100)
            h = max(o, c) + rng.uniform(10, 80)
            l = min(o, c) - rng.uniform(10, 80)
        elif i == ob_idx:
            # Bearish OB candle at zone level
            o = zone_high
            c = zone_low
            h = zone_high + rng.uniform(5, 20)
            l = zone_low - rng.uniform(5, 20)
        elif ob_idx < i <= ob_idx + impulse_candles:
            # Bullish impulse candles: each moves >= 1% up from previous close
            # so the OB + impulse pattern is always detectable regardless of
            # the zone_high / base_price ratio.
            prev_close = float(closes_list[-1]) if closes_list else zone_high
            move = prev_close * 0.012  # 1.2% per candle (above SD_IMPULSE_PCT=0.5%)
            c = prev_close + move
            o = prev_close + rng.uniform(0, move * 0.1)
            h = c + rng.uniform(10, 40)
            l = o - rng.uniform(5, 15)
        else:
            # Continuation
            last_close = closes_list[-1] if closes_list else base_price
            c = last_close + rng.uniform(10, 80)
            o = last_close + rng.uniform(-20, 20)
            h = max(o, c) + rng.uniform(10, 60)
            l = min(o, c) - rng.uniform(5, 30)

        closes_list.append(float(c))
        opens_list.append(float(o))
        highs_list.append(float(h))
        lows_list.append(float(l))

    return _assemble(
        times,
        np.array(opens_list),
        np.array(highs_list),
        np.array(lows_list),
        np.array(closes_list),
        rng.uniform(100, 1000, size=n),
    )


def make_supply_zone_candles(
    n: int = 150,
    base_price: float = 40000.0,
    zone_high: float = 40400.0,
    zone_low: float = 40200.0,
    impulse_candles: int = 4,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Return candles containing a valid supply zone (bullish OB + bearish impulse).

    This satisfies Gate 3 pass conditions for bearish bias.
    """
    rng = np.random.default_rng(seed)
    times = _make_timestamps(n)

    ob_idx = int(n * 0.6)
    impulse_step = (zone_low - base_price) / impulse_candles * 2

    closes_list = []
    opens_list = []
    highs_list = []
    lows_list = []

    for i in range(n):
        if i < ob_idx:
            c = base_price + rng.uniform(-200, 200)
            o = c + rng.uniform(-100, 100)
            h = max(o, c) + rng.uniform(10, 80)
            l = min(o, c) - rng.uniform(10, 80)
        elif i == ob_idx:
            # Bullish OB candle at zone level
            o = zone_low
            c = zone_high
            h = zone_high + rng.uniform(5, 20)
            l = zone_low - rng.uniform(5, 20)
        elif ob_idx < i <= ob_idx + impulse_candles:
            # Bearish impulse candles: each moves >= 1% down from previous close
            prev_close = float(closes_list[-1]) if closes_list else zone_high
            move = prev_close * 0.012
            c = prev_close - move
            o = prev_close - rng.uniform(0, move * 0.1)
            h = o + rng.uniform(5, 15)
            l = c - rng.uniform(10, 40)
        else:
            last_close = closes_list[-1] if closes_list else base_price
            c = last_close - rng.uniform(10, 80)
            o = last_close + rng.uniform(-20, 20)
            h = max(o, c) + rng.uniform(5, 30)
            l = min(o, c) - rng.uniform(10, 60)

        closes_list.append(float(c))
        opens_list.append(float(o))
        highs_list.append(float(h))
        lows_list.append(float(l))

    return _assemble(
        times,
        np.array(opens_list),
        np.array(highs_list),
        np.array(lows_list),
        np.array(closes_list),
        rng.uniform(100, 1000, size=n),
    )


# ---------------------------------------------------------------------------
# Liquidity tap candles
# ---------------------------------------------------------------------------

def make_tap_at_zone_candles(
    n: int = 200,
    base_price: float = 41000.0,
    zone_high: float = 40200.0,
    zone_low: float = 40000.0,
    tap_idx: int = 180,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Return candles that include a wick tap into [zone_low, zone_high] at tap_idx.

    The candle at tap_idx has its low at zone_high - 10 (inside the zone),
    satisfying Gate 4 pass conditions for bullish bias.
    """
    rng = np.random.default_rng(seed)
    times = _make_timestamps(n)

    closes = np.full(n, base_price) + rng.uniform(-100, 100, size=n)
    opens = closes + rng.uniform(-80, 80, size=n)
    highs = np.maximum(opens, closes) + rng.uniform(10, 60, size=n)
    lows = np.minimum(opens, closes) - rng.uniform(10, 60, size=n)

    # Force tap at tap_idx: low dips into zone
    lows[tap_idx] = zone_high - 10.0
    highs[tap_idx] = max(highs[tap_idx], closes[tap_idx] + 20)

    return _assemble(times, opens, highs, lows, closes,
                     rng.uniform(100, 1000, size=n))


def make_no_tap_candles(
    n: int = 200,
    base_price: float = 42000.0,
    zone_high: float = 40200.0,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Return candles that stay well above zone_high — no tap into any demand zone.

    Satisfies Gate 4 fail conditions (no tap detected).
    """
    rng = np.random.default_rng(seed)
    times = _make_timestamps(n)

    # Price stays >= base_price, well above zone_high
    closes = base_price + rng.uniform(100, 500, size=n)
    opens = closes + rng.uniform(-80, 80, size=n)
    highs = np.maximum(opens, closes) + rng.uniform(10, 60, size=n)
    lows = np.minimum(opens, closes) - rng.uniform(10, 30, size=n)

    # Enforce lows stay above zone_high
    lows = np.maximum(lows, zone_high + 50)

    return _assemble(times, opens, highs, lows, closes,
                     rng.uniform(100, 1000, size=n))


# ---------------------------------------------------------------------------
# Utility candles
# ---------------------------------------------------------------------------

def make_minimal_candles(
    n: int = 15,
    base_price: float = 40000.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Return the minimum number of candles needed to pass length checks."""
    rng = np.random.default_rng(seed)
    times = _make_timestamps(n)
    closes = base_price + rng.uniform(-50, 50, size=n)
    opens = closes + rng.uniform(-30, 30, size=n)
    highs = np.maximum(opens, closes) + rng.uniform(5, 20, size=n)
    lows = np.minimum(opens, closes) - rng.uniform(5, 20, size=n)
    return _assemble(times, opens, highs, lows, closes,
                     rng.uniform(50, 200, size=n))


def make_insufficient_candles(
    n: int = 3,
    base_price: float = 40000.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Return fewer candles than any gate minimum — forces FAIL on all gates."""
    rng = np.random.default_rng(seed)
    times = _make_timestamps(n)
    closes = base_price + rng.uniform(-10, 10, size=n)
    opens = closes.copy()
    highs = closes + 5.0
    lows = closes - 5.0
    return _assemble(times, opens, highs, lows, closes,
                     np.ones(n) * 100.0)


# ---------------------------------------------------------------------------
# Raw ccxt OHLCV format
# ---------------------------------------------------------------------------

def make_phemex_ohlcv_raw(
    n: int = 200,
    start_price: float = 40000.0,
    step: float = 10.0,
    seed: int = 42,
) -> list:
    """
    Return raw ccxt OHLCV list: [[timestamp_ms, open, high, low, close, volume], ...]

    Used to mock ccxt.phemex.fetch_ohlcv() in phemex_feed unit tests.
    """
    rng = np.random.default_rng(seed)
    base_ts = int(_BASE_TS.timestamp() * 1000)
    interval_ms = 15 * 60 * 1000  # 15 minutes

    rows = []
    price = start_price
    for i in range(n):
        ts = base_ts + i * interval_ms
        o = price + rng.uniform(-50, 50)
        c = o + step + rng.uniform(-20, 20)
        h = max(o, c) + rng.uniform(5, 30)
        l = min(o, c) - rng.uniform(5, 30)
        v = rng.uniform(50, 500)
        rows.append([ts, round(o, 2), round(h, 2), round(l, 2), round(c, 2), round(v, 2)])
        price = c

    return rows
