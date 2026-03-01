"""
Unit tests for phemex_feed.py

Tests:
  - fetch_candles returns a correctly shaped DataFrame
  - Column names and dtypes are correct
  - TTL cache is hit on the second call (no second exchange call)
  - Returns None on NetworkError
  - Returns None on ExchangeError
  - Returns None on unexpected exception
  - LTF fetches 200 rows, MTF 150, HTF 100
  - Returned DataFrame is read-only (numpy arrays are not writeable)
  - Raises ValueError for unsupported timeframe
  - clear_cache() forces a fresh fetch on the next call
"""

import time
from unittest.mock import MagicMock, patch, call

import numpy as np
import pandas as pd
import pytest
import ccxt

import phemex_feed
from phemex_feed import (
    fetch_candles,
    fetch_all,
    set_exchange,
    clear_cache,
    OHLCV_COLUMNS,
    LTF_TF,
    MTF_TF,
    HTF_TF,
    LTF_LIMIT,
    MTF_LIMIT,
    HTF_LIMIT,
)
from tests.fixtures.candles import make_phemex_ohlcv_raw


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_exchange(n: int = 200, start_price: float = 40000.0) -> MagicMock:
    """Return a mock ccxt.phemex whose fetch_ohlcv returns n rows."""
    mock = MagicMock(spec=ccxt.phemex)
    mock.fetch_ohlcv.return_value = make_phemex_ohlcv_raw(n=n, start_price=start_price)
    return mock


@pytest.fixture(autouse=True)
def reset_feed():
    """Reset phemex_feed module state before every test."""
    clear_cache()
    phemex_feed._exchange = None
    yield
    clear_cache()
    phemex_feed._exchange = None


# ---------------------------------------------------------------------------
# Shape and schema
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_fetch_candles_returns_dataframe():
    """fetch_candles returns a non-empty pandas DataFrame."""
    mock = _make_mock_exchange(n=200)
    set_exchange(mock)

    df = fetch_candles(LTF_TF)

    assert df is not None
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 200


@pytest.mark.unit
def test_fetch_candles_correct_columns():
    """Returned DataFrame has the standard OHLCV column set."""
    mock = _make_mock_exchange(n=200)
    set_exchange(mock)

    df = fetch_candles(LTF_TF)

    assert list(df.columns) == OHLCV_COLUMNS


@pytest.mark.unit
def test_fetch_candles_correct_dtypes():
    """open_time is datetime64 UTC; numeric columns are float64."""
    mock = _make_mock_exchange(n=200)
    set_exchange(mock)

    df = fetch_candles(LTF_TF)

    assert pd.api.types.is_datetime64_any_dtype(df["open_time"])
    for col in ("open", "high", "low", "close", "volume"):
        assert df[col].dtype == np.float64, f"Column {col!r} is not float64"


# ---------------------------------------------------------------------------
# TTL cache behaviour
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_fetch_candles_cache_hit_on_second_call():
    """The second call within TTL returns cached data without calling exchange."""
    mock = _make_mock_exchange(n=200)
    set_exchange(mock)

    df1 = fetch_candles(LTF_TF)
    df2 = fetch_candles(LTF_TF)

    # Exchange should be called exactly once
    assert mock.fetch_ohlcv.call_count == 1
    # Both calls return the same object
    assert df1 is df2


@pytest.mark.unit
def test_clear_cache_forces_refetch():
    """After clear_cache(), the next fetch calls the exchange again."""
    mock = _make_mock_exchange(n=200)
    set_exchange(mock)

    fetch_candles(LTF_TF)
    clear_cache()
    fetch_candles(LTF_TF)

    assert mock.fetch_ohlcv.call_count == 2


@pytest.mark.unit
def test_expired_cache_triggers_refetch(monkeypatch):
    """Once the TTL has elapsed, the next call fetches fresh data."""
    mock = _make_mock_exchange(n=200)
    set_exchange(mock)

    # First fetch populates cache
    fetch_candles(LTF_TF)
    assert mock.fetch_ohlcv.call_count == 1

    # Advance time past TTL by patching time.time
    original_time = time.time()
    monkeypatch.setattr(
        "phemex_feed.time.time",
        lambda: original_time + phemex_feed._TTL[LTF_TF] + 1,
    )

    fetch_candles(LTF_TF)
    assert mock.fetch_ohlcv.call_count == 2


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_fetch_candles_returns_none_on_network_error():
    """NetworkError is caught and None is returned so the tick can skip."""
    mock = MagicMock(spec=ccxt.phemex)
    mock.fetch_ohlcv.side_effect = ccxt.NetworkError("connection reset")
    set_exchange(mock)

    result = fetch_candles(LTF_TF)

    assert result is None


@pytest.mark.unit
def test_fetch_candles_returns_none_on_exchange_error():
    """ExchangeError (e.g. auth failure) is caught and None is returned."""
    mock = MagicMock(spec=ccxt.phemex)
    mock.fetch_ohlcv.side_effect = ccxt.ExchangeError("invalid signature")
    set_exchange(mock)

    result = fetch_candles(LTF_TF)

    assert result is None


@pytest.mark.unit
def test_fetch_candles_returns_none_on_unexpected_exception():
    """Any unexpected exception is caught and None is returned."""
    mock = MagicMock(spec=ccxt.phemex)
    mock.fetch_ohlcv.side_effect = RuntimeError("unknown crash")
    set_exchange(mock)

    result = fetch_candles(LTF_TF)

    assert result is None


@pytest.mark.unit
def test_fetch_candles_returns_none_on_empty_response():
    """An empty list from the exchange returns None instead of an empty DataFrame."""
    mock = MagicMock(spec=ccxt.phemex)
    mock.fetch_ohlcv.return_value = []
    set_exchange(mock)

    result = fetch_candles(LTF_TF)

    assert result is None


@pytest.mark.unit
def test_fetch_candles_raises_on_unsupported_timeframe():
    """Unsupported timeframe raises ValueError immediately."""
    mock = _make_mock_exchange()
    set_exchange(mock)

    with pytest.raises(ValueError, match="Unsupported timeframe"):
        fetch_candles("5m")


# ---------------------------------------------------------------------------
# Candle limits per timeframe
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_ltf_requests_correct_limit():
    """fetch_candles('15m') requests LTF_LIMIT candles from the exchange."""
    mock = _make_mock_exchange(n=LTF_LIMIT)
    set_exchange(mock)

    fetch_candles(LTF_TF)

    mock.fetch_ohlcv.assert_called_once_with(
        phemex_feed.SYMBOL, timeframe=LTF_TF, limit=LTF_LIMIT
    )


@pytest.mark.unit
def test_mtf_requests_correct_limit():
    """fetch_candles('1h') requests MTF_LIMIT candles from the exchange."""
    mock = _make_mock_exchange(n=MTF_LIMIT)
    set_exchange(mock)

    fetch_candles(MTF_TF)

    mock.fetch_ohlcv.assert_called_once_with(
        phemex_feed.SYMBOL, timeframe=MTF_TF, limit=MTF_LIMIT
    )


@pytest.mark.unit
def test_htf_requests_correct_limit():
    """fetch_candles('4h') requests HTF_LIMIT candles from the exchange."""
    mock = _make_mock_exchange(n=HTF_LIMIT)
    set_exchange(mock)

    fetch_candles(HTF_TF)

    mock.fetch_ohlcv.assert_called_once_with(
        phemex_feed.SYMBOL, timeframe=HTF_TF, limit=HTF_LIMIT
    )


# ---------------------------------------------------------------------------
# Read-only DataFrame
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_returned_dataframe_is_readonly():
    """Numeric columns in the returned DataFrame are not writeable."""
    mock = _make_mock_exchange(n=200)
    set_exchange(mock)

    df = fetch_candles(LTF_TF)

    for col in ("open", "high", "low", "close", "volume"):
        arr = df[col].values
        assert not arr.flags.writeable, (
            f"Column '{col}' should be read-only but writeable flag is True"
        )


@pytest.mark.unit
def test_mutating_readonly_dataframe_raises():
    """Attempting in-place mutation of the returned DataFrame raises ValueError."""
    mock = _make_mock_exchange(n=200)
    set_exchange(mock)

    df = fetch_candles(LTF_TF)

    with pytest.raises(ValueError):
        df["close"].values[0] = 99999.0


# ---------------------------------------------------------------------------
# fetch_all()
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_fetch_all_returns_all_three_timeframes():
    """fetch_all() returns a dict with keys for all three timeframes."""
    mock = _make_mock_exchange(n=200)
    set_exchange(mock)

    result = fetch_all()

    assert set(result.keys()) == {LTF_TF, MTF_TF, HTF_TF}
    for tf, df in result.items():
        assert df is not None, f"Timeframe {tf!r} returned None"
        assert isinstance(df, pd.DataFrame)


@pytest.mark.unit
def test_fetch_all_on_partial_error_returns_none_for_failed_timeframe():
    """If one timeframe fails, fetch_all returns None for that TF only."""
    def side_effect(symbol, timeframe, limit):
        if timeframe == MTF_TF:
            raise ccxt.NetworkError("timeout")
        return make_phemex_ohlcv_raw(n=limit)

    mock = MagicMock(spec=ccxt.phemex)
    mock.fetch_ohlcv.side_effect = side_effect
    set_exchange(mock)

    result = fetch_all()

    assert result[LTF_TF] is not None
    assert result[MTF_TF] is None
    assert result[HTF_TF] is not None
