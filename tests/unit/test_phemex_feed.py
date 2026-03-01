"""
Unit tests for phemex_feed.py

Tests:
  - fetch_candles returns a correctly shaped DataFrame
  - Column names and dtypes are correct
  - TTL cache is hit on the second call (no second HTTP call)
  - Returns None on HTTPStatusError
  - Returns None on RequestError (network failure)
  - Returns None on unexpected exception
  - LTF fetches 200 rows, MTF 150, HTF 100
  - Returned DataFrame is read-only (numpy arrays are not writeable)
  - Raises ValueError for unsupported timeframe
  - clear_cache() forces a fresh fetch on the next call
"""

import time
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import httpx

import phemex_feed
from phemex_feed import (
    fetch_candles,
    fetch_all,
    clear_cache,
    OHLCV_COLUMNS,
    LTF_TF,
    MTF_TF,
    HTF_TF,
    LTF_LIMIT,
    MTF_LIMIT,
    HTF_LIMIT,
    MEXC_KLINES_URL,
    SYMBOL,
    _MEXC_INTERVAL,
)
from tests.fixtures.candles import make_phemex_ohlcv_raw


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_response(n: int = 200, start_price: float = 40000.0) -> MagicMock:
    """Return a mock httpx.Response whose .json() returns raw OHLCV data."""
    mock = MagicMock()
    mock.json.return_value = make_phemex_ohlcv_raw(n=n, start_price=start_price)
    mock.raise_for_status.return_value = None
    return mock


@pytest.fixture(autouse=True)
def reset_feed():
    """Reset phemex_feed module state before every test."""
    clear_cache()
    yield
    clear_cache()


# ---------------------------------------------------------------------------
# Shape and schema
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_fetch_candles_returns_dataframe():
    """fetch_candles returns a non-empty pandas DataFrame."""
    with patch("httpx.get", return_value=_make_mock_response(n=200)):
        df = fetch_candles(LTF_TF)

    assert df is not None
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 200


@pytest.mark.unit
def test_fetch_candles_correct_columns():
    """Returned DataFrame has the standard OHLCV column set."""
    with patch("httpx.get", return_value=_make_mock_response(n=200)):
        df = fetch_candles(LTF_TF)

    assert list(df.columns) == OHLCV_COLUMNS


@pytest.mark.unit
def test_fetch_candles_correct_dtypes():
    """open_time is datetime64 UTC; numeric columns are float64."""
    with patch("httpx.get", return_value=_make_mock_response(n=200)):
        df = fetch_candles(LTF_TF)

    assert pd.api.types.is_datetime64_any_dtype(df["open_time"])
    for col in ("open", "high", "low", "close", "volume"):
        assert df[col].dtype == np.float64, f"Column {col!r} is not float64"


# ---------------------------------------------------------------------------
# TTL cache behaviour
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_fetch_candles_cache_hit_on_second_call():
    """The second call within TTL returns cached data without calling MEXC."""
    mock_get = MagicMock(return_value=_make_mock_response(n=200))
    with patch("httpx.get", mock_get):
        df1 = fetch_candles(LTF_TF)
        df2 = fetch_candles(LTF_TF)

    assert mock_get.call_count == 1
    assert df1 is df2


@pytest.mark.unit
def test_clear_cache_forces_refetch():
    """After clear_cache(), the next fetch calls MEXC again."""
    mock_get = MagicMock(return_value=_make_mock_response(n=200))
    with patch("httpx.get", mock_get):
        fetch_candles(LTF_TF)
        clear_cache()
        fetch_candles(LTF_TF)

    assert mock_get.call_count == 2


@pytest.mark.unit
def test_expired_cache_triggers_refetch(monkeypatch):
    """Once the TTL has elapsed, the next call fetches fresh data."""
    mock_get = MagicMock(return_value=_make_mock_response(n=200))
    with patch("httpx.get", mock_get):
        fetch_candles(LTF_TF)
        assert mock_get.call_count == 1

        original_time = time.time()
        monkeypatch.setattr(
            "phemex_feed.time.time",
            lambda: original_time + phemex_feed._TTL[LTF_TF] + 1,
        )
        fetch_candles(LTF_TF)

    assert mock_get.call_count == 2


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_fetch_candles_returns_none_on_http_error():
    """HTTPStatusError is caught and None is returned so the tick can skip."""
    mock_resp = MagicMock()
    mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
        "404 Not Found", request=MagicMock(), response=MagicMock()
    )
    with patch("httpx.get", return_value=mock_resp):
        result = fetch_candles(LTF_TF)

    assert result is None


@pytest.mark.unit
def test_fetch_candles_returns_none_on_network_error():
    """RequestError (network failure) is caught and None is returned."""
    with patch("httpx.get", side_effect=httpx.RequestError(
        "connection reset", request=MagicMock()
    )):
        result = fetch_candles(LTF_TF)

    assert result is None


@pytest.mark.unit
def test_fetch_candles_returns_none_on_unexpected_exception():
    """Any unexpected exception is caught and None is returned."""
    with patch("httpx.get", side_effect=RuntimeError("unknown crash")):
        result = fetch_candles(LTF_TF)

    assert result is None


@pytest.mark.unit
def test_fetch_candles_returns_none_on_empty_response():
    """An empty list from MEXC returns None instead of an empty DataFrame."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = []
    mock_resp.raise_for_status.return_value = None
    with patch("httpx.get", return_value=mock_resp):
        result = fetch_candles(LTF_TF)

    assert result is None


@pytest.mark.unit
def test_fetch_candles_raises_on_unsupported_timeframe():
    """Unsupported timeframe raises ValueError immediately."""
    with pytest.raises(ValueError, match="Unsupported timeframe"):
        fetch_candles("5m")


# ---------------------------------------------------------------------------
# Candle limits per timeframe
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_ltf_requests_correct_limit():
    """fetch_candles('15m') requests LTF_LIMIT candles with the right params."""
    mock_get = MagicMock(return_value=_make_mock_response(n=LTF_LIMIT))
    with patch("httpx.get", mock_get):
        fetch_candles(LTF_TF)

    mock_get.assert_called_once_with(
        MEXC_KLINES_URL,
        params={"symbol": SYMBOL, "interval": _MEXC_INTERVAL[LTF_TF], "limit": LTF_LIMIT},
        timeout=10.0,
    )


@pytest.mark.unit
def test_mtf_requests_correct_limit():
    """fetch_candles('1h') requests MTF_LIMIT candles with the right params."""
    mock_get = MagicMock(return_value=_make_mock_response(n=MTF_LIMIT))
    with patch("httpx.get", mock_get):
        fetch_candles(MTF_TF)

    mock_get.assert_called_once_with(
        MEXC_KLINES_URL,
        params={"symbol": SYMBOL, "interval": _MEXC_INTERVAL[MTF_TF], "limit": MTF_LIMIT},
        timeout=10.0,
    )


@pytest.mark.unit
def test_htf_requests_correct_limit():
    """fetch_candles('4h') requests HTF_LIMIT candles with the right params."""
    mock_get = MagicMock(return_value=_make_mock_response(n=HTF_LIMIT))
    with patch("httpx.get", mock_get):
        fetch_candles(HTF_TF)

    mock_get.assert_called_once_with(
        MEXC_KLINES_URL,
        params={"symbol": SYMBOL, "interval": _MEXC_INTERVAL[HTF_TF], "limit": HTF_LIMIT},
        timeout=10.0,
    )


# ---------------------------------------------------------------------------
# Read-only DataFrame
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_returned_dataframe_is_readonly():
    """Numeric columns in the returned DataFrame are not writeable."""
    with patch("httpx.get", return_value=_make_mock_response(n=200)):
        df = fetch_candles(LTF_TF)

    for col in ("open", "high", "low", "close", "volume"):
        arr = df[col].values
        assert not arr.flags.writeable, (
            f"Column '{col}' should be read-only but writeable flag is True"
        )


@pytest.mark.unit
def test_mutating_readonly_dataframe_raises():
    """Attempting in-place mutation of the returned DataFrame raises ValueError."""
    with patch("httpx.get", return_value=_make_mock_response(n=200)):
        df = fetch_candles(LTF_TF)

    with pytest.raises(ValueError):
        df["close"].values[0] = 99999.0


# ---------------------------------------------------------------------------
# fetch_all()
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_fetch_all_returns_all_three_timeframes():
    """fetch_all() returns a dict with keys for all three timeframes."""
    mock_get = MagicMock(return_value=_make_mock_response(n=200))
    with patch("httpx.get", mock_get):
        result = fetch_all()

    assert set(result.keys()) == {LTF_TF, MTF_TF, HTF_TF}
    for tf, df in result.items():
        assert df is not None, f"Timeframe {tf!r} returned None"
        assert isinstance(df, pd.DataFrame)


@pytest.mark.unit
def test_fetch_all_on_partial_error_returns_none_for_failed_timeframe():
    """If one timeframe fails, fetch_all returns None for that TF only."""
    def side_effect(url, params, timeout):
        if params["interval"] == _MEXC_INTERVAL[MTF_TF]:
            raise httpx.RequestError("timeout", request=MagicMock())
        resp = MagicMock()
        resp.json.return_value = make_phemex_ohlcv_raw(n=params["limit"])
        resp.raise_for_status.return_value = None
        return resp

    with patch("httpx.get", side_effect=side_effect):
        result = fetch_all()

    assert result[LTF_TF] is not None
    assert result[MTF_TF] is None
    assert result[HTF_TF] is not None
