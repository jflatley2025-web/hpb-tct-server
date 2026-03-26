"""
tests/unit/test_backtest_ingest.py — Unit Tests for Data Ingestion
===================================================================
Tests integrity checks, upsert idempotency, gap detection,
and data validation without requiring live API calls or DB.
"""

import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pandas as pd
import numpy as np

from backtest.config import TIMEFRAME_SECONDS, VALID_TIMEFRAMES
from backtest.ingest import IntegrityError, check_integrity


# ── Helpers ───────────────────────────────────────────────────────────

def _make_candle_df(
    start: datetime,
    tf: str,
    count: int,
    base_price: float = 50000.0,
    gap_at: int = None,
    gap_size: int = 1,
) -> pd.DataFrame:
    """Create a synthetic candle DataFrame."""
    tf_seconds = TIMEFRAME_SECONDS[tf]
    rows = []
    for i in range(count):
        extra = 0
        if gap_at is not None and i >= gap_at:
            extra = tf_seconds * gap_size  # insert gap
        t = start + timedelta(seconds=tf_seconds * i + extra)
        rows.append({
            "open_time": t,
            "open": base_price + i,
            "high": base_price + i + 50,
            "low": base_price + i - 50,
            "close": base_price + i + 10,
            "volume": 100.0,
        })
    df = pd.DataFrame(rows)
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    return df


# ── Integrity Checks ─────────────────────────────────────────────────

class TestIntegrityChecks:
    """Test all 6 integrity validations."""

    def test_empty_data_fails(self):
        df = pd.DataFrame(columns=["open_time", "open", "high", "low", "close", "volume"])
        with pytest.raises(IntegrityError, match="No data"):
            check_integrity(df, "1h")

    def test_duplicate_timestamps_fail(self):
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        df = _make_candle_df(start, "1h", 10)
        # Insert duplicate
        df = pd.concat([df, df.iloc[[5]]], ignore_index=True)
        df.sort_values("open_time", inplace=True)
        with pytest.raises(IntegrityError, match="duplicate"):
            check_integrity(df, "1h")

    def test_non_chronological_fails(self):
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        df = _make_candle_df(start, "1h", 10)
        # Swap two rows
        df.iloc[3], df.iloc[7] = df.iloc[7].copy(), df.iloc[3].copy()
        with pytest.raises(IntegrityError, match="not chronologically ordered"):
            check_integrity(df, "1h")

    def test_future_timestamps_fail(self):
        # Create candles far in the future
        start = datetime(2099, 1, 1, tzinfo=timezone.utc)
        df = _make_candle_df(start, "1h", 10)
        with pytest.raises(IntegrityError, match="future timestamps"):
            check_integrity(df, "1h")

    def test_naive_timestamps_fail(self):
        start = datetime(2026, 1, 1)  # no tzinfo
        rows = []
        for i in range(10):
            t = start + timedelta(hours=i)
            rows.append({
                "open_time": t,
                "open": 50000 + i, "high": 50050 + i,
                "low": 49950 + i, "close": 50010 + i,
                "volume": 100.0,
            })
        df = pd.DataFrame(rows)
        # pandas without utc=True creates naive timestamps
        with pytest.raises(IntegrityError, match="not timezone-aware"):
            check_integrity(df, "1h")

    def test_low_coverage_fails(self):
        """Data with many gaps should fail coverage check."""
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        # Create 10 candles spanning 100 hours (90% gap)
        rows = []
        for i in range(10):
            t = start + timedelta(hours=i * 10)
            rows.append({
                "open_time": t,
                "open": 50000, "high": 50050,
                "low": 49950, "close": 50010,
                "volume": 100.0,
            })
        df = pd.DataFrame(rows)
        df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
        with pytest.raises(IntegrityError, match="coverage"):
            check_integrity(df, "1h")

    def test_valid_data_passes(self):
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        df = _make_candle_df(start, "1h", 100)
        # Should not raise
        check_integrity(df, "1h")

    def test_valid_5m_data_passes(self):
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        df = _make_candle_df(start, "5m", 200)
        check_integrity(df, "5m")

    def test_valid_4h_data_passes(self):
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        df = _make_candle_df(start, "4h", 50)
        check_integrity(df, "4h")


# ── Gap Detection ─────────────────────────────────────────────────────

class TestGapDetection:
    def test_gaps_logged_but_pass(self):
        """Large gaps are logged as warnings but don't cause failure
        (crypto can have quiet periods)."""
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        df = _make_candle_df(start, "1h", 50, gap_at=25, gap_size=3)
        # Should pass (gaps are warnings, not errors) as long as coverage is OK
        # With gap_size=3, we have ~50 candles over ~53 hours ≈ 94% coverage
        check_integrity(df, "1h")


# ── Config Validation ─────────────────────────────────────────────────

class TestConfigValidation:
    def test_all_timeframes_have_seconds(self):
        for tf in VALID_TIMEFRAMES:
            assert tf in TIMEFRAME_SECONDS

    def test_timeframe_seconds_values(self):
        assert TIMEFRAME_SECONDS["1m"] == 60
        assert TIMEFRAME_SECONDS["5m"] == 300
        assert TIMEFRAME_SECONDS["15m"] == 900
        assert TIMEFRAME_SECONDS["1h"] == 3600
        assert TIMEFRAME_SECONDS["4h"] == 14400
        assert TIMEFRAME_SECONDS["1d"] == 86400


# ── Upsert Idempotency (mocked) ──────────────────────────────────────

class TestUpsertIdempotency:
    @patch("backtest.ingest.get_connection")
    def test_upsert_on_conflict_do_nothing(self, mock_conn):
        """Verify that upsert SQL uses ON CONFLICT DO NOTHING."""
        from backtest.ingest import upsert_candles

        # Create mock connection and cursor
        conn = MagicMock()
        cur = MagicMock()
        conn.cursor.return_value.__enter__ = MagicMock(return_value=cur)
        conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        cur.rowcount = 1

        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        df = _make_candle_df(start, "1h", 3)
        df["close_time"] = df["open_time"] + timedelta(hours=1)

        upsert_candles(conn, df, "BTCUSDT", "1h")

        # Verify execute was called (once per row)
        assert cur.execute.call_count == 3
        # Verify ON CONFLICT in SQL
        sql = cur.execute.call_args_list[0][0][0]
        assert "ON CONFLICT" in sql
        assert "DO NOTHING" in sql
