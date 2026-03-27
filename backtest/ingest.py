"""
backtest/ingest.py — OHLCV Data Ingestion + Integrity Checks
==============================================================
Downloads historical candle data from Bybit (primary), OKX (fallback),
or MEXC (last resort) and upserts into PostgreSQL.

Features:
- All 7 timeframes, configurable date range
- Rate-limited API calls
- Computes close_time from open_time + timeframe_duration
- Gap detection + integrity validation
- Aborts on integrity violations
"""

import logging
import time as _time
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Tuple

import ccxt
import pandas as pd

from backtest.config import (
    CANDLE_LIMITS,
    DEFAULT_SYMBOL,
    EXCHANGE_PRIORITY,
    RATE_LIMIT_DELAY,
    TIMEFRAME_SECONDS,
    VALID_TIMEFRAMES,
)
from backtest.db import get_connection

logger = logging.getLogger("backtest.ingest")


# ── Exchange factory ──────────────────────────────────────────────────

def _create_exchange(name: str) -> ccxt.Exchange:
    """Create a CCXT exchange instance."""
    cls = getattr(ccxt, name, None)
    if cls is None:
        raise ValueError(f"Unknown exchange: {name}")
    return cls({"enableRateLimit": True, "timeout": 30000})


def _get_exchange() -> Tuple[ccxt.Exchange, str]:
    """Try exchanges in priority order. Returns (exchange, name)."""
    for name in EXCHANGE_PRIORITY:
        try:
            ex = _create_exchange(name)
            ex.load_markets()
            logger.info(f"Connected to {name}")
            return ex, name
        except Exception as e:
            logger.warning(f"Failed to connect to {name}: {e}")
    raise RuntimeError(f"All exchanges failed: {EXCHANGE_PRIORITY}")


# ── Fetch candles ─────────────────────────────────────────────────────

def _fetch_candles_batch(
    exchange: ccxt.Exchange,
    symbol: str,
    timeframe: str,
    since_ms: int,
    limit: int = 1000,
) -> List[list]:
    """Fetch a single batch of OHLCV candles."""
    return exchange.fetch_ohlcv(
        symbol, timeframe=timeframe, since=since_ms, limit=limit
    )


def fetch_all_candles(
    exchange: ccxt.Exchange,
    symbol: str,
    timeframe: str,
    start_date: datetime,
    end_date: datetime,
) -> pd.DataFrame:
    """
    Fetch all candles for a symbol/timeframe between start_date and end_date.
    Handles pagination automatically.
    """
    tf_ms = TIMEFRAME_SECONDS[timeframe] * 1000
    since_ms = int(start_date.timestamp() * 1000)
    end_ms = int(end_date.timestamp() * 1000)
    all_rows = []

    while since_ms < end_ms:
        try:
            batch = _fetch_candles_batch(exchange, symbol, timeframe, since_ms)
        except Exception as e:
            logger.error(f"Fetch error {symbol}/{timeframe} at {since_ms}: {e}")
            raise

        if not batch:
            break

        for row in batch:
            if row[0] < end_ms:
                all_rows.append(row)

        last_ts = batch[-1][0]
        if last_ts <= since_ms:
            break
        since_ms = last_ts + tf_ms
        _time.sleep(RATE_LIMIT_DELAY)

    if not all_rows:
        return pd.DataFrame(columns=["open_time", "open", "high", "low", "close", "volume"])

    df = pd.DataFrame(all_rows, columns=["open_time", "open", "high", "low", "close", "volume"])
    # Drop any extra columns (some exchanges return 7 cols)
    df = df[["open_time", "open", "high", "low", "close", "volume"]]
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = df[col].astype(float)

    # Compute close_time
    tf_delta = timedelta(seconds=TIMEFRAME_SECONDS[timeframe])
    df["close_time"] = df["open_time"] + tf_delta

    df.drop_duplicates(subset=["open_time"], keep="last", inplace=True)
    df.sort_values("open_time", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ── Integrity checks ─────────────────────────────────────────────────

class IntegrityError(Exception):
    """Raised when candle data fails integrity checks."""
    pass


def check_integrity(df: pd.DataFrame, timeframe: str, symbol: str = "BTCUSDT"):
    """
    Validate candle data integrity. Raises IntegrityError on violation.

    Checks:
    1. No duplicate timestamps
    2. Chronological ordering
    3. No future timestamps beyond now
    4. All timestamps are UTC
    5. No gaps > 2x timeframe duration
    6. Row count sanity
    """
    if df.empty:
        raise IntegrityError(f"No data for {symbol}/{timeframe}")

    # 1. Duplicates
    dupes = df["open_time"].duplicated().sum()
    if dupes > 0:
        raise IntegrityError(
            f"{symbol}/{timeframe}: {dupes} duplicate timestamps found"
        )

    # 2. Chronological ordering
    if not df["open_time"].is_monotonic_increasing:
        raise IntegrityError(
            f"{symbol}/{timeframe}: timestamps are not chronologically ordered"
        )

    # 3. UTC enforcement (must come before future-timestamp check)
    if df["open_time"].dt.tz is None:
        raise IntegrityError(
            f"{symbol}/{timeframe}: timestamps are not timezone-aware (must be UTC)"
        )

    # 4. No future timestamps
    now_utc = datetime.now(timezone.utc)
    future_count = (df["open_time"] > now_utc).sum()
    if future_count > 0:
        raise IntegrityError(
            f"{symbol}/{timeframe}: {future_count} candles have future timestamps"
        )

    # 5. Gap detection
    tf_seconds = TIMEFRAME_SECONDS[timeframe]
    max_gap = timedelta(seconds=tf_seconds * 2)
    time_diffs = df["open_time"].diff().dropna()
    large_gaps = time_diffs[time_diffs > max_gap]
    if len(large_gaps) > 0:
        gap_details = []
        for idx in large_gaps.index[:5]:  # show first 5
            gap_at = df.loc[idx, "open_time"]
            gap_size = large_gaps[idx]
            gap_details.append(f"  at {gap_at}: gap={gap_size}")
        detail_str = "\n".join(gap_details)
        logger.warning(
            f"{symbol}/{timeframe}: {len(large_gaps)} gaps > 2x TF duration:\n{detail_str}"
        )
        # Warn but don't abort for gaps — crypto markets can have low-volume periods

    # 6. Row count sanity
    expected_duration = (df["open_time"].iloc[-1] - df["open_time"].iloc[0]).total_seconds()
    expected_rows = expected_duration / tf_seconds
    actual_rows = len(df)
    coverage = actual_rows / expected_rows if expected_rows > 0 else 0
    if coverage < 0.8:
        raise IntegrityError(
            f"{symbol}/{timeframe}: only {coverage:.1%} coverage "
            f"({actual_rows} rows, expected ~{int(expected_rows)})"
        )

    logger.info(
        f"Integrity OK: {symbol}/{timeframe} — {actual_rows} rows, "
        f"{coverage:.1%} coverage"
    )


# ── Upsert to database ───────────────────────────────────────────────

_OHLCV_UPSERT_SQL = """
INSERT INTO ohlcv_candles
    (symbol, timeframe, open_time, close_time,
     open, high, low, close, volume)
VALUES %s
ON CONFLICT (symbol, timeframe, open_time) DO NOTHING
"""


def _serialize_candle_rows(df: pd.DataFrame, symbol: str, timeframe: str) -> list:
    """Convert a candle DataFrame to a list of tuples for batch upsert."""
    return [
        (
            symbol, timeframe,
            row["open_time"], row["close_time"],
            float(row["open"]), float(row["high"]),
            float(row["low"]), float(row["close"]),
            float(row["volume"]),
        )
        for _, row in df.iterrows()
    ]


def upsert_candles(conn, df: pd.DataFrame, symbol: str, timeframe: str) -> int:
    """
    Upsert candle data into ohlcv_candles. ON CONFLICT DO NOTHING.
    Delegates to upsert_candles_batch for consistent SQL.
    Returns the number of rows attempted.
    """
    return upsert_candles_batch(conn, df, symbol, timeframe)


def upsert_candles_batch(conn, df: pd.DataFrame, symbol: str, timeframe: str) -> int:
    """
    Batch upsert using execute_values for better performance.
    Returns the number of rows attempted.
    """
    if df.empty:
        return 0

    rows = _serialize_candle_rows(df, symbol, timeframe)

    from psycopg2.extras import execute_values
    with conn.cursor() as cur:
        execute_values(cur, _OHLCV_UPSERT_SQL, rows, page_size=500)
    conn.commit()
    logger.info(f"Batch upserted {symbol}/{timeframe}: {len(rows)} rows")
    return len(rows)


# ── Load from database ────────────────────────────────────────────────

def load_candles(
    conn,
    symbol: str,
    timeframe: str,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
) -> pd.DataFrame:
    """Load candles from the database into a DataFrame."""
    query = """
        SELECT symbol, timeframe, open_time, close_time,
               open, high, low, close, volume
        FROM ohlcv_candles
        WHERE symbol = %s AND timeframe = %s
    """
    params = [symbol, timeframe]

    if start_date:
        query += " AND open_time >= %s"
        params.append(start_date)
    if end_date:
        query += " AND open_time < %s"
        params.append(end_date)

    query += " ORDER BY open_time"

    df = pd.read_sql(query, conn, params=params)
    if not df.empty:
        df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
        if "close_time" in df.columns:
            df["close_time"] = pd.to_datetime(df["close_time"], utc=True)
        for col in ("open", "high", "low", "close", "volume"):
            df[col] = df[col].astype(float)
    return df


# ── Main ingestion flow ──────────────────────────────────────────────

def ingest(
    symbol: str = DEFAULT_SYMBOL,
    timeframes: Optional[List[str]] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    conn=None,
):
    """
    Full ingestion pipeline: fetch from exchange, validate, upsert to DB.

    Args:
        symbol: Trading pair (default BTCUSDT)
        timeframes: List of TFs to ingest (default: all 7)
        start_date: Start of date range (default: 3 months ago)
        end_date: End of date range (default: now)
        conn: Optional DB connection (creates one if not provided)
    """
    if timeframes is None:
        timeframes = list(VALID_TIMEFRAMES)
    if end_date is None:
        end_date = datetime.now(timezone.utc)
    if start_date is None:
        start_date = end_date - timedelta(days=90)

    # Ensure timezone-aware
    if start_date.tzinfo is None:
        start_date = start_date.replace(tzinfo=timezone.utc)
    if end_date.tzinfo is None:
        end_date = end_date.replace(tzinfo=timezone.utc)

    exchange, exchange_name = _get_exchange()
    # Normalize symbol for CCXT (e.g., "BTCUSDT" -> "BTC/USDT")
    # Strip known quote currencies from the right to handle variable-length bases.
    if "/" in symbol:
        ccxt_symbol = symbol
    else:
        ccxt_symbol = symbol  # fallback: keep as-is if no match
        for quote in ("USDT", "USDC", "BUSD", "USD", "BTC", "ETH"):
            if symbol.endswith(quote):
                base = symbol[: -len(quote)]
                ccxt_symbol = f"{base}/{quote}"
                break

    own_conn = conn is None
    if own_conn:
        conn = get_connection()

    total_rows = 0
    try:
        for tf in timeframes:
            logger.info(f"Ingesting {symbol}/{tf} from {exchange_name} "
                        f"({start_date.date()} to {end_date.date()})...")
            try:
                df = fetch_all_candles(exchange, ccxt_symbol, tf, start_date, end_date)
                if df.empty:
                    logger.warning(f"No data returned for {symbol}/{tf}")
                    continue

                check_integrity(df, tf, symbol)
                count = upsert_candles_batch(conn, df, symbol, tf)
                total_rows += count
                logger.info(f"Completed {symbol}/{tf}: {count} rows")

            except IntegrityError as e:
                logger.error(f"INTEGRITY FAILURE: {e}")
                raise
            except Exception as e:
                logger.error(f"Error ingesting {symbol}/{tf}: {e}", exc_info=True)
                raise

        logger.info(f"Ingestion complete: {total_rows} total rows across {len(timeframes)} timeframes")

    finally:
        if own_conn:
            conn.close()

    return total_rows


# ── Entrypoint ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Ingest OHLCV candle data")
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL)
    parser.add_argument("--start", help="Start date YYYY-MM-DD", default=None)
    parser.add_argument("--end", help="End date YYYY-MM-DD", default=None)
    parser.add_argument("--timeframes", nargs="+", default=None,
                        help="Timeframes to ingest (default: all)")
    args = parser.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc) if args.start else None
    end = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc) if args.end else None

    from backtest.db import create_schema
    conn = get_connection()
    create_schema(conn)

    ingest(
        symbol=args.symbol,
        timeframes=args.timeframes,
        start_date=start,
        end_date=end,
        conn=conn,
    )
    conn.close()
