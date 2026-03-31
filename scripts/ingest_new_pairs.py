#!/usr/bin/env python3
"""
scripts/ingest_new_pairs.py — Batch ingest historical candles for new trading pairs
====================================================================================
Fetches 1h, 4h, 1d, and 1m candles from 2025-01-01 to now for each new pair
and upserts them into the ohlcv_candles table in PostgreSQL.

Processes pairs sequentially so a single failure does not block the rest.
Validates integrity after each timeframe ingestion.

Usage:
    python -m scripts.ingest_new_pairs
    python -m scripts.ingest_new_pairs --pairs BCHUSDT DOGEUSDT
    python -m scripts.ingest_new_pairs --dry-run
"""

import argparse
import logging
import sys
from datetime import datetime, timezone

from backtest.db import get_connection, create_schema
from backtest.ingest import ingest

logger = logging.getLogger("ingest_new_pairs")

NEW_PAIRS = [
    "BCHUSDT",
    "WIFUSDT",
    "DOGEUSDT",
    "HBARUSDT",
    "FETUSDT",
    "XMRUSDT",
    "FARTCOINUSDT",
    "PEPEUSDT",
    "XRPUSDT",
]

# Required timeframes: 1h, 4h, 1d for live detection; 1m for post-hoc simulations
TIMEFRAMES = ["1d", "4h", "1h", "1m"]

START_DATE = datetime(2025, 1, 1, tzinfo=timezone.utc)


def main():
    parser = argparse.ArgumentParser(description="Ingest historical candles for new trading pairs")
    parser.add_argument(
        "--pairs", nargs="+", default=None,
        help="Specific pairs to ingest (default: all new pairs)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be ingested without running",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    pairs = args.pairs or NEW_PAIRS

    if args.dry_run:
        for symbol in pairs:
            for tf in TIMEFRAMES:
                logger.info("[DRY RUN] Would ingest %s/%s from %s to now", symbol, tf, START_DATE.date())
        return

    conn = get_connection()
    create_schema(conn)

    results = {}
    for symbol in pairs:
        logger.info("=" * 60)
        logger.info("Starting ingestion for %s", symbol)
        logger.info("=" * 60)
        try:
            rows = ingest(
                symbol=symbol,
                timeframes=TIMEFRAMES,
                start_date=START_DATE,
                end_date=datetime.now(timezone.utc),
                conn=conn,
            )
            results[symbol] = {"status": "ok", "rows": rows}
            logger.info("Ingestion complete for %s: %d rows", symbol, rows)
        except Exception as e:
            logger.error("FAILED ingestion for %s: %s", symbol, e, exc_info=True)
            results[symbol] = {"status": "error", "error": str(e)}
            # Continue with remaining pairs per requirement #8

    conn.close()

    # Summary table
    logger.info("")
    logger.info("=" * 60)
    logger.info("INGESTION SUMMARY")
    logger.info("=" * 60)
    logger.info("%-18s %-10s %s", "SYMBOL", "STATUS", "ROWS/ERROR")
    logger.info("-" * 60)
    for symbol in pairs:
        r = results.get(symbol, {"status": "not_run"})
        if r["status"] == "ok":
            logger.info("%-18s %-10s %d rows", symbol, "OK", r["rows"])
        else:
            logger.info("%-18s %-10s %s", symbol, "FAILED", r.get("error", "unknown"))

    failed = [s for s, r in results.items() if r["status"] != "ok"]
    if failed:
        logger.warning("Failed pairs: %s", failed)
        sys.exit(1)


if __name__ == "__main__":
    main()
