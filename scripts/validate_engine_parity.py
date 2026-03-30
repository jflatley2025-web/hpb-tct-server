"""
scripts/validate_engine_parity.py — Decision Engine Parity Validation
======================================================================

Pulls the last N TAKE signals from a completed/running backtest run and
re-evaluates each one through decision_engine_v2.decide() to verify that
the unified engine produces identical decisions to the original backtest.

Usage:
    # Run against the most recently completed run
    python -m scripts.validate_engine_parity

    # Run against a specific run ID
    python -m scripts.validate_engine_parity --run-id 29

    # Increase signal sample size
    python -m scripts.validate_engine_parity --run-id 29 --limit 500

    # Verbose: show every mismatch in detail
    python -m scripts.validate_engine_parity --verbose

Pass criteria: decision_match_rate >= 99%.

Output:
    PASS/FAIL summary with:
      - Total signals compared
      - Exact matches (TAKE→TAKE, PASS→PASS)
      - TAKE→PASS mismatches (new engine is stricter — expected)
      - PASS→TAKE mismatches (new engine is looser — investigate)
      - Failure code distribution for mismatches

NOTE: This script re-runs schematic detection on live candle data, not
on the exact historical candle slices used during the backtest. Minor
differences in detection output are possible when market structure has
evolved; these are NOT logic drift. True parity violations are cases
where the score + gate inputs are identical but the engine decides
differently.
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import pandas as pd
import psycopg2

# Add project root to path
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.config import (
    DB_HOST, DB_NAME, DB_PASSWORD, DB_PORT, DB_USER,
    ENTRY_THRESHOLD, MIN_RR, MTF_TIMEFRAMES,
)
from backtest.ingest import load_candles

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("parity_check")


# ── DB helpers ────────────────────────────────────────────────────────

def get_conn():
    return psycopg2.connect(
        host=DB_HOST, dbname=DB_NAME, user=DB_USER,
        password=DB_PASSWORD, port=DB_PORT,
    )


def fetch_signals(conn, run_id: int, limit: int) -> List[Dict]:
    """Pull the last `limit` signals (all decisions) for a given run."""
    cur = conn.cursor()
    cur.execute(
        """
        SELECT
            signal_time, timeframe, direction, model,
            final_decision, failure_code, score_1d,
            rcm_score, rcm_valid, local_displacement,
            rig_status, htf_bias, msce_session,
            entry_price, stop_price, target_price, rr,
            range_duration_hours, schematic_json
        FROM backtest_signals
        WHERE run_id = %s
        ORDER BY signal_time DESC
        LIMIT %s
        """,
        (run_id, limit),
    )
    cols = [d[0] for d in cur.description]
    rows = [dict(zip(cols, row)) for row in cur.fetchall()]
    cur.close()
    return rows


def fetch_run_info(conn, run_id: int) -> Dict:
    cur = conn.cursor()
    cur.execute(
        "SELECT start_date, end_date, status, total_trades FROM backtest_runs WHERE id = %s",
        (run_id,),
    )
    row = cur.fetchone()
    cur.close()
    if not row:
        return {}
    return {
        "start_date": row[0], "end_date": row[1],
        "status": row[2], "total_trades": row[3],
    }


def get_latest_run_id(conn) -> Optional[int]:
    cur = conn.cursor()
    cur.execute("SELECT id FROM backtest_runs ORDER BY id DESC LIMIT 1")
    row = cur.fetchone()
    cur.close()
    return row[0] if row else None


# ── Candle loading ────────────────────────────────────────────────────

_candle_cache: Dict[str, pd.DataFrame] = {}


def load_candles_for_time(conn, symbol: str, tf: str, signal_time: datetime) -> Optional[pd.DataFrame]:
    """
    Load candles up to (but not including) signal_time for a given TF.
    Uses a per-TF cache — loads once and slices per signal_time.
    """
    cache_key = f"{symbol}_{tf}"
    if cache_key not in _candle_cache:
        try:
            df = load_candles(
                conn,
                symbol=symbol,
                timeframe=tf,
                start_date=None,  # full history
                end_date=None,
            )
            if df is not None and len(df) > 0:
                _candle_cache[cache_key] = df
            else:
                _candle_cache[cache_key] = pd.DataFrame()
        except Exception as e:
            logger.warning("Could not load %s %s candles: %s", symbol, tf, e)
            _candle_cache[cache_key] = pd.DataFrame()

    full_df = _candle_cache[cache_key]
    if full_df.empty:
        return None

    # Return only fully-closed candles before signal_time.
    # Prefer close_time (exact); fall back to open_time only if absent.
    if "close_time" in full_df.columns:
        sliced = full_df[full_df["close_time"] < signal_time].copy()
    elif "open_time" in full_df.columns:
        sliced = full_df[full_df["open_time"] < signal_time].copy()
    else:
        return full_df
    return sliced if len(sliced) >= 10 else None


# ── Comparison logic ──────────────────────────────────────────────────

def _classify_mismatch(orig: str, v2: str) -> str:
    """Classify the direction of a mismatch."""
    if orig == "TAKE" and v2 == "PASS":
        return "TAKE→PASS (v2 stricter)"
    if orig == "SKIP" and v2 == "TAKE":
        return "PASS→TAKE (v2 looser)"
    return f"{orig}→{v2}"


class NoSignalsFound(Exception):
    pass


def run_parity_check(
    run_id: int,
    limit: int = 200,
    verbose: bool = False,
    symbol: str = "BTCUSDT",
) -> Dict:
    """
    Main parity check entry point.

    Returns a result dict with match_rate, mismatch details, etc.
    Raises NoSignalsFound if the run exists but has no signals in the DB.
    Raises ValueError if the run_id is not found.
    """
    from decision_engine_v2 import decide

    conn = get_conn()
    try:
        run_info = fetch_run_info(conn, run_id)
        if not run_info:
            raise ValueError(f"Run ID {run_id} not found in backtest_runs")

        logger.info(
            "Parity check: run_id=%d symbol=%s status=%s trades=%s — fetching %d signals",
            run_id, symbol, run_info.get("status"),
            run_info.get("total_trades"), limit,
        )

        signals = fetch_signals(conn, run_id, limit)

        if not signals:
            raise NoSignalsFound(f"No signals found for run_id={run_id}")

        logger.info("Fetched %d signals — loading candle history...", len(signals))

        results = {
            "run_id": run_id,
            "signals_compared": 0,
            "exact_matches": 0,
            "take_to_pass": 0,          # v2 is stricter (expected)
            "pass_to_take": 0,          # v2 is looser (investigate)
            "errors": 0,
            "match_rate": 0.0,
            "mismatch_details": [],
            "failure_code_counts": {},
        }

        for sig in signals:
            signal_time = sig["signal_time"]
            if signal_time.tzinfo is None:
                signal_time = signal_time.replace(tzinfo=timezone.utc)

            orig_decision = sig["final_decision"]  # "TAKE" or "SKIP"
            orig_score = sig.get("score_1d", 0) or 0

            # Build candles_by_tf from stored history at signal_time
            candles_by_tf: Dict[str, pd.DataFrame] = {}
            for tf in MTF_TIMEFRAMES + ["1d", "4h"]:
                df = load_candles_for_time(conn, symbol, tf, signal_time)
                if df is not None:
                    candles_by_tf[tf] = df

            if not candles_by_tf:
                results["errors"] += 1
                continue

            # Build context (stateless — no DD protection or BOS dedup for parity test)
            ctx = {
                "current_price": float(sig.get("entry_price") or 0) or 0.0,
                "current_time": signal_time,
                "entry_threshold": ENTRY_THRESHOLD,
                "min_rr": MIN_RR,
            }
            # Use stored entry price as current_price proxy; fall back to 0
            if ctx["current_price"] == 0 and sig.get("stop_price"):
                ctx["current_price"] = float(sig["stop_price"])

            try:
                v2_result = decide(candles_by_tf, ctx)
            except Exception as e:
                logger.warning("decide() error for signal @ %s: %s", signal_time, e)
                results["errors"] += 1
                continue

            v2_decision = v2_result["decision"]  # "TAKE" or "PASS"
            # Normalise: backtest uses "SKIP", engine uses "PASS"
            orig_normalised = "TAKE" if orig_decision == "TAKE" else "PASS"

            results["signals_compared"] += 1

            if orig_normalised == v2_decision:
                results["exact_matches"] += 1
            else:
                mismatch_type = _classify_mismatch(orig_normalised, v2_decision)
                if orig_normalised == "TAKE" and v2_decision == "PASS":
                    results["take_to_pass"] += 1
                else:
                    results["pass_to_take"] += 1

                fc = v2_result.get("failure_code") or "none"
                results["failure_code_counts"][fc] = results["failure_code_counts"].get(fc, 0) + 1

                detail = {
                    "signal_time": signal_time.isoformat(),
                    "tf": sig.get("timeframe"),
                    "model": sig.get("model"),
                    "direction": sig.get("direction"),
                    "orig_decision": orig_normalised,
                    "v2_decision": v2_decision,
                    "mismatch_type": mismatch_type,
                    "orig_score": orig_score,
                    "v2_score": v2_result.get("score", 0),
                    "v2_failure_code": fc,
                    "v2_reason": v2_result.get("reason"),
                    "orig_failure_code": sig.get("failure_code"),
                }
                results["mismatch_details"].append(detail)

                if verbose:
                    logger.warning(
                        "MISMATCH @ %s | %s %s %s | orig=%s v2=%s | "
                        "orig_score=%s v2_score=%s | v2_fc=%s",
                        signal_time.isoformat(),
                        sig.get("timeframe"), sig.get("model"), sig.get("direction"),
                        orig_normalised, v2_decision,
                        orig_score, v2_result.get("score", 0), fc,
                    )

        total = results["signals_compared"]
        if total > 0:
            results["match_rate"] = round(results["exact_matches"] / total * 100, 2)

        return results
    finally:
        conn.close()


def _print_report(r: Dict):
    total = r["signals_compared"]
    if total == 0:
        print("ERROR: No signals could be compared.")
        return

    print("\n" + "=" * 60)
    print("PARITY VALIDATION REPORT")
    print("=" * 60)
    print(f"  Run ID:           {r['run_id']}")
    print(f"  Signals compared: {total}")
    print(f"  Exact matches:    {r['exact_matches']} ({r['match_rate']:.2f}%)")
    print(f"  TAKE→PASS:        {r['take_to_pass']}  (v2 stricter — expected)")
    print(f"  PASS→TAKE:        {r['pass_to_take']}  (v2 looser  — investigate)")
    print(f"  Errors:           {r['errors']}")
    print()

    if r["failure_code_counts"]:
        print("  Failure code distribution (mismatches):")
        for fc, cnt in sorted(r["failure_code_counts"].items(), key=lambda x: -x[1]):
            print(f"    {fc:<40} {cnt}")
    print()

    pass_threshold = 99.0
    min_signals = 100
    total = r["signals_compared"]
    if total < min_signals:
        print(f"  RESULT: ✗ FAIL  (insufficient signals: {total} compared < {min_signals} required)")
        print(f"           Increase --limit or ensure the run has >= {min_signals} signals.")
    elif r["match_rate"] >= pass_threshold:
        print(f"  RESULT: ✓ PASS  (match_rate={r['match_rate']:.2f}% >= {pass_threshold}%)")
    else:
        print(f"  RESULT: ✗ FAIL  (match_rate={r['match_rate']:.2f}% < {pass_threshold}%)")
        print()
        if r["pass_to_take"]:
            print("  ACTION REQUIRED: PASS→TAKE mismatches mean v2 is accepting signals")
            print("  the backtest would have blocked. These need investigation before")
            print("  enabling USE_UNIFIED_ENGINE = True.")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Validate decision_engine_v2 parity")
    parser.add_argument("--run-id", type=int, default=None, help="Backtest run ID (default: latest)")
    parser.add_argument("--limit", type=int, default=200, help="Max signals to compare (default: 200)")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Symbol (default: BTCUSDT)")
    parser.add_argument("--verbose", action="store_true", help="Print each mismatch")
    parser.add_argument("--json", dest="as_json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    # Resolve run_id
    run_id = args.run_id
    if run_id is None:
        conn = get_conn()
        run_id = get_latest_run_id(conn)
        conn.close()
        if run_id is None:
            logger.error("No backtest runs found in DB")
            sys.exit(1)
        logger.info("No --run-id specified, using latest: run_id=%d", run_id)

    try:
        result = run_parity_check(
            run_id=run_id,
            limit=args.limit,
            verbose=args.verbose,
            symbol=args.symbol,
        )
    except NoSignalsFound as e:
        logger.error("%s", e)
        sys.exit(1)

    if args.as_json:
        # Serialize datetimes for JSON output
        print(json.dumps(result, default=str, indent=2))
    else:
        _print_report(result)

    # Exit 1 if parity check failed — requires both minimum signal count AND match rate.
    # A 100% match rate on 5 signals is meaningless; enforce the floor before trusting the rate.
    if result.get("signals_compared", 0) < 100 or result.get("match_rate", 0) < 99.0:
        sys.exit(1)


if __name__ == "__main__":
    main()
