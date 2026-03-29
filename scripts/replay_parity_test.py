"""
scripts/replay_parity_test.py — Offline Engine Parity Replay Test
==================================================================

Loads historical candles from the database (last 3–7 days by default) and
walks forward step-by-step, running BOTH decision engines at every timestep:

    decision_v2   = decide(candles_by_tf, context)           # unified engine
    decision_bt   = run_gate_pipeline(state, ...)            # backtest engine

Logs every mismatch and prints a summary match rate.

Pass criteria: match_rate >= 99.5%

Usage:
    python -m scripts.replay_parity_test
    python -m scripts.replay_parity_test --days 7 --symbol ETHUSDT
    python -m scripts.replay_parity_test --verbose
    python -m scripts.replay_parity_test --json

The script intentionally uses a FRESH BacktestState with no open trades and
no DD history so the comparison isolates signal-level gate logic only.
State-dependent gates (DD hard block, BOS dedup, compression) are bypassed
in the v2 context via omitted optional fields — matching replay_mode behaviour.
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.config import (
    ENTRY_THRESHOLD,
    MIN_RR,
    MTF_TIMEFRAMES,
    STARTING_BALANCE,
    timeframe_to_seconds,
)
from backtest.db import get_connection, create_schema, create_run, fail_run, complete_run
from backtest.ingest import load_candles
from backtest.runner import BacktestState, run_gate_pipeline
from decision_engine_v2 import decide

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("replay_parity")

PASS_THRESHOLD = 0.995  # 99.5% required for replay parity

# Step interval for the replay walk-forward loop.
# 1h balances granularity vs runtime; each step loads a candle slice up to
# that timestamp for all TFs.
DEFAULT_STEP_HOURS = 1


# ── Candle helpers ────────────────────────────────────────────────────────────

_candle_cache: Dict[str, pd.DataFrame] = {}


def _load_full_history(symbol: str, timeframes: List[str], start: datetime) -> None:
    """Pre-load candle history for all TFs into cache."""
    for tf in timeframes:
        key = f"{symbol}_{tf}"
        if key in _candle_cache:
            continue
        logger.info("Loading %s %s candles...", symbol, tf)
        try:
            df = load_candles(symbol=symbol, timeframe=tf, start_date=start, end_date=None)
            _candle_cache[key] = df if (df is not None and len(df) > 0) else pd.DataFrame()
        except Exception as e:
            logger.warning("Could not load %s %s: %s", symbol, tf, e)
            _candle_cache[key] = pd.DataFrame()


def _get_candles_at(symbol: str, tf: str, up_to: datetime) -> Optional[pd.DataFrame]:
    """Return candles with open_time < up_to (no future leakage)."""
    key = f"{symbol}_{tf}"
    full = _candle_cache.get(key)
    if full is None or full.empty:
        return None
    if "open_time" not in full.columns:
        return full
    sliced = full[full["open_time"] < up_to]
    return sliced if len(sliced) >= 10 else None


def _current_price(candles_by_tf: Dict[str, pd.DataFrame]) -> float:
    """Extract the last close price from the most granular available TF."""
    for tf in ["30m", "1h", "4h", "1d"]:
        df = candles_by_tf.get(tf)
        if df is not None and len(df) > 0 and "close" in df.columns:
            return float(df.iloc[-1]["close"])
    return 0.0


# ── Mismatch logging ──────────────────────────────────────────────────────────

_mismatch_log_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "logs",
    "replay_parity_mismatches.jsonl",
)


def _log_mismatch(record: Dict) -> None:
    """Append a mismatch record to logs/replay_parity_mismatches.jsonl."""
    try:
        os.makedirs(os.path.dirname(_mismatch_log_path), exist_ok=True)
        with open(_mismatch_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except Exception as e:
        logger.warning("Could not write mismatch log: %s", e)


# ── Core replay loop ──────────────────────────────────────────────────────────

def run_replay(
    symbol: str = "BTCUSDT",
    days: int = 5,
    step_hours: int = DEFAULT_STEP_HOURS,
    verbose: bool = False,
) -> Dict:
    """
    Walk forward through the last `days` of candle data, comparing decide() vs
    run_gate_pipeline() at every step_hours interval.

    Returns a result dict with total, matches, mismatches, match_rate, pass bool.
    """
    conn = get_connection()
    create_schema(conn)

    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=days + 2)  # +2 for warmup

    replay_start = end_time - timedelta(days=days)
    step_delta = timedelta(hours=step_hours)

    # Create a replay run in DB so run_gate_pipeline can log signals
    run_id = create_run(
        conn=conn,
        name=f"replay_parity_{symbol}_{days}d",
        start_date=replay_start,
        end_date=end_time,
        step_interval=f"{step_hours}h",
        starting_balance=STARTING_BALANCE,
        config={
            "symbol": symbol,
            "days": days,
            "step_hours": step_hours,
            "mode": "replay_parity",
            "entry_threshold": ENTRY_THRESHOLD,
            "min_rr": MIN_RR,
        },
    )
    logger.info("Created replay run #%d (%dd, %dh steps, symbol=%s)", run_id, days, step_hours, symbol)

    # Pre-load candle history for all TFs
    tfs = list(dict.fromkeys(MTF_TIMEFRAMES + ["1d", "4h", "1h", "30m"]))
    _load_full_history(symbol, tfs, start_time)

    state = BacktestState(
        current_time=replay_start,
        equity=STARTING_BALANCE,
        peak_equity=STARTING_BALANCE,
    )

    results: Dict = {
        "run_id": run_id,
        "symbol": symbol,
        "days": days,
        "total": 0,
        "matches": 0,
        "mismatches": 0,
        "match_rate": 0.0,
        "errors": 0,
        "mismatch_details": [],
        "pass": False,
    }

    current_time = replay_start
    step_num = 0

    try:
        while current_time <= end_time:
            state.current_time = current_time
            state.current_step = step_num

            # Build candles_by_tf snapshot — no future leakage
            candles_by_tf: Dict[str, pd.DataFrame] = {}
            for tf in tfs:
                df = _get_candles_at(symbol, tf, current_time)
                if df is not None:
                    candles_by_tf[tf] = df

            if len(candles_by_tf) < 2:
                # Not enough history yet (warmup period)
                current_time += step_delta
                step_num += 1
                continue

            price = _current_price(candles_by_tf)
            if price <= 0:
                current_time += step_delta
                step_num += 1
                continue

            results["total"] += 1

            # ── Run unified engine (v2) ────────────────────────────────
            v2_decision = "ERROR"
            v2_failure_code = None
            v2_model = None
            v2_tf = None
            try:
                v2_result = decide(
                    candles_by_tf,
                    {
                        "current_price": price,
                        "current_time": current_time,
                        "entry_threshold": ENTRY_THRESHOLD,
                        "min_rr": MIN_RR,
                        # No DD state or BOS dedup — stateless comparison
                    },
                )
                v2_decision = v2_result.get("decision", "PASS")
                v2_failure_code = v2_result.get("failure_code")
                v2_model = v2_result.get("model")
                v2_tf = v2_result.get("timeframe")
            except Exception as e:
                logger.warning("[step %d] decide() error: %s", step_num, e)
                results["errors"] += 1
                current_time += step_delta
                step_num += 1
                continue

            # ── Run backtest gate pipeline ─────────────────────────────
            bt_decision = "PASS"
            bt_model = None
            bt_tf = None
            try:
                bt_result = run_gate_pipeline(
                    state=state,
                    candles_by_tf=candles_by_tf,
                    current_price=price,
                    current_time=current_time,
                    run_id=run_id,
                    conn=conn,
                    replay_mode=True,
                    entry_threshold=ENTRY_THRESHOLD,
                    min_rr=MIN_RR,
                )
                if bt_result is not None:
                    bt_decision = "TAKE"
                    bt_model = bt_result.get("model")
                    bt_tf = bt_result.get("timeframe")
            except Exception as e:
                logger.warning("[step %d] run_gate_pipeline() error: %s", step_num, e)
                results["errors"] += 1
                current_time += step_delta
                step_num += 1
                continue

            # ── Compare ───────────────────────────────────────────────
            match = v2_decision == bt_decision

            if match:
                results["matches"] += 1
            else:
                results["mismatches"] += 1
                mismatch_record = {
                    "step": step_num,
                    "timestamp": current_time.isoformat(),
                    "symbol": symbol,
                    "price": price,
                    "v2_decision": v2_decision,
                    "bt_decision": bt_decision,
                    "v2_failure_code": v2_failure_code,
                    "v2_model": v2_model,
                    "v2_tf": v2_tf,
                    "bt_model": bt_model,
                    "bt_tf": bt_tf,
                }
                results["mismatch_details"].append(mismatch_record)
                _log_mismatch(mismatch_record)

                if verbose:
                    logger.warning(
                        "MISMATCH @ %s | v2=%s bt=%s | v2_fc=%s v2_tf=%s",
                        current_time.isoformat(), v2_decision, bt_decision,
                        v2_failure_code, v2_tf,
                    )

            # Advance step (do NOT update state.open_trade — stateless replay)
            state.last_signal_time = current_time
            current_time += step_delta
            step_num += 1

    except Exception as e:
        logger.error("Replay loop error: %s", e, exc_info=True)
        fail_run(conn, run_id, str(e))
        conn.close()
        raise

    total = results["total"]
    if total > 0:
        results["match_rate"] = results["matches"] / total
        results["pass"] = results["match_rate"] >= PASS_THRESHOLD

    complete_run(
        conn=conn,
        run_id=run_id,
        final_balance=STARTING_BALANCE,  # No trades executed — equity unchanged
        total_trades=0,
        wins=0,
        losses=0,
        max_drawdown_pct=0.0,
    )
    conn.close()

    return results


# ── Reporting ─────────────────────────────────────────────────────────────────

def _print_report(r: Dict) -> None:
    total = r["total"]
    if total == 0:
        print("ERROR: No steps could be compared — check candle data availability.")
        return

    match_rate_pct = r["match_rate"] * 100
    status = "✅" if r["pass"] else "❌"

    print()
    print("=" * 60)
    print("REPLAY PARITY RESULT")
    print("=" * 60)
    print(f"  Run ID:       {r['run_id']}")
    print(f"  Symbol:       {r['symbol']}")
    print(f"  Days:         {r['days']}")
    print(f"  Total steps:  {total}")
    print(f"  Matches:      {r['matches']}")
    print(f"  Mismatches:   {r['mismatches']}")
    print(f"  Errors:       {r['errors']}")
    print(f"  Match rate:   {match_rate_pct:.1f}% {status}")
    print()

    if r["mismatch_details"]:
        print("  MISMATCHES:")
        for d in r["mismatch_details"][:20]:  # cap display at 20
            print(
                f"    [{d['timestamp']}] v2={d['v2_decision']} bt={d['bt_decision']} "
                f"fc={d['v2_failure_code']} tf={d['v2_tf']}"
            )
        if len(r["mismatch_details"]) > 20:
            print(f"    ... and {len(r['mismatch_details']) - 20} more (see logs/replay_parity_mismatches.jsonl)")
    else:
        print("  No mismatches found.")

    pass_pct = PASS_THRESHOLD * 100
    print()
    if r["pass"]:
        print(f"  RESULT: ✅ PASS  (match_rate={match_rate_pct:.1f}% >= {pass_pct:.1f}%)")
    else:
        print(f"  RESULT: ❌ FAIL  (match_rate={match_rate_pct:.1f}% < {pass_pct:.1f}%)")
    print("=" * 60)
    print()


# ── Failure conditions check ──────────────────────────────────────────────────

def check_failure_conditions(r: Dict) -> List[str]:
    """
    Returns a list of triggered STOP conditions (any should halt deployment).

    Failure conditions (per spec):
      - Match rate < 99%
      - Any TAKE in v2 missing from backtest (PASS→TAKE direction in v2 — investigate)
    """
    failures = []
    if r["match_rate"] < 0.99:
        failures.append(f"match_rate={r['match_rate']*100:.1f}% < 99% STOP threshold")
    v2_looser = [d for d in r["mismatch_details"] if d["v2_decision"] == "TAKE" and d["bt_decision"] == "PASS"]
    if v2_looser:
        failures.append(
            f"{len(v2_looser)} case(s) where v2=TAKE but backtest=PASS "
            "(v2 is looser — investigate before enabling USE_UNIFIED_ENGINE)"
        )
    return failures


# ── Entrypoint ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Replay historical candles through both engines to verify decision parity"
    )
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Symbol (default: BTCUSDT)")
    parser.add_argument("--days", type=int, default=5, help="Days of history to replay (default: 5)")
    parser.add_argument(
        "--step-hours", type=int, default=DEFAULT_STEP_HOURS,
        help=f"Step interval in hours (default: {DEFAULT_STEP_HOURS})",
    )
    parser.add_argument("--verbose", action="store_true", help="Print every mismatch as it occurs")
    parser.add_argument("--json", dest="as_json", action="store_true", help="Output result as JSON")
    args = parser.parse_args()

    result = run_replay(
        symbol=args.symbol,
        days=args.days,
        step_hours=args.step_hours,
        verbose=args.verbose,
    )

    if args.as_json:
        print(json.dumps(result, default=str, indent=2))
    else:
        _print_report(result)

    failures = check_failure_conditions(result)
    if failures:
        print("STOP CONDITIONS TRIGGERED:")
        for f in failures:
            print(f"  ⛔ {f}")
        sys.exit(1)


if __name__ == "__main__":
    main()
