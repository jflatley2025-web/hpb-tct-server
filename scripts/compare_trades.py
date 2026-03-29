"""
scripts/compare_trades.py — Trade-Level Parity Check (Live vs Backtest)
========================================================================

Compares live trades from schematics_5b_trade_log.json against backtest
trades from the DB for a given run_id.

Matching is fuzzy on timestamp (±30 min tolerance) since live execution
has latency and backtest uses candle open_time. Exact match requires
agreement on direction, model, and timeframe.

Usage:
    python -m scripts.compare_trades --run-id 29
    python -m scripts.compare_trades --run-id 29 --window-minutes 60
    python -m scripts.compare_trades --run-id 29 --json

Comparison fields (per spec):
    - timestamp   (fuzzy ±window_minutes)
    - direction   (exact)
    - model       (exact)
    - timeframe   (exact)

Output:
    Trade Parity:
    Matched Trades: 18/19
    Mismatch:
    - Missing trade at 2025-09-11 10:00 (live=LONG 1h Model_1, no backtest match)
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.db import get_connection, normalize_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("compare_trades")

_DEFAULT_TRADE_LOG = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "schematics_5b_trade_log.json",
)

# Default fuzzy timestamp tolerance
DEFAULT_WINDOW_MINUTES = 30

# Fields used for matching after timestamp alignment
MATCH_FIELDS = ["direction", "model", "timeframe"]


# ── Live trade loading ────────────────────────────────────────────────────────

def _parse_ts(ts_str: Optional[str]) -> Optional[datetime]:
    """Parse an ISO timestamp string to UTC-aware datetime. Returns None on failure."""
    if not ts_str:
        return None
    try:
        dt = datetime.fromisoformat(ts_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        return None


def load_live_trades(log_path: str) -> List[Dict]:
    """Load trade_history from schematics_5b_trade_log.json."""
    if not os.path.exists(log_path):
        logger.error("Trade log not found: %s", log_path)
        return []
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        trades = data.get("trade_history", [])
        logger.info("Loaded %d live trades from %s", len(trades), log_path)
        return trades
    except Exception as e:
        logger.error("Failed to load live trade log: %s", e)
        return []


def _normalise_live_trade(t: Dict) -> Dict:
    """Normalise a raw live trade dict to standard comparison fields."""
    return {
        "timestamp": _parse_ts(t.get("opened_at") or t.get("entry_time") or t.get("timestamp")),
        "direction": (t.get("direction") or "").upper(),
        "model": normalize_model(t.get("model") or t.get("schematic")),
        "timeframe": t.get("timeframe") or t.get("tf"),
        "entry_price": t.get("entry_price"),
        "_raw": t,
    }


# ── Backtest trade loading ────────────────────────────────────────────────────

def load_backtest_trades(conn, run_id: int) -> List[Dict]:
    """Load all trades for a given backtest run_id from backtest_trades."""
    cur = conn.cursor()
    cur.execute(
        """
        SELECT trade_num, timeframe, direction, model,
               entry_price, stop_price, target_price, rr,
               opened_at, closed_at, exit_reason, is_win, pnl_dollars
        FROM backtest_trades
        WHERE run_id = %s
        ORDER BY opened_at ASC
        """,
        (run_id,),
    )
    cols = [d[0] for d in cur.description]
    rows = [dict(zip(cols, row, strict=True)) for row in cur.fetchall()]
    cur.close()
    logger.info("Loaded %d backtest trades for run_id=%d", len(rows), run_id)
    return rows


def _normalise_bt_trade(t: Dict) -> Dict:
    """Normalise a backtest trade row to standard comparison fields."""
    ts = t.get("opened_at")
    if ts is not None and getattr(ts, "tzinfo", None) is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return {
        "timestamp": ts,
        "direction": (t.get("direction") or "").upper(),
        "model": normalize_model(t.get("model")),
        "timeframe": t.get("timeframe"),
        "entry_price": t.get("entry_price"),
        "_raw": t,
    }


# ── Matching logic ────────────────────────────────────────────────────────────

def _fields_match(live: Dict, bt: Dict) -> bool:
    """Return True if direction, model, and timeframe all agree."""
    for field in MATCH_FIELDS:
        lv = (live.get(field) or "").strip().lower()
        bv = (bt.get(field) or "").strip().lower()
        if lv != bv:
            return False
    return True


def _find_bt_match(
    live: Dict,
    bt_trades: List[Dict],
    used_indices: set,
    window: timedelta,
) -> Optional[int]:
    """
    Find the best matching backtest trade for a given live trade.

    Returns the index into bt_trades, or None if no match found.
    Priority: exact field match within window. Earliest-within-window wins
    on tie so trades are consumed in order.
    """
    if live["timestamp"] is None:
        return None

    candidates: List[Tuple[timedelta, int]] = []
    for i, bt in enumerate(bt_trades):
        if i in used_indices:
            continue
        if bt["timestamp"] is None:
            continue
        delta = abs(live["timestamp"] - bt["timestamp"])
        if delta <= window and _fields_match(live, bt):
            candidates.append((delta, i))

    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]


# ── Core comparison ───────────────────────────────────────────────────────────

def compare_trades(
    run_id: int,
    trade_log_path: str = _DEFAULT_TRADE_LOG,
    window_minutes: int = DEFAULT_WINDOW_MINUTES,
) -> Dict:
    """
    Compare live trades against backtest trades for run_id.

    Returns a result dict with:
        live_count, bt_count, matched, live_only, bt_only,
        match_rate, mismatch_details, pass
    """
    conn = get_connection()

    raw_live = load_live_trades(trade_log_path)
    raw_bt = load_backtest_trades(conn, run_id)
    conn.close()

    if not raw_live and not raw_bt:
        return {
            "error": "No trades found in either live log or backtest DB.",
            "run_id": run_id,
        }

    live_trades = [_normalise_live_trade(t) for t in raw_live]
    bt_trades = [_normalise_bt_trade(t) for t in raw_bt]

    window = timedelta(minutes=window_minutes)
    used_bt_indices: set = set()

    matched: List[Dict] = []
    live_only: List[Dict] = []  # in live but not backtest
    bt_only: List[Dict] = []    # in backtest but not live

    for live in live_trades:
        match_idx = _find_bt_match(live, bt_trades, used_bt_indices, window)
        if match_idx is not None:
            used_bt_indices.add(match_idx)
            matched.append({
                "live_ts": live["timestamp"].isoformat() if live["timestamp"] else None,
                "bt_ts": bt_trades[match_idx]["timestamp"].isoformat() if bt_trades[match_idx]["timestamp"] else None,
                "direction": live["direction"],
                "model": live["model"],
                "timeframe": live["timeframe"],
                "live_entry": live["entry_price"],
                "bt_entry": bt_trades[match_idx]["entry_price"],
            })
        else:
            live_only.append({
                "timestamp": live["timestamp"].isoformat() if live["timestamp"] else None,
                "direction": live["direction"],
                "model": live["model"],
                "timeframe": live["timeframe"],
                "entry_price": live["entry_price"],
            })

    for i, bt in enumerate(bt_trades):
        if i not in used_bt_indices:
            bt_only.append({
                "timestamp": bt["timestamp"].isoformat() if bt["timestamp"] else None,
                "direction": bt["direction"],
                "model": bt["model"],
                "timeframe": bt["timeframe"],
                "entry_price": bt["entry_price"],
            })

    total_live = len(live_trades)
    total_bt = len(bt_trades)
    total_union = max(total_live, total_bt)
    match_rate = len(matched) / total_union if total_union > 0 else 1.0

    return {
        "run_id": run_id,
        "window_minutes": window_minutes,
        "live_count": total_live,
        "bt_count": total_bt,
        "matched": len(matched),
        "live_only": live_only,    # trades in live but not backtest
        "bt_only": bt_only,        # trades in backtest but not live
        "match_rate": round(match_rate, 4),
        "match_details": matched,
        "pass": len(live_only) == 0 and len(bt_only) == 0,
    }


# ── Reporting ─────────────────────────────────────────────────────────────────

def _print_report(r: Dict) -> None:
    if "error" in r:
        print(f"ERROR: {r['error']}")
        return

    total_live = r["live_count"]
    total_bt = r["bt_count"]
    matched = r["matched"]
    match_rate_pct = r["match_rate"] * 100
    status = "✅" if r["pass"] else "❌"

    print()
    print("=" * 60)
    print("TRADE PARITY REPORT")
    print("=" * 60)
    print(f"  Run ID:           {r['run_id']}")
    print(f"  Window:           ±{r['window_minutes']}min timestamp tolerance")
    print(f"  Live trades:      {total_live}")
    print(f"  Backtest trades:  {total_bt}")
    print(f"  Matched trades:   {matched}/{max(total_live, total_bt)}  ({match_rate_pct:.1f}%)")
    print()

    if r["live_only"]:
        print("  IN LIVE BUT NOT BACKTEST (investigate):")
        for t in r["live_only"]:
            print(
                f"    ⚠ {t['timestamp']}  {t['direction']} {t['timeframe']} "
                f"{t['model']}  entry={t['entry_price']}"
            )

    if r["bt_only"]:
        print()
        print("  IN BACKTEST BUT NOT LIVE:")
        for t in r["bt_only"]:
            print(
                f"    ⚠ {t['timestamp']}  {t['direction']} {t['timeframe']} "
                f"{t['model']}  entry={t['entry_price']}"
            )

    if not r["live_only"] and not r["bt_only"]:
        print("  All trades matched.")

    print()
    if r["pass"]:
        print("  RESULT: ✅ PASS  — Trade-level alignment confirmed")
    else:
        print(f"  RESULT: ❌ FAIL  {status} — Unmatched trades require investigation")
    print("=" * 60)
    print()


# ── Entrypoint ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compare live trades vs backtest trades for a given run_id"
    )
    parser.add_argument(
        "--run-id", type=int, required=True,
        help="Backtest run ID to compare against",
    )
    parser.add_argument(
        "--trade-log", type=str, default=_DEFAULT_TRADE_LOG,
        help=f"Path to live trade log JSON (default: {_DEFAULT_TRADE_LOG})",
    )
    parser.add_argument(
        "--window-minutes", type=int, default=DEFAULT_WINDOW_MINUTES,
        help=f"Timestamp match tolerance in minutes (default: {DEFAULT_WINDOW_MINUTES})",
    )
    parser.add_argument(
        "--json", dest="as_json", action="store_true",
        help="Output as JSON",
    )
    args = parser.parse_args()

    result = compare_trades(
        run_id=args.run_id,
        trade_log_path=args.trade_log,
        window_minutes=args.window_minutes,
    )

    if args.as_json:
        print(json.dumps(result, default=str, indent=2))
    else:
        _print_report(result)

    if not result.get("pass", False):
        sys.exit(1)


if __name__ == "__main__":
    main()
