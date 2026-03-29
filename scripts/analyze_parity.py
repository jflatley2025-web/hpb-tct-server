"""
scripts/analyze_parity.py — Live vs Backtest Decision Parity Analyzer
=======================================================================

Reads logs/decision_parity.jsonl (written by schematics_5b_trader.py on every
decision cycle) and reports the match rate between the legacy gate pipeline and
decision_engine_v2.decide().

Usage:
    python -m scripts.analyze_parity               # last 200 records
    python -m scripts.analyze_parity --limit 500
    python -m scripts.analyze_parity --json        # machine-readable output
    python -m scripts.analyze_parity --verbose     # print every mismatch

Pass criteria: match_rate >= 99%

Output example:
    MATCH RATE: 98.7%  ✗ FAIL

    MISMATCH BREAKDOWN:
    - 15m: 6 mismatches
    - Model_3: 3 mismatches
    - Asia session: 5 mismatches

    TOP REASONS:
    - FAIL_RR_FILTER: 4
    - FAIL_15M_ASIA_FILTER: 3
"""

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("analyze_parity")

# Default JSONL path — relative to project root so the script works from any CWD.
_DEFAULT_LOG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "logs",
    "decision_parity.jsonl",
)

PASS_THRESHOLD = 0.99  # 99% match rate required


# ── Loading ───────────────────────────────────────────────────────────────────

def load_records(log_path: str, limit: int) -> List[Dict]:
    """Load the last `limit` records from the JSONL parity log."""
    if not os.path.exists(log_path):
        logger.error("Parity log not found: %s", log_path)
        return []

    records: List[Dict] = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning("Bad JSON line skipped: %s", e)

    # Return last N records (most recent)
    return records[-limit:] if len(records) > limit else records


# ── Analysis ──────────────────────────────────────────────────────────────────

def analyze_parity(limit: int = 200, log_path: Optional[str] = None) -> Dict:
    """
    Load last `limit` parity records and compute match statistics.

    Returns a result dict with:
        match_rate          — float 0.0–1.0
        total               — records analyzed
        matches             — exact match count
        mismatches          — mismatch count
        by_timeframe        — mismatch counts keyed by TF
        by_model            — mismatch counts keyed by model
        by_session          — mismatch counts keyed by session
        by_failure_code     — mismatch counts keyed by v2_failure_code
        mismatch_details    — list of mismatch records (for verbose output)
        pass                — bool: match_rate >= PASS_THRESHOLD
    """
    path = log_path or _DEFAULT_LOG_PATH
    records = load_records(path, limit)

    result: Dict = {
        "total": 0,
        "matches": 0,
        "mismatches": 0,
        "match_rate": 0.0,
        "by_timeframe": defaultdict(int),
        "by_model": defaultdict(int),
        "by_session": defaultdict(int),
        "by_failure_code": defaultdict(int),
        "mismatch_details": [],
        "pass": False,
    }

    if not records:
        logger.warning("No parity records found in: %s", path)
        return result

    for rec in records:
        result["total"] += 1
        match = rec.get("match", True)

        if match:
            result["matches"] += 1
        else:
            result["mismatches"] += 1

            tf = rec.get("timeframe") or "unknown"
            model = rec.get("model") or "unknown"
            session = rec.get("session") or "unknown"
            fc = rec.get("v2_failure_code") or rec.get("gate_debug", {}).get("failure_code") or "none"

            result["by_timeframe"][tf] += 1
            result["by_model"][model] += 1
            result["by_session"][session] += 1
            result["by_failure_code"][fc] += 1

            result["mismatch_details"].append({
                "timestamp": rec.get("timestamp"),
                "timeframe": tf,
                "model": model,
                "session": session,
                "legacy_decision": rec.get("legacy_decision"),
                "v2_decision": rec.get("v2_decision"),
                "failure_code": fc,
                "reason_v2": rec.get("reason_v2"),
                "rr": rec.get("rr"),
                "displacement": rec.get("displacement"),
                "htf_bias": rec.get("htf_bias"),
                "gate_debug": rec.get("gate_debug"),
            })

    total = result["total"]
    if total > 0:
        result["match_rate"] = result["matches"] / total
        result["pass"] = result["match_rate"] >= PASS_THRESHOLD

    # Convert defaultdicts to plain dicts for serialization
    result["by_timeframe"] = dict(result["by_timeframe"])
    result["by_model"] = dict(result["by_model"])
    result["by_session"] = dict(result["by_session"])
    result["by_failure_code"] = dict(result["by_failure_code"])

    return result


# ── Reporting ─────────────────────────────────────────────────────────────────

def _print_report(r: Dict, verbose: bool = False) -> None:
    total = r["total"]
    if total == 0:
        print("ERROR: No parity records found. Ensure schematics_5b_trader.py is running.")
        return

    match_rate_pct = r["match_rate"] * 100
    status = "✅" if r["pass"] else "❌"

    print()
    print("=" * 60)
    print("LIVE vs BACKTEST PARITY REPORT")
    print("=" * 60)
    print(f"  Records analyzed: {total}")
    print(f"  Matches:          {r['matches']}")
    print(f"  Mismatches:       {r['mismatches']}")
    print(f"  MATCH RATE:       {match_rate_pct:.1f}% {status}")
    print()

    if r["mismatches"] > 0:
        print("  MISMATCH BREAKDOWN:")
        by_tf = sorted(r["by_timeframe"].items(), key=lambda x: -x[1])
        for tf, cnt in by_tf:
            print(f"    - {tf}: {cnt} mismatches")

        print()
        by_model = sorted(r["by_model"].items(), key=lambda x: -x[1])
        for model, cnt in by_model:
            print(f"    - {model}: {cnt} mismatches")

        print()
        by_session = sorted(r["by_session"].items(), key=lambda x: -x[1])
        for sess, cnt in by_session:
            print(f"    - {sess} session: {cnt} mismatches")

        print()
        print("  TOP FAILURE CODES:")
        by_fc = sorted(r["by_failure_code"].items(), key=lambda x: -x[1])
        for fc, cnt in by_fc[:10]:
            print(f"    - {fc}: {cnt}")

        if verbose and r["mismatch_details"]:
            print()
            print("  MISMATCH DETAILS:")
            for d in r["mismatch_details"]:
                print(
                    f"    [{d['timestamp']}] {d['timeframe']} {d['model']} "
                    f"legacy={d['legacy_decision']} v2={d['v2_decision']} "
                    f"fc={d['failure_code']} rr={d['rr']} disp={d['displacement']}"
                )
                if d.get("gate_debug"):
                    gd = d["gate_debug"]
                    print(
                        f"      gate_debug: rig={gd.get('rig_status')} "
                        f"1a={gd.get('gate_1a_pass')} rcm={gd.get('rcm_score')} "
                        f"session={gd.get('session')}"
                    )
    else:
        print("  No mismatches found in this sample.")

    pass_threshold_pct = PASS_THRESHOLD * 100
    print()
    if r["pass"]:
        print(f"  RESULT: ✅ PASS  (match_rate={match_rate_pct:.1f}% >= {pass_threshold_pct:.0f}%)")
    else:
        print(f"  RESULT: ❌ FAIL  (match_rate={match_rate_pct:.1f}% < {pass_threshold_pct:.0f}%)")
        if total < 20:
            print(f"          WARNING: Only {total} records — more data needed for reliable analysis.")
    print("=" * 60)
    print()


# ── Entrypoint ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Analyze live vs backtest decision parity from logs/decision_parity.jsonl"
    )
    parser.add_argument(
        "--limit", type=int, default=200,
        help="Number of most recent records to analyze (default: 200)",
    )
    parser.add_argument(
        "--log-path", type=str, default=None,
        help=f"Path to decision_parity.jsonl (default: {_DEFAULT_LOG_PATH})",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print details of every mismatch",
    )
    parser.add_argument(
        "--json", dest="as_json", action="store_true",
        help="Output result as JSON instead of human-readable report",
    )
    args = parser.parse_args()

    result = analyze_parity(limit=args.limit, log_path=args.log_path)

    if args.as_json:
        print(json.dumps(result, default=str, indent=2))
    else:
        _print_report(result, verbose=args.verbose)

    # Exit 1 on failure so CI/monitoring can detect regressions
    if not result["pass"] and result["total"] >= 20:
        sys.exit(1)


if __name__ == "__main__":
    main()
