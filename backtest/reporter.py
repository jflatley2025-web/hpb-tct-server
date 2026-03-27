"""
backtest/reporter.py — Backtest Reporting & Analytics
======================================================
Generates comprehensive metrics from backtest results stored in PostgreSQL.

Standard:  Win rate, P&L, max drawdown, profit factor, avg R:R, avg hold time
Advanced:  Expectancy, MFE/MAE, consecutive losses, risk of ruin, RIG efficiency,
           gate failure distribution, session breakdown, execution quality metrics
"""

import json
import logging
import math
from datetime import timedelta
from typing import Dict, List, Optional

import pandas as pd

from backtest.db import get_connection, get_cursor

logger = logging.getLogger("backtest.reporter")


def load_run(conn, run_id: int) -> dict:
    """Load a backtest run record."""
    from psycopg2.extras import RealDictCursor
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("SELECT * FROM backtest_runs WHERE id = %s", (run_id,))
        row = cur.fetchone()
    return dict(row) if row else {}


def load_trades(conn, run_id: int) -> pd.DataFrame:
    """Load all trades for a run into a DataFrame."""
    return pd.read_sql(
        "SELECT * FROM backtest_trades WHERE run_id = %s ORDER BY trade_num",
        conn, params=(run_id,),
    )


def load_signals(conn, run_id: int) -> pd.DataFrame:
    """Load all signals for a run into a DataFrame."""
    return pd.read_sql(
        "SELECT * FROM backtest_signals WHERE run_id = %s ORDER BY signal_time",
        conn, params=(run_id,),
    )


# --Standard Metrics --────────────────────────────────────────────────

def compute_standard_metrics(trades: pd.DataFrame, run: dict) -> dict:
    """Compute core trading metrics."""
    if trades.empty:
        return {"total_trades": 0, "error": "No trades"}

    total = len(trades)
    wins = trades["is_win"].sum()
    losses = total - wins
    win_rate = (wins / total) * 100

    winning_trades = trades[trades["is_win"] == True]
    losing_trades = trades[trades["is_win"] == False]

    avg_win = winning_trades["pnl_dollars"].mean() if len(winning_trades) > 0 else 0
    avg_loss = abs(losing_trades["pnl_dollars"].mean()) if len(losing_trades) > 0 else 0

    gross_profit = winning_trades["pnl_dollars"].sum() if len(winning_trades) > 0 else 0
    gross_loss = abs(losing_trades["pnl_dollars"].sum()) if len(losing_trades) > 0 else 0

    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")

    total_pnl = trades["pnl_dollars"].sum()
    total_pnl_pct = ((run.get("final_balance", 0) - run.get("starting_balance", 0))
                     / run.get("starting_balance", 1)) * 100

    avg_rr = trades["rr"].mean() if "rr" in trades.columns else 0

    # Hold time
    if "opened_at" in trades.columns and "closed_at" in trades.columns:
        trades_with_times = trades.dropna(subset=["opened_at", "closed_at"])
        if not trades_with_times.empty:
            hold_times = (pd.to_datetime(trades_with_times["closed_at"])
                          - pd.to_datetime(trades_with_times["opened_at"]))
            avg_hold = hold_times.mean()
        else:
            avg_hold = timedelta(0)
    else:
        avg_hold = timedelta(0)

    return {
        "total_trades": total,
        "wins": int(wins),
        "losses": int(losses),
        "win_rate": round(win_rate, 2),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "profit_factor": round(profit_factor, 3),
        "total_pnl": round(total_pnl, 2),
        "total_pnl_pct": round(total_pnl_pct, 2),
        "max_drawdown_pct": round(run.get("max_drawdown_pct", 0) or 0, 2),
        "avg_rr": round(avg_rr, 2),
        "avg_hold_time": str(avg_hold),
        "starting_balance": run.get("starting_balance", 0),
        "final_balance": run.get("final_balance", 0),
    }


# --Advanced Metrics --────────────────────────────────────────────────

def compute_advanced_metrics(trades: pd.DataFrame, signals: pd.DataFrame) -> dict:
    """Compute advanced HPB-specific analytics."""
    if trades.empty:
        return {}

    metrics = {}

    # Expectancy
    total = len(trades)
    wins = trades["is_win"].sum()
    losses = total - wins
    winning_trades = trades[trades["is_win"] == True]
    losing_trades = trades[trades["is_win"] == False]
    avg_win = winning_trades["pnl_dollars"].mean() if len(winning_trades) > 0 else 0
    avg_loss = abs(losing_trades["pnl_dollars"].mean()) if len(losing_trades) > 0 else 0
    win_rate = wins / total if total > 0 else 0
    loss_rate = losses / total if total > 0 else 0
    metrics["expectancy"] = round((win_rate * avg_win) - (loss_rate * avg_loss), 2)

    # MFE/MAE distributions
    if "mfe" in trades.columns and trades["mfe"].notna().any():
        metrics["mfe_mean"] = round(trades["mfe"].mean(), 4)
        metrics["mfe_median"] = round(trades["mfe"].median(), 4)
        metrics["mae_mean"] = round(trades["mae"].mean(), 4)
        metrics["mae_median"] = round(trades["mae"].median(), 4)

    # Max consecutive losses
    metrics["max_consecutive_losses"] = _max_consecutive(trades, is_win=False)
    metrics["max_consecutive_wins"] = _max_consecutive(trades, is_win=True)

    # Risk of ruin (simplified approximation)
    if win_rate > 0 and loss_rate > 0 and avg_win > 0:
        # Kelly-based approximation
        edge = (win_rate * avg_win - loss_rate * avg_loss) / avg_win
        if edge > 0:
            metrics["risk_of_ruin_approx"] = round(
                ((loss_rate / win_rate) ** 10) * 100, 4  # 10-unit account
            )
        else:
            metrics["risk_of_ruin_approx"] = 100.0
    else:
        metrics["risk_of_ruin_approx"] = None

    # Return / drawdown ratio
    total_pnl_pct = trades["pnl_pct"].sum()
    max_dd = trades["balance_after"].expanding().apply(
        lambda x: ((x.cummax() - x.iloc[-1]) / x.cummax()).max() * 100
        if len(x) > 0 else 0
    ).max() if "balance_after" in trades.columns else 0
    if max_dd and max_dd > 0:
        metrics["return_drawdown_ratio"] = round(total_pnl_pct / max_dd, 2)
    else:
        metrics["return_drawdown_ratio"] = None

    # --Signal analytics --────────────────────────────────────────
    if not signals.empty:
        total_signals = len(signals)
        takes = (signals["final_decision"] == "TAKE").sum()
        skips = (signals["final_decision"] == "SKIP").sum()
        metrics["total_signals"] = total_signals
        metrics["signals_taken"] = int(takes)
        metrics["signals_skipped"] = int(skips)
        metrics["take_rate"] = round((takes / total_signals) * 100, 2) if total_signals > 0 else 0

        # RIG efficiency
        rig_blocks = signals[signals["rig_status"] == "BLOCK"]
        metrics["rig_total_blocks"] = len(rig_blocks)
        metrics["rig_block_rate"] = round(
            (len(rig_blocks) / total_signals) * 100, 2
        ) if total_signals > 0 else 0

        # Gate failure distribution
        failure_counts = {}
        if "failure_code" in signals.columns:
            fc = signals["failure_code"].dropna().value_counts()
            failure_counts = fc.to_dict()
        metrics["gate_failure_distribution"] = failure_counts

        # Session breakdown
        if "msce_session" in signals.columns:
            session_stats = {}
            for session_name in signals["msce_session"].dropna().unique():
                session_sigs = signals[signals["msce_session"] == session_name]
                session_takes = session_sigs[session_sigs["final_decision"] == "TAKE"]
                session_stats[session_name] = {
                    "signals": len(session_sigs),
                    "taken": len(session_takes),
                }
            metrics["session_breakdown"] = session_stats

    # --Execution quality --───────────────────────────────────────
    if not signals.empty and "latency_to_entry_seconds" in signals.columns:
        taken = signals[signals["final_decision"] == "TAKE"]
        if not taken.empty:
            metrics["avg_latency_to_entry"] = round(
                taken["latency_to_entry_seconds"].mean(), 2
            )

    # Adverse excursion analysis
    if "mae" in trades.columns and trades["mae"].notna().any():
        adverse_threshold = -0.01  # 1% adverse excursion
        adverse_count = (trades["mae"] < adverse_threshold).sum()
        metrics["pct_trades_adverse_gt_1pct"] = round(
            (adverse_count / len(trades)) * 100, 2
        )

    return metrics


def _max_consecutive(trades: pd.DataFrame, is_win: bool) -> int:
    """Count maximum consecutive wins or losses."""
    if trades.empty:
        return 0
    streak = 0
    max_streak = 0
    for _, row in trades.iterrows():
        if row["is_win"] == is_win:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
    return max_streak


# --Session P&L Breakdown --──────────────────────────────────────────

def compute_session_pnl(trades: pd.DataFrame, signals: pd.DataFrame) -> dict:
    """Break down P&L by trading session (Asia/London/NY)."""
    if trades.empty or signals.empty:
        return {}

    from backtest.session import get_session_name

    result = {}
    for _, trade in trades.iterrows():
        opened = pd.to_datetime(trade["opened_at"])
        if pd.isna(opened):
            continue
        session = get_session_name(opened.to_pydatetime())
        if session not in result:
            result[session] = {"trades": 0, "wins": 0, "pnl": 0.0}
        result[session]["trades"] += 1
        if trade["is_win"]:
            result[session]["wins"] += 1
        result[session]["pnl"] += trade["pnl_dollars"] or 0

    for session in result:
        t = result[session]["trades"]
        result[session]["win_rate"] = round(
            (result[session]["wins"] / t) * 100, 2
        ) if t > 0 else 0
        result[session]["pnl"] = round(result[session]["pnl"], 2)

    return result


# --Full Report --─────────────────────────────────────────────────────

def generate_report(run_id: int, conn=None, output_json: bool = False) -> dict:
    """Generate a full report for a backtest run."""
    own_conn = conn is None
    if own_conn:
        conn = get_connection()

    try:
        run = load_run(conn, run_id)
        if not run:
            return {"error": f"Run {run_id} not found"}

        trades = load_trades(conn, run_id)
        signals = load_signals(conn, run_id)

        report = {
            "run": {
                "id": run_id,
                "name": run.get("name"),
                "status": run.get("status"),
                "start_date": str(run.get("start_date")),
                "end_date": str(run.get("end_date")),
                "step_interval": run.get("step_interval"),
            },
            "standard": compute_standard_metrics(trades, run),
            "advanced": compute_advanced_metrics(trades, signals),
            "session_pnl": compute_session_pnl(trades, signals),
        }

        if output_json:
            return report

        _print_report(report)
        return report

    finally:
        if own_conn:
            conn.close()


def _print_report(report: dict):
    """Print report to console."""
    print(f"\n{'='*60}")
    print(f"  BACKTEST REPORT -- Run #{report['run']['id']}")
    print(f"  {report['run']['name']}")
    print(f"  {report['run']['start_date']} to {report['run']['end_date']}")
    print(f"  Step: {report['run']['step_interval']}")
    print(f"{'='*60}\n")

    std = report.get("standard", {})
    print("--STANDARD METRICS --")
    print(f"  Total Trades:    {std.get('total_trades', 0)}")
    print(f"  Wins / Losses:   {std.get('wins', 0)} / {std.get('losses', 0)}")
    print(f"  Win Rate:        {std.get('win_rate', 0):.1f}%")
    print(f"  Profit Factor:   {std.get('profit_factor', 0):.3f}")
    print(f"  Total P&L:       ${std.get('total_pnl', 0):.2f} ({std.get('total_pnl_pct', 0):+.2f}%)")
    print(f"  Max Drawdown:    {std.get('max_drawdown_pct', 0):.2f}%")
    print(f"  Avg R:R:         {std.get('avg_rr', 0):.2f}")
    print(f"  Avg Hold Time:   {std.get('avg_hold_time', '-')}")
    print(f"  Balance:         ${std.get('starting_balance', 0):.2f} -> ${std.get('final_balance', 0):.2f}")

    adv = report.get("advanced", {})
    if adv:
        print(f"\n--ADVANCED METRICS --")
        print(f"  Expectancy:          ${adv.get('expectancy', 0):.2f}")
        if adv.get("mfe_mean") is not None:
            print(f"  MFE (mean/median):   {adv['mfe_mean']:.4f} / {adv['mfe_median']:.4f}")
            print(f"  MAE (mean/median):   {adv['mae_mean']:.4f} / {adv['mae_median']:.4f}")
        print(f"  Max Consec Losses:   {adv.get('max_consecutive_losses', 0)}")
        print(f"  Max Consec Wins:     {adv.get('max_consecutive_wins', 0)}")
        if adv.get("risk_of_ruin_approx") is not None:
            print(f"  Risk of Ruin:        {adv['risk_of_ruin_approx']:.2f}%")
        if adv.get("return_drawdown_ratio") is not None:
            print(f"  Return/DD Ratio:     {adv['return_drawdown_ratio']:.2f}")

        if adv.get("total_signals"):
            print(f"\n--SIGNAL ANALYTICS --")
            print(f"  Total Signals:   {adv['total_signals']}")
            print(f"  Taken / Skipped: {adv['signals_taken']} / {adv['signals_skipped']}")
            print(f"  Take Rate:       {adv['take_rate']:.1f}%")
            print(f"  RIG Blocks:      {adv['rig_total_blocks']} ({adv['rig_block_rate']:.1f}%)")

        if adv.get("gate_failure_distribution"):
            print(f"\n--GATE FAILURES --")
            for code, count in sorted(adv["gate_failure_distribution"].items(),
                                       key=lambda x: -x[1]):
                print(f"  {code}: {count}")

    session_pnl = report.get("session_pnl", {})
    if session_pnl:
        print(f"\n--SESSION BREAKDOWN --")
        for session, stats in session_pnl.items():
            print(f"  {session:12s}: {stats['trades']} trades, "
                  f"WR={stats['win_rate']:.0f}%, P&L=${stats['pnl']:.2f}")

    if adv.get("pct_trades_adverse_gt_1pct") is not None:
        print(f"\n--EXECUTION QUALITY --")
        if adv.get("avg_latency_to_entry") is not None:
            print(f"  Avg Latency:     {adv['avg_latency_to_entry']:.2f}s")
        print(f"  Adverse >1%:     {adv['pct_trades_adverse_gt_1pct']:.1f}% of trades")

    print()


# --CLI Entrypoint --──────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Backtest Report Generator")
    parser.add_argument("run_id", type=int, help="Backtest run ID")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    report = generate_report(args.run_id, output_json=args.json)
    if args.json:
        print(json.dumps(report, indent=2, default=str))
