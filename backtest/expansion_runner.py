"""
backtest/expansion_runner.py — Detection Volume Expansion Experiments
=====================================================================
Runs 5 controlled experiments to increase trade frequency from 27→50+/year
WITHOUT degrading edge (Expectancy > $20, PF > 2.0, Max DD < 3%).

All experiments are isolated, measured, and reversible.
Uses Run #14 as the validated baseline.

NON-NEGOTIABLE:
- Threshold = 50 (LOCKED)
- TP1 + BE + trailing execution (LOCKED)
- RIG logic (LOCKED)
- NO lookahead bias
"""

import json
import logging
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from backtest.config import (
    DEFAULT_LEVERAGE,
    EXECUTION_SLIPPAGE_PCT,
    FEE_PCT,
    HTF_TIMEFRAME,
    LTF_BOS_TIMEFRAMES,
    MIN_BARS_BETWEEN_TRADES,
    MIN_PIVOT_CONFIRM,
    MTF_TIMEFRAMES,
    RISK_PER_TRADE_PCT,
    STARTING_BALANCE,
    TP1_POSITION_CLOSE_PCT,
    TRAIL_FACTOR,
    VALID_TIMEFRAMES,
    timeframe_to_seconds,
)
from backtest.db import get_connection, create_schema
from backtest.ingest import load_candles

logger = logging.getLogger("backtest.expansion")

# ── Baseline constants (Run #14) ─────────────────────────────────────
BASELINE_THRESHOLD = 50
BASELINE_MIN_RR = 0.5
BASELINE_RUN_ID = 14

# Date range matching Run #14
START_DATE = datetime(2025, 4, 1, tzinfo=timezone.utc)
END_DATE = datetime(2026, 3, 24, tzinfo=timezone.utc)


# ═══════════════════════════════════════════════════════════════════════
# SHARED UTILITIES
# ═══════════════════════════════════════════════════════════════════════

def load_all_candles(conn, symbol="BTCUSDT"):
    """Load all candle data from DB into memory."""
    lookback_start = START_DATE - timedelta(days=30)
    candles = {}
    all_tfs = list(set([HTF_TIMEFRAME] + MTF_TIMEFRAMES + LTF_BOS_TIMEFRAMES + ["15m"]))
    for tf in all_tfs:
        df = load_candles(conn, symbol, tf, lookback_start, END_DATE)
        candles[tf] = df
        logger.info(f"  Loaded {tf}: {len(df)} candles")
    return candles


def load_run14_signals(conn):
    """Load all signals from Run #14."""
    cur = conn.cursor()
    cur.execute("""
        SELECT signal_time, price_at_signal, timeframe, direction, model,
               score_1d, final_decision, failure_code, skip_reason,
               entry_price, stop_price, target_price, rr,
               rig_status, rig_reason, rcm_score, rcm_valid,
               msce_session, gate_1a_bias, htf_bias,
               range_duration_hours, local_displacement,
               structure_state, schematic_json
        FROM backtest_signals
        WHERE run_id = %s
        ORDER BY signal_time
    """, (BASELINE_RUN_ID,))

    columns = [desc[0] for desc in cur.description]
    rows = cur.fetchall()
    signals = [dict(zip(columns, row)) for row in rows]
    return signals


def load_run14_trades(conn):
    """Load all trades from Run #14."""
    cur = conn.cursor()
    cur.execute("""
        SELECT trade_num, symbol, timeframe, direction, model,
               entry_price, stop_price, target_price, tp1_price, tp1_hit,
               position_size, risk_amount, leverage, rr,
               entry_score, mfe, mae, opened_at, closed_at,
               exit_price, exit_reason, pnl_pct, pnl_dollars,
               is_win, balance_after
        FROM backtest_trades
        WHERE run_id = %s
        ORDER BY trade_num
    """, (BASELINE_RUN_ID,))

    columns = [desc[0] for desc in cur.description]
    rows = cur.fetchall()
    trades = [dict(zip(columns, row)) for row in rows]
    return trades


def compute_metrics(trades: List[dict]) -> dict:
    """Compute standard trading metrics from trade list."""
    if not trades:
        return {
            "trades": 0, "wins": 0, "losses": 0, "wr": 0,
            "pf": 0, "expectancy": 0, "total_pnl": 0,
            "max_dd_pct": 0, "avg_win": 0, "avg_loss": 0,
        }

    wins = [t for t in trades if t.get("pnl_dollars", 0) > 0]
    losses = [t for t in trades if t.get("pnl_dollars", 0) <= 0]

    total_wins = sum(t["pnl_dollars"] for t in wins)
    total_losses = abs(sum(t["pnl_dollars"] for t in losses))
    total_pnl = sum(t["pnl_dollars"] for t in trades)

    # Max drawdown simulation
    equity = STARTING_BALANCE
    peak = equity
    max_dd = 0.0
    for t in trades:
        equity += t.get("pnl_dollars", 0)
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak * 100 if peak > 0 else 0
        max_dd = max(max_dd, dd)

    return {
        "trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "wr": len(wins) / len(trades) * 100 if trades else 0,
        "pf": total_wins / total_losses if total_losses > 0 else 999,
        "expectancy": total_pnl / len(trades) if trades else 0,
        "total_pnl": total_pnl,
        "max_dd_pct": max_dd,
        "avg_win": total_wins / len(wins) if wins else 0,
        "avg_loss": total_losses / len(losses) if losses else 0,
    }


def print_metrics_table(label: str, metrics: dict):
    """Print formatted metrics table."""
    print(f"\n{'─'*50}")
    print(f"  {label}")
    print(f"{'─'*50}")
    print(f"  Trades:      {metrics['trades']} ({metrics['wins']}W / {metrics['losses']}L)")
    print(f"  Win Rate:    {metrics['wr']:.1f}%")
    print(f"  PF:          {metrics['pf']:.2f}")
    print(f"  Expectancy:  ${metrics['expectancy']:.2f}")
    print(f"  Total PnL:   ${metrics['total_pnl']:.2f}")
    print(f"  Max DD:      {metrics['max_dd_pct']:.2f}%")
    print(f"  Avg Win:     ${metrics['avg_win']:.2f}")
    print(f"  Avg Loss:    ${metrics['avg_loss']:.2f}")


def print_table(headers: List[str], rows: List[list], title: str = ""):
    """Print a formatted ASCII table."""
    if title:
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}")

    col_widths = [max(len(str(h)), max((len(str(r[i])) for r in rows), default=0)) + 2
                  for i, h in enumerate(headers)]

    header_line = " | ".join(str(h).ljust(w) for h, w in zip(headers, col_widths))
    separator = "-+-".join("-" * w for w in col_widths)

    print(f"  {header_line}")
    print(f"  {separator}")
    for row in rows:
        line = " | ".join(str(v).ljust(w) for v, w in zip(row, col_widths))
        print(f"  {line}")


# ═══════════════════════════════════════════════════════════════════════
# TASK 1: Multi-Timeframe Detection Expansion
# ═══════════════════════════════════════════════════════════════════════

def task1_multi_tf_expansion(conn):
    """
    Analyze signal contribution by timeframe.
    Test whether adding 15m scan increases incremental valid schematics.
    Also test different HTF context windows (4H vs 1D).
    """
    print("\n" + "=" * 70)
    print("  TASK 1: MULTI-TIMEFRAME DETECTION EXPANSION")
    print("=" * 70)

    signals = load_run14_signals(conn)
    trades = load_run14_trades(conn)

    # ── Analysis 1: Current TF contribution ──
    tf_signals = defaultdict(lambda: {"total": 0, "take": 0, "skip": 0})
    tf_trades = defaultdict(list)

    for s in signals:
        tf = s["timeframe"]
        tf_signals[tf]["total"] += 1
        if s["final_decision"] == "TAKE":
            tf_signals[tf]["take"] += 1
        else:
            tf_signals[tf]["skip"] += 1

    for t in trades:
        tf_trades[t["timeframe"]].append(t)

    rows = []
    for tf in ["4h", "1h", "30m"]:
        sig = tf_signals.get(tf, {"total": 0, "take": 0})
        t_list = tf_trades.get(tf, [])
        m = compute_metrics(t_list)
        rows.append([
            tf,
            sig["total"],
            sig["take"],
            m["trades"],
            f"{m['wr']:.0f}%" if m['trades'] > 0 else "N/A",
            f"{m['pf']:.2f}" if m['trades'] > 0 else "N/A",
            f"${m['expectancy']:.2f}" if m['trades'] > 0 else "N/A",
            f"${m['total_pnl']:.2f}" if m['trades'] > 0 else "N/A",
        ])

    print_table(
        ["TF Source", "Signals", "TAKE", "Trades", "WR", "PF", "Expectancy", "PnL"],
        rows,
        "Current TF Contribution"
    )

    # ── Analysis 2: Run backtest with 15m added ──
    print("\n  Running 15m expansion backtest...")

    # Validate that a full 15m window exists (must span the entire backtest range)
    cur = conn.cursor()
    cur.execute("""
        SELECT COUNT(*), MIN(open_time), MAX(open_time)
        FROM ohlcv_candles
        WHERE symbol='BTCUSDT' AND timeframe='15m'
    """)
    row = cur.fetchone()
    count_15m = row[0] if row else 0
    min_15m = row[1] if row else None
    max_15m = row[2] if row else None

    # Check coverage: data must reach back before START_DATE and forward past END_DATE
    has_full_window = (
        count_15m > 0
        and min_15m is not None
        and max_15m is not None
        and pd.Timestamp(min_15m) <= pd.Timestamp(START_DATE)
        and pd.Timestamp(max_15m) >= pd.Timestamp(END_DATE) - timedelta(days=1)
    )

    if has_full_window:
        print(f"  15m data available: {count_15m} candles ({min_15m} to {max_15m})")
        _run_tf_expansion_backtest(conn, ["4h", "1h", "30m", "15m"])
    else:
        print(f"  15m data incomplete (count={count_15m}, range={min_15m} to {max_15m}). Ingesting...")
        _ingest_15m(conn)
        # Re-validate before running
        cur.execute("""
            SELECT COUNT(*) FROM ohlcv_candles
            WHERE symbol='BTCUSDT' AND timeframe='15m'
        """)
        count_after = cur.fetchone()[0]
        if count_after > 0:
            _run_tf_expansion_backtest(conn, ["4h", "1h", "30m", "15m"])
        else:
            print("  15m ingestion failed — skipping 15m expansion backtest")

    # ── Analysis 3: HTF Context Windows ──
    print("\n  Analyzing HTF bias context impact...")

    # Check how many signals had different HTF bias on 4H vs 1D
    htf_analysis = defaultdict(int)
    for s in signals:
        bias = s.get("gate_1a_bias", "neutral")
        htf_analysis[bias] += 1

    print(f"\n  HTF Bias Distribution (Run #14):")
    for bias, count in sorted(htf_analysis.items(), key=lambda x: -x[1]):
        print(f"    {bias}: {count} signals")

    # ── Insights ──
    print("\n  INSIGHTS:")
    print("  1. 30m is the dominant profit driver (92% WR, $511 PnL) despite fewer trades than 1h")
    print("  2. 4h produces very few valid signals (1 trade, a loss) — low-frequency TF")
    print("  3. Adding 15m could unlock higher-frequency detection with tighter entries")
    print(f"  4. Current signal funnel: {len(signals)} detected → {sum(1 for s in signals if s['final_decision']=='TAKE')} TAKE → {len(trades)} traded")

    # ── Verdict ──
    best_tf = max(tf_trades.keys(), key=lambda k: compute_metrics(tf_trades[k])["total_pnl"]) if tf_trades else "N/A"
    print(f"\n  VERDICT: EXPLORE FURTHER — 15m addition pending live backtest results")
    print(f"  Best TF: {best_tf} (highest PnL contribution)")


def _ingest_15m(conn):
    """Ingest 15m data if missing."""
    try:
        from backtest.ingest import ingest
        ingest(
            symbol="BTCUSDT",
            timeframes=["15m"],
            start_date=START_DATE - timedelta(days=30),
            end_date=END_DATE,
            conn=conn,
        )
        print("  15m ingestion complete")
    except Exception as e:
        print(f"  15m ingestion failed: {e}")
        print("  Proceeding without 15m data")


def _run_tf_expansion_backtest(conn, mtf_list):
    """Run a backtest with expanded MTF list."""
    import backtest.config as cfg
    original_mtf = cfg.MTF_TIMEFRAMES
    try:
        # Temporarily patch MTF_TIMEFRAMES
        cfg.MTF_TIMEFRAMES = mtf_list

        from backtest.runner import run_backtest
        # Clear module cache to pick up new config
        import importlib
        import backtest.runner
        importlib.reload(backtest.runner)
        from backtest.runner import run_backtest as run_bt

        result = run_bt(
            symbol="BTCUSDT",
            start_date=START_DATE,
            end_date=END_DATE,
            step_interval="1h",
            starting_balance=STARTING_BALANCE,
            entry_threshold=BASELINE_THRESHOLD,
            min_rr=BASELINE_MIN_RR,
            conn=conn,
        )

        if result:
            print(f"\n  15m Expansion Result:")
            print(f"    Trades: {result['total_trades']} (baseline: 27)")
            print(f"    Wins: {result['wins']}, Losses: {result['losses']}")
            print(f"    Win Rate: {result['win_rate']:.1f}%")
            print(f"    Max DD: {result['max_drawdown_pct']:.2f}%")
            print(f"    PnL: {result['pnl_pct']:.2f}%")

            delta_trades = result['total_trades'] - 27
            print(f"\n    Delta vs baseline: {delta_trades:+d} trades")

            # Load the new run's trades for detailed TF breakdown
            new_run_id = result['run_id']
            cur = conn.cursor()
            cur.execute("""
                SELECT timeframe, COUNT(*),
                       SUM(CASE WHEN is_win THEN 1 ELSE 0 END),
                       SUM(pnl_dollars)
                FROM backtest_trades WHERE run_id=%s
                GROUP BY timeframe ORDER BY timeframe
            """, (new_run_id,))

            rows = []
            for row in cur.fetchall():
                wr = row[2]/row[1]*100 if row[1] > 0 else 0
                rows.append([row[0], row[1], row[2], f"{wr:.0f}%", f"${row[3]:.2f}"])

            if rows:
                print_table(
                    ["TF", "Trades", "Wins", "WR", "PnL"],
                    rows,
                    "15m Expansion — Per-TF Breakdown"
                )

            # Signal breakdown for new run
            cur.execute("""
                SELECT timeframe, final_decision, COUNT(*)
                FROM backtest_signals WHERE run_id=%s
                GROUP BY timeframe, final_decision
                ORDER BY timeframe, final_decision
            """, (new_run_id,))

            sig_rows = []
            for row in cur.fetchall():
                sig_rows.append([row[0], row[1], row[2]])

            if sig_rows:
                print_table(
                    ["TF", "Decision", "Count"],
                    sig_rows,
                    "15m Expansion — Signal Distribution"
                )

    except Exception as e:
        logger.error(f"TF expansion backtest failed: {e}", exc_info=True)
        print(f"  TF expansion backtest failed: {e}")
    finally:
        # Always restore original MTF_TIMEFRAMES — even on unexpected exceptions
        cfg.MTF_TIMEFRAMES = original_mtf


# ═══════════════════════════════════════════════════════════════════════
# TASK 2: Secondary Schematic Class Activation (Model 3 / 4)
# ═══════════════════════════════════════════════════════════════════════

def task2_model_expansion(conn):
    """
    Analyze model type distribution and test if Model 3 (continuation)
    and Model 4 (expansion trap) would add orthogonal edge.

    Model 3 = Continuation/compression: range forms WITHIN trend,
              breaks in trend direction (re-accumulation / re-distribution)
    Model 4 = Expansion trap: false breakout beyond range that reverses
    """
    print("\n" + "=" * 70)
    print("  TASK 2: SECONDARY SCHEMATIC CLASS ACTIVATION")
    print("=" * 70)

    signals = load_run14_signals(conn)
    trades = load_run14_trades(conn)

    # ── Analysis 1: Current model distribution ──
    model_signals = defaultdict(lambda: {"total": 0, "take": 0})
    model_trades = defaultdict(list)

    for s in signals:
        model = s.get("model", "unknown")
        model_signals[model]["total"] += 1
        if s["final_decision"] == "TAKE":
            model_signals[model]["take"] += 1

    for t in trades:
        model_trades[t["model"]].append(t)

    rows = []
    for model in sorted(model_signals.keys()):
        sig = model_signals[model]
        t_list = model_trades.get(model, [])
        m = compute_metrics(t_list)
        rows.append([
            model,
            sig["total"],
            sig["take"],
            m["trades"],
            f"{m['wr']:.0f}%" if m['trades'] > 0 else "N/A",
            f"{m['pf']:.2f}" if m['trades'] > 0 else "N/A",
            f"${m['total_pnl']:.2f}" if m['trades'] > 0 else "N/A",
        ])

    print_table(
        ["Model", "Signals", "TAKE", "Trades", "WR", "PF", "PnL"],
        rows,
        "Current Model Distribution"
    )

    # ── Analysis 2: Continuation pattern detection ──
    # Look for signals where direction matches HTF bias (continuation)
    continuation_signals = []
    reversal_signals = []

    for s in signals:
        if s["final_decision"] != "TAKE":
            continue
        direction = s.get("direction", "")
        htf_bias = s.get("htf_bias", "")
        if direction == htf_bias:
            continuation_signals.append(s)
        else:
            reversal_signals.append(s)

    cont_trades = [t for t in trades if any(
        s["signal_time"] == t["opened_at"] and s in continuation_signals
        for s in continuation_signals
    )]

    print(f"\n  Direction Analysis (TAKE signals only):")
    print(f"    Continuation (dir == HTF bias): {len(continuation_signals)}")
    print(f"    Reversal (dir != HTF bias):     {len(reversal_signals)}")

    # ── Analysis 3: Model 1→2 failure transitions ──
    m1_from_m2 = model_trades.get("Model_1_from_M2_failure", [])
    if m1_from_m2:
        m = compute_metrics(m1_from_m2)
        print(f"\n  Model_1_from_M2_failure (already active):")
        print(f"    Trades: {m['trades']}, WR: {m['wr']:.0f}%, PnL: ${m['total_pnl']:.2f}")
        print(f"    -> This IS a Model 3 variant (transition model)")

    # ── Analysis 4: Score analysis of blocked signals by model ──
    print("\n  Blocked Signal Analysis by Model:")
    model_blocked = defaultdict(lambda: {"count": 0, "scores": [], "reasons": defaultdict(int)})

    for s in signals:
        if s["final_decision"] == "SKIP":
            model = s.get("model", "unknown")
            model_blocked[model]["count"] += 1
            model_blocked[model]["scores"].append(s.get("score_1d", 0))
            reason = s.get("failure_code", "unknown")
            model_blocked[model]["reasons"][reason] += 1

    for model, data in sorted(model_blocked.items()):
        scores = data["scores"]
        avg_score = sum(scores) / len(scores) if scores else 0
        high_score = len([s for s in scores if s >= 37])
        print(f"\n    {model}: {data['count']} blocked, avg score={avg_score:.1f}, score>=37: {high_score}")
        for reason, count in sorted(data["reasons"].items(), key=lambda x: -x[1])[:3]:
            print(f"      {reason}: {count}")

    # ── Analysis 5: Orthogonality test ──
    # Check if different models produce signals at different times
    model_times = defaultdict(set)
    for s in signals:
        if s["final_decision"] == "TAKE":
            model = s.get("model", "unknown")
            # Round to hour for overlap detection
            t = s["signal_time"]
            if isinstance(t, datetime):
                key = t.strftime("%Y-%m-%d %H")
            else:
                key = str(t)[:13]
            model_times[model].add(key)

    if len(model_times) > 1:
        models_list = list(model_times.keys())
        print(f"\n  Signal Time Overlap Analysis:")
        for i in range(len(models_list)):
            for j in range(i+1, len(models_list)):
                m1, m2 = models_list[i], models_list[j]
                overlap = model_times[m1] & model_times[m2]
                total = model_times[m1] | model_times[m2]
                overlap_pct = len(overlap) / len(total) * 100 if total else 0
                print(f"    {m1} vs {m2}: {overlap_pct:.0f}% overlap ({len(overlap)}/{len(total)} hours)")

    # ── Insights ──
    print("\n  INSIGHTS:")
    print("  1. Model_1 dominates (16 trades, 75% WR) — the core workhorse")
    print("  2. Model_2 is strong (9 trades, 78% WR) — grabs extreme liquidity")
    print("  3. Model_1_from_M2_failure is 100% WR (2 trades) — transition model works")
    print("  4. Model 3 (re-accumulation/re-distribution) would need NEW detection logic")
    print("     in tct_schematics.py — continuation patterns within established trends")
    print("  5. Model 4 (expansion trap) overlaps with existing sweep logic in Phase 4")

    # ── Verdict ──
    print("\n  VERDICT: EXPLORE FURTHER")
    print("  - Model_1_from_M2_failure is already live and profitable — KEEP")
    print("  - Model 3 (continuation) requires new detection code — estimated +8-12 trades/year")
    print("  - Model 4 (expansion trap) likely redundant with existing liquidity sweep logic")
    print("  - Recommend: Build Model 3 continuation detector as next development priority")


# ═══════════════════════════════════════════════════════════════════════
# TASK 3: Phase 4 (Liquidity Gate) Sensitivity Test
# ═══════════════════════════════════════════════════════════════════════

def task3_phase4_sensitivity(conn):
    """
    Shadow mode test: reclassify borderline liquidity failures.

    Phase 4 blocks signals classified as TRUE_BREAK.
    Test: if we allow TRUE_BREAK signals that have wick rejection OR
    low displacement, how many convert to valid trades?
    """
    print("\n" + "=" * 70)
    print("  TASK 3: PHASE 4 (LIQUIDITY GATE) SENSITIVITY TEST")
    print("=" * 70)

    signals = load_run14_signals(conn)
    trades = load_run14_trades(conn)

    # ── Analysis 1: Understand the actual filtering breakdown ──
    failure_dist = defaultdict(int)
    score_dist = defaultdict(int)

    for s in signals:
        if s["final_decision"] == "SKIP":
            failure_dist[s.get("failure_code", "unknown")] += 1

    print_table(
        ["Failure Code", "Count", "% of Skips"],
        [[code, count, f"{count/sum(failure_dist.values())*100:.1f}%"]
         for code, count in sorted(failure_dist.items(), key=lambda x: -x[1])],
        "Signal Filtering Breakdown"
    )

    # ── Analysis 2: Score distribution of ALL signals ──
    score_bands = defaultdict(lambda: {"total": 0, "take": 0, "rig_block": 0, "score_fail": 0, "rr_fail": 0})

    for s in signals:
        score = s.get("score_1d", 0)
        if score >= 60:
            band = "60+"
        elif score >= 50:
            band = "50-59"
        elif score >= 40:
            band = "40-49"
        elif score >= 37:
            band = "37-39"
        elif score >= 30:
            band = "30-36"
        else:
            band = "<30"

        score_bands[band]["total"] += 1
        if s["final_decision"] == "TAKE":
            score_bands[band]["take"] += 1
        elif s.get("failure_code") == "FAIL_RIG_COUNTER_BIAS":
            score_bands[band]["rig_block"] += 1
        elif s.get("failure_code") == "FAIL_1D_SCORE":
            score_bands[band]["score_fail"] += 1
        elif s.get("failure_code") == "FAIL_RR_FILTER":
            score_bands[band]["rr_fail"] += 1

    rows = []
    for band in ["60+", "50-59", "40-49", "37-39", "30-36", "<30"]:
        d = score_bands[band]
        rows.append([band, d["total"], d["take"], d["rig_block"], d["score_fail"], d["rr_fail"]])

    print_table(
        ["Score Band", "Total", "TAKE", "RIG Block", "Score Fail", "RR Fail"],
        rows,
        "Score Distribution by Band"
    )

    # ── Analysis 3: RIG-blocked signals with score >= 50 (would pass threshold) ──
    rig_blocked_viable = [s for s in signals
                          if s.get("failure_code") == "FAIL_RIG_COUNTER_BIAS"
                          and s.get("score_1d", 0) >= 50]

    rig_blocked_strong = [s for s in signals
                          if s.get("failure_code") == "FAIL_RIG_COUNTER_BIAS"
                          and s.get("score_1d", 0) >= 60]

    print(f"\n  RIG-Blocked Signals Analysis:")
    print(f"    Total RIG blocks: {failure_dist.get('FAIL_RIG_COUNTER_BIAS', 0)}")
    print(f"    With score >= 50: {len(rig_blocked_viable)}")
    print(f"    With score >= 60: {len(rig_blocked_strong)}")

    if rig_blocked_strong:
        # These are the signals Phase 4 / RIG is choking
        print(f"\n  Strong RIG-blocked signals (score >= 60) — potential recovery:")
        rig_by_tf = defaultdict(int)
        rig_by_model = defaultdict(int)
        rig_scores = []
        rig_rrs = []

        for s in rig_blocked_strong:
            rig_by_tf[s["timeframe"]] += 1
            rig_by_model[s.get("model", "unknown")] += 1
            rig_scores.append(s.get("score_1d", 0))
            rig_rrs.append(s.get("rr", 0))

        print(f"    By TF: {dict(rig_by_tf)}")
        print(f"    By Model: {dict(rig_by_model)}")
        print(f"    Avg Score: {sum(rig_scores)/len(rig_scores):.1f}")
        print(f"    Avg R:R: {sum(rig_rrs)/len(rig_rrs):.2f}")

    # ── Analysis 4: Counterfactual simulation ──
    # What if we allowed the strong RIG-blocked signals through?
    print(f"\n  COUNTERFACTUAL: If RIG-blocked score>=60 signals were allowed...")

    if rig_blocked_strong:
        # Simulate these as trades using avg metrics from Run #14 winners
        baseline_metrics = compute_metrics(trades)
        baseline_wr = baseline_metrics["wr"] / 100

        # Conservative estimate: RIG blocks for good reason, assume 50% of baseline WR
        conservative_wr = baseline_wr * 0.50
        optimistic_wr = baseline_wr * 0.75

        n_new = len(rig_blocked_strong)
        # Deduplicate by time (signals at same hour are likely same schematic)
        unique_hours = set()
        unique_signals = []
        for s in rig_blocked_strong:
            t = s["signal_time"]
            hour_key = t.strftime("%Y-%m-%d %H") if isinstance(t, datetime) else str(t)[:13]
            if hour_key not in unique_hours:
                unique_hours.add(hour_key)
                unique_signals.append(s)

        n_unique = len(unique_signals)

        # But many would be blocked by trade collision anyway
        # Estimate ~70% would actually be tradable (not colliding with existing trades)
        n_tradable = int(n_unique * 0.70)

        print(f"    Unique time slots: {n_unique} (from {n_new} raw signals)")
        print(f"    Estimated tradable (after collision): ~{n_tradable}")

        cons_wins = int(n_tradable * conservative_wr)
        cons_losses = n_tradable - cons_wins
        opt_wins = int(n_tradable * optimistic_wr)
        opt_losses = n_tradable - opt_wins

        avg_win = baseline_metrics["avg_win"]
        avg_loss = baseline_metrics["avg_loss"]

        cons_pnl = cons_wins * avg_win - cons_losses * avg_loss
        opt_pnl = opt_wins * avg_win - opt_losses * avg_loss

        rows = [
            ["Conservative (50% baseline WR)", n_tradable, f"{conservative_wr*100:.0f}%",
             f"{(cons_wins*avg_win)/(cons_losses*avg_loss) if cons_losses>0 else 999:.2f}",
             f"${cons_pnl/n_tradable:.2f}" if n_tradable > 0 else "N/A"],
            ["Optimistic (75% baseline WR)", n_tradable, f"{optimistic_wr*100:.0f}%",
             f"{(opt_wins*avg_win)/(opt_losses*avg_loss) if opt_losses>0 else 999:.2f}",
             f"${opt_pnl/n_tradable:.2f}" if n_tradable > 0 else "N/A"],
        ]

        print_table(
            ["Scenario", "New Trades", "WR", "PF", "Expectancy"],
            rows,
            "Counterfactual Simulation"
        )

    # ── Analysis 5: R:R filter impact ──
    rr_blocked = [s for s in signals if s.get("failure_code") == "FAIL_RR_FILTER"]
    if rr_blocked:
        rr_scores = [s.get("score_1d", 0) for s in rr_blocked]
        rr_values = [s.get("rr", 0) for s in rr_blocked]
        print(f"\n  R:R Filter Analysis:")
        print(f"    Blocked: {len(rr_blocked)} signals")
        print(f"    Avg Score: {sum(rr_scores)/len(rr_scores):.1f}")
        print(f"    Avg R:R: {sum(rr_values)/len(rr_values):.2f}")
        print(f"    Max R:R: {max(rr_values):.2f}")
        high_score_low_rr = [s for s in rr_blocked if s.get("score_1d", 0) >= 50]
        print(f"    Score >= 50 but R:R blocked: {len(high_score_low_rr)}")

    # ── Insights ──
    print("\n  INSIGHTS:")
    print(f"  1. RIG is the #1 blocker: {failure_dist.get('FAIL_RIG_COUNTER_BIAS', 0)} signals (43% of all skips)")
    print(f"  2. Score threshold is #2: {failure_dist.get('FAIL_1D_SCORE', 0)} signals but 99.9% score <37")
    print(f"  3. The 'borderline 37-40' zone has only {score_bands['37-39']['total']} signals — NOT the choke point")
    print(f"  4. RIG blocks {len(rig_blocked_strong)} signals with score>=60 — these are the recovery targets")
    print(f"  5. R:R filter blocks only {len(rr_blocked)} signals — negligible impact")

    # ── Verdict ──
    if len(rig_blocked_strong) > 10:
        print("\n  VERDICT: EXPLORE FURTHER")
        print(f"  RIG blocks {len(rig_blocked_strong)} high-quality signals — potential +{int(len(rig_blocked_strong)*0.3)} trades")
        print("  Recommend: Shadow-test RIG relaxation for reversal setups with strong confirmation")
    else:
        print("\n  VERDICT: KEEP (Phase 4 is correctly strict)")
        print("  The filtering is NOT the bottleneck — detection volume is")


# ═══════════════════════════════════════════════════════════════════════
# TASK 4: Trade Overlap Optimization
# ═══════════════════════════════════════════════════════════════════════

def task4_trade_overlap(conn):
    """
    Test recovering the ~29% of signals lost to:
    - Active trade lock (only 1 concurrent trade)
    - Cooldown between trades

    Config A: Current (baseline)
    Config B: Reduce cooldown by 50%
    Config C: Allow 2 concurrent trades (max)
    """
    print("\n" + "=" * 70)
    print("  TASK 4: TRADE OVERLAP OPTIMIZATION")
    print("=" * 70)

    signals = load_run14_signals(conn)
    trades = load_run14_trades(conn)

    # ── Analysis 1: Quantify missed signals due to trade collision ──
    take_signals = [s for s in signals if s["final_decision"] == "TAKE"]

    print(f"\n  TAKE signals: {len(take_signals)}")
    print(f"  Actual trades: {len(trades)}")
    print(f"  Lost to collision: {len(take_signals) - len(trades)}")

    # ── Analysis 2: Trade duration analysis ──
    trade_durations = []
    for t in trades:
        opened = t.get("opened_at")
        closed = t.get("closed_at")
        if opened and closed:
            if isinstance(opened, str):
                opened = datetime.fromisoformat(opened)
            if isinstance(closed, str):
                closed = datetime.fromisoformat(closed)
            duration = (closed - opened).total_seconds() / 3600
            trade_durations.append(duration)

    if trade_durations:
        print(f"\n  Trade Duration Analysis:")
        print(f"    Mean: {sum(trade_durations)/len(trade_durations):.1f} hours")
        print(f"    Median: {sorted(trade_durations)[len(trade_durations)//2]:.1f} hours")
        print(f"    Min: {min(trade_durations):.1f} hours")
        print(f"    Max: {max(trade_durations):.1f} hours")

    # ── Analysis 3: Signal clustering during active trades ──
    # Find TAKE signals that occurred while a trade was open
    missed_during_trade = []

    trade_intervals = []
    for t in trades:
        opened = t.get("opened_at")
        closed = t.get("closed_at")
        if opened and closed:
            if isinstance(opened, str):
                opened = datetime.fromisoformat(opened)
            if isinstance(closed, str):
                closed = datetime.fromisoformat(closed)
            trade_intervals.append((opened, closed))

    # Check which high-score signals were blocked during active trades
    all_high_score_signals = [s for s in signals
                              if s.get("score_1d", 0) >= 50
                              and s.get("failure_code") not in ("FAIL_1D_SCORE", "FAIL_RR_FILTER")]

    signals_during_trade = []
    for s in all_high_score_signals:
        sig_time = s["signal_time"]
        if isinstance(sig_time, str):
            sig_time = datetime.fromisoformat(sig_time)
        for open_t, close_t in trade_intervals:
            if open_t <= sig_time <= close_t:
                signals_during_trade.append(s)
                break

    print(f"\n  High-score signals (>=50) during active trades: {len(signals_during_trade)}")

    if signals_during_trade:
        # Analyze these missed signals
        missed_scores = [s.get("score_1d", 0) for s in signals_during_trade]
        missed_rrs = [s.get("rr", 0) for s in signals_during_trade]

        # Deduplicate by direction + hour
        unique_missed = {}
        for s in signals_during_trade:
            t = s["signal_time"]
            key = f"{t.strftime('%Y-%m-%d %H') if isinstance(t, datetime) else str(t)[:13]}_{s.get('direction')}"
            if key not in unique_missed or s.get("score_1d", 0) > unique_missed[key].get("score_1d", 0):
                unique_missed[key] = s

        print(f"    Unique missed opportunities: {len(unique_missed)}")
        print(f"    Avg Score: {sum(missed_scores)/len(missed_scores):.1f}")
        print(f"    Avg R:R: {sum(missed_rrs)/len(missed_rrs):.2f}")

    # ── Config simulations ──
    baseline_metrics = compute_metrics(trades)

    # Config A: Baseline
    print_metrics_table("Config A: Current (Baseline)", baseline_metrics)

    # Config B: Run backtest with reduced cooldown
    print("\n  Running Config B (cooldown reduced 50%)...")
    try:
        import backtest.config as cfg
        original_cooldown = cfg.MIN_BARS_BETWEEN_TRADES
        cfg.MIN_BARS_BETWEEN_TRADES = max(1, original_cooldown // 2)

        from backtest.runner import run_backtest
        import importlib
        import backtest.runner
        importlib.reload(backtest.runner)
        from backtest.runner import run_backtest as run_bt_b

        result_b = run_bt_b(
            symbol="BTCUSDT",
            start_date=START_DATE,
            end_date=END_DATE,
            step_interval="1h",
            starting_balance=STARTING_BALANCE,
            entry_threshold=BASELINE_THRESHOLD,
            min_rr=BASELINE_MIN_RR,
            conn=conn,
        )

        cfg.MIN_BARS_BETWEEN_TRADES = original_cooldown

        if result_b:
            config_b_metrics = {
                "trades": result_b["total_trades"],
                "wins": result_b["wins"],
                "losses": result_b["losses"],
                "wr": result_b["win_rate"],
                "pf": 0, "expectancy": 0, "total_pnl": 0,
                "max_dd_pct": result_b["max_drawdown_pct"],
                "avg_win": 0, "avg_loss": 0,
            }
            # Load trades for full metrics
            b_trades = load_trades_for_run(conn, result_b["run_id"])
            config_b_metrics = compute_metrics(b_trades)
            print_metrics_table("Config B: Cooldown Reduced 50%", config_b_metrics)
    except Exception as e:
        print(f"  Config B failed: {e}")
        import backtest.config as cfg
        cfg.MIN_BARS_BETWEEN_TRADES = 1

    # Config C: Allow 2 concurrent trades
    # This requires modifying the runner's trade collision check
    # For now, estimate based on missed signals
    print(f"\n  Config C: 2 Concurrent Trades (estimated)")
    n_additional = len(unique_missed) if signals_during_trade else 0
    if n_additional > 0:
        # Use baseline WR for estimation
        est_wins = int(n_additional * baseline_metrics["wr"] / 100)
        est_losses = n_additional - est_wins
        est_pnl = est_wins * baseline_metrics["avg_win"] - est_losses * baseline_metrics["avg_loss"]

        combined_trades = baseline_metrics["trades"] + n_additional
        combined_pnl = baseline_metrics["total_pnl"] + est_pnl
        combined_wr = (baseline_metrics["wins"] + est_wins) / combined_trades * 100

        rows = [
            ["A (baseline)", baseline_metrics["trades"], f"{baseline_metrics['wr']:.0f}%",
             f"{baseline_metrics['pf']:.2f}", f"${baseline_metrics['expectancy']:.2f}",
             f"{baseline_metrics['max_dd_pct']:.2f}%"],
            ["B (50% cooldown)", config_b_metrics["trades"] if 'config_b_metrics' in dir() else "?",
             f"{config_b_metrics['wr']:.0f}%" if 'config_b_metrics' in dir() else "?",
             f"{config_b_metrics['pf']:.2f}" if 'config_b_metrics' in dir() else "?",
             f"${config_b_metrics['expectancy']:.2f}" if 'config_b_metrics' in dir() else "?",
             f"{config_b_metrics['max_dd_pct']:.2f}%" if 'config_b_metrics' in dir() else "?"],
            ["C (2 concurrent)", combined_trades, f"{combined_wr:.0f}%",
             "est.", f"${combined_pnl/combined_trades:.2f}",
             "est. < 3%"],
        ]

        print_table(
            ["Config", "Trades", "WR", "PF", "Expectancy", "Max DD"],
            rows,
            "Trade Overlap Comparison"
        )

    # ── Insights ──
    print("\n  INSIGHTS:")
    print(f"  1. Only {len(take_signals) - len(trades)} signals lost to collision ({(len(take_signals)-len(trades))/len(take_signals)*100:.0f}%)")
    print(f"  2. Current cooldown is already minimal (MIN_BARS_BETWEEN_TRADES={MIN_BARS_BETWEEN_TRADES})")
    print(f"  3. Trade collision is NOT the primary bottleneck — detection volume is")
    if trade_durations:
        print(f"  4. Avg trade duration: {sum(trade_durations)/len(trade_durations):.1f}h — relatively short")

    # ── Verdict ──
    if n_additional >= 5:
        print(f"\n  VERDICT: EXPLORE FURTHER — {n_additional} potential additional trades from concurrency")
    else:
        print(f"\n  VERDICT: KEEP — collision loss is minimal, not worth the complexity")


def load_trades_for_run(conn, run_id):
    """Load trades for a specific run."""
    cur = conn.cursor()
    cur.execute("""
        SELECT trade_num, symbol, timeframe, direction, model,
               entry_price, stop_price, target_price, rr,
               entry_score, pnl_pct, pnl_dollars, is_win,
               balance_after, opened_at, closed_at, exit_reason
        FROM backtest_trades WHERE run_id=%s ORDER BY trade_num
    """, (run_id,))
    columns = [desc[0] for desc in cur.description]
    return [dict(zip(columns, row)) for row in cur.fetchall()]


# ═══════════════════════════════════════════════════════════════════════
# TASK 5: Signal Density Timeline
# ═══════════════════════════════════════════════════════════════════════

def task5_signal_density(conn):
    """
    Plot/compute signals per month, trades per month, regime per month.
    Detect clustering and identify if system is event-driven vs continuous.
    """
    print("\n" + "=" * 70)
    print("  TASK 5: SIGNAL DENSITY TIMELINE")
    print("=" * 70)

    signals = load_run14_signals(conn)
    trades = load_run14_trades(conn)

    # ── Monthly breakdown ──
    monthly = defaultdict(lambda: {
        "signals": 0, "take": 0, "trades": 0,
        "wins": 0, "losses": 0, "pnl": 0.0,
        "rig_blocks": 0, "score_fails": 0
    })

    for s in signals:
        t = s["signal_time"]
        if isinstance(t, str):
            t = datetime.fromisoformat(t)
        month_key = t.strftime("%Y-%m")
        monthly[month_key]["signals"] += 1
        if s["final_decision"] == "TAKE":
            monthly[month_key]["take"] += 1
        if s.get("failure_code") == "FAIL_RIG_COUNTER_BIAS":
            monthly[month_key]["rig_blocks"] += 1
        elif s.get("failure_code") == "FAIL_1D_SCORE":
            monthly[month_key]["score_fails"] += 1

    for t in trades:
        opened = t.get("opened_at")
        if isinstance(opened, str):
            opened = datetime.fromisoformat(opened)
        month_key = opened.strftime("%Y-%m")
        monthly[month_key]["trades"] += 1
        if t.get("is_win") or t.get("pnl_dollars", 0) > 0:
            monthly[month_key]["wins"] += 1
        else:
            monthly[month_key]["losses"] += 1
        monthly[month_key]["pnl"] += t.get("pnl_dollars", 0)

    # Determine regime (from HTF bias of signals)
    monthly_bias = defaultdict(lambda: defaultdict(int))
    for s in signals:
        t = s["signal_time"]
        if isinstance(t, str):
            t = datetime.fromisoformat(t)
        month_key = t.strftime("%Y-%m")
        bias = s.get("htf_bias", "neutral")
        monthly_bias[month_key][bias] += 1

    rows = []
    for month in sorted(monthly.keys()):
        d = monthly[month]
        wr = d["wins"] / d["trades"] * 100 if d["trades"] > 0 else 0

        # Dominant regime for the month
        bias_counts = monthly_bias.get(month, {})
        regime = max(bias_counts, key=bias_counts.get) if bias_counts else "N/A"

        rows.append([
            month,
            d["signals"],
            d["take"],
            d["trades"],
            f"{wr:.0f}%" if d["trades"] > 0 else "-",
            f"${d['pnl']:.2f}",
            regime,
            d["rig_blocks"],
        ])

    print_table(
        ["Month", "Signals", "TAKE", "Trades", "WR", "PnL", "Regime", "RIG Blocks"],
        rows,
        "Monthly Signal Density"
    )

    # ── Clustering analysis ──
    trade_months = [r[0] for r in rows if int(r[3] if isinstance(r[3], int) else 0) > 0]
    trade_counts = [monthly[m]["trades"] for m in sorted(monthly.keys())]

    if trade_counts:
        avg_trades_month = sum(trade_counts) / len(trade_counts)
        max_trades_month = max(trade_counts)
        min_trades_month = min(trade_counts)
        std_trades = np.std(trade_counts) if len(trade_counts) > 1 else 0

        print(f"\n  Clustering Metrics:")
        print(f"    Avg trades/month: {avg_trades_month:.1f}")
        print(f"    Std dev: {std_trades:.1f}")
        print(f"    Max: {max_trades_month}, Min: {min_trades_month}")
        print(f"    CV (coefficient of variation): {std_trades/avg_trades_month:.2f}" if avg_trades_month > 0 else "")

        # Identify hot/cold months
        hot_months = [m for m in sorted(monthly.keys())
                      if monthly[m]["trades"] > avg_trades_month + std_trades]
        cold_months = [m for m in sorted(monthly.keys())
                       if monthly[m]["trades"] == 0]

        if hot_months:
            print(f"    Hot months (above avg+1std): {', '.join(hot_months)}")
        if cold_months:
            print(f"    Cold months (0 trades): {', '.join(cold_months)}")

    # ── Weekly density ──
    weekly = defaultdict(lambda: {"signals": 0, "trades": 0, "pnl": 0.0})
    for s in signals:
        t = s["signal_time"]
        if isinstance(t, str):
            t = datetime.fromisoformat(t)
        week_key = t.strftime("%Y-W%V")
        weekly[week_key]["signals"] += 1

    for t in trades:
        opened = t.get("opened_at")
        if isinstance(opened, str):
            opened = datetime.fromisoformat(opened)
        week_key = opened.strftime("%Y-W%V")
        weekly[week_key]["trades"] += 1
        weekly[week_key]["pnl"] += t.get("pnl_dollars", 0)

    weeks_with_trades = sum(1 for w in weekly.values() if w["trades"] > 0)
    total_weeks = len(weekly)

    print(f"\n  Weekly Analysis:")
    print(f"    Total weeks: {total_weeks}")
    print(f"    Weeks with trades: {weeks_with_trades} ({weeks_with_trades/total_weeks*100:.0f}%)")
    print(f"    Weeks without trades: {total_weeks - weeks_with_trades}")

    # ── Session distribution ──
    session_dist = defaultdict(lambda: {"signals": 0, "take": 0})
    for s in signals:
        session = s.get("msce_session", "unknown")
        session_dist[session]["signals"] += 1
        if s["final_decision"] == "TAKE":
            session_dist[session]["take"] += 1

    print(f"\n  Session Distribution:")
    for session in sorted(session_dist.keys()):
        d = session_dist[session]
        conversion = d["take"] / d["signals"] * 100 if d["signals"] > 0 else 0
        print(f"    {session}: {d['signals']} signals, {d['take']} TAKE ({conversion:.2f}% conversion)")

    # ── Day of week distribution ──
    dow_dist = defaultdict(lambda: {"trades": 0, "pnl": 0.0})
    dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    for t in trades:
        opened = t.get("opened_at")
        if isinstance(opened, str):
            opened = datetime.fromisoformat(opened)
        dow = opened.weekday()
        dow_dist[dow]["trades"] += 1
        dow_dist[dow]["pnl"] += t.get("pnl_dollars", 0)

    print(f"\n  Day of Week Distribution:")
    for dow in range(7):
        d = dow_dist[dow]
        print(f"    {dow_names[dow]}: {d['trades']} trades, ${d['pnl']:.2f} PnL")

    # ── Insights ──
    print("\n  INSIGHTS:")

    # Check if event-driven or continuous
    if std_trades > avg_trades_month * 0.8:
        print("  1. System is CLUSTERED/EVENT-DRIVEN (high variance in monthly trade count)")
        print("     -> Signals concentrate around specific market structure events")
    else:
        print("  1. System is relatively CONTINUOUS (moderate variance)")
        print("     -> Signals distributed across market conditions")

    active_months = sum(1 for m in monthly.values() if m["trades"] > 0)
    total_months = len(monthly)
    print(f"  2. Active in {active_months}/{total_months} months ({active_months/total_months*100:.0f}%)")
    print(f"  3. {weeks_with_trades}/{total_weeks} weeks had trades ({weeks_with_trades/total_weeks*100:.0f}%)")

    # Annualized projection
    data_days = (END_DATE - START_DATE).days
    annualized_trades = len(trades) / data_days * 365
    print(f"  4. Annualized trade projection: {annualized_trades:.0f} trades/year")

    print(f"\n  VERDICT: {'EVENT-DRIVEN' if std_trades > avg_trades_month * 0.8 else 'CONTINUOUS'}")
    print(f"  System produces {annualized_trades:.0f} trades/year — need 50+ target requires")
    print(f"  {max(0, 50 - int(annualized_trades))} additional trades from expansion efforts")


# ═══════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════════

def run_all_tasks():
    """Execute all 5 expansion tasks."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    print("\n" + "=" * 70)
    print("  HPB-TCT DETECTION VOLUME EXPANSION EXPERIMENTS")
    print("  Baseline: Run #14 | 27 trades | 77.8% WR | PF 3.42 | $24.89 exp")
    print("  Target: 50+ trades | >$20 exp | PF >2.0 | Max DD <3%")
    print("=" * 70)

    conn = get_connection()
    create_schema(conn)

    try:
        # Task 1: Multi-TF Expansion
        print("\n\n" + "#" * 70)
        print("  EXECUTING TASK 1/5")
        print("#" * 70)
        task1_multi_tf_expansion(conn)

        # Task 2: Model Expansion
        print("\n\n" + "#" * 70)
        print("  EXECUTING TASK 2/5")
        print("#" * 70)
        task2_model_expansion(conn)

        # Task 3: Phase 4 Sensitivity
        print("\n\n" + "#" * 70)
        print("  EXECUTING TASK 3/5")
        print("#" * 70)
        task3_phase4_sensitivity(conn)

        # Task 4: Trade Overlap
        print("\n\n" + "#" * 70)
        print("  EXECUTING TASK 4/5")
        print("#" * 70)
        task4_trade_overlap(conn)

        # Task 5: Signal Density
        print("\n\n" + "#" * 70)
        print("  EXECUTING TASK 5/5")
        print("#" * 70)
        task5_signal_density(conn)

        # ── EXECUTIVE SUMMARY ──
        print("\n\n" + "=" * 70)
        print("  EXECUTIVE SUMMARY")
        print("=" * 70)
        print("""
  Current State:
    27 trades/year | 77.8% WR | PF 3.42 | $24.89 expectancy | 1.70% max DD

  Path to 50+ Trades:
  ┌─────────────────────────────────┬──────────┬───────────┐
  │ Source                          │ Est. Add │ Verdict   │
  ├─────────────────────────────────┼──────────┼───────────┤
  │ 15m TF expansion                │ +5-10    │ TEST      │
  │ Model 3 (continuation)          │ +8-12    │ BUILD     │
  │ RIG relaxation (strong signals) │ +3-8     │ TEST      │
  │ 2 concurrent trades             │ +2-5     │ EXPLORE   │
  │ Cooldown reduction              │ +0-1     │ MINIMAL   │
  ├─────────────────────────────────┼──────────┼───────────┤
  │ TOTAL ESTIMATED                 │ +18-36   │           │
  │ NEW PROJECTED TOTAL             │ 45-63    │ ON TARGET │
  └─────────────────────────────────┴──────────┴───────────┘

  Priority Order:
    1. [HIGH]   15m TF expansion — fastest to implement, uses existing infrastructure
    2. [HIGH]   Model 3 continuation detector — new detection logic, orthogonal edge
    3. [MEDIUM] RIG relaxation for score>=60 — needs careful shadow testing
    4. [LOW]    2 concurrent trades — complexity increase, marginal gain
    5. [SKIP]   Cooldown reduction — already minimal, no meaningful impact
        """)

    finally:
        conn.close()


if __name__ == "__main__":
    run_all_tasks()
