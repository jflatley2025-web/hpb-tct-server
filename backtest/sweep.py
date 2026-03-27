"""
backtest/sweep.py -- Phase 2 Optimization Sweep Engine
======================================================
Runs parameterized backtest sweeps for:
  Task 1: Threshold x MIN_RR frequency expansion
  Task 2: TP1 split x TP1 level optimization
  Task 3: RIG counterfactual audit
  Task 4: Regime expansion check
  Task 5: Score band analysis

All tasks use the same 12-month dataset (Apr 2025 - Mar 2026) with 1h step
and 90-day warmup. Execution logic (TP1, trailing, RIG) remains identical
unless explicitly varied in Task 2.
"""

import json
import logging
import sys
import time
from datetime import datetime, timezone
from itertools import product

import pandas as pd

from backtest.config import (
    STARTING_BALANCE,
    WARMUP_DAYS,
    MIN_RR,
    TP1_POSITION_CLOSE_PCT,
    TRAIL_FACTOR,
)
from backtest.db import get_connection, create_schema
from backtest.runner import run_backtest
from backtest.reporter import (
    load_run,
    load_trades,
    load_signals,
    compute_standard_metrics,
    compute_advanced_metrics,
)

logger = logging.getLogger("backtest.sweep")

# -- Constants --------------------------------------------------------
SYMBOL = "BTCUSDT"
START_DATE = datetime(2025, 4, 1, tzinfo=timezone.utc)
END_DATE = datetime(2026, 3, 26, tzinfo=timezone.utc)
STEP = "1h"
WARMUP = WARMUP_DAYS  # 90 days


def _run_single(conn, label: str, **overrides) -> dict:
    """Run a single backtest with overrides, return enriched summary."""
    t0 = time.time()
    logger.info(f">> Starting: {label}")

    summary = run_backtest(
        symbol=SYMBOL,
        start_date=START_DATE,
        end_date=END_DATE,
        step_interval=STEP,
        starting_balance=STARTING_BALANCE,
        warmup_days=WARMUP,
        conn=conn,
        **overrides,
    )

    elapsed = time.time() - t0
    run_id = summary["run_id"]

    # Enrich with reporter metrics
    run_rec = load_run(conn, run_id)
    trades = load_trades(conn, run_id)
    signals = load_signals(conn, run_id)

    std = compute_standard_metrics(trades, run_rec)
    adv = compute_advanced_metrics(trades, signals)

    result = {
        "label": label,
        "run_id": run_id,
        "trades": std.get("total_trades", 0),
        "wins": std.get("wins", 0),
        "losses": std.get("losses", 0),
        "win_rate": std.get("win_rate", 0),
        "pf": std.get("profit_factor", 0),
        "expectancy": adv.get("expectancy", 0),
        "total_pnl": std.get("total_pnl", 0),
        "max_dd": std.get("max_drawdown_pct", 0),
        "avg_win": std.get("avg_win", 0),
        "avg_loss": std.get("avg_loss", 0),
        "wl_ratio": round(std.get("avg_win", 0) / std.get("avg_loss", 1), 2) if std.get("avg_loss", 0) > 0 else 0,
        "rig_blocks": adv.get("rig_total_blocks", 0),
        "elapsed_min": round(elapsed / 60, 1),
    }

    logger.info(
        f"OK {label}: {result['trades']}T, PF={result['pf']}, "
        f"E=${result['expectancy']}, DD={result['max_dd']}%, "
        f"elapsed={result['elapsed_min']}m"
    )
    return result


def _print_table(rows: list, title: str):
    """Print a formatted results table."""
    if not rows:
        print(f"\n{title}: No results")
        return

    print(f"\n{'='*100}")
    print(f"  {title}")
    print(f"{'='*100}")

    # Header
    cols = ["Label", "Run", "Trades", "W/L", "WR%", "PF", "E[$]", "PnL$", "DD%", "W/L$", "RIG", "Min"]
    widths = [28, 5, 6, 7, 6, 6, 7, 8, 6, 6, 6, 5]
    header = "".join(c.ljust(w) for c, w in zip(cols, widths))
    print(f"  {header}")
    print(f"  {'-'*sum(widths)}")

    for r in rows:
        wl = f"{r['wins']}/{r['losses']}"
        vals = [
            r["label"][:27],
            str(r["run_id"]),
            str(r["trades"]),
            wl,
            f"{r['win_rate']:.1f}",
            f"{r['pf']:.2f}",
            f"{r['expectancy']:.1f}",
            f"{r['total_pnl']:.0f}",
            f"{r['max_dd']:.2f}",
            f"{r['wl_ratio']:.2f}",
            str(r["rig_blocks"]),
            f"{r['elapsed_min']:.0f}",
        ]
        line = "".join(v.ljust(w) for v, w in zip(vals, widths))
        print(f"  {line}")

    print()


# -- TASK 1: Threshold x MIN_RR Sweep ---------------------------------

def task1_frequency_sweep(conn):
    """Sweep threshold in [45, 48, 50] x MIN_RR in [0.4, 0.5, 0.6]"""
    print("\n" + "#"*80)
    print("  TASK 1 -- TRADE FREQUENCY EXPANSION SWEEP")
    print("#"*80)

    thresholds = [45, 48, 50]
    min_rrs = [0.4, 0.5, 0.6]
    results = []

    for thresh, rr in product(thresholds, min_rrs):
        label = f"T{thresh}_RR{rr}"
        result = _run_single(
            conn, label,
            entry_threshold=thresh,
            min_rr=rr,
        )
        results.append(result)

    _print_table(results, "TASK 1: Threshold x MIN_RR Sweep Results")

    # Identify Pareto-optimal configs
    print("  PARETO ANALYSIS (higher trades WITHOUT PF collapse):")
    baseline = next((r for r in results if r["label"] == "T50_RR0.5"), results[0])
    print(f"  Baseline (T50/RR0.5): {baseline['trades']}T, PF={baseline['pf']}, E=${baseline['expectancy']}")
    print()

    pareto = [
        r for r in results
        if r["trades"] >= baseline["trades"]
        and r["pf"] >= 2.0
        and r["expectancy"] >= 20
        and r["max_dd"] < 3.0
    ]
    pareto.sort(key=lambda r: (r["trades"], r["pf"]), reverse=True)

    if pareto:
        print("  [PASS] Pareto-optimal configs (meet ALL success criteria):")
        for r in pareto:
            flag = " <- BEST" if r == pareto[0] else ""
            print(f"    {r['label']}: {r['trades']}T, PF={r['pf']}, E=${r['expectancy']}, DD={r['max_dd']}%{flag}")
    else:
        print("  [WARN] No config meets all criteria. Closest configs:")
        closest = sorted(results, key=lambda r: r["trades"], reverse=True)[:3]
        for r in closest:
            print(f"    {r['label']}: {r['trades']}T, PF={r['pf']}, E=${r['expectancy']}, DD={r['max_dd']}%")

    print()
    return results


# -- TASK 2: TP1 Optimization Curve -----------------------------------

def task2_tp1_optimization(conn, best_thresh=None, best_rr=None):
    """Sweep TP1 split in [0.4, 0.5, 0.6] x TP1 level in [0.4R, 0.5R, 0.6R]"""
    print("\n" + "#"*80)
    print("  TASK 2 -- TP1 OPTIMIZATION CURVE")
    print("#"*80)

    # Use best from Task 1 or defaults
    thresh = best_thresh or 50
    rr = best_rr or MIN_RR

    tp1_splits = [0.40, 0.50, 0.60]
    tp1_levels = [0.40, 0.50, 0.60]
    results = []

    for split, level in product(tp1_splits, tp1_levels):
        label = f"TP1_{int(split*100)}%@{level:.1f}R"
        result = _run_single(
            conn, label,
            entry_threshold=thresh,
            min_rr=rr,
            tp1_close_pct=split,
            tp1_level_pct=level,
        )
        results.append(result)

    _print_table(results, "TASK 2: TP1 Split x Level Optimization")

    # Rank by expectancy, then PF, then W/L ratio
    ranked = sorted(results, key=lambda r: (r["expectancy"], r["pf"], r["wl_ratio"]), reverse=True)
    print("  TOP 3 CONFIGS (ranked by Expectancy -> PF -> W/L Ratio):")
    for i, r in enumerate(ranked[:3], 1):
        flag = " <- RECOMMENDED" if i == 1 else ""
        print(
            f"    #{i} {r['label']}: E=${r['expectancy']}, PF={r['pf']}, "
            f"W/L={r['wl_ratio']}, WR={r['win_rate']}%{flag}"
        )

    # Check success criteria
    best = ranked[0]
    print(f"\n  SUCCESS CHECK:")
    print(f"    W/L ratio >= 1.05: {best['wl_ratio']} {'[PASS]' if best['wl_ratio'] >= 1.05 else '[FAIL]'}")
    print(f"    WR >= 70%: {best['win_rate']}% {'[PASS]' if best['win_rate'] >= 70 else '[FAIL]'}")
    print()
    return results


# -- TASK 3: RIG Counterfactual Audit --------------------------------

def task3_rig_counterfactual(conn):
    """Run with RIG disabled vs enabled, compare blocked signal outcomes."""
    print("\n" + "#"*80)
    print("  TASK 3 -- RIG EFFICIENCY AUDIT (COUNTERFACTUAL)")
    print("#"*80)

    # Run 1: Production (RIG enabled) -- use threshold 50 baseline
    result_rig_on = _run_single(conn, "RIG_ON (baseline)", entry_threshold=50)

    # Run 2: RIG disabled -- we simulate by setting a very low threshold
    # but keeping everything else. Actually, we need to disable RIG in the
    # gate pipeline. The cleanest approach: run with RIG enabled and then
    # query the blocked signals to simulate counterfactual trades.

    run_id_on = result_rig_on["run_id"]
    signals = load_signals(conn, run_id_on)

    # Get all RIG-blocked signals that had score >= threshold
    rig_blocked = signals[
        (signals["rig_status"] == "BLOCK") &
        (signals["score_1d"] >= 50)
    ].copy()

    total_blocked = len(rig_blocked)
    print(f"\n  RIG-blocked signals with score >= 50: {total_blocked}")

    if total_blocked == 0:
        print("  No high-score blocks to analyze.")
        return [result_rig_on]

    # For counterfactual: we can't replay trades easily, but we can
    # analyze the price action after each blocked signal to estimate
    # whether the trade would have won or lost.
    print(f"\n  Simulating counterfactual trades for {total_blocked} blocked signals...")

    # Load 1h candle data for forward simulation
    from backtest.ingest import load_candles
    candles_1h = load_candles(conn, SYMBOL, "1h", START_DATE, END_DATE)

    counterfactual_wins = 0
    counterfactual_losses = 0
    counterfactual_pnl = 0.0
    cf_details = []

    for _, sig in rig_blocked.iterrows():
        sig_time = pd.to_datetime(sig["signal_time"])
        if sig_time.tzinfo is None:
            sig_time = sig_time.tz_localize("UTC")

        entry_price = sig.get("entry_price", 0)
        stop_price = sig.get("stop_price", 0)
        target_price = sig.get("target_price", 0)
        direction = sig.get("direction", "")

        if not entry_price or not stop_price or not target_price:
            continue

        # Forward simulate using 1h candles after signal time
        future_candles = candles_1h[candles_1h["open_time"] > sig_time].head(200)
        if future_candles.empty:
            continue

        # Simulate: did price hit SL or TP first?
        # Also simulate TP1 at midpoint
        tp1_price = entry_price + (target_price - entry_price) * 0.5
        hit_tp1 = False
        result = "open"

        for _, candle in future_candles.iterrows():
            h = float(candle["high"])
            l = float(candle["low"])

            if direction == "bullish":
                sl_hit = l <= stop_price
                tp_hit = h >= target_price
                tp1_hit = h >= tp1_price
            else:
                sl_hit = h >= stop_price
                tp_hit = l <= target_price
                tp1_hit = l <= tp1_price

            if sl_hit and tp_hit:
                result = "loss"  # worst case
                break
            if sl_hit:
                if hit_tp1:
                    result = "breakeven"  # would have moved SL to BE
                else:
                    result = "loss"
                break
            if tp_hit:
                result = "win"
                break
            if tp1_hit:
                hit_tp1 = True

        # Estimate P&L
        if result == "win":
            sl_dist = abs(entry_price - stop_price)
            tp_dist = abs(target_price - entry_price)
            # Approximate: risk $50, win = rr * $50
            rr = tp_dist / sl_dist if sl_dist > 0 else 0
            est_pnl = 50 * rr * 0.5 + 50 * 0.5  # TP1 partial + full target (simplified)
            counterfactual_wins += 1
            counterfactual_pnl += est_pnl
        elif result == "loss":
            counterfactual_losses += 1
            counterfactual_pnl -= 50  # $50 risk
        elif result == "breakeven":
            counterfactual_pnl += 0  # BE after TP1, small win from partial
            counterfactual_wins += 1  # technically a small win

        cf_details.append({
            "time": str(sig_time),
            "direction": direction,
            "score": sig.get("score_1d", 0),
            "rr": sig.get("rr", 0),
            "result": result,
        })

    total_cf = counterfactual_wins + counterfactual_losses
    cf_wr = (counterfactual_wins / total_cf * 100) if total_cf > 0 else 0
    pct_losers = (counterfactual_losses / total_cf * 100) if total_cf > 0 else 0

    print(f"\n  COUNTERFACTUAL RESULTS ({total_cf} simulated trades):")
    print(f"  {'-'*50}")
    print(f"  Win rate of blocked signals:    {cf_wr:.1f}%")
    print(f"  Avg PnL of blocked signals:     ${counterfactual_pnl / total_cf:.2f}" if total_cf > 0 else "  N/A")
    print(f"  % blocked that would be losers: {pct_losers:.1f}%")
    print(f"  Estimated total PnL if unblocked: ${counterfactual_pnl:.2f}")

    # Result breakdown
    from collections import Counter
    outcome_counts = Counter(d["result"] for d in cf_details)
    print(f"\n  Outcome breakdown:")
    for outcome, count in outcome_counts.most_common():
        print(f"    {outcome}: {count} ({count/total_cf*100:.1f}%)" if total_cf > 0 else f"    {outcome}: {count}")

    # Key insight
    if pct_losers > 50:
        print(f"\n  [PASS] RIG IS PROTECTIVE: {pct_losers:.0f}% of blocked trades would have lost.")
        print(f"     RIG is alpha-preserving. Keep enabled.")
    else:
        print(f"\n  [WARN] RIG MAY BE OVER-FILTERING: only {pct_losers:.0f}% of blocked trades would have lost.")
        print(f"     Consider relaxing RIG conditions.")

    print()
    return [result_rig_on]


# -- TASK 4: Regime Expansion Check ----------------------------------

def task4_regime_analysis(conn):
    """Analyze signals and trades by HTF regime classification."""
    print("\n" + "#"*80)
    print("  TASK 4 -- REGIME EXPANSION CHECK")
    print("#"*80)

    # Run baseline to get signals
    result = _run_single(conn, "REGIME_ANALYSIS", entry_threshold=50)
    run_id = result["run_id"]

    signals = load_signals(conn, run_id)
    trades = load_trades(conn, run_id)

    if signals.empty:
        print("  No signals to analyze.")
        return [result]

    # Classify signals by HTF bias (regime proxy).
    # Each trade is assigned to at most one regime (the regime of the closest
    # TAKE signal within ±2h). The single-pass assignment prevents the same
    # trade from inflating multiple regime buckets.
    take_signals = signals[signals["final_decision"] == "TAKE"].copy()
    take_signals["signal_time_dt"] = pd.to_datetime(take_signals["signal_time"])
    trades_opened = pd.to_datetime(trades["opened_at"])

    trade_regime: dict = {}  # trade_idx -> regime string
    seen_trade_idxs: set = set()
    for _, trade in trades.iterrows():
        trade_idx = trade.name
        opened = trades_opened[trade_idx]
        if pd.isna(opened):
            continue
        window = take_signals[
            (take_signals["signal_time_dt"] >= opened - pd.Timedelta(hours=2)) &
            (take_signals["signal_time_dt"] <= opened + pd.Timedelta(hours=2))
        ]
        if window.empty:
            continue
        # Pick the temporally closest TAKE signal to assign the regime
        closest_idx = (window["signal_time_dt"] - opened).abs().idxmin()
        bias = window.loc[closest_idx, "gate_1a_bias"]
        regime = "unknown" if (pd.isna(bias) or bias == "") else bias
        if trade_idx not in seen_trade_idxs:
            seen_trade_idxs.add(trade_idx)
            trade_regime[trade_idx] = regime

    # Build per-regime trade lists from the single-pass assignment
    regime_trade_rows: dict = {}
    for trade_idx, regime in trade_regime.items():
        regime_trade_rows.setdefault(regime, []).append(trades.loc[trade_idx])

    regime_stats = {}
    for regime in signals["gate_1a_bias"].unique():
        reg_sigs = signals[signals["gate_1a_bias"] == regime]
        reg_takes = reg_sigs[reg_sigs["final_decision"] == "TAKE"]
        reg_skips = reg_sigs[reg_sigs["final_decision"] == "SKIP"]

        reg_trades_list = regime_trade_rows.get(regime, [])
        if reg_trades_list:
            reg_trades_df = pd.DataFrame(reg_trades_list)
            reg_wins = reg_trades_df["is_win"].sum()
            reg_pnl = reg_trades_df["pnl_dollars"].sum()
            reg_wr = (reg_wins / len(reg_trades_df)) * 100
        else:
            reg_trades_df = pd.DataFrame()
            reg_wins = 0
            reg_pnl = 0
            reg_wr = 0

        regime_stats[regime] = {
            "total_signals": len(reg_sigs),
            "taken": len(reg_takes),
            "skipped": len(reg_skips),
            "trades": len(reg_trades_df),
            "wins": int(reg_wins),
            "pnl": round(reg_pnl, 2),
            "win_rate": round(reg_wr, 1),
        }

    # Print regime table
    print(f"\n  {'Regime':<15} {'Signals':>8} {'Taken':>7} {'Skipped':>8} {'Trades':>7} {'Wins':>6} {'WR%':>6} {'PnL$':>8}")
    print(f"  {'-'*70}")
    for regime, stats in sorted(regime_stats.items(), key=lambda x: x[1]["total_signals"], reverse=True):
        print(
            f"  {regime:<15} {stats['total_signals']:>8} {stats['taken']:>7} "
            f"{stats['skipped']:>8} {stats['trades']:>7} {stats['wins']:>6} "
            f"{stats['win_rate']:>5.1f}% ${stats['pnl']:>7.0f}"
        )

    # Determine specialization
    total_trades = sum(s["trades"] for s in regime_stats.values())
    ranging_trades = sum(
        s["trades"] for r, s in regime_stats.items()
        if r in ("ranging", "neutral")
    )
    trend_trades = sum(
        s["trades"] for r, s in regime_stats.items()
        if r in ("bullish", "bearish", "trending")
    )

    print(f"\n  REGIME CONCENTRATION:")
    if total_trades > 0:
        print(f"    Range/Neutral trades: {ranging_trades} ({ranging_trades/total_trades*100:.0f}%)")
        print(f"    Trend trades:         {trend_trades} ({trend_trades/total_trades*100:.0f}%)")

    if ranging_trades > total_trades * 0.8:
        print(f"\n  [PASS] System is RANGE-SPECIALIZED (intentional)")
        print(f"     HPB schematics form during ranging markets -- this is expected.")
    elif trend_trades > total_trades * 0.3:
        print(f"\n  [WARN] System has TREND EXPOSURE -- review if detection logic captures trend setups")
    else:
        print(f"\n  [INFO] Mixed regime -- system operates across conditions")

    print()
    return [result]


# -- TASK 5: Score Band Expansion Analysis ---------------------------

def task5_score_band_analysis(conn):
    """Analyze score distribution to diagnose bimodality."""
    print("\n" + "#"*80)
    print("  TASK 5 -- SCORE BAND EXPANSION (BIMODALITY DIAGNOSIS)")
    print("#"*80)

    # Run baseline
    result = _run_single(conn, "SCORE_BAND_ANALYSIS", entry_threshold=50)
    run_id = result["run_id"]

    signals = load_signals(conn, run_id)
    if signals.empty:
        print("  No signals to analyze.")
        return [result]

    scores = signals["score_1d"].dropna()
    if scores.empty:
        print("  No scores available.")
        return [result]

    # Score distribution histogram
    print(f"\n  SCORE DISTRIBUTION (n={len(scores)}):")
    print(f"  {'-'*50}")

    bins = list(range(0, 101, 5))
    hist = pd.cut(scores, bins=bins, right=False).value_counts().sort_index()

    max_count = hist.max()
    for interval, count in hist.items():
        if count > 0:
            bar = "#" * int(count / max_count * 40) if max_count > 0 else ""
            pct = count / len(scores) * 100
            print(f"  {str(interval):<12} {count:>5} ({pct:>5.1f}%) {bar}")

    # Score stats
    print(f"\n  STATISTICS:")
    print(f"    Mean:   {scores.mean():.1f}")
    print(f"    Median: {scores.median():.1f}")
    print(f"    Std:    {scores.std():.1f}")
    print(f"    Min:    {scores.min():.0f}")
    print(f"    Max:    {scores.max():.0f}")

    # Bimodality detection: check gap between clusters
    below_thresh = scores[scores < 50]
    above_thresh = scores[scores >= 50]
    print(f"\n  CLUSTER ANALYSIS:")
    print(f"    Below threshold (< 50): {len(below_thresh)} signals, mean={below_thresh.mean():.1f}" if len(below_thresh) > 0 else "    Below threshold: 0")
    print(f"    Above threshold (>= 50): {len(above_thresh)} signals, mean={above_thresh.mean():.1f}" if len(above_thresh) > 0 else "    Above threshold: 0")

    # Gap analysis: check for dead zones
    mid_band = scores[(scores >= 42) & (scores <= 55)]
    print(f"    Mid-band (42-55):       {len(mid_band)} signals ({len(mid_band)/len(scores)*100:.1f}%)")

    if len(mid_band) < len(scores) * 0.1:
        print(f"\n  [WARN] BIMODALITY CONFIRMED: <10% of signals in mid-band")
        print(f"     Scores jump from ~{below_thresh.max():.0f} -> ~{above_thresh.min():.0f}" if len(below_thresh) > 0 and len(above_thresh) > 0 else "")
    else:
        print(f"\n  [PASS] Score distribution is reasonably continuous")

    # Analyze which gate failures cause the jump
    print(f"\n  GATE FAILURE DISTRIBUTION (for scores 35-55):")
    mid_signals = signals[(signals["score_1d"] >= 35) & (signals["score_1d"] <= 55)]
    if not mid_signals.empty and "failure_code" in mid_signals.columns:
        failure_dist = mid_signals["failure_code"].value_counts()
        for code, count in failure_dist.items():
            if code and str(code) != "nan":
                print(f"    {code}: {count} ({count/len(mid_signals)*100:.1f}%)")
        # Signals that passed all gates in this range
        passed = mid_signals[mid_signals["final_decision"] == "TAKE"]
        print(f"    TAKE (passed all): {len(passed)} ({len(passed)/len(mid_signals)*100:.1f}%)")

    # Direction + score analysis for signals near threshold
    print(f"\n  NEAR-THRESHOLD SIGNALS (score 45-55):")
    near = signals[(signals["score_1d"] >= 45) & (signals["score_1d"] <= 55)]
    if not near.empty:
        for direction in near["direction"].unique():
            d_sigs = near[near["direction"] == direction]
            d_takes = d_sigs[d_sigs["final_decision"] == "TAKE"]
            print(f"    {direction}: {len(d_sigs)} signals, {len(d_takes)} taken")

    print()
    return [result]


# -- Main Orchestrator ------------------------------------------------

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    conn = get_connection()
    create_schema(conn)

    all_results = {}

    try:
        # TASK 1: Frequency Expansion
        print("\n" + "="*80)
        print("  PHASE 2 OPTIMIZATION SWEEP -- STARTING")
        print("="*80)

        t1_results = task1_frequency_sweep(conn)
        all_results["task1"] = t1_results

        # Find best config from Task 1 for Task 2
        pareto = [
            r for r in t1_results
            if r["pf"] >= 2.0 and r["expectancy"] >= 20 and r["max_dd"] < 3.0
        ]
        if pareto:
            best_t1 = max(pareto, key=lambda r: (r["trades"], r["pf"]))
            # Parse label to extract params
            parts = best_t1["label"].split("_")
            best_thresh = int(parts[0][1:])  # T45 -> 45
            best_rr = float(parts[1][2:])    # RR0.4 -> 0.4
        else:
            best_thresh, best_rr = 50, 0.5

        # TASK 2: TP1 Optimization
        t2_results = task2_tp1_optimization(conn, best_thresh, best_rr)
        all_results["task2"] = t2_results

        # TASK 3: RIG Counterfactual
        t3_results = task3_rig_counterfactual(conn)
        all_results["task3"] = t3_results

        # TASK 4: Regime Analysis
        t4_results = task4_regime_analysis(conn)
        all_results["task4"] = t4_results

        # TASK 5: Score Band Analysis
        t5_results = task5_score_band_analysis(conn)
        all_results["task5"] = t5_results

        # -- FINAL SUMMARY ----------------------------------------
        print("\n" + "="*80)
        print("  PHASE 2 OPTIMIZATION -- COMPLETE")
        print("="*80)

        print(f"\n  Total runs executed: {sum(len(v) for v in all_results.values())}")
        print(f"  Best Task 1 config: T{best_thresh}/RR{best_rr}")

        if t2_results:
            best_tp1 = max(t2_results, key=lambda r: r["expectancy"])
            print(f"  Best Task 2 config: {best_tp1['label']}")

        print(f"\n  [PASS] SUCCESS CRITERIA CHECK:")
        # Check against best overall config
        all_runs = t1_results + t2_results
        best_overall = max(all_runs, key=lambda r: (
            r["trades"] >= 40,  # bool: meets trade count
            r["pf"] >= 2.0,     # bool: meets PF
            r["expectancy"],    # tiebreak: highest expectancy
        ))
        print(f"    Trade count >= 40/year:  {best_overall['trades']} {'[PASS]' if best_overall['trades'] >= 40 else '[FAIL]'}")
        print(f"    Expectancy > $20/trade: ${best_overall['expectancy']} {'[PASS]' if best_overall['expectancy'] > 20 else '[FAIL]'}")
        print(f"    PF > 2.0:               {best_overall['pf']} {'[PASS]' if best_overall['pf'] > 2.0 else '[FAIL]'}")
        print(f"    Max DD < 3%:            {best_overall['max_dd']}% {'[PASS]' if best_overall['max_dd'] < 3.0 else '[FAIL]'}")
        print(f"    Best config: {best_overall['label']}")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
