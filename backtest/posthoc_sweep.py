"""
backtest/posthoc_sweep.py -- Post-Hoc Parameter Sweep Analyzer
================================================================
Uses existing Run #14 signal data (21,683 signals) to simulate different
filter configurations WITHOUT re-running detection. Covers Tasks 1, 3, 4, 5.

For Task 2 (TP1 optimization), runs a single detection pass then replays
trade management with different TP1 parameters.

This is ~1000x faster than re-running full backtests.
"""

import logging
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from backtest.config import (
    DEFAULT_LEVERAGE,
    EXECUTION_SLIPPAGE_PCT,
    FEE_PCT,
    RISK_PER_TRADE_PCT,
    STARTING_BALANCE,
)
from backtest.db import get_connection
from backtest.ingest import load_candles
from backtest.reporter import (
    load_run,
    load_trades,
    load_signals,
)

logger = logging.getLogger("backtest.posthoc")

# Source run with full signal log
SOURCE_RUN_ID = 14
SYMBOL = "BTCUSDT"
START_DATE = datetime(2025, 4, 1, tzinfo=timezone.utc)
END_DATE = datetime(2026, 3, 26, tzinfo=timezone.utc)


# ==============================================================
# TASK 1: Post-Hoc Threshold x MIN_RR Sweep
# ==============================================================

def task1_posthoc_sweep(conn, signals_df):
    """
    For each (threshold, min_rr) combo, filter signals to TAKE,
    then forward-simulate trades on 1h candle data.
    """
    print("\n" + "="*80)
    print("  TASK 1 -- TRADE FREQUENCY EXPANSION (POST-HOC SWEEP)")
    print("="*80)

    # Load 1h candles for trade simulation
    candles_1h = load_candles(conn, SYMBOL, "1h", START_DATE, END_DATE)
    candles_1h = candles_1h.sort_values("open_time").reset_index(drop=True)

    # Only consider signals with score > 0 (structural detections)
    real_signals = signals_df[signals_df["score_1d"] > 0].copy()
    print(f"\n  Meaningful signals (score > 0): {len(real_signals)}")
    print(f"  Score range: {real_signals['score_1d'].min()} - {real_signals['score_1d'].max()}")

    thresholds = [35, 40, 45, 48, 50, 55]
    min_rrs = [0.3, 0.4, 0.5, 0.6]

    results = []
    for thresh in thresholds:
        for rr in min_rrs:
            # Filter: score >= threshold, rr >= min_rr, NOT rig-blocked
            eligible = real_signals[
                (real_signals["score_1d"] >= thresh) &
                (real_signals["rr"] >= rr) &
                (real_signals["rig_status"] != "BLOCK")
            ]

            if eligible.empty:
                results.append({
                    "label": f"T{thresh}/RR{rr}",
                    "thresh": thresh,
                    "min_rr": rr,
                    "eligible": 0,
                    "trades": 0,
                    "wins": 0,
                    "losses": 0,
                    "wr": 0,
                    "pf": 0,
                    "expectancy": 0,
                    "total_pnl": 0,
                    "max_dd": 0,
                })
                continue

            # Simulate trades sequentially (one at a time, no overlap)
            sim = _simulate_trades(eligible, candles_1h, STARTING_BALANCE)
            results.append({
                "label": f"T{thresh}/RR{rr}",
                "thresh": thresh,
                "min_rr": rr,
                "eligible": len(eligible),
                **sim,
            })

    # Print table
    print(f"\n  {'Config':<14} {'Elig':>5} {'Trades':>6} {'W/L':>7} {'WR%':>6} {'PF':>6} "
          f"{'E[$]':>7} {'PnL$':>8} {'DD%':>6}")
    print(f"  {'-'*75}")

    for r in results:
        wl = f"{r['wins']}/{r['losses']}" if r['trades'] > 0 else "-"
        print(
            f"  {r['label']:<14} {r['eligible']:>5} {r['trades']:>6} {wl:>7} "
            f"{r['wr']:>5.1f}% {r['pf']:>5.2f} {r['expectancy']:>6.1f} "
            f"${r['total_pnl']:>7.0f} {r['max_dd']:>5.2f}%"
        )

    # Pareto analysis
    print(f"\n  PARETO ANALYSIS:")
    baseline = next((r for r in results if r["label"] == "T50/RR0.5"), None)
    if baseline:
        print(f"  Baseline T50/RR0.5: {baseline['trades']}T, PF={baseline['pf']}, E=${baseline['expectancy']}")

    pareto = [
        r for r in results
        if r["trades"] > 0
        and r["pf"] >= 2.0
        and r["expectancy"] >= 20
        and r["max_dd"] < 3.0
    ]
    pareto.sort(key=lambda r: (r["trades"], r["pf"]), reverse=True)

    if pareto:
        print(f"\n  [PASS] Pareto-optimal configs (PF>=2, E>=$20, DD<3%):")
        for r in pareto[:5]:
            print(f"    {r['label']}: {r['trades']}T, PF={r['pf']:.2f}, E=${r['expectancy']:.1f}, DD={r['max_dd']:.2f}%")
        best = pareto[0]
        print(f"\n  >> RECOMMENDED: {best['label']} ({best['trades']} trades, PF={best['pf']:.2f})")
    else:
        print(f"\n  [WARN] No config meets all criteria. Best available:")
        by_trades = sorted([r for r in results if r["trades"] > 0], key=lambda r: r["trades"], reverse=True)
        for r in by_trades[:3]:
            print(f"    {r['label']}: {r['trades']}T, PF={r['pf']:.2f}, E=${r['expectancy']:.1f}, DD={r['max_dd']:.2f}%")

    # KEY INSIGHT
    print(f"\n  KEY INSIGHT:")
    total_nonzero = len(real_signals)
    non_rig_high = len(real_signals[
        (real_signals["score_1d"] >= 50) & (real_signals["rig_status"] != "BLOCK")
    ])
    print(f"    Total non-zero signals: {total_nonzero}")
    print(f"    Non-RIG score>=50: {non_rig_high}")
    print(f"    The detection engine produces ~{total_nonzero} meaningful signals/year.")
    print(f"    Trade frequency is DETECTION-BOUNDED, not threshold-bounded.")
    if non_rig_high < 40:
        print(f"    To reach 40+ trades/year, you need MORE SCHEMATICS, not looser filters.")

    return results


def _simulate_trades(eligible_signals, candles_1h, starting_balance,
                     tp1_close_pct=0.50, tp1_level_pct=0.50, trail_factor=0.50):
    """
    Forward-simulate trades from eligible signals using 1h candles.
    Applies TP1 partial close, BE stop, trailing stop.
    Returns dict with trade metrics.
    """
    equity = starting_balance
    peak_equity = starting_balance
    max_dd_pct = 0.0
    trades = []
    last_close_time = None

    for _, sig in eligible_signals.iterrows():
        sig_time = pd.to_datetime(sig["signal_time"])
        if sig_time.tzinfo is None:
            sig_time = sig_time.tz_localize("UTC")

        # Cooldown: skip if too close to last trade close
        if last_close_time is not None and sig_time <= last_close_time:
            continue

        entry_price = sig.get("entry_price", 0)
        stop_price = sig.get("stop_price", 0)
        target_price = sig.get("target_price", 0)
        direction = sig.get("direction", "")

        if not entry_price or not stop_price or not target_price:
            continue

        # Apply slippage
        if direction == "bullish":
            eff_entry = entry_price * (1 + EXECUTION_SLIPPAGE_PCT)
        else:
            eff_entry = entry_price * (1 - EXECUTION_SLIPPAGE_PCT)

        # Position sizing
        if direction == "bullish":
            sl_pct = abs(eff_entry - stop_price) / eff_entry * 100
        else:
            sl_pct = abs(stop_price - eff_entry) / eff_entry * 100

        if sl_pct <= 0:
            continue

        risk_amount = equity * (RISK_PER_TRADE_PCT / 100)
        position_size = (risk_amount / sl_pct) * 100

        # TP1 price
        tp1_price = eff_entry + (target_price - eff_entry) * tp1_level_pct

        # Forward simulate on 1h candles
        future = candles_1h[candles_1h["open_time"] > sig_time].head(500)
        if future.empty:
            continue

        tp1_hit = False
        realized_pnl = 0.0
        remaining_size = position_size
        current_stop = stop_price
        original_stop = stop_price
        highest = 0.0
        lowest = float('inf')
        exit_price = None
        exit_reason = None
        close_time = None

        for _, candle in future.iterrows():
            h = float(candle["high"])
            l = float(candle["low"])

            # Check TP1
            if not tp1_hit:
                tp1_triggered = (h >= tp1_price) if direction == "bullish" else (l <= tp1_price)
                if tp1_triggered:
                    partial_size = position_size * tp1_close_pct
                    if direction == "bullish":
                        partial_pnl = (tp1_price - eff_entry) * (partial_size / eff_entry)
                    else:
                        partial_pnl = (eff_entry - tp1_price) * (partial_size / eff_entry)
                    partial_fee = partial_size * FEE_PCT
                    realized_pnl += (partial_pnl - partial_fee)
                    remaining_size = position_size - partial_size
                    tp1_hit = True
                    current_stop = eff_entry  # BE
                    highest = h
                    lowest = l

            # Update trailing stop
            if tp1_hit:
                trail_dist = abs(target_price - eff_entry) * trail_factor
                if direction == "bullish":
                    highest = max(highest, h)
                    trail_stop = highest - trail_dist
                    current_stop = max(current_stop, trail_stop)
                else:
                    lowest = min(lowest, l)
                    trail_stop = lowest + trail_dist
                    current_stop = min(current_stop, trail_stop)

            # Check SL/TP
            if direction == "bullish":
                sl_hit = l <= current_stop
                tp_hit = h >= target_price
            else:
                sl_hit = h >= current_stop
                tp_hit = l <= target_price

            if sl_hit and tp_hit:
                exit_price = current_stop
                exit_reason = "be_after_tp1" if tp1_hit else "stop_hit"
                close_time = candle["open_time"]
                break
            if sl_hit:
                exit_price = current_stop
                if tp1_hit:
                    exit_reason = "be_after_tp1" if abs(current_stop - eff_entry) < 1 else "trailing_stop"
                else:
                    exit_reason = "stop_hit"
                close_time = candle["open_time"]
                break
            if tp_hit:
                exit_price = target_price
                exit_reason = "target_hit"
                close_time = candle["open_time"]
                break

        if exit_price is None:
            continue  # Never resolved within 500 candles

        # Apply slippage to exit
        if direction == "bullish":
            eff_exit = exit_price * (1 - EXECUTION_SLIPPAGE_PCT)
        else:
            eff_exit = exit_price * (1 + EXECUTION_SLIPPAGE_PCT)

        # P&L on remaining position
        if direction == "bullish":
            remaining_pnl = (eff_exit - eff_entry) * (remaining_size / eff_entry)
        else:
            remaining_pnl = (eff_entry - eff_exit) * (remaining_size / eff_entry)

        remaining_fee = remaining_size * FEE_PCT
        entry_fee = position_size * FEE_PCT
        total_pnl = realized_pnl + remaining_pnl - remaining_fee - entry_fee

        is_win = total_pnl > 0
        equity += total_pnl

        if equity > peak_equity:
            peak_equity = equity
        dd = ((peak_equity - equity) / peak_equity) * 100 if peak_equity > 0 else 0
        max_dd_pct = max(max_dd_pct, dd)

        last_close_time = close_time

        trades.append({
            "direction": direction,
            "entry": eff_entry,
            "exit": eff_exit,
            "exit_reason": exit_reason,
            "tp1_hit": tp1_hit,
            "pnl": round(total_pnl, 2),
            "is_win": is_win,
            "balance": round(equity, 2),
        })

    # Compute metrics
    total = len(trades)
    wins = sum(1 for t in trades if t["is_win"])
    losses = total - wins
    win_rate = (wins / total * 100) if total > 0 else 0

    winning = [t["pnl"] for t in trades if t["is_win"]]
    losing = [abs(t["pnl"]) for t in trades if not t["is_win"]]

    avg_win = np.mean(winning) if winning else 0
    avg_loss = np.mean(losing) if losing else 0
    gross_profit = sum(winning)
    gross_loss = sum(losing)
    pf = (gross_profit / gross_loss) if gross_loss > 0 else float('inf')
    total_pnl = sum(t["pnl"] for t in trades)

    wr_frac = wins / total if total > 0 else 0
    lr_frac = losses / total if total > 0 else 0
    expectancy = (wr_frac * avg_win) - (lr_frac * avg_loss)

    return {
        "trades": total,
        "wins": wins,
        "losses": losses,
        "wr": win_rate,
        "pf": round(pf, 3),
        "expectancy": round(expectancy, 2),
        "total_pnl": round(total_pnl, 2),
        "max_dd": round(max_dd_pct, 2),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "trade_details": trades,
    }


# ==============================================================
# TASK 2: TP1 Optimization (Post-Hoc Replay)
# ==============================================================

def task2_tp1_optimization(conn, signals_df, best_thresh=50, best_rr=0.5):
    """
    Replay trades with different TP1 split and TP1 level parameters.
    Uses the same eligible signals from the best Task 1 config.
    """
    print("\n" + "="*80)
    print("  TASK 2 -- TP1 OPTIMIZATION CURVE")
    print("="*80)

    candles_1h = load_candles(conn, SYMBOL, "1h", START_DATE, END_DATE)
    candles_1h = candles_1h.sort_values("open_time").reset_index(drop=True)

    real_signals = signals_df[signals_df["score_1d"] > 0].copy()
    eligible = real_signals[
        (real_signals["score_1d"] >= best_thresh) &
        (real_signals["rr"] >= best_rr) &
        (real_signals["rig_status"] != "BLOCK")
    ]
    print(f"\n  Using config T{best_thresh}/RR{best_rr}: {len(eligible)} eligible signals")

    tp1_splits = [0.30, 0.40, 0.50, 0.60, 0.70]
    tp1_levels = [0.30, 0.40, 0.50, 0.60, 0.70]
    results = []

    for split in tp1_splits:
        for level in tp1_levels:
            label = f"TP1_{int(split*100)}%@{level:.1f}R"
            sim = _simulate_trades(
                eligible, candles_1h, STARTING_BALANCE,
                tp1_close_pct=split,
                tp1_level_pct=level,
            )
            sim["label"] = label
            sim["split"] = split
            sim["level"] = level
            wl_ratio = round(sim["avg_win"] / sim["avg_loss"], 2) if sim["avg_loss"] > 0 else 0
            sim["wl_ratio"] = wl_ratio
            results.append(sim)

    # Print table
    print(f"\n  {'Config':<18} {'Trades':>6} {'W/L':>7} {'WR%':>6} {'PF':>6} "
          f"{'E[$]':>7} {'PnL$':>8} {'W/L$':>6} {'DD%':>6}")
    print(f"  {'-'*80}")

    for r in results:
        wl = f"{r['wins']}/{r['losses']}" if r['trades'] > 0 else "-"
        print(
            f"  {r['label']:<18} {r['trades']:>6} {wl:>7} "
            f"{r['wr']:>5.1f}% {r['pf']:>5.2f} {r['expectancy']:>6.1f} "
            f"${r['total_pnl']:>7.0f} {r['wl_ratio']:>5.2f} {r['max_dd']:>5.2f}%"
        )

    # Rank by expectancy
    ranked = sorted(results, key=lambda r: (r["expectancy"], r["pf"], r["wl_ratio"]), reverse=True)
    print(f"\n  TOP 5 CONFIGS (ranked by Expectancy -> PF -> W/L Ratio):")
    for i, r in enumerate(ranked[:5], 1):
        flag = " <- RECOMMENDED" if i == 1 else ""
        print(
            f"    #{i} {r['label']}: E=${r['expectancy']:.1f}, PF={r['pf']:.2f}, "
            f"W/L={r['wl_ratio']:.2f}, WR={r['wr']:.1f}%{flag}"
        )

    best = ranked[0]
    print(f"\n  SUCCESS CHECK:")
    print(f"    W/L ratio >= 1.05: {best['wl_ratio']} {'[PASS]' if best['wl_ratio'] >= 1.05 else '[FAIL]'}")
    print(f"    WR >= 70%: {best['wr']}% {'[PASS]' if best['wr'] >= 70 else '[FAIL]'}")

    return results


# ==============================================================
# TASK 3: RIG Counterfactual Audit
# ==============================================================

def task3_rig_counterfactual(conn, signals_df):
    """
    Simulate what happens if RIG-blocked high-score signals were taken.
    """
    print("\n" + "="*80)
    print("  TASK 3 -- RIG EFFICIENCY AUDIT (COUNTERFACTUAL)")
    print("="*80)

    candles_1h = load_candles(conn, SYMBOL, "1h", START_DATE, END_DATE)
    candles_1h = candles_1h.sort_values("open_time").reset_index(drop=True)

    # RIG-blocked signals with high scores
    rig_blocked = signals_df[
        (signals_df["rig_status"] == "BLOCK") &
        (signals_df["score_1d"] >= 50)
    ].copy()

    total_blocked_high = len(rig_blocked)
    print(f"\n  RIG-blocked signals with score >= 50: {total_blocked_high}")

    if total_blocked_high == 0:
        # Also check lower thresholds
        rig_blocked_all = signals_df[
            (signals_df["rig_status"] == "BLOCK") &
            (signals_df["score_1d"] > 0)
        ]
        print(f"  RIG-blocked signals with score > 0: {len(rig_blocked_all)}")
        for score, count in rig_blocked_all["score_1d"].value_counts().sort_index().items():
            print(f"    Score {score}: {count} blocked")
        # Use all non-zero blocked signals
        rig_blocked = rig_blocked_all.copy()
        total_blocked_high = len(rig_blocked)

    if total_blocked_high == 0:
        print("  No blocks to analyze.")
        return {}

    # Simulate: what if we took ALL these blocked signals as trades?
    sim_with_rig_off = _simulate_trades(rig_blocked, candles_1h, STARTING_BALANCE)

    # Also simulate production (non-blocked trades)
    prod_eligible = signals_df[
        (signals_df["score_1d"] >= 50) &
        (signals_df["rr"] >= 0.5) &
        (signals_df["rig_status"] != "BLOCK")
    ]
    sim_production = _simulate_trades(prod_eligible, candles_1h, STARTING_BALANCE)

    print(f"\n  {'Scenario':<25} {'Trades':>6} {'W/L':>7} {'WR%':>6} {'PF':>6} {'E[$]':>7} {'PnL$':>8}")
    print(f"  {'-'*70}")

    for label, sim in [("Production (RIG ON)", sim_production), ("Blocked Only (RIG OFF)", sim_with_rig_off)]:
        wl = f"{sim['wins']}/{sim['losses']}" if sim['trades'] > 0 else "-"
        print(
            f"  {label:<25} {sim['trades']:>6} {wl:>7} "
            f"{sim['wr']:>5.1f}% {sim['pf']:>5.2f} {sim['expectancy']:>6.1f} "
            f"${sim['total_pnl']:>7.0f}"
        )

    # Detailed outcome analysis for blocked trades
    blocked_details = sim_with_rig_off.get("trade_details", [])
    if blocked_details:
        outcome_counts = Counter(t["exit_reason"] for t in blocked_details)
        loser_count = sum(1 for t in blocked_details if not t["is_win"])
        winner_count = sum(1 for t in blocked_details if t["is_win"])
        total_cf = len(blocked_details)

        print(f"\n  COUNTERFACTUAL OUTCOME BREAKDOWN:")
        for reason, count in outcome_counts.most_common():
            print(f"    {reason}: {count} ({count/total_cf*100:.1f}%)")

        pct_losers = (loser_count / total_cf * 100) if total_cf > 0 else 0
        print(f"\n  % blocked trades that would have LOST: {pct_losers:.1f}%")
        print(f"  % blocked trades that would have WON:  {100-pct_losers:.1f}%")

        if pct_losers > 50:
            print(f"\n  [PASS] RIG IS PROTECTIVE -- {pct_losers:.0f}% of blocked trades are losers")
            print(f"  Recommendation: KEEP RIG enabled. It is alpha-preserving.")
        elif pct_losers > 30:
            print(f"\n  [INFO] RIG is MODERATELY protective -- {pct_losers:.0f}% losers")
            print(f"  Recommendation: Keep RIG but consider relaxing conditions.")
        else:
            print(f"\n  [WARN] RIG MAY BE OVER-FILTERING -- only {pct_losers:.0f}% losers")
            print(f"  Recommendation: Consider disabling or significantly relaxing RIG.")
    else:
        print(f"\n  No blocked trades resolved within simulation window.")

    return {"production": sim_production, "blocked": sim_with_rig_off}


# ==============================================================
# TASK 4: Regime Expansion Check
# ==============================================================

def task4_regime_analysis(conn, signals_df):
    """Analyze signals and trades by HTF regime."""
    print("\n" + "="*80)
    print("  TASK 4 -- REGIME EXPANSION CHECK")
    print("="*80)

    real_signals = signals_df[signals_df["score_1d"] > 0].copy()
    trades_df = load_trades(conn, SOURCE_RUN_ID)

    # Group by regime (htf_bias)
    print(f"\n  Signal Distribution by HTF Regime:")
    print(f"  {'Regime':<15} {'Signals':>8} {'Score>0':>8} {'Taken':>7} {'RIG Block':>10}")
    print(f"  {'-'*55}")

    for regime in sorted(signals_df["gate_1a_bias"].dropna().unique()):
        reg_all = signals_df[signals_df["gate_1a_bias"] == regime]
        reg_real = real_signals[real_signals["gate_1a_bias"] == regime]
        reg_takes = reg_all[reg_all["final_decision"] == "TAKE"]
        reg_rig = reg_all[reg_all["rig_status"] == "BLOCK"]
        print(
            f"  {regime:<15} {len(reg_all):>8} {len(reg_real):>8} "
            f"{len(reg_takes):>7} {len(reg_rig):>10}"
        )

    # Trade performance by regime
    # Match each trade to its TAKE signal to get the signal's HTF regime (gate_1a_bias).
    if not trades_df.empty:
        print(f"\n  Trade Performance by HTF Regime:")

        take_signals = signals_df[signals_df["final_decision"] == "TAKE"].copy()
        take_signals["signal_time_dt"] = pd.to_datetime(take_signals["signal_time"])

        regime_trades = defaultdict(list)
        for _, trade in trades_df.iterrows():
            opened = pd.to_datetime(trade["opened_at"])
            if pd.isna(opened):
                continue
            # Find the TAKE signal within ±2h of trade open
            sig_match = take_signals[
                (take_signals["signal_time_dt"] >= opened - pd.Timedelta(hours=2)) &
                (take_signals["signal_time_dt"] <= opened + pd.Timedelta(hours=2))
            ]
            if not sig_match.empty:
                regime = sig_match.iloc[0].get("gate_1a_bias", "unknown")
            else:
                regime = "unknown"
            regime_trades[regime].append(trade)

        print(f"\n  {'Regime':<15} {'Trades':>7} {'Wins':>5} {'WR%':>6} {'PnL$':>8} {'Avg PnL':>8}")
        print(f"  {'-'*56}")
        for regime in sorted(regime_trades.keys()):
            tlist = regime_trades[regime]
            t_wins = sum(1 for t in tlist if t["is_win"])
            t_pnl = sum(t["pnl_dollars"] or 0 for t in tlist)
            t_wr = (t_wins / len(tlist) * 100) if tlist else 0
            t_avg = (t_pnl / len(tlist)) if tlist else 0
            print(
                f"  {regime:<15} {len(tlist):>7} {t_wins:>5} "
                f"{t_wr:>5.1f}% ${t_pnl:>7.0f} ${t_avg:>7.1f}"
            )

    # Regime concentration
    total_signals = len(signals_df)
    regime_pcts = signals_df["gate_1a_bias"].value_counts() / total_signals * 100
    print(f"\n  REGIME CONCENTRATION (all signals):")
    for regime, pct in regime_pcts.items():
        print(f"    {regime}: {pct:.1f}%")

    dominant = regime_pcts.idxmax()
    print(f"\n  Dominant regime: {dominant} ({regime_pcts.max():.1f}%)")
    if dominant in ("ranging", "neutral"):
        print(f"  [PASS] System is RANGE-SPECIALIZED (intentional)")
        print(f"  HPB schematics form during consolidation -- this is expected.")
    else:
        print(f"  [INFO] System operates primarily in {dominant} regime")

    return {}


# ==============================================================
# TASK 5: Score Band Analysis
# ==============================================================

def task5_score_band_analysis(conn, signals_df):
    """Analyze score distribution and bimodality."""
    print("\n" + "="*80)
    print("  TASK 5 -- SCORE BAND EXPANSION (BIMODALITY DIAGNOSIS)")
    print("="*80)

    real_signals = signals_df[signals_df["score_1d"] > 0].copy()
    all_scores = signals_df["score_1d"].dropna()

    print(f"\n  Total signals: {len(signals_df)}")
    print(f"  Signals with score > 0: {len(real_signals)}")
    print(f"  Signals with score = 0: {len(signals_df) - len(real_signals)} (no structural detection)")

    if real_signals.empty:
        print("  No meaningful scores to analyze.")
        return {}

    scores = real_signals["score_1d"]

    # Detailed score histogram
    print(f"\n  SCORE DISTRIBUTION (non-zero signals, n={len(scores)}):")
    print(f"  {'Score':>7} {'Count':>7} {'%':>7} {'Bar'}")
    print(f"  {'-'*60}")

    score_counts = scores.value_counts().sort_index()
    max_count = score_counts.max()
    for score_val, count in score_counts.items():
        pct = count / len(scores) * 100
        bar = "#" * int(count / max_count * 30) if max_count > 0 else ""
        print(f"  {score_val:>7.0f} {count:>7} {pct:>6.1f}% {bar}")

    # Bimodality analysis
    below_50 = scores[scores < 50]
    above_50 = scores[scores >= 50]
    gap_zone = scores[(scores > 40) & (scores < 57)]

    print(f"\n  CLUSTER ANALYSIS:")
    print(f"    Below 50:     {len(below_50)} signals (mean={below_50.mean():.1f})" if len(below_50) > 0 else "    Below 50: 0")
    print(f"    Above 50:     {len(above_50)} signals (mean={above_50.mean():.1f})" if len(above_50) > 0 else "    Above 50: 0")
    print(f"    Gap zone (41-56): {len(gap_zone)} signals")

    if len(gap_zone) == 0 and len(below_50) > 0 and len(above_50) > 0:
        print(f"\n  [WARN] HARD BIMODALITY: Zero signals in 41-56 range!")
        print(f"    Lower cluster: scores {below_50.min():.0f}-{below_50.max():.0f}")
        print(f"    Upper cluster: scores {above_50.min():.0f}-{above_50.max():.0f}")
        print(f"    Gap width: {above_50.min() - below_50.max():.0f} points")
    elif len(gap_zone) < 3:
        print(f"\n  [WARN] NEAR-BIMODALITY: Only {len(gap_zone)} signals in gap zone")
    else:
        print(f"\n  [PASS] Score distribution is reasonably continuous")

    # Failure code analysis for mid-range signals
    print(f"\n  FAILURE ANALYSIS (signals with score 30-55):")
    mid_signals = real_signals[(real_signals["score_1d"] >= 30) & (real_signals["score_1d"] <= 55)]
    if not mid_signals.empty:
        failure_dist = mid_signals["failure_code"].value_counts()
        for code, count in failure_dist.items():
            if pd.notna(code):
                print(f"    {code}: {count}")
        no_failure = mid_signals["failure_code"].isna().sum()
        print(f"    (no failure / TAKE): {no_failure}")
    else:
        print(f"    No signals in range 30-55")

    # RIG impact on mid-range
    print(f"\n  RIG IMPACT ON NEAR-THRESHOLD SIGNALS:")
    near_thresh = real_signals[(real_signals["score_1d"] >= 45) & (real_signals["score_1d"] <= 65)]
    if not near_thresh.empty:
        rig_blocked = near_thresh[near_thresh["rig_status"] == "BLOCK"]
        print(f"    Score 45-65: {len(near_thresh)} signals, {len(rig_blocked)} RIG-blocked ({len(rig_blocked)/len(near_thresh)*100:.0f}%)")

    # Component contribution analysis
    print(f"\n  SCORE JUMP DIAGNOSIS:")
    print(f"    The scoring system uses DecisionTreeEvaluator.")
    print(f"    Score jumps from ~{below_50.max():.0f} to ~{above_50.min():.0f}" if len(below_50) > 0 and len(above_50) > 0 else "")
    print(f"    This suggests a BINARY GATE in the decision tree that adds ~{above_50.min() - below_50.max():.0f} points." if len(below_50) > 0 and len(above_50) > 0 else "")
    print(f"    To diagnose further: inspect DecisionTreeEvaluator.evaluate_schematic()")
    print(f"    and log individual component contributions.")

    return {}


# ==============================================================
# Main
# ==============================================================

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    conn = get_connection()

    # Load Run #14 signals (pre-computed)
    print(f"Loading Run #{SOURCE_RUN_ID} signals...")
    signals_df = load_signals(conn, SOURCE_RUN_ID)
    print(f"Loaded {len(signals_df)} signals")

    try:
        # TASK 1
        t1_results = task1_posthoc_sweep(conn, signals_df)

        # Find best Task 1 config
        pareto = [r for r in t1_results if r["pf"] >= 2.0 and r["expectancy"] >= 20 and r.get("trades", 0) > 0]
        if pareto:
            best_t1 = max(pareto, key=lambda r: (r["trades"], r["pf"]))
            best_thresh = best_t1["thresh"]
            best_rr = best_t1["min_rr"]
        else:
            best_thresh, best_rr = 50, 0.5

        # TASK 2
        t2_results = task2_tp1_optimization(conn, signals_df, best_thresh, best_rr)

        # TASK 3
        task3_rig_counterfactual(conn, signals_df)

        # TASK 4
        task4_regime_analysis(conn, signals_df)

        # TASK 5
        task5_score_band_analysis(conn, signals_df)

        # FINAL SUMMARY
        print("\n" + "="*80)
        print("  PHASE 2 OPTIMIZATION -- COMPLETE")
        print("="*80)

        print(f"\n  SUCCESS CRITERIA CHECK:")
        all_configs = t1_results
        best_overall = max(
            [r for r in all_configs if r.get("trades", 0) > 0],
            key=lambda r: (r["trades"] >= 40, r["pf"] >= 2.0, r["expectancy"]),
            default=None,
        )
        if best_overall:
            print(f"    Best config: {best_overall['label']}")
            print(f"    Trade count >= 40/year:  {best_overall['trades']} {'[PASS]' if best_overall['trades'] >= 40 else '[FAIL]'}")
            print(f"    Expectancy > $20/trade:  ${best_overall['expectancy']} {'[PASS]' if best_overall['expectancy'] > 20 else '[FAIL]'}")
            print(f"    PF > 2.0:                {best_overall['pf']} {'[PASS]' if best_overall['pf'] > 2.0 else '[FAIL]'}")
            print(f"    Max DD < 3%:             {best_overall['max_dd']}% {'[PASS]' if best_overall['max_dd'] < 3.0 else '[FAIL]'}")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
