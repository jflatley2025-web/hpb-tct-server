"""
backtest/threshold_sweep.py -- Post-hoc Threshold Sweep Simulator
=================================================================
Uses logged signals from a completed backtest run to simulate different
entry thresholds WITHOUT re-running the full engine. Accounts for trade
collision, slippage, fees, and intra-candle conflict resolution.
"""

import logging
import pandas as pd
from datetime import timedelta
from typing import Dict, List, Optional

from backtest.config import (
    EXECUTION_SLIPPAGE_PCT, FEE_PCT, DEFAULT_LEVERAGE,
    RISK_PER_TRADE_PCT, STARTING_BALANCE, MIN_BARS_BETWEEN_TRADES,
)
from backtest.db import get_connection
from backtest.ingest import load_candles

logger = logging.getLogger("backtest.threshold_sweep")


def simulate_threshold(
    run_id: int,
    threshold: int,
    conn=None,
    starting_balance: float = STARTING_BALANCE,
) -> dict:
    """
    Simulate a backtest run as if ENTRY_THRESHOLD was set to `threshold`.

    Uses the signal log from an existing run and 1m candles for outcome
    simulation. Properly handles:
    - Trade collision (one trade at a time)
    - Slippage on entry/exit
    - Fees on entry/exit
    - Intra-candle conflict (SL first on ambiguous candles)
    - Cooldown between trades
    """
    own_conn = conn is None
    if own_conn:
        conn = get_connection()

    try:
        from psycopg2.extras import RealDictCursor
        cur = conn.cursor(cursor_factory=RealDictCursor)

        # Load all signals that would qualify at this threshold
        # Include signals that passed all gates EXCEPT possibly score
        cur.execute("""
            SELECT *
            FROM backtest_signals
            WHERE run_id = %s
              AND score_1d >= %s
              AND gate_1a_pass = TRUE
              AND (failure_code IS NULL
                   OR failure_code = 'FAIL_1D_SCORE'
                   OR final_decision = 'TAKE')
            ORDER BY signal_time
        """, (run_id, threshold))
        signals = [dict(r) for r in cur.fetchall()]

        # Derive symbol from the run record rather than hardcoding BTCUSDT
        cur.execute(
            "SELECT config_json FROM backtest_runs WHERE id = %s", (run_id,)
        )
        run_row = cur.fetchone()
        run_symbol = "BTCUSDT"
        if run_row and run_row.get("config_json"):
            try:
                cfg = run_row["config_json"]
                if isinstance(cfg, str):
                    import json as _json
                    cfg = _json.loads(cfg)
                run_symbol = cfg.get("symbol", "BTCUSDT")
            except Exception:
                pass

        # Load 1m candles for outcome simulation
        candles_1m = load_candles(conn, run_symbol, '1m')
        candles_1m = candles_1m.sort_values('open_time').reset_index(drop=True)

        # Walk through signals simulating trades
        equity = starting_balance
        peak_equity = starting_balance
        max_drawdown_pct = 0.0
        trades = []
        in_trade = False
        trade_close_time = None

        for sig in signals:
            if in_trade:
                continue

            # Trade cooldown
            if trade_close_time is not None:
                cooldown = timedelta(hours=MIN_BARS_BETWEEN_TRADES)
                if sig['signal_time'] < trade_close_time + cooldown:
                    continue

            entry_price = sig.get('entry_price')
            stop_price = sig.get('stop_price')
            target_price = sig.get('target_price')
            direction = sig.get('direction')
            sig_time = sig['signal_time']

            if not all([entry_price, stop_price, target_price, direction]):
                continue

            # Apply entry slippage
            if direction == 'bullish':
                effective_entry = entry_price * (1 + EXECUTION_SLIPPAGE_PCT)
                sl_pct = abs(effective_entry - stop_price) / effective_entry * 100
            else:
                effective_entry = entry_price * (1 - EXECUTION_SLIPPAGE_PCT)
                sl_pct = abs(stop_price - effective_entry) / effective_entry * 100

            if sl_pct <= 0:
                continue

            # Position sizing
            risk_amount = equity * (RISK_PER_TRADE_PCT / 100)
            position_size = (risk_amount / sl_pct) * 100

            # Simulate outcome with 1m candles
            future = candles_1m[candles_1m['open_time'] > sig_time]
            max_hold_candles = 72 * 60  # 72 hours
            future = future.head(max_hold_candles)

            outcome = None
            exit_price = None
            exit_time = None
            mfe = 0.0
            mae = 0.0

            for _, c in future.iterrows():
                # Track MFE/MAE
                if direction == 'bullish':
                    mfe = max(mfe, c['high'] - effective_entry)
                    mae = min(mae, c['low'] - effective_entry)

                    # Intra-candle conflict: check SL first (worst case)
                    if c['low'] <= stop_price:
                        outcome = 'LOSS'
                        exit_price = stop_price
                        exit_time = c['open_time']
                        break
                    if c['high'] >= target_price:
                        outcome = 'WIN'
                        exit_price = target_price
                        exit_time = c['open_time']
                        break
                else:
                    mfe = max(mfe, effective_entry - c['low'])
                    mae = min(mae, effective_entry - c['high'])

                    if c['high'] >= stop_price:
                        outcome = 'LOSS'
                        exit_price = stop_price
                        exit_time = c['open_time']
                        break
                    if c['low'] <= target_price:
                        outcome = 'WIN'
                        exit_price = target_price
                        exit_time = c['open_time']
                        break

            if outcome is None:
                # Timeout -- close at last available price
                if not future.empty:
                    last_candle = future.iloc[-1]
                    exit_price = last_candle['close']
                    exit_time = last_candle['open_time']
                    if direction == 'bullish':
                        outcome = 'WIN' if exit_price > effective_entry else 'LOSS'
                    else:
                        outcome = 'WIN' if exit_price < effective_entry else 'LOSS'
                else:
                    continue

            # Apply exit slippage
            if direction == 'bullish':
                effective_exit = exit_price * (1 - EXECUTION_SLIPPAGE_PCT)
            else:
                effective_exit = exit_price * (1 + EXECUTION_SLIPPAGE_PCT)

            # Calculate P&L with fees
            if direction == 'bullish':
                raw_pnl_pct = (effective_exit - effective_entry) / effective_entry * 100
            else:
                raw_pnl_pct = (effective_entry - effective_exit) / effective_entry * 100

            # Apply fees (entry + exit)
            total_fee_pct = FEE_PCT * 2 * 100  # both sides, as percentage
            pnl_pct = raw_pnl_pct - total_fee_pct

            pnl_dollars = position_size * (pnl_pct / 100)
            equity += pnl_dollars
            peak_equity = max(peak_equity, equity)
            dd = (peak_equity - equity) / peak_equity * 100 if peak_equity > 0 else 0
            max_drawdown_pct = max(max_drawdown_pct, dd)

            trade_close_time = exit_time
            is_win = pnl_dollars > 0

            trades.append({
                'signal_time': sig_time,
                'score': sig['score_1d'],
                'direction': direction,
                'session': sig.get('msce_session', 'unknown'),
                'entry': effective_entry,
                'exit': effective_exit,
                'outcome': outcome,
                'is_win': is_win,
                'pnl_pct': round(pnl_pct, 4),
                'pnl_dollars': round(pnl_dollars, 2),
                'rr': sig.get('rr', 0),
                'mfe': round(mfe, 2),
                'mae': round(mae, 2),
                'balance_after': round(equity, 2),
                'hold_time': str(exit_time - sig_time) if exit_time else 'N/A',
            })

        # Compute summary metrics
        total = len(trades)
        wins = sum(1 for t in trades if t['is_win'])
        losses = total - wins
        win_rate = (wins / total * 100) if total > 0 else 0

        winning = [t for t in trades if t['is_win']]
        losing = [t for t in trades if not t['is_win']]

        gross_profit = sum(t['pnl_dollars'] for t in winning) if winning else 0
        gross_loss = abs(sum(t['pnl_dollars'] for t in losing)) if losing else 0
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float('inf')

        avg_win = (sum(t['pnl_dollars'] for t in winning) / len(winning)) if winning else 0
        avg_loss = (abs(sum(t['pnl_dollars'] for t in losing)) / len(losing)) if losing else 0

        expectancy = (win_rate/100 * avg_win) - ((100-win_rate)/100 * avg_loss)

        total_pnl = sum(t['pnl_dollars'] for t in trades)
        total_pnl_pct = ((equity - starting_balance) / starting_balance) * 100

        return {
            'threshold': threshold,
            'total_trades': total,
            'wins': wins,
            'losses': losses,
            'win_rate': round(win_rate, 1),
            'profit_factor': round(profit_factor, 3),
            'expectancy': round(expectancy, 2),
            'total_pnl': round(total_pnl, 2),
            'total_pnl_pct': round(total_pnl_pct, 2),
            'max_drawdown_pct': round(max_drawdown_pct, 2),
            'final_balance': round(equity, 2),
            'trades': trades,
        }

    finally:
        if own_conn:
            conn.close()


def run_sweep(
    run_id: int,
    thresholds: List[int] = None,
    conn=None,
) -> List[dict]:
    """Run threshold sweep across multiple values."""
    if thresholds is None:
        thresholds = [30, 37, 40, 45, 50, 52, 55, 57, 58, 60, 62, 65, 80]

    own_conn = conn is None
    if own_conn:
        conn = get_connection()

    results = []
    try:
        for thr in thresholds:
            result = simulate_threshold(run_id, thr, conn=conn)
            results.append(result)
    finally:
        if own_conn:
            conn.close()

    return results


def print_sweep_table(results: List[dict]):
    """Print threshold sweep results as a formatted table."""
    header = '{:>5} {:>6} {:>4} {:>4} {:>5} {:>7} {:>9} {:>8} {:>8} {:>7}'.format(
        'Thr', 'Trades', 'W', 'L', 'WR%', 'PF', 'Expect$', 'PnL$', 'PnL%', 'MaxDD%')
    print(header)
    print('-' * len(header))
    for r in results:
        pf_str = '{:.3f}'.format(r['profit_factor']) if r['profit_factor'] != float('inf') else 'inf'
        print('{:>5} {:>6} {:>4} {:>4} {:>4.1f}% {:>7} {:>8.2f} {:>7.2f} {:>7.2f}% {:>6.2f}%'.format(
            r['threshold'], r['total_trades'], r['wins'], r['losses'],
            r['win_rate'], pf_str, r['expectancy'],
            r['total_pnl'], r['total_pnl_pct'], r['max_drawdown_pct']))


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Threshold Sweep Simulator")
    parser.add_argument("run_id", type=int, help="Backtest run ID to sweep")
    parser.add_argument("--thresholds", nargs="+", type=int,
                        default=[30, 37, 40, 45, 50, 52, 55, 57, 58, 60, 62, 65])
    args = parser.parse_args()

    results = run_sweep(args.run_id, args.thresholds)
    print()
    print_sweep_table(results)

    # Find optimal
    valid = [r for r in results if r['total_trades'] >= 3 and r['expectancy'] > 0]
    if valid:
        best = max(valid, key=lambda x: x['expectancy'] * min(x['total_trades'], 20))
        print()
        print('=== RECOMMENDED THRESHOLD: {} ==='.format(best['threshold']))
        print('  Trades: {}, WR: {:.1f}%, PF: {:.3f}, Expect: ${:.2f}, MaxDD: {:.2f}%'.format(
            best['total_trades'], best['win_rate'], best['profit_factor'],
            best['expectancy'], best['max_drawdown_pct']))
