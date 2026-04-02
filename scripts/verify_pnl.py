"""Verify that BE/TP1 PnL matches trailing-stop exit model (not breakeven)."""
import sys; sys.path.insert(0, '.')
import os
os.environ['BACKTEST_DB_PASSWORD'] = 'Schlittlebah44!'
os.environ['BACKTEST_DB_USER'] = 'bulldog'
os.environ['BACKTEST_DB_NAME'] = 'first_db_local'
from backtest.db import get_connection
from backtest.config import FEE_PCT, EXECUTION_SLIPPAGE_PCT

with get_connection() as conn:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT entry_price, exit_price, tp1_price, target_price,
                   position_size, pnl_dollars, direction
            FROM backtest_trades
            WHERE run_id = 29 AND exit_reason = 'breakeven_after_tp1'
            ORDER BY pnl_dollars DESC LIMIT 5
        """)
        print("Verifying top 5 BE/TP1 trades: does PnL match trailing-stop exit?")
        print("=" * 70)
        for r in cur.fetchall():
            entry = float(r[0])
            exit_p = float(r[1])
            tp1 = float(r[2])
            target = float(r[3])
            pos = float(r[4])
            actual_pnl = float(r[5])
            direction = r[6]

            half = pos * 0.5
            # Apply slippage to entry and exits
            if direction == 'bullish':
                eff_entry = entry * (1 + EXECUTION_SLIPPAGE_PCT)
                eff_tp1 = tp1 * (1 - EXECUTION_SLIPPAGE_PCT)
                eff_exit = exit_p * (1 - EXECUTION_SLIPPAGE_PCT)
                tp1_pnl = half * (eff_tp1 - eff_entry) / eff_entry
                remain_pnl = half * (eff_exit - eff_entry) / eff_entry
            else:
                eff_entry = entry * (1 - EXECUTION_SLIPPAGE_PCT)
                eff_tp1 = tp1 * (1 + EXECUTION_SLIPPAGE_PCT)
                eff_exit = exit_p * (1 + EXECUTION_SLIPPAGE_PCT)
                tp1_pnl = half * (eff_entry - eff_tp1) / eff_entry
                remain_pnl = half * (eff_entry - eff_exit) / eff_entry

            tp1_fee = half * FEE_PCT
            remain_fee = half * FEE_PCT
            entry_fee = pos * FEE_PCT
            total = (tp1_pnl - tp1_fee) + (remain_pnl - remain_fee) - entry_fee

            # Also compute what PnL would be if remaining exited at breakeven
            if direction == 'bullish':
                be_remain_pnl = half * (eff_entry - eff_entry) / eff_entry  # = 0
            else:
                be_remain_pnl = half * (eff_entry - eff_entry) / eff_entry  # = 0
            be_total = (tp1_pnl - tp1_fee) + (be_remain_pnl - remain_fee) - entry_fee

            print(f"{direction:>8} entry=${entry:>10,.2f}  exit=${exit_p:>10,.2f}  tp1=${tp1:>10,.2f}")
            print(f"         Actual PnL:           ${actual_pnl:>10,.2f}")
            print(f"         Calc (trailing stop):  ${total:>10,.2f}  diff=${abs(total - actual_pnl):.2f}")
            print(f"         Calc (true breakeven): ${be_total:>10,.2f}")
            print(f"         Inflation:             ${actual_pnl - be_total:>10,.2f}")
            print()
