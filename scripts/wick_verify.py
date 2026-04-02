"""Verify what exit_price and PnL actually represent for BE/TP1 trades."""
import os
os.environ['BACKTEST_DB_PASSWORD'] = 'Schlittlebah44!'
os.environ['BACKTEST_DB_USER'] = 'bulldog'
os.environ['BACKTEST_DB_NAME'] = 'first_db_local'
from backtest.db import get_connection

with get_connection() as conn:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT entry_price, tp1_price, exit_price, position_size,
                   pnl_dollars, direction, stop_price, target_price
            FROM backtest_trades
            WHERE run_id = 29 AND exit_reason = 'breakeven_after_tp1'
            ORDER BY pnl_dollars DESC LIMIT 10
        """)
        print("Top 10 BE/TP1 trades by PnL:")
        print(f"{'Entry':>12} {'TP1':>12} {'Exit':>12} {'PnL':>10} {'Dir':<8}")
        print("-" * 60)
        for r in cur.fetchall():
            entry, tp1, exit_p = float(r[0]), float(r[1]), float(r[2])
            pos, pnl, d = float(r[3]), float(r[4]), r[5]
            fee = 0.0006

            if d == 'bullish':
                half_tp1 = 0.5 * pos * (tp1 - entry) / entry
                half_exit = 0.5 * pos * (exit_p - entry) / entry
            else:
                half_tp1 = 0.5 * pos * (entry - tp1) / entry
                half_exit = 0.5 * pos * (entry - exit_p) / entry

            total_fees = pos * fee * 2
            expect_tp1_be = half_tp1 - total_fees
            expect_both = half_tp1 + half_exit - total_fees

            print(f"${entry:>10,.2f} ${tp1:>10,.2f} ${exit_p:>10,.2f} ${pnl:>8,.2f} {d:<8}")
            print(f"  50%@TP1=${half_tp1:,.2f}  50%@exit=${half_exit:,.2f}  "
                  f"expect(tp1+BE)=${expect_tp1_be:,.2f}  expect(tp1+exit)=${expect_both:,.2f}  "
                  f"actual=${pnl:,.2f}")
            # Which model matches actual?
            tp1_be_diff = abs(pnl - expect_tp1_be)
            both_diff = abs(pnl - expect_both)
            winner = "TP1+BE" if tp1_be_diff < both_diff else "TP1+EXIT"
            print(f"  Closer match: {winner} (tp1+be diff=${tp1_be_diff:,.2f}, tp1+exit diff=${both_diff:,.2f})")
            print()
