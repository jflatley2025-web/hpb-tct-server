"""Compare Run 29 (old) vs Run 40 (fixed labeling) exit distributions."""
import sys; sys.path.insert(0, '.')
import os
os.environ['BACKTEST_DB_PASSWORD'] = 'Schlittlebah44!'
os.environ['BACKTEST_DB_USER'] = 'bulldog'
os.environ['BACKTEST_DB_NAME'] = 'first_db_local'
from backtest.db import get_connection

with get_connection() as conn:
    with conn.cursor() as cur:
        for rid, label in [(29, "Run 29 (OLD)"), (40, "Run 40 (FIXED)")]:
            cur.execute("""
                SELECT exit_reason, COUNT(*) as cnt,
                       ROUND(SUM(pnl_dollars)::numeric, 2) as total_pnl,
                       ROUND(AVG(pnl_dollars)::numeric, 2) as avg_pnl,
                       SUM(CASE WHEN is_win THEN 1 ELSE 0 END) as wins
                FROM backtest_trades WHERE run_id = %s
                GROUP BY exit_reason ORDER BY cnt DESC
            """, (rid,))
            rows = cur.fetchall()
            print(f"=== {label} Exit Distribution ===")
            for r in rows:
                reason, cnt, tot, avg, wins = r[0], r[1], float(r[2]), float(r[3]), r[4]
                print(f"  {reason:<25} {cnt:>4} trades  total=${tot:>10,.2f}  avg=${avg:>8,.2f}  wins={wins}")
            print()

        print("=" * 65)
        print(f"{'Metric':<30} {'Run 29 (OLD)':>16} {'Run 40 (FIXED)':>16}")
        print("-" * 65)

        for rid, label in [(29, "Run 29"), (40, "Run 40")]:
            cur.execute("""
                SELECT COUNT(*),
                       SUM(CASE WHEN is_win THEN 1 ELSE 0 END),
                       ROUND(SUM(pnl_dollars)::numeric, 2),
                       ROUND(MAX(balance_after)::numeric, 2)
                FROM backtest_trades WHERE run_id = %s
            """, (rid,))
            r = cur.fetchone()
            total, wins, pnl, final = r[0], r[1], float(r[2]), float(r[3])
            wr = 100 * wins / total if total else 0
            ret = 100 * pnl / 5000
            if rid == 29:
                old = (total, wins, total - wins, wr, pnl, final, ret)
            else:
                new = (total, wins, total - wins, wr, pnl, final, ret)

        metrics = [
            ("Total Trades", f"{old[0]}", f"{new[0]}"),
            ("Wins", f"{old[1]}", f"{new[1]}"),
            ("Losses", f"{old[2]}", f"{new[2]}"),
            ("Win Rate", f"{old[3]:.1f}%", f"{new[3]:.1f}%"),
            ("Net Profit", f"${old[4]:,.2f}", f"${new[4]:,.2f}"),
            ("Final Balance", f"${old[5]:,.2f}", f"${new[5]:,.2f}"),
            ("Return", f"{old[6]:.1f}%", f"{new[6]:.1f}%"),
        ]
        for name, v1, v2 in metrics:
            print(f"  {name:<28} {v1:>16} {v2:>16}")

        # Model breakdown for Run 40
        print()
        print("=== Run 40 Model Breakdown ===")
        cur.execute("""
            SELECT model, COUNT(*), SUM(CASE WHEN is_win THEN 1 ELSE 0 END),
                   ROUND(100.0 * SUM(CASE WHEN is_win THEN 1 ELSE 0 END) / COUNT(*), 1),
                   ROUND(SUM(pnl_dollars)::numeric, 2),
                   ROUND(AVG(rr)::numeric, 2)
            FROM backtest_trades WHERE run_id = 40
            GROUP BY model ORDER BY SUM(pnl_dollars) DESC
        """)
        for r in cur.fetchall():
            print(f"  {r[0]:<25} {r[1]:>4}t  {r[2]}W  WR={float(r[3]):.1f}%  PnL=${float(r[4]):>10,.2f}  avgRR={float(r[5]):.2f}")
