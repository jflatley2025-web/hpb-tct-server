"""Detailed sweep output script."""

if __name__ == "__main__":
    from backtest.threshold_sweep import simulate_threshold
    from backtest.db import get_connection

    conn = get_connection()
    try:
        result = simulate_threshold(6, 40, conn=conn)

        print("=== THRESHOLD 40: DETAILED TRADES ===")
        print("{:<22} {:>5} {:>8} {:>6} {:>8} {:>8} {:>8}".format(
            "Time", "Score", "Session", "Dir", "Outcome", "PnL$", "Balance"))
        print("-" * 78)
        for t in result["trades"]:
            print("{:<22} {:>5} {:>8} {:>6} {:>8} {:>7.2f} {:>8.2f}".format(
                str(t["signal_time"])[:19], t["score"], t["session"],
                t["direction"][:4], t["outcome"], t["pnl_dollars"],
                t["balance_after"]))

        print()
        result30 = simulate_threshold(6, 30, conn=conn)
        print("=== THRESHOLD 30: EXTRA TRADE(S) ===")
        for t in result30["trades"]:
            if t["score"] < 40:
                print("  Score={} time={} session={} outcome={} pnl=${:.2f}".format(
                    t["score"], str(t["signal_time"])[:19], t["session"],
                    t["outcome"], t["pnl_dollars"]))
    finally:
        conn.close()
