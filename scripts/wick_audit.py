"""wick_audit.py - TP1 vs Breakeven audit with wick-hit simulation for Run 29."""
import os, json
os.environ['BACKTEST_DB_PASSWORD'] = 'Schlittlebah44!'
os.environ['BACKTEST_DB_USER'] = 'bulldog'
os.environ['BACKTEST_DB_NAME'] = 'first_db_local'

from backtest.db import get_connection
from statistics import mean, median, stdev
from collections import defaultdict

with get_connection() as conn:
    with conn.cursor() as cur:
        # ── Load all Run 29 trades ──
        cur.execute("""
            SELECT id, entry_price, exit_price, tp1_price, stop_price, target_price,
                   direction, pnl_dollars, pnl_pct, rr, position_size, risk_amount,
                   opened_at, closed_at, exit_reason, tp1_hit, model, leverage
            FROM backtest_trades WHERE run_id = 29 ORDER BY opened_at
        """)
        cols = [d[0] for d in cur.description]
        trades = [dict(zip(cols, r)) for r in cur.fetchall()]

        cur.execute("""
            SELECT COUNT(*), MIN(open_time), MAX(open_time)
            FROM ohlcv_candles WHERE symbol = 'BTCUSDT' AND timeframe = '1h'
        """)
        ci = cur.fetchone()
        print(f"Candle data: {ci[0]} rows, {ci[1]} to {ci[2]}")
        print(f"Total trades: {len(trades)}")

        be_trades = [t for t in trades if t['exit_reason'] == 'breakeven_after_tp1']
        print(f"BE/TP1 trades: {len(be_trades)}")

        # ── Wick simulation for each BE/TP1 trade ──
        wick_results = []
        FEE = 0.0006

        for t in be_trades:
            entry = float(t['entry_price'])
            tp1 = float(t['tp1_price'])
            exit_p = float(t['exit_price'])
            direction = t['direction']
            opened = t['opened_at']
            closed = t['closed_at']
            pos_size = float(t['position_size'])
            risk_amt = float(t['risk_amount'])

            cur.execute("""
                SELECT open_time, open, high, low, close
                FROM ohlcv_candles
                WHERE symbol = 'BTCUSDT' AND timeframe = '1h'
                  AND open_time >= %s AND open_time <= %s
                ORDER BY open_time
            """, (opened, closed))
            candles = cur.fetchall()

            # Phase 1: find TP1 hit candle
            tp1_idx = None
            for i, c in enumerate(candles):
                h, lo = float(c[2]), float(c[3])
                if direction == 'bullish' and h >= tp1:
                    tp1_idx = i; break
                elif direction == 'bearish' and lo <= tp1:
                    tp1_idx = i; break

            if tp1_idx is None:
                wick_results.append({
                    'id': t['id'], 'wick_hit': False, 'candles_after_tp1': 0,
                    'wick_candle': None, 'orig_pnl': float(t['pnl_dollars']),
                    'wick_pnl': float(t['pnl_dollars']), 'direction': direction,
                    'entry': entry, 'tp1': tp1, 'exit': exit_p,
                    'pos_size': pos_size, 'risk': risk_amt,
                    'month': str(opened)[:7], 'model': t['model'],
                    'n_candles': len(candles),
                })
                continue

            # Phase 2: after TP1, check wick touches entry (BE level)
            wick_hit = False
            wick_candle = None
            for i in range(tp1_idx + 1, len(candles)):
                h, lo = float(candles[i][2]), float(candles[i][3])
                if direction == 'bullish' and lo <= entry:
                    wick_hit = True; wick_candle = i - tp1_idx; break
                elif direction == 'bearish' and h >= entry:
                    wick_hit = True; wick_candle = i - tp1_idx; break

            # Compute wick PnL: if wick hits BE, remaining 50% exits at entry
            if wick_hit:
                if direction == 'bullish':
                    tp1_gross = 0.5 * pos_size * (tp1 - entry) / entry
                else:
                    tp1_gross = 0.5 * pos_size * (entry - tp1) / entry
                total_fees = pos_size * FEE * 2
                wick_pnl = tp1_gross - total_fees
            else:
                wick_pnl = float(t['pnl_dollars'])

            wick_results.append({
                'id': t['id'], 'wick_hit': wick_hit,
                'candles_after_tp1': len(candles) - tp1_idx - 1,
                'wick_candle': wick_candle,
                'orig_pnl': float(t['pnl_dollars']),
                'wick_pnl': wick_pnl, 'direction': direction,
                'entry': entry, 'tp1': tp1, 'exit': exit_p,
                'pos_size': pos_size, 'risk': risk_amt,
                'month': str(opened)[:7], 'model': t['model'],
                'n_candles': len(candles),
            })

        # ══════════════════════════════════════════
        # RESULTS
        # ══════════════════════════════════════════
        wh = [r for r in wick_results if r['wick_hit']]
        ws = [r for r in wick_results if not r['wick_hit']]
        total_be = len(wick_results)

        print(f"\n{'='*65}")
        print(f" WICK-HIT SIMULATION RESULTS")
        print(f"{'='*65}")
        print(f"Total BE/TP1 trades:        {total_be}")
        print(f"Wick WOULD hit BE stop:     {len(wh)} ({100*len(wh)/total_be:.1f}%)")
        print(f"Wick survives (close-only): {len(ws)} ({100*len(ws)/total_be:.1f}%)")

        orig_total = sum(r['orig_pnl'] for r in wick_results)
        wick_total = sum(r['wick_pnl'] for r in wick_results)
        print(f"\nOriginal BE/TP1 total PnL:  ${orig_total:,.2f}")
        print(f"Wick-adjusted total PnL:    ${wick_total:,.2f}")
        reduction = orig_total - wick_total
        print(f"PnL reduction:              ${reduction:,.2f} ({100*reduction/orig_total:.1f}%)")

        if wh:
            print(f"\n--- Wick-hit trades ({len(wh)}) ---")
            print(f"  Avg original PnL:   ${mean([r['orig_pnl'] for r in wh]):,.2f}")
            print(f"  Avg wick PnL:       ${mean([r['wick_pnl'] for r in wh]):,.2f}")
            print(f"  Median wick PnL:    ${median([r['wick_pnl'] for r in wh]):,.2f}")
            print(f"  Avg candles to BE:  {mean([r['wick_candle'] for r in wh]):.1f}")
            losses = sum(1 for r in wh if r['wick_pnl'] <= 0)
            print(f"  Become losses:      {losses}/{len(wh)}")

        if ws:
            print(f"\n--- Surviving trades ({len(ws)}) ---")
            print(f"  Avg PnL (unchanged): ${mean([r['orig_pnl'] for r in ws]):,.2f}")

        # ══════════════════════════════════════════
        # FULL PORTFOLIO COMPARISON
        # ══════════════════════════════════════════
        non_be = [t for t in trades if t['exit_reason'] != 'breakeven_after_tp1']
        non_be_pnl = sum(float(t['pnl_dollars']) for t in non_be)
        orig_net = sum(float(t['pnl_dollars']) for t in trades)
        wick_net = non_be_pnl + wick_total

        orig_wins = sum(1 for t in trades if t['pnl_dollars'] > 0)
        wick_be_wins = sum(1 for r in wick_results if r['wick_pnl'] > 0)
        non_be_wins = sum(1 for t in non_be if t['pnl_dollars'] > 0)
        wick_wins = non_be_wins + wick_be_wins
        total_t = len(trades)

        orig_gp = sum(float(t['pnl_dollars']) for t in trades if t['pnl_dollars'] > 0)
        orig_gl = abs(sum(float(t['pnl_dollars']) for t in trades if t['pnl_dollars'] < 0))

        wick_all_pnls = [float(t['pnl_dollars']) for t in non_be] + [r['wick_pnl'] for r in wick_results]
        wick_gp = sum(p for p in wick_all_pnls if p > 0)
        wick_gl = abs(sum(p for p in wick_all_pnls if p < 0))

        pf_orig = orig_gp / orig_gl if orig_gl else float('inf')
        pf_wick = wick_gp / wick_gl if wick_gl else float('inf')

        print(f"\n{'='*65}")
        print(f" FULL PORTFOLIO: CANDLE-CLOSE vs WICK-HIT")
        print(f"{'='*65}")
        print(f"{'Metric':<30} {'Candle-Close':>16} {'Wick-Hit':>16}")
        print(f"{'-'*62}")
        print(f"{'Net Profit':<30} {'${:,.2f}'.format(orig_net):>16} {'${:,.2f}'.format(wick_net):>16}")
        print(f"{'Return':<30} {'{:.1f}%'.format(100*orig_net/5000):>16} {'{:.1f}%'.format(100*wick_net/5000):>16}")
        print(f"{'Win Rate':<30} {'{:.1f}%'.format(100*orig_wins/total_t):>16} {'{:.1f}%'.format(100*wick_wins/total_t):>16}")
        print(f"{'Profit Factor':<30} {'{:.2f}'.format(pf_orig):>16} {'{:.2f}'.format(pf_wick):>16}")
        print(f"{'Gross Profit':<30} {'${:,.2f}'.format(orig_gp):>16} {'${:,.2f}'.format(wick_gp):>16}")
        print(f"{'Gross Loss':<30} {'${:,.2f}'.format(orig_gl):>16} {'${:,.2f}'.format(wick_gl):>16}")
        print(f"{'Wins / Total':<30} {'{}/{}'.format(orig_wins, total_t):>16} {'{}/{}'.format(wick_wins, total_t):>16}")

        # ══════════════════════════════════════════
        # PER-EXIT-TYPE R:R
        # ══════════════════════════════════════════
        print(f"\n{'='*65}")
        print(f" AVG R:R BY EXIT TYPE")
        print(f"{'='*65}")
        for reason in ['breakeven_after_tp1', 'target_hit', 'trailing_stop', 'stop_hit']:
            subset = [t for t in trades if t['exit_reason'] == reason]
            if subset:
                avg_rr = mean([float(t['rr']) for t in subset])
                avg_pnl = mean([float(t['pnl_dollars']) for t in subset])
                print(f"  {reason:<25} {len(subset):>3} trades  Avg R:R={avg_rr:.2f}  Avg PnL=${avg_pnl:,.2f}")

        # ── Wick-adjusted R:R for BE/TP1 ──
        if wh:
            wick_rrs = []
            for r in wh:
                sl_dist = abs(r['entry'] - float([t for t in trades if t['id']==r['id']][0]['stop_price']))
                if sl_dist > 0:
                    actual_move = abs(r['tp1'] - r['entry'])  # only TP1 profit matters
                    wick_rrs.append(actual_move / sl_dist * 0.5)  # half position
            if wick_rrs:
                print(f"  {'BE/TP1 (wick-adjusted)':<25} {len(wh):>3} trades  Avg R:R={mean(wick_rrs):.2f}  Avg PnL=${mean([r['wick_pnl'] for r in wh]):,.2f}")

        # ══════════════════════════════════════════
        # MONTHLY BREAKDOWN
        # ══════════════════════════════════════════
        monthly = defaultdict(lambda: {
            'total': 0, 'be_total': 0, 'wick_hit': 0, 'wick_survive': 0,
            'orig_pnl': 0, 'orig_be_pnl': 0, 'wick_be_pnl': 0
        })
        for t in trades:
            m = str(t['opened_at'])[:7]
            monthly[m]['total'] += 1
            monthly[m]['orig_pnl'] += float(t['pnl_dollars'])
        for r in wick_results:
            m = r['month']
            monthly[m]['be_total'] += 1
            monthly[m]['orig_be_pnl'] += r['orig_pnl']
            monthly[m]['wick_be_pnl'] += r['wick_pnl']
            if r['wick_hit']:
                monthly[m]['wick_hit'] += 1
            else:
                monthly[m]['wick_survive'] += 1

        print(f"\n{'='*80}")
        print(f" MONTHLY BREAKDOWN")
        print(f"{'='*80}")
        print(f"{'Month':<9} {'Tr':>3} {'BE':>3} {'Wick':>4} {'Surv':>4} {'BE%':>5} {'Orig PnL':>12} {'Wick PnL':>12} {'Delta':>10}")
        print(f"{'-'*72}")
        for m in sorted(monthly):
            d = monthly[m]
            wick_adj = d['orig_pnl'] - d['orig_be_pnl'] + d['wick_be_pnl']
            delta = wick_adj - d['orig_pnl']
            bep = 100 * d['be_total'] / d['total'] if d['total'] else 0
            print(f"{m:<9} {d['total']:>3} {d['be_total']:>3} {d['wick_hit']:>4} {d['wick_survive']:>4} "
                  f"{bep:>4.0f}% ${d['orig_pnl']:>10,.2f} ${wick_adj:>10,.2f} ${delta:>8,.2f}")

        # ══════════════════════════════════════════
        # PER-MODEL WICK IMPACT
        # ══════════════════════════════════════════
        model_wick = defaultdict(lambda: {'total': 0, 'wick_hit': 0, 'orig_pnl': 0, 'wick_pnl': 0})
        for r in wick_results:
            mo = r['model']
            model_wick[mo]['total'] += 1
            model_wick[mo]['orig_pnl'] += r['orig_pnl']
            model_wick[mo]['wick_pnl'] += r['wick_pnl']
            if r['wick_hit']:
                model_wick[mo]['wick_hit'] += 1

        print(f"\n{'='*75}")
        print(f" PER-MODEL WICK IMPACT (BE/TP1 trades only)")
        print(f"{'='*75}")
        print(f"{'Model':<25} {'BE':>4} {'Wick':>5} {'Hit%':>6} {'Orig PnL':>12} {'Wick PnL':>12} {'Delta':>10}")
        for mo in sorted(model_wick):
            d = model_wick[mo]
            pct = 100 * d['wick_hit'] / d['total'] if d['total'] else 0
            delta = d['wick_pnl'] - d['orig_pnl']
            print(f"{mo:<25} {d['total']:>4} {d['wick_hit']:>5} {pct:>5.1f}% ${d['orig_pnl']:>10,.2f} ${d['wick_pnl']:>10,.2f} ${delta:>8,.2f}")

        # ══════════════════════════════════════════
        # WICK-HIT TIMING DISTRIBUTION
        # ══════════════════════════════════════════
        if wh:
            candle_dists = [r['wick_candle'] for r in wh]
            print(f"\n{'='*65}")
            print(f" WICK-HIT TIMING (candles after TP1 until BE wick)")
            print(f"{'='*65}")
            print(f"Min: {min(candle_dists)}, Max: {max(candle_dists)}, "
                  f"Avg: {mean(candle_dists):.1f}, Median: {median(candle_dists):.0f}")
            buckets = [('1-2 bars', 1, 2), ('3-5 bars', 3, 5), ('6-12 bars', 6, 12),
                       ('13-24 bars', 13, 24), ('25-48 bars', 25, 48), ('49+ bars', 49, 9999)]
            mx = max(sum(1 for c in candle_dists if lo <= c <= hi) for _, lo, hi in buckets)
            for label, lo, hi in buckets:
                cnt = sum(1 for c in candle_dists if lo <= c <= hi)
                bar = '#' * int(cnt * 35 / mx) if mx else ''
                print(f"  {label:<12} {cnt:>3} ({100*cnt/len(candle_dists):>5.1f}%)  {bar}")

        # ══════════════════════════════════════════
        # EQUITY CURVE & MAX DRAWDOWN
        # ══════════════════════════════════════════
        wick_pnl_map = {r['id']: r['wick_pnl'] for r in wick_results}
        orig_eq = [5000.0]
        wick_eq = [5000.0]
        for t in trades:
            tid = t['id']
            orig_eq.append(orig_eq[-1] + float(t['pnl_dollars']))
            if tid in wick_pnl_map:
                wick_eq.append(wick_eq[-1] + wick_pnl_map[tid])
            else:
                wick_eq.append(wick_eq[-1] + float(t['pnl_dollars']))

        def max_dd(eq):
            peak = eq[0]; dd = 0
            for v in eq:
                if v > peak: peak = v
                d = (peak - v) / peak * 100
                if d > dd: dd = d
            return dd

        print(f"\n{'='*65}")
        print(f" EQUITY CURVE & MAX DRAWDOWN")
        print(f"{'='*65}")
        print(f"{'Metric':<25} {'Candle-Close':>16} {'Wick-Hit':>16}")
        print(f"{'-'*57}")
        print(f"{'Starting equity':<25} {'$5,000.00':>16} {'$5,000.00':>16}")
        print(f"{'Final equity':<25} {'${:,.2f}'.format(orig_eq[-1]):>16} {'${:,.2f}'.format(wick_eq[-1]):>16}")
        print(f"{'Peak equity':<25} {'${:,.2f}'.format(max(orig_eq)):>16} {'${:,.2f}'.format(max(wick_eq)):>16}")
        print(f"{'Max drawdown':<25} {'{:.2f}%'.format(max_dd(orig_eq)):>16} {'{:.2f}%'.format(max_dd(wick_eq)):>16}")
        print(f"{'Return':<25} {'{:.1f}%'.format(100*(orig_eq[-1]-5000)/5000):>16} {'{:.1f}%'.format(100*(wick_eq[-1]-5000)/5000):>16}")
