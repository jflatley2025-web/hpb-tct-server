import psycopg2, json, os
from backtest.db import normalize_model

conn = psycopg2.connect(
    host=os.environ.get("BACKTEST_DB_HOST", "localhost"),
    dbname=os.environ.get("BACKTEST_DB_NAME", "first_db_local"),
    user=os.environ.get("BACKTEST_DB_USER", "bulldog"),
    password=os.environ.get("BACKTEST_DB_PASSWORD", ""),
    port=int(os.environ.get("BACKTEST_DB_PORT", "5432")),
)
cur = conn.cursor()

cur.execute("SELECT total_trades, wins, losses, win_rate, final_balance, max_drawdown_pct, starting_balance, completed_at, config_json FROM backtest_runs WHERE id=27")
r = cur.fetchone()
total, wins, losses, wr, final_bal, max_dd, start_bal, completed, cfg = r
cfg = json.loads(cfg) if isinstance(cfg, str) else (cfg or {})

cur.execute("SELECT SUM(pnl_dollars) FROM backtest_trades WHERE run_id=27 AND is_win=true")
gw = float(cur.fetchone()[0] or 0)
cur.execute("SELECT SUM(ABS(pnl_dollars)) FROM backtest_trades WHERE run_id=27 AND is_win=false")
gl = float(cur.fetchone()[0] or 0)
pf = gw/gl if gl else 9999
net = float(final_bal) - float(start_bal)
expectancy = net / total if total else 0
avg_win = gw / wins if wins else 0
avg_loss = gl / losses if losses else 0

cur.execute("SELECT trade_num, symbol, timeframe, direction, model, entry_price, stop_price, target_price, tp1_price, tp1_hit, position_size, risk_amount, leverage, rr, entry_score, mfe, mae, opened_at, closed_at, exit_price, exit_reason, pnl_pct, pnl_dollars, is_win, balance_after FROM backtest_trades WHERE run_id=27 ORDER BY trade_num")
trades = cur.fetchall()

cur.execute("SELECT model, COUNT(*) as n, SUM(CASE WHEN is_win THEN 1 ELSE 0 END) as w, ROUND((SUM(CASE WHEN is_win THEN 1.0 ELSE 0 END)/COUNT(*)*100)::numeric,1) as wr, ROUND(SUM(pnl_dollars)::numeric,2) as pnl, ROUND(AVG(pnl_dollars)::numeric,2) as avg_pnl, ROUND(SUM(CASE WHEN is_win THEN pnl_dollars ELSE 0 END)::numeric,2) as gw, ROUND(SUM(CASE WHEN NOT is_win THEN ABS(pnl_dollars) ELSE 0 END)::numeric,2) as gl FROM backtest_trades WHERE run_id=27 GROUP BY model ORDER BY model")
model_stats = cur.fetchall()

cur.execute("SELECT timeframe, COUNT(*), SUM(CASE WHEN is_win THEN 1 ELSE 0 END), ROUND((SUM(CASE WHEN is_win THEN 1.0 ELSE 0 END)/COUNT(*)*100)::numeric,1), ROUND(SUM(pnl_dollars)::numeric,2), ROUND(AVG(pnl_dollars)::numeric,2) FROM backtest_trades WHERE run_id=27 GROUP BY timeframe ORDER BY timeframe")
tf_stats = cur.fetchall()

cur.execute("SELECT DATE_TRUNC('month',opened_at), COUNT(*), SUM(CASE WHEN is_win THEN 1 ELSE 0 END), ROUND((SUM(CASE WHEN is_win THEN 1.0 ELSE 0 END)/COUNT(*)*100)::numeric,1), ROUND(SUM(pnl_dollars)::numeric,2) FROM backtest_trades WHERE run_id=27 GROUP BY 1 ORDER BY 1")
monthly = cur.fetchall()

conn.close()

def fmt(v): return f"{float(v):,.2f}"
def fmtp(v): return f"{float(v):.2f}%"

out = []
def l(s=''):
    out.append(s)

l('='*100)
_engine = cfg.get("engine_version", cfg.get("engine", 11))
l(f'BACKTEST REPORT  |  RUN 27  |  ENGINE v{_engine}  |  OFFICIAL v11 BASELINE')
l('='*100)
l('')
l('CONFIGURATION')
l('-'*60)
l('  Exchange:          Binance (simulated, closed-candle fills, no slippage model)')
l('  Pair:              BTCUSDT')
l('  Period:            2025-04-01  to  2026-03-24')
l('  Step interval:     1h')
l(f'  Score threshold:   {cfg.get("entry_threshold", cfg.get("threshold", 50))}')
l('  Warmup days:       90  (signals enabled from 2025-06-30)')
l('  Starting balance:  $5,000.00')
l(f'  Engine version:    {_engine}')
l(f'  Run completed:     {completed}')
l('')
l('CORE METRICS')
l('-'*60)
l(f'  Total trades:      {total}')
l(f'  Wins / Losses:     {wins} W  /  {losses} L')
l(f'  Win rate:          {float(wr):.1f}%')
l(f'  Profit factor:     {pf:.2f}')
l(f'  Expectancy:        ${expectancy:.2f} / trade')
l(f'  Avg winner:        ${avg_win:.2f}')
l(f'  Avg loser:         -${avg_loss:.2f}')
if avg_loss:
    l(f'  Win/Loss ratio:    {avg_win/avg_loss:.2f}x')
l(f'  Gross profit:      ${fmt(gw)}')
l(f'  Gross loss:        -${fmt(gl)}')
l(f'  Net profit:        ${fmt(net)}  (+{net/float(start_bal)*100:.1f}%)')
l(f'  Final balance:     ${fmt(final_bal)}')
l(f'  Max drawdown:      {float(max_dd):.2f}%')
l('')
l('PERFORMANCE BY MODEL')
l('-'*100)
l(f'  {"Model":<35} {"Trades":>6} {"Wins":>5} {"WR%":>6} {"Net PnL":>11} {"Avg PnL":>10} {"GrossWin":>11} {"GrossLoss":>11} {"PF":>6}')
l('  ' + '-'*98)
for ms in model_stats:
    mn, n, mw, mwr, mpnl, mavg, mgw, mgl = ms
    mn = normalize_model(mn) or mn  # map legacy labels (e.g. Model_3 -> Model_2_EXT)
    mpf = float(mgw)/float(mgl) if mgl else 9999
    pf_s = f'{mpf:.2f}' if mpf < 9999 else 'inf'
    l(f'  {mn:<35} {n:>6} {mw:>5} {float(mwr):>5.1f}% {float(mpnl):>11,.2f} {float(mavg):>10,.2f} {float(mgw):>11,.2f} {float(mgl):>11,.2f} {pf_s:>6}')
l('')
l('PERFORMANCE BY TIMEFRAME')
l('-'*80)
l(f'  {"TF":<8} {"Trades":>6} {"Wins":>5} {"WR%":>6} {"Net PnL":>11} {"Avg PnL":>10}')
l('  ' + '-'*55)
for tf in tf_stats:
    tfn, n, mw, mwr, mpnl, mavg = tf
    l(f'  {tfn:<8} {n:>6} {mw:>5} {float(mwr):>5.1f}% {float(mpnl):>11,.2f} {float(mavg):>10,.2f}')
l('')
l('MONTHLY PERFORMANCE')
l('-'*60)
l(f'  {"Month":<12} {"Trades":>6} {"Wins":>5} {"WR%":>6} {"Net PnL":>11}')
l('  ' + '-'*45)
for mo in monthly:
    mon, n, mw, mwr, mpnl = mo
    l(f'  {str(mon)[:7]:<12} {n:>6} {mw:>5} {float(mwr):>5.1f}% {float(mpnl):>11,.2f}')
l('')
l('PnL vs WIN RATE ANALYSIS')
l('-'*80)
l('')
l('  Model PnL vs WR (sorted by WR descending):')
l(f'  {"Model":<35} {"WR%":>6}  {"Net PnL":>11}  {"PnL/Trade":>10}')
l('  ' + '-'*70)
for ms in sorted(model_stats, key=lambda x: float(x[3]), reverse=True):
    mn, n, mw, mwr, mpnl, mavg, mgw, mgl = ms
    mn = normalize_model(mn) or mn
    l(f'  {mn:<35} {float(mwr):>5.1f}%  ${float(mpnl):>10,.2f}  ${float(mavg):>9,.2f}/trade')
l('')
l('  Timeframe PnL vs WR (sorted by avg PnL descending):')
l(f'  {"TF":<8} {"WR%":>6}  {"Net PnL":>11}  {"PnL/Trade":>10}')
l('  ' + '-'*50)
for tf in sorted(tf_stats, key=lambda x: float(x[5]), reverse=True):
    tfn, n, mw, mwr, mpnl, mavg = tf
    l(f'  {tfn:<8} {float(mwr):>5.1f}%  ${float(mpnl):>10,.2f}  ${float(mavg):>9,.2f}/trade')
l('')
l('  Key Observations:')
l('  - 15m: lowest WR (69.2%) and lowest avg PnL ($41/trade) -- primary target for v12/v13 hardening')
l('  - Model_2/15m: 8 trades, 62.5% WR, -$13 net -- only net-negative model x TF bucket')
l('  - Model_2_EXT/4h: 10 trades, 70% WR, $25 avg -- low edge; eliminated by FAIL_MODEL3_TF_FILTER in v12')
l('  - 4h TF: highest avg PnL ($152/trade), 29 trades -- best edge quality')
l('  - Model_1_from_M2_failure: 7/7 wins (100% WR, $85/trade) -- strongest sub-pattern')
if avg_loss:
    l(f'  - High PF (4.97) driven by avg win/loss ratio of {avg_win/avg_loss:.2f}x')
l('')
l('='*100)
l('TRADE-BY-TRADE LOG')
l('='*100)
l('')

for t in trades:
    (trade_num, symbol, timeframe, direction, model,
     entry_price, stop_price, target_price, tp1_price,
     tp1_hit, position_size, risk_amount, leverage, rr,
     entry_score, mfe, mae,
     opened_at, closed_at, exit_price, exit_reason,
     pnl_pct, pnl_dollars, is_win, balance_after) = t

    wl = 'WIN' if is_win else 'LOSS'
    tp1_s = f'${float(tp1_price):,.2f}' if tp1_price else 'N/A'
    tp1_hit_s = 'HIT' if tp1_hit else 'MISS'
    lev_s = f'{float(leverage):.1f}x' if leverage else '1.0x'
    score_s = f'{float(entry_score):.1f}' if entry_score else 'N/A'
    opened_s = str(opened_at)[:19] if opened_at else 'N/A'
    closed_s = str(closed_at)[:19] if closed_at else 'N/A'
    entry_f = float(entry_price)
    stop_f = float(stop_price)
    target_f = float(target_price)
    exit_f = float(exit_price)
    # position_size is USD notional — convert to BTC quantity for price-distance math
    size_notional = float(position_size)
    size_qty = size_notional / entry_f  # BTC quantity
    risk_f = float(risk_amount)
    rr_f = float(rr)
    pnl_f = float(pnl_dollars)
    pnl_pct_f = float(pnl_pct)
    bal_f = float(balance_after)
    stop_dist_pct = abs(entry_f - stop_f) / entry_f * 100
    target_dist_pct = abs(target_f - entry_f) / entry_f * 100
    exit_dist_pct = abs(exit_f - entry_f) / entry_f * 100
    # MFE/MAE are stored as absolute price deltas — convert to % of entry
    mfe_f = float(mfe) if mfe else 0.0
    mae_f = abs(float(mae)) if mae else 0.0
    mfe_s = f'{mfe_f:.2f} pts ({mfe_f / entry_f * 100:.3f}%)' if mfe else 'N/A'
    mae_s = f'{mae_f:.2f} pts ({mae_f / entry_f * 100:.3f}%)' if mae else 'N/A'
    bal_before = bal_f - pnl_f
    pnl_sign = '+' if pnl_f >= 0 else ''

    l('')
    l('  ' + '-'*90)
    l(f'  TRADE #{trade_num:03d}  |  {symbol}  |  {timeframe}  |  {direction.upper():<5}  |  {model:<30}  |  {wl}')
    l('  ' + '-'*90)
    l(f'  Exchange:       Binance (simulated)')
    l(f'  Pair:           {symbol}')
    l(f'  Opened:         {opened_s} UTC')
    l(f'  Closed:         {closed_s} UTC')
    l(f'  Direction:      {direction.upper()}')
    l(f'  Timeframe:      {timeframe}')
    l(f'  Model:          {model}')
    l('')
    l('  --- PRICES ---')
    l(f'  Entry price:    ${entry_f:,.2f}')
    l(f'  Stop loss:      ${stop_f:,.2f}  ({stop_dist_pct:.2f}% from entry  |  dollar risk ${abs(entry_f - stop_f) * size_qty:.2f})')
    l(f'  Target price:   ${target_f:,.2f}  ({target_dist_pct:.2f}% from entry  |  RR {rr_f:.2f})')
    l(f'  TP1 price:      {tp1_s:<15}  ({tp1_hit_s})')
    l(f'  Exit price:     ${exit_f:,.2f}  ({exit_dist_pct:.2f}% from entry)')
    l(f'  Exit reason:    {exit_reason}')
    l('')
    l('  --- POSITION ---')
    l(f'  Size:           {size_qty:.6f} BTC  (${size_notional:,.2f} notional)')
    l(f'  Risk amount:    ${risk_f:.2f}')
    l(f'  Leverage:       {lev_s}')
    l(f'  Entry score:    {score_s}')
    l('')
    l('  --- OUTCOME ---')
    l(f'  MFE:            {mfe_s}  (max favorable excursion)')
    l(f'  MAE:            {mae_s}  (max adverse excursion)')
    l(f'  PnL:            {pnl_sign}${pnl_f:,.2f}  ({pnl_sign}{pnl_pct_f:.2f}%)')
    l(f'  Balance before: ${bal_before:,.2f}')
    l(f'  Balance after:  ${bal_f:,.2f}')
    l(f'  Result:         {wl}')

l('')
l('='*100)
l(f'END OF REPORT  |  RUN 27  |  ENGINE v{_engine}  |  {total} TRADES  |  {float(wr):.1f}% WR  |  PF {pf:.2f}  |  +${fmt(net)}')
l('='*100)

print('\n'.join(out))
