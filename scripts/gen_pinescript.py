"""Generate PineScript trade overlay indicators for ETH and SOL backtests.
Includes tap 1/2/3 markers and Fibonacci range levels from schematic detection.
Splits data loading into batched functions to avoid PineScript block-size limits."""
import sys; sys.path.insert(0, '.')
import os
import json
os.environ['BACKTEST_DB_PASSWORD'] = 'Schlittlebah44!'
os.environ['BACKTEST_DB_USER'] = 'bulldog'
os.environ['BACKTEST_DB_NAME'] = 'first_db_local'

from backtest.db import get_connection

BATCH_SIZE = 8  # trades per loader function (keeps each function well under Pine limits)


def gen_pine(symbol, run_id):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, entry_price, exit_price, tp1_price, stop_price, target_price,
                       direction, exit_reason, is_win, pnl_dollars, model,
                       opened_at, closed_at, tp1_hit, rr, timeframe
                FROM backtest_trades WHERE run_id = %s ORDER BY opened_at
            """, (run_id,))
            tcols = [d[0] for d in cur.description]
            trades = [dict(zip(tcols, r)) for r in cur.fetchall()]

            cur.execute("""
                SELECT signal_time, entry_price, direction, model, timeframe,
                       schematic_json
                FROM backtest_signals
                WHERE run_id = %s AND final_decision = 'TAKE'
                  AND schematic_json IS NOT NULL
                ORDER BY signal_time
            """, (run_id,))
            signals = cur.fetchall()

            candle_maps = {}
            for tf in ["1h", "30m", "15m", "4h"]:
                cur.execute("""
                    SELECT open_time FROM ohlcv_candles
                    WHERE symbol = %s AND timeframe = %s
                    ORDER BY open_time
                """, (symbol, tf))
                candle_maps[tf] = [r[0] for r in cur.fetchall()]

    # Build signal lookup
    sig_lookup = {}
    for s in signals:
        sig_time, sig_entry, sig_dir, sig_model, sig_tf, sj = s
        if sj is None:
            continue
        if isinstance(sj, str):
            sj = json.loads(sj)
        key = (round(float(sig_entry), 1), sig_dir, sig_tf)
        sig_lookup[key] = (sj, sig_time, sig_tf)

    # Resolve taps + range for each trade
    trade_rows = []  # flat list of all numeric values per trade
    for t in trades:
        entry_r = round(float(t['entry_price']), 1)
        key = (entry_r, t['direction'], t['timeframe'])
        match = sig_lookup.get(key)

        taps = {'t1_ts': 0, 't1_p': 0.0, 't2_ts': 0, 't2_p': 0.0, 't3_ts': 0, 't3_p': 0.0,
                'rng_hi': 0.0, 'rng_lo': 0.0}

        if match:
            sj, sig_time, sig_tf = match
            cmap = candle_maps.get(sig_tf, [])
            for tap_num in [1, 2, 3]:
                tap_price = sj.get(f"tap{tap_num}_price")
                tap_idx = sj.get(f"tap{tap_num}_idx")
                if tap_price is not None and tap_idx is not None:
                    tap_idx = int(tap_idx)
                    taps[f't{tap_num}_p'] = float(tap_price)
                    if 0 <= tap_idx < len(cmap):
                        taps[f't{tap_num}_ts'] = int(cmap[tap_idx].timestamp() * 1000)
            rng_hi = sj.get("range_high")
            rng_lo = sj.get("range_low")
            if rng_hi is not None and rng_lo is not None:
                taps['rng_hi'] = float(rng_hi)
                taps['rng_lo'] = float(rng_lo)

        entry = float(t['entry_price'])
        exit_p = float(t['exit_price'])
        tp1 = float(t['tp1_price'])
        stop = float(t['stop_price'])
        target = float(t['target_price'])
        d = t['direction']
        reason = t['exit_reason']
        win = t['is_win']
        opened = t['opened_at']
        closed = t['closed_at']
        model = t['model']

        open_ts = int(opened.timestamp() * 1000)
        close_ts = int(closed.timestamp() * 1000)
        side = 1 if d == 'bullish' else -1
        w = 1 if win else 0
        rmap = {'stop_hit': 0, 'breakeven_after_tp1': 1, 'trailing_stop': 2,
                'target_hit': 3, 'backtest_end': 4}
        r = rmap.get(reason, 0)
        mmap = {'Model_1': 1, 'Model_2': 2, 'Model_3': 3,
                'Model_1_from_M2_failure': 4, 'Model_1_CONTINUATION': 5,
                'Model_2_CONTINUATION': 6}
        m = mmap.get(model, 0)

        trade_rows.append({
            'eP': entry, 'xP': exit_p, 'sP': stop, 'tP': target, 't1P': tp1,
            'oT': open_ts, 'cT': close_ts, 'sd': side, 'wn': w, 'er': r, 'md': m,
            'tap1T': taps['t1_ts'], 'tap1P': taps['t1_p'],
            'tap2T': taps['t2_ts'], 'tap2P': taps['t2_p'],
            'tap3T': taps['t3_ts'], 'tap3P': taps['t3_p'],
            'rHi': taps['rng_hi'], 'rLo': taps['rng_lo'],
        })

    n = len(trade_rows)

    # ── Build batched loader functions ──
    # Each function loads BATCH_SIZE trades worth of array.push calls
    float_arrays = ['eP', 'xP', 'sP', 'tP', 't1P', 'tap1P', 'tap2P', 'tap3P', 'rHi', 'rLo']
    int_arrays = ['oT', 'cT', 'sd', 'wn', 'er', 'md', 'tap1T', 'tap2T', 'tap3T']

    loader_functions = []
    num_batches = (n + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_idx in range(num_batches):
        start = batch_idx * BATCH_SIZE
        end = min(start + BATCH_SIZE, n)
        batch = trade_rows[start:end]

        lines = []
        for row in batch:
            for arr in float_arrays:
                lines.append(f"    array.push({arr},{row[arr]:.2f})")
            for arr in int_arrays:
                lines.append(f"    array.push({arr},{row[arr]})")

        func_name = f"_load_{batch_idx}"
        func_body = "\n".join(lines)
        loader_functions.append(f"{func_name}() =>\n{func_body}\n    true")

    loader_defs = "\n\n".join(loader_functions)

    # Init calls
    init_calls = "\n".join(f"    _load_{i}()" for i in range(num_batches))

    script = f"""// ================================================================
// HPB-TCT Backtest Trade Overlay - {symbol}
// Run {run_id} | {n} trades | Engine v14
// Includes Tap 1/2/3, Range Box, and Fibonacci levels
// Add to TradingView: Pine Editor > New Indicator > Paste > Add to Chart
// Use on {symbol} 1h chart (Binance or MEXC)
// ================================================================
//@version=5
indicator("HPB-TCT Trades - {symbol}", overlay=true, max_lines_count=500, max_labels_count=500, max_boxes_count=500)

// ---- Settings ----
showEntry   = input.bool(true, "Show Entry Markers")
showExit    = input.bool(true, "Show Exit Markers")
showLines   = input.bool(true, "Show SL/TP/TP1 Lines")
showPnL     = input.bool(true, "Show PnL on Exit")
showTaps    = input.bool(true, "Show Tap 1/2/3 Markers")
showRange   = input.bool(true, "Show Range Box")
showFibs    = input.bool(true, "Show Fib Levels (0.236, 0.382, 0.5, 0.618, 0.786)")
filterModel = input.int(0, "Filter Model (0=All, 1=M1, 2=M2, 3=M3, 4=M1f, 5=M1C, 6=M2C)", minval=0, maxval=6)

// ---- Colors ----
var color cLong    = #00e676
var color cShort   = #ff5252
var color cWin     = #00e676
var color cLoss    = #ff5252
var color cStop    = color.new(#ff5252, 60)
var color cTarget  = color.new(#00e676, 60)
var color cTP1     = color.new(#00bcd4, 60)
var color cTap1    = #ff9800
var color cTap2    = #ffeb3b
var color cTap3    = #e040fb
var color cRangeHi = color.new(#4fc3f7, 70)
var color cRangeBg = color.new(#4fc3f7, 92)
var color cFib236  = color.new(#ef5350, 50)
var color cFib382  = color.new(#ff9800, 50)
var color cFib500  = color.new(#ffeb3b, 40)
var color cFib618  = color.new(#ff9800, 50)
var color cFib786  = color.new(#ef5350, 50)

// ---- Trade Data Arrays ----
var int N = {n}
var float[] eP    = array.new_float(0)
var float[] xP    = array.new_float(0)
var float[] sP    = array.new_float(0)
var float[] tP    = array.new_float(0)
var float[] t1P   = array.new_float(0)
var int[]   oT    = array.new_int(0)
var int[]   cT    = array.new_int(0)
var int[]   sd    = array.new_int(0)
var int[]   wn    = array.new_int(0)
var int[]   er    = array.new_int(0)
var int[]   md    = array.new_int(0)
var int[]   tap1T = array.new_int(0)
var float[] tap1P = array.new_float(0)
var int[]   tap2T = array.new_int(0)
var float[] tap2P = array.new_float(0)
var int[]   tap3T = array.new_int(0)
var float[] tap3P = array.new_float(0)
var float[] rHi   = array.new_float(0)
var float[] rLo   = array.new_float(0)

// ---- Batched Data Loaders ----
{loader_defs}

// ---- Initialize on first bar ----
if barstate.isfirst
{init_calls}

// ---- Draw Logic ----
t = time
halfBar = 1800000

for i = 0 to N - 1
    if filterModel != 0 and array.get(md, i) != filterModel
        continue

    _oT = array.get(oT, i)
    _cT = array.get(cT, i)
    _eP = array.get(eP, i)
    _xP = array.get(xP, i)
    _sP = array.get(sP, i)
    _tP = array.get(tP, i)
    _t1 = array.get(t1P, i)
    _sd = array.get(sd, i)
    _wn = array.get(wn, i)
    _er = array.get(er, i)
    _md = array.get(md, i)

    mStr = _md == 1 ? "M1" : _md == 2 ? "M2" : _md == 3 ? "M3" : _md == 4 ? "M1f" : _md == 5 ? "M1C" : _md == 6 ? "M2C" : "?"
    rStr = _er == 0 ? "SL" : _er == 1 ? "BE" : _er == 2 ? "Trail" : _er == 3 ? "TP" : "End"

    // ---- Entry label ----
    if showEntry and t >= _oT - halfBar and t <= _oT + halfBar
        _col = _sd == 1 ? cLong : cShort
        _sty = _sd == 1 ? label.style_label_up : label.style_label_down
        _txt = (_sd == 1 ? "LONG " : "SHORT ") + mStr
        label.new(_oT, _eP, text=_txt, xloc=xloc.bar_time, style=_sty, color=_col, textcolor=color.white, size=size.small)

    // ---- Exit label ----
    if showExit and t >= _cT - halfBar and t <= _cT + halfBar
        _rCol = _wn == 1 ? cWin : cLoss
        _pnl = _sd == 1 ? (_xP - _eP) / _eP * 100 : (_eP - _xP) / _eP * 100
        _pStr = (_pnl >= 0 ? "+" : "") + str.tostring(_pnl, "#.##") + "%"
        _eTxt = rStr + " " + (showPnL ? _pStr : "")
        _eSty = _wn == 1 ? label.style_label_down : label.style_label_up
        label.new(_cT, _xP, text=_eTxt, xloc=xloc.bar_time, style=_eSty, color=_rCol, textcolor=color.white, size=size.tiny)

    // ---- SL / TP / TP1 / Entry lines ----
    if showLines and t >= _oT and t <= _oT + halfBar
        line.new(_oT, _eP, _cT, _eP, xloc=xloc.bar_time, color=_sd == 1 ? cLong : cShort, style=line.style_solid, width=2)
        line.new(_oT, _sP, _cT, _sP, xloc=xloc.bar_time, color=cStop, style=line.style_dashed, width=1)
        line.new(_oT, _tP, _cT, _tP, xloc=xloc.bar_time, color=cTarget, style=line.style_dashed, width=1)
        line.new(_oT, _t1, _cT, _t1, xloc=xloc.bar_time, color=cTP1, style=line.style_dotted, width=1)

    // ---- Tap 1/2/3 markers ----
    if showTaps
        _tap1T = array.get(tap1T, i)
        _tap1P = array.get(tap1P, i)
        _tap2T = array.get(tap2T, i)
        _tap2P = array.get(tap2P, i)
        _tap3T = array.get(tap3T, i)
        _tap3P = array.get(tap3P, i)

        if _tap1T > 0 and t >= _tap1T - halfBar and t <= _tap1T + halfBar
            label.new(_tap1T, _tap1P, text="T1", xloc=xloc.bar_time,
                 style=label.style_diamond, color=cTap1, textcolor=color.black, size=size.tiny)
        if _tap2T > 0 and t >= _tap2T - halfBar and t <= _tap2T + halfBar
            label.new(_tap2T, _tap2P, text="T2", xloc=xloc.bar_time,
                 style=label.style_diamond, color=cTap2, textcolor=color.black, size=size.tiny)
        if _tap3T > 0 and t >= _tap3T - halfBar and t <= _tap3T + halfBar
            label.new(_tap3T, _tap3P, text="T3", xloc=xloc.bar_time,
                 style=label.style_diamond, color=cTap3, textcolor=color.black, size=size.tiny)

    // ---- Range box + Fib levels ----
    _rHi = array.get(rHi, i)
    _rLo = array.get(rLo, i)
    _tap1T_r = array.get(tap1T, i)

    if _rHi > 0 and _rLo > 0 and _tap1T_r > 0
        _rangeStart = _tap1T_r
        _rangeEnd   = _oT

        if showRange and t >= _rangeStart and t <= _rangeStart + halfBar
            box.new(_rangeStart, _rHi, _rangeEnd, _rLo, xloc=xloc.bar_time,
                 border_color=cRangeHi, border_width=1, border_style=line.style_solid,
                 bgcolor=cRangeBg)
            label.new(_rangeStart, _rHi, text="RNG Hi " + str.tostring(_rHi, "#.##"),
                 xloc=xloc.bar_time, style=label.style_none,
                 textcolor=cRangeHi, size=size.tiny)
            label.new(_rangeStart, _rLo, text="RNG Lo " + str.tostring(_rLo, "#.##"),
                 xloc=xloc.bar_time, style=label.style_none,
                 textcolor=cRangeHi, size=size.tiny)

        if showFibs and t >= _rangeStart and t <= _rangeStart + halfBar
            _range = _rHi - _rLo
            _f236 = _rLo + _range * 0.236
            _f382 = _rLo + _range * 0.382
            _f500 = _rLo + _range * 0.5
            _f618 = _rLo + _range * 0.618
            _f786 = _rLo + _range * 0.786

            line.new(_rangeStart, _f236, _rangeEnd, _f236, xloc=xloc.bar_time,
                 color=cFib236, style=line.style_dotted, width=1)
            line.new(_rangeStart, _f382, _rangeEnd, _f382, xloc=xloc.bar_time,
                 color=cFib382, style=line.style_dotted, width=1)
            line.new(_rangeStart, _f500, _rangeEnd, _f500, xloc=xloc.bar_time,
                 color=cFib500, style=line.style_dotted, width=1)
            line.new(_rangeStart, _f618, _rangeEnd, _f618, xloc=xloc.bar_time,
                 color=cFib618, style=line.style_dotted, width=1)
            line.new(_rangeStart, _f786, _rangeEnd, _f786, xloc=xloc.bar_time,
                 color=cFib786, style=line.style_dotted, width=1)

            label.new(_rangeEnd, _f236, text="0.236", xloc=xloc.bar_time,
                 style=label.style_none, textcolor=cFib236, size=size.tiny)
            label.new(_rangeEnd, _f382, text="0.382", xloc=xloc.bar_time,
                 style=label.style_none, textcolor=cFib382, size=size.tiny)
            label.new(_rangeEnd, _f500, text="0.5", xloc=xloc.bar_time,
                 style=label.style_none, textcolor=cFib500, size=size.tiny)
            label.new(_rangeEnd, _f618, text="0.618", xloc=xloc.bar_time,
                 style=label.style_none, textcolor=cFib618, size=size.tiny)
            label.new(_rangeEnd, _f786, text="0.786", xloc=xloc.bar_time,
                 style=label.style_none, textcolor=cFib786, size=size.tiny)
"""
    return script


for sym, rid in [('ETHUSDT', 38), ('SOLUSDT', 39)]:
    script = gen_pine(sym, rid)
    path = os.path.join(os.path.dirname(__file__), '..', f'{sym}_backtest_trades.pine')
    path = os.path.abspath(path)
    with open(path, 'w') as f:
        f.write(script)
    lines = script.count('\n')
    print(f"Written {path} ({lines} lines, {len(script):,} chars, {sym} run {rid})")
