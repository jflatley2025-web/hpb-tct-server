"""Generate PineScript for Run 40 (BTCUSDT 1h) backtest trades.
Resolves tap timestamps by replicating the exact backtest DataFrame window."""
import sys; sys.path.insert(0, '.')
import os
import json
from datetime import datetime, timezone
os.environ['BACKTEST_DB_PASSWORD'] = 'Schlittlebah44!'
os.environ['BACKTEST_DB_USER'] = 'bulldog'
os.environ['BACKTEST_DB_NAME'] = 'first_db_local'

from backtest.db import get_connection

BATCH_SIZE = 6
SYMBOL = "BTCUSDT"
RUN_ID = 40
MAX_TRADES_PER_SCRIPT = 80


DETECTION_WINDOW = 200  # Must match backtest/runner.py line 440


def resolve_tap_times(cur, symbol, sig_tf, sig_time, sj):
    """Resolve tap indices to timestamps by replicating the exact backtest
    DataFrame window: last DETECTION_WINDOW candles with open_time < aligned_time.

    This matches get_last_closed() + .tail(DETECTION_WINDOW).reset_index(drop=True)
    from backtest/runner.py lines 247-251 and 569.
    """
    from backtest.config import timeframe_to_seconds
    tf_sec = timeframe_to_seconds(sig_tf)
    sig_ts = sig_time.timestamp()
    aligned_ts = sig_ts - (sig_ts % tf_sec)
    aligned_time = datetime.fromtimestamp(aligned_ts, tz=timezone.utc)

    cur.execute("""
        SELECT open_time FROM ohlcv_candles
        WHERE symbol = %s AND timeframe = %s AND open_time < %s
        ORDER BY open_time DESC LIMIT %s
    """, (symbol, sig_tf, aligned_time, DETECTION_WINDOW))
    window = list(reversed(cur.fetchall()))  # oldest first, 0-indexed

    results = {}
    for tn in [1, 2, 3]:
        tp = sj.get(f"tap{tn}_price")
        ti = sj.get(f"tap{tn}_idx")
        if tp is not None and ti is not None:
            ti = int(ti)
            results[f't{tn}_p'] = float(tp)
            if 0 <= ti < len(window):
                results[f't{tn}_ts'] = int(window[ti][0].timestamp() * 1000)
            else:
                results[f't{tn}_ts'] = 0
        else:
            results[f't{tn}_p'] = 0.0
            results[f't{tn}_ts'] = 0

    return results


def build_script(part_data, n, part_num, num_parts, total_trades, trade_range):
    float_map = {'eP': 'eP', 'xP': 'xP', 'sP': 'sP', 'tP': 'tP', 't1P': 't1P',
                 'tap1P': 't1_p', 'tap2P': 't2_p', 'tap3P': 't3_p',
                 'rHi': 'rng_hi', 'rLo': 'rng_lo'}
    int_map = {'oT': 'oT', 'cT': 'cT', 'sd': 'sd', 'wn': 'wn', 'er': 'er', 'md': 'md',
               'tap1T': 't1_ts', 'tap2T': 't2_ts', 'tap3T': 't3_ts'}

    loaders = []
    num_batches = (n + BATCH_SIZE - 1) // BATCH_SIZE
    for bi in range(num_batches):
        bs = bi * BATCH_SIZE
        be = min(bs + BATCH_SIZE, n)
        batch = part_data[bs:be]
        lines = []
        for row in batch:
            for pn, dk in float_map.items():
                lines.append(f"    array.push({pn},{row[dk]:.2f})")
            for pn, dk in int_map.items():
                lines.append(f"    array.push({pn},{row[dk]})")
        loaders.append(f"_ld{bi}() =>\n" + "\n".join(lines) + "\n    true")

    loader_defs = "\n\n".join(loaders)
    init_calls = "\n".join(f"    _ld{i}()" for i in range(num_batches))
    pt = f" Pt{part_num}" if num_parts > 1 else ""
    pn = f" | {trade_range} of {total_trades}" if num_parts > 1 else ""

    return f"""// ================================================================
// HPB-TCT Run 40{pt} - BTCUSDT 1h
// {n} trades{pn} | Engine v14
// TCT Model + Tap 1/2/3 + Range + Fibs
// ================================================================
//@version=5
indicator("Run 40{pt} - BTCUSDT", overlay=true,
     max_lines_count=500, max_labels_count=500, max_boxes_count=500)

grpT = "Trade Display"
showEntry = input.bool(true, "Entry Markers", group=grpT)
showExit  = input.bool(true, "Exit Markers", group=grpT)
showLines = input.bool(true, "SL/TP/TP1 Lines", group=grpT)
showPnL   = input.bool(true, "PnL on Exit", group=grpT)

grpS = "TCT Structure"
showTaps  = input.bool(true, "Tap 1/2/3", group=grpS)
showModel = input.bool(true, "Model Label", group=grpS)
showRange = input.bool(true, "Range Box", group=grpS)
showFibs  = input.bool(true, "Fib Levels", group=grpS)

grpF = "Filters"
fModel = input.int(0, "Model (0=All 1-7)", minval=0, maxval=7, group=grpF)
fDir   = input.int(0, "Dir (0=All 1=Long 2=Short)", minval=0, maxval=2, group=grpF)

var color cL=#00e676, var color cS=#ff5252
var color cW=#00e676, var color cX=#ff5252
var color cSL=color.new(#ff5252,60), var color cTG=color.new(#00e676,60)
var color cT1c=color.new(#00bcd4,60)
var color cTp1=#ff9800, var color cTp2=#ffeb3b, var color cTp3=#e040fb
var color cMd=#42a5f5
var color cRn=color.new(#4fc3f7,70), var color cRB=color.new(#4fc3f7,93)
var color cF2=color.new(#ef5350,55), var color cF3=color.new(#ff9800,55)
var color cF5=color.new(#ffeb3b,45), var color cF6=color.new(#ff9800,55)
var color cF7=color.new(#ef5350,55)

var int N={n}
var float[] eP=array.new_float(0), var float[] xP=array.new_float(0)
var float[] sP=array.new_float(0), var float[] tP=array.new_float(0)
var float[] t1P=array.new_float(0)
var int[] oT=array.new_int(0), var int[] cT=array.new_int(0)
var int[] sd=array.new_int(0), var int[] wn=array.new_int(0)
var int[] er=array.new_int(0), var int[] md=array.new_int(0)
var int[] tap1T=array.new_int(0), var float[] tap1P=array.new_float(0)
var int[] tap2T=array.new_int(0), var float[] tap2P=array.new_float(0)
var int[] tap3T=array.new_int(0), var float[] tap3P=array.new_float(0)
var float[] rHi=array.new_float(0), var float[] rLo=array.new_float(0)

{loader_defs}

if barstate.isfirst
{init_calls}

mN(m) => m==1?"M1":m==2?"M2":m==3?"M3":m==4?"M1f":m==5?"M1C":m==6?"M2C":m==7?"M2E":"?"
eN(e) => e==0?"SL":e==1?"BE":e==2?"Trail":e==3?"TP":"End"

t = time
hb = 1800000

for i = 0 to N - 1
    _md=array.get(md,i), _sd=array.get(sd,i)
    if fModel!=0 and _md!=fModel
        continue
    if fDir==1 and _sd!=1
        continue
    if fDir==2 and _sd!=-1
        continue

    _oT=array.get(oT,i), _cT=array.get(cT,i)
    _eP=array.get(eP,i), _xP=array.get(xP,i)
    _sP=array.get(sP,i), _tP=array.get(tP,i), _t1=array.get(t1P,i)
    _wn=array.get(wn,i), _er=array.get(er,i)
    _rH=array.get(rHi,i), _rL=array.get(rLo,i)
    _rS=_rH>0 and _rL>0?_rH-_rL:_eP*0.005
    _ng=_rS*0.15
    dC=_sd==1?cL:cS

    // -- Taps: placed on the actual 1h candle's wick at the tap time --
    // Accumulation (bullish): taps touch support -> label below candle low
    // Distribution (bearish): taps touch resistance -> label above candle high
    if showTaps
        _a1T=array.get(tap1T,i)
        _a2T=array.get(tap2T,i)
        _a3T=array.get(tap3T,i)
        // +-2 bar tolerance for TF alignment (4h tap -> nearest 1h bar)
        _tol = hb * 4
        if _sd == 1
            // Accumulation: label_up below the candle low
            if _a1T>0 and t>=_a1T-_tol and t<=_a1T+_tol
                label.new(bar_index, low, text="T1", style=label.style_label_up, color=cTp1, textcolor=color.black, size=size.tiny)
            if _a2T>0 and t>=_a2T-_tol and t<=_a2T+_tol
                label.new(bar_index, low, text="T2", style=label.style_label_up, color=cTp2, textcolor=color.black, size=size.tiny)
            if _a3T>0 and t>=_a3T-_tol and t<=_a3T+_tol
                label.new(bar_index, low, text="T3", style=label.style_label_up, color=cTp3, textcolor=color.black, size=size.tiny)
        else
            // Distribution: label_down above the candle high
            if _a1T>0 and t>=_a1T-_tol and t<=_a1T+_tol
                label.new(bar_index, high, text="T1", style=label.style_label_down, color=cTp1, textcolor=color.black, size=size.tiny)
            if _a2T>0 and t>=_a2T-_tol and t<=_a2T+_tol
                label.new(bar_index, high, text="T2", style=label.style_label_down, color=cTp2, textcolor=color.black, size=size.tiny)
            if _a3T>0 and t>=_a3T-_tol and t<=_a3T+_tol
                label.new(bar_index, high, text="T3", style=label.style_label_down, color=cTp3, textcolor=color.black, size=size.tiny)

    // -- Range + Fibs --
    _t1R=array.get(tap1T,i)
    if _rH>0 and _rL>0 and _t1R>0
        if showRange and t>=_t1R and t<=_t1R+hb
            box.new(_t1R,_rH,_oT,_rL,xloc=xloc.bar_time,border_color=cRn,border_width=1,bgcolor=cRB)
        if showFibs and t>=_t1R and t<=_t1R+hb
            _d=_rH-_rL
            line.new(_t1R,_rL+_d*0.236,_oT,_rL+_d*0.236,xloc=xloc.bar_time,color=cF2,style=line.style_dotted,width=1)
            line.new(_t1R,_rL+_d*0.382,_oT,_rL+_d*0.382,xloc=xloc.bar_time,color=cF3,style=line.style_dotted,width=1)
            line.new(_t1R,_rL+_d*0.5,_oT,_rL+_d*0.5,xloc=xloc.bar_time,color=cF5,style=line.style_dotted,width=1)
            line.new(_t1R,_rL+_d*0.618,_oT,_rL+_d*0.618,xloc=xloc.bar_time,color=cF6,style=line.style_dotted,width=1)
            line.new(_t1R,_rL+_d*0.786,_oT,_rL+_d*0.786,xloc=xloc.bar_time,color=cF7,style=line.style_dotted,width=1)
            label.new(_oT,_rL+_d*0.236,text=".236",xloc=xloc.bar_time,style=label.style_none,textcolor=cF2,size=size.tiny)
            label.new(_oT,_rL+_d*0.382,text=".382",xloc=xloc.bar_time,style=label.style_none,textcolor=cF3,size=size.tiny)
            label.new(_oT,_rL+_d*0.5,text=".5",xloc=xloc.bar_time,style=label.style_none,textcolor=cF5,size=size.tiny)
            label.new(_oT,_rL+_d*0.618,text=".618",xloc=xloc.bar_time,style=label.style_none,textcolor=cF6,size=size.tiny)
            label.new(_oT,_rL+_d*0.786,text=".786",xloc=xloc.bar_time,style=label.style_none,textcolor=cF7,size=size.tiny)

    // -- Model --
    if showModel and t>=_oT-hb and t<=_oT+hb
        _mY=_sd==1?(_rL>0?_rL-_ng*2:_eP-_ng*3):(_rH>0?_rH+_ng*2:_eP+_ng*3)
        label.new(_oT,_mY,text=mN(_md)+" "+(_sd==1?"LONG":"SHORT"),xloc=xloc.bar_time,style=label.style_none,textcolor=cMd,size=size.normal)

    // -- Entry --
    if showEntry and t>=_oT-hb and t<=_oT+hb
        label.new(_oT,_eP,text=str.tostring(_eP,"#.#"),xloc=xloc.bar_time,style=_sd==1?label.style_label_up:label.style_label_down,color=dC,textcolor=color.white,size=size.small)

    // -- Exit --
    if showExit and t>=_cT-hb and t<=_cT+hb
        _p=_sd==1?(_xP-_eP)/_eP*100:(_eP-_xP)/_eP*100
        _ps=showPnL?" "+(_p>=0?"+":"")+str.tostring(_p,"#.##")+"%":""
        label.new(_cT,_xP,text=eN(_er)+_ps,xloc=xloc.bar_time,style=_wn==1?label.style_label_down:label.style_label_up,color=_wn==1?cW:cX,textcolor=color.white,size=size.tiny)

    // -- Lines --
    if showLines and t>=_oT and t<=_oT+hb
        line.new(_oT,_eP,_cT,_eP,xloc=xloc.bar_time,color=dC,style=line.style_solid,width=2)
        line.new(_oT,_sP,_cT,_sP,xloc=xloc.bar_time,color=cSL,style=line.style_dashed,width=1)
        line.new(_oT,_tP,_cT,_tP,xloc=xloc.bar_time,color=cTG,style=line.style_dashed,width=1)
        if _t1>0
            line.new(_oT,_t1,_cT,_t1,xloc=xloc.bar_time,color=cT1c,style=line.style_dotted,width=1)
"""


def main():
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, entry_price, exit_price, tp1_price, stop_price, target_price,
                       direction, exit_reason, is_win, pnl_dollars, model,
                       opened_at, closed_at, tp1_hit, rr, timeframe
                FROM backtest_trades WHERE run_id = %s ORDER BY opened_at
            """, (RUN_ID,))
            tcols = [d[0] for d in cur.description]
            trades = [dict(zip(tcols, r)) for r in cur.fetchall()]

            cur.execute("""
                SELECT signal_time, entry_price, direction, model, timeframe, schematic_json
                FROM backtest_signals
                WHERE run_id = %s AND final_decision = 'TAKE' AND schematic_json IS NOT NULL
                ORDER BY signal_time
            """, (RUN_ID,))
            signals = cur.fetchall()

            # Build signal lookup
            sig_lookup = {}
            for s in signals:
                sig_time, sig_entry, sig_dir, sig_model, sig_tf, sj = s
                if sj is None:
                    continue
                if isinstance(sj, str):
                    sj = json.loads(sj)
                key = (round(float(sig_entry), 0), sig_dir, sig_tf)
                if key not in sig_lookup:
                    sig_lookup[key] = (sj, sig_time, sig_tf)

            # Resolve per-trade data with price-matched tap timestamps
            trade_data = []
            matched = 0
            tap_resolved = 0
            for t in trades:
                entry_r = round(float(t['entry_price']), 0)
                key = (entry_r, t['direction'], t['timeframe'])
                match = sig_lookup.get(key)

                taps = {'t1_ts': 0, 't1_p': 0.0, 't2_ts': 0, 't2_p': 0.0,
                        't3_ts': 0, 't3_p': 0.0, 'rng_hi': 0.0, 'rng_lo': 0.0}

                if match:
                    matched += 1
                    sj, sig_time, sig_tf = match

                    rh = sj.get("range_high")
                    rl = sj.get("range_low")
                    if rh is not None and rl is not None:
                        taps['rng_hi'] = float(rh)
                        taps['rng_lo'] = float(rl)

                    # Resolve tap timestamps using exact backtest window replication
                    resolved = resolve_tap_times(cur, SYMBOL, sig_tf, sig_time, sj)
                    taps.update(resolved)
                    if any(resolved.get(f't{n}_ts', 0) > 0 for n in [1, 2, 3]):
                        tap_resolved += 1

                entry = float(t['entry_price'])
                exit_p = float(t['exit_price'])
                tp1 = float(t['tp1_price'] or 0)
                stop = float(t['stop_price'])
                target = float(t['target_price'])
                open_ts = int(t['opened_at'].timestamp() * 1000)
                close_ts = int(t['closed_at'].timestamp() * 1000)
                side = 1 if t['direction'] == 'bullish' else -1
                w = 1 if t['is_win'] else 0
                rmap = {'stop_hit': 0, 'breakeven_after_tp1': 1, 'trailing_stop': 2,
                        'target_hit': 3, 'backtest_end': 4}
                r = rmap.get(t['exit_reason'], 0)
                mmap = {'Model_1': 1, 'Model_2': 2, 'Model_3': 3,
                        'Model_1_from_M2_failure': 4, 'Model_1_CONTINUATION': 5,
                        'Model_2_CONTINUATION': 6, 'Model_2_EXT': 7}
                m = mmap.get(t['model'], 0)
                trade_data.append({
                    'eP': entry, 'xP': exit_p, 'sP': stop, 'tP': target, 't1P': tp1,
                    'oT': open_ts, 'cT': close_ts, 'sd': side, 'wn': w, 'er': r, 'md': m,
                    **taps
                })

    total = len(trade_data)
    print(f"Trades: {total}, Signal matches: {matched}/{total}, Taps resolved: {tap_resolved}/{matched}")

    num_parts = (total + MAX_TRADES_PER_SCRIPT - 1) // MAX_TRADES_PER_SCRIPT
    print(f"Splitting into {num_parts} script(s)")

    for pi in range(num_parts):
        ps = pi * MAX_TRADES_PER_SCRIPT
        pe = min(ps + MAX_TRADES_PER_SCRIPT, total)
        part = trade_data[ps:pe]
        n = len(part)
        pnum = pi + 1
        tr = f"#{ps+1}-{pe}"
        script = build_script(part, n, pnum, num_parts, total, tr)
        suffix = f"_pt{pnum}" if num_parts > 1 else ""
        out_path = os.path.join(os.path.dirname(__file__), '..', f'BTCUSDT_run40_trades{suffix}.pine')
        out_path = os.path.abspath(out_path)
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(script)
        print(f"  Part {pnum}: {out_path} ({script.count(chr(10))} lines, {len(script):,} chars)")


if __name__ == "__main__":
    main()
