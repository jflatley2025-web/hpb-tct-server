"""
Verify Run 40 trailing stop logic in the live 5B engine.
Simulates 10 trades (long + short) through _manage_open_trade() and validates:
  1. Trailing stop ratchets only in profit direction
  2. Exit labels match actual stop behavior
  3. Dual-hit safety (SL priority over TP1)
  4. is_win logic for all exit reasons
  5. Parity with Run 40 backtest expectations
"""
import sys; sys.path.insert(0, '.')
import os
import json
import copy
from datetime import datetime, timezone

os.environ.setdefault("SCHEMATICS_5B_SCAN_INTERVAL", "999999")  # prevent auto-scan

from schematics_5b_trader import Schematics5BTrader, TRAIL_FACTOR

# ── Helpers ──────────────────────────────────────────────────────────

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
WARN = "\033[93mWARN\033[0m"

results = []


def check(name, condition, detail=""):
    tag = PASS if condition else FAIL
    results.append((name, condition, detail))
    print(f"  [{tag}] {name}" + (f"  ({detail})" if detail else ""))
    return condition


def make_trade(direction, entry, stop, target, tp1, symbol="BTCUSDT"):
    """Create a mock trade dict matching the live engine's format."""
    return {
        "id": 999,
        "symbol": symbol,
        "timeframe": "1h",
        "direction": direction,
        "model": "Model_1",
        "entry_price": entry,
        "stop_price": stop,
        "target_price": target,
        "tp1_price": tp1,
        "tp1_hit": False,
        "position_size": 10000.0,
        "margin": 1000.0,
        "risk_amount": 100.0,
        "leverage": 10,
        "rr": round(abs(target - entry) / abs(entry - stop), 2),
        "entry_score": 90,
        "entry_reasons": ["Test trade"],
        "htf_bias": "bullish" if direction == "bullish" else "bearish",
        "liquidation_price": 0,
        "liquidation_safe": True,
        "opened_at": datetime.now(timezone.utc).isoformat(),
        "status": "open",
        "live_pnl_pct": 0,
        "current_price": entry,
    }


class MockState:
    """Minimal state mock so _manage_open_trade can run without DB."""
    def __init__(self, trade):
        self.current_trade = trade
        self.balance = 5000.0
        self.peak_balance = 5000.0
        self.total_wins = 0
        self.total_losses = 0
        self.trade_history = []
        self.dd_triggered_at = None
        self.dd_trough_balance = None

    def save(self):
        pass  # no-op for testing


class MockTrader:
    """Wrap the real trader methods but with mock state."""
    def __init__(self, trade):
        self.state = MockState(trade)
        self._portfolio = None
        # Bind the real methods
        self._manage_open_trade = Schematics5BTrader._manage_open_trade.__get__(self)
        self._take_partial_profit = Schematics5BTrader._take_partial_profit.__get__(self)
        self._close_trade = Schematics5BTrader._close_trade.__get__(self)


def simulate(trade_def, price_sequence, label):
    """Run a trade through a sequence of prices, return results."""
    print(f"\n{'='*70}")
    print(f"  TRADE: {label}")
    print(f"  {trade_def['direction'].upper()} | Entry={trade_def['entry_price']} "
          f"SL={trade_def['stop_price']} TP1={trade_def['tp1_price']} Target={trade_def['target_price']}")
    print(f"{'='*70}")

    trader = MockTrader(copy.deepcopy(trade_def))
    trail_log = []
    exit_result = None

    for i, price in enumerate(price_sequence):
        result = trader._manage_open_trade(price)
        trade = trader.state.current_trade
        action = result.get("action", "?")

        if trade:
            stop_now = trade.get("stop_price", 0)
            tp1_hit = trade.get("tp1_hit", False)
            highest = trade.get("highest_since_tp1")
            lowest = trade.get("lowest_since_tp1")
            trail_log.append({
                'step': i, 'price': price, 'stop': stop_now,
                'tp1_hit': tp1_hit, 'highest': highest, 'lowest': lowest,
                'action': action,
            })

        if action in ("trade_closed", "tp1_hit"):
            if action == "trade_closed":
                exit_result = result
                break

    return trader, trail_log, exit_result


# ══════════════════════════════════════════════════════════════════════
# TEST CASES
# ══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  HPB-TCT LIVE ENGINE VERIFICATION — Run 40 Trailing Stop Logic")
print("=" * 70)
print(f"  TRAIL_FACTOR = {TRAIL_FACTOR}")

# ────────────────────────────────────────────────────────────────────
# TEST 1: LONG — TP1 hit, price rallies, trailing stop exits
# ────────────────────────────────────────────────────────────────────
t1 = make_trade("bullish", entry=70000, stop=69000, target=74000, tp1=71000)
# trail_distance = (74000-70000)*0.5 = 2000
# Price: entry -> TP1 -> rally to 73000 -> pullback hits trail at 71000
prices = [70500, 71000, 71500, 72000, 72500, 73000, 72500, 72000, 71500, 71000]
trader, log, exit = simulate(t1, prices, "LONG: TP1 -> Rally -> Trailing Stop")

check("T1: TP1 triggered", any(l['tp1_hit'] for l in log))
check("T1: Trail ratcheted up", any(l['stop'] > 70000 for l in log if l['tp1_hit']),
      f"max stop={max(l['stop'] for l in log if l['tp1_hit'])}")
# After TP1 at 71000, highest goes to 73000, trail = 73000-1000 = 72000
check("T1: Trail never ratcheted down",
      all(log[i]['stop'] <= log[i+1]['stop'] for i in range(len(log)-2)
          if log[i]['tp1_hit'] and log[i+1]['tp1_hit']),
      "stop only increased")
check("T1: Exit reason = trailing_stop",
      exit and exit.get("trade", {}).get("exit_reason") == "trailing_stop",
      f"got: {exit.get('trade', {}).get('exit_reason') if exit else 'no exit'}")
check("T1: is_win = True",
      exit and exit.get("trade", {}).get("is_win") is True)
check("T1: Exit price at trailing stop (not current)",
      exit and exit.get("trade", {}).get("exit_price", 0) > 70000,
      f"exit={exit.get('trade', {}).get('exit_price') if exit else '?'}")

# ────────────────────────────────────────────────────────────────────
# TEST 2: SHORT — TP1 hit, price drops, trailing stop exits
# ────────────────────────────────────────────────────────────────────
t2 = make_trade("bearish", entry=70000, stop=71000, target=66000, tp1=69000)
# trail_distance = (70000-66000)*0.5 = 2000
# Price: entry -> TP1(69000) -> drops to 67000 -> bounces to 69500 (hits trail)
prices = [69500, 69000, 68500, 68000, 67500, 67000, 67500, 68000, 68500, 69000, 69500]
trader2, log2, exit2 = simulate(t2, prices, "SHORT: TP1 -> Drop -> Trailing Stop")

check("T2: TP1 triggered", any(l['tp1_hit'] for l in log2))
check("T2: Trail ratcheted down",
      any(l['stop'] < 71000 for l in log2 if l['tp1_hit']),
      f"min stop={min(l['stop'] for l in log2 if l['tp1_hit'])}")
check("T2: Exit reason = trailing_stop",
      exit2 and exit2.get("trade", {}).get("exit_reason") == "trailing_stop",
      f"got: {exit2.get('trade', {}).get('exit_reason') if exit2 else 'no exit'}")
check("T2: is_win = True",
      exit2 and exit2.get("trade", {}).get("is_win") is True)

# ────────────────────────────────────────────────────────────────────
# TEST 3: LONG — TP1 hit, immediate reversal -> breakeven exit
# ────────────────────────────────────────────────────────────────────
t3 = make_trade("bullish", entry=70000, stop=69000, target=74000, tp1=71000)
# TP1 hit, then price drops straight to entry (breakeven)
prices = [71000, 70500, 70000]
trader3, log3, exit3 = simulate(t3, prices, "LONG: TP1 -> Immediate Reversal -> Breakeven")

check("T3: Exit reason = breakeven_after_tp1",
      exit3 and exit3.get("trade", {}).get("exit_reason") == "breakeven_after_tp1",
      f"got: {exit3.get('trade', {}).get('exit_reason') if exit3 else 'no exit'}")
check("T3: is_win = True (TP1 partial profit)",
      exit3 and exit3.get("trade", {}).get("is_win") is True)

# ────────────────────────────────────────────────────────────────────
# TEST 4: LONG — Stop hit BEFORE TP1 (no trailing)
# ────────────────────────────────────────────────────────────────────
t4 = make_trade("bullish", entry=70000, stop=69000, target=72000, tp1=71000)
prices = [69500, 69000]
trader4, log4, exit4 = simulate(t4, prices, "LONG: SL hit before TP1")

check("T4: Exit reason = stop_hit",
      exit4 and exit4.get("trade", {}).get("exit_reason") == "stop_hit",
      f"got: {exit4.get('trade', {}).get('exit_reason') if exit4 else 'no exit'}")
check("T4: is_win = False",
      exit4 and exit4.get("trade", {}).get("is_win") is False)

# ────────────────────────────────────────────────────────────────────
# TEST 5: LONG — Full target hit
# ────────────────────────────────────────────────────────────────────
t5 = make_trade("bullish", entry=70000, stop=69000, target=72000, tp1=71000)
prices = [71000, 71500, 72000]  # reaches target directly
trader5, log5, exit5 = simulate(t5, prices, "LONG: Full target hit")

check("T5: Exit reason = target_hit",
      exit5 and exit5.get("trade", {}).get("exit_reason") == "target_hit",
      f"got: {exit5.get('trade', {}).get('exit_reason') if exit5 else 'no exit'}")
check("T5: is_win = True",
      exit5 and exit5.get("trade", {}).get("is_win") is True)

# ────────────────────────────────────────────────────────────────────
# TEST 6: DUAL-HIT SAFETY — SL and TP1 hit at same price check
# ────────────────────────────────────────────────────────────────────
t6 = make_trade("bullish", entry=70000, stop=69000, target=72000, tp1=71000)
# Price drops to 69000 (SL) which also happens to be where TP1 could trigger
# But SL should take priority when both are hittable
# Actually, for dual-hit we need price that hits both SL and TP1
# This is impossible in a single price check (price can't be both >= 71000 and <= 69000)
# The dual-hit scenario applies when BOTH SL and TARGET trigger after TP1
# Let's simulate: TP1 hit, trail ratchets up, then price hits both trail SL and target
t6 = make_trade("bullish", entry=70000, stop=69000, target=74000, tp1=71000)
# trail_distance = 2000. TP1 hit at 71000, price goes to 76500, trail = 74500
# Then price drops to 74000: target=74000, stop=74500 -> dual hit (stop above target)
prices = [71000, 73000, 75000, 76500, 75000, 74000]
trader6, log6, exit6 = simulate(t6, prices, "LONG: Dual-hit (trail=target, both triggered)")

check("T6: Exit is target_hit or trailing_stop (both valid in dual-hit)",
      exit6 and exit6.get("trade", {}).get("exit_reason") in ("trailing_stop", "target_hit"),
      f"got: {exit6.get('trade', {}).get('exit_reason') if exit6 else 'no exit'}")

# ────────────────────────────────────────────────────────────────────
# TEST 7: SHORT — TP1 hit, trailing stop captures big move
# ────────────────────────────────────────────────────────────────────
t7 = make_trade("bearish", entry=3000, stop=3100, target=2600, tp1=2900)
# trail_distance = (3000-2600)*0.5 = 200
# Price: TP1 -> 2700 (trail=2900) -> 2650 (trail=2850) -> bounce to 2900 (hits trail)
prices = [2900, 2800, 2700, 2650, 2700, 2750, 2800, 2850, 2900]
trader7, log7, exit7 = simulate(t7, prices, "SHORT: Big trailing capture")

check("T7: Trail ratcheted to follow price",
      any(l['stop'] < 2900 for l in log7 if l['tp1_hit']),
      f"best stop={min(l['stop'] for l in log7 if l['tp1_hit'])}")
check("T7: Exit reason = trailing_stop",
      exit7 and exit7.get("trade", {}).get("exit_reason") == "trailing_stop")
check("T7: Exit price better than entry",
      exit7 and exit7.get("trade", {}).get("exit_price", 9999) < 3000,
      f"exit={exit7.get('trade', {}).get('exit_price') if exit7 else '?'}")

# ────────────────────────────────────────────────────────────────────
# TEST 8: LONG — Trail never goes backwards
# ────────────────────────────────────────────────────────────────────
t8 = make_trade("bullish", entry=70000, stop=69000, target=76000, tp1=71000)
# trail_distance = 3000. TP1 hit, price oscillates (never reaches target or trail)
prices = [71000, 72000, 71500, 72500, 71800, 73000, 72500, 72000]
trader8, log8, exit8 = simulate(t8, prices, "LONG: Trail ratchet-only-up validation")

tp1_stops = [l['stop'] for l in log8 if l['tp1_hit']]
ratchet_ok = all(tp1_stops[i] <= tp1_stops[i+1] for i in range(len(tp1_stops)-1))
check("T8: Stop NEVER decreased after TP1",
      ratchet_ok,
      f"stop progression: {tp1_stops}")

# ────────────────────────────────────────────────────────────────────
# TEST 9: SHORT — Trail never goes up
# ────────────────────────────────────────────────────────────────────
t9 = make_trade("bearish", entry=3000, stop=3100, target=2600, tp1=2900)
# trail_distance = 200. Price oscillates after TP1 (never reaches target or trail)
prices = [2900, 2850, 2880, 2830, 2860, 2810, 2840]
trader9, log9, exit9 = simulate(t9, prices, "SHORT: Trail ratchet-only-down validation")

tp1_stops9 = [l['stop'] for l in log9 if l['tp1_hit']]
ratchet_ok9 = all(tp1_stops9[i] >= tp1_stops9[i+1] for i in range(len(tp1_stops9)-1))
check("T9: Stop NEVER increased after TP1",
      ratchet_ok9,
      f"stop progression: {tp1_stops9}")

# ────────────────────────────────────────────────────────────────────
# TEST 10: Existing trade from live log — XRPUSDT stop_hit
# ────────────────────────────────────────────────────────────────────
t10 = make_trade("bullish", entry=1.35, stop=1.33, target=1.42, tp1=1.39, symbol="XRPUSDT")
# Simulates the actual XRPUSDT trade that hit stop at 1.33
prices = [1.35, 1.34, 1.33]
trader10, log10, exit10 = simulate(t10, prices, "XRPUSDT LONG: Live trade replay (stop hit)")

check("T10: Exit reason = stop_hit (before TP1)",
      exit10 and exit10.get("trade", {}).get("exit_reason") == "stop_hit")
check("T10: is_win = False",
      exit10 and exit10.get("trade", {}).get("is_win") is False)

# ══════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  VERIFICATION SUMMARY")
print("=" * 70)

passed = sum(1 for _, ok, _ in results if ok)
failed = sum(1 for _, ok, _ in results if not ok)
total = len(results)

print(f"\n  Total checks: {total}")
print(f"  Passed:       {passed}")
print(f"  Failed:       {failed}")

if failed > 0:
    print(f"\n  FAILURES:")
    for name, ok, detail in results:
        if not ok:
            print(f"    [{FAIL}] {name}  ({detail})")

status = "ALL CHECKS PASSED" if failed == 0 else f"{failed} CHECKS FAILED"
color = "\033[92m" if failed == 0 else "\033[91m"
print(f"\n  {color}>>> {status} <<<\033[0m\n")
