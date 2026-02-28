############# SMA Bot #1 — Simulated Trading via Phemex Live Data
#
# Strategy: Daily 20-SMA bias + 15m 20-SMA entry levels
# Data: Live from Phemex (read-only, no real orders placed)
# Execution: All trades are simulated in-memory and logged to JSON
#
# Original: moondev coding bot #1 (Phemex live)
# Modified: Simulation mode using Phemex market data

import ccxt
import pandas as pd
import numpy as np
import json
import os
import dontshare_config as ds
from datetime import datetime, timezone
import time
import schedule

# ================================================================
# PHEMEX CONNECTION (read-only — used for market data only)
# ================================================================
phemex = ccxt.phemex({
    'enableRateLimit': True,
    'apiKey': ds.xP_hmv_KEY,
    'secret': ds.xP_hmv_SECRET,
})

# ================================================================
# CONFIGURATION
# ================================================================
symbol = 'uBTCUSD'
pos_size = 30           # total position size in contracts
target = 8              # take profit at 8% PnL
max_loss = -9           # stop loss at -9% PnL
vol_decimal = .4        # OB volume ratio threshold for exit gating
leverage = 10           # simulated leverage multiplier
starting_balance = 1000 # simulated starting balance in USD
cooldown_minutes = 59   # minutes to wait between closing and re-entering

SIM_LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sim_trade_log.json')

# ================================================================
# SIMULATED POSITION TRACKER
# ================================================================
class SimulatedTrader:
    """Tracks a virtual trading account — no real orders are ever placed."""

    def __init__(self):
        self.balance = starting_balance
        self.position = None   # None = flat, or dict with trade details
        self.trade_history = []
        self.pending_orders = []  # simulated limit orders waiting to fill
        self.last_close_time = None
        self._load()

    # --- persistence ---

    def _load(self):
        if os.path.exists(SIM_LOG_PATH):
            try:
                with open(SIM_LOG_PATH, 'r') as f:
                    data = json.load(f)
                self.balance = data.get('balance', starting_balance)
                self.position = data.get('position')
                self.trade_history = data.get('trade_history', [])
                self.last_close_time = data.get('last_close_time')
                print(f'[SIM] Loaded state — balance=${self.balance:.2f}, trades={len(self.trade_history)}, in_pos={self.position is not None}')
            except Exception as e:
                print(f'[SIM] Could not load state: {e}')

    def save(self):
        try:
            data = {
                'balance': round(self.balance, 2),
                'position': self.position,
                'trade_history': self.trade_history,
                'last_close_time': self.last_close_time,
            }
            with open(SIM_LOG_PATH, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            print(f'[SIM] Failed to save state: {e}')

    # --- position queries (drop-in replacements for Phemex account calls) ---

    @property
    def in_position(self):
        return self.position is not None

    @property
    def is_long(self):
        if self.position is None:
            return None
        return self.position['side'] == 'long'

    @property
    def position_size(self):
        if self.position is None:
            return 0
        return self.position['size']

    @property
    def entry_price(self):
        if self.position is None:
            return 0
        return self.position['entry_price']

    # --- simulated order handling ---

    def place_limit_order(self, side, size, price):
        """Queue a simulated limit order (not filled yet)."""
        order = {
            'side': side,         # 'buy' or 'sell'
            'size': size,
            'price': price,
            'placed_at': datetime.now(timezone.utc).isoformat(),
        }
        self.pending_orders.append(order)
        print(f'[SIM ORDER] {side.upper()} {size} {symbol} @ ${price:.2f} (limit, pending)')
        return order

    def cancel_all_orders(self):
        """Cancel all pending simulated orders."""
        count = len(self.pending_orders)
        self.pending_orders = []
        if count > 0:
            print(f'[SIM] Cancelled {count} pending order(s)')

    def check_pending_fills(self, ask, bid):
        """Check if any pending limit orders would have filled at current prices.

        Buy limit fills when ask <= order price (market came down to us).
        Sell limit fills when bid >= order price (market came up to us).
        """
        if not self.pending_orders:
            return

        filled = []
        remaining = []

        for order in self.pending_orders:
            if order['side'] == 'buy' and ask <= order['price']:
                filled.append(order)
            elif order['side'] == 'sell' and bid >= order['price']:
                filled.append(order)
            else:
                remaining.append(order)

        self.pending_orders = remaining

        for order in filled:
            self._fill_order(order, ask if order['side'] == 'buy' else bid)

    def _fill_order(self, order, fill_price):
        """Process a simulated fill — open or add to position."""
        side = order['side']
        size = order['size']

        if self.position is None:
            # opening new position
            pos_side = 'long' if side == 'buy' else 'short'
            self.position = {
                'side': pos_side,
                'size': size,
                'entry_price': fill_price,
                'opened_at': datetime.now(timezone.utc).isoformat(),
                'fills': [{'price': fill_price, 'size': size}],
            }
            print(f'[SIM FILL] Opened {pos_side.upper()} {size} {symbol} @ ${fill_price:.2f}')
        else:
            # adding to existing position — compute weighted average entry
            old_size = self.position['size']
            old_entry = self.position['entry_price']
            new_size = old_size + size
            new_entry = ((old_entry * old_size) + (fill_price * size)) / new_size
            self.position['size'] = new_size
            self.position['entry_price'] = new_entry
            self.position['fills'].append({'price': fill_price, 'size': size})
            print(f'[SIM FILL] Added {size} to {self.position["side"].upper()} — total {new_size} @ avg ${new_entry:.2f}')

        self.save()

    def close_position(self, exit_price, reason='manual'):
        """Close the entire simulated position and record PnL."""
        if self.position is None:
            print('[SIM] No position to close')
            return

        side = self.position['side']
        entry = self.position['entry_price']
        size = self.position['size']

        if side == 'long':
            diff = exit_price - entry
        else:
            diff = entry - exit_price

        pnl_pct = (diff / entry) * leverage * 100
        # approximate USD PnL: for inverse contracts (uBTCUSD), size is in contracts
        # simplified: treat PnL% as applied to a notional based on current balance allocation
        notional = (size / pos_size) * self.balance * 0.5  # rough allocation
        pnl_usd = notional * (pnl_pct / 100)

        self.balance += pnl_usd

        trade_record = {
            'id': len(self.trade_history) + 1,
            'side': side,
            'size': size,
            'entry_price': entry,
            'exit_price': exit_price,
            'pnl_pct': round(pnl_pct, 2),
            'pnl_usd': round(pnl_usd, 2),
            'balance_after': round(self.balance, 2),
            'reason': reason,
            'opened_at': self.position.get('opened_at'),
            'closed_at': datetime.now(timezone.utc).isoformat(),
            'fills': self.position.get('fills', []),
        }
        self.trade_history.append(trade_record)
        self.last_close_time = time.time()

        is_win = pnl_pct > 0
        emoji = ':)' if is_win else ':('
        print(f'[SIM CLOSE] {emoji} {side.upper()} {size} {symbol} | entry=${entry:.2f} exit=${exit_price:.2f} | PnL={pnl_pct:+.2f}% (${pnl_usd:+.2f}) | reason={reason}')
        print(f'[SIM] Balance: ${self.balance:.2f} | Total trades: {len(self.trade_history)}')

        self.position = None
        self.cancel_all_orders()
        self.save()

    def print_summary(self):
        """Print a quick performance summary."""
        total = len(self.trade_history)
        if total == 0:
            print('[SIM] No trades yet')
            return
        wins = sum(1 for t in self.trade_history if t['pnl_pct'] > 0)
        losses = total - wins
        total_pnl = sum(t['pnl_usd'] for t in self.trade_history)
        win_rate = (wins / total) * 100
        print(f'[SIM SUMMARY] Trades={total} | Wins={wins} Losses={losses} | Win Rate={win_rate:.1f}% | Total PnL=${total_pnl:+.2f} | Balance=${self.balance:.2f}')


# global sim trader instance
sim = SimulatedTrader()


# ================================================================
# MARKET DATA FUNCTIONS (live Phemex — read only)
# ================================================================

def ask_bid():
    """Fetch current bid/ask from Phemex order book."""
    ob = phemex.fetch_order_book(symbol)
    bid = ob['bids'][0][0]
    ask = ob['asks'][0][0]
    return ask, bid  # ask_bid()[0] = ask, [1] = bid


def daily_sma():
    """Compute daily 20-SMA and generate BUY/SELL signal."""
    print('starting daily sma...')

    bars = phemex.fetch_ohlcv(symbol, timeframe='1d', limit=100)
    df_d = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df_d['timestamp'] = pd.to_datetime(df_d['timestamp'], unit='ms')

    df_d['sma20_d'] = df_d.close.rolling(20).mean()

    bid = ask_bid()[1]

    # if sma > bid = SELL, if sma < bid = BUY
    df_d.loc[df_d['sma20_d'] > bid, 'sig'] = 'SELL'
    df_d.loc[df_d['sma20_d'] < bid, 'sig'] = 'BUY'

    return df_d


def f15_sma():
    """Compute 15m 20-SMA and derive entry price levels."""
    print('starting 15 min sma...')

    bars = phemex.fetch_ohlcv(symbol, timeframe='15m', limit=100)
    df_f = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df_f['timestamp'] = pd.to_datetime(df_f['timestamp'], unit='ms')

    df_f['sma20_15'] = df_f.close.rolling(20).mean()

    # entry prices: buy near/below 15m SMA, sell near/above
    df_f['bp_1'] = df_f['sma20_15'] * 1.001   # 0.1% above SMA
    df_f['bp_2'] = df_f['sma20_15'] * 0.997    # 0.3% below SMA
    df_f['sp_1'] = df_f['sma20_15'] * 0.999    # 0.1% below SMA
    df_f['sp_2'] = df_f['sma20_15'] * 1.003    # 0.3% above SMA

    return df_f


def ob():
    """Sample order book volume over ~55 seconds to gauge bid/ask control.

    Returns True if volume ratio is under the vol_decimal threshold (weak control),
    meaning it's safe to exit if target is hit.
    """
    print('fetching order book data...')

    rows = []

    for x in range(11):
        book = phemex.fetch_order_book(symbol)
        bids = book['bids']
        asks = book['asks']

        sum_bidvol = sum(level[1] for level in bids)
        sum_askvol = sum(level[1] for level in asks)

        rows.append({'bid_vol': sum_bidvol, 'ask_vol': sum_askvol})

        if x < 10:  # don't sleep after last sample
            time.sleep(5)

    df = pd.DataFrame(rows)

    total_bidvol = df['bid_vol'].sum()
    total_askvol = df['ask_vol'].sum()
    print(f'last ~1m total Bid Vol: {total_bidvol} | Ask Vol: {total_askvol}')

    if total_bidvol > total_askvol:
        control_dec = total_askvol / total_bidvol
        print(f'Bulls in control: ratio={control_dec:.3f}')
    else:
        control_dec = total_bidvol / total_askvol
        print(f'Bears in control: ratio={control_dec:.3f}')

    vol_under_dec = control_dec < vol_decimal
    print(f'vol_under_dec={vol_under_dec} (threshold={vol_decimal})')

    return vol_under_dec


# ================================================================
# SIMULATED POSITION MANAGEMENT
# ================================================================

def open_positions():
    """Query simulated position state.
    Returns: (positions_info, in_position_bool, size, is_long)
    """
    if sim.in_position:
        return (sim.position, True, sim.position_size, sim.is_long)
    else:
        return (None, False, 0, None)


def kill_switch():
    """Close simulated position immediately at current market price."""
    print('[SIM] Starting kill switch...')

    if not sim.in_position:
        print('[SIM] No position to close')
        return

    ask = ask_bid()[0]
    bid = ask_bid()[1]

    # close at market: longs sell at bid, shorts buy at ask
    if sim.is_long:
        exit_price = bid
    else:
        exit_price = ask

    sim.close_position(exit_price, reason='kill_switch')


def sleep_on_close():
    """Enforce cooldown period after closing a trade.
    If we closed a trade within the last `cooldown_minutes` minutes, sleep 60s.
    """
    if sim.last_close_time is None:
        print('[SIM] No previous close, skipping cooldown')
        return

    elapsed_min = (time.time() - sim.last_close_time) / 60

    if elapsed_min < cooldown_minutes:
        print(f'[SIM] Only {elapsed_min:.1f}m since last close (cooldown={cooldown_minutes}m), sleeping 60s...')
        time.sleep(60)
    else:
        print(f'[SIM] {elapsed_min:.1f}m since last close, no cooldown needed')


def pnl_close():
    """Check simulated PnL and trigger exit if target/stop hit.
    Returns: (pnlclose, in_pos, size, long)
    """
    print('checking simulated PnL...')

    pnlclose = False
    in_pos = False
    size = 0
    long = None

    if not sim.in_position:
        print('[SIM] Not in position')
        return pnlclose, in_pos, size, long

    in_pos = True
    long = sim.is_long
    size = sim.position_size
    entry_price = sim.entry_price
    current_price = ask_bid()[1]

    if long:
        diff = current_price - entry_price
    else:
        diff = entry_price - current_price

    if entry_price > 0:
        perc = ((diff / entry_price) * leverage) * 100
    else:
        perc = 0

    print(f'[SIM PnL] {"LONG" if long else "SHORT"} | entry=${entry_price:.2f} | current=${current_price:.2f} | PnL={perc:+.2f}%')

    if perc > 0:
        print(f'[SIM] In a winning position ({perc:+.2f}%)')
        if perc > target:
            print(f'[SIM] Target hit ({perc:.2f}% > {target}%)! Checking OB volume...')
            pnlclose = True
            vol_under_dec = ob()
            if vol_under_dec:
                print(f'[SIM] Volume under threshold ({vol_decimal}), holding for 30s...')
                time.sleep(30)
            else:
                print(f'[SIM] Volume clear — closing at target')
                exit_price = ask_bid()[1] if long else ask_bid()[0]
                sim.close_position(exit_price, reason=f'target_hit_{perc:.1f}pct')
        else:
            print(f'[SIM] Target not hit yet ({perc:.2f}% < {target}%)')

    elif perc < 0:
        if perc <= max_loss:
            print(f'[SIM] Stop loss triggered ({perc:.2f}% <= {max_loss}%)')
            exit_price = ask_bid()[1] if long else ask_bid()[0]
            sim.close_position(exit_price, reason=f'stop_loss_{perc:.1f}pct')
        else:
            print(f'[SIM] In drawdown ({perc:.2f}%) but within tolerance (max_loss={max_loss}%)')

    return pnlclose, in_pos, size, long


# ================================================================
# MAIN BOT LOOP
# ================================================================

def bot():
    print('=' * 60)
    print(f'[BOT] Cycle at {datetime.now(timezone.utc).isoformat()}')
    print('=' * 60)

    # check fills on pending orders
    ask = ask_bid()[0]
    bid = ask_bid()[1]
    sim.check_pending_fills(ask, bid)

    # check PnL / exit conditions
    pnl_close()

    # cooldown after recent close
    sleep_on_close()

    # compute signals
    df_d = daily_sma()
    df_f = f15_sma()

    ask = ask_bid()[0]
    bid = ask_bid()[1]

    sig = df_d.iloc[-1]['sig']
    print(f'[BOT] Daily SMA signal: {sig} | bid=${bid}')

    open_size = pos_size / 2

    # check simulated position
    in_pos = sim.in_position
    curr_size = sim.position_size

    curr_p = bid
    last_sma15 = df_f.iloc[-1]['sma20_15']

    if (not in_pos) and (curr_size < pos_size):

        # cancel any stale pending orders before placing new ones
        sim.cancel_all_orders()

        if (sig == 'BUY') and (curr_p > last_sma15):
            bp_1 = df_f.iloc[-1]['bp_1']
            bp_2 = df_f.iloc[-1]['bp_2']
            print(f'[BOT] BUY signal — placing limit orders at bp_1=${bp_1:.2f}, bp_2=${bp_2:.2f}')

            sim.place_limit_order('buy', open_size, bp_1)
            sim.place_limit_order('buy', open_size, bp_2)

            print('[BOT] Orders placed, sleeping 2min...')
            time.sleep(120)

        elif (sig == 'SELL') and (curr_p < last_sma15):
            sp_1 = df_f.iloc[-1]['sp_1']
            sp_2 = df_f.iloc[-1]['sp_2']
            print(f'[BOT] SELL signal — placing limit orders at sp_1=${sp_1:.2f}, sp_2=${sp_2:.2f}')

            sim.place_limit_order('sell', open_size, sp_1)
            sim.place_limit_order('sell', open_size, sp_2)

            print('[BOT] Orders placed, sleeping 2min...')
            time.sleep(120)

        else:
            print(f'[BOT] No valid entry — sig={sig}, price vs SMA mismatch. Sleeping 10min...')
            time.sleep(600)
    else:
        print(f'[BOT] Already in position (size={curr_size}) or at max. Monitoring...')

    # print running summary every cycle
    sim.print_summary()


# ================================================================
# ENTRY POINT
# ================================================================
if __name__ == '__main__':
    print('[SIM BOT] Starting simulated SMA bot on Phemex data...')
    print(f'[SIM BOT] Symbol={symbol} | Size={pos_size} | Target={target}% | StopLoss={max_loss}% | Leverage={leverage}x')
    print(f'[SIM BOT] Starting balance: ${starting_balance}')
    print(f'[SIM BOT] Trade log: {SIM_LOG_PATH}')
    sim.print_summary()

    schedule.every(28).seconds.do(bot)

    while True:
        try:
            schedule.run_pending()
        except KeyboardInterrupt:
            print('\n[SIM BOT] Shutting down...')
            sim.print_summary()
            sim.save()
            break
        except Exception as e:
            print(f'[SIM BOT] Error: {e}')
            time.sleep(30)
