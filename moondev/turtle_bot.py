############# Turtle Strategy Bot #4 — Simulated Trading via Phemex Live Data
#
# Strategy: 55-bar High/Low breakout with 2×ATR stop and 0.2% take profit
#   Long:  bid >= 55-bar high AND open was below high → BUY limit @ bid
#   Short: bid <= 55-bar low  AND open was above low  → SELL limit @ bid
#   Exit:  take profit at TAKE_PROFIT_PCT% gain, stop loss at 2×ATR
#   Hours: 9:30am–4:00pm ET, Mon–Fri only. Force-close any open position
#          before market close on Friday.
#
# Timeframe is configurable — TIMEFRAME controls the bar size and is also
# how many bars define "55 bars" (55min on 1m, 55hrs on 1h, etc.)
#
# Data: Live from Phemex (read-only, no real orders placed)
# Execution: All trades are simulated in-memory and logged to JSON

import ccxt
import pandas as pd
import numpy as np
import json
import os
import logging
import time
from datetime import datetime, timezone, time as dtime
from zoneinfo import ZoneInfo
from typing import Optional, Dict, List

logger = logging.getLogger("TURTLE-BOT")

ET = ZoneInfo('America/New_York')

# ================================================================
# CONFIGURATION  (all tunable variables in one place)
# ================================================================
SYMBOL = 'BTC/USD:BTC'
TIMEFRAME = '1m'            # '1m' | '5m' | '15m' | '1h' | '4h'
LOOKBACK_BARS = 55          # classic turtle system uses 55-bar channel
ATR_PERIOD = 14             # bars used to calculate Average True Range
ATR_MULTIPLIER = 2.0        # stop loss = entry ± ATR_MULTIPLIER × ATR
TAKE_PROFIT_PCT = 0.2       # take profit threshold in percent (0.2 = 0.2%)
POS_SIZE = 30               # position size in contracts
LEVERAGE = 10               # simulated leverage multiplier
STARTING_BALANCE = 1000     # simulated starting balance in USD
COOLDOWN_MINUTES = 10       # wait after a close before re-entering
SCAN_INTERVAL = 60          # seconds between scan cycles (original bot ran every 60s)
# Phemex requires discrete limit values; 100 covers 55 bars + ATR lookback
CANDLE_LIMIT = 100

MARKET_OPEN  = dtime(9, 30)   # 9:30 AM ET
MARKET_CLOSE = dtime(16, 0)   # 4:00 PM ET

SIM_LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'turtle_trade_log.json')


# ================================================================
# PHEMEX CONNECTION (lazy init — no side effects at import time)
# ================================================================
_phemex_client: Optional[ccxt.phemex] = None


def _get_phemex() -> ccxt.phemex:
    """Lazy-init Phemex CCXT client. Only public endpoints are used."""
    global _phemex_client
    if _phemex_client is None:
        opts = {'enableRateLimit': True}
        api_key = os.getenv("PHEMEX_API_KEY", "")
        api_secret = os.getenv("PHEMEX_API_SECRET", "")
        if api_key and api_secret:
            opts['apiKey'] = api_key
            opts['secret'] = api_secret
        _phemex_client = ccxt.phemex(opts)
    return _phemex_client


# ================================================================
# TRADING HOURS (ET, Mon–Fri 9:30am–4:00pm)
# ================================================================

def in_trading_hours() -> bool:
    """Return True if current ET time is Mon–Fri between 9:30am and 4:00pm."""
    now = datetime.now(ET)
    if now.weekday() > 4:   # Saturday(5) or Sunday(6)
        return False
    return MARKET_OPEN <= now.time() < MARKET_CLOSE


def end_of_trading_week() -> bool:
    """Return True after 4:00pm ET on Friday — trigger force-close."""
    now = datetime.now(ET)
    return now.weekday() == 4 and now.time() >= MARKET_CLOSE


# ================================================================
# SIMULATED POSITION TRACKER
# ================================================================
class SimulatedTrader:
    """Tracks a virtual trading account — no real orders are ever placed."""

    def __init__(self):
        self.balance = STARTING_BALANCE
        self.position: Optional[Dict] = None
        self.trade_history: List[Dict] = []
        self.pending_orders: List[Dict] = []
        self.last_close_time: Optional[float] = None
        self.last_cycle_time: Optional[str] = None
        self.last_cycle_action: Optional[str] = None
        self.last_bid: Optional[float] = None
        self.last_open_price: Optional[float] = None
        self.last_55bar_high: Optional[float] = None
        self.last_55bar_low: Optional[float] = None
        self.last_atr: Optional[float] = None
        self.last_tp_price: Optional[float] = None    # only set when in position
        self.last_sl_price: Optional[float] = None    # only set when in position
        self.last_in_hours: bool = False
        self.last_error: Optional[str] = None
        self._load()

    def _load(self):
        if os.path.exists(SIM_LOG_PATH):
            try:
                with open(SIM_LOG_PATH, 'r') as f:
                    data = json.load(f)
                self.balance = data.get('balance', STARTING_BALANCE)
                self.position = data.get('position')
                self.trade_history = data.get('trade_history', [])
                self.last_close_time = data.get('last_close_time')
                logger.info(f"[SIM] Loaded state — balance=${self.balance:.2f}, trades={len(self.trade_history)}")
            except Exception as e:
                logger.warning(f"[SIM] Could not load state: {e}")

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
            logger.error(f"[SIM] Failed to save state: {e}")

    @property
    def in_position(self) -> bool:
        return self.position is not None

    @property
    def is_long(self) -> Optional[bool]:
        if self.position is None:
            return None
        return self.position['side'] == 'long'

    @property
    def entry_price(self) -> float:
        return self.position['entry_price'] if self.position else 0

    def place_limit_order(self, side: str, size: float, price: float) -> Dict:
        order = {
            'side': side,
            'size': size,
            'price': price,
            'placed_at': datetime.now(timezone.utc).isoformat(),
        }
        self.pending_orders.append(order)
        logger.info(f"[SIM ORDER] {side.upper()} {size} {SYMBOL} @ ${price:.2f} (limit, pending)")
        return order

    def cancel_all_orders(self):
        count = len(self.pending_orders)
        self.pending_orders = []
        if count > 0:
            logger.info(f"[SIM] Cancelled {count} pending order(s)")

    def check_pending_fills(self, ask: float, bid: float):
        """Buy limit fills when ask <= order price. Sell limit fills when bid >= order price."""
        if not self.pending_orders:
            return
        filled, remaining = [], []
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

    def _fill_order(self, order: Dict, fill_price: float):
        side = order['side']
        size = order['size']
        if self.position is None:
            pos_side = 'long' if side == 'buy' else 'short'
            self.position = {
                'side': pos_side,
                'size': size,
                'entry_price': fill_price,
                'opened_at': datetime.now(timezone.utc).isoformat(),
                'fills': [{'price': fill_price, 'size': size}],
            }
            logger.info(f"[SIM FILL] Opened {pos_side.upper()} {size} {SYMBOL} @ ${fill_price:.2f}")
        else:
            old_size = self.position['size']
            old_entry = self.position['entry_price']
            new_size = old_size + size
            new_entry = ((old_entry * old_size) + (fill_price * size)) / new_size
            self.position['size'] = new_size
            self.position['entry_price'] = new_entry
            self.position['fills'].append({'price': fill_price, 'size': size})
            logger.info(f"[SIM FILL] Added {size} to {self.position['side'].upper()} — "
                        f"total {new_size} @ avg ${new_entry:.2f}")
        self.save()

    def close_position(self, exit_price: float, reason: str = 'manual'):
        if self.position is None:
            return
        side = self.position['side']
        entry = self.position['entry_price']
        size = self.position['size']

        diff = (exit_price - entry) if side == 'long' else (entry - exit_price)
        pnl_pct = (diff / entry) * LEVERAGE * 100
        notional = (size / POS_SIZE) * self.balance * 0.5
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
        self.last_tp_price = None
        self.last_sl_price = None

        is_win = pnl_pct > 0
        logger.info(f"[SIM CLOSE] {'WIN' if is_win else 'LOSS'} {side.upper()} {size} {SYMBOL} | "
                    f"entry=${entry:.2f} exit=${exit_price:.2f} | "
                    f"PnL={pnl_pct:+.2f}% (${pnl_usd:+.2f}) | reason={reason}")

        self.position = None
        self.cancel_all_orders()
        self.save()

    def snapshot(self) -> Dict:
        """JSON-safe state dict for the dashboard API."""
        total = len(self.trade_history)
        wins = sum(1 for t in self.trade_history if t.get('pnl_pct', 0) > 0)
        losses = total - wins
        total_pnl = sum(t.get('pnl_usd', 0) for t in self.trade_history)
        win_rate = (wins / max(total, 1)) * 100
        return {
            'balance': round(self.balance, 2),
            'starting_balance': STARTING_BALANCE,
            'pnl_total': round(total_pnl, 2),
            'pnl_pct': round(((self.balance - STARTING_BALANCE) / STARTING_BALANCE) * 100, 2),
            'total_trades': total,
            'wins': wins,
            'losses': losses,
            'win_rate': round(win_rate, 2),
            'current_position': self.position,
            'pending_orders': self.pending_orders,
            'trade_history': self.trade_history[-50:],
            'last_cycle_time': self.last_cycle_time,
            'last_cycle_action': self.last_cycle_action,
            'last_bid': self.last_bid,
            'last_open_price': self.last_open_price,
            'last_55bar_high': self.last_55bar_high,
            'last_55bar_low': self.last_55bar_low,
            'last_atr': self.last_atr,
            'last_tp_price': self.last_tp_price,
            'last_sl_price': self.last_sl_price,
            'in_trading_hours': self.last_in_hours,
            'last_error': self.last_error,
            'config': {
                'symbol': SYMBOL,
                'timeframe': TIMEFRAME,
                'lookback_bars': LOOKBACK_BARS,
                'atr_period': ATR_PERIOD,
                'atr_multiplier': ATR_MULTIPLIER,
                'take_profit_pct': TAKE_PROFIT_PCT,
                'pos_size': POS_SIZE,
                'leverage': LEVERAGE,
                'scan_interval': SCAN_INTERVAL,
            },
        }


# ================================================================
# SINGLETON ACCESS
# ================================================================
_bot_instance: Optional[SimulatedTrader] = None


def get_turtle_bot() -> SimulatedTrader:
    global _bot_instance
    if _bot_instance is None:
        _bot_instance = SimulatedTrader()
    return _bot_instance


# ================================================================
# MARKET DATA (live Phemex — read only)
# ================================================================

def ask_bid() -> tuple:
    phemex = _get_phemex()
    ob = phemex.fetch_order_book(SYMBOL)
    bid = ob['bids'][0][0]
    ask = ob['asks'][0][0]
    return ask, bid


def fetch_candle_data() -> pd.DataFrame:
    """Fetch LOOKBACK_BARS candles for TIMEFRAME and compute ATR.

    Returns DataFrame with columns: timestamp, open, high, low, close, volume, ATR.
    """
    phemex = _get_phemex()
    bars = phemex.fetch_ohlcv(SYMBOL, timeframe=TIMEFRAME, limit=CANDLE_LIMIT)
    df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

    # True Range: greatest of (H-L), |H-prev_C|, |L-prev_C|
    prev_close = df['close'].shift(1)
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - prev_close).abs(),
        (df['low'] - prev_close).abs(),
    ], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(ATR_PERIOD).mean()

    return df


# ================================================================
# BREAKOUT DETECTION
# ================================================================

def detect_breakout(df: pd.DataFrame, bid: float, open_price: float) -> Dict:
    """Check for 55-bar high/low breakout with an open-price confirmation gate.

    The open-price gate (from the original code) ensures we only signal on
    candles where price actually crossed the level intra-bar:
      Long:  bid >= 55-bar high  AND  open_price < 55-bar high
      Short: bid <= 55-bar low   AND  open_price > 55-bar low
    """
    # Use only the last LOOKBACK_BARS rows so the rolling window is stable
    window = df.tail(LOOKBACK_BARS)
    high_55 = float(window['high'].max())
    low_55 = float(window['low'].min())
    atr = float(df['ATR'].iloc[-1])

    result = {
        'buy_signal': False,
        'sell_signal': False,
        'high_55': high_55,
        'low_55': low_55,
        'atr': atr,
        'buy_price': None,
        'sell_price': None,
    }

    if bid >= high_55 and open_price < high_55:
        # Price cracked above the 55-bar high this bar — go long
        result['buy_signal'] = True
        result['buy_price'] = bid          # post-only limit at the bid
        logger.info(f"[TURTLE] BUY signal — bid={bid:.2f} >= 55-bar-high={high_55:.2f} | open={open_price:.2f}")

    elif bid <= low_55 and open_price > low_55:
        # Price cracked below the 55-bar low this bar — go short
        result['sell_signal'] = True
        result['sell_price'] = bid         # post-only limit at the bid
        logger.info(f"[TURTLE] SELL signal — bid={bid:.2f} <= 55-bar-low={low_55:.2f} | open={open_price:.2f}")

    else:
        logger.info(f"[TURTLE] No breakout — low={low_55:.2f} < bid={bid:.2f} < high={high_55:.2f} | ATR={atr:.2f}")

    return result


# ================================================================
# SCAN CYCLE — one full bot iteration
# ================================================================

def scan_cycle() -> Dict:
    """Run one full Turtle bot cycle. Returns a result dict for the dashboard."""
    sim = get_turtle_bot()
    now_str = datetime.now(timezone.utc).isoformat()
    result = {"timestamp": now_str, "action": "monitoring"}

    try:
        # 1. Always check pending order fills regardless of trading hours
        ask, bid = ask_bid()
        sim.last_bid = bid
        sim.check_pending_fills(ask, bid)

        # 2. Force-close if past Friday 4pm ET (end of trading week)
        if end_of_trading_week():
            sim.last_in_hours = False
            if sim.in_position:
                exit_price = bid if sim.is_long else ask
                sim.close_position(exit_price, reason='end_of_week_friday')
                result["action"] = "closed_end_of_week"
            else:
                sim.cancel_all_orders()
                result["action"] = "out_of_hours_eow_no_position"
            sim.last_cycle_time = now_str
            sim.last_cycle_action = result["action"]
            return result

        # 3. Check if within Mon–Fri 9:30–16:00 ET
        in_hours = in_trading_hours()
        sim.last_in_hours = in_hours

        if not in_hours:
            # Cancel any resting orders outside trading hours; hold open positions
            sim.cancel_all_orders()
            result["action"] = "outside_trading_hours"
            sim.last_cycle_time = now_str
            sim.last_cycle_action = result["action"]
            return result

        # 4. PnL / exit check for open position
        if sim.in_position:
            entry_price = sim.entry_price
            is_long = sim.is_long

            # Fetch fresh candle data to get current ATR for the stop loss
            df = fetch_candle_data()
            atr = float(df['ATR'].iloc[-1])
            sim.last_atr = atr if not np.isnan(atr) else sim.last_atr

            if is_long:
                tp_price = entry_price * (1 + TAKE_PROFIT_PCT / 100)
                sl_price = entry_price - (atr * ATR_MULTIPLIER)
            else:
                tp_price = entry_price * (1 - TAKE_PROFIT_PCT / 100)
                sl_price = entry_price + (atr * ATR_MULTIPLIER)

            sim.last_tp_price = round(tp_price, 2)
            sim.last_sl_price = round(sl_price, 2)

            hit_tp = (bid >= tp_price) if is_long else (bid <= tp_price)
            hit_sl = (bid <= sl_price) if is_long else (bid >= sl_price)

            diff = (bid - entry_price) if is_long else (entry_price - bid)
            perc = ((diff / entry_price) * LEVERAGE * 100) if entry_price > 0 else 0

            logger.info(f"[PnL] {'LONG' if is_long else 'SHORT'} entry=${entry_price:.2f} "
                        f"bid=${bid:.2f} PnL={perc:+.2f}% | TP=${tp_price:.2f} SL=${sl_price:.2f}")

            if hit_tp:
                exit_price = bid if is_long else ask
                sim.close_position(exit_price, reason=f'take_profit_{perc:.2f}pct')
                result["action"] = f"closed_tp_{perc:.2f}pct"
            elif hit_sl:
                exit_price = bid if is_long else ask
                sim.close_position(exit_price, reason=f'stop_loss_2atr_{perc:.2f}pct')
                result["action"] = f"closed_sl_{perc:.2f}pct"
            else:
                result["action"] = f"in_position_pnl_{perc:+.2f}pct"

        # 5. Cooldown check
        if sim.last_close_time is not None:
            elapsed_min = (time.time() - sim.last_close_time) / 60
            if elapsed_min < COOLDOWN_MINUTES:
                result["action"] = f"cooldown_{elapsed_min:.0f}m"
                sim.last_cycle_time = now_str
                sim.last_cycle_action = result["action"]
                return result

        # 6. Scan for new breakout entry if flat
        if not sim.in_position:
            df = fetch_candle_data()
            ask, bid = ask_bid()
            sim.last_bid = bid

            # The ticker open mirrors the original bot's openPrice (current bar's open)
            ticker = _get_phemex().fetch_ticker(SYMBOL)
            open_price = ticker.get('open') or float(df['open'].iloc[-1])
            sim.last_open_price = open_price

            atr = float(df['ATR'].iloc[-1])
            sim.last_atr = atr if not np.isnan(atr) else sim.last_atr

            breakout = detect_breakout(df, bid, open_price)
            sim.last_55bar_high = breakout['high_55']
            sim.last_55bar_low = breakout['low_55']

            sim.cancel_all_orders()

            if breakout['buy_signal']:
                price = breakout['buy_price']
                sim.place_limit_order('buy', POS_SIZE, price)
                result["action"] = f"turtle_buy_limit_{price:.2f}"

            elif breakout['sell_signal']:
                price = breakout['sell_price']
                sim.place_limit_order('sell', POS_SIZE, price)
                result["action"] = f"turtle_sell_limit_{price:.2f}"

            else:
                result["action"] = (
                    f"no_breakout_H={breakout['high_55']:.0f}"
                    f"_L={breakout['low_55']:.0f}"
                    f"_ATR={breakout['atr']:.0f}"
                )

        sim.last_error = None

    except Exception as e:
        logger.error(f"[CYCLE] Error: {e}", exc_info=True)
        result["action"] = f"error: {e}"
        sim.last_error = str(e)

    sim.last_cycle_time = now_str
    sim.last_cycle_action = result["action"]
    return result


# ================================================================
# STANDALONE ENTRY POINT
# ================================================================
if __name__ == '__main__':
    import schedule as sched

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(message)s')
    sim = get_turtle_bot()

    logger.info(f"[TURTLE BOT] Starting — {SYMBOL} | {TIMEFRAME} | 55-bar breakout | "
                f"TP={TAKE_PROFIT_PCT}% SL=2×ATR | Lev={LEVERAGE}x")
    logger.info(f"[TURTLE BOT] Balance: ${sim.balance:.2f} | Log: {SIM_LOG_PATH}")
    logger.info(f"[TURTLE BOT] Trading hours: Mon–Fri {MARKET_OPEN}–{MARKET_CLOSE} ET")

    sched.every(SCAN_INTERVAL).seconds.do(scan_cycle)

    while True:
        try:
            sched.run_pending()
            time.sleep(1)
        except KeyboardInterrupt:
            logger.info("[TURTLE BOT] Shutting down...")
            sim.save()
            break
        except Exception as e:
            logger.error(f"[TURTLE BOT] Error: {e}")
            time.sleep(30)
