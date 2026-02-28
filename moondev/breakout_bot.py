############# Breakout Bot #2 — Simulated Trading via Phemex Live Data
#
# Strategy: 3-day support/resistance on 15m candles.
#           Breakout above resistance → BUY retest.
#           Breakdown below support → SELL retest.
# Data: Live from Phemex (read-only, no real orders placed)
# Execution: All trades are simulated in-memory and logged to JSON
#
# Original: moondev coding bot #2 (Phemex live breakout)
# Modified: Simulation mode, server-importable module

import ccxt
import pandas as pd
import numpy as np
import json
import os
import logging
import time
from datetime import datetime, timezone
from typing import Optional, Dict, List

logger = logging.getLogger("BREAKOUT-BOT")

# ================================================================
# CONFIGURATION
# ================================================================
SYMBOL = 'BTC/USD:BTC'
POS_SIZE = 30           # position size in contracts
TARGET_PCT = 9          # take profit at 9% PnL
MAX_LOSS_PCT = -8       # stop loss at -8% PnL
VOL_DECIMAL = 0.4       # OB volume ratio threshold for exit gating
LEVERAGE = 10           # simulated leverage multiplier
STARTING_BALANCE = 1000 # simulated starting balance in USD
COOLDOWN_MINUTES = 10   # minutes to wait between closing and re-entering
SCAN_INTERVAL = 28      # seconds between scan cycles
LOOKBACK_BARS = 289     # 96 bars/day * 3 days + 1
SMA_PERIOD = 20

SIM_LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'breakout_trade_log.json')


# ================================================================
# PHEMEX CONNECTION (lazy init — shared with sma_bot if both run)
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
        self.last_support: Optional[float] = None
        self.last_resistance: Optional[float] = None
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
    def position_size(self) -> float:
        if self.position is None:
            return 0
        return self.position['size']

    @property
    def entry_price(self) -> float:
        if self.position is None:
            return 0
        return self.position['entry_price']

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
            logger.info(f"[SIM FILL] Added {size} to {self.position['side'].upper()} — total {new_size} @ avg ${new_entry:.2f}")
        self.save()

    def close_position(self, exit_price: float, reason: str = 'manual'):
        if self.position is None:
            return
        side = self.position['side']
        entry = self.position['entry_price']
        size = self.position['size']

        if side == 'long':
            diff = exit_price - entry
        else:
            diff = entry - exit_price

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

        is_win = pnl_pct > 0
        logger.info(f"[SIM CLOSE] {'WIN' if is_win else 'LOSS'} {side.upper()} {size} {SYMBOL} | "
                     f"entry=${entry:.2f} exit=${exit_price:.2f} | PnL={pnl_pct:+.2f}% (${pnl_usd:+.2f}) | reason={reason}")

        self.position = None
        self.cancel_all_orders()
        self.save()

    def snapshot(self) -> Dict:
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
            'last_support': self.last_support,
            'last_resistance': self.last_resistance,
            'last_error': self.last_error,
            'config': {
                'symbol': SYMBOL,
                'pos_size': POS_SIZE,
                'target_pct': TARGET_PCT,
                'max_loss_pct': MAX_LOSS_PCT,
                'leverage': LEVERAGE,
                'lookback_bars': LOOKBACK_BARS,
                'scan_interval': SCAN_INTERVAL,
            },
        }


# ================================================================
# SINGLETON ACCESS
# ================================================================
_bot_instance: Optional[SimulatedTrader] = None


def get_breakout_bot() -> SimulatedTrader:
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


def fetch_15m_data() -> pd.DataFrame:
    """Fetch 15m candles (3 days), compute 20-SMA, support, resistance.

    Phemex kline endpoint only accepts discrete limit values: 5, 10, 50, 100, 500, 1000.
    We request 500 (the nearest valid value above LOOKBACK_BARS=289) and use the
    rolling window for support/resistance so the extra bars don't affect the result.
    """
    phemex = _get_phemex()
    bars = phemex.fetch_ohlcv(SYMBOL, timeframe='15m', limit=500)
    df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['sma20'] = df['close'].rolling(SMA_PERIOD).mean()
    df['support'] = df['close'].rolling(LOOKBACK_BARS).min()
    df['resis'] = df['close'].rolling(LOOKBACK_BARS).max()
    return df


# ================================================================
# BREAKOUT DETECTION
# ================================================================

def detect_breakout(df: pd.DataFrame, bid: float) -> Dict:
    """Check if price is breaking above resistance or below support.

    Returns dict with breakout/breakdown flags and retest prices.
    """
    curr_resis = df['resis'].iloc[-1]
    curr_support = df['support'].iloc[-1]

    result = {
        'buy_breakout': False,
        'sell_breakdown': False,
        'breakout_price': None,
        'breakdown_price': None,
        'resistance': curr_resis,
        'support': curr_support,
    }

    if bid > curr_resis:
        # breaking upward — buy retest at resistance + 0.1%
        result['buy_breakout'] = True
        result['breakout_price'] = curr_resis * 1.001
        logger.info(f"[BREAKOUT] Breaking UP — bid=${bid:.2f} > resis=${curr_resis:.2f}, retest @ ${result['breakout_price']:.2f}")

    elif bid < curr_support:
        # breaking downward — sell retest at support - 0.1%
        result['sell_breakdown'] = True
        result['breakdown_price'] = curr_support * 0.999
        logger.info(f"[BREAKOUT] Breaking DOWN — bid=${bid:.2f} < support=${curr_support:.2f}, retest @ ${result['breakdown_price']:.2f}")

    else:
        logger.info(f"[BREAKOUT] In range — support=${curr_support:.2f} < bid=${bid:.2f} < resis=${curr_resis:.2f}")

    return result


# ================================================================
# SCAN CYCLE — one full bot iteration
# ================================================================

def scan_cycle() -> Dict:
    """Run one full breakout bot cycle. Returns a result dict for the dashboard."""
    sim = get_breakout_bot()
    now_str = datetime.now(timezone.utc).isoformat()
    result = {"timestamp": now_str, "action": "monitoring"}

    try:
        # 1. Check pending order fills
        ask, bid = ask_bid()
        sim.last_bid = bid
        sim.check_pending_fills(ask, bid)

        # 2. Check PnL / exit conditions if in position
        if sim.in_position:
            entry_price = sim.entry_price
            is_long = sim.is_long

            if is_long:
                diff = bid - entry_price
            else:
                diff = entry_price - bid

            perc = ((diff / entry_price) * LEVERAGE * 100) if entry_price > 0 else 0

            logger.info(f"[PnL] {'LONG' if is_long else 'SHORT'} entry=${entry_price:.2f} now=${bid:.2f} PnL={perc:+.2f}%")

            if perc > TARGET_PCT:
                exit_price = bid if is_long else ask
                sim.close_position(exit_price, reason=f'target_hit_{perc:.1f}pct')
                result["action"] = f"closed_target_{perc:.1f}pct"
            elif perc <= MAX_LOSS_PCT:
                exit_price = bid if is_long else ask
                sim.close_position(exit_price, reason=f'stop_loss_{perc:.1f}pct')
                result["action"] = f"closed_stoploss_{perc:.1f}pct"
            else:
                result["action"] = f"in_position_pnl_{perc:+.1f}pct"

        # 3. Cooldown check
        if sim.last_close_time is not None:
            elapsed_min = (time.time() - sim.last_close_time) / 60
            if elapsed_min < COOLDOWN_MINUTES:
                result["action"] = f"cooldown_{elapsed_min:.0f}m"
                sim.last_cycle_time = now_str
                sim.last_cycle_action = result["action"]
                return result

        # 4. Detect breakout and place orders if not in position
        if not sim.in_position:
            df = fetch_15m_data()
            ask, bid = ask_bid()
            sim.last_bid = bid

            breakout = detect_breakout(df, bid)
            sim.last_support = breakout['support']
            sim.last_resistance = breakout['resistance']

            sim.cancel_all_orders()

            if breakout['buy_breakout']:
                price = breakout['breakout_price']
                sim.place_limit_order('buy', POS_SIZE, price)
                result["action"] = f"breakout_buy_at_{price:.2f}"

            elif breakout['sell_breakdown']:
                price = breakout['breakdown_price']
                sim.place_limit_order('sell', POS_SIZE, price)
                result["action"] = f"breakdown_sell_at_{price:.2f}"

            else:
                result["action"] = f"in_range_S={breakout['support']:.0f}_R={breakout['resistance']:.0f}"

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
    sim = get_breakout_bot()

    logger.info(f"[BREAKOUT BOT] Starting — {SYMBOL} | Size={POS_SIZE} | TP={TARGET_PCT}% SL={MAX_LOSS_PCT}% | Lev={LEVERAGE}x")
    logger.info(f"[BREAKOUT BOT] Balance: ${sim.balance:.2f} | Log: {SIM_LOG_PATH}")

    sched.every(SCAN_INTERVAL).seconds.do(scan_cycle)

    while True:
        try:
            sched.run_pending()
            time.sleep(1)
        except KeyboardInterrupt:
            logger.info("[BREAKOUT BOT] Shutting down...")
            sim.save()
            break
        except Exception as e:
            logger.error(f"[BREAKOUT BOT] Error: {e}")
            time.sleep(30)
