############# Consolidation Pop Bot #5 — Simulated Trading via Phemex Live Data
#
# Strategy: Buy the low of a consolidation range, sell the high.
#   Consolidation is detected when ATR/close < CONSOLIDATION_PCT (price moving sideways).
#   Walk backward through candles to find where consolidation began, then use
#   that window's low/high as the range.
#
#   Long  entry: bid is in the lower 1/3 of range  → BUY  limit @ bid, TP=+0.3%, SL=-0.25%
#   Short entry: bid is in the upper 1/3 of range  → SELL limit @ bid, TP=-0.3%, SL=+0.25%
#
#   Thesis: most traders short the top of the range; we buy the bottom and
#   take profit quickly when price briefly cracks the top.
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
from datetime import datetime, timezone
from typing import Optional, Dict, List, Tuple

logger = logging.getLogger("CONSOL-BOT")

# ================================================================
# CONFIGURATION
# ================================================================
SYMBOL = 'BTC/USD:BTC'
TIMEFRAME = '5m'            # '1m' | '3m' | '5m' | '15m' | '1h'
CANDLE_LIMIT = 20           # candles to fetch (original: limit=20)
ATR_PERIOD = 14             # period for ATR rolling average
CONSOLIDATION_PCT = 0.7     # if ATR/close% < this, price is consolidating
TP_PCT = 0.3                # take profit percent from entry
SL_PCT = 0.25               # stop loss percent from entry
POS_SIZE = 30               # position size in contracts
LEVERAGE = 10               # simulated leverage multiplier
STARTING_BALANCE = 1000     # simulated starting balance in USD
COOLDOWN_MINUTES = 5        # wait after a close before re-entering
SCAN_INTERVAL = 20          # seconds between scan cycles (original: every 20s)

SIM_LOG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'consolidation_pop_trade_log.json'
)


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
        self.last_signal: Optional[str] = None
        self.last_bid: Optional[float] = None
        self.last_tr_deviance: Optional[float] = None   # ATR/close as %
        self.last_consol_low: Optional[float] = None    # bottom of current range
        self.last_consol_high: Optional[float] = None   # top of current range
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

    def place_limit_order(self, side: str, size: float, price: float,
                          take_profit: float, stop_loss: float) -> Dict:
        order = {
            'side': side,
            'size': size,
            'price': price,
            'take_profit': take_profit,
            'stop_loss': stop_loss,
            'placed_at': datetime.now(timezone.utc).isoformat(),
        }
        self.pending_orders.append(order)
        logger.info(f"[SIM ORDER] {side.upper()} {size} {SYMBOL} @ ${price:.2f} "
                    f"TP=${take_profit:.2f} SL=${stop_loss:.2f}")
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
                'take_profit': order['take_profit'],
                'stop_loss': order['stop_loss'],
                'opened_at': datetime.now(timezone.utc).isoformat(),
                'fills': [{'price': fill_price, 'size': size}],
            }
            logger.info(f"[SIM FILL] Opened {pos_side.upper()} {size} {SYMBOL} @ ${fill_price:.2f}")
        else:
            # Scale into existing position (shouldn't normally happen in this strategy)
            old_size = self.position['size']
            old_entry = self.position['entry_price']
            new_size = old_size + size
            new_entry = ((old_entry * old_size) + (fill_price * size)) / new_size
            self.position['size'] = new_size
            self.position['entry_price'] = new_entry
            self.position['fills'].append({'price': fill_price, 'size': size})
            logger.info(f"[SIM FILL] Added to {self.position['side'].upper()} — "
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
            'last_signal': self.last_signal,
            'last_bid': self.last_bid,
            'last_tr_deviance': round(self.last_tr_deviance, 3) if self.last_tr_deviance is not None else None,
            'last_consol_low': self.last_consol_low,
            'last_consol_high': self.last_consol_high,
            'last_error': self.last_error,
            'config': {
                'symbol': SYMBOL,
                'timeframe': TIMEFRAME,
                'candle_limit': CANDLE_LIMIT,
                'consolidation_pct': CONSOLIDATION_PCT,
                'tp_pct': TP_PCT,
                'sl_pct': SL_PCT,
                'pos_size': POS_SIZE,
                'leverage': LEVERAGE,
                'scan_interval': SCAN_INTERVAL,
            },
        }


# ================================================================
# SINGLETON ACCESS
# ================================================================
_bot_instance: Optional[SimulatedTrader] = None


def get_consolidation_pop_bot() -> SimulatedTrader:
    global _bot_instance
    if _bot_instance is None:
        _bot_instance = SimulatedTrader()
    return _bot_instance


# ================================================================
# MARKET DATA (live Phemex — read only)
# ================================================================

def ask_bid() -> Tuple[float, float]:
    phemex = _get_phemex()
    ob = phemex.fetch_order_book(SYMBOL)
    bid = ob['bids'][0][0]
    ask = ob['asks'][0][0]
    return ask, bid


def fetch_candle_data() -> pd.DataFrame:
    """Fetch CANDLE_LIMIT bars and compute True Range + ATR.

    Phemex accepts discrete limit values (100, 500, 1000); CANDLE_LIMIT=20
    is below the minimum, so we fetch 100 and slice to avoid a request error.
    ATR requires ATR_PERIOD warm-up bars, so the extra bars are useful.
    """
    phemex = _get_phemex()
    # Fetch enough bars: max of our candle limit and ATR period, padded for warmup
    fetch_count = max(CANDLE_LIMIT + ATR_PERIOD + 5, 100)
    bars = phemex.fetch_ohlcv(SYMBOL, timeframe=TIMEFRAME, limit=fetch_count)
    df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

    # True Range per bar: max(H-L, |H-prev_C|, |L-prev_C|)
    prev_close = df['close'].shift(1)
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - prev_close).abs(),
        (df['low'] - prev_close).abs(),
    ], axis=1).max(axis=1)
    # ATR = rolling mean of TR (mirrors pandas_ta's default wilder smoothing approximation)
    df['ATR'] = tr.rolling(ATR_PERIOD).mean()

    # Return only the most recent CANDLE_LIMIT rows — matches original limit param
    return df.tail(CANDLE_LIMIT).reset_index(drop=True)


# ================================================================
# CONSOLIDATION DETECTION
# ================================================================

def get_consolidation_range(df: pd.DataFrame) -> Tuple[float, float]:
    """Find the low/high of the current consolidation window.

    Walks backward from the most recent candle. The first bar where
    ATR/close > CONSOLIDATION_PCT marks the boundary — everything after
    that bar (i.e., more recent) is the consolidation window.

    If no boundary is found, the entire df range is returned.
    Port of get_extreme_of_consolidation() from functions.py.
    """
    for i in range(len(df) - 1, -1, -1):
        row = df.iloc[i]
        atr = row['ATR']
        close = row['close']
        if pd.isna(atr) or close == 0:
            continue
        if (atr / close) * 100 > CONSOLIDATION_PCT:
            # Consolidation window = all rows after index i
            window = df.iloc[i + 1:]
            if window.empty:
                break
            return float(window['low'].min()), float(window['high'].max())

    return float(df['low'].min()), float(df['high'].max())


def detect_signal(df: pd.DataFrame, bid: float) -> Dict:
    """Check for consolidation and whether bid is in the entry zone.

    Returns a dict describing the signal, consolidation range, and order prices.
    """
    last_atr = float(df['ATR'].iloc[-1])
    last_close = float(df['close'].iloc[-1])

    if last_close == 0 or np.isnan(last_atr):
        return {
            'in_consolidation': False,
            'signal': 'NO_DATA',
            'tr_deviance': None,
            'consol_low': None,
            'consol_high': None,
            'buy_price': None,
            'sell_price': None,
            'tp_buy': None,
            'sl_buy': None,
            'tp_sell': None,
            'sl_sell': None,
        }

    tr_deviance = (last_atr / last_close) * 100
    in_consolidation = tr_deviance < CONSOLIDATION_PCT

    result = {
        'in_consolidation': in_consolidation,
        'tr_deviance': tr_deviance,
        'signal': 'NOT_CONSOLIDATING',
        'consol_low': None,
        'consol_high': None,
        'buy_price': None,
        'sell_price': None,
        'tp_buy': None,
        'sl_buy': None,
        'tp_sell': None,
        'sl_sell': None,
    }

    if not in_consolidation:
        logger.info(f"[CONSOL] Not consolidating — TR dev={tr_deviance:.3f}% >= {CONSOLIDATION_PCT}%")
        return result

    consol_low, consol_high = get_consolidation_range(df)
    range_size = consol_high - consol_low
    result['consol_low'] = consol_low
    result['consol_high'] = consol_high

    if range_size <= 0:
        result['signal'] = 'ZERO_RANGE'
        return result

    one_third = range_size / 3
    lower_third_top = consol_low + one_third      # upper edge of the bottom 1/3
    upper_third_bot = consol_high - one_third     # lower edge of the top 1/3

    if bid <= lower_third_top:
        # Price in lower 1/3 — long entry
        result['signal'] = 'LONG_ENTRY'
        result['buy_price'] = bid
        result['tp_buy'] = round(bid * (1 + TP_PCT / 100), 2)
        result['sl_buy'] = round(bid * (1 - SL_PCT / 100), 2)
        logger.info(f"[CONSOL] LONG entry zone — bid={bid:.2f} <= lower1/3 top={lower_third_top:.2f} "
                    f"| range=[{consol_low:.2f}–{consol_high:.2f}] | TP={result['tp_buy']:.2f} SL={result['sl_buy']:.2f}")

    elif bid >= upper_third_bot:
        # Price in upper 1/3 — short entry
        result['signal'] = 'SHORT_ENTRY'
        result['sell_price'] = bid
        result['tp_sell'] = round(bid * (1 - TP_PCT / 100), 2)
        result['sl_sell'] = round(bid * (1 + SL_PCT / 100), 2)
        logger.info(f"[CONSOL] SHORT entry zone — bid={bid:.2f} >= upper1/3 bot={upper_third_bot:.2f} "
                    f"| range=[{consol_low:.2f}–{consol_high:.2f}] | TP={result['tp_sell']:.2f} SL={result['sl_sell']:.2f}")

    else:
        result['signal'] = 'CONSOLIDATING_MID'
        logger.info(f"[CONSOL] In consolidation mid-range — bid={bid:.2f} "
                    f"range=[{consol_low:.2f}–{consol_high:.2f}] TR dev={tr_deviance:.3f}%")

    return result


# ================================================================
# SCAN CYCLE — one full bot iteration
# ================================================================

def scan_cycle() -> Dict:
    """Run one full Consolidation Pop bot cycle. Returns a result dict for the dashboard."""
    sim = get_consolidation_pop_bot()
    now_str = datetime.now(timezone.utc).isoformat()
    result = {"timestamp": now_str, "action": "monitoring"}

    try:
        # 1. Check pending order fills
        ask, bid = ask_bid()
        sim.last_bid = bid
        sim.check_pending_fills(ask, bid)

        # 2. Check TP/SL if in position (stored on the position dict at fill time)
        if sim.in_position:
            pos = sim.position
            is_long = sim.is_long
            tp = pos.get('take_profit')
            sl = pos.get('stop_loss')

            hit_tp = (bid >= tp) if (is_long and tp) else (bid <= tp) if tp else False
            hit_sl = (bid <= sl) if (is_long and sl) else (bid >= sl) if sl else False

            entry = sim.entry_price
            diff = (bid - entry) if is_long else (entry - bid)
            perc = ((diff / entry) * LEVERAGE * 100) if entry > 0 else 0

            logger.info(f"[PnL] {'LONG' if is_long else 'SHORT'} entry=${entry:.2f} "
                        f"bid=${bid:.2f} PnL={perc:+.2f}% | TP=${tp} SL=${sl}")

            if hit_tp:
                exit_price = bid if is_long else ask
                sim.close_position(exit_price, reason=f'take_profit_{TP_PCT}pct')
                result["action"] = f"closed_tp_{perc:+.2f}pct"
            elif hit_sl:
                exit_price = bid if is_long else ask
                sim.close_position(exit_price, reason=f'stop_loss_{SL_PCT}pct')
                result["action"] = f"closed_sl_{perc:+.2f}pct"
            else:
                result["action"] = f"in_position_pnl_{perc:+.2f}pct"

        # 3. Cooldown check
        if sim.last_close_time is not None:
            elapsed_min = (time.time() - sim.last_close_time) / 60
            if elapsed_min < COOLDOWN_MINUTES:
                result["action"] = f"cooldown_{elapsed_min:.0f}m"
                sim.last_cycle_time = now_str
                sim.last_cycle_action = result["action"]
                return result

        # 4. Look for new consolidation entry if flat
        if not sim.in_position:
            df = fetch_candle_data()
            ask, bid = ask_bid()
            sim.last_bid = bid

            sig = detect_signal(df, bid)
            sim.last_signal = sig['signal']
            sim.last_tr_deviance = sig['tr_deviance']
            sim.last_consol_low = sig['consol_low']
            sim.last_consol_high = sig['consol_high']

            sim.cancel_all_orders()

            if sig['signal'] == 'LONG_ENTRY':
                sim.place_limit_order(
                    'buy', POS_SIZE, sig['buy_price'],
                    take_profit=sig['tp_buy'],
                    stop_loss=sig['sl_buy'],
                )
                result["action"] = f"consol_buy_{sig['buy_price']:.2f}_TP={sig['tp_buy']}_SL={sig['sl_buy']}"

            elif sig['signal'] == 'SHORT_ENTRY':
                sim.place_limit_order(
                    'sell', POS_SIZE, sig['sell_price'],
                    take_profit=sig['tp_sell'],
                    stop_loss=sig['sl_sell'],
                )
                result["action"] = f"consol_sell_{sig['sell_price']:.2f}_TP={sig['tp_sell']}_SL={sig['sl_sell']}"

            else:
                dev_str = f"{sig['tr_deviance']:.3f}%" if sig['tr_deviance'] is not None else "N/A"
                result["action"] = f"{sig['signal']}_dev={dev_str}"

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
    sim = get_consolidation_pop_bot()

    logger.info(f"[CONSOL BOT] Starting — {SYMBOL} | {TIMEFRAME} | "
                f"Consol={CONSOLIDATION_PCT}% | TP={TP_PCT}% SL={SL_PCT}% | Lev={LEVERAGE}x")
    logger.info(f"[CONSOL BOT] Balance: ${sim.balance:.2f} | Log: {SIM_LOG_PATH}")

    sched.every(SCAN_INTERVAL).seconds.do(scan_cycle)

    while True:
        try:
            sched.run_pending()
            time.sleep(1)
        except KeyboardInterrupt:
            logger.info("[CONSOL BOT] Shutting down...")
            sim.save()
            break
        except Exception as e:
            logger.error(f"[CONSOL BOT] Error: {e}")
            time.sleep(30)
