############# SMA Bot #1 — Simulated Trading via Phemex Live Data
#
# Strategy: Daily 20-SMA bias + 15m 20-SMA entry levels
# Data: Live from Phemex (read-only, no real orders placed)
# Execution: All trades are simulated in-memory and logged to JSON
#
# Can run standalone (python 10_bot1.py) or imported by the server
# as a background auto-scan loop with a dashboard.

import ccxt
import pandas as pd
import numpy as np
import json
import os
import logging
import time
from datetime import datetime, timezone
from typing import Optional, Dict, List

logger = logging.getLogger("SMA-BOT")

# ================================================================
# CONFIGURATION
# ================================================================
SYMBOL = 'uBTCUSD'
POS_SIZE = 30           # total position size in contracts
TARGET_PCT = 8          # take profit at 8% PnL
MAX_LOSS_PCT = -9       # stop loss at -9% PnL
VOL_DECIMAL = 0.4       # OB volume ratio threshold for exit gating
LEVERAGE = 10           # simulated leverage multiplier
STARTING_BALANCE = 1000 # simulated starting balance in USD
COOLDOWN_MINUTES = 59   # minutes to wait between closing and re-entering
SCAN_INTERVAL = 28      # seconds between scan cycles

SIM_LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sim_trade_log.json')


# ================================================================
# PHEMEX CONNECTION (lazy init — no side effects at import time)
# ================================================================
_phemex_client: Optional[ccxt.phemex] = None


def _get_phemex() -> ccxt.phemex:
    """Lazy-init Phemex CCXT client from environment variables."""
    global _phemex_client
    if _phemex_client is None:
        api_key = os.getenv("PHEMEX_API_KEY", "")
        api_secret = os.getenv("PHEMEX_API_SECRET", "")
        if not api_key or not api_secret:
            logger.warning("[PHEMEX] PHEMEX_API_KEY or PHEMEX_API_SECRET not set")
        _phemex_client = ccxt.phemex({
            'enableRateLimit': True,
            'apiKey': api_key,
            'secret': api_secret,
        })
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

    # --- position queries ---

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

    # --- simulated order handling ---

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
        logger.info(f"[SIM] Balance: ${self.balance:.2f} | Total trades: {len(self.trade_history)}")

        self.position = None
        self.cancel_all_orders()
        self.save()

    def snapshot(self) -> Dict:
        """JSON-safe state for the dashboard API."""
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
            'last_error': self.last_error,
            'config': {
                'symbol': SYMBOL,
                'pos_size': POS_SIZE,
                'target_pct': TARGET_PCT,
                'max_loss_pct': MAX_LOSS_PCT,
                'leverage': LEVERAGE,
                'scan_interval': SCAN_INTERVAL,
            },
        }


# ================================================================
# SINGLETON ACCESS
# ================================================================
_sim_instance: Optional[SimulatedTrader] = None


def get_sma_bot() -> SimulatedTrader:
    global _sim_instance
    if _sim_instance is None:
        _sim_instance = SimulatedTrader()
    return _sim_instance


# ================================================================
# MARKET DATA (live Phemex — read only)
# ================================================================

def ask_bid() -> tuple:
    phemex = _get_phemex()
    ob = phemex.fetch_order_book(SYMBOL)
    bid = ob['bids'][0][0]
    ask = ob['asks'][0][0]
    return ask, bid


def daily_sma() -> pd.DataFrame:
    phemex = _get_phemex()
    bars = phemex.fetch_ohlcv(SYMBOL, timeframe='1d', limit=100)
    df_d = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df_d['timestamp'] = pd.to_datetime(df_d['timestamp'], unit='ms')
    df_d['sma20_d'] = df_d.close.rolling(20).mean()

    bid = ask_bid()[1]
    df_d.loc[df_d['sma20_d'] > bid, 'sig'] = 'SELL'
    df_d.loc[df_d['sma20_d'] < bid, 'sig'] = 'BUY'
    return df_d


def f15_sma() -> pd.DataFrame:
    phemex = _get_phemex()
    bars = phemex.fetch_ohlcv(SYMBOL, timeframe='15m', limit=100)
    df_f = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df_f['timestamp'] = pd.to_datetime(df_f['timestamp'], unit='ms')
    df_f['sma20_15'] = df_f.close.rolling(20).mean()

    df_f['bp_1'] = df_f['sma20_15'] * 1.001
    df_f['bp_2'] = df_f['sma20_15'] * 0.997
    df_f['sp_1'] = df_f['sma20_15'] * 0.999
    df_f['sp_2'] = df_f['sma20_15'] * 1.003
    return df_f


def ob_volume_check() -> bool:
    """Sample order book ~55s. Returns True if volume ratio under threshold (safe to exit)."""
    rows = []
    phemex = _get_phemex()
    for x in range(11):
        book = phemex.fetch_order_book(SYMBOL)
        sum_bidvol = sum(level[1] for level in book['bids'])
        sum_askvol = sum(level[1] for level in book['asks'])
        rows.append({'bid_vol': sum_bidvol, 'ask_vol': sum_askvol})
        if x < 10:
            time.sleep(5)

    df = pd.DataFrame(rows)
    total_bidvol = df['bid_vol'].sum()
    total_askvol = df['ask_vol'].sum()

    if total_bidvol > total_askvol:
        control_dec = total_askvol / total_bidvol
    else:
        control_dec = total_bidvol / total_askvol

    logger.info(f"[OB] Bid={total_bidvol} Ask={total_askvol} ratio={control_dec:.3f} threshold={VOL_DECIMAL}")
    return control_dec < VOL_DECIMAL


# ================================================================
# SCAN CYCLE — one full bot iteration (called by server or standalone)
# ================================================================

def scan_cycle() -> Dict:
    """Run one full bot cycle. Returns a result dict for the dashboard."""
    sim = get_sma_bot()
    now_str = datetime.now(timezone.utc).isoformat()
    result = {"timestamp": now_str, "action": "monitoring"}

    try:
        # 1. Check pending order fills
        ask, bid = ask_bid()
        sim.last_bid = bid
        sim.check_pending_fills(ask, bid)

        # 2. Check PnL / exit conditions
        if sim.in_position:
            entry_price = sim.entry_price
            current_price = bid
            is_long = sim.is_long

            if is_long:
                diff = current_price - entry_price
            else:
                diff = entry_price - current_price

            perc = ((diff / entry_price) * LEVERAGE * 100) if entry_price > 0 else 0

            logger.info(f"[PnL] {'LONG' if is_long else 'SHORT'} entry=${entry_price:.2f} now=${current_price:.2f} PnL={perc:+.2f}%")

            if perc > TARGET_PCT:
                logger.info(f"[PnL] Target hit ({perc:.2f}% > {TARGET_PCT}%), checking OB volume...")
                vol_under = ob_volume_check()
                if not vol_under:
                    exit_price = bid if is_long else ask
                    sim.close_position(exit_price, reason=f'target_hit_{perc:.1f}pct')
                    result["action"] = f"closed_target_{perc:.1f}pct"
                else:
                    logger.info("[PnL] Volume under threshold, holding position")
                    result["action"] = "holding_vol_gate"

            elif perc <= MAX_LOSS_PCT:
                logger.info(f"[PnL] Stop loss triggered ({perc:.2f}% <= {MAX_LOSS_PCT}%)")
                exit_price = bid if is_long else ask
                sim.close_position(exit_price, reason=f'stop_loss_{perc:.1f}pct')
                result["action"] = f"closed_stoploss_{perc:.1f}pct"

            else:
                result["action"] = f"in_position_pnl_{perc:+.1f}pct"

        # 3. Cooldown check
        if sim.last_close_time is not None:
            elapsed_min = (time.time() - sim.last_close_time) / 60
            if elapsed_min < COOLDOWN_MINUTES:
                logger.info(f"[COOLDOWN] {elapsed_min:.1f}m since last close, waiting...")
                result["action"] = f"cooldown_{elapsed_min:.0f}m"
                sim.last_cycle_time = now_str
                sim.last_cycle_action = result["action"]
                return result

        # 4. Compute signals and place orders if not in position
        if not sim.in_position:
            df_d = daily_sma()
            df_f = f15_sma()
            ask, bid = ask_bid()

            sig = df_d.iloc[-1]['sig']
            sim.last_signal = sig
            last_sma15 = df_f.iloc[-1]['sma20_15']
            open_size = POS_SIZE / 2

            logger.info(f"[SIGNAL] Daily={sig} | bid=${bid} | 15m_sma=${last_sma15:.2f}")

            sim.cancel_all_orders()

            if (sig == 'BUY') and (bid > last_sma15):
                bp_1 = df_f.iloc[-1]['bp_1']
                bp_2 = df_f.iloc[-1]['bp_2']
                sim.place_limit_order('buy', open_size, bp_1)
                sim.place_limit_order('buy', open_size, bp_2)
                result["action"] = f"placed_buy_orders_bp1={bp_1:.2f}_bp2={bp_2:.2f}"

            elif (sig == 'SELL') and (bid < last_sma15):
                sp_1 = df_f.iloc[-1]['sp_1']
                sp_2 = df_f.iloc[-1]['sp_2']
                sim.place_limit_order('sell', open_size, sp_1)
                sim.place_limit_order('sell', open_size, sp_2)
                result["action"] = f"placed_sell_orders_sp1={sp_1:.2f}_sp2={sp_2:.2f}"

            else:
                result["action"] = f"no_entry_sig={sig}_sma_mismatch"

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
    sim = get_sma_bot()

    logger.info(f"[SMA BOT] Starting — {SYMBOL} | Size={POS_SIZE} | TP={TARGET_PCT}% SL={MAX_LOSS_PCT}% | Lev={LEVERAGE}x")
    logger.info(f"[SMA BOT] Balance: ${sim.balance:.2f} | Log: {SIM_LOG_PATH}")

    sched.every(SCAN_INTERVAL).seconds.do(scan_cycle)

    while True:
        try:
            sched.run_pending()
            time.sleep(1)
        except KeyboardInterrupt:
            logger.info("[SMA BOT] Shutting down...")
            sim.save()
            break
        except Exception as e:
            logger.error(f"[SMA BOT] Error: {e}")
            time.sleep(30)
