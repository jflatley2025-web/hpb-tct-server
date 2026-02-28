############# Stoch RSI + Nadarya-Watson Bot #6 — Simulated Trading via Phemex Live Data
#
# Strategy: Combines two indicator signals for entry and exit.
#
#   Entry (when flat):
#     BUY  if nadarya_buy_signal  OR stoch_rsi has been oversold  (< 10) once  in last 14 bars
#     SELL if nadarya_sell_signal OR stoch_rsi has been overbought (> 90) once  in last 14 bars
#
#   Exit (when in position):
#     Long  close if nadarya_sell_signal OR stoch_rsi overbought  twice in last 14 bars
#     Short close if nadarya_buy_signal  OR stoch_rsi oversold    twice in last 14 bars
#
#   Nadarya-Watson: Gaussian kernel regression over all candles →
#     smooth envelope; crossover below lower band = buy, above upper band = sell.
#   Stoch RSI: stochastic of RSI(14), smoothed → 0-100 oscillator.
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

logger = logging.getLogger("STOCH-NADARYA-BOT")

# ================================================================
# CONFIGURATION
# ================================================================
SYMBOL = 'BTC/USD:BTC'
TIMEFRAME = '1h'            # original: '1h'
CANDLE_LIMIT = 100          # bars to fetch (valid Phemex discrete limit)
RSI_PERIOD = 14             # RSI look-back
STOCH_PERIOD = 14           # stochastic look-back applied to RSI
SMOOTH_K = 3                # SMA smoothing on %K
RSI_TARGETS = [10, 90]      # [oversold, overbought] thresholds
RSI_WINDOW = 14             # bars to check for oversold/overbought count
NW_BANDWIDTH = 8.0          # Gaussian kernel bandwidth (higher = smoother)
NW_MULT = 3.0               # envelope half-width in residual std devs
POS_SIZE = 30               # position size in contracts
LEVERAGE = 10               # simulated leverage multiplier
STARTING_BALANCE = 1000     # simulated starting balance in USD
COOLDOWN_MINUTES = 10       # wait after a close before re-entering
SCAN_INTERVAL = 60          # seconds between scan cycles

SIM_LOG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'stoch_nadarya_trade_log.json'
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
        # Indicator state for the dashboard
        self.last_stoch_rsi: Optional[float] = None
        self.last_nadarya_buy: bool = False
        self.last_nadarya_sell: bool = False
        self.last_nw_est: Optional[float] = None
        self.last_nw_upper: Optional[float] = None
        self.last_nw_lower: Optional[float] = None
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
        logger.info(f"[SIM ORDER] {side.upper()} {size} {SYMBOL} @ ${price:.2f}")
        return order

    def cancel_all_orders(self):
        count = len(self.pending_orders)
        self.pending_orders = []
        if count > 0:
            logger.info(f"[SIM] Cancelled {count} pending order(s)")

    def check_pending_fills(self, ask: float, bid: float):
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
            'last_stoch_rsi': round(self.last_stoch_rsi, 2) if self.last_stoch_rsi is not None else None,
            'last_nadarya_buy': self.last_nadarya_buy,
            'last_nadarya_sell': self.last_nadarya_sell,
            'last_nw_est': round(self.last_nw_est, 2) if self.last_nw_est is not None else None,
            'last_nw_upper': round(self.last_nw_upper, 2) if self.last_nw_upper is not None else None,
            'last_nw_lower': round(self.last_nw_lower, 2) if self.last_nw_lower is not None else None,
            'last_error': self.last_error,
            'config': {
                'symbol': SYMBOL,
                'timeframe': TIMEFRAME,
                'rsi_targets': RSI_TARGETS,
                'rsi_window': RSI_WINDOW,
                'nw_bandwidth': NW_BANDWIDTH,
                'nw_mult': NW_MULT,
                'pos_size': POS_SIZE,
                'leverage': LEVERAGE,
                'scan_interval': SCAN_INTERVAL,
            },
        }


# ================================================================
# SINGLETON ACCESS
# ================================================================
_bot_instance: Optional[SimulatedTrader] = None


def get_stoch_nadarya_bot() -> SimulatedTrader:
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
    return ob['asks'][0][0], ob['bids'][0][0]


def fetch_candle_data() -> pd.DataFrame:
    """Fetch CANDLE_LIMIT 1h bars from Phemex."""
    phemex = _get_phemex()
    bars = phemex.fetch_ohlcv(SYMBOL, timeframe=TIMEFRAME, limit=CANDLE_LIMIT)
    df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df


# ================================================================
# NADARYA-WATSON KERNEL REGRESSION
# ================================================================

def calc_nadarya(df: pd.DataFrame) -> Tuple[bool, bool]:
    """Compute Gaussian kernel regression envelope and detect crossover signals.

    For each bar i the estimate is the kernel-weighted average of all closes:
      ŷ[i] = Σⱼ K(i,j) * close[j]  /  Σⱼ K(i,j)
      K(i,j) = exp(-0.5 * ((i-j) / bandwidth)²)

    Envelope:
      upper = ŷ + NW_MULT * std(close - ŷ)
      lower = ŷ - NW_MULT * std(close - ŷ)

    Signals (most recent completed bar vs the one before it):
      buy_signal:  close[-2] < lower[-2]  and  close[-1] > lower[-1]  (crossed above lower)
      sell_signal: close[-2] > upper[-2]  and  close[-1] < upper[-1]  (crossed below upper)

    Adds columns nw_est, nw_upper, nw_lower to df in-place.
    Returns (buy_signal, sell_signal).
    """
    close = df['close'].values.astype(float)
    n = len(close)

    # Build weight matrix vectorised — shape (n, n); W[i,j] = kernel(i,j)
    idx = np.arange(n, dtype=float)
    i_grid, j_grid = np.meshgrid(idx, idx, indexing='ij')   # (n, n)
    W = np.exp(-0.5 * ((i_grid - j_grid) / NW_BANDWIDTH) ** 2)
    est = (W @ close) / W.sum(axis=1)

    residuals = close - est
    std = float(np.std(residuals))

    upper = est + NW_MULT * std
    lower = est - NW_MULT * std

    df['nw_est']   = est
    df['nw_upper'] = upper
    df['nw_lower'] = lower

    # Use the last two bars for crossover detection
    buy_signal  = bool(close[-2] < lower[-2] and close[-1] > lower[-1])
    sell_signal = bool(close[-2] > upper[-2] and close[-1] < upper[-1])

    logger.info(f"[NW] est={est[-1]:.2f} upper={upper[-1]:.2f} lower={lower[-1]:.2f} "
                f"close={close[-1]:.2f} buy={buy_signal} sell={sell_signal}")

    return buy_signal, sell_signal


# ================================================================
# STOCHASTIC RSI
# ================================================================

def calc_stoch_rsi(df: pd.DataFrame) -> None:
    """Compute Stochastic RSI and add 'stoch_rsi' column to df in-place.

    Steps:
      1. RSI(close, RSI_PERIOD) using Wilder's EMA (com = RSI_PERIOD - 1)
      2. Stochastic of RSI over STOCH_PERIOD bars: 100*(RSI-min)/(max-min)
      3. Smooth with SMOOTH_K-period SMA
    """
    close = df['close']
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(com=RSI_PERIOD - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=RSI_PERIOD - 1, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    rsi_min = rsi.rolling(STOCH_PERIOD).min()
    rsi_max = rsi.rolling(STOCH_PERIOD).max()
    denom = (rsi_max - rsi_min).replace(0, np.nan)
    stoch = 100 * (rsi - rsi_min) / denom

    df['stoch_rsi'] = stoch.rolling(SMOOTH_K).mean()
    df['rsi'] = rsi


def is_oversold(series: pd.Series, window: int, times: int, target: float) -> bool:
    """True if series has been below target at least `times` times in the last `window` bars."""
    recent = series.dropna().iloc[-window:]
    return int((recent < target).sum()) >= times


def is_overbought(series: pd.Series, window: int, times: int, target: float) -> bool:
    """True if series has been above target at least `times` times in the last `window` bars."""
    recent = series.dropna().iloc[-window:]
    return int((recent > target).sum()) >= times


# ================================================================
# SCAN CYCLE — one full bot iteration
# ================================================================

def scan_cycle() -> Dict:
    """Run one full Stoch RSI + Nadarya-Watson cycle. Returns result dict for the dashboard."""
    sim = get_stoch_nadarya_bot()
    now_str = datetime.now(timezone.utc).isoformat()
    result = {"timestamp": now_str, "action": "monitoring"}

    try:
        # 1. Check pending order fills
        ask, bid = ask_bid()
        sim.last_bid = bid
        sim.check_pending_fills(ask, bid)

        # 2. Fetch indicators — needed for both exit and entry logic
        df = fetch_candle_data()
        nadarya_buy, nadarya_sell = calc_nadarya(df)
        calc_stoch_rsi(df)

        # Update indicator state for the dashboard
        sim.last_nadarya_buy  = nadarya_buy
        sim.last_nadarya_sell = nadarya_sell
        stoch_last = df['stoch_rsi'].dropna().iloc[-1] if df['stoch_rsi'].notna().any() else None
        sim.last_stoch_rsi = float(stoch_last) if stoch_last is not None else None
        sim.last_nw_est    = float(df['nw_est'].iloc[-1])
        sim.last_nw_upper  = float(df['nw_upper'].iloc[-1])
        sim.last_nw_lower  = float(df['nw_lower'].iloc[-1])

        logger.info(f"[INDICATORS] nadarya_buy={nadarya_buy} nadarya_sell={nadarya_sell} "
                    f"stoch_rsi={sim.last_stoch_rsi}")

        # 3. Exit logic — checked on every scan while in position
        if sim.in_position:
            is_long = sim.is_long
            entry = sim.entry_price
            diff = (bid - entry) if is_long else (entry - bid)
            perc = ((diff / entry) * LEVERAGE * 100) if entry > 0 else 0

            if is_long:
                # Close long: nadarya says sell OR stoch overbought twice in window
                exit_nadarya = nadarya_sell
                exit_stoch   = is_overbought(df['stoch_rsi'], RSI_WINDOW, times=2, target=RSI_TARGETS[1])
                if exit_nadarya or exit_stoch:
                    reason = 'nadarya_sell' if exit_nadarya else 'stoch_overbought_2x'
                    sim.close_position(bid, reason=reason)
                    result["action"] = f"closed_long_{reason}_pnl_{perc:+.2f}pct"
                else:
                    result["action"] = f"hold_long_pnl_{perc:+.2f}pct"
            else:
                # Close short: nadarya says buy OR stoch oversold twice in window
                exit_nadarya = nadarya_buy
                exit_stoch   = is_oversold(df['stoch_rsi'], RSI_WINDOW, times=2, target=RSI_TARGETS[0])
                if exit_nadarya or exit_stoch:
                    reason = 'nadarya_buy' if exit_nadarya else 'stoch_oversold_2x'
                    sim.close_position(ask, reason=reason)
                    result["action"] = f"closed_short_{reason}_pnl_{perc:+.2f}pct"
                else:
                    result["action"] = f"hold_short_pnl_{perc:+.2f}pct"

        # 4. Cooldown check
        if sim.last_close_time is not None:
            elapsed_min = (time.time() - sim.last_close_time) / 60
            if elapsed_min < COOLDOWN_MINUTES:
                result["action"] = f"cooldown_{elapsed_min:.0f}m"
                sim.last_cycle_time = now_str
                sim.last_cycle_action = result["action"]
                return result

        # 5. Entry logic — only when flat
        if not sim.in_position:
            # Refresh bid/ask after indicator computation
            ask, bid = ask_bid()
            sim.last_bid = bid

            oversold_entry  = is_oversold(df['stoch_rsi'],  RSI_WINDOW, times=1, target=RSI_TARGETS[0])
            overbought_entry = is_overbought(df['stoch_rsi'], RSI_WINDOW, times=1, target=RSI_TARGETS[1])

            if nadarya_buy or oversold_entry:
                signal_reason = 'nadarya_buy' if nadarya_buy else 'stoch_oversold'
                sim.last_signal = signal_reason.upper()
                sim.cancel_all_orders()
                sim.place_limit_order('buy', POS_SIZE, bid)
                result["action"] = f"buy_{signal_reason}_at_{bid:.2f}"

            elif nadarya_sell or overbought_entry:
                signal_reason = 'nadarya_sell' if nadarya_sell else 'stoch_overbought'
                sim.last_signal = signal_reason.upper()
                sim.cancel_all_orders()
                sim.place_limit_order('sell', POS_SIZE, bid)
                result["action"] = f"sell_{signal_reason}_at_{bid:.2f}"

            else:
                srsi_str = f"{sim.last_stoch_rsi:.1f}" if sim.last_stoch_rsi is not None else "N/A"
                sim.last_signal = 'NO_SIGNAL'
                result["action"] = f"no_signal_stoch={srsi_str}"

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
    sim = get_stoch_nadarya_bot()

    logger.info(f"[STOCH-NADARYA BOT] Starting — {SYMBOL} | {TIMEFRAME} | "
                f"RSI targets={RSI_TARGETS} window={RSI_WINDOW} | "
                f"NW bandwidth={NW_BANDWIDTH} mult={NW_MULT} | Lev={LEVERAGE}x")
    logger.info(f"[STOCH-NADARYA BOT] Balance: ${sim.balance:.2f} | Log: {SIM_LOG_PATH}")

    sched.every(SCAN_INTERVAL).seconds.do(scan_cycle)

    while True:
        try:
            sched.run_pending()
            time.sleep(1)
        except KeyboardInterrupt:
            logger.info("[STOCH-NADARYA BOT] Shutting down...")
            sim.save()
            break
        except Exception as e:
            logger.error(f"[STOCH-NADARYA BOT] Error: {e}")
            time.sleep(30)
