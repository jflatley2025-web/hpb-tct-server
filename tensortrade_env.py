import os
import pandas as pd
import requests
import numpy as np
import random
from datetime import datetime

# ================================================================
# GLOBAL CONFIG — MEXC integration
# ================================================================
MEXC_URL_BASE = os.getenv("MEXC_URL_BASE", "https://api.mexc.com")
SYMBOL = os.getenv("SYMBOL", "BTCUSDT")
MEXC_KEY = os.getenv("MEXC_KEY")
MEXC_SECRET = os.getenv("MEXC_SECRET")
AUTH_MODE = bool(MEXC_KEY and MEXC_SECRET)

print(f"[INIT] HPB–TCT v21.2 TensorTrade Environment (Exchange=MEXC)")
print(f"[MEXC] Auth Mode: {'🔒 PRIVATE' if AUTH_MODE else '🌐 PUBLIC'} | Symbol={SYMBOL}")

VALID_INTERVALS = {
    "1m","3m","5m","15m","30m","1h","2h","4h","6h","8h",
    "12h","1d","3d","1w","1M"
}


# ================================================================
# HPBContextualReward — lightweight reward computation
# ================================================================
class HPBContextualReward:
    """A lightweight contextual reward class."""
    def __init__(self):
        self.last_price = None

    def compute(self, current_price):
        """Reward is percentage change from last tick."""
        if self.last_price is None:
            self.last_price = current_price
            return 0.0
        reward = (current_price - self.last_price) / self.last_price
        self.last_price = current_price
        return reward


# ================================================================
# HPB_TensorTrade_Env — unified simulation + training environment
# ================================================================
class HPB_TensorTrade_Env:
    """Unified environment for HPB–TCT AutoLearn (MEXC Feed Version)."""

    def __init__(self, symbol=SYMBOL, interval="1h", window=100):
        self.symbol = symbol
        self.interval = interval
        self.window = window
        self.feed = None
        self.reward_scheme = HPBContextualReward()
        self.current_step = 0
        self.total_episodes = 0
        print(f"[INIT] HPB–TCT Environment initialized ({symbol}, {interval})")

    # ============================================================
    # MEXC Candle Fetch
    # ============================================================
    def mexc_get_candles(self, limit=300):
        """Fetch OHLCV candles from MEXC."""
        interval = self.interval.lower()
        if interval not in VALID_INTERVALS:
            print(f"[MEXC] Unsupported interval '{interval}' — using 1h fallback.")
            interval = "1h"

        url = f"{MEXC_URL_BASE}/api/v3/klines"
        params = {"symbol": self.symbol, "interval": interval, "limit": limit}

        try:
            res = requests.get(url, params=params, timeout=15)
            if res.status_code != 200:
                print(f"[MEXC_FAIL] {interval} — HTTP {res.status_code}")
                return None

            data = res.json()
            if not isinstance(data, list) or len(data) == 0:
                print(f"[MEXC_WARN] Empty response for {self.symbol} {interval}")
                return None

            df = pd.DataFrame(data, columns=[
                "open_time","open","high","low","close","volume",
                "close_time","quote_vol","trades","taker_base",
                "taker_quote","ignore"
            ])
            df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
            df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
            df = df.astype({
                "open": float, "high": float, "low": float,
                "close": float, "volume": float
            })
            df = df[["open_time","open","high","low","close","volume"]].sort_values("open_time")
            print(f"[MEXC] Loaded {len(df)} candles ({self.symbol}, {interval})")
            return df

        except Exception as e:
            print(f"[MEXC_ERR] {e}")
            return None

    # ============================================================
    # Feed Builder
    # ============================================================
    def build_feed(self):
        """Creates and stores a DataFrame feed."""
        df = self.mexc_get_candles()
        if df is None or df.empty:
            raise ValueError("[BUILD_FEED] Failed to build feed from MEXC.")
        self.feed = df
        print("[BUILD_FEED] Feed successfully built from MEXC.")
        return df

    # ============================================================
    # Live Price Fetch
    # ============================================================
    def fetch_live_price(self):
        """Get current market price from MEXC ticker."""
        url = f"{MEXC_URL_BASE}/api/v3/ticker/price"
        params = {"symbol": self.symbol}
        try:
            res = requests.get(url, params=params, timeout=10)
            data = res.json()
            price = float(data.get("price", 0))
            print(f"[LIVE] {self.symbol} price: {price}")
            return price
        except Exception as e:
            print(f"[LIVE] Failed to fetch live price: {e}")
            return None

    # ============================================================
    # Normalization Utility
    # ============================================================
    def normalize_feed(self, df):
        """Normalize OHLCV columns to [0,1] for TensorTrade compatibility."""
        if df is None or df.empty:
            return np.zeros((1, 5))
        arr = df[["open","high","low","close","volume"]].values
        min_v = np.min(arr, axis=0)
        max_v = np.max(arr, axis=0)
        norm = (arr - min_v) / (max_v - min_v + 1e-9)
        return norm

    # ============================================================
    # AutoLearn Core Training Entry Point
    # ============================================================
    def auto_train(self, episodes: int = 5):
        """
        HPB AutoLearn cycle runner.
        Called directly from /train and /bot/train endpoints.
        """
        print(f"[AUTO_TRAIN] Starting {episodes} AutoLearn episodes (MEXC feed)...")
        if self.feed is None:
            self.build_feed()

        prices = self.feed["close"].values
        rewards = []

        for ep in range(episodes):
            reward = self.run_cycle(ep, prices)
            rewards.append(reward)
            self.total_episodes += 1

        avg_reward = float(np.mean(rewards)) if rewards else 0.0
        bias = "bullish" if avg_reward > 0 else "bearish" if avg_reward < 0 else "neutral"
        confidence = round(abs(avg_reward) * 100, 4)

        print(f"[AUTO_TRAIN] Completed {episodes} episodes. Bias={bias}, Confidence={confidence}%")
        return {
            "episodes": episodes,
            "avg_reward": avg_reward,
            "last_bias": bias,
            "confidence": confidence
        }

    # ============================================================
    # Internal Learning Loop
    # ============================================================
    def run_cycle(self, episode: int, prices=None):
        """Simulate one learning cycle — simple price reward loop."""
        if prices is None:
            if self.feed is None:
                self.build_feed()
            prices = self.feed["close"].values

        total_reward = 0
        for i in range(1, len(prices)):
            total_reward += self.reward_scheme.compute(prices[i])

        noise = random.uniform(-0.001, 0.001)  # pseudo-random learning noise
        total_reward += noise

        print(f"[RUN_CYCLE] Episode {episode + 1} reward: {total_reward:.6f}")
        return total_reward
