import pandas as pd
import requests
import numpy as np
import random
from datetime import datetime

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
    """Unified environment for HPB–TCT AutoLearn"""
    def __init__(self, symbol="BTC-USDT-SWAP", interval="1H", window=100):
        self.symbol = symbol
        self.interval = interval
        self.window = window
        self.feed = None
        self.reward_scheme = HPBContextualReward()
        self.current_step = 0
        self.total_episodes = 0
        print(f"[INIT] HPB–TCT (Lite) Environment initialized ({symbol}, {interval})")

    # ============================================================
    # OKX Candle Fetch
    # ============================================================
    def okx_get_candles(self, limit=300):
        """Fetch OHLCV candles from OKX."""
        url = f"https://www.okx.com/api/v5/market/candles?instId={self.symbol}&bar={self.interval}&limit={limit}"
        res = requests.get(url)
        data = res.json()

        if "data" not in data:
            raise ValueError(f"Invalid OKX response: {data}")

        candles = list(reversed(data["data"]))
        df = pd.DataFrame([c[:6] for c in candles],
                          columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype(float), unit="ms", utc=True)
        df = df.astype({"open": float, "high": float, "low": float, "close": float, "volume": float})
        print(f"[OKX] Loaded {len(df)} candles from OKX.")
        return df

    # ============================================================
    # Feed Builder
    # ============================================================
    def build_feed(self):
        """Creates and stores a DataFrame feed."""
        df = self.okx_get_candles()
        self.feed = df
        print("[BUILD_FEED] Feed successfully built.")
        return df

    # ============================================================
    # Live Price Fetch
    # ============================================================
    def fetch_live_price(self):
        """Get current market price from OKX ticker."""
        url = f"https://www.okx.com/api/v5/market/ticker?instId={self.symbol}"
        try:
            res = requests.get(url)
            data = res.json()
            price = float(data["data"][0]["last"])
            print(f"[LIVE] {self.symbol} price: {price}")
            return price
        except Exception as e:
            print(f"[LIVE] Failed to fetch live price: {e}")
            return None

    # ============================================================
    # AutoLearn Core Training Entry Point
    # ============================================================
    def auto_train(self, episodes: int = 5):
        """
        HPB AutoLearn cycle runner.
        Called directly from /train and /bot/train endpoints.
        """
        print(f"[AUTO_TRAIN] Starting {episodes} AutoLearn episodes...")
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
