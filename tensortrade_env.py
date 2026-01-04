import pandas as pd
import requests
from datetime import datetime
import numpy as np


class HPBContextualReward:
    """A lightweight custom reward class."""
    def __init__(self):
        self.last_price = None

    def compute(self, current_price):
        if self.last_price is None:
            self.last_price = current_price
            return 0.0
        reward = (current_price - self.last_price) / self.last_price
        self.last_price = current_price
        return reward


class HPB_TensorTrade_Env:
    """Simplified self-contained environment — no TensorTrade internals."""
    def __init__(self, symbol="BTC-USDT-SWAP", interval="1H", window=100):
        self.symbol = symbol
        self.interval = interval
        self.window = window
        self.feed = None
        self.reward_scheme = HPBContextualReward()
        print(f"[INIT] HPB–TCT (Lite) Environment initialized ({symbol}, {interval})")

    def okx_get_candles(self, limit=300):
        """Fetch and clean OKX market data."""
        url = f"https://www.okx.com/api/v5/market/candles?instId={self.symbol}&bar={self.interval}&limit={limit}"
        res = requests.get(url)
        data = res.json()

        if "data" not in data:
            raise ValueError(f"Invalid OKX response: {data}")

        candles = list(reversed(data["data"]))
        df = pd.DataFrame([c[:6] for c in candles],
                          columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype(float), unit="ms", utc=True)
        df = df.astype({
            "open": float, "high": float, "low": float, "close": float, "volume": float
        })
        print(f"[OKX] Loaded {len(df)} candles from OKX.")
        return df

    def build_feed(self):
        """Creates a clean feed DataFrame."""
        df = self.okx_get_candles()
        self.feed = df
        print("[BUILD_FEED] Feed successfully built.")
        return df

    def simulate_training(self, episodes=10):
        """Fake 'training' loop — keeps your flow but works standalone."""
        if self.feed is None:
            self.build_feed()

        prices = self.feed["close"].values
        total_reward = 0
        for i in range(1, min(episodes, len(prices))):
            reward = self.reward_scheme.compute(prices[i])
            total_reward += reward
        print(f"[TRAIN] Completed {episodes} pseudo-episodes. Reward: {total_reward:.6f}")

    def fetch_live_price(self):
        """Gets the latest OKX ticker price."""
        url = f"https://www.okx.com/api/v5/market/ticker?instId={self.symbol}"
        res = requests.get(url)
        data = res.json()
        try:
            price = float(data["data"][0]["last"])
            print(f"[LIVE] Current {self.symbol} price: {price}")
            return price
        except Exception:
            print("[LIVE] Failed to fetch live price.")
            return None
