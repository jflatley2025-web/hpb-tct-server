"""
risk_model.py
────────────────────────────────────────────
Phase 7.2 — HPB–TCT Risk Management (Live Market Integration)
Uses live BTC-USDT OHLC data from OKX for volatility and stop/take levels.
────────────────────────────────────────────
"""

import numpy as np
import httpx
from datetime import datetime

async def fetch_live_prices(symbol="BTC-USDT", interval="1H", limit=100):
    """Fetch latest prices from OKX (live candles)."""
    try:
        url = f"https://www.okx.com/api/v5/market/candles?instId={symbol}-SWAP&bar={interval}&limit={limit}"
        async with httpx.AsyncClient() as client:
            r = await client.get(url)
        data = r.json().get("data", [])
        if not data:
            return []
        closes = [float(row[4]) for row in data]
        return closes[::-1]  # reverse chronological order
    except Exception as e:
        print(f"[RISK_MODEL ERROR] fetch_live_prices: {e}")
        return []

def smooth_confidence(conf_history, window=5):
    """Smooth recent confidence levels using a rolling average."""
    if len(conf_history) < 1:
        return 0.0
    return float(np.mean(conf_history[-window:]))

def volatility_band(prices, multiplier=1.8):
    """Compute stop-loss and take-profit using rolling volatility."""
    if len(prices) < 2:
        return (0, 0)
    std = np.std(prices[-50:])
    mid = prices[-1]
    return (mid - multiplier * std, mid + multiplier * std)

def derive_signal(conf, bias, reward_summary, price):
    """Compute the trade bias signal."""
    if conf < 0.2:
        return "WAIT"
    if "NEG" in str(reward_summary).upper():
        return "SELL" if bias == "Distribution" else "WAIT"
    if "POS" in str(reward_summary).upper():
        return "BUY" if bias == "Accumulation" else "WAIT"
    if conf > 0.7 and bias == "Accumulation":
        return "BUY"
    elif conf > 0.7 and bias == "Distribution":
        return "SELL"
    return "HOLD"

async def compute_risk_profile(context):
    """
    Compute a full risk profile from live BTC-USDT data and algorithmic bias.
    """
    try:
        conf = context.get("ExecutionConfidence_Total", 0)
        gates = context.get("gates", {})
        phase = gates.get("1A", {}).get("bias", "Neutral").title()
        reward_summary = context.get("Reward_Summary", "N/A")

        conf_smooth = smooth_confidence(context.setdefault("_conf_history", []))
        conf_weight = round((conf + conf_smooth) / 2, 3)

        # Fetch live BTC-USDT prices
        prices = await fetch_live_prices("BTC-USDT", "1H", 100)
        if not prices:
            raise ValueError("No live data available from OKX.")

        current_price = prices[-1]
        stop, take = volatility_band(prices)

        signal = derive_signal(conf_weight, phase, reward_summary, current_price)
        risk_score = round(conf_weight * (1.5 if signal in ["BUY", "SELL"] else 1.0), 3)

        profile = {
            "timestamp": datetime.utcnow().isoformat(),
            "phase": phase,
            "signal": signal,
            "risk_score": risk_score,
            "confidence_smoothed": conf_weight,
            "stop_loss": round(float(stop), 2),
            "take_profit": round(float(take), 2),
            "current_price": round(float(current_price), 2),
        }

        # Track confidence history for smoothing
        context["_conf_history"].append(conf)
        if len(context["_conf_history"]) > 100:
            context["_conf_history"].pop(0)

        print(f"[RISK] Signal={signal} | Price={current_price:.2f} | Stop={stop:.2f} | Take={take:.2f}")
        return profile

    except Exception as e:
        return {"error": str(e)}
