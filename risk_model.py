"""
risk_model.py
────────────────────────────────────────────
Phase 7.3 — Multi-Timeframe Risk Model
Integrates 15m + 1h + 4h OKX data into a composite risk profile.
────────────────────────────────────────────
"""

import numpy as np
import httpx
from datetime import datetime

# ───────────────────────────────
# Utility Functions
# ───────────────────────────────
async def fetch_live_prices(symbol="BTC-USDT", interval="1H", limit=100):
    """Fetch recent closing prices from OKX."""
    url = f"https://www.okx.com/api/v5/market/candles?instId={symbol}-SWAP&bar={interval}&limit={limit}"
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(url)
    data = r.json().get("data", [])
    if not data:
        return []
    closes = [float(row[4]) for row in data]
    return closes[::-1]

def smooth_confidence(history, window=5):
    return float(np.mean(history[-window:])) if len(history) else 0.0

def volatility(prices):
    if len(prices) < 2:
        return 0.0
    return np.std(prices[-50:]) / np.mean(prices[-50:])

def derive_signal(conf, bias, reward):
    if conf < 0.2:
        return "WAIT"
    if "NEG" in str(reward).upper():
        return "SELL" if bias == "Distribution" else "WAIT"
    if "POS" in str(reward).upper():
        return "BUY" if bias == "Accumulation" else "WAIT"
    if conf > 0.7 and bias == "Accumulation":
        return "BUY"
    if conf > 0.7 and bias == "Distribution":
        return "SELL"
    return "HOLD"

# ───────────────────────────────
# Composite Risk Profile
# ───────────────────────────────
async def compute_risk_profile(context):
    """
    Compute multi-timeframe risk metrics and composite signal.
    """
    try:
        conf = context.get("ExecutionConfidence_Total", 0.0)
        gates = context.get("gates", {})
        phase = gates.get("1A", {}).get("bias", "Neutral").title()
        reward = context.get("Reward_Summary", "N/A")

        context.setdefault("_conf_history", []).append(conf)
        if len(context["_conf_history"]) > 100:
            context["_conf_history"].pop(0)
        conf_smooth = smooth_confidence(context["_conf_history"])
        conf_weight = round((conf + conf_smooth) / 2, 3)

        # Fetch multi-timeframe prices
        tf_map = {"15m": 0.3, "1H": 0.5, "4H": 0.2}   # weights
        tf_data = {}
        async with httpx.AsyncClient(timeout=10) as client:
            for tf in tf_map.keys():
                url = f"https://www.okx.com/api/v5/market/candles?instId=BTC-USDT-SWAP&bar={tf}&limit=100"
                resp = await client.get(url)
                data = resp.json().get("data", [])
                closes = [float(r[4]) for r in data][::-1]
                tf_data[tf] = closes

        # Compute volatility & price per TF
        vols, prices = {}, {}
        for tf, arr in tf_data.items():
            vols[tf] = volatility(arr)
            prices[tf] = arr[-1] if arr else np.nan

        # Weighted volatility score
        v_score = sum(vols[tf] * w for tf, w in tf_map.items())
        price_now = np.nanmean(list(prices.values()))
        risk_weight = min(v_score * (1 + conf_weight), 3.0)

        signal = derive_signal(conf_weight, phase, reward)

        stop = price_now * (1 - 0.015 * risk_weight)
        take = price_now * (1 + 0.015 * risk_weight)

        profile = {
            "timestamp": datetime.utcnow().isoformat(),
            "phase": phase,
            "signal": signal,
            "confidence_smoothed": conf_weight,
            "risk_score": round(risk_weight, 3),
            "volatility_score": round(v_score, 4),
            "stop_loss": round(stop, 2),
            "take_profit": round(take, 2),
            "current_price": round(price_now, 2),
            "timeframes": {
                tf: {
                    "price": round(prices[tf], 2),
                    "volatility": round(vols[tf], 5)
                } for tf in tf_map
            }
        }

        print(f"[RISK-MTF] {signal} | {price_now:.2f} | v={v_score:.4f} | r={risk_weight:.2f}")
        return profile

    except Exception as e:
        print(f"[RISK-MTF ERROR] {e}")
        return {"error": str(e)}
