"""
risk_model.py
────────────────────────────────────────────
Phase 7 — HPB–TCT Risk Management & Signal Generation
Implements dynamic risk weighting and signal bias computation.
────────────────────────────────────────────
"""

import numpy as np
from datetime import datetime

def smooth_confidence(conf_history, window=5):
    """Smooth recent confidence levels using a rolling average."""
    if len(conf_history) < 1:
        return 0.0
    return float(np.mean(conf_history[-window:]))

def volatility_band(prices, multiplier=1.5):
    """Compute volatility-based stop and take-profit bands."""
    if len(prices) < 2:
        return (0, 0)
    std = np.std(prices[-50:])
    return (prices[-1] - multiplier * std, prices[-1] + multiplier * std)

def derive_signal(conf, bias, reward_summary):
    """
    Derive a structured trade signal.
    - conf: recent execution confidence
    - bias: accumulation/distribution phase
    - reward_summary: current reward summary (qualitative)
    """
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

def compute_risk_profile(context):
    """
    Build a risk profile dict based on the latest live_context_cache.
    """
    try:
        conf = context.get("ExecutionConfidence_Total", 0)
        gates = context.get("gates", {})
        phase = gates.get("1A", {}).get("bias", "Neutral").title()
        reward_summary = context.get("Reward_Summary", "N/A")

        conf_smooth = smooth_confidence(context.setdefault("_conf_history", []))
        conf_weight = round((conf + conf_smooth) / 2, 3)

        # Estimate volatility-based bands
        prices = np.random.normal(45000, 500, 100)  # placeholder (replace with live later)
        stop, take = volatility_band(prices)

        signal = derive_signal(conf_weight, phase, reward_summary)
        risk_score = round(conf_weight * (1.5 if signal in ["BUY", "SELL"] else 1.0), 3)

        profile = {
            "timestamp": datetime.utcnow().isoformat(),
            "phase": phase,
            "signal": signal,
            "risk_score": risk_score,
            "confidence_smoothed": conf_weight,
            "stop_loss": round(float(stop), 2),
            "take_profit": round(float(take), 2),
        }

        # Append confidence to history for smoothing
        context["_conf_history"].append(conf)
        if len(context["_conf_history"]) > 100:
            context["_conf_history"].pop(0)

        return profile

    except Exception as e:
        return {"error": str(e)}
