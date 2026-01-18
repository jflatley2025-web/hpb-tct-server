"""
hpb_nvs_module.py
High-Probability Model – Narrative-Volatility Synchronization (NVS)

Usage:
    from hpb_nvs_module import get_nvs_score
    nvs = get_nvs_score(bearer_token="YOUR_X_BEARER_TOKEN", htf_bias="bullish")
    print(nvs)
"""

import os, math, requests, re
from datetime import datetime, timedelta

# ---------------------------------------------------------------------
# Core NVS logic
# ---------------------------------------------------------------------

def fetch_tweets(bearer_token: str, query: str, max_results: int = 50):
    """Fetch tweets for a given query using X API v2 recent search endpoint."""
    url = "https://api.x.com/2/tweets/search/recent"
    headers = {"Authorization": f"Bearer {bearer_token}"}
    params = {
        "query": query,
        "max_results": max_results,
        "tweet.fields": "created_at,lang,text,public_metrics"
    }
    resp = requests.get(url, headers=headers, params=params, timeout=10)
    if resp.status_code != 200:
        raise RuntimeError(f"X API error {resp.status_code}: {resp.text}")
    return [t["text"] for t in resp.json().get("data", [])]


def sentiment_score(texts):
    """Very lightweight polarity heuristic for crypto tweets."""
    bull_words = ["buy", "long", "moon", "pump", "bull", "accum", "breakout"]
    bear_words = ["sell", "short", "dump", "bear", "down", "crash"]
    bull, bear = 0, 0
    for t in texts:
        txt = t.lower()
        bull += sum(w in txt for w in bull_words)
        bear += sum(w in txt for w in bear_words)
    total = bull + bear
    if total == 0:
        return 0.5   # neutral
    return bull / total   # 1 = very bullish, 0 = very bearish


def get_nvs_score(bearer_token: str, htf_bias: str = "neutral"):
    """
    Return a dict with NVS metrics.
    htf_bias: "bullish" | "bearish" | "neutral"
    """
    tickers = ["$BTC", "$ETH", "$SOL", "$DOGE", "$SUI", "$PEPE", "$AI"]
    all_texts = []
    for t in tickers:
        try:
            all_texts += fetch_tweets(bearer_token, t)
        except Exception:
            continue

    if not all_texts:
        return {"timestamp": datetime.utcnow().isoformat(),
                "status": "NO_DATA", "NVS_score": 0.0}

    sentiment = sentiment_score(all_texts)

    # map bias to numerical direction
    bias_dir = {"bullish": 1, "bearish": 0}.get(htf_bias, 0.5)
    context_match = 1 - abs(sentiment - bias_dir)     # 1 = aligned
    volatility_spread = max(0.05, abs(0.5 - sentiment))  # pseudo-spread proxy
    narrative_strength = min(1.0, math.log1p(len(all_texts)) / 5)

    nvs_score = (narrative_strength * context_match) / (1 + volatility_spread)
    nvs_score = round(min(1.0, max(0.0, nvs_score)), 3)

    return {
        "timestamp": datetime.utcnow().isoformat(),
        "tweets_analyzed": len(all_texts),
        "sentiment": round(sentiment, 3),
        "htf_bias": htf_bias,
        "context_match": round(context_match, 3),
        "narrative_strength": round(narrative_strength, 3),
        "volatility_spread": round(volatility_spread, 3),
        "NVS_score": nvs_score,
        "status": "OK"
    }

# ---------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------
if __name__ == "__main__":
    token = os.getenv("X_BEARER_TOKEN", "")
    if not token:
        print("Set environment variable X_BEARER_TOKEN or pass manually.")
    else:
        print(get_nvs_score(token, "bullish"))
