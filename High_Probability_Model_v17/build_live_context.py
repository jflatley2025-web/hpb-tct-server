import requests
import pandas as pd
import json


def okx_get_candles(instId="BTC-USDT-SWAP", bar="1H", limit=200):
    """Fetch and sanitize OKX candle data for TensorTrade."""
    print("[DEBUG] Using updated okx_get_candles function (fixing 9→6 column mismatch)")

    OKX_BASE = "https://www.okx.com/api/v5"
    url = f"{OKX_BASE}/market/candles?instId={instId}&bar={bar}&limit={limit}"
    response = requests.get(url)
    data = response.json()

    if "data" not in data:
        raise ValueError("Invalid OKX response: " + json.dumps(data))

    candles = list(reversed(data["data"]))
    print("[DEBUG] First candle (raw):", candles[0])
    print("[DEBUG] Candle length:", len(candles[0]))

    df = pd.DataFrame([c[:6] for c in candles],
                      columns=["timestamp", "open", "high", "low", "close", "volume"])

    df["timestamp"] = pd.to_datetime(df["timestamp"].astype(float), unit="ms", utc=True)
    df = df.astype({
        "open": float,
        "high": float,
        "low": float,
        "close": float,
        "volume": float
    })

    print("[DEBUG] Final DataFrame shape:", df.shape)
    return df


def build_live_context(source_preference="OKX"):
    print(f"[DEBUG] build_live_context() invoked with source: {source_preference}")

    df = okx_get_candles(instId="BTC-USDT-SWAP", bar="1H", limit=200)
    print(f"[DEBUG] DataFrame built with shape: {df.shape}")

    context = {
        "exchange": "OKX",
        "symbol": "BTC-USDT-SWAP",
        "interval": "1H",
        "data": df
    }

    print("[DEBUG] Live context constructed successfully.")
    return context
