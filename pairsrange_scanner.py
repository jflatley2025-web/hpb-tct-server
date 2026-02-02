import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# =========================
# CONFIG
# =========================

TIMEFRAMES = {
    "1m": 300,
    "5m": 300,
    "15m": 300,
    "1h": 300,
    "4h": 300,
    "1d": 200,
    "3d": 200,
}

MIN_RANGE_HOURS = 24
V_SHAPE_THRESHOLD = 0.65      # % retrace too fast = V-shape
MAX_INTERNAL_VOL = 0.35       # too much internal volatility = bad range

# =========================
# EXCHANGE
# =========================

exchange = ccxt.bybit({
    "enableRateLimit": True,
    "options": {"defaultType": "swap"}
})


# =========================
# DATA FETCH
# =========================

def fetch_ohlcv(symbol, timeframe, limit):
    try:
        data = exchange.fetch_ohlcv(
            symbol,
            timeframe=timeframe,
            limit=limit
        )
        df = pd.DataFrame(
            data,
            columns=["ts", "open", "high", "low", "close", "volume"]
        )
        df["ts"] = pd.to_datetime(df["ts"], unit="ms")
        return df
    except Exception:
        return None


# =========================
# RANGE DETECTION
# =========================

def detect_range(df):
    """
    Detects if HTF data forms a valid range.
    Returns dict with range info or None.
    """

    high = df["high"].max()
    low = df["low"].min()
    range_size = high - low

    if range_size <= 0:
        return None

    # duration
    duration_hours = (df["ts"].iloc[-1] - df["ts"].iloc[0]).total_seconds() / 3600
    if duration_hours < MIN_RANGE_HOURS:
        return None

    # V-shape detection
    mid_idx = len(df) // 2
    first_leg = abs(df["close"].iloc[mid_idx] - df["close"].iloc[0])
    second_leg = abs(df["close"].iloc[-1] - df["close"].iloc[mid_idx])

    if first_leg > 0 and (second_leg / first_leg) > V_SHAPE_THRESHOLD:
        return None

    # internal volatility (chop detector)
    internal_vol = df["close"].std() / range_size
    if internal_vol > MAX_INTERNAL_VOL:
        return None

    return {
        "range_high": high,
        "range_low": low,
        "range_eq": (high + low) / 2,
        "duration_hours": duration_hours,
        "internal_vol": internal_vol
    }


# =========================
# RANGE SCORE (RPS)
# =========================

def score_range(range_info):
    """
    Mechanical RPS scoring (0–10)
    """

    score = 10.0

    # duration bonus
    if range_info["duration_hours"] < 48:
        score -= 1.0
    elif range_info["duration_hours"] < 72:
        score -= 0.5

    # internal volatility penalty
    score -= range_info["internal_vol"] * 5

    return round(max(score, 0), 2)


# =========================
# MAIN SCAN
# =========================

def scan_pairs(pairs):
    stage_counts = {
        "total": len(pairs),
        "htf_range": 0,
        "valid_range": 0,
        "rps_9_6": 0
    }

    results = []

    for symbol in pairs:
        symbol = symbol.strip()
        if not symbol:
            continue

        df_1d = fetch_ohlcv(symbol, "1d", TIMEFRAMES["1d"])
        df_3d = fetch_ohlcv(symbol, "3d", TIMEFRAMES["3d"])

        if df_1d is None or df_3d is None:
            continue

        range_info = detect_range(df_1d)

        if not range_info:
            continue

        stage_counts["htf_range"] += 1

        rps = score_range(range_info)

        stage_counts["valid_range"] += 1

        if rps >= 9.6:
            stage_counts["rps_9_6"] += 1

            results.append({
                "symbol": symbol,
                "RPS": rps,
                "range_high": range_info["range_high"],
                "range_low": range_info["range_low"],
                "range_eq": range_info["range_eq"],
                "duration_hours": round(range_info["duration_hours"], 1)
            })

    return results, stage_counts


# =========================
# RUN
# =========================

if __name__ == "__main__":
    with open("pairs.txt") as f:
        pairs = f.readlines()

    results, counts = scan_pairs(pairs)

    print("\n=== SCAN SUMMARY ===")
    for k, v in counts.items():
        print(f"{k}: {v}")

    print("\n=== QUALIFIED RANGES (RPS >= 9.6) ===")
    for r in sorted(results, key=lambda x: -x["RPS"]):
        print(r)
