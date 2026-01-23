@app.get(”/api/signals”)
async def get_trading_signals():
“””
Complete Trading Signals Endpoint
Combines gate validation + TCT model detection for actionable signals
“””
try:
from tct_model_detector import detect_tct_models

```
    htf_df = await fetch_mexc_candles(SYMBOL, "4h", 100)
    ltf_df = await fetch_mexc_candles(SYMBOL, "15m", 200)
    
    if htf_df is None or ltf_df is None:
        return JSONResponse({"error": "Failed to fetch market data", "signals": []}, status_code=500)
    
    current_price = float(ltf_df.iloc[-1]['close'])
    
    models = detect_tct_models(ltf_df)
    
    best_acc = models["accumulation_models"][0] if models["accumulation_models"] else None
    best_dist = models["distribution_models"][0] if models["distribution_models"] else None
    
    signals = []
    
    # Validate accumulation model
    if best_acc:
        ltf_candles = []
        for idx, row in ltf_df.iterrows():
            ltf_candles.append({
                'open_time': str(row['open_time']),
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': float(row['volume'])
            })
        
        acc_range = await detect_best_range(ltf_candles)
        context = {
            "htf_candles": htf_df,
            "ltf_candles": ltf_df,
            "detected_range": acc_range,
            "current_price": current_price
        }
        
        validation = validate_gates(context)
        
        if validation["Action"] == "EXECUTE" and validation["1A"]["bias"] == "bullish":
            signals.append({
                "signal_type": "BUY",
                "model": best_acc["model_type"],
                "entry_zone": best_acc["range_low"],
                "target": best_acc["target"],
                "stop_loss": best_acc["invalidation"],
                "confidence": validation["ExecutionConfidence_Total"],
                "quality_score": best_acc["quality_score"],
                "duration_hours": best_acc["duration_hours"],
                "current_price": current_price,
                "risk_reward": round((best_acc["target"] - current_price) / (current_price - best_acc["invalidation"]), 2) if current_price > best_acc["invalidation"] else 0
            })
    
    # Validate distribution model
    if best_dist:
        ltf_candles = []
        for idx, row in ltf_df.iterrows():
            ltf_candles.append({
                'open_time': str(row['open_time']),
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': float(row['volume'])
            })
        
        dist_range = await detect_best_range(ltf_candles)
        context = {
            "htf_candles": htf_df,
            "ltf_candles": ltf_df,
            "detected_range": dist_range,
            "current_price": current_price
        }
        
        validation = validate_gates(context)
        
        if validation["Action"] == "EXECUTE" and validation["1A"]["bias"] == "bearish":
            signals.append({
                "signal_type": "SELL",
                "model": best_dist["model_type"],
                "entry_zone": best_dist["range_high"],
                "target": best_dist["target"],
                "stop_loss": best_dist["invalidation"],
                "confidence": validation["ExecutionConfidence_Total"],
                "quality_score": best_dist["quality_score"],
                "duration_hours": best_dist["duration_hours"],
                "current_price": current_price,
                "risk_reward": round((current_price - best_dist["target"]) / (best_dist["invalidation"] - current_price), 2) if best_dist["invalidation"] > current_price else 0
            })
    
    return JSONResponse({
        "symbol": SYMBOL,
        "timestamp": datetime.utcnow().isoformat(),
        "signals": signals,
        "total_signals": len(signals),
        "models_analyzed": {
            "accumulation": len(models["accumulation_models"]),
            "distribution": len(models["distribution_models"])
        }
    })

except Exception as e:
    logger.error(f"[SIGNALS_ERROR] {e}")
    return JSONResponse({"error": str(e), "signals": []}, status_code=500)
```

@app.get(”/api/scan”)
async def manual_scan():
“”“Manual range scan”””
try:
ltf_df = await fetch_mexc_candles(SYMBOL, “15m”, 200)
htf_df = await fetch_mexc_candles(SYMBOL, “4h”, 200)

```
    results = {"LTF": [], "HTF": []}
    
    if ltf_df is not None:
        ltf_candles = ltf_df.to_dict('records')
        ltf_range = await detect_best_range(ltf_candles)
        if ltf_range:
            results["LTF"] = [ltf_range]
    
    if htf_df is not None:
        htf_candles = htf_df.to_dict('records')
        htf_range = await detect_best_range(htf_candles)
        if htf_range:
            results["HTF"] = [htf_range]
    
    return JSONResponse(results)
except Exception as e:
    logger.error(f"[SCAN_ERROR] {e}")
    return JSONResponse({"error": str(e)}, status_code=500)
```

@app.get(”/api/price”)
async def live_price():
“”“Fetch current MEXC ticker price”””
url = f”{MEXC_URL_BASE}/api/v3/ticker/price?symbol={SYMBOL}”
try:
async with httpx.AsyncClient(timeout=10) as c:
r = await c.get(url)
if r.status_code == 200:
data = r.json()
return {“symbol”: SYMBOL, “price”: float(data[“price”])}
else:
return {“error”: f”HTTP {r.status_code}”}
except Exception as e:
logger.error(f”[PRICE_FAIL] {e}”)
return {“error”: str(e)}

@app.get(”/api/train”)
async def run_training(episodes: int = 5):
“”“Run TensorTrade AutoLearn training”””
try:
from tensortrade_env import HPB_TensorTrade_Env

```
    env = HPB_TensorTrade_Env(symbol=SYMBOL, interval="1h")
    result = env.auto_train(episodes)
    
    return JSONResponse(result)
except ImportError:
    return JSONResponse({
        "error": "TensorTrade environment module not found",
        "episodes": 0
    }, status_code=500)
except Exception as e:
    logger.error(f"[TRAIN_ERROR] {e}")
    return JSONResponse({"error": str(e)}, status_code=500)
```

# ================================================================

# ENTRY POINT

# ================================================================

if **name** == “**main**”:
import uvicorn
uvicorn.run(“server_mexc:app”, host=“0.0.0.0”, port=PORT, reload=True)