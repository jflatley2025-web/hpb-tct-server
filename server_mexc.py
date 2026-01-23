"""
server_mexc.py — HPB–TCT v21.2 MEXC Feed + Range Detection + Gate Validation Server
Author: HPB–TCT Dev Team
Date: 2026-01-23

Features:
- MEXC REST API for all market data
- Auto-scans LTF/HTF ranges in background every 2 minutes
- Complete 7-gate validation system (1A, 1B, 1C, RCM, MSCE, RIG, 1D)
- TCT Model Detection integration
- Unified endpoints: /status, /api/ranges, /api/validate, /api/models, /api/signals, /api/train
- Compatible with Render environment variables
"""

import os
import asyncio
import logging
import httpx
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from loguru import logger

# ================================================================
# CONFIGURATION
# ================================================================
PORT = int(os.getenv("PORT", 10000))
SYMBOL = os.getenv("SYMBOL", "BTCUSDT").replace("/", "").replace("-", "").upper()
MEXC_KEY = os.getenv("MEXC_KEY", None)
MEXC_SECRET = os.getenv("MEXC_SECRET", None)
MEXC_URL_BASE = "https://api.mexc.com"

app = FastAPI(title="HPB–TCT v21.2 MEXC Server", version="21.2")

latest_ranges = {"LTF": [], "HTF": []}
scan_interval_sec = 120

# ================================================================
# LOGGING CONFIG
# ================================================================
logging.basicConfig(level=logging.INFO)
logger.info(f"[INIT] MEXC Auth Mode: {'🔒 PRIVATE' if MEXC_KEY and MEXC_SECRET else '🌐 PUBLIC'}")
logger.info(f"[INIT] HPB–TCT v21.2 Ready — Symbol={SYMBOL}, Port={PORT}")


# ================================================================
# MARKET STRUCTURE DETECTION
# ================================================================
class MarketStructure:
    """Detects market structure using 6-candle rule"""
    
    @staticmethod
    def detect_pivots(candles: pd.DataFrame) -> Dict:
        """Implements 6-candle rule: 2 consecutive bullish/bearish = pivot"""
        if len(candles) < 6:
            return {"highs": [], "lows": [], "trend": "neutral"}
        
        pivots = {"highs": [], "lows": []}
        
        for i in range(2, len(candles) - 2):
            # Pivot high
            if (candles.iloc[i-2]['close'] < candles.iloc[i-1]['close'] and
                candles.iloc[i-1]['close'] < candles.iloc[i]['close'] and
                candles.iloc[i]['close'] > candles.iloc[i+1]['close'] and
                candles.iloc[i+1]['close'] > candles.iloc[i+2]['close']):
                pivots["highs"].append({
                    "idx": i,
                    "price": candles.iloc[i]['high'],
                    "time": str(candles.iloc[i].get('open_time', ''))
                })
            
            # Pivot low
            if (candles.iloc[i-2]['close'] > candles.iloc[i-1]['close'] and
                candles.iloc[i-1]['close'] > candles.iloc[i]['close'] and
                candles.iloc[i]['close'] < candles.iloc[i+1]['close'] and
                candles.iloc[i+1]['close'] < candles.iloc[i+2]['close']):
                pivots["lows"].append({
                    "idx": i,
                    "price": candles.iloc[i]['low'],
                    "time": str(candles.iloc[i].get('open_time', ''))
                })
        
        # Determine trend
        if len(pivots["highs"]) >= 2 and len(pivots["lows"]) >= 2:
            recent_highs = [p["price"] for p in pivots["highs"][-2:]]
            recent_lows = [p["price"] for p in pivots["lows"][-2:]]
            
            if recent_highs[-1] > recent_highs[-2] and recent_lows[-1] > recent_lows[-2]:
                pivots["trend"] = "bullish"
            elif recent_highs[-1] < recent_highs[-2] and recent_lows[-1] < recent_lows[-2]:
                pivots["trend"] = "bearish"
            else:
                pivots["trend"] = "ranging"
        else:
            pivots["trend"] = "neutral"
        
        return pivots
    
    @staticmethod
    def detect_bos(candles: pd.DataFrame, pivots: Dict) -> Optional[Dict]:
        """Detects Break of Structure"""
        if not pivots["highs"] or not pivots["lows"]:
            return None
        
        last_high = pivots["highs"][-1]["price"]
        last_low = pivots["lows"][-1]["price"]
        current_price = candles.iloc[-1]['close']
        
        bos = None
        if current_price > last_high:
            bos = {"type": "bullish", "level": last_high, "price": current_price}
        elif current_price < last_low:
            bos = {"type": "bearish", "level": last_low, "price": current_price}
        
        return bos


# ================================================================
# GATE VALIDATION FUNCTIONS
# ================================================================
def validate_1A(context: Dict) -> Dict:
    """Gate 1A: HTF Bias Detection"""
    try:
        htf_data = context.get("htf_candles")
        if htf_data is None or len(htf_data) < 50:
            return {"passed": False, "bias": "neutral", "confidence": 0.0, "reason": "Insufficient HTF data"}
        
        ms = MarketStructure()
        pivots = ms.detect_pivots(htf_data)
        bos = ms.detect_bos(htf_data, pivots)
        
        bias = pivots["trend"]
        confidence = 0.5
        if len(pivots["highs"]) >= 3 and len(pivots["lows"]) >= 3:
            confidence += 0.2
        if bos is not None:
            confidence += 0.3
        
        return {
            "passed": confidence > 0.5,
            "bias": bias,
            "confidence": min(confidence, 1.0),
            "pivots_count": {"highs": len(pivots["highs"]), "lows": len(pivots["lows"])},
            "bos": bos,
            "reason": None if confidence > 0.5 else "Weak HTF structure"
        }
    except Exception as e:
        logger.error(f"[1A_ERROR] {e}")
        return {"passed": False, "bias": "neutral", "confidence": 0.0, "reason": str(e)}


def validate_1B(context: Dict) -> Dict:
    """Gate 1B: LTF Confirmation"""
    try:
        ltf_data = context.get("ltf_candles")
        gate_1a = context.get("1A", {})
        htf_bias = gate_1a.get("bias", "neutral")
        
        if ltf_data is None or len(ltf_data) < 20:
            return {"passed": False, "confidence": 0.0, "reason": "Insufficient LTF data"}
        
        ms = MarketStructure()
        ltf_pivots = ms.detect_pivots(ltf_data)
        ltf_bos = ms.detect_bos(ltf_data, ltf_pivots)
        
        alignment = False
        if htf_bias == "bullish" and ltf_pivots["trend"] in ["bullish", "ranging"]:
            alignment = True
        elif htf_bias == "bearish" and ltf_pivots["trend"] in ["bearish", "ranging"]:
            alignment = True
        
        confidence = 0.6 if alignment else 0.2
        if ltf_bos and ltf_bos.get("type") == htf_bias:
            confidence += 0.3
        
        return {
            "passed": confidence > 0.5,
            "confidence": min(confidence, 1.0),
            "ltf_trend": ltf_pivots["trend"],
            "ltf_bos": ltf_bos,
            "alignment": alignment,
            "reason": None if alignment else "LTF/HTF misalignment"
        }
    except Exception as e:
        logger.error(f"[1B_ERROR] {e}")
        return {"passed": False, "confidence": 0.0, "reason": str(e)}


def validate_1C(context: Dict) -> Dict:
    """Gate 1C: Confidence Scoring"""
    try:
        gate_1a = context.get("1A", {})
        gate_1b = context.get("1B", {})
        
        conf_1a = gate_1a.get("confidence", 0.0)
        conf_1b = gate_1b.get("confidence", 0.0)
        
        total_confidence = (conf_1a * 0.6) + (conf_1b * 0.4)
        
        if gate_1a.get("passed") and gate_1b.get("passed"):
            total_confidence += 0.1
        
        passed = total_confidence >= 0.65
        
        return {
            "passed": passed,
            "confidence": min(total_confidence, 1.0),
            "score": round(total_confidence * 100, 2),
            "reason": None if passed else f"Confidence {total_confidence*100:.1f}% < 65%"
        }
    except Exception as e:
        logger.error(f"[1C_ERROR] {e}")
        return {"passed": False, "confidence": 0.0, "reason": str(e)}


def validate_RCM(context: Dict) -> Dict:
    """RCM: Range Context Module"""
    try:
        range_data = context.get("detected_range")
        if not range_data:
            return {"valid": False, "confidence": 0.0, "reason": "No range detected"}
        
        range_high = range_data.get("high")
        range_low = range_data.get("low")
        range_duration_hours = range_data.get("duration_hours", 0)
        candles = range_data.get("candles", [])
        
        if range_duration_hours < 24:
            return {
                "valid": False,
                "confidence": 0.0,
                "range_duration_hours": range_duration_hours,
                "reason": f"Range duration {range_duration_hours}h < 24h minimum"
            }
        
        if not candles or len(candles) < 10:
            return {"valid": False, "confidence": 0.0, "reason": "Insufficient candles in range"}
        
        eq = (range_high + range_low) / 2
        
        # Analyze smoothness
        price_moves = []
        for i in range(1, len(candles)):
            if isinstance(candles[i], dict):
                move = abs(candles[i].get('c', 0) - candles[i-1].get('c', 0))
            else:
                move = abs(candles[i]['close'] - candles[i-1]['close'])
            price_moves.append(move)
        
        avg_move = np.mean(price_moves) if price_moves else 0
        range_size = range_high - range_low
        
        smoothness = 1 - min(avg_move / (range_size + 1e-9), 1.0)
        duration_score = min(range_duration_hours / 34, 1.0)
        confidence = (smoothness * 0.6) + (duration_score * 0.4)
        
        return {
            "valid": confidence > 0.6,
            "confidence": min(confidence, 1.0),
            "range_high": range_high,
            "range_low": range_low,
            "range_eq": eq,
            "range_duration_hours": range_duration_hours,
            "smoothness": round(smoothness, 3),
            "reason": None if confidence > 0.6 else "Low quality range"
        }
    except Exception as e:
        logger.error(f"[RCM_ERROR] {e}")
        return {"valid": False, "confidence": 0.0, "reason": str(e)}


def validate_MSCE(context: Dict) -> Dict:
    """MSCE: Multi-Session Context Engine"""
    try:
        utc_hour = datetime.utcnow().hour
        
        if 0 <= utc_hour < 8:
            session = "Asia"
            weight = 0.95
            manipulation_risk = 0.7
        elif 8 <= utc_hour < 16:
            session = "London"
            weight = 1.05
            manipulation_risk = 0.5
        else:
            session = "NY"
            weight = 1.15
            manipulation_risk = 0.8
        
        gate_1a = context.get("1A", {})
        htf_bias = gate_1a.get("bias", "neutral")
        session_bias = htf_bias
        
        session_start_hours = {"Asia": 0, "London": 8, "NY": 16}
        hours_into_session = (utc_hour - session_start_hours[session]) % 24
        in_first_hour = hours_into_session < 1
        
        if in_first_hour:
            manipulation_risk = min(manipulation_risk + 0.2, 1.0)
        
        return {
            "passed": True,
            "session": session,
            "session_bias": session_bias,
            "weight": weight,
            "confidence": weight / 1.15,
            "manipulation_risk": manipulation_risk,
            "first_hour": in_first_hour,
            "utc_hour": utc_hour
        }
    except Exception as e:
        logger.error(f"[MSCE_ERROR] {e}")
        return {"passed": False, "session": "Unknown", "confidence": 0.0, "reason": str(e)}


def validate_RIG(context: Dict) -> Dict:
    """RIG: Range Integrity Gate"""
    try:
        rcm = context.get("RCM", {})
        msce = context.get("MSCE", {})
        gate_1a = context.get("1A", {})
        
        htf_bias = gate_1a.get("bias", "neutral")
        session_bias = msce.get("session_bias", htf_bias)
        range_valid = rcm.get("valid", False)
        range_duration = rcm.get("range_duration_hours", 0)
        
        range_high = rcm.get("range_high", 0)
        range_low = rcm.get("range_low", 0)
        current_price = context.get("current_price", 0)
        
        if range_high > range_low:
            local_disp = abs(current_price - (range_high + range_low) / 2) / (range_high - range_low)
        else:
            local_disp = 0
        
        MIN_DURATION = 24
        DISP_THRESHOLD = 0.25
        
        if (range_valid and range_duration >= MIN_DURATION and 
            local_disp < DISP_THRESHOLD and session_bias != htf_bias and htf_bias != "neutral"):
            
            return {
                "passed": False,
                "Gate": "RIG",
                "reason": f"Counter-bias {msce.get('session')} session during intact HTF range",
                "confidence": 0.0,
                "htf_bias": htf_bias,
                "session_bias": session_bias,
                "range_duration_hours": range_duration,
                "local_displacement": round(local_disp, 3)
            }
        
        return {
            "passed": True,
            "Gate": "RIG",
            "reason": None,
            "confidence": 1.0,
            "htf_bias": htf_bias,
            "session_bias": session_bias
        }
    except Exception as e:
        logger.error(f"[RIG_ERROR] {e}")
        return {"passed": True, "Gate": "RIG", "confidence": 0.5, "reason": f"Error: {str(e)}"}


def validate_1D(context: Dict) -> Dict:
    """Gate 1D: Final Execution Gate"""
    try:
        gate_1a = context.get("1A", {})
        gate_1b = context.get("1B", {})
        gate_1c = context.get("1C", {})
        rcm = context.get("RCM", {})
        msce = context.get("MSCE", {})
        rig = context.get("RIG", {})
        
        critical_pass = (
            gate_1a.get("passed", False) and
            gate_1b.get("passed", False) and
            gate_1c.get("passed", False) and
            rcm.get("valid", False) and
            rig.get("passed", False)
        )
        
        if not critical_pass:
            failed_gates = []
            if not gate_1a.get("passed"): failed_gates.append("1A")
            if not gate_1b.get("passed"): failed_gates.append("1B")
            if not gate_1c.get("passed"): failed_gates.append("1C")
            if not rcm.get("valid"): failed_gates.append("RCM")
            if not rig.get("passed"): failed_gates.append("RIG")
            
            return {
                "passed": False,
                "score": 0.0,
                "ExecutionConfidence_Total": 0.0,
                "reason": f"Failed gates: {', '.join(failed_gates)}",
                "failed_gates": failed_gates
            }
        
        conf_1a = gate_1a.get("confidence", 0.0)
        conf_1b = gate_1b.get("confidence", 0.0)
        conf_1c = gate_1c.get("confidence", 0.0)
        conf_rcm = rcm.get("confidence", 0.0)
        conf_msce = msce.get("confidence", 0.0)
        
        total_confidence = (
            conf_1a * 0.25 +
            conf_1b * 0.20 +
            conf_1c * 0.25 +
            conf_rcm * 0.20 +
            conf_msce * 0.10
        )
        
        execution_threshold = 0.70
        passed = total_confidence >= execution_threshold
        
        return {
            "passed": passed,
            "score": round(total_confidence, 3),
            "ExecutionConfidence_Total": round(total_confidence * 100, 2),
            "threshold": execution_threshold * 100,
            "reason": None if passed else f"Confidence {total_confidence*100:.1f}% < 70%",
            "breakdown": {
                "1A": round(conf_1a * 100, 1),
                "1B": round(conf_1b * 100, 1),
                "1C": round(conf_1c * 100, 1),
                "RCM": round(conf_rcm * 100, 1),
                "MSCE": round(conf_msce * 100, 1)
            }
        }
    except Exception as e:
        logger.error(f"[1D_ERROR] {e}")
        return {"passed": False, "score": 0.0, "reason": str(e)}


def validate_gates(context: Dict) -> Dict:
    """Master gate validation orchestrator"""
    try:
        context["1A"] = validate_1A(context)
        context["1B"] = validate_1B(context)
        context["1C"] = validate_1C(context)
        context["RCM"] = validate_RCM(context)
        context["MSCE"] = validate_MSCE(context)
        context["RIG"] = validate_RIG(context)
        context["1D"] = validate_1D(context)
        
        if context["1D"]["passed"]:
            context["ExecutionConfidence_Total"] = context["1D"]["ExecutionConfidence_Total"]
            context["Reward_Summary"] = "VALID_STRUCTURE"
            context["Action"] = "EXECUTE"
        else:
            context["ExecutionConfidence_Total"] = 0.0
            if not context["RIG"]["passed"]:
                context["Reward_Summary"] = "INVALID_RIG"
                context["RIG_reason"] = context["RIG"]["reason"]
            elif not context["RCM"]["valid"]:
                context["Reward_Summary"] = "INVALID_RANGE"
            else:
                context["Reward_Summary"] = "FAILED_GATES"
            context["Action"] = "NO_TRADE"
        
        context["timestamp"] = datetime.utcnow().isoformat()
        logger.info(f"[VALIDATE_GATES] Action={context['Action']}, Confidence={context['ExecutionConfidence_Total']}%")
        
        return context
    except Exception as e:
        logger.error(f"[VALIDATE_GATES_ERROR] {e}")
        context["ExecutionConfidence_Total"] = 0.0
        context["Reward_Summary"] = f"ERROR: {str(e)}"
        context["Action"] = "NO_TRADE"
        return context


# ================================================================
# DATA FETCHING UTILITIES
# ================================================================
async def fetch_mexc_candles(symbol: str, interval: str, limit: int = 500) -> Optional[pd.DataFrame]:
    """Fetch OHLCV candles from MEXC"""
    url = f"{MEXC_URL_BASE}/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.get(url, params=params)
            if response.status_code != 200:
                logger.error(f"[MEXC_FETCH] HTTP {response.status_code}")
                return None
            
            data = response.json()
            if not data:
                return None
            
            # MEXC returns 8 columns: [timestamp, open, high, low, close, volume, close_time, quote_volume]
            df = pd.DataFrame(data, columns=[
                "open_time", "open", "high", "low", "close", "volume", "close_time", "quote_vol"
            ])
            
            df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = df[col].astype(float)
            
            df = df[["open_time", "open", "high", "low", "close", "volume"]].sort_values("open_time")
            return df
    
    except Exception as e:
        logger.error(f"[MEXC_FETCH_ERROR] {e}")
        return None


# ================================================================
# RANGE SCANNER (Simplified)
# ================================================================
async def detect_best_range(candles: List) -> Optional[Dict]:
    """Simple range detection from candles"""
    if not candles or len(candles) < 10:
        return None
    
    highs = [c.get('h', c.get('high', 0)) for c in candles]
    lows = [c.get('l', c.get('low', 0)) for c in candles]
    
    range_high = max(highs)
    range_low = min(lows)
    
    # Calculate duration
    if isinstance(candles[0].get('t'), int):
        start_time = candles[0]['t']
        end_time = candles[-1]['t']
        duration_hours = (end_time - start_time) / (1000 * 3600)
    else:
        duration_hours = len(candles) * 0.25  # Estimate
    
    return {
        "high": range_high,
        "low": range_low,
        "duration_hours": duration_hours,
        "candles": candles
    }


async def background_range_updater():
    """Background task to update ranges"""
    global latest_ranges
    while True:
        try:
            # Fetch LTF data
            ltf_df = await fetch_mexc_candles(SYMBOL, "15m", 200)
            if ltf_df is not None:
                ltf_candles = ltf_df.to_dict('records')
                ltf_range = await detect_best_range(ltf_candles)
                if ltf_range:
                    latest_ranges["LTF"] = [ltf_range]
            
            # Fetch HTF data
            htf_df = await fetch_mexc_candles(SYMBOL, "4h", 200)
            if htf_df is not None:
                htf_candles = htf_df.to_dict('records')
                htf_range = await detect_best_range(htf_candles)
                if htf_range:
                    latest_ranges["HTF"] = [htf_range]
            
            logger.info("[AUTO_UPDATE] ✅ Range cache updated.")
        except Exception as e:
            logger.error(f"[AUTO_UPDATE_ERROR] {e}")
        
        await asyncio.sleep(scan_interval_sec)


# ================================================================
# ENDPOINTS
# ================================================================
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(background_range_updater())
    logger.info("[STARTUP] Background range updater started.")


@app.get("/")
async def root():
    return JSONResponse({
        "service": "HPB–TCT v21.2 (MEXC + Gate Validation + Model Detection)",
        "status": "running",
        "symbol": SYMBOL,
        "auth_mode": "private" if MEXC_KEY else "public",
        "endpoints": {
            "health": "/status",
            "market_data": {
                "/api/price": "Current ticker price",
                "/api/ranges": "Cached LTF/HTF ranges"
            },
            "analysis": {
                "/api/validate": "Complete 7-gate validation",
                "/api/models": "TCT model detection (Model 1/2, Accumulation/Distribution)",
                "/api/signals": "⭐ Complete trading signals (models + validation)",
                "/api/scan": "Manual range scan"
            },
            "training": {
                "/api/train": "TensorTrade AutoLearn training"
            }
        },
        "version": "21.2",
        "features": [
            "7-Gate Validation System (1A, 1B, 1C, RCM, MSCE, RIG, 1D)",
            "TCT Model Detection (Model 1, Model 2, Extended)",
            "Liquidity Curve Analysis (RTZ)",
            "Supply/Demand Zone Detection",
            "Multi-Session Context (Asia/London/NY)",
            "Range Integrity Protection",
            "Actionable Trading Signals"
        ]
    })


@app.get("/status")
async def get_status():
    return {"status": "OK", "symbol": SYMBOL, "auth_mode": "private" if MEXC_KEY else "public"}


@app.get("/api/ranges")
async def get_ranges():
    return latest_ranges


@app.get("/api/validate")
async def validate_current_setup():
    """
    Complete gate validation endpoint
    Fetches current market data and runs all 7 gates
    """
    try:
        # Fetch HTF and LTF data
        htf_df = await fetch_mexc_candles(SYMBOL, "4h", 100)
        ltf_df = await fetch_mexc_candles(SYMBOL, "15m", 100)
        
        if htf_df is None or ltf_df is None:
            return JSONResponse({
                "error": "Failed to fetch market data",
                "Action": "NO_TRADE"
            }, status_code=500)
        
        # Get current price
        current_price = ltf_df.iloc[-1]['close']
        
        # Detect range from LTF
        ltf_candles = ltf_df.to_dict('records')
        detected_range = await detect_best_range(ltf_candles)
        
        # Build context
        context = {
            "htf_candles": htf_df,
            "ltf_candles": ltf_df,
            "detected_range": detected_range,
            "current_price": current_price,
            "symbol": SYMBOL
        }
        
        # Run validation
        result = validate_gates(context)
        
        # Clean up for JSON serialization
        result.pop("htf_candles", None)
        result.pop("ltf_candles", None)
        
        return JSONResponse(result)
    
    except Exception as e:
        logger.error(f"[VALIDATE_ERROR] {e}")
        return JSONResponse({
            "error": str(e),
            "Action": "NO_TRADE",
            "ExecutionConfidence_Total": 0.0
        }, status_code=500)


@app.get("/api/models")
async def detect_models(interval: str = "15m", limit: int = 200):
    """
    TCT Model Detection Endpoint
    Detects Model 1, Model 2, Extended patterns (Accumulation & Distribution)
    """
    try:
        from tct_model_detector import detect_tct_models
        
        candles = await fetch_mexc_candles(SYMBOL, interval, limit)
        
        if candles is None or len(candles) < 50:
            return JSONResponse({
                "error": "Insufficient data for model detection",
                "accumulation_models": [],
                "distribution_models": []
            }, status_code=400)
        
        result = detect_tct_models(candles)
        result["symbol"] = SYMBOL
        result["interval"] = interval
        result["candles_analyzed"] = len(candles)
        
        logger.info(f"[MODELS] Found {result['total_models']} TCT models ({interval})")
        
        return JSONResponse(result)
    
    except ImportError:
        return JSONResponse({
            "error": "TCT model detector module not found. Ensure tct_model_detector.py is in the same directory.",
            "accumulation_models": [],
            "distribution_models": []
        }, status_code=500)
    
    except Exception as e:
        logger.error(f"[MODELS_ERROR] {e}")
        return JSONResponse({
            "error": str(e),
            "accumulation_models": [],
            "distribution_models": []
        }, status_code=500)


@app.get("/api/signals")
async def get_trading_signals():
    """
    Complete Trading Signals Endpoint
    Combines gate validation + TCT model detection for actionable signals
    """
    try:
        from tct_model_detector import detect_tct_models
        
        htf_df = await fetch_mexc_candles(SYMBOL, "4h", 100)
        ltf_df = await fetch_mexc_candles(SYMBOL, "15m", 200)
        
        if htf_df is None or ltf_df is None:
            return JSONResponse({"error": "Failed to fetch market data", "signals": []}, status_code=500)
        
        current_price = ltf_df.iloc[-1]['close']
        
        models = detect_tct_models(ltf_df)
        
        best_acc = models["accumulation_models"][0] if models["accumulation_models"] else None
        best_dist = models["distribution_models"][0] if models["distribution_models"] else None
        
        signals = []
        
        # Validate accumulation model
        if best_acc:
            acc_range = await detect_best_range(ltf_df.to_dict('records'))
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
            dist_range = await detect_best_range(ltf_df.to_dict('records'))
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


@app.get("/api/scan")
async def manual_scan():
    """Manual range scan"""
    try:
        ltf_df = await fetch_mexc_candles(SYMBOL, "15m", 200)
        htf_df = await fetch_mexc_candles(SYMBOL, "4h", 200)
        
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


@app.get("/api/price")
async def live_price():
    """Fetch current MEXC ticker price"""
    url = f"{MEXC_URL_BASE}/api/v3/ticker/price?symbol={SYMBOL}"
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.get(url)
            if r.status_code == 200:
                data = r.json()
                return {"symbol": SYMBOL, "price": float(data["price"])}
            else:
                return {"error": f"HTTP {r.status_code}"}
    except Exception as e:
        logger.error(f"[PRICE_FAIL] {e}")
        return {"error": str(e)}


@app.get("/api/train")
async def run_training(episodes: int = 5):
    """Run TensorTrade AutoLearn training"""
    try:
        from tensortrade_env import HPB_TensorTrade_Env
        
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


# ================================================================
# ENTRY POINT
# ================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server_mexc:app", host="0.0.0.0", port=PORT, reload=True)