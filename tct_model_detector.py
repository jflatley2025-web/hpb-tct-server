"""
tct_model_detector.py — HPB–TCT v21.2 Complete Model Detection System
Author: HPB-TCT Dev Team
Date: 2026-01-23

Detects Model 1, Model 2, and Extended TCT patterns based on:
- 2025-HP-TCT-model-variables final version.pdf
- 2025-Liquidity PDFs
- 2025-S_D-_E_REVIEW38Pages.pdf
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional
import logging

logger = logging.getLogger("TCT-Detector")


class LiquidityCurveDetector:
    """Detects high-quality liquidity curves (RTZ - Return to Zone)"""
    
    @staticmethod
    def detect_accumulation_curve(candles: pd.DataFrame, tap1_idx: int, tap2_idx: int) -> Dict:
        """Detects buy-side liquidity curve (higher lows stacking up)"""
        if tap1_idx >= tap2_idx or tap2_idx >= len(candles):
            return {"valid": False, "quality": 0.0, "smoothness": 0.0}

        curve_candles = candles.iloc[tap1_idx:tap2_idx+1].copy()
        if len(curve_candles) < 3:
            return {"valid": False, "quality": 0.0, "smoothness": 0.0}

        # Find pivot lows
        lows = []
        for i in range(1, len(curve_candles) - 1):
            if (curve_candles.iloc[i]['low'] < curve_candles.iloc[i-1]['low'] and
                curve_candles.iloc[i]['low'] < curve_candles.iloc[i+1]['low']):
                lows.append({"idx": i, "price": float(curve_candles.iloc[i]['low'])})

        if len(lows) < 2:
            return {"valid": False, "quality": 0.0, "smoothness": 0.0}
        
        # Check for higher lows
        higher_lows_count = sum(1 for i in range(1, len(lows)) if lows[i]["price"] > lows[i-1]["price"])
        quality = higher_lows_count / max(len(lows) - 1, 1)
        
        # Smoothness
        price_changes = curve_candles['close'].diff().abs()
        avg_change = price_changes.mean()
        range_size = curve_candles['high'].max() - curve_candles['low'].min()
        smoothness = 1 - min(avg_change / (range_size + 1e-9), 1.0)
        
        total_quality = (quality * 0.5) + (smoothness * 0.5)
        
        return {
            "valid": total_quality > 0.6,
            "quality": round(total_quality, 3),
            "smoothness": round(smoothness, 3),
            "curve_type": "accumulation"
        }
    
    @staticmethod
    def detect_distribution_curve(candles: pd.DataFrame, tap1_idx: int, tap2_idx: int) -> Dict:
        """Detects sell-side liquidity curve (lower highs stacking down)"""
        if tap1_idx >= tap2_idx or tap2_idx >= len(candles):
            return {"valid": False, "quality": 0.0, "smoothness": 0.0}

        curve_candles = candles.iloc[tap1_idx:tap2_idx+1].copy()
        if len(curve_candles) < 3:
            return {"valid": False, "quality": 0.0, "smoothness": 0.0}

        # Find pivot highs
        highs = []
        for i in range(1, len(curve_candles) - 1):
            if (curve_candles.iloc[i]['high'] > curve_candles.iloc[i-1]['high'] and
                curve_candles.iloc[i]['high'] > curve_candles.iloc[i+1]['high']):
                highs.append({"idx": i, "price": float(curve_candles.iloc[i]['high'])})

        if len(highs) < 2:
            return {"valid": False, "quality": 0.0, "smoothness": 0.0}
        
        # Check for lower highs
        lower_highs_count = sum(1 for i in range(1, len(highs)) if highs[i]["price"] < highs[i-1]["price"])
        quality = lower_highs_count / max(len(highs) - 1, 1)
        
        # Smoothness
        price_changes = curve_candles['close'].diff().abs()
        avg_change = price_changes.mean()
        range_size = curve_candles['high'].max() - curve_candles['low'].min()
        smoothness = 1 - min(avg_change / (range_size + 1e-9), 1.0)
        
        total_quality = (quality * 0.5) + (smoothness * 0.5)
        
        return {
            "valid": total_quality > 0.6,
            "quality": round(total_quality, 3),
            "smoothness": round(smoothness, 3),
            "curve_type": "distribution"
        }


class TCTModelDetector:
    """Main TCT Model Detection Engine"""
    
    def __init__(self, candles: pd.DataFrame):
        self.candles = candles.copy()
        self.lc_detector = LiquidityCurveDetector()
    
    def detect_accumulation_models(self) -> List[Dict]:
        """Detect accumulation models (bullish)"""
        models = []
        
        for i in range(10, len(self.candles) - 30):
            try:
                tap1 = self._find_tap1_acc(i)
                if not tap1:
                    continue
                
                tap2 = self._find_tap2_acc(tap1)
                if not tap2:
                    continue
                
                duration = self._calc_duration(tap1["idx"], tap2["idx"])
                if duration < 24:
                    continue
                
                curve = self.lc_detector.detect_accumulation_curve(self.candles, tap1["idx"], tap2["idx"])
                if not curve["valid"]:
                    continue
                
                tap3_m1 = self._find_tap3_model1_acc(tap1, tap2)
                tap3_m2 = self._find_tap3_model2_acc(tap1, tap2)
                
                if tap3_m1:
                    models.append(self._build_model_1_acc(tap1, tap2, tap3_m1, curve, duration))
                if tap3_m2:
                    models.append(self._build_model_2_acc(tap1, tap2, tap3_m2, curve, duration))
            except:
                continue
        
        models.sort(key=lambda x: x["quality_score"], reverse=True)
        return models[:5]
    
    def detect_distribution_models(self) -> List[Dict]:
        """Detect distribution models (bearish)"""
        models = []
        
        for i in range(10, len(self.candles) - 30):
            try:
                tap1 = self._find_tap1_dist(i)
                if not tap1:
                    continue
                
                tap2 = self._find_tap2_dist(tap1)
                if not tap2:
                    continue
                
                duration = self._calc_duration(tap1["idx"], tap2["idx"])
                if duration < 24:
                    continue
                
                curve = self.lc_detector.detect_distribution_curve(self.candles, tap1["idx"], tap2["idx"])
                if not curve["valid"]:
                    continue
                
                tap3_m1 = self._find_tap3_model1_dist(tap1, tap2)
                tap3_m2 = self._find_tap3_model2_dist(tap1, tap2)
                
                if tap3_m1:
                    models.append(self._build_model_1_dist(tap1, tap2, tap3_m1, curve, duration))
                if tap3_m2:
                    models.append(self._build_model_2_dist(tap1, tap2, tap3_m2, curve, duration))
            except:
                continue
        
        models.sort(key=lambda x: x["quality_score"], reverse=True)
        return models[:5]
    
    # ACCUMULATION TAP FINDERS
    def _find_tap1_acc(self, start_idx: int) -> Optional[Dict]:
        if start_idx >= len(self.candles) - 10:
            return None
        window = self.candles.iloc[start_idx:min(start_idx+10, len(self.candles))]
        if len(window) == 0:
            return None
        low_idx = int(window['low'].idxmin())
        return {
            "idx": low_idx,
            "price": float(self.candles.iloc[low_idx]['low']),
            "time": str(self.candles.iloc[low_idx]['open_time']),
            "type": "tap1_acc"
        }
    
    def _find_tap2_acc(self, tap1: Dict) -> Optional[Dict]:
        start_idx = tap1["idx"] + 5
        if start_idx >= len(self.candles) - 5:
            return None
        window = self.candles.iloc[start_idx:min(start_idx+20, len(self.candles))]
        if len(window) == 0:
            return None
        high_idx = int(window['high'].idxmax())
        if self.candles.iloc[high_idx]['high'] <= tap1["price"] * 1.005:
            return None
        return {
            "idx": high_idx,
            "price": float(self.candles.iloc[high_idx]['high']),
            "time": str(self.candles.iloc[high_idx]['open_time']),
            "type": "tap2_acc"
        }
    
    def _find_tap3_model1_acc(self, tap1: Dict, tap2: Dict) -> Optional[Dict]:
        start_idx = tap2["idx"] + 3
        if start_idx >= len(self.candles) - 3:
            return None
        window = self.candles.iloc[start_idx:min(start_idx+15, len(self.candles))]
        if len(window) == 0:
            return None
        low_idx = int(window['low'].idxmin())
        tap3_price = float(self.candles.iloc[low_idx]['low'])
        
        if tap3_price <= tap1["price"]:
            return None
        range_size = tap2["price"] - tap1["price"]
        if tap3_price > tap1["price"] + (range_size * 0.5):
            return None
        
        return {
            "idx": low_idx,
            "price": tap3_price,
            "time": str(self.candles.iloc[low_idx]['open_time']),
            "type": "tap3_m1",
            "deviation": False
        }
    
    def _find_tap3_model2_acc(self, tap1: Dict, tap2: Dict) -> Optional[Dict]:
        start_idx = tap2["idx"] + 3
        if start_idx >= len(self.candles) - 3:
            return None
        window = self.candles.iloc[start_idx:min(start_idx+20, len(self.candles))]
        if len(window) == 0:
            return None
        low_idx = int(window['low'].idxmin())
        tap3_price = float(self.candles.iloc[low_idx]['low'])
        
        tap1_tolerance = tap1["price"] * 0.02
        if abs(tap3_price - tap1["price"]) > tap1_tolerance and tap3_price > tap1["price"]:
            return None
        
        return {
            "idx": low_idx,
            "price": tap3_price,
            "time": str(self.candles.iloc[low_idx]['open_time']),
            "type": "tap3_m2",
            "deviation": tap3_price < tap1["price"]
        }
    
    # DISTRIBUTION TAP FINDERS
    def _find_tap1_dist(self, start_idx: int) -> Optional[Dict]:
        if start_idx >= len(self.candles) - 10:
            return None
        window = self.candles.iloc[start_idx:min(start_idx+10, len(self.candles))]
        if len(window) == 0:
            return None
        high_idx = int(window['high'].idxmax())
        return {
            "idx": high_idx,
            "price": float(self.candles.iloc[high_idx]['high']),
            "time": str(self.candles.iloc[high_idx]['open_time']),
            "type": "tap1_dist"
        }
    
    def _find_tap2_dist(self, tap1: Dict) -> Optional[Dict]:
        start_idx = tap1["idx"] + 5
        if start_idx >= len(self.candles) - 5:
            return None
        window = self.candles.iloc[start_idx:min(start_idx+20, len(self.candles))]
        if len(window) == 0:
            return None
        low_idx = int(window['low'].idxmin())
        if self.candles.iloc[low_idx]['low'] >= tap1["price"] * 0.995:
            return None
        return {
            "idx": low_idx,
            "price": float(self.candles.iloc[low_idx]['low']),
            "time": str(self.candles.iloc[low_idx]['open_time']),
            "type": "tap2_dist"
        }
    
    def _find_tap3_model1_dist(self, tap1: Dict, tap2: Dict) -> Optional[Dict]:
        start_idx = tap2["idx"] + 3
        if start_idx >= len(self.candles) - 3:
            return None
        window = self.candles.iloc[start_idx:min(start_idx+15, len(self.candles))]
        if len(window) == 0:
            return None
        high_idx = int(window['high'].idxmax())
        tap3_price = float(self.candles.iloc[high_idx]['high'])
        
        if tap3_price >= tap1["price"]:
            return None
        range_size = tap1["price"] - tap2["price"]
        if tap3_price < tap2["price"] + (range_size * 0.5):
            return None
        
        return {
            "idx": high_idx,
            "price": tap3_price,
            "time": str(self.candles.iloc[high_idx]['open_time']),
            "type": "tap3_m1",
            "deviation": False
        }
    
    def _find_tap3_model2_dist(self, tap1: Dict, tap2: Dict) -> Optional[Dict]:
        start_idx = tap2["idx"] + 3
        if start_idx >= len(self.candles) - 3:
            return None
        window = self.candles.iloc[start_idx:min(start_idx+20, len(self.candles))]
        if len(window) == 0:
            return None
        high_idx = int(window['high'].idxmax())
        tap3_price = float(self.candles.iloc[high_idx]['high'])
        
        tap1_tolerance = tap1["price"] * 0.02
        if abs(tap3_price - tap1["price"]) > tap1_tolerance and tap3_price < tap1["price"]:
            return None
        
        return {
            "idx": high_idx,
            "price": tap3_price,
            "time": str(self.candles.iloc[high_idx]['open_time']),
            "type": "tap3_m2",
            "deviation": tap3_price > tap1["price"]
        }
    
    # MODEL BUILDERS
    def _build_model_1_acc(self, tap1, tap2, tap3, curve, duration) -> Dict:
        quality = self._calc_quality(curve, duration, tap3.get("deviation", False))
        return {
            "model_type": "Model_1_Accumulation",
            "direction": "bullish",
            "tap1": tap1,
            "tap2": tap2,
            "tap3": tap3,
            "range_low": tap1["price"],
            "range_high": tap2["price"],
            "range_eq": (tap1["price"] + tap2["price"]) / 2,
            "duration_hours": duration,
            "liquidity_curve": curve,
            "quality_score": quality,
            "target": tap2["price"],
            "invalidation": tap1["price"],
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def _build_model_2_acc(self, tap1, tap2, tap3, curve, duration) -> Dict:
        quality = self._calc_quality(curve, duration, tap3.get("deviation", False))
        if tap3.get("deviation"):
            quality = min(quality + 0.1, 1.0)
        return {
            "model_type": "Model_2_Accumulation",
            "direction": "bullish",
            "tap1": tap1,
            "tap2": tap2,
            "tap3": tap3,
            "range_low": tap1["price"],
            "range_high": tap2["price"],
            "range_eq": (tap1["price"] + tap2["price"]) / 2,
            "duration_hours": duration,
            "liquidity_curve": curve,
            "quality_score": quality,
            "target": tap2["price"],
            "invalidation": tap3["price"] if tap3.get("deviation") else tap1["price"],
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def _build_model_1_dist(self, tap1, tap2, tap3, curve, duration) -> Dict:
        quality = self._calc_quality(curve, duration, tap3.get("deviation", False))
        return {
            "model_type": "Model_1_Distribution",
            "direction": "bearish",
            "tap1": tap1,
            "tap2": tap2,
            "tap3": tap3,
            "range_high": tap1["price"],
            "range_low": tap2["price"],
            "range_eq": (tap1["price"] + tap2["price"]) / 2,
            "duration_hours": duration,
            "liquidity_curve": curve,
            "quality_score": quality,
            "target": tap2["price"],
            "invalidation": tap1["price"],
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def _build_model_2_dist(self, tap1, tap2, tap3, curve, duration) -> Dict:
        quality = self._calc_quality(curve, duration, tap3.get("deviation", False))
        return {
            "model_type": "Model_2_Distribution",
            "direction": "bearish",
            "tap1": tap1,
            "tap2": tap2,
            "tap3": tap3,
            "range_high": tap1["price"],
            "range_low": tap2["price"],
            "range_eq": (tap1["price"] + tap2["price"]) / 2,
            "duration_hours": duration,
            "liquidity_curve": curve,
            "quality_score": quality,
            "target": tap2["price"],
            "invalidation": tap3["price"] if tap3.get("deviation") else tap1["price"],
            "timestamp": datetime.utcnow().isoformat()
        }
    
    # UTILITIES
    def _calc_duration(self, idx1: int, idx2: int) -> float:
        if idx1 >= idx2 or idx2 >= len(self.candles):
            return 0
        try:
            t1 = pd.to_datetime(self.candles.iloc[idx1]['open_time'])
            t2 = pd.to_datetime(self.candles.iloc[idx2]['open_time'])
            return round((t2 - t1).total_seconds() / 3600, 2)
        except:
            return round((idx2 - idx1) * 0.25, 2)
    
    def _calc_quality(self, curve: Dict, duration: float, deviation: bool) -> float:
        score = curve.get("quality", 0.5) * 0.5
        if duration >= 34:
            score += 0.2
        elif duration >= 24:
            score += 0.15
        else:
            score += 0.05
        if deviation:
            score += 0.15
        score += curve.get("smoothness", 0.5) * 0.15
        return round(min(score, 1.0), 3)


def detect_tct_models(candles: pd.DataFrame) -> Dict:
    """Main entry point for TCT model detection"""
    if len(candles) < 50:
        return {
            "accumulation_models": [],
            "distribution_models": [],
            "total_models": 0,
            "error": "Insufficient data (need 50+ candles)",
            "timestamp": datetime.utcnow().isoformat()
        }
    
    try:
        detector = TCTModelDetector(candles)
        acc = detector.detect_accumulation_models()
        dist = detector.detect_distribution_models()
        
        return {
            "accumulation_models": acc,
            "distribution_models": dist,
            "total_models": len(acc) + len(dist),
            "candles_analyzed": len(candles),
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"[ERROR] {e}")
        return {
            "accumulation_models": [],
            "distribution_models": [],
            "total_models": 0,
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("="*60)
    print("TCT MODEL DETECTOR TEST")
    print("="*60)
    
    dates = pd.date_range('2026-01-01', periods=200, freq='1H')
    prices = []
    base = 100000
    
    for i in range(200):
        if i < 50:
            prices.append(base + np.random.uniform(-500, 500))
        elif i < 100:
            prices.append(base + (i-50) * 40 + np.random.uniform(-300, 300))
        else:
            prices.append(base + 2000 - (i-100) * 15 + np.random.uniform(-400, 400))
    
    df = pd.DataFrame({
        'open_time': dates,
        'open': prices,
        'high': [p + np.random.uniform(100, 300) for p in prices],
        'low': [p - np.random.uniform(100, 300) for p in prices],
        'close': [p + np.random.uniform(-200, 200) for p in prices],
        'volume': np.random.uniform(100, 1000, 200)
    })
    
    print(f"\nAnalyzing {len(df)} candles...")
    result = detect_tct_models(df)
    
    print(f"\n{'='*60}")
    print(f"Total Models: {result['total_models']}")
    print(f"Accumulation: {len(result['accumulation_models'])}")
    print(f"Distribution: {len(result['distribution_models'])}")
    
    for i, m in enumerate(result['accumulation_models'], 1):
        print(f"\n  #{i} {m['model_type']}")
        print(f"    Quality: {m['quality_score']} ⭐")
        print(f"    Duration: {m['duration_hours']}h")
        print(f"    Range: ${m['range_low']:.0f} - ${m['range_high']:.0f}")
        print(f"    Target: ${m['target']:.0f}")
    
    print(f"\n{'='*60}")
    print("✅ TEST COMPLETE")
    print(f"{'='*60}\n")
