# ================================================================
# HPB–TCT v21.2 MEXC Feed + Range Detection + Gate Validation Server
# ================================================================

import os
import asyncio
import logging
import httpx
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional

from fastapi import FastAPI
from fastapi.responses import JSONResponse, HTMLResponse
from loguru import logger

from tct_schematics import detect_tct_schematics


# ================================================================
# NUMPY SERIALIZATION HELPER
# ================================================================

def convert_numpy_types(obj):
    """
    Recursively convert numpy types to Python native types for JSON serialization.
    Handles: numpy.bool_, numpy.int64, numpy.float64, numpy.ndarray, etc.
    """
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif pd.isna(obj):
        return None
    else:
        return obj


# ================================================================
# CONFIGURATION
# ================================================================

PORT = int(os.getenv("PORT", 10000))
SYMBOL = os.getenv("SYMBOL", "BTCUSDT").replace("/", "").replace("-", "").upper()
MEXC_KEY = os.getenv("MEXC_KEY")
MEXC_SECRET = os.getenv("MEXC_SECRET")
MEXC_URL_BASE = "https://api.mexc.com"

app = FastAPI(title="HPB–TCT v21.2 MEXC Server", version="21.2")

latest_ranges = {"LTF": [], "HTF": []}
scan_interval_sec = 120

logging.basicConfig(level=logging.INFO)
logger.info(f"[INIT] HPB–TCT v21.2 Ready — Symbol={SYMBOL}, Port={PORT}")

# ================================================================
# MARKET STRUCTURE
# ================================================================

class MarketStructure:
    """Detects market structure using 6-candle rule"""

    @staticmethod
    def detect_pivots(candles: pd.DataFrame) -> Dict:
        if len(candles) < 6:
            return {"highs": [], "lows": [], "trend": "neutral"}

        pivots = {"highs": [], "lows": []}

        for i in range(2, len(candles) - 2):
            if (
                candles.iloc[i - 2]["close"] < candles.iloc[i - 1]["close"]
                and candles.iloc[i - 1]["close"] < candles.iloc[i]["close"]
                and candles.iloc[i]["close"] > candles.iloc[i + 1]["close"]
                and candles.iloc[i + 1]["close"] > candles.iloc[i + 2]["close"]
            ):
                pivots["highs"].append({"idx": int(i), "price": float(candles.iloc[i]["high"])})

            if (
                candles.iloc[i - 2]["close"] > candles.iloc[i - 1]["close"]
                and candles.iloc[i - 1]["close"] > candles.iloc[i]["close"]
                and candles.iloc[i]["close"] < candles.iloc[i + 1]["close"]
                and candles.iloc[i + 1]["close"] < candles.iloc[i + 2]["close"]
            ):
                pivots["lows"].append({"idx": int(i), "price": float(candles.iloc[i]["low"])})

        if len(pivots["highs"]) >= 2 and len(pivots["lows"]) >= 2:
            h1, h2 = pivots["highs"][-2:]
            l1, l2 = pivots["lows"][-2:]
            if h2["price"] > h1["price"] and l2["price"] > l1["price"]:
                pivots["trend"] = "bullish"
            elif h2["price"] < h1["price"] and l2["price"] < l1["price"]:
                pivots["trend"] = "bearish"
            else:
                pivots["trend"] = "ranging"
        else:
            pivots["trend"] = "neutral"

        return pivots

    @staticmethod
    def detect_bos(candles: pd.DataFrame, pivots: Dict) -> Optional[Dict]:
        if not pivots["highs"] or not pivots["lows"]:
            return None

        price = candles.iloc[-1]["close"]
        if price > pivots["highs"][-1]["price"]:
            return {"type": "bullish", "price": price}
        if price < pivots["lows"][-1]["price"]:
            return {"type": "bearish", "price": price}
        return None

# ================================================================
# FAIR VALUE GAP (FVG) DETECTION
# ================================================================

class FairValueGap:
    """Detects Fair Value Gaps (inefficiencies) - 3-candle pattern"""

    @staticmethod
    def detect_fvgs(candles: pd.DataFrame) -> Dict:
        """
        Detect FVGs using 3-candle pattern.

        Bullish FVG: candle[i-1].high < candle[i+1].low (gap below)
        Bearish FVG: candle[i-1].low > candle[i+1].high (gap above)

        Returns: Dict with bullish_fvgs and bearish_fvgs lists
        """
        if len(candles) < 3:
            return {"bullish_fvgs": [], "bearish_fvgs": []}

        bullish_fvgs = []
        bearish_fvgs = []

        for i in range(1, len(candles) - 1):
            c_prev = candles.iloc[i - 1]
            c_curr = candles.iloc[i]
            c_next = candles.iloc[i + 1]

            # Bullish FVG: gap between prev high and next low
            if c_prev["high"] < c_next["low"]:
                bullish_fvgs.append({
                    "idx": i,
                    "top": float(c_next["low"]),
                    "bottom": float(c_prev["high"]),
                    "time": str(c_curr["open_time"]) if "open_time" in c_curr.index else i,
                    "gap_size": float(c_next["low"] - c_prev["high"])
                })

            # Bearish FVG: gap between prev low and next high
            if c_prev["low"] > c_next["high"]:
                bearish_fvgs.append({
                    "idx": i,
                    "top": float(c_prev["low"]),
                    "bottom": float(c_next["high"]),
                    "time": str(c_curr["open_time"]) if "open_time" in c_curr.index else i,
                    "gap_size": float(c_prev["low"] - c_next["high"])
                })

        return {"bullish_fvgs": bullish_fvgs, "bearish_fvgs": bearish_fvgs}

# ================================================================
# ORDER BLOCK DETECTION (OBIF - Order Block with Inefficiency)
# ================================================================

class OrderBlock:
    """
    Detects Order Blocks with Fair Value Gap requirement.

    An Order Block is a single candle representing supply/demand area
    that MUST have a Fair Value Gap (inefficiency).

    - Bullish OBIF: Last bearish candle before bullish expansion with FVG
    - Bearish OBIF: Last bullish candle before bearish expansion with FVG
    """

    @staticmethod
    def detect_order_blocks(candles: pd.DataFrame, fvgs: Dict) -> Dict:
        """
        Detect Order Blocks using FVG requirement.

        Args:
            candles: DataFrame with OHLC data
            fvgs: Dict from FairValueGap.detect_fvgs()

        Returns: Dict with bullish_obs and bearish_obs lists
        """
        bullish_obs = []
        bearish_obs = []

        # Process bullish FVGs (demand zones)
        for fvg in fvgs.get("bullish_fvgs", []):
            idx = fvg["idx"]
            if idx - 1 < 0 or idx + 2 >= len(candles):
                continue

            # Middle candle is the order block candidate
            ob_candle = candles.iloc[idx]
            expansion_candle = candles.iloc[idx + 1]

            # Check if ob_candle is bearish (close < open)
            is_bearish = ob_candle["close"] < ob_candle["open"]

            # Check for bullish expansion (significant move up)
            expansion_size = expansion_candle["high"] - expansion_candle["low"]
            ob_size = ob_candle["high"] - ob_candle["low"]

            if is_bearish and expansion_size > ob_size * 1.5:
                bullish_obs.append({
                    "idx": idx,
                    "type": "bullish",
                    "top": float(ob_candle["high"]),  # Include wick
                    "bottom": float(ob_candle["low"]),  # Include wick
                    "time": str(ob_candle["open_time"]) if "open_time" in ob_candle.index else idx,
                    "fvg": fvg,
                    "mitigated": False
                })

        # Process bearish FVGs (supply zones)
        for fvg in fvgs.get("bearish_fvgs", []):
            idx = fvg["idx"]
            if idx - 1 < 0 or idx + 2 >= len(candles):
                continue

            # Middle candle is the order block candidate
            ob_candle = candles.iloc[idx]
            expansion_candle = candles.iloc[idx + 1]

            # Check if ob_candle is bullish (close > open)
            is_bullish = ob_candle["close"] > ob_candle["open"]

            # Check for bearish expansion (significant move down)
            expansion_size = expansion_candle["high"] - expansion_candle["low"]
            ob_size = ob_candle["high"] - ob_candle["low"]

            if is_bullish and expansion_size > ob_size * 1.5:
                bearish_obs.append({
                    "idx": idx,
                    "type": "bearish",
                    "top": float(ob_candle["high"]),  # Include wick
                    "bottom": float(ob_candle["low"]),  # Include wick
                    "time": str(ob_candle["open_time"]) if "open_time" in ob_candle.index else idx,
                    "fvg": fvg,
                    "mitigated": False
                })

        return {"bullish_obs": bullish_obs, "bearish_obs": bearish_obs}

# ================================================================
# STRUCTURE SUPPLY/DEMAND (SS SD)
# ================================================================

class StructureSupplyDemand:
    """
    Detects Structure Supply/Demand zones.

    Structure S/D represents multiple candles with FVG following market structure.
    - Demand: Entire bullish structure move before bearish expansion (with FVG)
    - Supply: Entire bearish structure move before bullish expansion (with FVG)
    """

    @staticmethod
    def detect_structure_zones(candles: pd.DataFrame, fvgs: Dict, pivots: Dict) -> Dict:
        """
        Detect Structure Supply/Demand zones using market structure and FVGs.

        Args:
            candles: DataFrame with OHLC data
            fvgs: Dict from FairValueGap.detect_fvgs()
            pivots: Dict from MarketStructure.detect_pivots()

        Returns: Dict with demand_zones and supply_zones lists
        """
        demand_zones = []
        supply_zones = []

        # Process bullish structure (demand zones)
        for fvg in fvgs.get("bullish_fvgs", []):
            idx = fvg["idx"]
            if idx < 5:
                continue

            # Look for structure move before this FVG
            structure_start = max(0, idx - 10)
            structure_candles = candles.iloc[structure_start:idx]

            # Check if this forms a bullish structure (higher lows)
            lows = structure_candles["low"].values
            if len(lows) >= 2:
                # Simple check: are recent lows generally higher?
                is_bullish_structure = lows[-1] > lows[0]

                if is_bullish_structure:
                    demand_zones.append({
                        "type": "demand",
                        "top": float(structure_candles["high"].max()),
                        "bottom": float(structure_candles["low"].min()),
                        "start_idx": structure_start,
                        "end_idx": idx,
                        "fvg": fvg,
                        "candle_count": len(structure_candles),
                        "mitigated": False
                    })

        # Process bearish structure (supply zones)
        for fvg in fvgs.get("bearish_fvgs", []):
            idx = fvg["idx"]
            if idx < 5:
                continue

            # Look for structure move before this FVG
            structure_start = max(0, idx - 10)
            structure_candles = candles.iloc[structure_start:idx]

            # Check if this forms a bearish structure (lower highs)
            highs = structure_candles["high"].values
            if len(highs) >= 2:
                # Simple check: are recent highs generally lower?
                is_bearish_structure = highs[-1] < highs[0]

                if is_bearish_structure:
                    supply_zones.append({
                        "type": "supply",
                        "top": float(structure_candles["high"].max()),
                        "bottom": float(structure_candles["low"].min()),
                        "start_idx": structure_start,
                        "end_idx": idx,
                        "fvg": fvg,
                        "candle_count": len(structure_candles),
                        "mitigated": False
                    })

        return {"demand_zones": demand_zones, "supply_zones": supply_zones}

# ================================================================
# ZONE LOCATION SCORING
# ================================================================

class ZoneScoring:
    """
    Scores Supply/Demand zones based on 3 key locations:
    1. Pivot points following market structure
    2. Within ranges (demand in discount, supply in premium)
    3. Above/below range for deviations
    """

    @staticmethod
    def score_zones(zones: Dict, pivots: Dict, detected_range: Optional[Dict], current_price: float) -> List[Dict]:
        """
        Score and filter zones based on location quality.

        Args:
            zones: Combined dict with order_blocks and structure_zones
            pivots: Market structure pivots
            detected_range: Current detected range (if any)
            current_price: Current market price

        Returns: List of scored zones sorted by strength
        """
        all_zones = []

        # Process Order Blocks
        for ob in zones.get("bullish_obs", []):
            score = ZoneScoring._calculate_location_score(
                zone_type="demand",
                zone_top=ob["top"],
                zone_bottom=ob["bottom"],
                pivots=pivots,
                detected_range=detected_range,
                current_price=current_price
            )
            all_zones.append({
                **ob,
                "zone_class": "order_block",
                "location_score": score["score"],
                "location_type": score["location"],
                "strength": score["score"] * 100
            })

        for ob in zones.get("bearish_obs", []):
            score = ZoneScoring._calculate_location_score(
                zone_type="supply",
                zone_top=ob["top"],
                zone_bottom=ob["bottom"],
                pivots=pivots,
                detected_range=detected_range,
                current_price=current_price
            )
            all_zones.append({
                **ob,
                "zone_class": "order_block",
                "location_score": score["score"],
                "location_type": score["location"],
                "strength": score["score"] * 100
            })

        # Process Structure Zones
        for zone in zones.get("demand_zones", []):
            score = ZoneScoring._calculate_location_score(
                zone_type="demand",
                zone_top=zone["top"],
                zone_bottom=zone["bottom"],
                pivots=pivots,
                detected_range=detected_range,
                current_price=current_price
            )
            all_zones.append({
                **zone,
                "zone_class": "structure",
                "location_score": score["score"],
                "location_type": score["location"],
                "strength": score["score"] * 85  # Slightly lower than OB
            })

        for zone in zones.get("supply_zones", []):
            score = ZoneScoring._calculate_location_score(
                zone_type="supply",
                zone_top=zone["top"],
                zone_bottom=zone["bottom"],
                pivots=pivots,
                detected_range=detected_range,
                current_price=current_price
            )
            all_zones.append({
                **zone,
                "zone_class": "structure",
                "location_score": score["score"],
                "location_type": score["location"],
                "strength": score["score"] * 85  # Slightly lower than OB
            })

        # Sort by strength descending
        all_zones.sort(key=lambda z: z["strength"], reverse=True)
        return all_zones

    @staticmethod
    def _calculate_location_score(
        zone_type: str,
        zone_top: float,
        zone_bottom: float,
        pivots: Dict,
        detected_range: Optional[Dict],
        current_price: float
    ) -> Dict:
        """Calculate location-based score for a zone."""
        zone_mid = (zone_top + zone_bottom) / 2

        # Location 1: Pivot points (highest quality)
        if ZoneScoring._is_at_pivot(zone_mid, pivots, zone_type):
            return {"score": 1.0, "location": "pivot"}

        # Location 2: Within range (discount/premium)
        if detected_range:
            range_high = detected_range.get("high", 0)
            range_low = detected_range.get("low", 0)
            range_mid = (range_high + range_low) / 2

            # Demand should be in discount (lower 50%)
            if zone_type == "demand" and zone_mid < range_mid and zone_mid >= range_low:
                return {"score": 0.85, "location": "range_discount"}

            # Supply should be in premium (upper 50%)
            if zone_type == "supply" and zone_mid > range_mid and zone_mid <= range_high:
                return {"score": 0.85, "location": "range_premium"}

        # Location 3: Above/below range for deviations
        if detected_range:
            range_high = detected_range.get("high", 0)
            range_low = detected_range.get("low", 0)

            # Demand below range low
            if zone_type == "demand" and zone_mid < range_low:
                return {"score": 0.75, "location": "below_range"}

            # Supply above range high
            if zone_type == "supply" and zone_mid > range_high:
                return {"score": 0.75, "location": "above_range"}

        # No specific location match
        return {"score": 0.5, "location": "none"}

    @staticmethod
    def _is_at_pivot(zone_mid: float, pivots: Dict, zone_type: str) -> bool:
        """Check if zone is at a pivot point."""
        tolerance = 0.005  # 0.5% tolerance

        if zone_type == "demand":
            # Check swing lows
            for pivot in pivots.get("lows", [])[-3:]:  # Last 3 lows
                pivot_price = pivot.get("price", 0)
                if abs(zone_mid - pivot_price) / pivot_price < tolerance:
                    return True

        elif zone_type == "supply":
            # Check swing highs
            for pivot in pivots.get("highs", [])[-3:]:  # Last 3 highs
                pivot_price = pivot.get("price", 0)
                if abs(zone_mid - pivot_price) / pivot_price < tolerance:
                    return True

        return False

# ================================================================
# MULTI-TIMEFRAME VALIDATION (38-Page PDF Methodology)
# ================================================================

class MultiTimeframeValidator:
    """
    Validates HTF order blocks by drilling down through multiple timeframes
    to find "true inefficiency" as described in 38-page PDF.

    Process: HTF → Intermediate TF → LTF to find where true inefficiency exists
    """

    @staticmethod
    def validate_htf_zone_on_ltf(htf_zone: Dict, ltf_candles: pd.DataFrame) -> Dict:
        """
        Validate if HTF zone has true inefficiency on LTF.

        Returns validation status with LTF structure details.
        """
        if ltf_candles is None or len(ltf_candles) < 10:
            return {"has_true_inefficiency": False, "ltf_structure": None}

        zone_top = htf_zone.get("top", 0)
        zone_bottom = htf_zone.get("bottom", 0)

        # Find candles within HTF zone range
        in_zone = ltf_candles[
            (ltf_candles["low"] <= zone_top) &
            (ltf_candles["high"] >= zone_bottom)
        ]

        if len(in_zone) == 0:
            return {"has_true_inefficiency": False, "ltf_structure": None}

        # Detect FVG on LTF within this zone
        fvg_detector = FairValueGap()
        ltf_fvgs = fvg_detector.detect_fvgs(in_zone)

        zone_type = htf_zone.get("type", "unknown")
        ltf_fvg_list = ltf_fvgs.get("bullish_fvgs" if zone_type == "demand" else "bearish_fvgs", [])

        has_inefficiency = len(ltf_fvg_list) > 0

        # Check for LTF structure (higher highs/higher lows or lower highs/lower lows)
        ltf_structure = None
        if has_inefficiency and len(in_zone) >= 6:
            ltf_structure = MultiTimeframeValidator._detect_ltf_structure(in_zone, zone_type)

        return {
            "has_true_inefficiency": has_inefficiency,
            "ltf_structure": ltf_structure,
            "ltf_fvg_count": len(ltf_fvg_list),
            "ltf_fvgs": ltf_fvg_list[:3] if ltf_fvg_list else []  # Top 3 for detail
        }

    @staticmethod
    def _detect_ltf_structure(candles: pd.DataFrame, zone_type: str) -> Optional[Dict]:
        """Detect if LTF shows proper structure (HH/HL or LH/LL)."""
        if len(candles) < 6:
            return None

        highs = candles["high"].values
        lows = candles["low"].values

        if zone_type == "demand":
            # Look for higher highs and higher lows (bullish structure)
            has_hh = highs[-1] > highs[0]
            has_hl = lows[-1] > lows[0]

            if has_hh and has_hl:
                return {
                    "type": "bullish_structure",
                    "higher_highs": True,
                    "higher_lows": True,
                    "strength": 0.9
                }

        elif zone_type == "supply":
            # Look for lower highs and lower lows (bearish structure)
            has_lh = highs[-1] < highs[0]
            has_ll = lows[-1] < lows[0]

            if has_lh and has_ll:
                return {
                    "type": "bearish_structure",
                    "lower_highs": True,
                    "lower_lows": True,
                    "strength": 0.9
                }

        return None

# ================================================================
# ZONE REFINEMENT & PARTIAL MITIGATION
# ================================================================

class ZoneRefinement:
    """
    Refines zones based on mitigation and identifies un-mitigated portions.

    Per 38-page PDF: "Market doesn't always fill entire inefficiency -
    sometimes just taps and rejects. Resize order block to fit bottom edge
    of inefficiency."
    """

    @staticmethod
    def refine_zone_after_mitigation(zone: Dict, recent_candles: pd.DataFrame) -> Dict:
        """
        Refine zone boundaries based on partial mitigation.

        Returns refined zone with updated boundaries showing un-mitigated area.
        """
        zone_top = zone.get("top", 0)
        zone_bottom = zone.get("bottom", 0)
        zone_type = zone.get("type", "unknown")

        # Find candles that entered the zone
        touched_zone = recent_candles[
            (recent_candles["low"] <= zone_top) &
            (recent_candles["high"] >= zone_bottom)
        ]

        if len(touched_zone) == 0:
            # No mitigation yet
            return {
                **zone,
                "mitigation_percent": 0.0,
                "refined_top": zone_top,
                "refined_bottom": zone_bottom,
                "is_refined": False
            }

        # Calculate how much of zone was mitigated
        if zone_type == "supply":
            # For supply, check lowest low that entered zone
            lowest_penetration = touched_zone["low"].min()
            mitigation_percent = ((zone_top - lowest_penetration) / (zone_top - zone_bottom)) * 100

            # Refine: move bottom edge to lowest penetration point
            refined_bottom = lowest_penetration
            refined_top = zone_top

        else:  # demand
            # For demand, check highest high that entered zone
            highest_penetration = touched_zone["high"].max()
            mitigation_percent = ((highest_penetration - zone_bottom) / (zone_top - zone_bottom)) * 100

            # Refine: move top edge to highest penetration point
            refined_top = highest_penetration
            refined_bottom = zone_bottom

        return {
            **zone,
            "mitigation_percent": min(mitigation_percent, 100.0),
            "refined_top": float(refined_top),
            "refined_bottom": float(refined_bottom),
            "is_refined": True,
            "original_top": zone_top,
            "original_bottom": zone_bottom
        }

# ================================================================
# PIVOT QUALITY & LIQUIDITY SWEEP DETECTION
# ================================================================

class PivotQualityScorer:
    """
    Scores pivots based on quality and liquidity sweep characteristics.

    Per 38-page PDF: "1st pivot point after extreme liquidity sweep =
    extremely high quality. Block itself swept liquidity from left and
    visible in macro pivot point."
    """

    @staticmethod
    def score_pivot_quality(zone: Dict, pivots: Dict, candles: pd.DataFrame) -> Dict:
        """
        Score zone quality based on pivot position and liquidity characteristics.

        Returns enhanced score with pivot rank (1st, 2nd, 3rd) and liquidity sweep detection.
        """
        zone_mid = (zone.get("top", 0) + zone.get("bottom", 0)) / 2
        zone_type = zone.get("type", "unknown")

        # Determine if zone is at pivot and which pivot (1st, 2nd, 3rd)
        pivot_rank = PivotQualityScorer._get_pivot_rank(zone_mid, pivots, zone_type)

        # Check for liquidity sweep
        has_liquidity_sweep = PivotQualityScorer._detect_liquidity_sweep(
            zone, candles, pivots, zone_type
        )

        # Calculate quality score
        base_score = 0.5

        if pivot_rank == 1:
            base_score = 1.0  # 1st pivot = highest quality
        elif pivot_rank == 2:
            base_score = 0.85  # 2nd pivot = high quality
        elif pivot_rank == 3:
            base_score = 0.70  # 3rd pivot = good quality

        # Boost if has liquidity sweep
        if has_liquidity_sweep:
            base_score = min(base_score * 1.15, 1.0)

        return {
            "pivot_rank": pivot_rank,
            "has_liquidity_sweep": has_liquidity_sweep,
            "pivot_quality_score": base_score,
            "quality_label": PivotQualityScorer._get_quality_label(base_score, pivot_rank, has_liquidity_sweep)
        }

    @staticmethod
    def _get_pivot_rank(zone_mid: float, pivots: Dict, zone_type: str) -> int:
        """Determine if zone is at 1st, 2nd, or 3rd pivot point."""
        tolerance = 0.01  # 1% tolerance

        pivot_list = pivots.get("lows", []) if zone_type == "demand" else pivots.get("highs", [])

        # Check last 3 pivots (most recent)
        for rank, pivot in enumerate(reversed(pivot_list[-3:]), start=1):
            pivot_price = pivot.get("price", 0)
            if abs(zone_mid - pivot_price) / pivot_price < tolerance:
                return rank

        return 0  # Not at any pivot

    @staticmethod
    def _detect_liquidity_sweep(zone: Dict, candles: pd.DataFrame, pivots: Dict, zone_type: str) -> bool:
        """
        Detect if zone formed with liquidity sweep.

        Liquidity sweep = wick through previous high/low followed by reversal.
        """
        zone_idx = zone.get("idx", zone.get("start_idx", 0))

        if zone_idx < 5 or zone_idx >= len(candles) - 1:
            return False

        lookback = candles.iloc[max(0, zone_idx - 10):zone_idx]
        zone_candle = candles.iloc[zone_idx]

        if len(lookback) < 3:
            return False

        if zone_type == "supply":
            # Check if zone candle wicked above recent highs
            recent_high = lookback["high"].max()
            zone_high = zone_candle["high"]
            zone_close = zone_candle["close"]

            # Sweep = high exceeds recent high but close is below it
            if zone_high > recent_high and zone_close < recent_high:
                return True

        else:  # demand
            # Check if zone candle wicked below recent lows
            recent_low = lookback["low"].min()
            zone_low = zone_candle["low"]
            zone_close = zone_candle["close"]

            # Sweep = low breaks recent low but close is above it
            if zone_low < recent_low and zone_close > recent_low:
                return True

        return False

    @staticmethod
    def _get_quality_label(score: float, pivot_rank: int, has_sweep: bool) -> str:
        """Generate human-readable quality label."""
        if pivot_rank == 1 and has_sweep:
            return "EXTREME_HIGH_QUALITY"
        elif pivot_rank == 1:
            return "HIGH_QUALITY"
        elif pivot_rank == 2 and has_sweep:
            return "HIGH_QUALITY"
        elif pivot_rank == 2:
            return "GOOD_QUALITY"
        elif pivot_rank == 3:
            return "MEDIUM_QUALITY"
        elif has_sweep:
            return "MEDIUM_QUALITY"
        else:
            return "LOW_QUALITY"

# ================================================================
# OVERLAPPING ZONE DETECTOR
# ================================================================

class OverlappingZoneDetector:
    """
    Detects if LTF structure zones properly OVERLAP (not just within) HTF zones.

    Per 38-page PDF: "If LTF zone is just within but not overlapping,
    it's not representing the entire HTF block. Overlapping means boundaries
    extend beyond both top and bottom."
    """

    @staticmethod
    def check_overlap(htf_zone: Dict, ltf_zone: Dict) -> Dict:
        """
        Check if LTF zone overlaps HTF zone properly.

        True overlap = LTF zone boundaries extend beyond HTF zone boundaries.
        """
        htf_top = htf_zone.get("top", 0)
        htf_bottom = htf_zone.get("bottom", 0)
        ltf_top = ltf_zone.get("top", 0)
        ltf_bottom = ltf_zone.get("bottom", 0)

        # Check if LTF is just "within" HTF (not overlapping)
        ltf_within = (ltf_top <= htf_top and ltf_bottom >= htf_bottom)

        # Check if LTF overlaps HTF (extends beyond boundaries)
        ltf_overlaps_top = ltf_top > htf_top and ltf_bottom <= htf_top and ltf_bottom >= htf_bottom
        ltf_overlaps_bottom = ltf_bottom < htf_bottom and ltf_top >= htf_bottom and ltf_top <= htf_top
        ltf_fully_overlaps = ltf_top > htf_top and ltf_bottom < htf_bottom

        is_overlapping = ltf_overlaps_top or ltf_overlaps_bottom or ltf_fully_overlaps

        # Calculate overlap percentage
        overlap_top = min(htf_top, ltf_top)
        overlap_bottom = max(htf_bottom, ltf_bottom)
        overlap_range = overlap_top - overlap_bottom
        htf_range = htf_top - htf_bottom

        overlap_percent = (overlap_range / htf_range * 100) if htf_range > 0 else 0

        return {
            "is_overlapping": bool(is_overlapping),
            "is_within_only": bool(ltf_within and not is_overlapping),
            "overlap_percent": float(min(overlap_percent, 100.0)),
            "represents_htf": bool(is_overlapping),  # Only overlapping zones represent HTF
            "overlap_quality": "EXCELLENT" if overlap_percent > 80 else "GOOD" if overlap_percent > 50 else "WEAK"
        }

    @staticmethod
    def find_overlapping_zones(htf_zones: List[Dict], ltf_zones: List[Dict]) -> List[Dict]:
        """
        Find all LTF zones that properly overlap HTF zones.

        Returns list of HTF zones with their overlapping LTF zones.
        """
        enhanced_htf_zones = []

        for htf_zone in htf_zones:
            overlapping_ltf = []

            for ltf_zone in ltf_zones:
                # Check type match
                if htf_zone.get("type") != ltf_zone.get("type"):
                    continue

                overlap_result = OverlappingZoneDetector.check_overlap(htf_zone, ltf_zone)

                if overlap_result["is_overlapping"]:
                    overlapping_ltf.append({
                        **ltf_zone,
                        "overlap_details": overlap_result
                    })

            enhanced_htf_zones.append({
                **htf_zone,
                "has_ltf_overlap": bool(len(overlapping_ltf) > 0),
                "overlapping_ltf_zones": overlapping_ltf,
                "ltf_overlap_count": len(overlapping_ltf)
            })

        return enhanced_htf_zones

# ================================================================
# RANGE DETECTION (TCT Mentorship Lecture 2 - Pure TCT Methodology)
# ================================================================

class TCTRangeDetector:
    """
    Detects and validates ranges using PURE TCT Lecture 2 methodology.

    Pure TCT Methodology from "TCT-2024-mentorship-Lecture-2-Ranges-AI-Text-Only.pdf":
    - Range = price moving sideways (consolidation phase)
    - Purpose: Restore balance between buyers and sellers after aggressive expansion
    - Uptrend: Pull range from TOP to BOTTOM (high → low)
    - Downtrend: Pull range from BOTTOM to TOP (low → high)
    - Range confirmed when price moves back to EQUILIBRIUM (0.5 fib level)
    - Six Candle Rule: Range valid if 2 candles up, 2 down, 2 up (uptrend) or inverse
    - Deviation Limit (DL): 30% of range size - threshold for deviation vs break
    - Premium (above 0.5) vs Discount (below 0.5) pricing zones
    - Ranges within ranges: Multiple timeframe nesting
    - Good range = horizontal price action, Bad range = sharp V-shaped moves
    """

    DEVIATION_LIMIT_PERCENT = 0.30  # TCT: 30% of range size for DL

    @staticmethod
    def detect_ranges(candles: pd.DataFrame, pivots: Dict) -> Dict:
        """
        Detect ranges using TCT Lecture 2 methodology.

        TCT: "A range is when price is simply moving sideways"
        TCT: "After aggressive expansion, price likes to form a range"

        Returns: Dict with detected ranges and their properties
        """
        if len(candles) < 10:
            return {"ranges": [], "active_range": None}

        ranges = []
        trend = pivots.get("trend", "neutral")

        # Get swing points for range detection
        swing_highs = pivots.get("highs", [])
        swing_lows = pivots.get("lows", [])

        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return {"ranges": [], "active_range": None}

        # TCT: Detect ranges based on trend direction
        # "When we're trending up, we pull our range from top to bottom"
        # "When we're trending down, we pull our range from bottom to top"

        # Find potential range formations
        for i in range(len(swing_highs) - 1):
            for j in range(len(swing_lows) - 1):
                high_pivot = swing_highs[i]
                low_pivot = swing_lows[j]

                # TCT: Ensure proper sequence (high comes before low for uptrend range)
                if abs(high_pivot["idx"] - low_pivot["idx"]) > 50:
                    continue  # Too far apart

                range_high = high_pivot["price"]
                range_low = low_pivot["price"]

                if range_high <= range_low:
                    continue  # Invalid range

                equilibrium = (range_high + range_low) / 2

                # TCT: Check for equilibrium touch (range confirmation)
                # "When we have a move back to the equilibrium, that's when the range is confirmed"
                eq_touched = TCTRangeDetector._check_equilibrium_touch(
                    candles, high_pivot["idx"], low_pivot["idx"], equilibrium
                )

                if not eq_touched:
                    continue

                # TCT: Validate with Six Candle Rule
                six_candle_valid = TCTRangeDetector._validate_six_candle_rule(
                    candles, high_pivot["idx"], low_pivot["idx"], trend
                )

                # Calculate range properties
                range_size = range_high - range_low
                deviation_limit_high = range_high + (range_size * TCTRangeDetector.DEVIATION_LIMIT_PERCENT)
                deviation_limit_low = range_low - (range_size * TCTRangeDetector.DEVIATION_LIMIT_PERCENT)

                # Determine range quality
                quality = TCTRangeDetector._assess_range_quality(
                    candles, high_pivot["idx"], low_pivot["idx"], range_high, range_low
                )

                ranges.append({
                    "range_high": float(range_high),
                    "range_low": float(range_low),
                    "equilibrium": float(equilibrium),
                    "range_size": float(range_size),
                    "deviation_limit_high": float(deviation_limit_high),
                    "deviation_limit_low": float(deviation_limit_low),
                    "high_idx": int(high_pivot["idx"]),
                    "low_idx": int(low_pivot["idx"]),
                    "direction": "uptrend" if high_pivot["idx"] < low_pivot["idx"] else "downtrend",
                    "equilibrium_touched": bool(eq_touched),
                    "six_candle_valid": bool(six_candle_valid),
                    "quality": quality,
                    "is_confirmed": bool(eq_touched and six_candle_valid)
                })

        # Sort by recency and quality
        ranges.sort(key=lambda r: (r["is_confirmed"], r["quality"].get("score", 0)), reverse=True)

        # Get most recent active range
        active_range = ranges[0] if ranges else None

        return {
            "ranges": ranges[:5],  # Top 5 ranges
            "active_range": active_range,
            "total_ranges_found": len(ranges)
        }

    @staticmethod
    def _check_equilibrium_touch(candles: pd.DataFrame, high_idx: int, low_idx: int, equilibrium: float) -> bool:
        """
        Check if price touched equilibrium after range formation.

        TCT: "Move back up towards the equilibrium of the range - that's when the range is confirmed"
        TCT: "Your range low is already confirmed the moment we have a move back up touching the equilibrium"
        """
        start_idx = min(high_idx, low_idx)
        end_idx = max(high_idx, low_idx)

        # Check candles after the range formed
        check_start = end_idx + 1
        check_end = min(check_start + 20, len(candles))

        if check_start >= len(candles):
            return False

        for i in range(check_start, check_end):
            candle = candles.iloc[i]
            # TCT: Check if any candle touched the equilibrium (0.5 level)
            if candle["low"] <= equilibrium <= candle["high"]:
                return True

        return False

    @staticmethod
    def _validate_six_candle_rule(candles: pd.DataFrame, high_idx: int, low_idx: int, trend: str) -> bool:
        """
        Validate range using TCT Six Candle Rule.

        TCT: "A range is only valid on a certain timeframe if the six candle rule applies"
        TCT: "You want to have two candles up, two candles down, and two candles back up again"
        TCT: "That's when the range is valid on that specific timeframe"
        """
        start_idx = min(high_idx, low_idx)
        end_idx = max(high_idx, low_idx)

        if end_idx - start_idx < 6:
            return False  # Not enough candles

        # Count directional candles in the range
        up_candles = 0
        down_candles = 0
        neutral_candles = 0

        for i in range(start_idx, min(end_idx + 1, len(candles))):
            candle = candles.iloc[i]
            if candle["close"] > candle["open"]:
                up_candles += 1
            elif candle["close"] < candle["open"]:
                down_candles += 1
            else:
                neutral_candles += 1

        # TCT: Need at least 2 candles each direction for valid six candle rule
        return up_candles >= 2 and down_candles >= 2

    @staticmethod
    def _assess_range_quality(candles: pd.DataFrame, high_idx: int, low_idx: int,
                               range_high: float, range_low: float) -> Dict:
        """
        Assess range quality using TCT methodology.

        TCT: "Be rational - a range is when prices are going sideways"
        TCT: "V-shaped moves are often Market structure, not ranges"
        TCT: "When the V gets very extended and wide - that's when you have a good range"
        TCT: "Good range = horizontal price action, Bad range = sharp V-shaped moves"
        """
        start_idx = min(high_idx, low_idx)
        end_idx = max(high_idx, low_idx)
        range_candles = candles.iloc[start_idx:end_idx + 1]

        if len(range_candles) < 3:
            return {"score": 0.0, "quality_label": "INVALID", "is_horizontal": False}

        # Calculate how "horizontal" the range is
        equilibrium = (range_high + range_low) / 2
        range_size = range_high - range_low

        # Measure time spent in each zone
        time_in_premium = 0
        time_in_discount = 0
        time_near_equilibrium = 0

        for _, candle in range_candles.iterrows():
            mid_price = (candle["high"] + candle["low"]) / 2
            if mid_price > equilibrium + (range_size * 0.1):
                time_in_premium += 1
            elif mid_price < equilibrium - (range_size * 0.1):
                time_in_discount += 1
            else:
                time_near_equilibrium += 1

        total_candles = len(range_candles)

        # TCT: Good ranges have balanced time in premium/discount
        # V-shaped moves spend all time on one side
        balance_ratio = min(time_in_premium, time_in_discount) / max(time_in_premium, time_in_discount, 1)

        # Calculate how wide/extended the range is (horizontality)
        candle_span = end_idx - start_idx
        price_change_per_candle = range_size / max(candle_span, 1)
        avg_candle_size = range_candles["high"].mean() - range_candles["low"].mean()

        # TCT: Extended V (horizontal) is good, sharp V is bad
        horizontality = 1.0 - min(price_change_per_candle / avg_candle_size, 1.0) if avg_candle_size > 0 else 0

        # Calculate overall score
        score = (balance_ratio * 0.4 + horizontality * 0.4 + (time_near_equilibrium / total_candles) * 0.2)

        if score >= 0.7:
            quality_label = "EXCELLENT"
        elif score >= 0.5:
            quality_label = "GOOD"
        elif score >= 0.3:
            quality_label = "MODERATE"
        else:
            quality_label = "WEAK"

        return {
            "score": float(round(score, 3)),
            "quality_label": quality_label,
            "is_horizontal": bool(horizontality > 0.5),
            "balance_ratio": float(round(balance_ratio, 3)),
            "horizontality": float(round(horizontality, 3)),
            "time_in_premium": int(time_in_premium),
            "time_in_discount": int(time_in_discount),
            "time_near_equilibrium": int(time_near_equilibrium)
        }


class TCTDeviationDetector:
    """
    Detects range deviations using TCT Lecture 2 methodology.

    Pure TCT Methodology:
    - Deviation = price exceeds range high/low but doesn't break it
    - Wick deviation = easiest (never broke structure)
    - Candle close deviation = bad break of structure (closes back inside quickly)
    - Deviation Limit (DL) = 30% of range size threshold
    - TCT: "When breaking range but not closing above DL, it's still a deviation"
    - TCT: "Extend range to deviation high after deviation comes back inside"
    """

    @staticmethod
    def detect_deviations(candles: pd.DataFrame, active_range: Dict) -> Dict:
        """
        Detect deviations from the active range.

        TCT: "A deviation is when price exceeds your range high or low but it does not break it"
        TCT: "When this happens we can expect a reversal towards the range low or close to it"

        Returns: Dict with high_deviations and low_deviations
        """
        if not active_range:
            return {"high_deviations": [], "low_deviations": [], "total_deviations": 0}

        range_high = active_range["range_high"]
        range_low = active_range["range_low"]
        dl_high = active_range["deviation_limit_high"]
        dl_low = active_range["deviation_limit_low"]
        range_size = active_range["range_size"]

        high_deviations = []
        low_deviations = []

        # Start checking after range formation
        start_idx = max(active_range.get("high_idx", 0), active_range.get("low_idx", 0)) + 1

        i = start_idx
        while i < len(candles):
            candle = candles.iloc[i]

            # TCT: Check for high deviation
            if candle["high"] > range_high:
                deviation = TCTDeviationDetector._classify_deviation(
                    candles, i, "high", range_high, dl_high, range_size
                )
                if deviation:
                    high_deviations.append(deviation)
                    # Skip ahead past the deviation
                    i += deviation.get("duration_candles", 1)
                    continue

            # TCT: Check for low deviation
            if candle["low"] < range_low:
                deviation = TCTDeviationDetector._classify_deviation(
                    candles, i, "low", range_low, dl_low, range_size
                )
                if deviation:
                    low_deviations.append(deviation)
                    # Skip ahead past the deviation
                    i += deviation.get("duration_candles", 1)
                    continue

            i += 1

        return {
            "high_deviations": high_deviations,
            "low_deviations": low_deviations,
            "total_deviations": len(high_deviations) + len(low_deviations),
            "has_deviation": len(high_deviations) + len(low_deviations) > 0
        }

    @staticmethod
    def _classify_deviation(candles: pd.DataFrame, start_idx: int, direction: str,
                            range_level: float, dl_level: float, range_size: float) -> Optional[Dict]:
        """
        Classify deviation type using TCT methodology.

        TCT Deviation Types:
        1. Wick deviation - "Easiest deviations will always be the wicks"
        2. Candle close deviation - "Bad break of structure falls into category of deviations"
        3. DL-based deviation - "When breaking range but not closing above DL"

        TCT: "The bad break of structure was whenever you break your range low
        but the second candle immediately comes and closes back inside"
        """
        if start_idx >= len(candles):
            return None

        candle = candles.iloc[start_idx]
        exceeded_dl = False
        is_wick = False
        is_bad_bos = False
        duration = 1
        max_deviation = 0.0
        came_back_inside = False

        if direction == "high":
            # TCT: Check if it's a wick deviation (no close above range high)
            is_wick = candle["close"] <= range_level and candle["high"] > range_level

            # Check if exceeded DL
            exceeded_dl = candle["close"] > dl_level

            # Calculate max deviation
            max_deviation = candle["high"] - range_level

            # TCT: Check for bad BOS (closes above then quickly comes back)
            if candle["close"] > range_level:
                # Look for quick return inside range
                for j in range(start_idx + 1, min(start_idx + 5, len(candles))):
                    next_candle = candles.iloc[j]
                    duration += 1
                    max_deviation = max(max_deviation, next_candle["high"] - range_level)

                    if next_candle["close"] <= range_level:
                        is_bad_bos = True
                        came_back_inside = True
                        break

                    # TCT: If closes above DL, it's likely a real break
                    if next_candle["close"] > dl_level:
                        exceeded_dl = True

        else:  # direction == "low"
            # TCT: Check if it's a wick deviation (no close below range low)
            is_wick = candle["close"] >= range_level and candle["low"] < range_level

            # Check if exceeded DL
            exceeded_dl = candle["close"] < dl_level

            # Calculate max deviation
            max_deviation = range_level - candle["low"]

            # TCT: Check for bad BOS
            if candle["close"] < range_level:
                for j in range(start_idx + 1, min(start_idx + 5, len(candles))):
                    next_candle = candles.iloc[j]
                    duration += 1
                    max_deviation = max(max_deviation, range_level - next_candle["low"])

                    if next_candle["close"] >= range_level:
                        is_bad_bos = True
                        came_back_inside = True
                        break

                    if next_candle["close"] < dl_level:
                        exceeded_dl = True

        # TCT: If exceeded DL with close, it's likely a real break, not deviation
        if exceeded_dl and not came_back_inside:
            return None  # This is a range break, not a deviation

        # Determine deviation type
        if is_wick:
            deviation_type = "WICK"
            quality = "EXCELLENT"  # TCT: "Wicks are always the easiest deviations"
        elif is_bad_bos:
            deviation_type = "BAD_BOS"
            quality = "GOOD"  # TCT: "Bad break of structure falls into category of deviations"
        else:
            deviation_type = "CANDLE_CLOSE"
            quality = "MODERATE"

        deviation_percent = (max_deviation / range_size) * 100

        return {
            "direction": direction,
            "type": deviation_type,
            "quality": quality,
            "start_idx": int(start_idx),
            "duration_candles": int(duration),
            "max_deviation_price": float(max_deviation),
            "max_deviation_percent": float(round(deviation_percent, 2)),
            "exceeded_dl": bool(exceeded_dl),
            "came_back_inside": bool(came_back_inside),
            "is_valid_deviation": bool(not exceeded_dl or came_back_inside)
        }


class TCTPremiumDiscountClassifier:
    """
    Classifies price zones as Premium or Discount using TCT methodology.

    Pure TCT Methodology:
    - Premium = Above equilibrium (0.5) to range high
    - Discount = Below equilibrium (0.5) to range low
    - TCT: "From equilibrium towards the range low - that's what we call Discount pricing"
    - TCT: "Above the 0.5 to range high is what we call Premium pricing"
    - TCT: "We can have many rotations from premium back to discount"
    """

    @staticmethod
    def classify_current_position(current_price: float, active_range: Dict) -> Dict:
        """
        Classify current price position within range.

        TCT: "Premium section = above 0.5 to range high"
        TCT: "Discount section = from equilibrium towards range low"
        """
        if not active_range:
            return {"zone": "UNKNOWN", "distance_to_eq": 0.0}

        range_high = active_range["range_high"]
        range_low = active_range["range_low"]
        equilibrium = active_range["equilibrium"]
        range_size = active_range["range_size"]

        # Calculate position
        if current_price >= range_high:
            zone = "ABOVE_RANGE"
            position_percent = 100 + ((current_price - range_high) / range_size * 100)
        elif current_price <= range_low:
            zone = "BELOW_RANGE"
            position_percent = -((range_low - current_price) / range_size * 100)
        elif current_price > equilibrium:
            zone = "PREMIUM"
            position_percent = 50 + ((current_price - equilibrium) / (range_high - equilibrium) * 50)
        elif current_price < equilibrium:
            zone = "DISCOUNT"
            position_percent = ((current_price - range_low) / (equilibrium - range_low) * 50)
        else:
            zone = "EQUILIBRIUM"
            position_percent = 50.0

        distance_to_eq = abs(current_price - equilibrium)
        distance_to_eq_percent = (distance_to_eq / range_size) * 100

        # TCT: Trading bias based on zone
        if zone == "DISCOUNT":
            trading_bias = "LONG"  # TCT: Look for longs in discount
            bias_strength = (equilibrium - current_price) / (equilibrium - range_low)
        elif zone == "PREMIUM":
            trading_bias = "SHORT"  # TCT: Look for shorts in premium
            bias_strength = (current_price - equilibrium) / (range_high - equilibrium)
        elif zone == "ABOVE_RANGE":
            trading_bias = "SHORT"  # Above range = potential deviation short
            bias_strength = 0.8
        elif zone == "BELOW_RANGE":
            trading_bias = "LONG"  # Below range = potential deviation long
            bias_strength = 0.8
        else:
            trading_bias = "NEUTRAL"
            bias_strength = 0.0

        return {
            "zone": zone,
            "position_percent": round(position_percent, 2),
            "distance_to_equilibrium": round(distance_to_eq, 6),
            "distance_to_equilibrium_percent": round(distance_to_eq_percent, 2),
            "trading_bias": trading_bias,
            "bias_strength": round(bias_strength, 3)
        }

    @staticmethod
    def get_zone_targets(active_range: Dict) -> Dict:
        """
        Get trading targets based on current range zones.

        TCT: "If we deviated the high, we can expect a reversal towards the range low"
        TCT: "If you deviate the low, we can expect a move back up towards the Range High"
        """
        if not active_range:
            return {"premium_target": None, "discount_target": None}

        range_high = active_range["range_high"]
        range_low = active_range["range_low"]
        equilibrium = active_range["equilibrium"]
        range_size = active_range["range_size"]

        # TCT: Define key target levels
        return {
            "range_high": float(range_high),
            "range_low": float(range_low),
            "equilibrium": float(equilibrium),
            "premium_entry_zone": {
                "top": float(range_high),
                "bottom": float(equilibrium + range_size * 0.1)  # Upper 40% of range
            },
            "discount_entry_zone": {
                "top": float(equilibrium - range_size * 0.1),  # Lower 40% of range
                "bottom": float(range_low)
            },
            "deviation_limit_high": float(active_range["deviation_limit_high"]),
            "deviation_limit_low": float(active_range["deviation_limit_low"])
        }


class TCTRangesWithinRanges:
    """
    Detects nested ranges (ranges within ranges) using TCT methodology.

    Pure TCT Methodology:
    - TCT: "Ranges within ranges are so so so common"
    - TCT: "You literally almost all of the time have it - range within the range"
    - TCT: "High time frame ranges are more important than low time frame ranges"
    - TCT: "Watch your most recent expansions - that's the most important structure"
    - TCT: "The same way HTF market structure is more important, HTF ranges are more important"
    """

    @staticmethod
    def detect_nested_ranges(htf_ranges: List[Dict], ltf_ranges: List[Dict]) -> Dict:
        """
        Detect ranges nested within higher timeframe ranges.

        TCT: "This black range inside, in the Black Range we have a red range,
        and in the red range we have a blue range"

        Returns: Dict with nested range relationships
        """
        nested_relationships = []

        for htf_range in htf_ranges:
            htf_high = htf_range["range_high"]
            htf_low = htf_range["range_low"]

            ltf_inside = []
            for ltf_range in ltf_ranges:
                ltf_high = ltf_range["range_high"]
                ltf_low = ltf_range["range_low"]

                # Check if LTF range is inside HTF range
                if ltf_high <= htf_high and ltf_low >= htf_low:
                    # Calculate position within HTF range
                    htf_size = htf_high - htf_low
                    ltf_mid = (ltf_high + ltf_low) / 2
                    position_in_htf = ((ltf_mid - htf_low) / htf_size) * 100

                    ltf_inside.append({
                        **ltf_range,
                        "position_in_htf_percent": float(round(position_in_htf, 2)),
                        "is_in_premium": bool(position_in_htf > 50),
                        "is_in_discount": bool(position_in_htf < 50)
                    })

            if ltf_inside:
                nested_relationships.append({
                    "htf_range": htf_range,
                    "ltf_ranges_inside": ltf_inside,
                    "ltf_count": len(ltf_inside)
                })

        return {
            "nested_relationships": nested_relationships,
            "htf_ranges_with_nesting": len(nested_relationships),
            "total_nested_ltf_ranges": sum(rel["ltf_count"] for rel in nested_relationships)
        }

    @staticmethod
    def identify_most_recent_rotation(candles: pd.DataFrame, active_range: Dict) -> Dict:
        """
        Identify the most recent rotation within a range.

        TCT: "When we're ranging, the most important Market structure pool
        to watch is just your most recent expansion"
        TCT: "Just watch your most recent rotations within the range"
        """
        if not active_range or len(candles) < 10:
            return {"rotation": None, "direction": "unknown"}

        range_high = active_range["range_high"]
        range_low = active_range["range_low"]
        equilibrium = active_range["equilibrium"]

        # Find recent price action
        recent_candles = candles.tail(20)

        # Determine rotation direction
        first_close = recent_candles.iloc[0]["close"]
        last_close = recent_candles.iloc[-1]["close"]

        if last_close > first_close:
            direction = "UP"
            # TCT: "We're moving up - watch structure from low up"
        else:
            direction = "DOWN"
            # TCT: "We're moving down - watch structure from high down"

        # Find rotation extremes
        rotation_high = recent_candles["high"].max()
        rotation_low = recent_candles["low"].min()

        # Calculate rotation within range context
        crossed_equilibrium = (
            recent_candles["low"].min() < equilibrium < recent_candles["high"].max()
        )

        return {
            "direction": direction,
            "rotation_high": float(rotation_high),
            "rotation_low": float(rotation_low),
            "crossed_equilibrium": crossed_equilibrium,
            "started_in_zone": "PREMIUM" if first_close > equilibrium else "DISCOUNT",
            "ended_in_zone": "PREMIUM" if last_close > equilibrium else "DISCOUNT"
        }


# ================================================================
# LIQUIDITY DETECTION (TCT Mentorship Lecture 4 - Pure TCT Methodology)
# ================================================================

class LiquidityDetector:
    """
    Detects Buy-Side Liquidity (BSL) and Sell-Side Liquidity (SSL) pools.

    Pure TCT Methodology from Liquidity PDFs:
    - BSL = Buy-Side Liquidity above highs (stop losses from shorts)
    - SSL = Sell-Side Liquidity below lows (stop losses from longs)
    - Primary highs/lows = visible pivot points
    - Internal highs/lows = price action between primary highs/lows
    - Equal highs/lows = exact same price levels (amazing liquidity targets)
    - Non-liquidity highs/lows = grabbed liquidity from left, NOT liquidity targets
    - Exception: If price spends time near swept level, it becomes liquidity curve
    """

    @staticmethod
    def detect_liquidity_pools(candles: pd.DataFrame, pivots: Dict, current_price: float) -> Dict:
        """
        Detect BSL (above highs) and SSL (below lows) liquidity pools.

        TCT Methodology:
        - Primary highs/lows = visible pivot points (major liquidity)
        - Internal highs/lows = between primaries (less significant)
        - Equal highs/lows = exact same price (amazing liquidity targets)

        Returns: Dict with bsl_pools and ssl_pools lists
        """
        bsl_pools = []  # Buy-side liquidity (above price)
        ssl_pools = []  # Sell-side liquidity (below price)

        # Classify pivots as primary vs internal (TCT: visible pivots vs between pivots)
        primary_highs, internal_highs = LiquidityDetector._classify_primary_vs_internal(
            pivots.get("highs", []), is_highs=True
        )

        primary_lows, internal_lows = LiquidityDetector._classify_primary_vs_internal(
            pivots.get("lows", []), is_lows=True
        )

        # Detect equal highs (TCT: "amazing liquidity targets")
        equal_highs = LiquidityDetector._detect_equal_levels(
            [p["price"] for p in pivots.get("highs", [])],
            tolerance=0.0001  # Exact same price level (e.g., 1.07833 = 1.07833)
        )

        # Detect equal lows (TCT: "amazing liquidity targets")
        equal_lows = LiquidityDetector._detect_equal_levels(
            [p["price"] for p in pivots.get("lows", [])],
            tolerance=0.0001  # Exact same price level
        )

        # Create BSL pools from primary highs
        # TCT: Primary highs are visible pivot points representing liquidity above
        for pivot in primary_highs:
            price = pivot["price"]

            # TCT: Check if this is a non-liquidity high (grabbed liquidity from left)
            is_non_liquidity = LiquidityDetector._is_non_liquidity_high(
                pivot, pivots.get("highs", []), candles
            )

            # TCT: Only use liquidity highs (not non-liquidity highs) for BSL pools
            if not is_non_liquidity:
                # TCT: Equal highs are "amazing liquidity targets"
                is_equal = any(abs(price - eq) / eq < 0.0001 for eq in equal_highs)

                bsl_pools.append({
                    "type": "BSL",
                    "price": float(price),
                    "idx": int(pivot["idx"]),
                    "is_primary": True,
                    "is_equal": bool(is_equal),
                    "strength": 1.0 if is_equal else 0.8,  # TCT: Equal = stronger
                    "distance_from_price": float((price - current_price) / current_price * 100)
                })

        # Create SSL pools from primary lows
        # TCT: Primary lows are visible pivot points representing liquidity below
        for pivot in primary_lows:
            price = pivot["price"]

            # TCT: Check if this is a non-liquidity low (grabbed liquidity from left)
            is_non_liquidity = LiquidityDetector._is_non_liquidity_low(
                pivot, pivots.get("lows", []), candles
            )

            # TCT: Only use liquidity lows (not non-liquidity lows) for SSL pools
            if not is_non_liquidity:
                # TCT: Equal lows are "amazing liquidity targets"
                is_equal = any(abs(price - eq) / eq < 0.0001 for eq in equal_lows)

                ssl_pools.append({
                    "type": "SSL",
                    "price": float(price),
                    "idx": int(pivot["idx"]),
                    "is_primary": True,
                    "is_equal": bool(is_equal),
                    "strength": 1.0 if is_equal else 0.8,  # TCT: Equal = stronger
                    "distance_from_price": float((current_price - price) / current_price * 100)
                })

        # TCT: Add internal highs/lows as weaker liquidity (stacking up)
        # "Internal highs stacking up here" creating buy-side liquidity
        for pivot in internal_highs:
            price = pivot["price"]
            bsl_pools.append({
                "type": "BSL",
                "price": float(price),
                "idx": int(pivot["idx"]),
                "is_primary": False,
                "is_equal": False,
                "strength": 0.5,  # TCT: Internal = weaker than primary
                "distance_from_price": float((price - current_price) / current_price * 100)
            })

        # "Internal lows stacking up" creating sell-side liquidity
        for pivot in internal_lows:
            price = pivot["price"]
            ssl_pools.append({
                "type": "SSL",
                "price": float(price),
                "idx": int(pivot["idx"]),
                "is_primary": False,
                "is_equal": False,
                "strength": 0.5,  # TCT: Internal = weaker than primary
                "distance_from_price": float((current_price - price) / current_price * 100)
            })

        # Sort by strength
        bsl_pools.sort(key=lambda x: x["strength"], reverse=True)
        ssl_pools.sort(key=lambda x: x["strength"], reverse=True)

        return {
            "bsl_pools": bsl_pools,
            "ssl_pools": ssl_pools,
            "equal_highs": equal_highs,
            "equal_lows": equal_lows,
            "primary_highs_count": len(primary_highs),
            "primary_lows_count": len(primary_lows),
            "internal_highs_count": len(internal_highs),
            "internal_lows_count": len(internal_lows)
        }

    @staticmethod
    def _classify_primary_vs_internal(pivots: List[Dict], is_highs: bool = False, is_lows: bool = False) -> tuple:
        """
        Classify pivots as primary vs internal using TCT methodology.

        TCT Methodology:
        - Primary highs/lows = visible pivot points (major swings)
        - Internal highs/lows = high/low creations between primaries
        - "Visible pivot points are our primary highs"
        - "Internal highs are our high creations in between our primary highs"
        """
        if len(pivots) < 2:
            return pivots, []

        # Sort by index
        sorted_pivots = sorted(pivots, key=lambda x: x["idx"])

        # Calculate relative significance based on price range
        prices = [p["price"] for p in sorted_pivots]
        price_range = max(prices) - min(prices)

        if price_range == 0:
            return sorted_pivots, []

        primary = []
        internal = []

        # TCT: Window-based classification to identify visible pivots
        window_size = 5
        for i, pivot in enumerate(sorted_pivots):
            start = max(0, i - window_size)
            end = min(len(sorted_pivots), i + window_size + 1)
            window = sorted_pivots[start:end]
            window_prices = [p["price"] for p in window]

            if is_highs:
                # TCT: Primary highs are visible/prominent highs in the window
                is_primary = pivot["price"] >= sorted(window_prices)[-2] if len(window_prices) >= 2 else True
            elif is_lows:
                # TCT: Primary lows are visible/prominent lows in the window
                is_primary = pivot["price"] <= sorted(window_prices)[1] if len(window_prices) >= 2 else True
            else:
                is_primary = True

            if is_primary:
                primary.append(pivot)
            else:
                internal.append(pivot)

        return primary, internal

    @staticmethod
    def _detect_equal_levels(prices: List[float], tolerance: float = 0.0001) -> List[float]:
        """
        Detect equal highs/lows using TCT methodology.

        TCT: "Equal lows are amazing liquidity targets"
        Example: "Both lows length equals 1.07833" - exact same price
        "Indicated that it purposely went there to grab it to the exact amount"

        Returns list of price levels that have 2+ touches at exact same level.
        """
        if len(prices) < 2:
            return []

        equal_levels = []
        processed = set()

        for i, price1 in enumerate(prices):
            if i in processed:
                continue

            matches = [price1]
            for j, price2 in enumerate(prices[i+1:], start=i+1):
                # TCT: Equal means exact same price (very tight tolerance)
                if abs(price1 - price2) / price1 < tolerance:
                    matches.append(price2)
                    processed.add(j)

            # TCT: If 2+ matches at exact same level, it's an equal level (amazing liquidity target)
            if len(matches) >= 2:
                equal_levels.append(sum(matches) / len(matches))
                processed.add(i)

        return equal_levels

    @staticmethod
    def _is_non_liquidity_high(pivot: Dict, all_highs: List[Dict], candles: pd.DataFrame) -> bool:
        """
        Check if high is a non-liquidity high using TCT methodology.

        TCT: "If a high grabs liquidity from a previous high - that high is NOT a liquidity target"
        "Short sellers place their stop loss just above the previous high"
        "Often you will see a more aggressive expansion towards the downside from it"
        "That non liquidity high should not be apart of your plan for return to zone"

        Exception: "If liquidity gets swept than immediately starts creating price action
        near that high... more stop losses get set above that liquidity high, more liquidity
        gets stacked → becomes part of liquidity curve"

        Returns True if this is a non-liquidity high (should NOT be used as liquidity target)
        """
        idx = pivot["idx"]
        price = pivot["price"]

        if idx < 3 or idx >= len(candles) - 2:
            return False

        # TCT: Find previous highs to check if this grabbed liquidity from left
        previous_highs = [h for h in all_highs if h["idx"] < idx]
        if not previous_highs:
            return False

        # TCT: Check if this high grabbed liquidity from a previous high
        grabbed_liquidity_from_left = any(price > h["price"] for h in previous_highs[-3:])

        if not grabbed_liquidity_from_left:
            return False  # Didn't grab liquidity, so it's a valid liquidity high

        # TCT: Check for aggressive expansion downward after grabbing liquidity
        next_candles = candles.iloc[idx+1:min(idx+5, len(candles))]
        if len(next_candles) == 0:
            return False

        high_candle = candles.iloc[idx]
        lowest_after = next_candles["low"].min()
        expansion_size = high_candle["high"] - lowest_after
        high_size = high_candle["high"] - high_candle["low"]

        # TCT: "Often you will see a more aggressive expansion towards the downside"
        has_aggressive_expansion = expansion_size > high_size * 1.5

        if not has_aggressive_expansion:
            return False  # No aggressive expansion, treat as liquidity high

        # TCT EXCEPTION: "If price is spending time nearby that liquidity high"
        # "Price action near that liquidity high is going to create more short orders
        # and people tend to put their stop loss above that liquidity high"
        time_spent_near = sum(
            1 for c in next_candles.itertuples()
            if abs(c.close - price) / price < 0.01
        )

        # TCT: If spent time near level, it becomes part of liquidity curve
        if time_spent_near >= 2:
            return False  # Exception: IS liquidity because price spent time nearby

        return True  # It's a non-liquidity high (grabbed liquidity + aggressive expansion)

    @staticmethod
    def _is_non_liquidity_low(pivot: Dict, all_lows: List[Dict], candles: pd.DataFrame) -> bool:
        """
        Check if low is a non-liquidity low using TCT methodology.

        TCT: Same concept as non-liquidity high but inverted
        "The exact same thing happens from a liquidity low perspective"
        "We have a low right here and this low grabbed that liquidity from here"

        Returns True if this is a non-liquidity low (should NOT be used as liquidity target)
        """
        idx = pivot["idx"]
        price = pivot["price"]

        if idx < 3 or idx >= len(candles) - 2:
            return False

        # TCT: Find previous lows to check if this grabbed liquidity from left
        previous_lows = [l for l in all_lows if l["idx"] < idx]
        if not previous_lows:
            return False

        # TCT: Check if this low grabbed liquidity from a previous low
        grabbed_liquidity_from_left = any(price < l["price"] for l in previous_lows[-3:])

        if not grabbed_liquidity_from_left:
            return False  # Didn't grab liquidity, so it's a valid liquidity low

        # TCT: Check for aggressive expansion upward after grabbing liquidity
        next_candles = candles.iloc[idx+1:min(idx+5, len(candles))]
        if len(next_candles) == 0:
            return False

        low_candle = candles.iloc[idx]
        highest_after = next_candles["high"].max()
        expansion_size = highest_after - low_candle["low"]
        low_size = low_candle["high"] - low_candle["low"]

        # TCT: Aggressive expansion upward (inverted from non-liquidity high)
        has_aggressive_expansion = expansion_size > low_size * 1.5

        if not has_aggressive_expansion:
            return False  # No aggressive expansion, treat as liquidity low

        # TCT EXCEPTION: "Same goes for liquidity lows - only if we spend time above
        # that liquidity low can we start building a liquidity curve above it"
        time_spent_near = sum(
            1 for c in next_candles.itertuples()
            if abs(c.close - price) / price < 0.01
        )

        # TCT: If spent time near level, it becomes part of liquidity curve
        if time_spent_near >= 2:
            return False  # Exception: IS liquidity because price spent time nearby

        return True  # It's a non-liquidity low (grabbed liquidity + aggressive expansion)

# ================================================================
# LIQUIDITY CURVE GENERATION (TCT Methodology)
# ================================================================

class LiquidityCurveGenerator:
    """
    Generates liquidity curves by connecting primary highs or lows.

    Pure TCT Methodology from PDFs:
    - "Buy-side liquidity curve getting generated"
    - "Sell-side liquidity curve" with progressively lower primary highs
    - "Points all don't need to connect - just need a few points for proper liquidity curve"
    - "Typically you want each primary high to be a lower one to the previous one"
    - "Primary lows should form progressively higher lows" (for buy-side curve)
    - "Two lower highs" "Range high touches liquidity curve"
    - 3-tap pattern: 1st tap → 2nd tap → 3rd tap (Model 2 TCT)
    """

    @staticmethod
    def generate_curves(liquidity_pools: Dict, candles: pd.DataFrame) -> Dict:
        """
        Generate liquidity curves from liquidity pools using TCT methodology.

        TCT: "Buy-side liquidity curve getting generated"
        TCT: "Sell-side liquidity curve" from primary highs
        TCT: "Points all don't need to connect - just need a few points for proper liquidity curve"

        Returns: Dict with sell_side_curves and buy_side_curves
        """
        bsl_pools = liquidity_pools.get("bsl_pools", [])
        ssl_pools = liquidity_pools.get("ssl_pools", [])

        # TCT: Filter primary pools only (visible pivot points)
        # "Visible pivot points are our primary highs"
        primary_bsl = [p for p in bsl_pools if p.get("is_primary", False)]
        primary_ssl = [p for p in ssl_pools if p.get("is_primary", False)]

        # TCT: Generate sell-side curves from progressively lower primary highs
        sell_side_curves = LiquidityCurveGenerator._generate_sell_side_curves(primary_bsl, candles)

        # TCT: Generate buy-side curves from progressively higher primary lows
        buy_side_curves = LiquidityCurveGenerator._generate_buy_side_curves(primary_ssl, candles)

        return {
            "sell_side_curves": sell_side_curves,
            "buy_side_curves": buy_side_curves,
            "total_curves": len(sell_side_curves) + len(buy_side_curves)
        }

    @staticmethod
    def _generate_sell_side_curves(primary_highs: List[Dict], candles: pd.DataFrame) -> List[Dict]:
        """
        Generate sell-side liquidity curves from progressively lower primary highs.

        TCT Methodology:
        - "Typically you want each primary high to be a lower one to the previous one"
        - "Two lower highs" forming the curve
        - "Points all don't need to connect - just need a few points for proper liquidity curve"
        - 1st tap, 2nd tap, 3rd tap pattern (Model 2)
        """
        if len(primary_highs) < 3:
            return []

        # Sort by index (chronological order)
        sorted_highs = sorted(primary_highs, key=lambda x: x["idx"])

        curves = []

        # TCT: Look for sequences of 3+ progressively lower highs
        for i in range(len(sorted_highs) - 2):
            sequence = []
            current_high = sorted_highs[i]
            sequence.append(current_high)

            for j in range(i + 1, len(sorted_highs)):
                next_high = sorted_highs[j]

                # TCT: "Each primary high to be a lower one to the previous one"
                if next_high["price"] < sequence[-1]["price"]:
                    sequence.append(next_high)

                # TCT: Stop if we found enough points (don't need all points to connect)
                if len(sequence) >= 5:
                    break

            # TCT: Valid curve needs at least 3 points (1st, 2nd, 3rd tap)
            if len(sequence) >= 3:
                # Verify it's a valid descending sequence (lower highs)
                is_valid = all(
                    sequence[k]["price"] > sequence[k+1]["price"]
                    for k in range(len(sequence)-1)
                )

                if is_valid:
                    tap_count = len(sequence)

                    curves.append({
                        "type": "sell_side",
                        "taps": sequence,
                        "tap_count": tap_count,
                        "first_tap": sequence[0],  # TCT: 1st tap
                        "second_tap": sequence[1] if len(sequence) >= 2 else None,  # TCT: 2nd tap
                        "third_tap": sequence[2] if len(sequence) >= 3 else None,  # TCT: 3rd tap
                        "highest_price": float(sequence[0]["price"]),
                        "lowest_price": float(sequence[-1]["price"]),
                        "quality": "EXCELLENT" if tap_count >= 4 else "GOOD" if tap_count == 3 else "WEAK"
                    })

        return curves

    @staticmethod
    def _generate_buy_side_curves(primary_lows: List[Dict], candles: pd.DataFrame) -> List[Dict]:
        """
        Generate buy-side liquidity curves from progressively higher primary lows.

        TCT Methodology:
        - "Buy-side liquidity curve getting generated"
        - Primary lows should form progressively higher lows (accumulation model)
        - "Primary lows stacking up creating buy-side liquidity"
        - "Internal lows stacking up" below the curve
        - 1st tap (initial range low), 2nd tap, 3rd tap
        """
        if len(primary_lows) < 3:
            return []

        # Sort by index (chronological order)
        sorted_lows = sorted(primary_lows, key=lambda x: x["idx"])

        curves = []

        # TCT: Look for sequences of 3+ progressively higher lows
        for i in range(len(sorted_lows) - 2):
            sequence = []
            current_low = sorted_lows[i]
            sequence.append(current_low)

            for j in range(i + 1, len(sorted_lows)):
                next_low = sorted_lows[j]

                # TCT: Each primary low should be higher than the previous (higher lows)
                if next_low["price"] > sequence[-1]["price"]:
                    sequence.append(next_low)

                # TCT: Stop if we found enough points (don't need all points to connect)
                if len(sequence) >= 5:
                    break

            # TCT: Valid curve needs at least 3 points (1st, 2nd, 3rd tap)
            if len(sequence) >= 3:
                # Verify it's a valid ascending sequence (higher lows)
                is_valid = all(
                    sequence[k]["price"] < sequence[k+1]["price"]
                    for k in range(len(sequence)-1)
                )

                if is_valid:
                    tap_count = len(sequence)

                    curves.append({
                        "type": "buy_side",
                        "taps": sequence,
                        "tap_count": tap_count,
                        "first_tap": sequence[0],  # TCT: 1st tap (initial range low)
                        "second_tap": sequence[1] if len(sequence) >= 2 else None,  # TCT: 2nd tap
                        "third_tap": sequence[2] if len(sequence) >= 3 else None,  # TCT: 3rd tap
                        "lowest_price": float(sequence[0]["price"]),
                        "highest_price": float(sequence[-1]["price"]),
                        "quality": "EXCELLENT" if tap_count >= 4 else "GOOD" if tap_count == 3 else "WEAK"
                    })

        return curves

# ================================================================
# EXTREME LIQUIDITY TARGET IDENTIFICATION (TCT 2nd Tap Observations)
# ================================================================

class ExtremeLiquidityTarget:
    """
    Identifies extreme liquidity POI after 2nd tap using TCT methodology.

    Pure TCT Methodology from PDFs:
    - "2nd Tap Observations for Extreme Liquidity Point of Interest"
    - "Just understanding that your extreme liquidity is going to be that first higher low POI
      you would use this as your extreme liquidity target" (for accumulation)
    - For distribution: first lower high after 2nd tap
    - "If extreme liquidity gets grabbed in 5 minutes after 2nd tap, then focus on 2nd higher low"
    - "Equal lows are amazing liquidity targets" - use 2nd higher low if it's equal to next low
    - "If we used that 1st lower low after 2nd tap we would have been waiting for extreme
      liquidity target that was never revisited because the 2nd lower low was actually the
      extreme liquidity target"
    """

    @staticmethod
    def identify_extreme_targets(curves: Dict, candles: pd.DataFrame) -> Dict:
        """
        Identify extreme liquidity targets for each curve using TCT methodology.

        TCT: "2nd Tap Observations for Extreme Liquidity Point of Interest"
        - Must have at least 2 taps to identify extreme liquidity target
        - For accumulation (buy-side): first higher low after 2nd tap
        - For distribution (sell-side): first lower high after 2nd tap

        Returns: Dict with targets for each curve type
        """
        sell_side_targets = []
        buy_side_targets = []

        # TCT: Process sell-side curves (distribution model)
        for curve in curves.get("sell_side_curves", []):
            if curve.get("tap_count", 0) >= 2:
                target = ExtremeLiquidityTarget._find_extreme_target_for_distribution(
                    curve, candles
                )
                if target:
                    sell_side_targets.append(target)

        # TCT: Process buy-side curves (accumulation model)
        for curve in curves.get("buy_side_curves", []):
            if curve.get("tap_count", 0) >= 2:
                target = ExtremeLiquidityTarget._find_extreme_target_for_accumulation(
                    curve, candles
                )
                if target:
                    buy_side_targets.append(target)

        return {
            "sell_side_targets": sell_side_targets,
            "buy_side_targets": buy_side_targets,
            "total_targets": len(sell_side_targets) + len(buy_side_targets)
        }

    @staticmethod
    def _find_extreme_target_for_distribution(curve: Dict, candles: pd.DataFrame) -> Optional[Dict]:
        """
        Find first lower high after 2nd tap for distribution/sell-side using TCT methodology.

        TCT: "For distribution (sell-side): first lower high after 2nd tap"
        This is the inverted version of the accumulation model shown in the PDFs.
        """
        second_tap = curve.get("second_tap")
        if not second_tap:
            return None

        second_tap_idx = second_tap["idx"]
        second_tap_price = second_tap["price"]

        # TCT: Look for first lower high after 2nd tap
        # Search window should be sufficient to find the extreme target
        search_window = candles.iloc[second_tap_idx+1:min(second_tap_idx+20, len(candles))]

        for i, candle in enumerate(search_window.itertuples(), start=second_tap_idx+1):
            # TCT: Check if this forms a lower high (below 2nd tap price)
            if candle.high < second_tap_price:
                # TCT: Found first lower high - this is the extreme liquidity POI
                return {
                    "type": "distribution",
                    "curve_type": "sell_side",
                    "target_price": float(candle.high),
                    "target_idx": i,
                    "second_tap_price": float(second_tap_price),
                    "is_swept": False,  # Will be updated when checking current price
                    "quality": "EXTREME_LIQUIDITY_POI"
                }

        return None

    @staticmethod
    def _find_extreme_target_for_accumulation(curve: Dict, candles: pd.DataFrame) -> Optional[Dict]:
        """
        Find first higher low after 2nd tap for accumulation/buy-side using TCT methodology.

        TCT: "Just understanding that your extreme liquidity is going to be that first higher
        low POI you would use this as your extreme liquidity target"

        TCT: "1st Low after the 2nd tap is going to be our extreme liquidity"

        TCT: Important note - if this gets swept quickly (e.g., "5 minutes after 2nd tap"),
        then should focus on 2nd higher low instead. But this function returns the first
        higher low, and the endpoint logic will handle whether it was swept.
        """
        second_tap = curve.get("second_tap")
        if not second_tap:
            return None

        second_tap_idx = second_tap["idx"]
        second_tap_price = second_tap["price"]

        # TCT: "Look for first higher low after 2nd tap"
        # "We have a high, a low, higher high, and higher low in that first expansion after the 2nd tap"
        search_window = candles.iloc[second_tap_idx+1:min(second_tap_idx+20, len(candles))]

        for i, candle in enumerate(search_window.itertuples(), start=second_tap_idx+1):
            # TCT: Check if this forms a higher low (above 2nd tap price)
            if candle.low > second_tap_price:
                # TCT: Found first higher low - this is the extreme liquidity POI
                # "You would use this as your extreme liquidity target"
                return {
                    "type": "accumulation",
                    "curve_type": "buy_side",
                    "target_price": float(candle.low),
                    "target_idx": i,
                    "second_tap_price": float(second_tap_price),
                    "is_swept": False,  # Will be updated when checking current price
                    "quality": "EXTREME_LIQUIDITY_POI"
                }

        return None

# ================================================================
# LIQUIDITY VOID DETECTION (TCT Methodology)
# ================================================================

class LiquidityVoidDetector:
    """
    Detects liquidity voids using TCT methodology.

    Pure TCT from PDFs:
    - "There is no supply because of this" - areas without supply/demand
    - "This is so important to understand for high probablity trades because there is
      nothing in here that can hold price down" (no order blocks/zones to block movement)
    - Voids = areas with no S&D zones where price can move freely/aggressively
    - "If price returns to our deviation here than the move down will be quick because
      all the liquidity below will cause the sell orders to get triggered and price to
      move towards the downside"
    """

    @staticmethod
    def detect_voids(zones: List[Dict], candles: pd.DataFrame, current_price: float) -> Dict:
        """
        Detect liquidity voids between S&D zones using TCT methodology.

        TCT: "There is nothing in here that can hold price down/up"
        Areas without order blocks or structure zones = voids where price moves freely

        Returns: Dict with void regions
        """
        if not zones or len(candles) < 10:
            return {"voids": [], "total_voids": 0}

        # Get price range from candles
        price_high = float(candles["high"].max())
        price_low = float(candles["low"].min())
        price_range = price_high - price_low

        # Divide price range into segments
        segment_count = 20
        segment_size = price_range / segment_count

        # Create segments
        segments = []
        for i in range(segment_count):
            segment_low = price_low + (i * segment_size)
            segment_high = segment_low + segment_size
            segments.append({
                "low": segment_low,
                "high": segment_high,
                "mid": (segment_low + segment_high) / 2,
                "has_zone": False
            })

        # Mark segments that have zones
        for zone in zones:
            zone_top = zone.get("top", zone.get("refined_top", 0))
            zone_bottom = zone.get("bottom", zone.get("refined_bottom", 0))
            zone_mid = (zone_top + zone_bottom) / 2

            for segment in segments:
                # Check if zone overlaps this segment
                if (zone_bottom <= segment["high"] and zone_top >= segment["low"]):
                    segment["has_zone"] = True

        # Find consecutive segments without zones (voids)
        voids = []
        current_void = None

        for segment in segments:
            if not segment["has_zone"]:
                if current_void is None:
                    # Start new void
                    current_void = {
                        "low": segment["low"],
                        "high": segment["high"],
                        "segment_count": 1
                    }
                else:
                    # Extend current void
                    current_void["high"] = segment["high"]
                    current_void["segment_count"] += 1
            else:
                if current_void is not None:
                    # End current void if it's significant (2+ segments)
                    if current_void["segment_count"] >= 2:
                        void_range = current_void["high"] - current_void["low"]
                        void_mid = (current_void["high"] + current_void["low"]) / 2

                        voids.append({
                            "low": float(current_void["low"]),
                            "high": float(current_void["high"]),
                            "mid": float(void_mid),
                            "size": float(void_range),
                            "size_percent": float(void_range / price_range * 100),
                            "distance_from_price": float((void_mid - current_price) / current_price * 100),
                            "quality": "LARGE" if current_void["segment_count"] >= 4 else "MEDIUM"
                        })

                    current_void = None

        # Check final void
        if current_void is not None and current_void["segment_count"] >= 2:
            void_range = current_void["high"] - current_void["low"]
            void_mid = (current_void["high"] + current_void["low"]) / 2

            voids.append({
                "low": float(current_void["low"]),
                "high": float(current_void["high"]),
                "mid": float(void_mid),
                "size": float(void_range),
                "size_percent": float(void_range / price_range * 100),
                "distance_from_price": float((void_mid - current_price) / current_price * 100),
                "quality": "LARGE" if current_void["segment_count"] >= 4 else "MEDIUM"
            })

        return {
            "voids": voids,
            "total_voids": len(voids),
            "largest_void": max(voids, key=lambda v: v["size"]) if voids else None
        }

# ================================================================
# GATES
# ================================================================

def validate_1A(context: Dict) -> Dict:
    try:
        htf = context.get("htf_candles")
        if htf is None or len(htf) < 50:
            return {"passed": False, "bias": "neutral", "confidence": 0.0}

        ms = MarketStructure()
        pivots = ms.detect_pivots(htf)
        bos = ms.detect_bos(htf, pivots)

        conf = 0.5 + (0.3 if bos else 0.0)
        return {
            "passed": conf > 0.5,
            "bias": pivots["trend"],
            "confidence": min(conf, 1.0),
        }
    except Exception as e:
        return {"passed": False, "bias": "neutral", "confidence": 0.0, "error": str(e)}

def validate_1B(context: Dict) -> Dict:
    try:
        ltf = context.get("ltf_candles")
        htf_bias = context.get("1A", {}).get("bias", "neutral")

        if ltf is None or len(ltf) < 20:
            return {"passed": False, "confidence": 0.0}

        ms = MarketStructure()
        pivots = ms.detect_pivots(ltf)
        bos = ms.detect_bos(ltf, pivots)

        align = pivots["trend"] in (htf_bias, "ranging")
        conf = 0.6 if align else 0.2
        if bos and bos["type"] == htf_bias:
            conf += 0.3

        return {"passed": conf > 0.5, "confidence": min(conf, 1.0)}
    except Exception as e:
        return {"passed": False, "confidence": 0.0, "error": str(e)}

def validate_1C(context: Dict) -> Dict:
    try:
        c = context.get("1A", {}).get("confidence", 0) * 0.6
        c += context.get("1B", {}).get("confidence", 0) * 0.4
        return {"passed": c >= 0.65, "confidence": min(c, 1.0)}
    except Exception as e:
        return {"passed": False, "confidence": 0.0, "error": str(e)}

def validate_RCM(context: Dict) -> Dict:
    try:
        r = context.get("detected_range")
        if not r or r["duration_hours"] < 24:
            return {"valid": False, "confidence": 0.0}

        conf = min(r["duration_hours"] / 34, 1.0)
        return {"valid": conf > 0.6, "confidence": conf}
    except Exception as e:
        return {"valid": False, "confidence": 0.0, "error": str(e)}

def validate_MSCE(context: Dict) -> Dict:
    utc = datetime.utcnow().hour
    if utc < 8:
        return {"confidence": 0.95}
    elif utc < 16:
        return {"confidence": 1.05}
    return {"confidence": 1.15}

def validate_RIG(context: Dict) -> Dict:
    rcm = context.get("RCM", {})
    if rcm.get("valid"):
        return {"passed": True}
    return {"passed": False}

def validate_1D(context: Dict) -> Dict:
    try:
        if not all([
            context["1A"]["passed"],
            context["1B"]["passed"],
            context["1C"]["passed"],
            context["RCM"]["valid"],
            context["RIG"]["passed"]
        ]):
            return {"passed": False, "ExecutionConfidence_Total": 0.0}

        total = (
            context["1A"]["confidence"] * 0.25 +
            context["1B"]["confidence"] * 0.20 +
            context["1C"]["confidence"] * 0.25 +
            context["RCM"]["confidence"] * 0.20 +
            context["MSCE"]["confidence"] * 0.10
        )
        return {"passed": total >= 0.7, "ExecutionConfidence_Total": round(total * 100, 2)}
    except Exception:
        return {"passed": False, "ExecutionConfidence_Total": 0.0}

def validate_gates(context: Dict) -> Dict:
    context["1A"] = validate_1A(context)
    context["1B"] = validate_1B(context)
    context["1C"] = validate_1C(context)
    context["RCM"] = validate_RCM(context)
    context["MSCE"] = validate_MSCE(context)
    context["RIG"] = validate_RIG(context)
    context["1D"] = validate_1D(context)

    context["Action"] = "EXECUTE" if context["1D"]["passed"] else "NO_TRADE"
    return context

# ================================================================
# DATA FETCHING
# ================================================================

async def fetch_mexc_candles(symbol: str, interval: str, limit: int = 200) -> Optional[pd.DataFrame]:
    url = f"{MEXC_URL_BASE}/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.get(url, params=params)
            if r.status_code != 200:
                return None

            df = pd.DataFrame(r.json(), columns=[
                "open_time", "open", "high", "low", "close", "volume",
                "close_time", "quote_vol"
            ])
            df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
            for c in ["open", "high", "low", "close", "volume"]:
                df[c] = df[c].astype(float)
            return df
    except Exception as e:
        logger.error(f"[MEXC_FETCH_ERROR] {e}")
        return None

async def detect_best_range(candles: List) -> Optional[Dict]:
    """Simple range detection from candles"""
    if not candles or len(candles) < 10:
        return None

    highs = [c.get('high', c.get('h', 0)) for c in candles]
    lows = [c.get('low', c.get('l', 0)) for c in candles]

    return {
        "high": max(highs),
        "low": min(lows),
        "duration_hours": len(candles) * 0.25,
        "candles": candles
    }

# ================================================================
# ENDPOINTS
# ================================================================

@app.get("/")
async def root():
    return {
        "service": "HPB–TCT v21.2 (MEXC + Gate Validation + Full TCT)",
        "status": "running",
        "symbol": SYMBOL,
        "version": "21.2",
        "endpoints": {
            "/dashboard": "Interactive TCT Dashboard with chart and all metrics",
            "/status": "Health check",
            "/api/validate": "7-gate validation",
            "/api/price": "Current price",
            "/api/candles": "OHLC candle data (for charting)",
            "/api/ranges": "Range detection & deviations (TCT Mentorship Lecture 2)",
            "/api/zones": "Supply & Demand zones (TCT Mentorship Lecture 3)",
            "/api/liquidity": "Liquidity pools, curves & targets (TCT Lecture 4)"
        },
        "tct_lectures": {
            "lecture_1": "Market Structure (MarketStructure class)",
            "lecture_2": "Ranges (TCTRangeDetector, TCTDeviationDetector, TCTPremiumDiscountClassifier)",
            "lecture_3": "Supply & Demand (StructureSupplyDemand, ZoneScoring, ZoneRefinement)",
            "lecture_4": "Liquidity (LiquidityDetector, LiquidityCurveGenerator, ExtremeLiquidityTarget)"
        }
    }

@app.get("/status")
async def get_status():
    return {"status": "OK", "symbol": SYMBOL, "timestamp": datetime.utcnow().isoformat()}

@app.get("/api/price")
async def live_price():
    url = f"{MEXC_URL_BASE}/api/v3/ticker/price?symbol={SYMBOL}"
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.get(url)
            if r.status_code == 200:
                return {"symbol": SYMBOL, "price": float(r.json()["price"])}
            return {"error": f"HTTP {r.status_code}"}
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/candles")
async def get_candles(interval: str = "4h", limit: int = 100):
    """
    Fetch candles from MEXC - server-side to avoid CORS issues.

    Args:
        interval: Timeframe (1m, 5m, 15m, 1h, 4h, 1d)
        limit: Number of candles (max 1000)

    Returns:
        List of candles with time, open, high, low, close
    """
    try:
        df = await fetch_mexc_candles(SYMBOL, interval, min(limit, 500))
        if df is None:
            return JSONResponse({"error": "Failed to fetch candles from MEXC"}, status_code=500)

        candles = []
        for _, row in df.iterrows():
            candles.append({
                "time": int(row["open_time"].timestamp()),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"])
            })

        return {
            "symbol": SYMBOL,
            "interval": interval,
            "count": len(candles),
            "candles": candles
        }
    except Exception as e:
        logger.error(f"[CANDLES_ERROR] {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    """
    Interactive TCT Dashboard with candlestick chart and all TCT metrics.
    Displays: Market Structure, Ranges, Supply/Demand Zones, Liquidity, Deviations.
    """
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>HPB-TCT Dashboard</title>
    <script src="https://unpkg.com/lightweight-charts@4.1.0/dist/lightweight-charts.standalone.production.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #0a0a0f;
            color: #e0e0e0;
            min-height: 100vh;
        }
        .header {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            padding: 15px 25px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid #2d2d44;
        }
        .header h1 {
            font-size: 1.5rem;
            color: #00d4ff;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .header .price-display {
            font-size: 1.8rem;
            font-weight: bold;
            color: #00ff88;
        }
        .header .symbol { color: #888; font-size: 0.9rem; }
        .main-container {
            display: grid;
            grid-template-columns: 1fr 320px;
            gap: 15px;
            padding: 15px;
            height: calc(100vh - 70px);
        }
        .chart-section {
            background: #12121a;
            border-radius: 10px;
            overflow: hidden;
            border: 1px solid #2d2d44;
        }
        #chart { width: 100%; height: 100%; }
        .metrics-panel {
            display: flex;
            flex-direction: column;
            gap: 10px;
            overflow-y: auto;
            padding-right: 5px;
        }
        .metric-card {
            background: #12121a;
            border-radius: 8px;
            padding: 12px;
            border: 1px solid #2d2d44;
        }
        .metric-card h3 {
            font-size: 0.85rem;
            color: #00d4ff;
            margin-bottom: 10px;
            padding-bottom: 6px;
            border-bottom: 1px solid #2d2d44;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .metric-card h3 .badge {
            font-size: 0.7rem;
            padding: 2px 6px;
            border-radius: 4px;
            font-weight: normal;
        }
        .badge-bullish { background: rgba(0, 255, 136, 0.2); color: #00ff88; }
        .badge-bearish { background: rgba(255, 68, 68, 0.2); color: #ff4444; }
        .badge-neutral { background: rgba(255, 193, 7, 0.2); color: #ffc107; }
        .badge-premium { background: rgba(255, 68, 68, 0.2); color: #ff6b6b; }
        .badge-discount { background: rgba(0, 255, 136, 0.2); color: #00ff88; }
        .metric-row {
            display: flex;
            justify-content: space-between;
            padding: 4px 0;
            font-size: 0.8rem;
        }
        .metric-row .label { color: #888; }
        .metric-row .value { color: #e0e0e0; font-weight: 500; }
        .metric-row .value.bullish { color: #00ff88; }
        .metric-row .value.bearish { color: #ff4444; }
        .metric-row .value.warning { color: #ffc107; }
        .zone-list { margin-top: 8px; }
        .zone-item {
            background: #1a1a2e;
            border-radius: 4px;
            padding: 6px 8px;
            margin-bottom: 4px;
            font-size: 0.75rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .zone-item.demand { border-left: 3px solid #00ff88; }
        .zone-item.supply { border-left: 3px solid #ff4444; }
        .zone-item.bsl { border-left: 3px solid #ff6b6b; }
        .zone-item.ssl { border-left: 3px solid #4ecdc4; }
        .range-viz {
            background: #1a1a2e;
            border-radius: 6px;
            padding: 10px;
            margin-top: 8px;
        }
        .range-bar {
            height: 30px;
            background: linear-gradient(to bottom, rgba(255,68,68,0.3) 0%, rgba(255,68,68,0.3) 50%, rgba(0,255,136,0.3) 50%, rgba(0,255,136,0.3) 100%);
            border-radius: 4px;
            position: relative;
            margin: 10px 0;
        }
        .range-bar .eq-line {
            position: absolute;
            top: 50%;
            left: 0;
            right: 0;
            height: 2px;
            background: #ffc107;
        }
        .range-bar .price-marker {
            position: absolute;
            width: 8px;
            height: 8px;
            background: #00d4ff;
            border-radius: 50%;
            transform: translate(-50%, -50%);
            box-shadow: 0 0 10px #00d4ff;
        }
        .range-labels {
            display: flex;
            justify-content: space-between;
            font-size: 0.7rem;
            color: #888;
        }
        .validation-gates {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 4px;
            margin-top: 8px;
        }
        .gate {
            text-align: center;
            padding: 6px 4px;
            border-radius: 4px;
            font-size: 0.65rem;
        }
        .gate.pass { background: rgba(0, 255, 136, 0.2); color: #00ff88; }
        .gate.fail { background: rgba(255, 68, 68, 0.2); color: #ff4444; }
        .gate.pending { background: rgba(136, 136, 136, 0.2); color: #888; }
        .loading {
            display: flex;
            align-items: center;
            justify-content: center;
            height: 100%;
            color: #888;
        }
        .loading::after {
            content: '';
            width: 20px;
            height: 20px;
            border: 2px solid #2d2d44;
            border-top-color: #00d4ff;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin-left: 10px;
        }
        @keyframes spin { to { transform: rotate(360deg); } }
        .deviation-indicator {
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            margin-right: 4px;
        }
        .deviation-indicator.wick { background: #ffc107; }
        .deviation-indicator.candle { background: #ff6b6b; }
        .deviation-indicator.dl { background: #9b59b6; }
        .tct-lecture {
            font-size: 0.65rem;
            color: #666;
            font-style: italic;
            margin-bottom: 8px;
        }
        .schematic-item {
            background: #1a1a2e;
            border-radius: 6px;
            padding: 10px;
            margin-bottom: 8px;
            border-left: 3px solid #00d4ff;
        }
        .schematic-item.accumulation { border-left-color: #00ff88; }
        .schematic-item.distribution { border-left-color: #ff4444; }
        .schematic-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 6px;
        }
        .schematic-type {
            font-size: 0.75rem;
            font-weight: bold;
            color: #e0e0e0;
        }
        .schematic-quality {
            font-size: 0.65rem;
            padding: 2px 6px;
            border-radius: 3px;
            background: rgba(0, 212, 255, 0.2);
            color: #00d4ff;
        }
        .schematic-levels {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 6px;
            margin-top: 6px;
        }
        .level-box {
            text-align: center;
            padding: 4px;
            border-radius: 4px;
            font-size: 0.7rem;
        }
        .level-box.entry { background: rgba(0, 212, 255, 0.15); color: #00d4ff; }
        .level-box.stop { background: rgba(255, 68, 68, 0.15); color: #ff4444; }
        .level-box.target { background: rgba(0, 255, 136, 0.15); color: #00ff88; }
        .level-label { font-size: 0.6rem; color: #888; display: block; }
        .level-price { font-weight: bold; }
        .schematic-meta {
            display: flex;
            gap: 8px;
            margin-top: 6px;
            font-size: 0.65rem;
            color: #888;
        }
        .schematic-meta .rr { color: #ffc107; }
        .schematic-meta .safe { color: #00ff88; }
        .schematic-meta .unsafe { color: #ff4444;
        }
        .refresh-btn {
            background: #00d4ff;
            color: #0a0a0f;
            border: none;
            padding: 6px 12px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 0.8rem;
            font-weight: 600;
        }
        .refresh-btn:hover { background: #00b8e6; }
        .timeframe-selector {
            display: flex;
            gap: 5px;
        }
        .tf-btn {
            background: #1a1a2e;
            color: #888;
            border: 1px solid #2d2d44;
            padding: 4px 10px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 0.75rem;
        }
        .tf-btn.active { background: #00d4ff; color: #0a0a0f; border-color: #00d4ff; }
    </style>
</head>
<body>
    <div class="header">
        <h1>HPB-TCT Dashboard <span class="symbol">BTCUSDT</span></h1>
        <div style="display: flex; align-items: center; gap: 15px;">
            <div class="timeframe-selector">
                <button class="tf-btn" data-tf="15m">15m</button>
                <button class="tf-btn active" data-tf="4h">4H</button>
            </div>
            <div class="price-display" id="currentPrice">--</div>
            <button class="refresh-btn" onclick="refreshData()">Refresh</button>
        </div>
    </div>

    <div class="main-container">
        <div class="chart-section">
            <div id="chart"></div>
        </div>

        <div class="metrics-panel">
            <!-- Market Structure -->
            <div class="metric-card">
                <h3>Market Structure <span class="badge badge-neutral" id="trendBadge">--</span></h3>
                <div class="tct-lecture">TCT Lecture 1</div>
                <div class="metric-row">
                    <span class="label">HTF Trend (4H)</span>
                    <span class="value" id="htfTrend">--</span>
                </div>
                <div class="metric-row">
                    <span class="label">LTF Trend (15m)</span>
                    <span class="value" id="ltfTrend">--</span>
                </div>
                <div class="metric-row">
                    <span class="label">HTF Pivots</span>
                    <span class="value" id="htfPivots">--</span>
                </div>
            </div>

            <!-- Active Range -->
            <div class="metric-card">
                <h3>Active Range <span class="badge" id="zoneBadge">--</span></h3>
                <div class="tct-lecture">TCT Lecture 2 - Ranges</div>
                <div class="metric-row">
                    <span class="label">Range High</span>
                    <span class="value" id="rangeHigh">--</span>
                </div>
                <div class="metric-row">
                    <span class="label">Equilibrium (0.5)</span>
                    <span class="value warning" id="rangeEq">--</span>
                </div>
                <div class="metric-row">
                    <span class="label">Range Low</span>
                    <span class="value" id="rangeLow">--</span>
                </div>
                <div class="range-viz" id="rangeViz" style="display:none;">
                    <div class="range-labels">
                        <span>Premium</span>
                        <span>Discount</span>
                    </div>
                    <div class="range-bar">
                        <div class="eq-line"></div>
                        <div class="price-marker" id="priceMarker"></div>
                    </div>
                    <div class="range-labels">
                        <span id="rangeHighLabel">--</span>
                        <span id="rangeLowLabel">--</span>
                    </div>
                </div>
                <div class="metric-row">
                    <span class="label">Trading Bias</span>
                    <span class="value" id="tradingBias">--</span>
                </div>
                <div class="metric-row">
                    <span class="label">Bias Strength</span>
                    <span class="value" id="biasStrength">--</span>
                </div>
            </div>

            <!-- Deviations -->
            <div class="metric-card">
                <h3>Deviations <span class="badge badge-neutral" id="devBadge">0</span></h3>
                <div class="tct-lecture">TCT Lecture 2 - DL: 30%</div>
                <div class="metric-row">
                    <span class="label"><span class="deviation-indicator wick"></span>Wick Devs</span>
                    <span class="value" id="wickDevs">0</span>
                </div>
                <div class="metric-row">
                    <span class="label"><span class="deviation-indicator candle"></span>Bad BOS</span>
                    <span class="value" id="candleDevs">0</span>
                </div>
                <div class="metric-row">
                    <span class="label"><span class="deviation-indicator dl"></span>DL Devs</span>
                    <span class="value" id="dlDevs">0</span>
                </div>
            </div>

            <!-- Supply & Demand Zones -->
            <div class="metric-card">
                <h3>S&D Zones <span class="badge badge-neutral" id="zoneCount">0</span></h3>
                <div class="tct-lecture">TCT Lecture 3</div>
                <div class="metric-row">
                    <span class="label">HTF Zones</span>
                    <span class="value" id="htfZones">--</span>
                </div>
                <div class="metric-row">
                    <span class="label">High Quality</span>
                    <span class="value bullish" id="hqZones">--</span>
                </div>
                <div class="zone-list" id="topZones"></div>
            </div>

            <!-- Liquidity -->
            <div class="metric-card">
                <h3>Liquidity Pools <span class="badge badge-neutral" id="liqCount">0</span></h3>
                <div class="tct-lecture">TCT Lecture 4</div>
                <div class="metric-row">
                    <span class="label">BSL Pools (Above)</span>
                    <span class="value bearish" id="bslCount">--</span>
                </div>
                <div class="metric-row">
                    <span class="label">SSL Pools (Below)</span>
                    <span class="value bullish" id="sslCount">--</span>
                </div>
                <div class="metric-row">
                    <span class="label">Equal Highs</span>
                    <span class="value" id="eqHighs">--</span>
                </div>
                <div class="metric-row">
                    <span class="label">Equal Lows</span>
                    <span class="value" id="eqLows">--</span>
                </div>
                <div class="zone-list" id="liqPools"></div>
            </div>

            <!-- 7-Gate Validation -->
            <div class="metric-card">
                <h3>7-Gate Validation <span class="badge" id="actionBadge">--</span></h3>
                <div class="validation-gates" id="gates">
                    <div class="gate pending">G1</div>
                    <div class="gate pending">G2</div>
                    <div class="gate pending">G3</div>
                    <div class="gate pending">G4</div>
                    <div class="gate pending">G5</div>
                    <div class="gate pending">G6</div>
                    <div class="gate pending">G7</div>
                    <div class="gate pending">--</div>
                </div>
                <div class="metric-row" style="margin-top: 8px;">
                    <span class="label">Recommendation</span>
                    <span class="value" id="recommendation">--</span>
                </div>
            </div>

            <!-- TCT Schematics (Lecture 5A + 5B) -->
            <div class="metric-card">
                <h3>TCT Schematics <span class="badge badge-neutral" id="schematicsBadge">--</span></h3>
                <div class="tct-lecture">Lecture 5A + 5B Methodology</div>
                <div id="schematicsContent">
                    <div class="metric-row">
                        <span class="label">Loading...</span>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        let chart, candleSeries, lineSeries = [];
        let additionalSeries = []; // For liquidity curves
        let currentTimeframe = '4h';
        let isLoading = false;
        let lastCandles = []; // Store candles for index-to-time mapping

        // Fetch with retry and timeout
        async function fetchWithRetry(url, options = {}, retries = 3, timeout = 20000) {
            for (let i = 0; i < retries; i++) {
                try {
                    const controller = new AbortController();
                    const timeoutId = setTimeout(() => controller.abort(), timeout);

                    const response = await fetch(url, {
                        ...options,
                        signal: controller.signal
                    });
                    clearTimeout(timeoutId);

                    if (!response.ok) {
                        throw new Error(`HTTP ${response.status}`);
                    }
                    return await response.json();
                } catch (e) {
                    console.warn(`Fetch attempt ${i + 1} failed for ${url}:`, e.message);
                    if (i === retries - 1) throw e;
                    // Exponential backoff: 1s, 2s, 4s
                    await new Promise(r => setTimeout(r, Math.pow(2, i) * 1000));
                }
            }
        }

        // Show loading state for a section
        function setLoading(sectionId, loading) {
            const badge = document.getElementById(sectionId);
            if (badge && loading) {
                badge.textContent = '...';
                badge.className = 'badge badge-neutral';
            }
        }

        // Show error state for a section
        function setError(sectionId, error = true) {
            const badge = document.getElementById(sectionId);
            if (badge && error) {
                badge.textContent = 'ERR';
                badge.className = 'badge badge-bearish';
            }
        }

        // Initialize chart
        function initChart() {
            const container = document.getElementById('chart');
            chart = LightweightCharts.createChart(container, {
                width: container.clientWidth,
                height: container.clientHeight,
                layout: {
                    background: { color: '#12121a' },
                    textColor: '#888',
                },
                grid: {
                    vertLines: { color: '#1e1e2d' },
                    horzLines: { color: '#1e1e2d' },
                },
                crosshair: {
                    mode: LightweightCharts.CrosshairMode.Normal,
                },
                rightPriceScale: {
                    borderColor: '#2d2d44',
                },
                timeScale: {
                    borderColor: '#2d2d44',
                    timeVisible: true,
                },
            });

            candleSeries = chart.addCandlestickSeries({
                upColor: '#00ff88',
                downColor: '#ff4444',
                borderUpColor: '#00ff88',
                borderDownColor: '#ff4444',
                wickUpColor: '#00ff88',
                wickDownColor: '#ff4444',
            });

            window.addEventListener('resize', () => {
                chart.applyOptions({
                    width: container.clientWidth,
                    height: container.clientHeight,
                });
            });
        }

        // Fetch candle data from server (avoids CORS issues)
        async function fetchCandles(interval = '4h', limit = 100) {
            try {
                const data = await fetchWithRetry(
                    `/api/candles?interval=${interval}&limit=${limit}`,
                    {}, 3, 20000
                );
                if (data.error) {
                    console.error('Candles API error:', data.error);
                    return [];
                }
                return data.candles || [];
            } catch (e) {
                console.error('Failed to fetch candles:', e);
                return [];
            }
        }

        // Add horizontal line to chart
        function addPriceLine(price, color, title, lineStyle = 0, lineWidth = 1) {
            return candleSeries.createPriceLine({
                price: price,
                color: color,
                lineWidth: lineWidth,
                lineStyle: lineStyle,
                axisLabelVisible: true,
                title: title,
            });
        }

        // Add a range band (shaded area between two prices)
        function addRangeBand(high, low, candles) {
            if (!candles || candles.length === 0) return;

            const startTime = candles[0].time;
            const endTime = candles[candles.length - 1].time;
            const eq = (high + low) / 2;

            // Premium zone (above EQ) - red tint
            const premiumSeries = chart.addAreaSeries({
                topColor: 'rgba(255, 68, 68, 0.15)',
                bottomColor: 'rgba(255, 68, 68, 0.05)',
                lineColor: 'rgba(255, 68, 68, 0.0)',
                lineWidth: 0,
                priceScaleId: 'right',
            });
            premiumSeries.setData([
                { time: startTime, value: high },
                { time: endTime, value: high }
            ]);
            additionalSeries.push(premiumSeries);

            // Discount zone (below EQ) - green tint
            const discountSeries = chart.addAreaSeries({
                topColor: 'rgba(0, 255, 136, 0.05)',
                bottomColor: 'rgba(0, 255, 136, 0.15)',
                lineColor: 'rgba(0, 255, 136, 0.0)',
                lineWidth: 0,
                priceScaleId: 'right',
            });
            discountSeries.setData([
                { time: startTime, value: low },
                { time: endTime, value: low }
            ]);
            additionalSeries.push(discountSeries);
        }

        // Add liquidity curve to chart
        function addLiquidityCurve(curve, candles, isSellSide = true) {
            if (!curve.taps || curve.taps.length < 2 || !candles || candles.length === 0) return;

            const color = isSellSide ? '#ff6b6b' : '#4ecdc4';
            const curveData = [];

            curve.taps.forEach(tap => {
                const idx = tap.idx;
                if (idx >= 0 && idx < candles.length) {
                    curveData.push({
                        time: candles[idx].time,
                        value: tap.price
                    });
                }
            });

            if (curveData.length >= 2) {
                // Sort by time
                curveData.sort((a, b) => a.time - b.time);

                const curveSeries = chart.addLineSeries({
                    color: color,
                    lineWidth: 2,
                    lineStyle: LightweightCharts.LineStyle.Dashed,
                    crosshairMarkerVisible: true,
                    priceScaleId: 'right',
                });
                curveSeries.setData(curveData);
                additionalSeries.push(curveSeries);

                // Add markers at tap points
                const markers = curveData.map((point, i) => ({
                    time: point.time,
                    position: isSellSide ? 'aboveBar' : 'belowBar',
                    color: color,
                    shape: 'circle',
                    text: `T${i + 1}`
                }));
                curveSeries.setMarkers(markers);
            }
        }

        // Clear all price lines and additional series
        function clearPriceLines() {
            lineSeries.forEach(line => {
                try { candleSeries.removePriceLine(line); } catch(e) {}
            });
            lineSeries = [];

            // Remove additional series (liquidity curves, range bands)
            additionalSeries.forEach(series => {
                try { chart.removeSeries(series); } catch(e) {}
            });
            additionalSeries = [];
        }

        // Fetch and display all TCT data
        async function refreshData() {
            if (isLoading) return;
            isLoading = true;

            // Show loading states
            setLoading('trendBadge', true);
            setLoading('zoneBadge', true);
            setLoading('zoneCount', true);
            setLoading('liqCount', true);
            setLoading('actionBadge', true);
            setLoading('schematicsBadge', true);

            // Fetch candles and update chart
            lastCandles = await fetchCandles(currentTimeframe, 100);
            if (lastCandles.length > 0) {
                candleSeries.setData(lastCandles);
                const lastPrice = lastCandles[lastCandles.length - 1].close;
                document.getElementById('currentPrice').textContent = '$' + lastPrice.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2});
            }

            clearPriceLines();

            // Fetch all API data in parallel with individual error handling
            const [rangesResult, zonesResult, liqResult, valResult, schematicsResult] = await Promise.allSettled([
                fetchWithRetry('/api/ranges', {}, 3, 25000),
                fetchWithRetry('/api/zones', {}, 3, 25000),
                fetchWithRetry('/api/liquidity', {}, 3, 25000),
                fetchWithRetry('/api/validate', {}, 3, 25000),
                fetchWithRetry('/api/schematics', {}, 3, 30000)
            ]);

            // Process ranges (pass candles for range band)
            if (rangesResult.status === 'fulfilled' && !rangesResult.value.error) {
                updateRangesUI(rangesResult.value, lastCandles);
            } else {
                console.error('Ranges error:', rangesResult.reason || rangesResult.value?.error);
                setError('trendBadge');
                setError('zoneBadge');
            }

            // Process zones
            if (zonesResult.status === 'fulfilled' && !zonesResult.value.error) {
                updateZonesUI(zonesResult.value);
            } else {
                console.error('Zones error:', zonesResult.reason || zonesResult.value?.error);
                setError('zoneCount');
            }

            // Process liquidity (pass candles for curve drawing)
            if (liqResult.status === 'fulfilled' && !liqResult.value.error) {
                updateLiquidityUI(liqResult.value, lastCandles);
            } else {
                console.error('Liquidity error:', liqResult.reason || liqResult.value?.error);
                setError('liqCount');
            }

            // Process validation
            if (valResult.status === 'fulfilled') {
                updateValidationUI(valResult.value);
            } else {
                console.error('Validation error:', valResult.reason);
                setError('actionBadge');
            }

            // Process schematics
            if (schematicsResult.status === 'fulfilled' && !schematicsResult.value.error) {
                updateSchematicsUI(schematicsResult.value);
            } else {
                console.error('Schematics error:', schematicsResult.reason || schematicsResult.value?.error);
                setError('schematicsBadge');
            }

            isLoading = false;
        }

        function updateRangesUI(data, candles = []) {
            if (data.error) return;

            // Market structure
            const htfTrend = data.market_structure?.htf_trend || 'neutral';
            const ltfTrend = data.market_structure?.ltf_trend || 'neutral';
            document.getElementById('htfTrend').textContent = htfTrend.toUpperCase();
            document.getElementById('htfTrend').className = 'value ' + (htfTrend === 'bullish' ? 'bullish' : htfTrend === 'bearish' ? 'bearish' : '');
            document.getElementById('ltfTrend').textContent = ltfTrend.toUpperCase();
            document.getElementById('ltfTrend').className = 'value ' + (ltfTrend === 'bullish' ? 'bullish' : ltfTrend === 'bearish' ? 'bearish' : '');

            const trendBadge = document.getElementById('trendBadge');
            trendBadge.textContent = htfTrend.toUpperCase();
            trendBadge.className = 'badge badge-' + (htfTrend === 'bullish' ? 'bullish' : htfTrend === 'bearish' ? 'bearish' : 'neutral');

            // Active range
            const activeRange = data.htf_ranges?.active_range || data.ltf_ranges?.active_range;
            if (activeRange) {
                const high = activeRange.range_high || activeRange.high;
                const low = activeRange.range_low || activeRange.low;
                const eq = activeRange.equilibrium || ((high + low) / 2);

                document.getElementById('rangeHigh').textContent = '$' + high?.toLocaleString(undefined, {maximumFractionDigits: 2});
                document.getElementById('rangeEq').textContent = '$' + eq?.toLocaleString(undefined, {maximumFractionDigits: 2});
                document.getElementById('rangeLow').textContent = '$' + low?.toLocaleString(undefined, {maximumFractionDigits: 2});

                // Add PROMINENT range lines to chart (thicker lines)
                if (high) lineSeries.push(addPriceLine(high, '#ff4444', 'RANGE HIGH', 0, 2));
                if (eq) lineSeries.push(addPriceLine(eq, '#ffc107', 'EQ (0.5)', 1, 2));
                if (low) lineSeries.push(addPriceLine(low, '#00ff88', 'RANGE LOW', 0, 2));

                // Add Premium/Discount shaded zones
                if (high && low && candles.length > 0) {
                    // Premium zone (high to eq) - subtle red
                    const premiumData = candles.map(c => ({ time: c.time, value: high }));
                    const premiumSeries = chart.addLineSeries({
                        color: 'rgba(255, 68, 68, 0.4)',
                        lineWidth: 1,
                        lineStyle: LightweightCharts.LineStyle.Dotted,
                        priceLineVisible: false,
                        lastValueVisible: false,
                    });
                    premiumSeries.setData(premiumData);
                    additionalSeries.push(premiumSeries);

                    // Discount zone (eq to low) - subtle green
                    const discountData = candles.map(c => ({ time: c.time, value: low }));
                    const discountSeries = chart.addLineSeries({
                        color: 'rgba(0, 255, 136, 0.4)',
                        lineWidth: 1,
                        lineStyle: LightweightCharts.LineStyle.Dotted,
                        priceLineVisible: false,
                        lastValueVisible: false,
                    });
                    discountSeries.setData(discountData);
                    additionalSeries.push(discountSeries);

                    // Equilibrium line (prominent yellow dashed)
                    const eqData = candles.map(c => ({ time: c.time, value: eq }));
                    const eqSeries = chart.addLineSeries({
                        color: '#ffc107',
                        lineWidth: 2,
                        lineStyle: LightweightCharts.LineStyle.Dashed,
                        priceLineVisible: false,
                        lastValueVisible: false,
                    });
                    eqSeries.setData(eqData);
                    additionalSeries.push(eqSeries);
                }

                // Range visualization in sidebar
                document.getElementById('rangeViz').style.display = 'block';
                document.getElementById('rangeHighLabel').textContent = '$' + high?.toLocaleString(undefined, {maximumFractionDigits: 0});
                document.getElementById('rangeLowLabel').textContent = '$' + low?.toLocaleString(undefined, {maximumFractionDigits: 0});

                // Position price marker
                const currentPrice = data.current_price;
                if (currentPrice && high && low) {
                    const pct = ((high - currentPrice) / (high - low)) * 100;
                    const marker = document.getElementById('priceMarker');
                    marker.style.top = Math.min(100, Math.max(0, pct)) + '%';
                    marker.style.left = '50%';
                }
            }

            // Current position
            const position = data.current_position || {};
            const zone = position.zone || 'UNKNOWN';
            const zoneBadge = document.getElementById('zoneBadge');
            zoneBadge.textContent = zone;
            zoneBadge.className = 'badge badge-' + (zone === 'PREMIUM' ? 'premium' : zone === 'DISCOUNT' ? 'discount' : 'neutral');

            document.getElementById('tradingBias').textContent = position.trading_bias || '--';
            document.getElementById('tradingBias').className = 'value ' + (position.trading_bias === 'SELL' ? 'bearish' : position.trading_bias === 'BUY' ? 'bullish' : '');
            document.getElementById('biasStrength').textContent = position.bias_strength ? (position.bias_strength * 100).toFixed(0) + '%' : '--';

            // Deviations
            const htfDevs = data.deviations?.htf_deviations || {};
            const ltfDevs = data.deviations?.ltf_deviations || {};
            const totalDevs = (htfDevs.total_deviations || 0) + (ltfDevs.total_deviations || 0);

            document.getElementById('devBadge').textContent = totalDevs;
            document.getElementById('wickDevs').textContent = (htfDevs.wick_deviations?.length || 0) + (ltfDevs.wick_deviations?.length || 0);
            document.getElementById('candleDevs').textContent = (htfDevs.candle_close_deviations?.length || 0) + (ltfDevs.candle_close_deviations?.length || 0);
            document.getElementById('dlDevs').textContent = (htfDevs.dl_deviations?.length || 0) + (ltfDevs.dl_deviations?.length || 0);

            document.getElementById('htfPivots').textContent =
                (data.htf_ranges?.total_ranges || 0) + ' ranges, ' +
                (data.htf_ranges?.confirmed_ranges || 0) + ' confirmed';
        }

        function updateZonesUI(data) {
            if (data.error) return;

            const htfTotal = data.htf_zones?.total_zones || 0;
            const hqCount = data.htf_zones?.high_quality_count || 0;

            document.getElementById('htfZones').textContent = htfTotal;
            document.getElementById('hqZones').textContent = hqCount;
            document.getElementById('zoneCount').textContent = htfTotal;

            // Display top zones
            const topZonesEl = document.getElementById('topZones');
            topZonesEl.innerHTML = '';

            const topZones = data.htf_zones?.top_3_high_quality || data.htf_zones?.top_3_all || [];
            topZones.slice(0, 3).forEach(zone => {
                const zoneType = zone.type || (zone.top > data.current_price ? 'supply' : 'demand');
                const div = document.createElement('div');
                div.className = 'zone-item ' + zoneType;
                div.innerHTML = `
                    <span>${zoneType.toUpperCase()}</span>
                    <span>$${zone.top?.toLocaleString(undefined, {maximumFractionDigits: 0})} - $${zone.bottom?.toLocaleString(undefined, {maximumFractionDigits: 0})}</span>
                `;
                topZonesEl.appendChild(div);

                // Add zone to chart
                if (zone.top && zone.bottom) {
                    const color = zoneType === 'demand' ? 'rgba(0, 255, 136, 0.3)' : 'rgba(255, 68, 68, 0.3)';
                    // Note: Lightweight Charts doesn't support rectangles natively, using price lines
                    lineSeries.push(addPriceLine(zone.top, zoneType === 'demand' ? '#00ff88' : '#ff4444', zoneType.charAt(0).toUpperCase(), 2));
                }
            });
        }

        function updateLiquidityUI(data, candles = []) {
            if (data.error) return;

            const bslPools = data.htf_liquidity?.bsl_pools || [];
            const sslPools = data.htf_liquidity?.ssl_pools || [];

            document.getElementById('bslCount').textContent = bslPools.length;
            document.getElementById('sslCount').textContent = sslPools.length;
            document.getElementById('eqHighs').textContent = data.htf_liquidity?.equal_highs?.length || 0;
            document.getElementById('eqLows').textContent = data.htf_liquidity?.equal_lows?.length || 0;
            document.getElementById('liqCount').textContent = bslPools.length + sslPools.length;

            // Display top liquidity pools
            const liqPoolsEl = document.getElementById('liqPools');
            liqPoolsEl.innerHTML = '';

            // Top 2 BSL pools
            bslPools.slice(0, 2).forEach(pool => {
                const div = document.createElement('div');
                div.className = 'zone-item bsl';
                div.innerHTML = `
                    <span>BSL ${pool.is_equal ? '(EQ)' : ''}</span>
                    <span>$${pool.price?.toLocaleString(undefined, {maximumFractionDigits: 0})}</span>
                `;
                liqPoolsEl.appendChild(div);

                // Add to chart with thicker line
                if (pool.price) {
                    lineSeries.push(addPriceLine(pool.price, '#ff6b6b', 'BSL', 1, 1));
                }
            });

            // Top 2 SSL pools
            sslPools.slice(0, 2).forEach(pool => {
                const div = document.createElement('div');
                div.className = 'zone-item ssl';
                div.innerHTML = `
                    <span>SSL ${pool.is_equal ? '(EQ)' : ''}</span>
                    <span>$${pool.price?.toLocaleString(undefined, {maximumFractionDigits: 0})}</span>
                `;
                liqPoolsEl.appendChild(div);

                // Add to chart with thicker line
                if (pool.price) {
                    lineSeries.push(addPriceLine(pool.price, '#4ecdc4', 'SSL', 1, 1));
                }
            });

            // === DRAW LIQUIDITY CURVES ON CHART ===
            const sellSideCurves = data.htf_curves?.sell_side_curves || [];
            const buySideCurves = data.htf_curves?.buy_side_curves || [];

            // Draw best quality sell-side curve (descending highs - bearish liquidity)
            if (sellSideCurves.length > 0 && candles.length > 0) {
                // Sort by quality/tap count to get best curve
                const bestSellCurve = sellSideCurves.sort((a, b) =>
                    (b.quality || b.tap_count || 0) - (a.quality || a.tap_count || 0)
                )[0];

                if (bestSellCurve && bestSellCurve.taps && bestSellCurve.taps.length >= 2) {
                    addLiquidityCurve(bestSellCurve, candles, true);
                    console.log('Drew sell-side liquidity curve with', bestSellCurve.taps.length, 'taps');
                }
            }

            // Draw best quality buy-side curve (ascending lows - bullish liquidity)
            if (buySideCurves.length > 0 && candles.length > 0) {
                // Sort by quality/tap count to get best curve
                const bestBuyCurve = buySideCurves.sort((a, b) =>
                    (b.quality || b.tap_count || 0) - (a.quality || a.tap_count || 0)
                )[0];

                if (bestBuyCurve && bestBuyCurve.taps && bestBuyCurve.taps.length >= 2) {
                    addLiquidityCurve(bestBuyCurve, candles, false);
                    console.log('Drew buy-side liquidity curve with', bestBuyCurve.taps.length, 'taps');
                }
            }
        }

        function updateValidationUI(data) {
            if (data.error) {
                document.getElementById('recommendation').textContent = data.Action || 'ERROR';
                return;
            }

            const gates = data.gates || {};
            const gatesEl = document.getElementById('gates');
            const gateNames = ['G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7'];
            const gateKeys = ['htf_trend', 'ltf_alignment', 'sd_zone', 'liquidity', 'fvg', 'entry_model', 'rr_ratio'];

            let passCount = 0;
            gatesEl.innerHTML = '';

            gateKeys.forEach((key, i) => {
                const gate = gates[key];
                const passed = gate?.passed || gate?.valid || false;
                if (passed) passCount++;

                const div = document.createElement('div');
                div.className = 'gate ' + (passed ? 'pass' : 'fail');
                div.textContent = gateNames[i];
                div.title = gate?.reason || key;
                gatesEl.appendChild(div);
            });

            // Summary gate
            const summaryDiv = document.createElement('div');
            summaryDiv.className = 'gate ' + (passCount >= 5 ? 'pass' : 'fail');
            summaryDiv.textContent = passCount + '/7';
            gatesEl.appendChild(summaryDiv);

            const action = data.Action || (passCount >= 5 ? 'VALID_SETUP' : 'NO_TRADE');
            const actionBadge = document.getElementById('actionBadge');
            actionBadge.textContent = action;
            actionBadge.className = 'badge badge-' + (action.includes('LONG') || action.includes('VALID') ? 'bullish' : action.includes('SHORT') ? 'bearish' : 'neutral');

            document.getElementById('recommendation').textContent = action;
            document.getElementById('recommendation').className = 'value ' + (action.includes('LONG') || action.includes('VALID') ? 'bullish' : action.includes('SHORT') ? 'bearish' : 'warning');
        }

        function updateSchematicsUI(data) {
            const contentEl = document.getElementById('schematicsContent');
            const badgeEl = document.getElementById('schematicsBadge');

            if (data.error) {
                contentEl.innerHTML = '<div class="metric-row"><span class="label">Error loading schematics</span></div>';
                badgeEl.textContent = 'ERR';
                badgeEl.className = 'badge badge-bearish';
                return;
            }

            // Combine HTF and LTF schematics
            const htfSchematics = data.htf_schematics?.schematics || [];
            const ltfSchematics = data.ltf_schematics?.schematics || [];
            const allSchematics = [...htfSchematics.slice(0, 2), ...ltfSchematics.slice(0, 2)];

            if (allSchematics.length === 0) {
                contentEl.innerHTML = '<div class="metric-row"><span class="label">No active schematics detected</span></div>';
                badgeEl.textContent = '0';
                badgeEl.className = 'badge badge-neutral';
                return;
            }

            // Update badge
            const totalCount = (data.htf_schematics?.summary?.total || 0) + (data.ltf_schematics?.summary?.total || 0);
            const hasAccum = allSchematics.some(s => s.direction === 'bullish');
            const hasDist = allSchematics.some(s => s.direction === 'bearish');
            badgeEl.textContent = totalCount;
            badgeEl.className = 'badge badge-' + (hasAccum && !hasDist ? 'bullish' : hasDist && !hasAccum ? 'bearish' : 'neutral');

            // Build schematic cards
            let html = '';
            allSchematics.forEach((s, i) => {
                const isAccum = s.direction === 'bullish';
                const typeClass = isAccum ? 'accumulation' : 'distribution';
                const typeLabel = s.schematic_type?.replace(/_/g, ' ').toUpperCase() || (isAccum ? 'ACCUMULATION' : 'DISTRIBUTION');
                const quality = Math.round((s.quality_score || 0) * 100);
                const entry = s.entry?.price;
                const stop = s.stop_loss?.price;
                const target = s.target?.price;
                const rr = s.risk_reward;
                const isSafe = s.entry?.is_safe !== false;
                const isConfirmed = s.is_confirmed;

                // Lecture 5B enhancements
                const enhancements = s.lecture_5b_enhancements || {};
                const meetsRR = enhancements.meets_minimum_rr;
                const has6CR = enhancements.htf_validation?.all_taps_valid_6cr;
                const hasTrendline = enhancements.has_trendline_confluence;

                // Lecture 6 enhancements
                const l6 = s.lecture_6_enhancements || {};
                const hasConversion = l6.has_conversion;
                const hasDualDev = l6.has_dual_deviation;
                const hasWOV = l6.has_wov_opportunity;
                const hasM1toM2 = l6.has_m1_to_m2_opportunity;
                const followBias = l6.follow_through_bias;
                const enhancedTarget = l6.enhanced_target;

                html += '<div class="schematic-item ' + typeClass + '">';
                html += '<div class="schematic-header">';
                html += '<span class="schematic-type">' + typeLabel + '</span>';
                html += '<span class="schematic-quality">' + quality + '%</span>';
                html += '</div>';

                if (entry && stop && target) {
                    html += '<div class="schematic-levels">';
                    html += '<div class="level-box entry"><span class="level-label">ENTRY</span><span class="level-price">$' + entry.toLocaleString(undefined, {maximumFractionDigits: 0}) + '</span></div>';
                    html += '<div class="level-box stop"><span class="level-label">STOP</span><span class="level-price">$' + stop.toLocaleString(undefined, {maximumFractionDigits: 0}) + '</span></div>';
                    html += '<div class="level-box target"><span class="level-label">TARGET</span><span class="level-price">$' + target.toLocaleString(undefined, {maximumFractionDigits: 0}) + '</span></div>';
                    html += '</div>';
                }

                html += '<div class="schematic-meta">';
                if (rr) html += '<span class="rr">R:R ' + rr.toFixed(1) + '</span>';
                html += '<span class="' + (isSafe ? 'safe' : 'unsafe') + '">' + (isSafe ? 'Safe Entry' : 'Caution: S/D Zone') + '</span>';
                if (isConfirmed) html += '<span class="safe">Confirmed</span>';
                if (has6CR) html += '<span class="safe">6CR Valid</span>';
                if (hasTrendline) html += '<span class="safe">TL Confluence</span>';
                html += '</div>';

                // Lecture 6 advanced indicators
                if (hasConversion || hasDualDev || hasWOV || hasM1toM2 || followBias) {
                    html += '<div class="schematic-meta l6-indicators" style="margin-top:4px;border-top:1px solid #333;padding-top:4px;">';
                    if (hasConversion) html += '<span style="color:#ff9800;">Converted</span>';
                    if (hasDualDev) html += '<span style="color:#e91e63;">Dual Dev</span>';
                    if (hasWOV) html += '<span style="color:#00bcd4;">WOV Entry</span>';
                    if (hasM1toM2) html += '<span style="color:#9c27b0;">M1→M2</span>';
                    if (followBias && followBias !== 'neutral') html += '<span style="color:#8bc34a;">' + followBias + '</span>';
                    if (enhancedTarget) html += '<span style="color:#ffc107;">Ext: $' + enhancedTarget.toLocaleString(undefined, {maximumFractionDigits: 0}) + '</span>';
                    html += '</div>';
                }
                html += '</div>';
            });

            contentEl.innerHTML = html;
        }

        // Timeframe selector
        document.querySelectorAll('.tf-btn').forEach(btn => {
            btn.addEventListener('click', async () => {
                document.querySelectorAll('.tf-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                currentTimeframe = btn.dataset.tf;
                await refreshData();
            });
        });

        // Initialize
        initChart();
        refreshData();

        // Auto-refresh every 30 seconds
        setInterval(refreshData, 30000);
    </script>
</body>
</html>
    """
    return HTMLResponse(content=html_content)

@app.get("/api/zones")
async def detect_zones():
    """
    Detect and score Supply/Demand zones using TCT Mentorship methodology.

    Returns:
        - Order Blocks (with FVG requirement)
        - Structure Supply/Demand zones
        - Location-based scoring (pivot, range discount/premium, deviation)
        - Top zones sorted by strength
    """
    try:
        # Fetch candles
        htf_df = await fetch_mexc_candles(SYMBOL, "4h", 100)
        ltf_df = await fetch_mexc_candles(SYMBOL, "15m", 200)

        if htf_df is None or ltf_df is None:
            return JSONResponse({"error": "Failed to fetch data"}, status_code=500)

        current_price = float(ltf_df.iloc[-1]["close"])

        # Detect FVGs
        fvg_detector = FairValueGap()
        htf_fvgs = fvg_detector.detect_fvgs(htf_df)
        ltf_fvgs = fvg_detector.detect_fvgs(ltf_df)

        # Detect Order Blocks
        ob_detector = OrderBlock()
        htf_obs = ob_detector.detect_order_blocks(htf_df, htf_fvgs)
        ltf_obs = ob_detector.detect_order_blocks(ltf_df, ltf_fvgs)

        # Detect market structure
        ms = MarketStructure()
        htf_pivots = ms.detect_pivots(htf_df)
        ltf_pivots = ms.detect_pivots(ltf_df)

        # Detect Structure Supply/Demand
        ssd_detector = StructureSupplyDemand()
        htf_structure = ssd_detector.detect_structure_zones(htf_df, htf_fvgs, htf_pivots)
        ltf_structure = ssd_detector.detect_structure_zones(ltf_df, ltf_fvgs, ltf_pivots)

        # Convert LTF to dict for range detection
        ltf_candles = []
        for _, row in ltf_df.iterrows():
            ltf_candles.append({
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close'])
            })

        detected_range = await detect_best_range(ltf_candles)

        # Combine zones
        htf_zones = {
            "bullish_obs": htf_obs.get("bullish_obs", []),
            "bearish_obs": htf_obs.get("bearish_obs", []),
            "demand_zones": htf_structure.get("demand_zones", []),
            "supply_zones": htf_structure.get("supply_zones", [])
        }

        ltf_zones = {
            "bullish_obs": ltf_obs.get("bullish_obs", []),
            "bearish_obs": ltf_obs.get("bearish_obs", []),
            "demand_zones": ltf_structure.get("demand_zones", []),
            "supply_zones": ltf_structure.get("supply_zones", [])
        }

        # Score zones (basic scoring)
        scorer = ZoneScoring()
        htf_scored = scorer.score_zones(htf_zones, htf_pivots, detected_range, current_price)
        ltf_scored = scorer.score_zones(ltf_zones, ltf_pivots, detected_range, current_price)

        # === ENHANCED ANALYSIS (38-Page PDF Methodology) ===

        # 1. Multi-Timeframe Validation - Validate HTF zones on LTF
        mtf_validator = MultiTimeframeValidator()
        for zone in htf_scored:
            mtf_result = mtf_validator.validate_htf_zone_on_ltf(zone, ltf_df)
            zone["mtf_validation"] = mtf_result
            # Boost strength if has true LTF inefficiency
            if mtf_result.get("has_true_inefficiency"):
                zone["strength"] = zone.get("strength", 0) * 1.2

        # 2. Zone Refinement - Detect partial mitigation and refine boundaries
        zone_refiner = ZoneRefinement()
        for zone in htf_scored:
            refined = zone_refiner.refine_zone_after_mitigation(zone, ltf_df)
            zone.update(refined)

        for zone in ltf_scored:
            refined = zone_refiner.refine_zone_after_mitigation(zone, ltf_df)
            zone.update(refined)

        # 3. Pivot Quality & Liquidity Sweep Scoring
        pivot_scorer = PivotQualityScorer()
        for zone in htf_scored:
            pivot_quality = pivot_scorer.score_pivot_quality(zone, htf_pivots, htf_df)
            zone["pivot_quality"] = pivot_quality
            # Boost strength based on pivot quality
            zone["strength"] = zone.get("strength", 0) * pivot_quality.get("pivot_quality_score", 1.0)

        for zone in ltf_scored:
            pivot_quality = pivot_scorer.score_pivot_quality(zone, ltf_pivots, ltf_df)
            zone["pivot_quality"] = pivot_quality
            zone["strength"] = zone.get("strength", 0) * pivot_quality.get("pivot_quality_score", 1.0)

        # 4. Overlapping Zone Detection - Find LTF zones that overlap HTF zones
        overlap_detector = OverlappingZoneDetector()

        # Separate demand and supply for overlap detection
        htf_demand = [z for z in htf_scored if z.get("type") == "demand"]
        htf_supply = [z for z in htf_scored if z.get("type") == "supply"]
        ltf_demand = [z for z in ltf_scored if z.get("type") == "demand"]
        ltf_supply = [z for z in ltf_scored if z.get("type") == "supply"]

        enhanced_htf_demand = overlap_detector.find_overlapping_zones(htf_demand, ltf_demand)
        enhanced_htf_supply = overlap_detector.find_overlapping_zones(htf_supply, ltf_supply)
        enhanced_htf_all = enhanced_htf_demand + enhanced_htf_supply

        # Boost strength for zones with LTF overlap
        for zone in enhanced_htf_all:
            if zone.get("has_ltf_overlap"):
                zone["strength"] = zone.get("strength", 0) * 1.15

        # Re-sort by final strength
        enhanced_htf_all.sort(key=lambda z: z.get("strength", 0), reverse=True)
        ltf_scored.sort(key=lambda z: z.get("strength", 0), reverse=True)

        # Filter for fresh (non-mitigated) zones only
        # Per PDF: Only use zones with < 50% mitigation
        htf_fresh = [z for z in enhanced_htf_all if z.get("mitigation_percent", 0) < 50]
        ltf_fresh = [z for z in ltf_scored if z.get("mitigation_percent", 0) < 50]

        # Identify high-quality vs low-quality zones
        htf_high_quality = [z for z in htf_fresh if z.get("pivot_quality", {}).get("quality_label", "") in ["EXTREME_HIGH_QUALITY", "HIGH_QUALITY"]]
        htf_low_quality = [z for z in htf_fresh if z.get("pivot_quality", {}).get("quality_label", "") == "LOW_QUALITY"]

        return JSONResponse({
            "symbol": SYMBOL,
            "current_price": current_price,
            "methodology": "TCT Mentorship Lecture 3 + 38-Page PDF Enhanced",
            "htf_zones": {
                "timeframe": "4h",
                "total_zones": len(htf_fresh),
                "high_quality_count": len(htf_high_quality),
                "low_quality_count": len(htf_low_quality),
                "top_3_all": htf_fresh[:3],
                "top_3_high_quality": htf_high_quality[:3]
            },
            "ltf_zones": {
                "timeframe": "15m",
                "total_zones": len(ltf_fresh),
                "top_3": ltf_fresh[:3]
            },
            "detected_range": detected_range,
            "summary": {
                "htf_fvg_count": len(htf_fvgs.get("bullish_fvgs", [])) + len(htf_fvgs.get("bearish_fvgs", [])),
                "ltf_fvg_count": len(ltf_fvgs.get("bullish_fvgs", [])) + len(ltf_fvgs.get("bearish_fvgs", [])),
                "htf_order_blocks": len(htf_obs.get("bullish_obs", [])) + len(htf_obs.get("bearish_obs", [])),
                "ltf_order_blocks": len(ltf_obs.get("bullish_obs", [])) + len(ltf_obs.get("bearish_obs", [])),
                "htf_zones_with_ltf_validation": sum(1 for z in htf_fresh if z.get("mtf_validation", {}).get("has_true_inefficiency")),
                "htf_zones_with_ltf_overlap": sum(1 for z in htf_fresh if z.get("has_ltf_overlap")),
                "zones_with_liquidity_sweep": sum(1 for z in htf_fresh if z.get("pivot_quality", {}).get("has_liquidity_sweep"))
            },
            "enhancements_applied": {
                "multi_timeframe_validation": True,
                "zone_refinement": True,
                "pivot_quality_scoring": True,
                "overlapping_zone_detection": True,
                "liquidity_sweep_detection": True
            }
        })

    except Exception as e:
        logger.error(f"[ZONES_ERROR] {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/api/ranges")
async def detect_ranges():
    """
    Detect and analyze ranges using PURE TCT Lecture 2 methodology.

    Pure TCT Methodology from "TCT-2024-mentorship-Lecture-2-Ranges-AI-Text-Only.pdf":
    - Range = price moving sideways (consolidation phase)
    - Purpose: Restore balance between buyers and sellers after aggressive expansion
    - Uptrend: Pull range from TOP to BOTTOM (high → low)
    - Downtrend: Pull range from BOTTOM to TOP (low → high)
    - Range confirmed when price moves back to EQUILIBRIUM (0.5 fib level)
    - Six Candle Rule: Range valid if 2 candles up, 2 down, 2 up (uptrend) or inverse
    - Deviation Limit (DL): 30% of range size - threshold for deviation vs break
    - Wick deviation = easiest (never broke structure)
    - Candle close deviation = bad break of structure
    - Premium (above 0.5) vs Discount (below 0.5) pricing zones
    - Ranges within ranges: Multiple timeframe nesting
    - Good range = horizontal price action, Bad range = sharp V-shaped moves

    Returns:
        - HTF and LTF detected ranges with quality scoring
        - Active range with premium/discount classification
        - Deviations (wick, candle close, DL-based)
        - Current position classification
        - Nested ranges (ranges within ranges)
        - Trading targets based on range zones
    """
    try:
        # Fetch candles
        htf_df = await fetch_mexc_candles(SYMBOL, "4h", 100)
        ltf_df = await fetch_mexc_candles(SYMBOL, "15m", 200)

        if htf_df is None or ltf_df is None:
            return JSONResponse({"error": "Failed to fetch data"}, status_code=500)

        current_price = float(ltf_df.iloc[-1]["close"])

        # Detect market structure (pivots)
        ms = MarketStructure()
        htf_pivots = ms.detect_pivots(htf_df)
        ltf_pivots = ms.detect_pivots(ltf_df)

        # === STEP 2: RANGE DETECTION (TCT Lecture 2) ===

        # 1. Detect Ranges on both timeframes
        range_detector = TCTRangeDetector()
        htf_ranges_result = range_detector.detect_ranges(htf_df, htf_pivots)
        ltf_ranges_result = range_detector.detect_ranges(ltf_df, ltf_pivots)

        htf_ranges = htf_ranges_result.get("ranges", [])
        ltf_ranges = ltf_ranges_result.get("ranges", [])

        # Get active ranges (most relevant confirmed ranges)
        htf_active_range = htf_ranges_result.get("active_range")
        ltf_active_range = ltf_ranges_result.get("active_range")

        # FALLBACK: If no TCT range detected, create a simple range from recent price action
        # This ensures the dashboard always has range data to display
        if not htf_active_range and len(htf_df) >= 20:
            recent_high = float(htf_df.tail(50)["high"].max())
            recent_low = float(htf_df.tail(50)["low"].min())
            eq = (recent_high + recent_low) / 2
            range_size = recent_high - recent_low
            htf_active_range = {
                "range_high": recent_high,
                "range_low": recent_low,
                "equilibrium": eq,
                "range_size": range_size,
                "deviation_limit_high": recent_high + (range_size * 0.30),
                "deviation_limit_low": recent_low - (range_size * 0.30),
                "is_fallback": True,  # Mark as fallback (not TCT validated)
                "is_confirmed": False
            }

        if not ltf_active_range and len(ltf_df) >= 20:
            recent_high = float(ltf_df.tail(100)["high"].max())
            recent_low = float(ltf_df.tail(100)["low"].min())
            eq = (recent_high + recent_low) / 2
            range_size = recent_high - recent_low
            ltf_active_range = {
                "range_high": recent_high,
                "range_low": recent_low,
                "equilibrium": eq,
                "range_size": range_size,
                "deviation_limit_high": recent_high + (range_size * 0.30),
                "deviation_limit_low": recent_low - (range_size * 0.30),
                "is_fallback": True,  # Mark as fallback (not TCT validated)
                "is_confirmed": False
            }

        # 2. Detect Deviations from active ranges
        deviation_detector = TCTDeviationDetector()

        htf_deviations = deviation_detector.detect_deviations(htf_df, htf_active_range)
        ltf_deviations = deviation_detector.detect_deviations(ltf_df, ltf_active_range)

        # 3. Classify Current Position (Premium/Discount)
        position_classifier = TCTPremiumDiscountClassifier()

        # Use LTF active range for current position if available, else HTF
        active_range_for_position = ltf_active_range or htf_active_range

        current_position = position_classifier.classify_current_position(
            current_price, active_range_for_position
        )

        # Get trading targets
        zone_targets = position_classifier.get_zone_targets(active_range_for_position)

        # 4. Detect Nested Ranges (Ranges within Ranges)
        nesting_detector = TCTRangesWithinRanges()
        nested_ranges = nesting_detector.detect_nested_ranges(htf_ranges, ltf_ranges)

        # 5. Identify Most Recent Rotation within active range
        htf_rotation = nesting_detector.identify_most_recent_rotation(htf_df, htf_active_range)
        ltf_rotation = nesting_detector.identify_most_recent_rotation(ltf_df, ltf_active_range)

        # === SUMMARY STATISTICS ===

        # Count confirmed vs unconfirmed ranges
        htf_confirmed = [r for r in htf_ranges if r.get("is_confirmed")]
        ltf_confirmed = [r for r in ltf_ranges if r.get("is_confirmed")]

        # Count range quality distribution
        htf_quality_dist = {}
        for r in htf_ranges:
            quality = r.get("quality", {}).get("quality_label", "UNKNOWN")
            htf_quality_dist[quality] = htf_quality_dist.get(quality, 0) + 1

        ltf_quality_dist = {}
        for r in ltf_ranges:
            quality = r.get("quality", {}).get("quality_label", "UNKNOWN")
            ltf_quality_dist[quality] = ltf_quality_dist.get(quality, 0) + 1

        # Convert numpy types to native Python types for JSON serialization
        response_data = convert_numpy_types({
            "symbol": SYMBOL,
            "current_price": current_price,
            "methodology": "TCT Mentorship Lecture 2 - Ranges",
            "htf_ranges": {
                "timeframe": "4h",
                "total_ranges": len(htf_ranges),
                "confirmed_ranges": len(htf_confirmed),
                "active_range": htf_active_range,
                "all_ranges": htf_ranges[:5],  # Top 5
                "quality_distribution": htf_quality_dist
            },
            "ltf_ranges": {
                "timeframe": "15m",
                "total_ranges": len(ltf_ranges),
                "confirmed_ranges": len(ltf_confirmed),
                "active_range": ltf_active_range,
                "all_ranges": ltf_ranges[:5],  # Top 5
                "quality_distribution": ltf_quality_dist
            },
            "deviations": {
                "htf_deviations": htf_deviations,
                "ltf_deviations": ltf_deviations,
                "total_htf_deviations": htf_deviations.get("total_deviations", 0),
                "total_ltf_deviations": ltf_deviations.get("total_deviations", 0)
            },
            "current_position": current_position,
            "zone_targets": zone_targets,
            "nested_ranges": nested_ranges,
            "rotations": {
                "htf_rotation": htf_rotation,
                "ltf_rotation": ltf_rotation
            },
            "market_structure": {
                "htf_trend": htf_pivots.get("trend", "neutral"),
                "ltf_trend": ltf_pivots.get("trend", "neutral")
            },
            "summary": {
                "htf_ranges_found": htf_ranges_result.get("total_ranges_found", 0),
                "ltf_ranges_found": ltf_ranges_result.get("total_ranges_found", 0),
                "htf_has_active_range": htf_active_range is not None,
                "ltf_has_active_range": ltf_active_range is not None,
                "current_zone": current_position.get("zone", "UNKNOWN"),
                "trading_bias": current_position.get("trading_bias", "NEUTRAL"),
                "htf_deviations_detected": htf_deviations.get("has_deviation", False),
                "ltf_deviations_detected": ltf_deviations.get("has_deviation", False),
                "nested_htf_ranges": nested_ranges.get("htf_ranges_with_nesting", 0),
                "total_nested_ltf_ranges": nested_ranges.get("total_nested_ltf_ranges", 0)
            },
            "tct_concepts": {
                "six_candle_rule": "Range valid if 2 candles up, 2 down, 2 up (or inverse)",
                "deviation_limit": "30% of range size - threshold for deviation vs break",
                "premium_zone": "Above equilibrium (0.5) to range high",
                "discount_zone": "Below equilibrium (0.5) to range low",
                "equilibrium": "0.5 fib level - range confirmation point",
                "wick_deviation": "Easiest deviation - never broke structure",
                "bad_bos_deviation": "Candle close deviation - closes back inside quickly"
            }
        })
        return JSONResponse(response_data)

    except Exception as e:
        logger.error(f"[RANGES_ERROR] {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/api/liquidity")
async def detect_liquidity():
    """
    Detect liquidity pools, curves, and targets using PURE TCT Lecture 4 methodology.

    Pure TCT Methodology from PDFs:
    - BSL (Buy-Side Liquidity) = liquidity above primary/internal highs
    - SSL (Sell-Side Liquidity) = liquidity below primary/internal lows
    - Primary highs/lows = visible pivot points
    - Internal highs/lows = between primaries (stacking up liquidity)
    - Equal highs/lows = amazing liquidity targets (exact same price)
    - Liquidity curves = connecting progressively lower/higher primaries
    - 2nd tap observations → extreme liquidity POI (first higher low / first lower high)
    - Liquidity voids = areas with no S&D zones (price moves freely)
    - Non-liquidity highs/lows = grabbed liquidity from left (NOT valid targets)
    - Exception: price spending time near = becomes part of liquidity curve

    Returns:
        - BSL/SSL pools with primary/internal classification
        - Liquidity curves (3-tap patterns: 1st, 2nd, 3rd)
        - Extreme liquidity targets (after 2nd tap observation)
        - Liquidity voids
        - Integration with S&D zones (high-probability vs low-probability)
    """
    try:
        # Fetch candles
        htf_df = await fetch_mexc_candles(SYMBOL, "4h", 100)
        ltf_df = await fetch_mexc_candles(SYMBOL, "15m", 200)

        if htf_df is None or ltf_df is None:
            return JSONResponse({"error": "Failed to fetch data"}, status_code=500)

        current_price = float(ltf_df.iloc[-1]["close"])

        # Detect market structure (pivots)
        ms = MarketStructure()
        htf_pivots = ms.detect_pivots(htf_df)
        ltf_pivots = ms.detect_pivots(ltf_df)

        # Detect FVGs
        fvg_detector = FairValueGap()
        htf_fvgs = fvg_detector.detect_fvgs(htf_df)
        ltf_fvgs = fvg_detector.detect_fvgs(ltf_df)

        # Detect Order Blocks
        ob_detector = OrderBlock()
        htf_obs = ob_detector.detect_order_blocks(htf_df, htf_fvgs)
        ltf_obs = ob_detector.detect_order_blocks(ltf_df, ltf_fvgs)

        # Detect Structure Supply/Demand
        ssd_detector = StructureSupplyDemand()
        htf_structure = ssd_detector.detect_structure_zones(htf_df, htf_fvgs, htf_pivots)
        ltf_structure = ssd_detector.detect_structure_zones(ltf_df, ltf_fvgs, ltf_pivots)

        # === STEP 4: LIQUIDITY DETECTION ===

        # 1. Detect Liquidity Pools (BSL/SSL)
        liq_detector = LiquidityDetector()
        htf_liquidity = liq_detector.detect_liquidity_pools(htf_df, htf_pivots, current_price)
        ltf_liquidity = liq_detector.detect_liquidity_pools(ltf_df, ltf_pivots, current_price)

        # 2. Generate Liquidity Curves
        curve_generator = LiquidityCurveGenerator()
        htf_curves = curve_generator.generate_curves(htf_liquidity, htf_df)
        ltf_curves = curve_generator.generate_curves(ltf_liquidity, ltf_df)

        # 3. Identify Extreme Liquidity Targets
        target_detector = ExtremeLiquidityTarget()
        htf_targets = target_detector.identify_extreme_targets(htf_curves, htf_df)
        ltf_targets = target_detector.identify_extreme_targets(ltf_curves, ltf_df)

        # 4. Combine all zones for void detection
        all_htf_zones = (
            htf_obs.get("bullish_obs", []) +
            htf_obs.get("bearish_obs", []) +
            htf_structure.get("demand_zones", []) +
            htf_structure.get("supply_zones", [])
        )

        all_ltf_zones = (
            ltf_obs.get("bullish_obs", []) +
            ltf_obs.get("bearish_obs", []) +
            ltf_structure.get("demand_zones", []) +
            ltf_structure.get("supply_zones", [])
        )

        # 5. Detect Liquidity Voids
        void_detector = LiquidityVoidDetector()
        htf_voids = void_detector.detect_voids(all_htf_zones, htf_df, current_price)
        ltf_voids = void_detector.detect_voids(all_ltf_zones, ltf_df, current_price)

        # === INTEGRATION: Link Liquidity with S&D Zones ===

        # Find S&D zones that are AT liquidity pools (high-probability zones)
        htf_zones_at_liquidity = []
        for zone in all_htf_zones:
            zone_mid = (zone.get("top", 0) + zone.get("bottom", 0)) / 2
            zone_type = zone.get("type", "unknown")

            # Check if zone is at BSL pool
            at_bsl = any(
                abs(zone_mid - pool["price"]) / pool["price"] < 0.01
                for pool in htf_liquidity.get("bsl_pools", [])
                if pool.get("is_primary", False)
            )

            # Check if zone is at SSL pool
            at_ssl = any(
                abs(zone_mid - pool["price"]) / pool["price"] < 0.01
                for pool in htf_liquidity.get("ssl_pools", [])
                if pool.get("is_primary", False)
            )

            if at_bsl or at_ssl:
                htf_zones_at_liquidity.append({
                    **zone,
                    "at_liquidity_pool": True,
                    "liquidity_type": "BSL" if at_bsl else "SSL",
                    "high_probability": True
                })

        # Find zones inside liquidity voids (low-probability zones)
        htf_zones_in_voids = []
        for zone in all_htf_zones:
            zone_mid = (zone.get("top", 0) + zone.get("bottom", 0)) / 2

            in_void = any(
                void["low"] <= zone_mid <= void["high"]
                for void in htf_voids.get("voids", [])
            )

            if in_void:
                htf_zones_in_voids.append({
                    **zone,
                    "in_liquidity_void": True,
                    "low_probability": True
                })

        # Check if liquidity sweep occurred at extreme targets
        for target in htf_targets.get("sell_side_targets", []) + htf_targets.get("buy_side_targets", []):
            target_price = target["target_price"]

            # Check if current price swept the target
            if target["type"] == "distribution":
                # For sell-side, check if price went below target
                target["is_swept"] = current_price < target_price
            else:
                # For buy-side, check if price went above target
                target["is_swept"] = current_price > target_price

        return JSONResponse({
            "symbol": SYMBOL,
            "current_price": current_price,
            "methodology": "TCT Mentorship Lecture 4 - Liquidity",
            "htf_liquidity": {
                "timeframe": "4h",
                "bsl_pools": htf_liquidity.get("bsl_pools", [])[:10],  # Top 10
                "ssl_pools": htf_liquidity.get("ssl_pools", [])[:10],
                "equal_highs": htf_liquidity.get("equal_highs", []),
                "equal_lows": htf_liquidity.get("equal_lows", []),
                "primary_highs_count": htf_liquidity.get("primary_highs_count", 0),
                "primary_lows_count": htf_liquidity.get("primary_lows_count", 0),
                "internal_highs_count": htf_liquidity.get("internal_highs_count", 0),
                "internal_lows_count": htf_liquidity.get("internal_lows_count", 0)
            },
            "ltf_liquidity": {
                "timeframe": "15m",
                "bsl_pools": ltf_liquidity.get("bsl_pools", [])[:10],
                "ssl_pools": ltf_liquidity.get("ssl_pools", [])[:10],
                "equal_highs": ltf_liquidity.get("equal_highs", []),
                "equal_lows": ltf_liquidity.get("equal_lows", [])
            },
            "htf_curves": {
                "sell_side_curves": htf_curves.get("sell_side_curves", []),
                "buy_side_curves": htf_curves.get("buy_side_curves", []),
                "total_curves": htf_curves.get("total_curves", 0)
            },
            "ltf_curves": {
                "sell_side_curves": ltf_curves.get("sell_side_curves", []),
                "buy_side_curves": ltf_curves.get("buy_side_curves", []),
                "total_curves": ltf_curves.get("total_curves", 0)
            },
            "extreme_targets": {
                "htf": htf_targets,
                "ltf": ltf_targets,
                "total_targets": htf_targets.get("total_targets", 0) + ltf_targets.get("total_targets", 0)
            },
            "liquidity_voids": {
                "htf_voids": htf_voids.get("voids", []),
                "ltf_voids": ltf_voids.get("voids", []),
                "htf_largest_void": htf_voids.get("largest_void"),
                "ltf_largest_void": ltf_voids.get("largest_void")
            },
            "integration": {
                "zones_at_liquidity_pools": htf_zones_at_liquidity,
                "zones_in_voids": htf_zones_in_voids,
                "high_probability_zone_count": len(htf_zones_at_liquidity),
                "low_probability_zone_count": len(htf_zones_in_voids)
            },
            "summary": {
                "total_bsl_pools": len(htf_liquidity.get("bsl_pools", [])),
                "total_ssl_pools": len(htf_liquidity.get("ssl_pools", [])),
                "total_equal_highs": len(htf_liquidity.get("equal_highs", [])),
                "total_equal_lows": len(htf_liquidity.get("equal_lows", [])),
                "total_curves": htf_curves.get("total_curves", 0),
                "total_extreme_targets": htf_targets.get("total_targets", 0),
                "total_voids": htf_voids.get("total_voids", 0),
                "zones_enhanced_by_liquidity": len(htf_zones_at_liquidity)
            }
        })

    except Exception as e:
        logger.error(f"[LIQUIDITY_ERROR] {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/api/validate")
async def validate_current_setup():
    """Complete 7-gate validation endpoint"""
    try:
        htf_df = await fetch_mexc_candles(SYMBOL, "4h", 100)
        ltf_df = await fetch_mexc_candles(SYMBOL, "15m", 100)

        if htf_df is None or ltf_df is None:
            return JSONResponse({"error": "Failed to fetch data", "Action": "NO_TRADE"}, status_code=500)

        current_price = float(ltf_df.iloc[-1]["close"])

        # Convert to dict with timestamps as strings
        ltf_candles = []
        for _, row in ltf_df.iterrows():
            ltf_candles.append({
                'open_time': str(row['open_time']),
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': float(row['volume'])
            })

        detected_range = await detect_best_range(ltf_candles)

        context = {
            "htf_candles": htf_df,
            "ltf_candles": ltf_df,
            "detected_range": detected_range,
            "current_price": current_price,
            "symbol": SYMBOL
        }

        result = validate_gates(context)
        result.pop("htf_candles", None)
        result.pop("ltf_candles", None)

        return JSONResponse(result)
    except Exception as e:
        logger.error(f"[VALIDATE_ERROR] {e}")
        return JSONResponse({"error": str(e), "Action": "NO_TRADE"}, status_code=500)

@app.get("/api/schematics")
async def get_tct_schematics():
    """
    TCT Schematics endpoint - Lecture 5A + 5B + 6 Advanced methodology

    Detects TCT Accumulation and Distribution schematics:
    - Model 1: Two successive deviations (Tap1 → Tap2 → Tap3 deeper)
    - Model 2: One deviation then higher low/lower high at extreme liquidity or demand/supply

    Lecture 6 Advanced Features:
    - Schematic conversion (distribution → accumulation and vice versa)
    - Dual-side deviation awareness with risk-on/risk-off triggers
    - LTF-to-HTF range transition detection
    - Multi-timeframe schematic validity checking
    - WOV-in-WOV (schematic within schematic) for R:R optimization
    - Model 1 to Model 2 flow with position management
    - Context-based follow-through prediction (premium/discount zones)

    Returns schematics with entry, stop loss, target levels, and advanced enhancements.
    """
    try:
        # Fetch HTF (4h) and LTF (15m) candles
        htf_df = await fetch_mexc_candles(SYMBOL, "4h", 200)
        ltf_df = await fetch_mexc_candles(SYMBOL, "15m", 200)

        if htf_df is None or ltf_df is None:
            return JSONResponse({"error": "Failed to fetch candle data"}, status_code=500)

        current_price = float(ltf_df.iloc[-1]["close"])

        # Detect ranges for both timeframes (convert to list for range detection)
        def df_to_candles(df):
            candles = []
            for _, row in df.iterrows():
                candles.append({
                    'open_time': str(row['open_time']),
                    'open': float(row['open']),
                    'high': float(row['high']),
                    'low': float(row['low']),
                    'close': float(row['close']),
                    'volume': float(row['volume'])
                })
            return candles

        htf_candles_list = df_to_candles(htf_df)
        ltf_candles_list = df_to_candles(ltf_df)

        # Detect ranges for both timeframes
        htf_ranges = await detect_best_range(htf_candles_list)
        ltf_ranges = await detect_best_range(ltf_candles_list)

        # Convert single range to list if needed
        htf_range_list = [htf_ranges] if htf_ranges and not isinstance(htf_ranges, list) else (htf_ranges or [])
        ltf_range_list = [ltf_ranges] if ltf_ranges and not isinstance(ltf_ranges, list) else (ltf_ranges or [])

        # Detect TCT schematics on both timeframes (pass DataFrame, not list)
        htf_schematics_result = detect_tct_schematics(htf_df, htf_range_list)
        ltf_schematics_result = detect_tct_schematics(ltf_df, ltf_range_list)

        # Extract schematic lists from result dict
        htf_schematics = (
            htf_schematics_result.get("accumulation_schematics", []) +
            htf_schematics_result.get("distribution_schematics", [])
        )
        ltf_schematics = (
            ltf_schematics_result.get("accumulation_schematics", []) +
            ltf_schematics_result.get("distribution_schematics", [])
        )

        # Filter and sort schematics by quality
        def filter_active_schematics(schematics, current_price):
            """Filter to schematics that are still valid for trading"""
            active = []
            for s in schematics:
                if not isinstance(s, dict):
                    continue
                # Check if schematic is still valid (price hasn't hit target or stop)
                entry = s.get('entry', {}).get('price')
                target = s.get('target', {}).get('price')
                stop = s.get('stop_loss', {}).get('price')

                if entry and target and stop:
                    # For long (accumulation)
                    if s.get('direction') == 'bullish':
                        if current_price < target and current_price > stop:
                            active.append(s)
                    # For short (distribution)
                    elif s.get('direction') == 'bearish':
                        if current_price > target and current_price < stop:
                            active.append(s)
                else:
                    # No complete trade management yet, still include
                    active.append(s)
            return sorted(active, key=lambda x: x.get('quality_score', 0), reverse=True)

        htf_active = filter_active_schematics(htf_schematics, current_price)
        ltf_active = filter_active_schematics(ltf_schematics, current_price)

        # Summarize schematics
        def summarize_schematics(schematics):
            return {
                'total': len(schematics),
                'model_1_accumulation': sum(1 for s in schematics if s.get('schematic_type') == 'model_1_accumulation'),
                'model_2_accumulation': sum(1 for s in schematics if s.get('schematic_type') == 'model_2_accumulation'),
                'model_1_distribution': sum(1 for s in schematics if s.get('schematic_type') == 'model_1_distribution'),
                'model_2_distribution': sum(1 for s in schematics if s.get('schematic_type') == 'model_2_distribution'),
                'confirmed': sum(1 for s in schematics if s.get('status') == 'confirmed'),
                'forming': sum(1 for s in schematics if s.get('status') == 'forming'),
            }

        # Convert numpy types to native Python types for JSON serialization
        response_data = convert_numpy_types({
            "symbol": SYMBOL,
            "current_price": current_price,
            "methodology": "TCT Mentorship Lecture 5A + 5B + 6 - Advanced TCT Schematics",
            "htf_schematics": {
                "timeframe": "4h",
                "schematics": htf_active[:5],  # Top 5 by quality
                "summary": summarize_schematics(htf_active)
            },
            "ltf_schematics": {
                "timeframe": "15m",
                "schematics": ltf_active[:5],  # Top 5 by quality
                "summary": summarize_schematics(ltf_active)
            },
            "trading_rules": {
                "model_1": "Two successive deviations - Tap2 below Tap1, Tap3 below Tap2 (accumulation) or above (distribution)",
                "model_2": "One deviation then higher low/lower high - must grab extreme liquidity OR mitigate extreme demand/supply",
                "entry": "Wait for BOS confirmation from lowest/highest point between Tap2 and Tap3",
                "stop_loss": "Below Tap3 for longs, above Tap3 for shorts",
                "target": "Opposite range extreme (Wyckoff high for longs, Wyckoff low for shorts)",
                "six_candle_rule": "Each tap pivot must pass 6-candle rule for valid schematic on that timeframe"
            },
            "lecture_6_rules": {
                "schematic_conversion": "Distribution can convert to accumulation (and vice versa) when opposite deviation occurs",
                "dual_side_deviation": "When both sides deviate, trigger risk-on (more aggressive) until range extreme breaks",
                "ltf_htf_transition": "LTF schematics can grow into HTF ranges - watch for nested structures",
                "multi_tf_validity": "Check if tap2/tap3 close enough to merge on HTF - affects target selection",
                "wov_in_wov": "Look for schematic within schematic for dramatic R:R improvement",
                "m1_to_m2_flow": "Model 1 can flow into Model 2 - add to position at M2 entry, same stop, extended target",
                "context_follow_through": "Premium zone expects distribution, discount zone expects accumulation"
            },
            "summary": {
                "total_htf_schematics": len(htf_active),
                "total_ltf_schematics": len(ltf_active),
                "best_htf_quality": htf_active[0].get('quality_score', 0) if htf_active else 0,
                "best_ltf_quality": ltf_active[0].get('quality_score', 0) if ltf_active else 0,
            }
        })
        return JSONResponse(response_data)

    except Exception as e:
        logger.error(f"[SCHEMATICS_ERROR] {e}")
        import traceback
        logger.error(traceback.format_exc())
        return JSONResponse({"error": str(e)}, status_code=500)

# ================================================================
# ENTRY POINT
# ================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server_mexc:app", host="0.0.0.0", port=PORT, reload=True)
