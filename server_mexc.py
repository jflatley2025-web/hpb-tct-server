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
from fastapi.responses import JSONResponse
from loguru import logger

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
                pivots["highs"].append({"idx": i, "price": candles.iloc[i]["high"]})

            if (
                candles.iloc[i - 2]["close"] > candles.iloc[i - 1]["close"]
                and candles.iloc[i - 1]["close"] > candles.iloc[i]["close"]
                and candles.iloc[i]["close"] < candles.iloc[i + 1]["close"]
                and candles.iloc[i + 1]["close"] < candles.iloc[i + 2]["close"]
            ):
                pivots["lows"].append({"idx": i, "price": candles.iloc[i]["low"]})

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
                    "time": c_curr["open_time"] if "open_time" in c_curr.index else i,
                    "gap_size": float(c_next["low"] - c_prev["high"])
                })

            # Bearish FVG: gap between prev low and next high
            if c_prev["low"] > c_next["high"]:
                bearish_fvgs.append({
                    "idx": i,
                    "top": float(c_prev["low"]),
                    "bottom": float(c_next["high"]),
                    "time": c_curr["open_time"] if "open_time" in c_curr.index else i,
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
                    "time": ob_candle["open_time"] if "open_time" in ob_candle.index else idx,
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
                    "time": ob_candle["open_time"] if "open_time" in ob_candle.index else idx,
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
            "is_overlapping": is_overlapping,
            "is_within_only": ltf_within and not is_overlapping,
            "overlap_percent": min(overlap_percent, 100.0),
            "represents_htf": is_overlapping,  # Only overlapping zones represent HTF
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
                "has_ltf_overlap": len(overlapping_ltf) > 0,
                "overlapping_ltf_zones": overlapping_ltf,
                "ltf_overlap_count": len(overlapping_ltf)
            })

        return enhanced_htf_zones

# ================================================================
# LIQUIDITY DETECTION (TCT Mentorship Lecture 4)
# ================================================================

class LiquidityDetector:
    """
    Detects Buy-Side Liquidity (BSL) and Sell-Side Liquidity (SSL) pools.

    Methodology from Liquidity PDFs:
    - BSL = Liquidity above highs (stop losses above resistance)
    - SSL = Liquidity below lows (stop losses below support)
    - Primary highs/lows = visible pivot points (major liquidity pools)
    - Internal highs/lows = price action between primaries (less significant)
    - Equal highs/lows = multiple touches at same level (strong liquidity targets)
    - Non-liquidity highs/lows = highs/lows that grabbed previous liquidity
    """

    @staticmethod
    def detect_liquidity_pools(candles: pd.DataFrame, pivots: Dict, current_price: float) -> Dict:
        """
        Detect BSL (above highs) and SSL (below lows) liquidity pools.

        Returns: Dict with bsl_pools and ssl_pools lists
        """
        bsl_pools = []  # Buy-side liquidity (above price)
        ssl_pools = []  # Sell-side liquidity (below price)

        # Classify pivots as primary vs internal
        primary_highs, internal_highs = LiquidityDetector._classify_primary_vs_internal(
            pivots.get("highs", []), is_highs=True
        )

        primary_lows, internal_lows = LiquidityDetector._classify_primary_vs_internal(
            pivots.get("lows", []), is_lows=True
        )

        # Detect equal highs (BSL)
        equal_highs = LiquidityDetector._detect_equal_levels(
            [p["price"] for p in pivots.get("highs", [])],
            tolerance=0.001  # 0.1% tolerance
        )

        # Detect equal lows (SSL)
        equal_lows = LiquidityDetector._detect_equal_levels(
            [p["price"] for p in pivots.get("lows", [])],
            tolerance=0.001
        )

        # Create BSL pools from primary highs
        for pivot in primary_highs:
            price = pivot["price"]

            # Check if this is a non-liquidity high
            is_non_liquidity = LiquidityDetector._is_non_liquidity_high(
                pivot, pivots.get("highs", []), candles
            )

            if not is_non_liquidity:
                is_equal = any(abs(price - eq) / eq < 0.001 for eq in equal_highs)

                bsl_pools.append({
                    "type": "BSL",
                    "price": float(price),
                    "idx": pivot["idx"],
                    "is_primary": True,
                    "is_equal": is_equal,
                    "strength": 1.0 if is_equal else 0.75,
                    "distance_from_price": float((price - current_price) / current_price * 100)
                })

        # Create SSL pools from primary lows
        for pivot in primary_lows:
            price = pivot["price"]

            # Check if this is a non-liquidity low
            is_non_liquidity = LiquidityDetector._is_non_liquidity_low(
                pivot, pivots.get("lows", []), candles
            )

            if not is_non_liquidity:
                is_equal = any(abs(price - eq) / eq < 0.001 for eq in equal_lows)

                ssl_pools.append({
                    "type": "SSL",
                    "price": float(price),
                    "idx": pivot["idx"],
                    "is_primary": True,
                    "is_equal": is_equal,
                    "strength": 1.0 if is_equal else 0.75,
                    "distance_from_price": float((current_price - price) / current_price * 100)
                })

        # Add internal highs/lows as weaker liquidity
        for pivot in internal_highs:
            price = pivot["price"]
            bsl_pools.append({
                "type": "BSL",
                "price": float(price),
                "idx": pivot["idx"],
                "is_primary": False,
                "is_equal": False,
                "strength": 0.4,
                "distance_from_price": float((price - current_price) / current_price * 100)
            })

        for pivot in internal_lows:
            price = pivot["price"]
            ssl_pools.append({
                "type": "SSL",
                "price": float(price),
                "idx": pivot["idx"],
                "is_primary": False,
                "is_equal": False,
                "strength": 0.4,
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
        Classify pivots as primary (visible, significant) or internal (between primaries).

        Primary pivots = major swing points that stand out
        Internal pivots = smaller swings between primaries
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

        # Window-based classification
        window_size = 5
        for i, pivot in enumerate(sorted_pivots):
            start = max(0, i - window_size)
            end = min(len(sorted_pivots), i + window_size + 1)
            window = sorted_pivots[start:end]
            window_prices = [p["price"] for p in window]

            if is_highs:
                # For highs: primary if it's among the highest in window
                is_primary = pivot["price"] >= sorted(window_prices)[-2] if len(window_prices) >= 2 else True
            elif is_lows:
                # For lows: primary if it's among the lowest in window
                is_primary = pivot["price"] <= sorted(window_prices)[1] if len(window_prices) >= 2 else True
            else:
                is_primary = True

            if is_primary:
                primary.append(pivot)
            else:
                internal.append(pivot)

        return primary, internal

    @staticmethod
    def _detect_equal_levels(prices: List[float], tolerance: float = 0.001) -> List[float]:
        """
        Detect equal highs/lows (multiple touches at same price level).

        Returns list of price levels that have 2+ touches within tolerance.
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
                if abs(price1 - price2) / price1 < tolerance:
                    matches.append(price2)
                    processed.add(j)

            # If 2+ matches, it's an equal level
            if len(matches) >= 2:
                equal_levels.append(sum(matches) / len(matches))
                processed.add(i)

        return equal_levels

    @staticmethod
    def _is_non_liquidity_high(pivot: Dict, all_highs: List[Dict], candles: pd.DataFrame) -> bool:
        """
        Check if high is a non-liquidity high.

        Non-liquidity high = high that grabbed liquidity from previous high,
        followed by aggressive expansion downward.

        Exception: If price spent time near the swept level, it IS liquidity.
        """
        idx = pivot["idx"]
        price = pivot["price"]

        if idx < 5 or idx >= len(candles) - 2:
            return False

        # Find previous highs
        previous_highs = [h for h in all_highs if h["idx"] < idx]
        if not previous_highs:
            return False

        # Check if this high swept a previous high
        swept_previous = any(price > h["price"] for h in previous_highs[-3:])

        if not swept_previous:
            return False

        # Check for aggressive expansion downward after this high
        next_candles = candles.iloc[idx+1:min(idx+4, len(candles))]
        if len(next_candles) == 0:
            return False

        high_candle = candles.iloc[idx]
        lowest_after = next_candles["low"].min()
        expansion_size = high_candle["high"] - lowest_after
        high_size = high_candle["high"] - high_candle["low"]

        # Aggressive expansion = move down > 2x the high candle size
        has_aggressive_expansion = expansion_size > high_size * 2.0

        if not has_aggressive_expansion:
            return False

        # Exception: Check if price spent time near this level
        time_spent_near = sum(
            1 for c in next_candles.itertuples()
            if abs(c.close - price) / price < 0.01
        )

        # If spent 2+ candles near level, it IS liquidity (not non-liquidity)
        if time_spent_near >= 2:
            return False

        return True  # It's a non-liquidity high

    @staticmethod
    def _is_non_liquidity_low(pivot: Dict, all_lows: List[Dict], candles: pd.DataFrame) -> bool:
        """
        Check if low is a non-liquidity low.

        Non-liquidity low = low that grabbed liquidity from previous low,
        followed by aggressive expansion upward.
        """
        idx = pivot["idx"]
        price = pivot["price"]

        if idx < 5 or idx >= len(candles) - 2:
            return False

        # Find previous lows
        previous_lows = [l for l in all_lows if l["idx"] < idx]
        if not previous_lows:
            return False

        # Check if this low swept a previous low
        swept_previous = any(price < l["price"] for l in previous_lows[-3:])

        if not swept_previous:
            return False

        # Check for aggressive expansion upward after this low
        next_candles = candles.iloc[idx+1:min(idx+4, len(candles))]
        if len(next_candles) == 0:
            return False

        low_candle = candles.iloc[idx]
        highest_after = next_candles["high"].max()
        expansion_size = highest_after - low_candle["low"]
        low_size = low_candle["high"] - low_candle["low"]

        # Aggressive expansion = move up > 2x the low candle size
        has_aggressive_expansion = expansion_size > low_size * 2.0

        if not has_aggressive_expansion:
            return False

        # Exception: Check if price spent time near this level
        time_spent_near = sum(
            1 for c in next_candles.itertuples()
            if abs(c.close - price) / price < 0.01
        )

        if time_spent_near >= 2:
            return False

        return True

# ================================================================
# LIQUIDITY CURVE GENERATION
# ================================================================

class LiquidityCurveGenerator:
    """
    Generates liquidity curves by connecting primary highs or lows.

    Methodology from PDFs:
    - Sell-side liquidity curve: Connect progressively lower primary highs
    - Buy-side liquidity curve: Connect progressively higher primary lows
    - Don't need all points to connect, just a few to form proper curve
    - 3-tap pattern: 1st tap → 2nd tap → 3rd tap
    """

    @staticmethod
    def generate_curves(liquidity_pools: Dict, candles: pd.DataFrame) -> Dict:
        """
        Generate liquidity curves from liquidity pools.

        Returns: Dict with sell_side_curves and buy_side_curves
        """
        bsl_pools = liquidity_pools.get("bsl_pools", [])
        ssl_pools = liquidity_pools.get("ssl_pools", [])

        # Filter primary pools only
        primary_bsl = [p for p in bsl_pools if p.get("is_primary", False)]
        primary_ssl = [p for p in ssl_pools if p.get("is_primary", False)]

        # Generate sell-side curves (from BSL - highs)
        sell_side_curves = LiquidityCurveGenerator._generate_sell_side_curves(primary_bsl, candles)

        # Generate buy-side curves (from SSL - lows)
        buy_side_curves = LiquidityCurveGenerator._generate_buy_side_curves(primary_ssl, candles)

        return {
            "sell_side_curves": sell_side_curves,
            "buy_side_curves": buy_side_curves,
            "total_curves": len(sell_side_curves) + len(buy_side_curves)
        }

    @staticmethod
    def _generate_sell_side_curves(primary_highs: List[Dict], candles: pd.DataFrame) -> List[Dict]:
        """
        Generate sell-side liquidity curves from progressively lower highs.
        """
        if len(primary_highs) < 3:
            return []

        # Sort by index
        sorted_highs = sorted(primary_highs, key=lambda x: x["idx"])

        curves = []

        # Look for sequences of 3+ lower highs
        for i in range(len(sorted_highs) - 2):
            sequence = []
            current_high = sorted_highs[i]
            sequence.append(current_high)

            for j in range(i + 1, len(sorted_highs)):
                next_high = sorted_highs[j]

                # Check if next high is lower than current
                if next_high["price"] < sequence[-1]["price"]:
                    sequence.append(next_high)

                # Stop if we found 5+ points or price goes back up
                if len(sequence) >= 5:
                    break

            # Valid curve needs at least 3 points
            if len(sequence) >= 3:
                # Check if it's a valid descending sequence
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
                        "first_tap": sequence[0],
                        "second_tap": sequence[1] if len(sequence) >= 2 else None,
                        "third_tap": sequence[2] if len(sequence) >= 3 else None,
                        "highest_price": float(sequence[0]["price"]),
                        "lowest_price": float(sequence[-1]["price"]),
                        "quality": "EXCELLENT" if tap_count >= 4 else "GOOD" if tap_count == 3 else "WEAK"
                    })

        return curves

    @staticmethod
    def _generate_buy_side_curves(primary_lows: List[Dict], candles: pd.DataFrame) -> List[Dict]:
        """
        Generate buy-side liquidity curves from progressively higher lows.
        """
        if len(primary_lows) < 3:
            return []

        # Sort by index
        sorted_lows = sorted(primary_lows, key=lambda x: x["idx"])

        curves = []

        # Look for sequences of 3+ higher lows
        for i in range(len(sorted_lows) - 2):
            sequence = []
            current_low = sorted_lows[i]
            sequence.append(current_low)

            for j in range(i + 1, len(sorted_lows)):
                next_low = sorted_lows[j]

                # Check if next low is higher than current
                if next_low["price"] > sequence[-1]["price"]:
                    sequence.append(next_low)

                # Stop if we found 5+ points or price goes back down
                if len(sequence) >= 5:
                    break

            # Valid curve needs at least 3 points
            if len(sequence) >= 3:
                # Check if it's a valid ascending sequence
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
                        "first_tap": sequence[0],
                        "second_tap": sequence[1] if len(sequence) >= 2 else None,
                        "third_tap": sequence[2] if len(sequence) >= 3 else None,
                        "lowest_price": float(sequence[0]["price"]),
                        "highest_price": float(sequence[-1]["price"]),
                        "quality": "EXCELLENT" if tap_count >= 4 else "GOOD" if tap_count == 3 else "WEAK"
                    })

        return curves

# ================================================================
# EXTREME LIQUIDITY TARGET IDENTIFICATION
# ================================================================

class ExtremeLiquidityTarget:
    """
    Identifies extreme liquidity POI (Point of Interest) after 2nd tap.

    Per PDF: "After 2nd tap observation, identify extreme liquidity target:
    - For accumulation: First higher low after 2nd tap
    - For distribution: First lower high after 2nd tap"
    """

    @staticmethod
    def identify_extreme_targets(curves: Dict, candles: pd.DataFrame) -> Dict:
        """
        Identify extreme liquidity targets for each curve.

        Returns: Dict with targets for each curve type
        """
        sell_side_targets = []
        buy_side_targets = []

        # Process sell-side curves (distribution)
        for curve in curves.get("sell_side_curves", []):
            if curve.get("tap_count", 0) >= 2:
                target = ExtremeLiquidityTarget._find_extreme_target_for_distribution(
                    curve, candles
                )
                if target:
                    sell_side_targets.append(target)

        # Process buy-side curves (accumulation)
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
        Find first lower high after 2nd tap (for distribution/sell-side).
        """
        second_tap = curve.get("second_tap")
        if not second_tap:
            return None

        second_tap_idx = second_tap["idx"]
        second_tap_price = second_tap["price"]

        # Look for first lower high after 2nd tap
        search_window = candles.iloc[second_tap_idx+1:min(second_tap_idx+20, len(candles))]

        for i, candle in enumerate(search_window.itertuples(), start=second_tap_idx+1):
            # Check if this forms a lower high
            if candle.high < second_tap_price:
                # Found first lower high
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
        Find first higher low after 2nd tap (for accumulation/buy-side).
        """
        second_tap = curve.get("second_tap")
        if not second_tap:
            return None

        second_tap_idx = second_tap["idx"]
        second_tap_price = second_tap["price"]

        # Look for first higher low after 2nd tap
        search_window = candles.iloc[second_tap_idx+1:min(second_tap_idx+20, len(candles))]

        for i, candle in enumerate(search_window.itertuples(), start=second_tap_idx+1):
            # Check if this forms a higher low
            if candle.low > second_tap_price:
                # Found first higher low
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
# LIQUIDITY VOID DETECTION
# ================================================================

class LiquidityVoidDetector:
    """
    Detects liquidity voids - areas with no S&D zones where price can move easily.

    Per PDF: Areas without order blocks or structure zones are voids
    where liquidity is sparse and price can move freely.
    """

    @staticmethod
    def detect_voids(zones: List[Dict], candles: pd.DataFrame, current_price: float) -> Dict:
        """
        Detect liquidity voids between S&D zones.

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
        "service": "HPB–TCT v21.2 (MEXC + Gate Validation + Liquidity)",
        "status": "running",
        "symbol": SYMBOL,
        "version": "21.2",
        "endpoints": {
            "/status": "Health check",
            "/api/validate": "7-gate validation",
            "/api/price": "Current price",
            "/api/zones": "Supply & Demand zones (TCT Mentorship Lecture 3)",
            "/api/liquidity": "Liquidity pools, curves & targets (TCT Lecture 4)"
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

@app.get("/api/liquidity")
async def detect_liquidity():
    """
    Detect liquidity pools, curves, and targets using TCT Lecture 4 methodology.

    Returns:
        - BSL (Buy-Side Liquidity) pools above highs
        - SSL (Sell-Side Liquidity) pools below lows
        - Liquidity curves (sell-side and buy-side)
        - Extreme liquidity targets (after 2nd tap)
        - Liquidity voids (areas with no S&D zones)
        - Integration with existing S&D zones
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

# ================================================================
# ENTRY POINT
# ================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server_mexc:app", host="0.0.0.0", port=PORT, reload=True)
