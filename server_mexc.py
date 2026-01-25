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
        "service": "HPB–TCT v21.2 (MEXC + Gate Validation)",
        "status": "running",
        "symbol": SYMBOL,
        "version": "21.2",
        "endpoints": {
            "/status": "Health check",
            "/api/validate": "7-gate validation",
            "/api/price": "Current price",
            "/api/zones": "Supply & Demand zones (TCT Mentorship)"
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
