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
from po3_schematics import detect_po3_schematics
from trade_execution import generate_execution_plan, calculate_leverage_comparison, calculate_capital_allocation
from market_structure import MarketStructure, evaluate_rtz


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

# Load leverage coin list for pair selection
COIN_LIST = []
COIN_LIST_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "leverage_coin_list_extended.txt")
try:
    with open(COIN_LIST_PATH, "r") as f:
        for line in f:
            pair = line.strip()
            # Filter out non-pair lines, dated futures contracts, and BNB (exchange-tied)
            if pair and pair.endswith("USDT") and "-" not in pair and not pair.endswith(".txt"):
                # Exclude BNB and any Binance-exchange-tied pairs
                base = pair.replace("USDT", "")
                if base.upper() in ("BNB", "BIFI", "CAKE", "XVS", "ALPACA", "BURGER", "SFP", "LINA", "TWT"):
                    continue
                COIN_LIST.append(pair)
    COIN_LIST = sorted(set(COIN_LIST))
    logger.info(f"[INIT] Loaded {len(COIN_LIST)} trading pairs from coin list")
except FileNotFoundError:
    COIN_LIST = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT"]
    logger.warning(f"[INIT] Coin list file not found, using defaults")

def resolve_symbol(symbol_param: Optional[str] = None) -> str:
    """Resolve which symbol to use: param override or global default."""
    if symbol_param:
        s = symbol_param.strip().upper().replace("/", "").replace("-", "")
        if s:
            return s
    return SYMBOL

app = FastAPI(title="HPB–TCT v21.2 MEXC Server", version="21.2")

latest_ranges = {"LTF": [], "HTF": []}
scan_interval_sec = 120

# ===== TOP 5 SETUP SCANNER STATE (pairsrange_scanner approach) =====
top_5_setups = []           # Top 5 pairs with highest RPS (Range Probability Score)
forming_setups = []         # Forming (unconfirmed) TCT models from top 5 pairs
scanner_status = {
    "last_scan": None,
    "next_scan": None,
    "pairs_scanned": 0,
    "total_pairs": 0,
    "is_scanning": False,
    "scan_duration_sec": 0,
}
SCANNER_INTERVAL_SEC = 4 * 60 * 60  # 4 hours
# Range scanner config (rebuilt using /ranges page detect_ranges logic)
RANGE_MIN_RPS = 6.5                  # minimum quality score to qualify (0-10)

logging.basicConfig(level=logging.INFO)
logger.info(f"[INIT] HPB–TCT v21.2 Ready — Symbol={SYMBOL}, Port={PORT}")

# ================================================================
# MARKET STRUCTURE  (TCT Mentorship – Lecture 1)
# ================================================================
# Rebuilt implementation now lives in market_structure.py
# MarketStructure class imported at top of file.
# See market_structure.py for full documentation of:
#   - 6-candle rule pivots (inside bar exclusion)
#   - MSH/MSL confirmation (revisit rule)
#   - BOS with Good/Bad quality classification
#   - Wick/SFP detection (3 post-wick scenarios)
#   - CHoCH via domino effect
#   - Level 1/2/3 hierarchy (direction-based)
#   - Domino Effect confirmation chain (L3 → L2 → L1)
#   - RTZ quality scoring
#   - Trend classification + trend shift detection
#   - Expectational Order Flow (EOF)
# ================================================================

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
    Detects Structure Supply/Demand zones using TCT Lecture 1 market structure.

    Structure S/D represents multiple candles with FVG following confirmed
    market structure (6-candle rule pivots, confirmed MS highs/lows, BOS).
    - Demand: Bullish structure move (confirmed HL) with FVG → zone at the HL
    - Supply: Bearish structure move (confirmed LH) with FVG → zone at the LH
    """

    @staticmethod
    def detect_structure_zones(candles: pd.DataFrame, fvgs: Dict, pivots: Dict) -> Dict:
        """
        Detect Structure Supply/Demand zones using confirmed MS points and FVGs.

        Uses TCT Lecture 1 market structure:
        - MS highs/lows (confirmed via 6-candle rule)
        - BOS events (candle CLOSE through MS level)
        - EOF (expectational order flow) for bias confirmation

        Args:
            candles: DataFrame with OHLC data
            fvgs: Dict from FairValueGap.detect_fvgs()
            pivots: Dict from MarketStructure.detect_pivots()

        Returns: Dict with demand_zones and supply_zones lists
        """
        demand_zones = []
        supply_zones = []

        ms_highs = pivots.get("ms_highs", [])
        ms_lows = pivots.get("ms_lows", [])
        bos_events = pivots.get("bos_events", [])
        trend = pivots.get("trend", "neutral")
        eof = pivots.get("eof", {})

        # --- Demand zones: FVG near confirmed MS lows (higher lows in bullish) ---
        for fvg in fvgs.get("bullish_fvgs", []):
            idx = fvg["idx"]
            if idx < 5:
                continue

            structure_start = max(0, idx - 10)
            structure_candles = candles.iloc[structure_start:idx]

            # Check confirmed MS structure: is there a confirmed MS low near this FVG?
            at_confirmed_ms_low = False
            ms_low_match = None
            for ms_l in ms_lows:
                if abs(ms_l["idx"] - idx) <= 12:
                    at_confirmed_ms_low = True
                    ms_low_match = ms_l
                    break

            # Check for higher lows in the local region (confirmed structure)
            has_bullish_structure = False
            nearby_lows = [l for l in ms_lows if structure_start <= l["idx"] <= idx + 2]
            if len(nearby_lows) >= 2:
                has_bullish_structure = nearby_lows[-1]["price"] > nearby_lows[-2]["price"]
            elif len(nearby_lows) >= 1:
                # Single confirmed MS low with bullish FVG = valid demand
                has_bullish_structure = True

            # Fallback: simple direction check if no confirmed MS lows nearby
            if not at_confirmed_ms_low and not has_bullish_structure:
                lows = structure_candles["low"].values
                if len(lows) >= 2:
                    has_bullish_structure = lows[-1] > lows[0]

            if has_bullish_structure or at_confirmed_ms_low:
                # Score: zones at confirmed MS lows with BOS are strongest
                ms_quality = "confirmed" if at_confirmed_ms_low else "inferred"
                has_bos_support = any(
                    b["type"] == "bullish" and abs(b["bos_idx"] - idx) <= 10
                    for b in bos_events
                )

                demand_zones.append({
                    "type": "demand",
                    "top": float(structure_candles["high"].max()),
                    "bottom": float(structure_candles["low"].min()),
                    "start_idx": structure_start,
                    "end_idx": idx,
                    "fvg": fvg,
                    "candle_count": len(structure_candles),
                    "mitigated": False,
                    "ms_quality": ms_quality,
                    "at_ms_low": at_confirmed_ms_low,
                    "ms_low": ms_low_match,
                    "has_bos_support": has_bos_support,
                    "eof_aligned": eof.get("bias") == "bullish",
                })

        # --- Supply zones: FVG near confirmed MS highs (lower highs in bearish) ---
        for fvg in fvgs.get("bearish_fvgs", []):
            idx = fvg["idx"]
            if idx < 5:
                continue

            structure_start = max(0, idx - 10)
            structure_candles = candles.iloc[structure_start:idx]

            # Check confirmed MS structure: is there a confirmed MS high near this FVG?
            at_confirmed_ms_high = False
            ms_high_match = None
            for ms_h in ms_highs:
                if abs(ms_h["idx"] - idx) <= 12:
                    at_confirmed_ms_high = True
                    ms_high_match = ms_h
                    break

            # Check for lower highs in the local region (confirmed structure)
            has_bearish_structure = False
            nearby_highs = [h for h in ms_highs if structure_start <= h["idx"] <= idx + 2]
            if len(nearby_highs) >= 2:
                has_bearish_structure = nearby_highs[-1]["price"] < nearby_highs[-2]["price"]
            elif len(nearby_highs) >= 1:
                has_bearish_structure = True

            # Fallback: simple direction check
            if not at_confirmed_ms_high and not has_bearish_structure:
                highs = structure_candles["high"].values
                if len(highs) >= 2:
                    has_bearish_structure = highs[-1] < highs[0]

            if has_bearish_structure or at_confirmed_ms_high:
                ms_quality = "confirmed" if at_confirmed_ms_high else "inferred"
                has_bos_support = any(
                    b["type"] == "bearish" and abs(b["bos_idx"] - idx) <= 10
                    for b in bos_events
                )

                supply_zones.append({
                    "type": "supply",
                    "top": float(structure_candles["high"].max()),
                    "bottom": float(structure_candles["low"].min()),
                    "start_idx": structure_start,
                    "end_idx": idx,
                    "fvg": fvg,
                    "candle_count": len(structure_candles),
                    "mitigated": False,
                    "ms_quality": ms_quality,
                    "at_ms_high": at_confirmed_ms_high,
                    "ms_high": ms_high_match,
                    "has_bos_support": has_bos_support,
                    "eof_aligned": eof.get("bias") == "bearish",
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
        """Check if zone is at a confirmed MS pivot point (TCT Lecture 1)."""
        tolerance = 0.005  # 0.5% tolerance

        if zone_type == "demand":
            # Prefer confirmed MS lows, fall back to raw pivots
            ms_lows = pivots.get("ms_lows", pivots.get("lows", []))
            for pivot in ms_lows[-3:]:  # Last 3 confirmed MS lows
                pivot_price = pivot.get("price", 0)
                if pivot_price > 0 and abs(zone_mid - pivot_price) / pivot_price < tolerance:
                    return True

        elif zone_type == "supply":
            # Prefer confirmed MS highs, fall back to raw pivots
            ms_highs = pivots.get("ms_highs", pivots.get("highs", []))
            for pivot in ms_highs[-3:]:  # Last 3 confirmed MS highs
                pivot_price = pivot.get("price", 0)
                if pivot_price > 0 and abs(zone_mid - pivot_price) / pivot_price < tolerance:
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

        TCT Methodology (using Lecture 1 confirmed MS points):
        - Confirmed MS highs/lows = primary liquidity (strongest)
        - 6CR pivots not confirmed as MS = internal liquidity
        - Equal highs/lows = amazing liquidity targets
        - BOS events indicate which levels have been grabbed

        Returns: Dict with bsl_pools and ssl_pools lists
        """
        bsl_pools = []  # Buy-side liquidity (above price)
        ssl_pools = []  # Sell-side liquidity (below price)

        # Use confirmed MS highs/lows as primary (TCT Lecture 1)
        ms_highs = pivots.get("ms_highs", [])
        ms_lows = pivots.get("ms_lows", [])
        raw_highs = pivots.get("highs", [])
        raw_lows = pivots.get("lows", [])
        bos_events = pivots.get("bos_events", [])

        # Confirmed MS points are primary; remaining 6CR pivots are internal
        ms_high_idxs = {h["idx"] for h in ms_highs}
        ms_low_idxs = {l["idx"] for l in ms_lows}

        primary_highs = ms_highs if ms_highs else []
        internal_highs = [h for h in raw_highs if h["idx"] not in ms_high_idxs]

        primary_lows = ms_lows if ms_lows else []
        internal_lows = [l for l in raw_lows if l["idx"] not in ms_low_idxs]

        # Fallback: if no confirmed MS points, use window-based classification
        if not primary_highs and raw_highs:
            primary_highs, internal_highs = LiquidityDetector._classify_primary_vs_internal(
                raw_highs, is_highs=True
            )
        if not primary_lows and raw_lows:
            primary_lows, internal_lows = LiquidityDetector._classify_primary_vs_internal(
                raw_lows, is_lows=True
            )

        # BOS-grabbed levels: levels that were broken by a BOS candle close
        grabbed_high_prices = {b["broken_level"] for b in bos_events if b["type"] == "bullish"}
        grabbed_low_prices = {b["broken_level"] for b in bos_events if b["type"] == "bearish"}

        # Detect equal highs (TCT: "amazing liquidity targets")
        all_high_prices = [p["price"] for p in (ms_highs or raw_highs)]
        equal_highs = LiquidityDetector._detect_equal_levels(
            all_high_prices,
            tolerance=0.0001
        )

        # Detect equal lows (TCT: "amazing liquidity targets")
        all_low_prices = [p["price"] for p in (ms_lows or raw_lows)]
        equal_lows = LiquidityDetector._detect_equal_levels(
            all_low_prices,
            tolerance=0.0001
        )

        # Create BSL pools from primary highs (confirmed MS highs)
        for pivot in primary_highs:
            price = pivot["price"]

            # Check if level was already grabbed by BOS (candle CLOSE above)
            is_grabbed = any(abs(price - gp) / price < 0.001 for gp in grabbed_high_prices)

            # TCT: Check if this is a non-liquidity high (grabbed liquidity from left)
            is_non_liquidity = LiquidityDetector._is_non_liquidity_high(
                pivot, raw_highs or primary_highs, candles
            )

            # TCT: Only use liquidity highs (not non-liquidity, not already grabbed)
            if not is_non_liquidity and not is_grabbed:
                is_equal = any(abs(price - eq) / eq < 0.0001 for eq in equal_highs)

                bsl_pools.append({
                    "type": "BSL",
                    "price": float(price),
                    "idx": int(pivot["idx"]),
                    "is_primary": True,
                    "is_equal": bool(is_equal),
                    "is_confirmed_ms": True,
                    "strength": 1.0 if is_equal else 0.85,
                    "distance_from_price": float((price - current_price) / current_price * 100)
                })

        # Create SSL pools from primary lows (confirmed MS lows)
        for pivot in primary_lows:
            price = pivot["price"]

            # Check if level was already grabbed by BOS (candle CLOSE below)
            is_grabbed = any(abs(price - gp) / price < 0.001 for gp in grabbed_low_prices)

            # TCT: Check if this is a non-liquidity low
            is_non_liquidity = LiquidityDetector._is_non_liquidity_low(
                pivot, raw_lows or primary_lows, candles
            )

            # TCT: Only use liquidity lows (not non-liquidity, not already grabbed)
            if not is_non_liquidity and not is_grabbed:
                is_equal = any(abs(price - eq) / eq < 0.0001 for eq in equal_lows)

                ssl_pools.append({
                    "type": "SSL",
                    "price": float(price),
                    "idx": int(pivot["idx"]),
                    "is_primary": True,
                    "is_equal": bool(is_equal),
                    "is_confirmed_ms": True,
                    "strength": 1.0 if is_equal else 0.85,
                    "distance_from_price": float((current_price - price) / current_price * 100)
                })

        # TCT: Add internal highs/lows as weaker liquidity (stacking up)
        # These are 6CR pivots NOT confirmed as MS highs/lows
        for pivot in internal_highs:
            price = pivot["price"]
            is_grabbed = any(abs(price - gp) / price < 0.001 for gp in grabbed_high_prices)
            if not is_grabbed:
                bsl_pools.append({
                    "type": "BSL",
                    "price": float(price),
                    "idx": int(pivot["idx"]),
                    "is_primary": False,
                    "is_equal": False,
                    "is_confirmed_ms": False,
                    "strength": 0.5,
                    "distance_from_price": float((price - current_price) / current_price * 100)
                })

        for pivot in internal_lows:
            price = pivot["price"]
            is_grabbed = any(abs(price - gp) / price < 0.001 for gp in grabbed_low_prices)
            if not is_grabbed:
                ssl_pools.append({
                    "type": "SSL",
                    "price": float(price),
                    "idx": int(pivot["idx"]),
                    "is_primary": False,
                    "is_equal": False,
                    "is_confirmed_ms": False,
                    "strength": 0.5,
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

# Shared httpx client — reuses connections instead of opening one per request
_shared_client: Optional[httpx.AsyncClient] = None
_scanner_consecutive_errors = 0  # Track consecutive fetch errors for adaptive backoff

async def _get_shared_client() -> httpx.AsyncClient:
    global _shared_client
    if _shared_client is None or _shared_client.is_closed:
        _shared_client = httpx.AsyncClient(
            timeout=15,
            limits=httpx.Limits(max_connections=5, max_keepalive_connections=3),
        )
    return _shared_client

async def fetch_mexc_candles(symbol: str, interval: str, limit: int = 200) -> Optional[pd.DataFrame]:
    global _scanner_consecutive_errors
    # Normalize intervals to MEXC-supported values (MEXC spot uses 60m not 1h, etc.)
    _MEXC_INTERVAL_MAP = {"1h": "60m", "2h": "4h"}
    interval = _MEXC_INTERVAL_MAP.get(interval, interval)
    url = f"{MEXC_URL_BASE}/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}

    try:
        client = await _get_shared_client()
        r = await client.get(url, params=params)

        if r.status_code == 429:
            # Rate limited — back off significantly
            _scanner_consecutive_errors += 1
            logger.warning(f"[MEXC] Rate limited (429) for {symbol}/{interval}, backing off")
            await asyncio.sleep(min(5 * _scanner_consecutive_errors, 30))
            return None

        if r.status_code != 200:
            _scanner_consecutive_errors += 1
            return None

        _scanner_consecutive_errors = max(0, _scanner_consecutive_errors - 1)  # Decay on success
        data = r.json()
        if not data or not isinstance(data, list):
            return None

        df = pd.DataFrame(data, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_vol"
        ])
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = df[c].astype(float)
        return df
    except Exception as e:
        _scanner_consecutive_errors += 1
        logger.error(f"[MEXC_FETCH_ERROR] {symbol}/{interval}: {e}")
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
# TOP 5 RANGE SCANNER (rebuilt using /ranges page detect_ranges logic)
# Uses the same sliding-window consolidation detection, quality scoring,
# touch counting, and EQ-touched checks as the /ranges debug page.
# Excludes BNB-exchange pairs and pairs listed less than 1 year.
# ================================================================

PAIR_MIN_AGE_DAYS = 365  # Minimum pair listing age to qualify


def score_range_quality(range_info: Dict) -> float:
    """
    Score a detected range (0–10) using the /ranges page quality metrics.
    Considers: quality rating, touch counts, consolidation ratio, EQ touched,
    candle count (duration), and range size relative to price.
    """
    score = 0.0

    # Quality base score (from detect_ranges quality assessment)
    quality = range_info.get("quality", "bad")
    if quality == "good":
        score += 5.0
    elif quality == "moderate":
        score += 3.0
    else:
        score += 1.0

    # EQ touched bonus — confirmed range per TCT rules
    if range_info.get("eq_touched", False):
        score += 1.5

    # Consolidation ratio bonus (lower = tighter range = better)
    # consolidation_ratio is avg_candle_size / range_size
    # We don't store it directly, but candles and range_size tell us the duration
    candles = range_info.get("candles", 0)
    if candles >= 40:
        score += 1.5  # Long duration range
    elif candles >= 20:
        score += 1.0

    # Range size as % of price — meaningful ranges get bonus
    range_size = range_info.get("range_size", 0)
    range_high = range_info.get("high", 1)
    if range_high > 0:
        range_pct = (range_size / range_high) * 100
        if 2.0 <= range_pct <= 15.0:
            score += 1.0  # Sweet spot — not too tight, not too wide

    return round(min(score, 10.0), 2)


async def scan_pair_range(symbol: str) -> Optional[Dict]:
    """
    Scan a single pair using the same detect_ranges() logic as the /ranges page.
    Fetches 1D candles, runs sliding-window range detection, scores the best range.
    Also enforces minimum pair age (1 year) by checking candle history.
    Returns a setup dict if score qualifies, or None.
    """
    try:
        # Fetch 1D candles — request 400 to check pair age (365+ days)
        df = await fetch_mexc_candles(symbol, "1d", 400)
        if df is None or len(df) < 30:
            return None

        # Pair age filter: earliest candle must be >= 365 days ago
        earliest_candle = df["open_time"].iloc[0]
        latest_candle = df["open_time"].iloc[-1]
        pair_age_days = (latest_candle - earliest_candle).total_seconds() / 86400
        if pair_age_days < PAIR_MIN_AGE_DAYS:
            return None

        current_price = float(df.iloc[-1]["close"])

        # Use the same detect_ranges() function as the /ranges page
        all_ranges = detect_ranges(df, lookback=len(df))
        if not all_ranges:
            return None

        # Score each detected range and pick the best
        best_range = None
        best_score = 0.0

        for r in all_ranges:
            rps = score_range_quality(r)
            if rps > best_score:
                best_score = rps
                best_range = r

        if not best_range or best_score < RANGE_MIN_RPS:
            return None

        return {
            "symbol": symbol,
            "timeframe": "1d",
            "RPS": best_score,
            "range_high": round(best_range["high"], 6),
            "range_low": round(best_range["low"], 6),
            "range_eq": round(best_range["equilibrium"], 6),
            "range_size": round(best_range["range_size"], 6),
            "deviation_limit": round(best_range["deviation_limit"], 6),
            "upper_dl": round(best_range["upper_dl"], 6),
            "lower_dl": round(best_range["lower_dl"], 6),
            "quality": best_range["quality"],
            "eq_touched": best_range["eq_touched"],
            "candles": best_range["candles"],
            "trend_before": best_range.get("trend_before", "neutral"),
            "current_price": round(current_price, 6),
            "pair_age_days": round(pair_age_days),
        }

    except Exception as e:
        logger.debug(f"[RANGE_SCANNER] Error scanning {symbol}: {e}")
        return None


async def run_full_scan():
    """
    Scan all pairs for high-probability ranges using the /ranges page detect_ranges logic.
    Fetches 1D candles, runs sliding-window range detection, scores best range per pair.
    Excludes BNB-exchange pairs and pairs under 1 year old.
    Runs as a background task every 4 hours.
    """
    global top_5_setups

    if scanner_status["is_scanning"]:
        logger.info("[RANGE_SCANNER] Scan already in progress, skipping")
        return

    scanner_status["is_scanning"] = True
    scanner_status["total_pairs"] = len(COIN_LIST)
    scanner_status["pairs_scanned"] = 0
    start_time = datetime.utcnow()

    logger.info(f"[RANGE_SCANNER] Starting range scan of {len(COIN_LIST)} pairs (1D timeframe, detect_ranges logic, score >= {RANGE_MIN_RPS})")

    all_qualified = []

    global _scanner_consecutive_errors
    _scanner_consecutive_errors = 0  # Reset at scan start

    for i, symbol in enumerate(COIN_LIST):
        scanner_status["pairs_scanned"] = i + 1

        try:
            result = await scan_pair_range(symbol)
            if result:
                all_qualified.append(result)
        except Exception as e:
            logger.debug(f"[RANGE_SCANNER] Exception for {symbol}: {e}")

        # Adaptive rate limiting
        base_delay = 0.35
        if _scanner_consecutive_errors > 10:
            delay = min(base_delay + (_scanner_consecutive_errors * 0.5), 10.0)
            logger.info(f"[RANGE_SCANNER] Throttling: {delay:.1f}s delay (errors={_scanner_consecutive_errors})")
        elif _scanner_consecutive_errors > 3:
            delay = base_delay + (_scanner_consecutive_errors * 0.2)
        else:
            delay = base_delay
        await asyncio.sleep(delay)

        # Log progress every 50 pairs
        if (i + 1) % 50 == 0:
            logger.info(f"[RANGE_SCANNER] Progress: {i + 1}/{len(COIN_LIST)} pairs, {len(all_qualified)} qualified")

    # Sort by RPS (highest first) and take top 5
    all_qualified.sort(key=lambda x: x["RPS"], reverse=True)
    if all_qualified:
        top_5_setups = all_qualified[:5]

    end_time = datetime.utcnow()
    duration = (end_time - start_time).total_seconds()

    scanner_status["is_scanning"] = False
    scanner_status["last_scan"] = end_time.isoformat()
    scanner_status["scan_duration_sec"] = round(duration)
    scanner_status["next_scan"] = (end_time + pd.Timedelta(seconds=SCANNER_INTERVAL_SEC)).isoformat()

    kept = " (kept previous)" if not all_qualified and top_5_setups else ""
    logger.info(
        f"[RANGE_SCANNER] Scan complete in {duration:.0f}s — "
        f"{len(COIN_LIST)} pairs, {len(all_qualified)} qualified (RPS >= {RANGE_MIN_RPS}){kept}, "
        f"Top 5: {[s['symbol'] + ' RPS=' + str(s['RPS']) for s in top_5_setups]}"
    )

    # After main scan, scan top 5 pairs for forming (unconfirmed) models
    asyncio.create_task(scan_forming_setups())


async def scan_pair_for_forming(symbol: str, timeframe: str) -> List[Dict]:
    """
    Scan a single pair/timeframe for ALL schematics including forming (unconfirmed) ones.
    Returns list of forming schematic dicts with full tap/range data for charting.
    """
    try:
        df = await fetch_mexc_candles(symbol, timeframe, 200)
        if df is None or len(df) < 30:
            return []

        current_price = float(df.iloc[-1]["close"])

        candles_list = []
        for _, row in df.iterrows():
            candles_list.append({
                'open_time': str(row['open_time']),
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': float(row['volume'])
            })

        detected_range = await detect_best_range(candles_list)
        range_list = [detected_range] if detected_range and not isinstance(detected_range, list) else (detected_range or [])

        schematics_result = detect_tct_schematics(df, range_list)

        all_schematics = (
            schematics_result.get("accumulation_schematics", []) +
            schematics_result.get("distribution_schematics", [])
        )

        forming = []
        for s in all_schematics:
            if not isinstance(s, dict):
                continue

            # Include both forming AND confirmed schematics
            entry = s.get('entry', {}).get('price')
            target = s.get('target', {}).get('price')
            stop = s.get('stop_loss', {}).get('price')
            if not entry or not target or not stop:
                continue

            # Check setup is still relevant (price hasn't blown past levels)
            direction = s.get('direction', '')
            if direction == 'bullish' and current_price >= target:
                continue
            if direction == 'bearish' and current_price <= target:
                continue

            model = s.get('model', '')
            schematic_type = s.get('schematic_type', '')
            if 'Model_2' in (model or '') or 'model_2' in schematic_type:
                model_label = 'M2'
            else:
                model_label = 'M1'

            if direction == 'bullish':
                setup_type = f"{model_label} Accumulation"
            elif direction == 'bearish':
                setup_type = f"{model_label} Distribution"
            else:
                setup_type = schematic_type.replace('_', ' ')

            forming.append({
                "symbol": symbol,
                "timeframe": timeframe,
                "setup_type": setup_type,
                "model": model_label,
                "direction": direction,
                "is_confirmed": s.get('is_confirmed', False),
                "status": "confirmed" if s.get('is_confirmed', False) else "forming",
                "entry": round(entry, 8),
                "stop_loss": round(stop, 8),
                "target": round(target, 8),
                "risk_reward": round(s.get('risk_reward', 0), 2),
                "quality_score": round(s.get('quality_score', 0), 3),
                "current_price": round(current_price, 8),
                # Tap data for chart overlays
                "tap1": s.get('tap1'),
                "tap2": s.get('tap2'),
                "tap3": s.get('tap3'),
                "range": s.get('range'),
                "bos_confirmation": s.get('bos_confirmation'),
                "wyckoff_high": s.get('wyckoff_high'),
                "wyckoff_low": s.get('wyckoff_low'),
            })

        return forming

    except Exception as e:
        logger.debug(f"[FORMING] Error scanning {symbol}/{timeframe}: {e}")
        return []


async def scan_forming_setups():
    """Scan the current top 5 setup pairs for forming (unconfirmed) TCT models."""
    global forming_setups

    if not top_5_setups:
        return

    logger.info(f"[FORMING] Scanning top 5 pairs for forming TCT models...")

    all_forming = []
    seen_pairs = set()

    for setup in top_5_setups:
        symbol = setup["symbol"]
        timeframe = setup["timeframe"]
        pair_key = f"{symbol}/{timeframe}"

        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)

        results = await scan_pair_for_forming(symbol, timeframe)
        all_forming.extend(results)
        await asyncio.sleep(0.2)

    # Sort: forming first, then by quality score descending
    all_forming.sort(key=lambda x: (x["is_confirmed"], -x["quality_score"]))

    forming_setups = all_forming
    logger.info(f"[FORMING] Found {len(forming_setups)} schematics ({sum(1 for f in forming_setups if not f['is_confirmed'])} forming, {sum(1 for f in forming_setups if f['is_confirmed'])} confirmed)")


async def scanner_loop():
    """Background loop that runs the scanner every 4 hours."""
    # Initial scan after a short delay to let the server start
    await asyncio.sleep(10)
    await run_full_scan()

    while True:
        await asyncio.sleep(SCANNER_INTERVAL_SEC)
        await run_full_scan()


@app.on_event("startup")
async def startup_event():
    """Start the background scanner on server startup."""
    asyncio.create_task(scanner_loop())
    logger.info(f"[SCANNER] Background scanner started — interval: {SCANNER_INTERVAL_SEC}s ({SCANNER_INTERVAL_SEC // 3600}h)")


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up shared httpx client on shutdown."""
    global _shared_client
    if _shared_client and not _shared_client.is_closed:
        await _shared_client.aclose()
        _shared_client = None


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
            "lecture_4": "Liquidity (LiquidityDetector, LiquidityCurveGenerator, ExtremeLiquidityTarget)",
            "lecture_7": "Risk Management (Position Sizing, Leverage, Compounding, Equity Simulation)",
            "lecture_8": "PO3 Schematics (Power of Three: Range → Manipulation → Expansion)",
            "lecture_9": "Trade Execution (Position Sizing, Leverage Safety, Partial TPs, Trailing SL)"
        }
    }

@app.get("/status")
async def get_status():
    return {"status": "OK", "symbol": SYMBOL, "timestamp": datetime.utcnow().isoformat()}

@app.get("/api/price")
async def live_price(symbol: Optional[str] = None):
    sym = resolve_symbol(symbol)
    url = f"{MEXC_URL_BASE}/api/v3/ticker/price?symbol={sym}"
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.get(url)
            if r.status_code == 200:
                return {"symbol": sym, "price": float(r.json()["price"])}
            return {"error": f"HTTP {r.status_code}"}
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/coin-list")
async def get_coin_list():
    """Return the full list of available trading pairs, categorized."""
    # Major pairs for quick access
    majors = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT",
              "DOGEUSDT", "AVAXUSDT", "DOTUSDT", "LINKUSDT", "LTCUSDT", "MATICUSDT"]
    # DeFi
    defi = ["AAVEUSDT", "UNIUSDT", "MKRUSDT", "COMPUSDT", "CRVUSDT", "SNXUSDT",
            "LDOUSDT", "DYDXUSDT", "GMXUSDT", "PENDLEUSDT", "JUPUSDT", "RAYDIUMUSDT"]
    # Layer 1/2
    layer = ["APTUSDT", "SEIUSDT", "SUIUSDT", "NEARUSDT", "ATOMUSDT", "ICPUSDT",
             "INJUSDT", "TIAUSDT", "MANTAUSDT", "ARBUSDT", "OPUSDT", "STRKUSDT"]
    # Meme
    meme = ["1000PEPEUSDT", "1000FLOKIUSDT", "1000BONKUSDT", "SHIBUSDT", "WIFUSDT",
            "BOMEUSDT", "MEWUSDT", "POPCATUSDT", "GOATUSDT", "PNUTUSDT", "FARTCOINUSDT"]
    # AI
    ai = ["RENDERUSDT", "FETUSDT", "AGIXUSDT", "OCEANUSDT", "AIXBTUSDT", "TAOBTCUSDT",
          "GRASSUSDT", "CGPTUSDT", "AIOZUSDT"]

    return {
        "total": len(COIN_LIST),
        "categories": {
            "majors": [p for p in majors if p in COIN_LIST],
            "defi": [p for p in defi if p in COIN_LIST],
            "layer_1_2": [p for p in layer if p in COIN_LIST],
            "meme": [p for p in meme if p in COIN_LIST],
            "ai": [p for p in ai if p in COIN_LIST],
        },
        "all": COIN_LIST
    }

@app.get("/api/top-setups")
async def get_top_setups():
    """
    Return the top 5 pairs with the highest range quality score.
    Uses the same detect_ranges() sliding-window logic as the /ranges page.
    Excludes BNB-exchange pairs and pairs listed less than 1 year.
    Scanned every 4 hours across all pairs on 1D timeframe.
    """
    return {
        "top_setups": top_5_setups,
        "scanner_status": scanner_status,
        "scan_method": "detect_ranges (ranges-page logic)",
        "score_threshold": RANGE_MIN_RPS,
        "total_pairs_in_list": len(COIN_LIST),
    }

@app.get("/api/forming-setups")
async def get_forming_setups():
    """Return forming (unconfirmed) TCT models from top 5 setup pairs."""
    return {
        "forming_setups": forming_setups,
        "total": len(forming_setups),
        "forming_count": sum(1 for f in forming_setups if not f["is_confirmed"]),
        "confirmed_count": sum(1 for f in forming_setups if f["is_confirmed"]),
    }


@app.get("/api/schematic-data")
async def get_schematic_data(symbol: str, timeframe: str = "4h", type: str = "tct"):
    """
    Return full schematic + candle data for the schematic chart page.
    Fetches fresh candles and runs schematic detection with full tap/range data.
    type='tct' for TCT schematics, type='po3' for PO3 schematics.
    """
    try:
        df = await fetch_mexc_candles(symbol, timeframe, 200)
        if df is None or len(df) < 30:
            return {"error": "Insufficient candle data", "symbol": symbol}

        current_price = float(df.iloc[-1]["close"])

        candles_list = []
        candles_json = []
        for _, row in df.iterrows():
            candle = {
                'open_time': str(row['open_time']),
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': float(row['volume'])
            }
            candles_list.append(candle)
            # LightweightCharts format
            ts = row['open_time']
            if hasattr(ts, 'timestamp'):
                epoch = int(ts.timestamp())
            else:
                epoch = int(pd.Timestamp(ts).timestamp())
            candles_json.append({
                'time': epoch,
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
            })

        detected_range = await detect_best_range(candles_list)
        range_list = [detected_range] if detected_range and not isinstance(detected_range, list) else (detected_range or [])

        if type == "po3":
            # PO3 schematic detection
            po3_result = detect_po3_schematics(df, range_list)
            all_po3 = po3_result.get("bullish_po3", []) + po3_result.get("bearish_po3", [])

            # Convert PO3 range indices to timestamps
            po3_with_timestamps = []
            for p in all_po3:
                if not isinstance(p, dict):
                    continue
                po3 = dict(p)
                ri = po3.get("range_indices", {})
                for idx_key in ["range_start", "range_end", "manipulation_start", "manipulation_end"]:
                    idx = ri.get(idx_key)
                    if idx is not None and 0 <= idx < len(candles_json):
                        ri[idx_key + "_time"] = candles_json[idx]["time"]
                po3_with_timestamps.append(po3)

            return convert_numpy_types({
                "symbol": symbol,
                "timeframe": timeframe,
                "type": "po3",
                "current_price": current_price,
                "candles": candles_json,
                "po3_schematics": po3_with_timestamps,
                "ranges": range_list,
            })
        else:
            # TCT schematic detection
            schematics_result = detect_tct_schematics(df, range_list)

            all_schematics = (
                schematics_result.get("accumulation_schematics", []) +
                schematics_result.get("distribution_schematics", [])
            )

            # Convert schematic tap indices to timestamps for chart overlay
            schematics_with_timestamps = []
            for s in all_schematics:
                if not isinstance(s, dict):
                    continue
                sch = dict(s)
                # Map tap indices to candle timestamps
                for tap_key in ['tap1', 'tap2', 'tap3']:
                    tap = sch.get(tap_key)
                    if tap and isinstance(tap, dict) and 'idx' in tap:
                        idx = int(tap['idx'])
                        if 0 <= idx < len(candles_json):
                            tap['time'] = candles_json[idx]['time']
                # Map BOS index
                bos = sch.get('bos_confirmation')
                if bos and isinstance(bos, dict) and 'bos_idx' in bos:
                    idx = int(bos['bos_idx'])
                    if 0 <= idx < len(candles_json):
                        bos['bos_time'] = candles_json[idx]['time']
                schematics_with_timestamps.append(sch)

            return convert_numpy_types({
                "symbol": symbol,
                "timeframe": timeframe,
                "type": "tct",
                "current_price": current_price,
                "candles": candles_json,
                "schematics": schematics_with_timestamps,
                "ranges": range_list,
            })

    except Exception as e:
        logger.error(f"[SCHEMATIC-DATA] Error for {symbol}/{timeframe}: {e}")
        return {"error": str(e), "symbol": symbol}


@app.get("/schematic-chart", response_class=HTMLResponse)
async def schematic_chart_page(symbol: str = "BTCUSDT", timeframe: str = "4h", type: str = "tct"):
    """
    Dedicated schematic chart page showing TCT model overlays (type=tct) or PO3 overlays (type=po3).
    Range boxes, S/D zones, tap circles, BOS markers, deviation limits, PO3 phases.
    """
    chart_type = type  # tct or po3
    html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>""" + ("PO3" if chart_type == "po3" else "TCT") + """ Schematic — """ + symbol + """ """ + timeframe.upper() + """</title>
    <script src="https://unpkg.com/lightweight-charts@4.1.0/dist/lightweight-charts.standalone.production.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { background: #0a0a12; color: #e0e0e0; font-family: 'Segoe UI', sans-serif; overflow: hidden; }
        .header {
            display: flex; align-items: center; justify-content: space-between;
            padding: 10px 20px; background: #12121a; border-bottom: 1px solid #1e1e2d;
        }
        .header h1 { font-size: 1.1rem; color: #e040fb; }
        .header h1 span.pair { color: #00d4ff; }
        .header h1 span.tf { color: #888; font-weight: 400; font-size: 0.9rem; margin-left: 8px; }
        .back-link { color: #888; text-decoration: none; font-size: 0.8rem; padding: 4px 12px; border: 1px solid #333; border-radius: 4px; }
        .back-link:hover { color: #e0e0e0; border-color: #555; }
        .chart-container { width: 100%; height: calc(100vh - 95px); }
        .legend {
            display: flex; gap: 16px; padding: 6px 20px; background: #12121a;
            border-top: 1px solid #1e1e2d; font-size: 0.7rem; color: #888; flex-wrap: wrap;
        }
        .legend-item { display: flex; align-items: center; gap: 4px; }
        .legend-dot { width: 10px; height: 10px; border-radius: 50%; }
        .legend-line { width: 16px; height: 2px; }
        .legend-box { width: 14px; height: 10px; border-radius: 2px; }
        .schematic-cards {
            position: absolute; top: 50px; right: 10px; z-index: 10;
            display: flex; flex-direction: column; gap: 6px; max-height: calc(100vh - 110px);
            overflow-y: auto; padding: 4px;
        }
        .schematic-card {
            background: rgba(18, 18, 26, 0.95); border: 1px solid #2d2d44;
            border-radius: 6px; padding: 8px 12px; min-width: 220px;
            font-size: 0.7rem; backdrop-filter: blur(8px);
        }
        .schematic-card.forming { border-left: 3px solid #ffc107; }
        .schematic-card.confirmed { border-left: 3px solid #00ff88; }
        .sc-title { font-weight: 700; margin-bottom: 4px; }
        .sc-title.bullish { color: #00ff88; }
        .sc-title.bearish { color: #ff4444; }
        .sc-status { font-size: 0.6rem; padding: 1px 6px; border-radius: 3px; margin-left: 6px; }
        .sc-status.forming { background: rgba(255,193,7,0.2); color: #ffc107; }
        .sc-status.confirmed { background: rgba(0,255,136,0.2); color: #00ff88; }
        .sc-row { display: flex; justify-content: space-between; color: #888; margin: 2px 0; }
        .sc-row .val { color: #e0e0e0; font-weight: 600; }
        .sc-taps { display: flex; gap: 6px; margin-top: 4px; }
        .sc-tap { font-size: 0.6rem; padding: 2px 5px; border-radius: 3px; background: rgba(255,255,255,0.05); }
        .sc-tap.deviation { background: rgba(224,64,251,0.15); color: #e040fb; }
    </style>
</head>
<body>
    <div class="header">
        <h1>TCT Schematic <span class="pair">""" + symbol.replace("USDT", "/USDT") + """</span><span class="tf">""" + timeframe.upper() + """</span></h1>
        <a href="/dashboard" class="back-link">Back to Dashboard</a>
    </div>
    <div class="chart-container" id="chartContainer"></div>
    <div class="schematic-cards" id="schematicCards"></div>
    <div class="legend">
        <div class="legend-item"><div class="legend-box" style="background:rgba(128,128,128,0.2);border:1px solid #888;"></div> Range</div>
        <div class="legend-item"><div class="legend-box" style="background:rgba(255,68,68,0.2);border:1px solid #ff4444;"></div> Supply Zone</div>
        <div class="legend-item"><div class="legend-box" style="background:rgba(120,80,255,0.2);border:1px solid #7850ff;"></div> Demand Zone</div>
        <div class="legend-item"><div class="legend-dot" style="background:#e040fb;"></div> Tap (Deviation)</div>
        <div class="legend-item"><div class="legend-dot" style="background:#00d4ff;"></div> Tap (Range)</div>
        <div class="legend-item"><div class="legend-line" style="background:#ffc107;"></div> Deviation Limit</div>
        <div class="legend-item"><div class="legend-line" style="background:#00ff88;"></div> BOS</div>
        <div class="legend-item"><div class="legend-line" style="background:#00d4ff;"></div> Entry</div>
        <div class="legend-item"><div class="legend-line" style="background:#ff4444;"></div> Stop Loss</div>
    </div>

    <script>
        const SYMBOL = '""" + symbol + """';
        const TIMEFRAME = '""" + timeframe + """';
        const CHART_TYPE = '""" + chart_type + """';

        let chart, candleSeries;
        const overlays = [];

        function initChart() {
            const container = document.getElementById('chartContainer');
            chart = LightweightCharts.createChart(container, {
                width: container.clientWidth,
                height: container.clientHeight,
                layout: { background: { color: '#0a0a12' }, textColor: '#888' },
                grid: { vertLines: { color: '#1a1a2a' }, horzLines: { color: '#1a1a2a' } },
                crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
                rightPriceScale: { borderColor: '#2d2d44' },
                timeScale: { borderColor: '#2d2d44', timeVisible: true },
            });

            candleSeries = chart.addCandlestickSeries({
                upColor: '#00ff88', downColor: '#ff4444',
                borderUpColor: '#00ff88', borderDownColor: '#ff4444',
                wickUpColor: '#00ff88', wickDownColor: '#ff4444',
            });

            window.addEventListener('resize', () => {
                chart.applyOptions({ width: container.clientWidth, height: container.clientHeight });
            });
        }

        function fmt(p) {
            if (p == null || isNaN(p)) return '--';
            if (p >= 10000) return '$' + p.toLocaleString(undefined, {maximumFractionDigits: 0});
            if (p >= 100)   return '$' + p.toLocaleString(undefined, {maximumFractionDigits: 2});
            if (p >= 1)     return '$' + p.toFixed(2);
            if (p >= 0.01)  return '$' + p.toFixed(4);
            return '$' + p.toPrecision(4);
        }

        function drawHLine(price, color, title, style, width) {
            return candleSeries.createPriceLine({
                price: price, color: color, lineWidth: width || 1,
                lineStyle: style || LightweightCharts.LineStyle.Solid,
                axisLabelVisible: true, title: title || '',
            });
        }

        function drawZone(high, low, candles, color, startIdx) {
            if (!candles || candles.length < 2) return;
            const si = startIdx || 0;
            const startTime = candles[si].time;
            const endTime = candles[candles.length - 1].time;
            // Extend zone into future
            const lastInterval = candles.length > 1 ? candles[candles.length-1].time - candles[candles.length-2].time : 3600;
            const futureTime = endTime + lastInterval * 20;

            const topSeries = chart.addAreaSeries({
                topColor: color.replace(')', ',0.25)').replace('rgb', 'rgba'),
                bottomColor: color.replace(')', ',0.08)').replace('rgb', 'rgba'),
                lineColor: color.replace(')', ',0.4)').replace('rgb', 'rgba'),
                lineWidth: 1, priceScaleId: 'right',
                lastValueVisible: false, crosshairMarkerVisible: false,
            });
            topSeries.setData([
                { time: startTime, value: high },
                { time: futureTime, value: high },
            ]);
            overlays.push(topSeries);

            const botSeries = chart.addAreaSeries({
                topColor: color.replace(')', ',0.08)').replace('rgb', 'rgba'),
                bottomColor: color.replace(')', ',0.02)').replace('rgb', 'rgba'),
                lineColor: color.replace(')', ',0.4)').replace('rgb', 'rgba'),
                lineWidth: 1, priceScaleId: 'right',
                lastValueVisible: false, crosshairMarkerVisible: false,
            });
            botSeries.setData([
                { time: startTime, value: low },
                { time: futureTime, value: low },
            ]);
            overlays.push(botSeries);
        }

        function drawSchematics(data) {
            const candles = data.candles;
            const schematics = data.schematics || [];
            const cardsEl = document.getElementById('schematicCards');

            if (schematics.length === 0) {
                cardsEl.innerHTML = '<div class="schematic-card"><div class="sc-title" style="color:#888;">No schematics detected on this timeframe</div></div>';
                return;
            }

            let cardsHTML = '';

            schematics.forEach((s, idx) => {
                const isConfirmed = s.is_confirmed;
                const statusCls = isConfirmed ? 'confirmed' : 'forming';
                const statusText = isConfirmed ? 'CONFIRMED' : 'FORMING';
                const direction = s.direction || 'unknown';
                const dirCls = direction === 'bullish' ? 'bullish' : 'bearish';
                const typeLabel = (s.schematic_type || '').replace(/_/g, ' ').replace(/\\b\\w/g, c => c.toUpperCase());

                const entry = s.entry?.price;
                const stop = s.stop_loss?.price;
                const target = s.target?.price;
                const rr = s.risk_reward;
                const quality = s.quality_score;

                // --- Draw range box ---
                const range = s.range;
                if (range && range.high && range.low) {
                    const rangeStartIdx = Math.max(0, (s.tap1?.idx || 0) - 5);
                    drawZone(range.high, range.low, candles, 'rgb(128,128,128)', rangeStartIdx);

                    // Deviation limits (DL)
                    if (range.dl_high) {
                        drawHLine(range.dl_high, 'rgba(255,193,7,0.5)', 'DL High', LightweightCharts.LineStyle.Dotted, 1);
                    }
                    if (range.dl_low) {
                        drawHLine(range.dl_low, 'rgba(255,193,7,0.5)', 'DL Low', LightweightCharts.LineStyle.Dotted, 1);
                    }
                    // Equilibrium
                    if (range.equilibrium) {
                        drawHLine(range.equilibrium, 'rgba(128,128,128,0.4)', 'EQ', LightweightCharts.LineStyle.Dotted, 1);
                    }
                }

                // --- Draw supply/demand zone at target ---
                if (direction === 'bullish' && target && range) {
                    // Supply zone at top (target area)
                    const zoneSize = (range.high - range.low) * 0.08;
                    drawZone(target + zoneSize, target - zoneSize, candles, 'rgb(255,68,68)', Math.max(0, (s.tap1?.idx || 0)));
                } else if (direction === 'bearish' && target && range) {
                    // Demand zone at bottom (target area)
                    const zoneSize = (range.high - range.low) * 0.08;
                    drawZone(target + zoneSize, target - zoneSize, candles, 'rgb(120,80,255)', Math.max(0, (s.tap1?.idx || 0)));
                }

                // --- Draw tap markers ---
                const markers = [];
                [s.tap1, s.tap2, s.tap3].forEach((tap, ti) => {
                    if (tap && tap.time) {
                        const isDev = tap.is_deviation;
                        const tapNum = ti + 1;
                        const isLow = direction === 'bullish';
                        markers.push({
                            time: tap.time,
                            position: isLow ? 'belowBar' : 'aboveBar',
                            color: isDev ? '#e040fb' : '#00d4ff',
                            shape: 'circle',
                            text: 'T' + tapNum + (isDev ? ' (Dev)' : ''),
                        });
                    }
                });

                // --- BOS marker ---
                const bos = s.bos_confirmation;
                if (bos && bos.bos_time) {
                    markers.push({
                        time: bos.bos_time,
                        position: direction === 'bullish' ? 'aboveBar' : 'belowBar',
                        color: '#00ff88',
                        shape: 'arrowUp',
                        text: 'BOS',
                    });
                }

                // Add markers to chart (only first schematic to avoid clutter)
                if (idx === 0 && markers.length > 0) {
                    candleSeries.setMarkers(markers);
                }

                // --- Draw entry/stop/target lines ---
                if (entry) drawHLine(entry, '#00d4ff', 'Entry', LightweightCharts.LineStyle.Solid, 2);
                if (stop) drawHLine(stop, '#ff4444', 'Stop Loss', LightweightCharts.LineStyle.Dashed, 1);
                if (target) drawHLine(target, '#00ff88', 'Target', LightweightCharts.LineStyle.Dashed, 1);

                // --- Build card HTML ---
                cardsHTML += '<div class="schematic-card ' + statusCls + '">';
                cardsHTML += '<div class="sc-title ' + dirCls + '">' + typeLabel;
                cardsHTML += '<span class="sc-status ' + statusCls + '">' + statusText + '</span></div>';

                if (entry) cardsHTML += '<div class="sc-row"><span>Entry</span><span class="val">' + fmt(entry) + '</span></div>';
                if (stop) cardsHTML += '<div class="sc-row"><span>Stop Loss</span><span class="val">' + fmt(stop) + '</span></div>';
                if (target) cardsHTML += '<div class="sc-row"><span>Target</span><span class="val">' + fmt(target) + '</span></div>';
                if (rr) cardsHTML += '<div class="sc-row"><span>R:R</span><span class="val">' + rr.toFixed(1) + '</span></div>';
                if (quality) cardsHTML += '<div class="sc-row"><span>Quality</span><span class="val">' + Math.round(quality * 100) + '%</span></div>';

                // Tap indicators
                cardsHTML += '<div class="sc-taps">';
                [s.tap1, s.tap2, s.tap3].forEach((tap, ti) => {
                    if (tap) {
                        const cls = tap.is_deviation ? 'deviation' : '';
                        cardsHTML += '<span class="sc-tap ' + cls + '">T' + (ti+1) + ': ' + fmt(tap.price) + '</span>';
                    }
                });
                cardsHTML += '</div>';

                cardsHTML += '</div>';
            });

            cardsEl.innerHTML = cardsHTML;
        }

        function drawPO3Schematics(data) {
            const candles = data.candles;
            const po3List = data.po3_schematics || [];
            const cardsEl = document.getElementById('schematicCards');

            if (po3List.length === 0) {
                cardsEl.innerHTML = '<div class="schematic-card"><div class="sc-title" style="color:#888;">No PO3 schematics detected on this timeframe</div></div>';
                return;
            }

            let cardsHTML = '';
            const allMarkers = [];

            po3List.forEach((p, idx) => {
                const isBull = p.direction === 'bullish';
                const dirCls = isBull ? 'bullish' : 'bearish';
                const dirLabel = isBull ? 'Bullish PO3' : 'Bearish PO3';
                const phase = (p.phase || 'range').replace(/_/g, ' ');
                const entry = p.entry?.price;
                const stop = p.stop_loss?.price;
                const target = p.target?.price;
                const rr = p.risk_reward;
                const quality = p.quality_score;
                const ri = p.range_indices || {};
                const rangeInfo = p.range || {};

                // Draw range box
                if (rangeInfo.high && rangeInfo.low) {
                    const startIdx = ri.range_start || 0;
                    drawZone(rangeInfo.high, rangeInfo.low, candles, 'rgb(128,128,128)', startIdx);

                    // Equilibrium line
                    if (rangeInfo.equilibrium) {
                        drawHLine(rangeInfo.equilibrium, 'rgba(128,128,128,0.4)', 'EQ', LightweightCharts.LineStyle.Dotted, 1);
                    }
                }

                // Draw manipulation zone
                const manipInfo = p.manipulation || {};
                if (isBull && manipInfo.low && rangeInfo.low) {
                    drawZone(rangeInfo.low, manipInfo.low, candles, 'rgb(255,68,68)', ri.manipulation_start || 0);
                } else if (!isBull && manipInfo.high && rangeInfo.high) {
                    drawZone(manipInfo.high || rangeInfo.high, rangeInfo.high, candles, 'rgb(0,255,136)', ri.manipulation_start || 0);
                }

                // DL2 limit line
                if (manipInfo.dl2_limit) {
                    drawHLine(manipInfo.dl2_limit, 'rgba(255,193,7,0.5)', 'DL2', LightweightCharts.LineStyle.Dotted, 1);
                }

                // Phase markers
                if (ri.range_start_time) {
                    allMarkers.push({
                        time: ri.range_start_time,
                        position: 'aboveBar',
                        color: '#888',
                        shape: 'square',
                        text: 'Range Start',
                    });
                }
                if (ri.manipulation_start_time) {
                    allMarkers.push({
                        time: ri.manipulation_start_time,
                        position: isBull ? 'belowBar' : 'aboveBar',
                        color: '#ff4444',
                        shape: 'circle',
                        text: 'Manipulation',
                    });
                }
                if (ri.manipulation_end_time) {
                    allMarkers.push({
                        time: ri.manipulation_end_time,
                        position: isBull ? 'belowBar' : 'aboveBar',
                        color: '#e040fb',
                        shape: 'circle',
                        text: 'Manip End',
                    });
                }

                // Entry/Stop/Target lines (first PO3 only to avoid clutter)
                if (idx === 0) {
                    if (entry) drawHLine(entry, '#00d4ff', 'Entry', LightweightCharts.LineStyle.Solid, 2);
                    if (stop) drawHLine(stop, '#ff4444', 'Stop Loss', LightweightCharts.LineStyle.Dashed, 1);
                    if (target) drawHLine(target, '#00ff88', 'Target', LightweightCharts.LineStyle.Dashed, 1);
                }

                // Card HTML
                cardsHTML += '<div class="schematic-card ' + (p.has_expansion ? 'confirmed' : 'forming') + '">';
                cardsHTML += '<div class="sc-title ' + dirCls + '">' + dirLabel;
                cardsHTML += '<span class="sc-status ' + (p.has_expansion ? 'confirmed' : 'forming') + '">' + phase.toUpperCase() + '</span></div>';

                if (entry) cardsHTML += '<div class="sc-row"><span>Entry</span><span class="val">' + fmt(entry) + '</span></div>';
                if (stop) cardsHTML += '<div class="sc-row"><span>Stop Loss</span><span class="val">' + fmt(stop) + '</span></div>';
                if (target) cardsHTML += '<div class="sc-row"><span>Target</span><span class="val">' + fmt(target) + '</span></div>';
                if (rr) cardsHTML += '<div class="sc-row"><span>R:R</span><span class="val">' + rr.toFixed(1) + '</span></div>';
                if (quality) cardsHTML += '<div class="sc-row"><span>Quality</span><span class="val">' + Math.round(quality * 100) + '%</span></div>';

                // Manipulation depth
                const devPct = manipInfo.deviation_pct || 0;
                cardsHTML += '<div class="sc-row"><span>Dev Depth</span><span class="val">' + devPct.toFixed(1) + '% / 30%</span></div>';

                // TCT model inside manipulation
                if (p.tct_model?.detected) {
                    cardsHTML += '<div class="sc-row"><span>TCT Model</span><span class="val" style="color:#e040fb;">' + (p.tct_model.type || 'detected') + '</span></div>';
                }

                // Exception type
                if (p.exception) {
                    const excLabel = p.exception === 'exception_1_two_tap' ? '2-Tap Exception' : p.exception === 'exception_2_internal_tct' ? 'Internal TCT' : p.exception;
                    cardsHTML += '<div class="sc-row"><span>Exception</span><span class="val" style="color:#ffc107;">' + excLabel + '</span></div>';
                }

                // Range info
                if (rangeInfo.high && rangeInfo.low) {
                    cardsHTML += '<div class="sc-row"><span>Range</span><span class="val">' + fmt(rangeInfo.low) + ' — ' + fmt(rangeInfo.high) + '</span></div>';
                }

                cardsHTML += '</div>';
            });

            // Set all markers
            if (allMarkers.length > 0) {
                allMarkers.sort((a, b) => a.time - b.time);
                candleSeries.setMarkers(allMarkers);
            }

            cardsEl.innerHTML = cardsHTML;
        }

        async function loadData() {
            try {
                const resp = await fetch('/api/schematic-data?symbol=' + SYMBOL + '&timeframe=' + TIMEFRAME + '&type=' + CHART_TYPE);
                const data = await resp.json();

                if (data.error) {
                    document.getElementById('schematicCards').innerHTML =
                        '<div class="schematic-card"><div class="sc-title" style="color:#ff4444;">Error: ' + data.error + '</div></div>';
                    return;
                }

                if (data.candles && data.candles.length > 0) {
                    candleSeries.setData(data.candles);
                }

                if (CHART_TYPE === 'po3') {
                    drawPO3Schematics(data);
                } else {
                    drawSchematics(data);
                }
            } catch (e) {
                console.error('Failed to load schematic data:', e);
            }
        }

        initChart();
        loadData();

        // Auto-refresh every 60 seconds
        setInterval(loadData, 60000);
    </script>
</body>
</html>"""
    return HTMLResponse(content=html)


@app.get("/api/po3-data")
async def get_po3_data(symbol: str, timeframe: str = "4h"):
    """
    Return full PO3 schematic + candle data for the PO3 chart page.
    Fetches fresh candles and runs PO3 detection with full range/manipulation/expansion data.
    """
    try:
        df = await fetch_mexc_candles(symbol, timeframe, 200)
        if df is None or len(df) < 30:
            return {"error": "Insufficient candle data", "symbol": symbol}

        current_price = float(df.iloc[-1]["close"])

        candles_list = []
        candles_json = []
        for _, row in df.iterrows():
            candle = {
                'open_time': str(row['open_time']),
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': float(row['volume'])
            }
            candles_list.append(candle)
            ts = row['open_time']
            if hasattr(ts, 'timestamp'):
                epoch = int(ts.timestamp())
            else:
                epoch = int(pd.Timestamp(ts).timestamp())
            candles_json.append({
                'time': epoch,
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
            })

        detected_range = await detect_best_range(candles_list)
        range_list = [detected_range] if detected_range and not isinstance(detected_range, list) else (detected_range or [])

        po3_result = detect_po3_schematics(df, range_list)
        all_po3 = po3_result.get("bullish_po3", []) + po3_result.get("bearish_po3", [])

        # Also get TCT schematics for the manipulation phase overlay
        tct_result = detect_tct_schematics(df, range_list)
        all_tct = (
            tct_result.get("accumulation_schematics", []) +
            tct_result.get("distribution_schematics", [])
        )

        # Map tap indices to timestamps for TCT overlays
        tct_with_timestamps = []
        for s in all_tct:
            if not isinstance(s, dict):
                continue
            sch = dict(s)
            for tap_key in ['tap1', 'tap2', 'tap3']:
                tap = sch.get(tap_key)
                if tap and isinstance(tap, dict) and 'idx' in tap:
                    idx = int(tap['idx'])
                    if 0 <= idx < len(candles_json):
                        tap['time'] = candles_json[idx]['time']
            bos = sch.get('bos_confirmation')
            if bos and isinstance(bos, dict) and 'bos_idx' in bos:
                idx = int(bos['bos_idx'])
                if 0 <= idx < len(candles_json):
                    bos['bos_time'] = candles_json[idx]['time']
            tct_with_timestamps.append(sch)

        return convert_numpy_types({
            "symbol": symbol,
            "timeframe": timeframe,
            "current_price": current_price,
            "candles": candles_json,
            "po3_schematics": all_po3,
            "tct_schematics": tct_with_timestamps,
            "ranges": range_list,
        })

    except Exception as e:
        logger.error(f"[PO3-DATA] Error for {symbol}/{timeframe}: {e}")
        return {"error": str(e), "symbol": symbol}


@app.get("/po3-chart", response_class=HTMLResponse)
async def po3_chart_page(symbol: str = "BTCUSDT", timeframe: str = "4h"):
    """
    Dedicated PO3 schematic chart page showing Power of Three overlays
    with range boxes, manipulation zones, TCT model taps, and expansion targets.
    Based on TCT Mentorship Lecture 8.
    """
    html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PO3 Schematic — """ + symbol + """ """ + timeframe.upper() + """</title>
    <script src="https://unpkg.com/lightweight-charts@4.1.0/dist/lightweight-charts.standalone.production.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { background: #0a0a12; color: #e0e0e0; font-family: 'Segoe UI', sans-serif; overflow: hidden; }
        .header {
            display: flex; align-items: center; justify-content: space-between;
            padding: 10px 20px; background: #12121a; border-bottom: 1px solid #1e1e2d;
        }
        .header h1 { font-size: 1.1rem; color: #ff9800; }
        .header h1 span.pair { color: #00d4ff; }
        .header h1 span.tf { color: #888; font-weight: 400; font-size: 0.9rem; margin-left: 8px; }
        .back-link { color: #888; text-decoration: none; font-size: 0.8rem; padding: 4px 12px; border: 1px solid #333; border-radius: 4px; }
        .back-link:hover { color: #e0e0e0; border-color: #555; }
        .chart-container { width: 100%; height: calc(100vh - 95px); }
        .legend {
            display: flex; gap: 16px; padding: 6px 20px; background: #12121a;
            border-top: 1px solid #1e1e2d; font-size: 0.7rem; color: #888; flex-wrap: wrap;
        }
        .legend-item { display: flex; align-items: center; gap: 4px; }
        .legend-dot { width: 10px; height: 10px; border-radius: 50%; }
        .legend-line { width: 16px; height: 2px; }
        .legend-box { width: 14px; height: 10px; border-radius: 2px; }
        .po3-cards {
            position: absolute; top: 50px; right: 10px; z-index: 10;
            display: flex; flex-direction: column; gap: 6px; max-height: calc(100vh - 110px);
            overflow-y: auto; padding: 4px;
        }
        .po3-card {
            background: rgba(18, 18, 26, 0.95); border: 1px solid #2d2d44;
            border-radius: 6px; padding: 8px 12px; min-width: 240px;
            font-size: 0.7rem; backdrop-filter: blur(8px);
        }
        .po3-card.bullish { border-left: 3px solid #00ff88; }
        .po3-card.bearish { border-left: 3px solid #ff4444; }
        .pc-title { font-weight: 700; margin-bottom: 4px; }
        .pc-title.bullish { color: #00ff88; }
        .pc-title.bearish { color: #ff4444; }
        .pc-phase { display: inline-block; font-size: 0.6rem; padding: 1px 6px; border-radius: 3px; margin-left: 6px; }
        .pc-phase.range { background: rgba(128,128,128,0.2); color: #aaa; }
        .pc-phase.manipulation, .pc-phase.manipulation_complete { background: rgba(255,152,0,0.2); color: #ff9800; }
        .pc-phase.expansion { background: rgba(0,255,136,0.2); color: #00ff88; }
        .pc-row { display: flex; justify-content: space-between; color: #888; margin: 2px 0; }
        .pc-row .val { color: #e0e0e0; font-weight: 600; }
        .pc-tags { display: flex; gap: 4px; flex-wrap: wrap; margin-top: 4px; }
        .pc-tag { font-size: 0.55rem; padding: 1px 5px; border-radius: 3px; background: rgba(255,255,255,0.05); color: #aaa; }
        .pc-tag.tct { background: rgba(224,64,251,0.15); color: #e040fb; }
        .pc-tag.exception { background: rgba(255,152,0,0.15); color: #ff9800; }
        .pc-tag.compressed { background: rgba(0,188,212,0.15); color: #00bcd4; }
        .pc-tag.expanding { background: rgba(0,255,136,0.15); color: #00ff88; }
        .pc-manip-bar { margin: 4px 0; }
        .bar-track { height: 4px; background: #1a1a2a; border-radius: 2px; overflow: hidden; }
        .bar-fill { height: 100%; border-radius: 2px; }
        .bar-fill.bullish { background: linear-gradient(90deg, #00ff88, #00bcd4); }
        .bar-fill.bearish { background: linear-gradient(90deg, #ff4444, #ff9800); }
        .bar-labels { display: flex; justify-content: space-between; font-size: 0.55rem; color: #666; margin-top: 1px; }
    </style>
</head>
<body>
    <div class="header">
        <h1>PO3 Schematic <span class="pair">""" + symbol.replace("USDT", "/USDT") + """</span><span class="tf">""" + timeframe.upper() + """</span></h1>
        <a href="/dashboard" class="back-link">Back to Dashboard</a>
    </div>
    <div class="chart-container" id="chartContainer"></div>
    <div class="po3-cards" id="po3Cards"></div>
    <div class="legend">
        <div class="legend-item"><div class="legend-box" style="background:rgba(128,128,128,0.2);border:1px solid #888;"></div> PO3 Range</div>
        <div class="legend-item"><div class="legend-box" style="background:rgba(255,152,0,0.2);border:1px solid #ff9800;"></div> Manipulation Zone</div>
        <div class="legend-item"><div class="legend-box" style="background:rgba(0,255,136,0.15);border:1px solid #00ff88;"></div> Expansion Zone</div>
        <div class="legend-item"><div class="legend-dot" style="background:#e040fb;"></div> TCT Tap (Deviation)</div>
        <div class="legend-item"><div class="legend-dot" style="background:#00d4ff;"></div> TCT Tap (Range)</div>
        <div class="legend-item"><div class="legend-line" style="background:#ffc107;"></div> Deviation Limit (DL2)</div>
        <div class="legend-item"><div class="legend-line" style="background:#00ff88;"></div> BOS / Entry</div>
        <div class="legend-item"><div class="legend-line" style="background:#ff4444;"></div> Stop Loss</div>
        <div class="legend-item"><div class="legend-line" style="background:#00d4ff;"></div> Target</div>
    </div>

    <script>
        const SYMBOL = '""" + symbol + """';
        const TIMEFRAME = '""" + timeframe + """';

        let chart, candleSeries;
        const overlays = [];

        function initChart() {
            const container = document.getElementById('chartContainer');
            chart = LightweightCharts.createChart(container, {
                width: container.clientWidth,
                height: container.clientHeight,
                layout: { background: { color: '#0a0a12' }, textColor: '#888' },
                grid: { vertLines: { color: '#1a1a2a' }, horzLines: { color: '#1a1a2a' } },
                crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
                rightPriceScale: { borderColor: '#2d2d44' },
                timeScale: { borderColor: '#2d2d44', timeVisible: true },
            });

            candleSeries = chart.addCandlestickSeries({
                upColor: '#00ff88', downColor: '#ff4444',
                borderUpColor: '#00ff88', borderDownColor: '#ff4444',
                wickUpColor: '#00ff88', wickDownColor: '#ff4444',
            });

            window.addEventListener('resize', () => {
                chart.applyOptions({ width: container.clientWidth, height: container.clientHeight });
            });
        }

        function fmt(p) {
            if (p == null || isNaN(p)) return '--';
            if (p >= 10000) return '$' + p.toLocaleString(undefined, {maximumFractionDigits: 0});
            if (p >= 100)   return '$' + p.toLocaleString(undefined, {maximumFractionDigits: 2});
            if (p >= 1)     return '$' + p.toFixed(2);
            if (p >= 0.01)  return '$' + p.toFixed(4);
            return '$' + p.toPrecision(4);
        }

        function drawHLine(price, color, title, style, width) {
            return candleSeries.createPriceLine({
                price: price, color: color, lineWidth: width || 1,
                lineStyle: style || LightweightCharts.LineStyle.Solid,
                axisLabelVisible: true, title: title || '',
            });
        }

        function drawZone(high, low, candles, color, startIdx) {
            if (!candles || candles.length < 2) return;
            const si = startIdx || 0;
            const startTime = candles[si].time;
            const endTime = candles[candles.length - 1].time;
            const lastInterval = candles.length > 1 ? candles[candles.length-1].time - candles[candles.length-2].time : 3600;
            const futureTime = endTime + lastInterval * 20;

            const topSeries = chart.addAreaSeries({
                topColor: color.replace(')', ',0.25)').replace('rgb', 'rgba'),
                bottomColor: color.replace(')', ',0.08)').replace('rgb', 'rgba'),
                lineColor: color.replace(')', ',0.4)').replace('rgb', 'rgba'),
                lineWidth: 1, priceScaleId: 'right',
                lastValueVisible: false, crosshairMarkerVisible: false,
            });
            topSeries.setData([
                { time: startTime, value: high },
                { time: futureTime, value: high },
            ]);
            overlays.push(topSeries);

            const botSeries = chart.addAreaSeries({
                topColor: color.replace(')', ',0.08)').replace('rgb', 'rgba'),
                bottomColor: color.replace(')', ',0.02)').replace('rgb', 'rgba'),
                lineColor: color.replace(')', ',0.4)').replace('rgb', 'rgba'),
                lineWidth: 1, priceScaleId: 'right',
                lastValueVisible: false, crosshairMarkerVisible: false,
            });
            botSeries.setData([
                { time: startTime, value: low },
                { time: futureTime, value: low },
            ]);
            overlays.push(botSeries);
        }

        function drawPO3Overlays(data) {
            const candles = data.candles;
            const po3List = data.po3_schematics || [];
            const tctList = data.tct_schematics || [];
            const cardsEl = document.getElementById('po3Cards');

            if (po3List.length === 0 && tctList.length === 0) {
                cardsEl.innerHTML = '<div class="po3-card"><div class="pc-title" style="color:#888;">No PO3 schematics detected on this timeframe</div>' +
                    '<div style="color:#666;font-size:0.6rem;margin-top:4px;">PO3 = Range \\u2192 Manipulation \\u2192 Expansion</div></div>';

                // Still draw TCT schematics if present
                if (tctList.length > 0) {
                    drawTCTOnChart(tctList, candles);
                }
                return;
            }

            let cardsHTML = '';

            // Draw PO3 range and manipulation zones
            po3List.forEach((p, idx) => {
                const isBull = p.direction === 'bullish';
                const dirClass = isBull ? 'bullish' : 'bearish';
                const phase = p.phase || 'range';
                const quality = Math.round((p.quality_score || 0) * 100);
                const entry = p.entry?.price;
                const stop = p.stop_loss?.price;
                const target = p.target?.price;
                const rr = p.risk_reward;

                // Draw PO3 main range box (gray)
                if (p.range && p.range.high && p.range.low) {
                    drawZone(p.range.high, p.range.low, candles, 'rgb(128,128,128)', 0);

                    // DL2 lines
                    const rangeSize = p.range.high - p.range.low;
                    const dl2High = p.range.high + rangeSize * 0.3;
                    const dl2Low = p.range.low - rangeSize * 0.3;
                    drawHLine(dl2High, 'rgba(255,193,7,0.4)', 'DL2 High', LightweightCharts.LineStyle.Dotted, 1);
                    drawHLine(dl2Low, 'rgba(255,193,7,0.4)', 'DL2 Low', LightweightCharts.LineStyle.Dotted, 1);
                }

                // Draw manipulation zone (orange)
                const manipInfo = p.manipulation || {};
                if (manipInfo.range_high && manipInfo.range_low) {
                    drawZone(manipInfo.range_high, manipInfo.range_low, candles, 'rgb(255,152,0)', Math.floor(candles.length * 0.3));
                }

                // Draw entry/stop/target lines
                if (entry) drawHLine(entry, '#00d4ff', 'Entry', LightweightCharts.LineStyle.Solid, 2);
                if (stop) drawHLine(stop, '#ff4444', 'Stop Loss', LightweightCharts.LineStyle.Dashed, 1);
                if (target) drawHLine(target, '#00ff88', 'Target (PO3)', LightweightCharts.LineStyle.Dashed, 2);

                // Build PO3 card
                cardsHTML += '<div class="po3-card ' + dirClass + '">';
                cardsHTML += '<div class="pc-title ' + dirClass + '">' + (isBull ? 'BULLISH' : 'BEARISH') + ' PO3';
                cardsHTML += '<span class="pc-phase ' + phase.replace(' ', '_') + '">' + phase.replace('_', ' ') + '</span></div>';

                if (entry) cardsHTML += '<div class="pc-row"><span>Entry</span><span class="val">' + fmt(entry) + '</span></div>';
                if (stop) cardsHTML += '<div class="pc-row"><span>Stop Loss</span><span class="val">' + fmt(stop) + '</span></div>';
                if (target) cardsHTML += '<div class="pc-row"><span>Target</span><span class="val">' + fmt(target) + '</span></div>';
                if (rr) cardsHTML += '<div class="pc-row"><span>R:R</span><span class="val">' + rr.toFixed(1) + '</span></div>';
                cardsHTML += '<div class="pc-row"><span>Quality</span><span class="val">' + quality + '%</span></div>';

                // Manipulation depth bar
                const devPct = manipInfo.deviation_pct || 0;
                const barWidth = Math.min(100, (devPct / 30) * 100);
                cardsHTML += '<div class="pc-manip-bar"><div class="bar-track"><div class="bar-fill ' + dirClass + '" style="width:' + barWidth + '%"></div></div>';
                cardsHTML += '<div class="bar-labels"><span>Dev: ' + devPct.toFixed(1) + '%</span><span>DL2: 30%</span></div></div>';

                // Tags
                cardsHTML += '<div class="pc-tags">';
                if (p.tct_model?.detected) cardsHTML += '<span class="pc-tag tct">TCT ' + (p.tct_model.type || 'Model') + '</span>';
                if (p.exception) {
                    const excLabel = p.exception === 'exception_1_two_tap' ? '2-Tap Exception' : p.exception === 'exception_2_internal_tct' ? 'Internal TCT' : p.exception;
                    cardsHTML += '<span class="pc-tag exception">' + excLabel + '</span>';
                }
                if (p.has_compression) cardsHTML += '<span class="pc-tag compressed">Compressed</span>';
                if (p.has_liquidity_both_sides) cardsHTML += '<span class="pc-tag">Dual Liquidity</span>';
                if (p.has_expansion) cardsHTML += '<span class="pc-tag expanding">Expanding</span>';
                cardsHTML += '</div>';

                if (p.range) {
                    cardsHTML += '<div style="margin-top:4px;font-size:0.55rem;color:#555;">Range: ' + fmt(p.range.low) + ' — ' + fmt(p.range.high) + '</div>';
                }
                cardsHTML += '</div>';
            });

            cardsEl.innerHTML = cardsHTML;

            // Draw TCT schematics (from manipulation phase) on chart
            drawTCTOnChart(tctList, candles);
        }

        function drawTCTOnChart(tctList, candles) {
            const allMarkers = [];
            tctList.slice(0, 3).forEach((s, idx) => {
                const direction = s.direction || 'unknown';

                // Draw range box for TCT
                if (s.range && s.range.high && s.range.low) {
                    const rangeStartIdx = Math.max(0, (s.tap1?.idx || 0) - 3);
                    drawZone(s.range.high, s.range.low, candles, direction === 'bullish' ? 'rgb(120,80,255)' : 'rgb(255,68,68)', rangeStartIdx);
                }

                // Tap markers
                [s.tap1, s.tap2, s.tap3].forEach((tap, ti) => {
                    if (tap && tap.time) {
                        const isDev = tap.is_deviation;
                        const isLow = direction === 'bullish';
                        allMarkers.push({
                            time: tap.time,
                            position: isLow ? 'belowBar' : 'aboveBar',
                            color: isDev ? '#e040fb' : '#00d4ff',
                            shape: 'circle',
                            text: 'T' + (ti + 1) + (isDev ? ' (Dev)' : ''),
                        });
                    }
                });

                // BOS marker
                const bos = s.bos_confirmation;
                if (bos && bos.bos_time) {
                    allMarkers.push({
                        time: bos.bos_time,
                        position: direction === 'bullish' ? 'aboveBar' : 'belowBar',
                        color: '#00ff88',
                        shape: direction === 'bullish' ? 'arrowUp' : 'arrowDown',
                        text: 'BOS',
                    });
                }
            });

            if (allMarkers.length > 0) {
                allMarkers.sort((a, b) => a.time - b.time);
                candleSeries.setMarkers(allMarkers);
            }
        }

        async function loadData() {
            try {
                const resp = await fetch('/api/po3-data?symbol=' + SYMBOL + '&timeframe=' + TIMEFRAME);
                const data = await resp.json();

                if (data.error) {
                    document.getElementById('po3Cards').innerHTML =
                        '<div class="po3-card"><div class="pc-title" style="color:#ff4444;">Error: ' + data.error + '</div></div>';
                    return;
                }

                if (data.candles && data.candles.length > 0) {
                    candleSeries.setData(data.candles);
                }

                drawPO3Overlays(data);
            } catch (e) {
                console.error('Failed to load PO3 data:', e);
            }
        }

        initChart();
        loadData();

        // Auto-refresh every 60 seconds
        setInterval(loadData, 60000);
    </script>
</body>
</html>"""
    return HTMLResponse(content=html)


@app.get("/api/scan-now")
async def trigger_scan():
    """Manually trigger a range scan (if not already running)."""
    if scanner_status["is_scanning"]:
        return {"status": "already_scanning", "progress": f"{scanner_status['pairs_scanned']}/{scanner_status['total_pairs']}"}
    asyncio.create_task(run_full_scan())
    return {"status": "scan_started", "total_pairs": len(COIN_LIST), "scan_method": "pairsrange_scanner (RPS)", "rps_threshold": RANGE_MIN_RPS}

@app.get("/api/candles")
async def get_candles(interval: str = "4h", limit: int = 100, symbol: Optional[str] = None):
    """
    Fetch candles from MEXC - server-side to avoid CORS issues.

    Args:
        interval: Timeframe (1m, 5m, 15m, 1h, 4h, 1d)
        limit: Number of candles (max 1000)
        symbol: Trading pair (defaults to BTCUSDT)

    Returns:
        List of candles with time, open, high, low, close
    """
    try:
        sym = resolve_symbol(symbol)
        df = await fetch_mexc_candles(sym, interval, min(limit, 500))
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
            "symbol": sym,
            "interval": interval,
            "count": len(candles),
            "candles": candles
        }
    except Exception as e:
        logger.error(f"[CANDLES_ERROR] {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/api/risk-calculator")
async def risk_calculator(
    account_balance: float = 10000,
    risk_pct: float = 1.0,
    stop_loss_pct: float = 0.26,
    risk_reward: float = 3.0,
    market: str = "crypto",
    gold_price: float = 2000,
    leverage: float = 10.0,
    symbol: str = None,
    direction: str = None
):
    """
    TCT Lecture 7 — Risk Management Calculator.

    Calculates position size, leverage requirements, margin, and projected outcomes
    using the TCT risk management methodology.

    Now integrates TCT Lecture 1 market structure for bias assessment:
    - When symbol is provided, fetches live MS data (trend, EOF, BOS)
    - Provides confluence score based on whether trade direction aligns with MS

    Args:
        account_balance: Total account balance in USD
        risk_pct: Risk percentage per trade (1-3% recommended)
        stop_loss_pct: Stop-loss size as percentage of position
        risk_reward: Target risk-to-reward ratio
        market: Market type (crypto, forex, gold)
        gold_price: Current gold price (only used if market=gold)
        leverage: Leverage being used
        symbol: Optional trading pair for live market structure analysis
        direction: Optional trade direction ('long' or 'short') for bias check

    Returns:
        Complete risk management profile with position sizing, leverage, and projections.
    """
    try:
        risk_amount = account_balance * (risk_pct / 100)

        # Position size formulas from TCT Lecture 7
        if stop_loss_pct <= 0:
            return JSONResponse({"error": "Stop-loss percentage must be greater than 0"}, status_code=400)

        raw_position_size = (risk_amount / stop_loss_pct) * 100

        if market == "forex":
            position_size_lots = raw_position_size / 100000
            position_display = {"lots": round(position_size_lots, 4), "units": round(raw_position_size, 2)}
        elif market == "gold":
            lot_value = gold_price * 100
            position_size_lots = raw_position_size / lot_value if lot_value > 0 else 0
            position_display = {"lots": round(position_size_lots, 4), "lot_value": round(lot_value, 2), "units": round(raw_position_size, 2)}
        else:
            position_display = {"usd": round(raw_position_size, 2)}

        # Leverage calculation
        min_leverage_needed = raw_position_size / account_balance if account_balance > 0 else 0
        used_margin = raw_position_size / leverage if leverage > 0 else raw_position_size
        free_margin = account_balance - used_margin

        # Profit/Loss at SL and TP
        loss_at_sl = risk_amount
        profit_at_tp = risk_amount * risk_reward
        tp_pct_gain = (profit_at_tp / account_balance) * 100

        # Compounding projection (5% per week, 35 trading weeks)
        weekly_rate = 0.05
        weeks_per_year = 35
        compounding = []
        balance = account_balance
        for year in range(1, 4):
            balance = balance * ((1 + weekly_rate) ** weeks_per_year)
            compounding.append({"year": year, "balance": round(balance, 2)})

        # Losing streak simulation (6 losses then wins at given RR)
        streak_balance = account_balance
        loss_multiplier = 1 - (risk_pct / 100)
        win_multiplier = 1 + (risk_pct * risk_reward / 100)

        losing_streak = []
        for i in range(6):
            streak_balance *= loss_multiplier
            losing_streak.append({"trade": i + 1, "balance": round(streak_balance, 2), "result": "loss"})

        winning_streak = []
        for i in range(3):
            streak_balance *= win_multiplier
            winning_streak.append({"trade": 7 + i, "balance": round(streak_balance, 2), "result": "win"})

        net_result_pct = ((streak_balance - account_balance) / account_balance) * 100

        profile = {
            "inputs": {
                "account_balance": account_balance,
                "risk_pct": risk_pct,
                "risk_amount": round(risk_amount, 2),
                "stop_loss_pct": stop_loss_pct,
                "risk_reward": risk_reward,
                "market": market,
                "leverage": leverage
            },
            "position_sizing": {
                "position_size": round(raw_position_size, 2),
                **position_display
            },
            "leverage_analysis": {
                "min_leverage_needed": round(min_leverage_needed, 2),
                "leverage_used": leverage,
                "used_margin": round(used_margin, 2),
                "free_margin": round(free_margin, 2),
                "margin_pct_of_account": round((used_margin / account_balance) * 100, 2) if account_balance > 0 else 0
            },
            "trade_outcome": {
                "loss_at_stop": round(loss_at_sl, 2),
                "profit_at_target": round(profit_at_tp, 2),
                "tp_account_gain_pct": round(tp_pct_gain, 2)
            },
            "compounding_projection": compounding,
            "streak_simulation": {
                "scenario": f"6 losses then 3 wins at {risk_reward}R",
                "trades": losing_streak + winning_streak,
                "final_balance": round(streak_balance, 2),
                "net_result_pct": round(net_result_pct, 2)
            },
            "rules": {
                "risk_range": "1-3% of account per trade",
                "min_rr": "2:1 minimum, 2.3-3:1 average",
                "margin_mode": "Always use ISOLATED margin",
                "liquidation_warning": "Ensure liquidation price stays outside stop-loss range",
                "weekly_target": "5% per week (~1% per day)"
            }
        }

        # Optional: Add market structure bias assessment if symbol provided
        if symbol and market == "crypto":
            try:
                sym = resolve_symbol(symbol)
                htf_df = await fetch_mexc_candles(sym, "4h", 100)
                if htf_df is not None and len(htf_df) >= 6:
                    ms = MarketStructure()
                    htf_pivots = ms.detect_pivots(htf_df)
                    htf_trend = htf_pivots.get("trend", "neutral")
                    htf_eof = htf_pivots.get("eof", {})
                    htf_bos = htf_pivots.get("bos_events", [])
                    last_bos = htf_bos[-1] if htf_bos else None

                    # Calculate confluence with trade direction
                    ms_bias = htf_eof.get("bias", "neutral")
                    trend_shift = htf_eof.get("trend_shift", False)

                    confluence = "neutral"
                    confluence_score = 0.5
                    if direction:
                        if direction == "long" and ms_bias == "bullish":
                            confluence = "aligned"
                            confluence_score = 1.0
                        elif direction == "short" and ms_bias == "bearish":
                            confluence = "aligned"
                            confluence_score = 1.0
                        elif direction == "long" and ms_bias == "bearish":
                            confluence = "against_bias"
                            confluence_score = 0.3
                        elif direction == "short" and ms_bias == "bullish":
                            confluence = "against_bias"
                            confluence_score = 0.3
                        elif ms_bias == "neutral":
                            confluence = "neutral"
                            confluence_score = 0.5

                    profile["market_structure_context"] = {
                        "htf_trend": htf_trend,
                        "eof_bias": ms_bias,
                        "eof_expectation": htf_eof.get("expectation", "undetermined"),
                        "trend_shift_detected": trend_shift,
                        "last_bos": {
                            "type": last_bos["type"],
                            "price": last_bos["bos_price"]
                        } if last_bos else None,
                        "trade_confluence": confluence,
                        "confluence_score": confluence_score,
                        "risk_note": (
                            "Trade aligns with HTF market structure bias"
                            if confluence == "aligned"
                            else "Trade is AGAINST HTF market structure bias - consider reducing size"
                            if confluence == "against_bias"
                            else "Neutral market structure - standard risk applies"
                        )
                    }
            except Exception as ms_err:
                profile["market_structure_context"] = {
                    "error": f"Could not fetch MS data: {str(ms_err)}"
                }

        return JSONResponse(convert_numpy_types(profile))

    except Exception as e:
        logger.error(f"[RISK_CALC_ERROR] {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    """
    Interactive TCT Dashboard with candlestick chart and all TCT metrics.
    Displays: Market Structure, Ranges, Supply/Demand Zones, Liquidity, Deviations, Risk Management.
    """
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, viewport-fit=cover, maximum-scale=5">
    <meta name="apple-mobile-web-app-capable" content="yes">
    <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
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
        #chart { width: 100%; height: calc(100% - 36px); }
        .chart-controls {
            display: flex;
            gap: 6px;
            padding: 4px 8px;
            background: #12121a;
            border-bottom: 1px solid #2d2d44;
        }
        .chart-ctrl-btn {
            background: #1a1a2e;
            color: #aaa;
            border: 1px solid #2d2d44;
            padding: 3px 10px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 0.7rem;
            font-weight: 600;
            transition: all 0.2s;
        }
        .chart-ctrl-btn:hover { background: #2d2d44; color: #e0e0e0; border-color: #00d4ff80; }
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
        .schematic-meta .unsafe { color: #ff4444; }
        .scan-btn {
            background: linear-gradient(135deg, #1a1a2e, #2d2d44);
            color: #00d4ff;
            border: 1px solid #00d4ff40;
            padding: 5px 12px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 0.7rem;
            font-weight: 600;
            flex: 1;
            text-align: center;
            transition: all 0.2s;
        }
        .scan-btn:hover { background: #00d4ff22; border-color: #00d4ff; }
        .scan-btn:disabled { opacity: 0.5; cursor: not-allowed; }
        .scan-btn.scanning { border-color: #ffc107; color: #ffc107; }
        .tf-group {
            margin-bottom: 8px;
        }
        .tf-group-header {
            font-size: 0.65rem;
            font-weight: 700;
            color: #888;
            padding: 4px 0 4px 0;
            border-bottom: 1px solid #1e1e2d;
            margin-bottom: 4px;
            display: flex;
            align-items: center;
            gap: 6px;
        }
        .tf-group-header .tf-label {
            color: #00d4ff;
            font-size: 0.6rem;
            padding: 1px 6px;
            border-radius: 3px;
            background: rgba(0,212,255,0.1);
        }
        .tf-group-header .tf-count {
            margin-left: auto;
            font-size: 0.55rem;
            color: #666;
        }
        .schematic-link {
            text-decoration: none;
            color: inherit;
            display: block;
            transition: transform 0.15s, box-shadow 0.15s;
            border-radius: 6px;
        }
        .schematic-link:hover {
            transform: translateX(2px);
            box-shadow: 0 0 8px rgba(0,212,255,0.15);
        }
        .schematic-link:hover .schematic-item,
        .schematic-link:hover .po3-item {
            border-color: #00d4ff;
        }
        .view-chart-hint {
            font-size: 0.55rem;
            color: #00d4ff88;
            text-align: right;
            margin-top: 2px;
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
            gap: 8px;
            align-items: center;
        }
        .pair-search-wrapper {
            position: relative;
            display: inline-block;
        }
        .pair-search {
            background: #1a1a2e;
            color: #00d4ff;
            border: 1px solid #2d2d44;
            padding: 6px 12px;
            border-radius: 6px;
            font-size: 0.85rem;
            font-weight: 700;
            min-width: 140px;
            width: 140px;
            cursor: text;
        }
        .pair-search:focus { outline: none; border-color: #00d4ff; box-shadow: 0 0 0 2px rgba(0,212,255,0.2); }
        .pair-search::placeholder { color: #555; font-weight: 400; }
        .pair-dropdown {
            display: none;
            position: absolute;
            top: 100%;
            left: 0;
            background: #12121a;
            border: 1px solid #2d2d44;
            border-radius: 6px;
            max-height: 400px;
            overflow-y: auto;
            z-index: 1000;
            min-width: 260px;
            margin-top: 4px;
            box-shadow: 0 8px 24px rgba(0,0,0,0.5);
        }
        .pair-dropdown.open { display: block; }
        .pair-dropdown-group { padding: 4px 0; }
        .pair-dropdown-group-label {
            padding: 6px 12px 4px;
            font-size: 0.6rem;
            color: #00d4ff;
            text-transform: uppercase;
            letter-spacing: 1px;
            font-weight: 600;
            border-bottom: 1px solid #1a1a2e;
        }
        .pair-dropdown-item {
            padding: 6px 12px;
            font-size: 0.78rem;
            color: #ccc;
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .pair-dropdown-item:hover { background: #1a1a2e; color: #00d4ff; }
        .pair-dropdown-item.active { background: rgba(0,212,255,0.1); color: #00d4ff; font-weight: 600; }
        .pair-dropdown-item .pair-base { font-weight: 600; }
        .pair-dropdown-item .pair-quote { color: #555; font-size: 0.65rem; }
        .tf-dropdown {
            background: #1a1a2e;
            color: #00d4ff;
            border: 1px solid #2d2d44;
            padding: 6px 12px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 0.8rem;
            font-weight: 600;
            appearance: none;
            -webkit-appearance: none;
            background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' fill='%2300d4ff' viewBox='0 0 16 16'%3E%3Cpath d='M1.5 5.5l6.5 6.5 6.5-6.5'/%3E%3C/svg%3E");
            background-repeat: no-repeat;
            background-position: right 8px center;
            padding-right: 28px;
            min-width: 80px;
        }
        .tf-dropdown:hover { border-color: #00d4ff; }
        .tf-dropdown:focus { outline: none; border-color: #00d4ff; box-shadow: 0 0 0 2px rgba(0,212,255,0.2); }
        .tf-dropdown option {
            background: #1a1a2e;
            color: #e0e0e0;
            padding: 4px;
        }
        .tf-dropdown optgroup {
            background: #12121a;
            color: #00d4ff;
            font-weight: 600;
        }
        .tf-label {
            font-size: 0.65rem;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 0.5px;
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
        /* === HIGHEST PROBABILITY SETUP PANEL === */
        .setup-panel {
            background: linear-gradient(135deg, rgba(0,212,255,0.08) 0%, rgba(0,255,136,0.05) 100%);
            border: 1px solid rgba(0,212,255,0.3);
            border-radius: 8px;
            padding: 10px 14px;
            margin-bottom: 10px;
        }
        .setup-panel h3 {
            font-size: 0.8rem;
            color: #00d4ff;
            margin-bottom: 8px;
            padding-bottom: 6px;
            border-bottom: 1px solid rgba(0,212,255,0.2);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .setup-direction {
            font-size: 0.75rem;
            font-weight: 700;
            padding: 2px 8px;
            border-radius: 4px;
        }
        .setup-direction.long { background: rgba(0,255,136,0.2); color: #00ff88; }
        .setup-direction.short { background: rgba(255,68,68,0.2); color: #ff4444; }
        .setup-direction.none { background: rgba(255,193,7,0.2); color: #ffc107; }
        .setup-levels {
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            gap: 6px;
            margin: 8px 0;
        }
        .setup-level-box {
            text-align: center;
            padding: 6px 4px;
            border-radius: 4px;
            font-size: 0.7rem;
        }
        .setup-level-box.entry { background: rgba(0,212,255,0.15); border: 1px solid rgba(0,212,255,0.3); }
        .setup-level-box.sl { background: rgba(255,68,68,0.15); border: 1px solid rgba(255,68,68,0.3); }
        .setup-level-box.tp { background: rgba(0,255,136,0.15); border: 1px solid rgba(0,255,136,0.3); }
        .setup-level-label { display: block; font-size: 0.6rem; color: #888; margin-bottom: 2px; }
        .setup-level-price { display: block; font-weight: 600; color: #e0e0e0; }
        .setup-meta {
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
            margin-top: 6px;
        }
        .setup-tag {
            font-size: 0.6rem;
            padding: 2px 6px;
            border-radius: 3px;
            background: rgba(255,255,255,0.05);
            color: #aaa;
        }
        .setup-tag.good { background: rgba(0,255,136,0.15); color: #00ff88; }
        .setup-tag.warn { background: rgba(255,193,7,0.15); color: #ffc107; }
        .setup-tag.bad { background: rgba(255,68,68,0.15); color: #ff4444; }
        .setup-confidence {
            height: 4px;
            background: #1a1a2e;
            border-radius: 2px;
            margin-top: 8px;
            overflow: hidden;
        }
        .setup-confidence-fill {
            height: 100%;
            border-radius: 2px;
            transition: width 0.5s ease;
        }

        /* ===== TOP 5 SCANNER PANEL ===== */
        .top5-panel {
            background: linear-gradient(135deg, rgba(224,64,251,0.08) 0%, rgba(0,212,255,0.05) 100%);
            border: 1px solid rgba(224,64,251,0.3);
            border-radius: 8px;
            padding: 10px 14px;
            margin-bottom: 10px;
        }
        .top5-panel h3 {
            font-size: 0.8rem;
            color: #e040fb;
            margin-bottom: 8px;
            padding-bottom: 6px;
            border-bottom: 1px solid rgba(224,64,251,0.2);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .top5-scanner-status {
            font-size: 0.55rem;
            color: #666;
            padding: 2px 6px;
            border-radius: 3px;
            background: rgba(255,255,255,0.05);
        }
        .top5-scanner-status.scanning {
            color: #ffc107;
            background: rgba(255,193,7,0.15);
            animation: pulse 1.5s infinite;
        }
        @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.5; } }
        .top5-item {
            padding: 8px;
            margin-bottom: 6px;
            border-radius: 6px;
            background: rgba(255,255,255,0.03);
            border-left: 3px solid #888;
            cursor: pointer;
            transition: background 0.2s;
        }
        .top5-item:hover { background: rgba(255,255,255,0.06); }
        .top5-item.qualified { border-left-color: #e040fb; }
        .top5-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 4px;
        }
        .top5-pair {
            font-size: 0.85rem;
            font-weight: 700;
            color: #e0e0e0;
        }
        .top5-tf {
            font-size: 0.65rem;
            color: #888;
            background: rgba(255,255,255,0.05);
            padding: 1px 5px;
            border-radius: 3px;
        }
        .top5-rps {
            font-size: 0.75rem;
            font-weight: 700;
            color: #e040fb;
            margin-bottom: 3px;
        }
        .top5-rps .rps-value { color: #00d4ff; }
        .top5-levels {
            display: flex;
            gap: 8px;
            font-size: 0.6rem;
            color: #888;
            flex-wrap: wrap;
        }
        .top5-levels span { white-space: nowrap; }
        .top5-levels .rps-highlight { color: #e040fb; font-weight: 600; }
        .top5-levels .eq-highlight { color: #ffc107; font-weight: 600; }
        .top5-tags {
            display: flex;
            gap: 4px;
            margin-top: 3px;
            flex-wrap: wrap;
        }
        .top5-tag {
            font-size: 0.5rem;
            padding: 1px 4px;
            border-radius: 2px;
            background: rgba(0,255,136,0.1);
            color: #00ff88;
        }
        .top5-tag.duration { background: rgba(0,212,255,0.1); color: #00d4ff; }
        .top5-tag.vol { background: rgba(255,193,7,0.1); color: #ffc107; }
        .top5-empty {
            text-align: center;
            padding: 20px 10px;
            color: #555;
            font-size: 0.75rem;
        }
        .top5-rank {
            font-size: 0.6rem;
            color: #e040fb;
            font-weight: 700;
            margin-right: 6px;
        }

        /* ===== FORMING MODELS PANEL ===== */
        .forming-panel {
            background: linear-gradient(135deg, rgba(255,193,7,0.08) 0%, rgba(224,64,251,0.05) 100%);
            border: 1px solid rgba(255,193,7,0.3);
            border-radius: 8px;
            padding: 10px 14px;
            margin-bottom: 10px;
        }
        .forming-panel h3 {
            font-size: 0.8rem;
            color: #ffc107;
            margin-bottom: 8px;
            padding-bottom: 6px;
            border-bottom: 1px solid rgba(255,193,7,0.2);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .forming-count {
            font-size: 0.55rem;
            color: #888;
            padding: 2px 6px;
            border-radius: 3px;
            background: rgba(255,255,255,0.05);
        }
        .forming-link {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 6px 8px;
            margin-bottom: 4px;
            border-radius: 5px;
            background: rgba(255,255,255,0.03);
            border-left: 3px solid #888;
            text-decoration: none;
            color: #e0e0e0;
            font-size: 0.75rem;
            font-weight: 600;
            transition: background 0.2s;
            cursor: pointer;
        }
        .forming-link:hover { background: rgba(255,255,255,0.08); }
        .forming-link.bullish { border-left-color: #00ff88; }
        .forming-link.bearish { border-left-color: #ff4444; }
        .forming-link .pair-name { flex: 1; }
        .forming-link .model-type {
            font-size: 0.6rem;
            font-weight: 400;
            color: #888;
            margin-left: 6px;
        }
        .forming-link .model-type.accum { color: #00ff88; }
        .forming-link .model-type.dist { color: #ff4444; }
        .forming-link .status-dot {
            width: 6px; height: 6px; border-radius: 50%; margin-left: 6px;
        }
        .forming-link .status-dot.forming { background: #ffc107; }
        .forming-link .status-dot.confirmed { background: #00ff88; }
        .forming-link .tf-badge {
            font-size: 0.55rem;
            color: #666;
            background: rgba(255,255,255,0.05);
            padding: 1px 4px;
            border-radius: 2px;
            margin-left: 4px;
        }
        .forming-empty {
            text-align: center;
            padding: 12px 8px;
            color: #555;
            font-size: 0.7rem;
        }

        /* ===== RISK MANAGEMENT (TCT Lecture 7) ===== */
        .risk-section { border-left: 3px solid #ffc107; }
        .risk-section h3 { color: #ffc107 !important; }
        .risk-input-group {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 6px;
            margin-bottom: 8px;
        }
        .risk-input {
            display: flex;
            flex-direction: column;
            gap: 2px;
        }
        .risk-input label {
            font-size: 0.65rem;
            color: #888;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .risk-input input, .risk-input select {
            background: #1a1a2e;
            border: 1px solid #2d2d44;
            border-radius: 4px;
            padding: 5px 8px;
            color: #e0e0e0;
            font-size: 0.8rem;
            font-family: inherit;
            outline: none;
            transition: border-color 0.2s;
        }
        .risk-input input:focus, .risk-input select:focus {
            border-color: #ffc107;
        }
        .risk-input.full-width {
            grid-column: 1 / -1;
        }
        .calc-btn {
            background: linear-gradient(135deg, #ffc107, #ff9800);
            color: #0a0a0f;
            border: none;
            padding: 8px 12px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 0.8rem;
            font-weight: 700;
            width: 100%;
            margin-top: 4px;
            letter-spacing: 0.5px;
            transition: opacity 0.2s;
        }
        .calc-btn:hover { opacity: 0.85; }
        .risk-results {
            margin-top: 10px;
            display: none;
        }
        .risk-results.active { display: block; }
        .risk-result-card {
            background: #1a1a2e;
            border-radius: 6px;
            padding: 10px;
            margin-bottom: 8px;
        }
        .risk-result-card h4 {
            font-size: 0.75rem;
            color: #ffc107;
            margin-bottom: 6px;
            padding-bottom: 4px;
            border-bottom: 1px solid #2d2d44;
        }
        .result-row {
            display: flex;
            justify-content: space-between;
            padding: 3px 0;
            font-size: 0.75rem;
        }
        .result-row .r-label { color: #888; }
        .result-row .r-value { color: #e0e0e0; font-weight: 600; }
        .result-row .r-value.profit { color: #00ff88; }
        .result-row .r-value.loss { color: #ff4444; }
        .result-row .r-value.highlight { color: #ffc107; }
        .result-row .r-value.info { color: #00d4ff; }

        /* Risk tabs */
        .risk-tabs {
            display: flex;
            gap: 2px;
            margin-bottom: 10px;
            background: #0a0a0f;
            border-radius: 4px;
            padding: 2px;
        }
        .risk-tab {
            flex: 1;
            padding: 5px 4px;
            text-align: center;
            font-size: 0.65rem;
            color: #888;
            background: transparent;
            border: none;
            border-radius: 3px;
            cursor: pointer;
            font-family: inherit;
            transition: all 0.2s;
        }
        .risk-tab.active {
            background: #ffc107;
            color: #0a0a0f;
            font-weight: 700;
        }
        .risk-tab-content { display: none; }
        .risk-tab-content.active { display: block; }

        /* Equity chart */
        .equity-chart-container {
            background: #1a1a2e;
            border-radius: 6px;
            padding: 10px;
            margin-top: 8px;
        }
        .equity-canvas {
            width: 100%;
            height: 160px;
            display: block;
        }
        .equity-legend {
            display: flex;
            justify-content: center;
            gap: 15px;
            margin-top: 6px;
            font-size: 0.65rem;
        }
        .equity-legend span {
            display: flex;
            align-items: center;
            gap: 4px;
        }
        .legend-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            display: inline-block;
        }
        .legend-dot.trader-a { background: #00ff88; }
        .legend-dot.trader-b { background: #ff4444; }

        /* Compounding table */
        .compound-table {
            width: 100%;
            font-size: 0.7rem;
            border-collapse: collapse;
            margin-top: 6px;
        }
        .compound-table th {
            color: #888;
            font-weight: 600;
            text-align: left;
            padding: 4px 6px;
            border-bottom: 1px solid #2d2d44;
            font-size: 0.65rem;
        }
        .compound-table td {
            padding: 4px 6px;
            color: #e0e0e0;
        }
        .compound-table tr:nth-child(even) td {
            background: rgba(255, 255, 255, 0.02);
        }
        .compound-growth { color: #00ff88 !important; font-weight: 600; }

        /* Streak simulation */
        .streak-bar-container {
            display: flex;
            gap: 2px;
            align-items: flex-end;
            height: 80px;
            margin-top: 8px;
            padding: 0 4px;
        }
        .streak-bar {
            flex: 1;
            border-radius: 2px 2px 0 0;
            position: relative;
            min-height: 4px;
            transition: height 0.3s ease;
        }
        .streak-bar.loss-bar { background: linear-gradient(to top, #ff4444, #ff6b6b); }
        .streak-bar.win-bar { background: linear-gradient(to top, #00ff88, #00ffaa); }
        .streak-bar .bar-label {
            position: absolute;
            bottom: -16px;
            left: 50%;
            transform: translateX(-50%);
            font-size: 0.55rem;
            color: #666;
            white-space: nowrap;
        }
        .streak-result {
            text-align: center;
            margin-top: 22px;
            font-size: 0.75rem;
            padding: 6px;
            border-radius: 4px;
        }
        .streak-result.positive {
            background: rgba(0, 255, 136, 0.1);
            color: #00ff88;
        }
        .streak-result.negative {
            background: rgba(255, 68, 68, 0.1);
            color: #ff4444;
        }

        /* Warning box */
        .risk-warning {
            background: rgba(255, 193, 7, 0.08);
            border: 1px solid rgba(255, 193, 7, 0.3);
            border-radius: 4px;
            padding: 8px;
            margin-top: 8px;
            font-size: 0.65rem;
            color: #ffc107;
            line-height: 1.4;
        }
        .risk-warning strong { color: #ff9800; }

        /* ===== PO3 SCHEMATICS (TCT Lecture 8) ===== */
        .po3-section { border-left: 3px solid #e040fb; }
        .po3-section h3 { color: #e040fb !important; }
        .po3-item {
            background: #1a1a2e;
            border-radius: 6px;
            padding: 10px;
            margin-bottom: 8px;
            border-left: 3px solid #e040fb;
        }
        .po3-item.bullish { border-left-color: #00ff88; }
        .po3-item.bearish { border-left-color: #ff4444; }
        .po3-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 6px;
        }
        .po3-type {
            font-size: 0.75rem;
            font-weight: bold;
            color: #e0e0e0;
        }
        .po3-phase {
            font-size: 0.6rem;
            padding: 2px 6px;
            border-radius: 3px;
            text-transform: uppercase;
            font-weight: 600;
        }
        .po3-phase.range { background: rgba(224, 64, 251, 0.2); color: #e040fb; }
        .po3-phase.manipulation { background: rgba(255, 152, 0, 0.2); color: #ff9800; }
        .po3-phase.manipulation_complete { background: rgba(0, 188, 212, 0.2); color: #00bcd4; }
        .po3-phase.expansion { background: rgba(0, 255, 136, 0.2); color: #00ff88; }
        .po3-quality {
            font-size: 0.65rem;
            padding: 2px 6px;
            border-radius: 3px;
            background: rgba(224, 64, 251, 0.2);
            color: #e040fb;
        }
        .po3-levels {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 6px;
            margin-top: 6px;
        }
        .po3-level-box {
            text-align: center;
            padding: 4px;
            border-radius: 4px;
            font-size: 0.7rem;
        }
        .po3-level-box.entry { background: rgba(0, 212, 255, 0.15); color: #00d4ff; }
        .po3-level-box.stop { background: rgba(255, 68, 68, 0.15); color: #ff4444; }
        .po3-level-box.target { background: rgba(0, 255, 136, 0.15); color: #00ff88; }
        .po3-level-label { font-size: 0.6rem; color: #888; display: block; }
        .po3-level-price { font-weight: bold; }
        .po3-meta {
            display: flex;
            gap: 8px;
            margin-top: 6px;
            font-size: 0.65rem;
            color: #888;
            flex-wrap: wrap;
        }
        .po3-meta .rr { color: #ffc107; }
        .po3-meta .tct-model { color: #00bcd4; }
        .po3-meta .exception { color: #ff9800; }
        .po3-meta .compression { color: #8bc34a; }
        .po3-meta .liq-both { color: #9c27b0; }
        .po3-manip-bar {
            margin-top: 6px;
            background: #12121a;
            border-radius: 4px;
            padding: 6px;
        }
        .po3-manip-bar .bar-track {
            height: 6px;
            background: #2d2d44;
            border-radius: 3px;
            position: relative;
            overflow: hidden;
        }
        .po3-manip-bar .bar-fill {
            height: 100%;
            border-radius: 3px;
            transition: width 0.3s;
        }
        .po3-manip-bar .bar-fill.bullish { background: linear-gradient(90deg, #ff4444, #00ff88); }
        .po3-manip-bar .bar-fill.bearish { background: linear-gradient(90deg, #00ff88, #ff4444); }
        .po3-manip-labels {
            display: flex;
            justify-content: space-between;
            font-size: 0.6rem;
            color: #666;
            margin-top: 3px;
        }
        /* Trade Execution (Lecture 9) — Teal/Cyan theme */
        .exec-section { border-left: 3px solid #00bcd4; }
        .exec-section .tct-lecture { color: #00bcd4; }
        .exec-tabs { display: flex; gap: 4px; margin: 10px 0 8px; }
        .exec-tab {
            flex: 1; padding: 6px 8px; border: 1px solid #2d2d44;
            background: #12121a; color: #888; font-size: 0.7rem;
            border-radius: 4px; cursor: pointer; text-align: center;
        }
        .exec-tab.active { background: rgba(0,188,212,0.15); color: #00bcd4; border-color: #00bcd4; }
        .exec-tab-content { display: none; }
        .exec-tab-content.active { display: block; }
        .exec-input-group {
            display: grid; grid-template-columns: 1fr 1fr; gap: 6px; margin-bottom: 10px;
        }
        .exec-input { display: flex; flex-direction: column; gap: 3px; }
        .exec-input label { font-size: 0.65rem; color: #888; text-transform: uppercase; }
        .exec-input input, .exec-input select {
            background: #12121a; border: 1px solid #2d2d44; color: #e0e0e0;
            padding: 6px 8px; border-radius: 4px; font-size: 0.8rem;
        }
        .exec-input input:focus { border-color: #00bcd4; outline: none; }
        .exec-btn {
            width: 100%; padding: 8px; background: linear-gradient(135deg, #00bcd4, #0097a7);
            color: #fff; border: none; border-radius: 4px; cursor: pointer;
            font-size: 0.8rem; font-weight: 600; margin-bottom: 10px;
        }
        .exec-btn:hover { opacity: 0.9; }
        .exec-results { display: none; }
        .exec-results.active { display: block; }
        .exec-result-card {
            background: #12121a; border-radius: 6px; padding: 10px;
            margin-bottom: 8px; border: 1px solid #1e1e2d;
        }
        .exec-result-card h4 {
            color: #00bcd4; font-size: 0.75rem; margin: 0 0 8px;
            text-transform: uppercase; letter-spacing: 0.5px;
        }
        .exec-result-card .result-row {
            display: flex; justify-content: space-between;
            padding: 3px 0; font-size: 0.75rem;
        }
        .exec-result-card .r-label { color: #888; }
        .exec-result-card .r-value { color: #e0e0e0; font-weight: 500; }
        .exec-result-card .r-value.safe { color: #00ff88; }
        .exec-result-card .r-value.danger { color: #ff4444; }
        .exec-result-card .r-value.highlight { color: #00bcd4; }
        .exec-result-card .r-value.profit { color: #00ff88; }
        .exec-result-card .r-value.loss { color: #ff4444; }
        .exec-checklist {
            list-style: none; padding: 0; margin: 0;
        }
        .exec-checklist li {
            padding: 4px 0; font-size: 0.7rem; color: #ccc;
            border-bottom: 1px solid #1e1e2d;
        }
        .exec-checklist li:before { content: "✓ "; color: #00bcd4; font-weight: bold; }
        .lev-compare-table {
            width: 100%; border-collapse: collapse; font-size: 0.65rem;
        }
        .lev-compare-table th {
            background: #1e1e2d; color: #00bcd4; padding: 4px 6px;
            text-align: left; font-weight: 500;
        }
        .lev-compare-table td {
            padding: 4px 6px; border-bottom: 1px solid #1e1e2d; color: #ccc;
        }
        .lev-compare-table tr.safe-row td { color: #00ff88; }
        .lev-compare-table tr.danger-row td { color: #ff4444; }
        .exec-warning {
            background: rgba(255,68,68,0.1); border: 1px solid rgba(255,68,68,0.3);
            border-radius: 4px; padding: 8px; margin-top: 8px;
            font-size: 0.7rem; color: #ff8888;
        }
        .exec-info {
            background: rgba(0,188,212,0.1); border: 1px solid rgba(0,188,212,0.3);
            border-radius: 4px; padding: 8px; margin-top: 8px;
            font-size: 0.7rem; color: #80deea;
        }

        /* ===== RESPONSIVE DESIGN ===== */

        /* Tablets landscape + small laptops (iPad Air, Asus PZ13 ProArt ~820-1100px) */
        @media (max-width: 1100px) {
            .main-container {
                grid-template-columns: 1fr 280px;
                gap: 10px;
                padding: 10px;
            }
            .header { padding: 10px 15px; }
            .header h1 { font-size: 1.2rem; }
            .header .price-display { font-size: 1.4rem; }
            .metric-card { padding: 10px; }
            .metric-card h3 { font-size: 0.8rem; }
        }

        /* Tablets portrait + large phones (iPad Air portrait ~820px, Asus PZ13 ProArt portrait) */
        @media (max-width: 860px) {
            .main-container {
                grid-template-columns: 1fr;
                height: auto;
                gap: 10px;
                padding: 10px;
            }
            .chart-section {
                height: 50vh;
                min-height: 350px;
            }
            #chart { height: calc(100% - 36px) !important; }
            .metrics-panel {
                max-height: none;
                overflow-y: visible;
                padding-right: 0;
            }
            .header {
                flex-direction: column;
                gap: 10px;
                align-items: flex-start;
                padding: 10px 15px;
            }
            .header > div {
                width: 100%;
                justify-content: space-between;
                flex-wrap: wrap;
                gap: 8px;
            }
            .timeframe-selector {
                flex-wrap: wrap;
                gap: 6px;
            }
            .pair-search { min-width: 120px; width: 120px; font-size: 0.8rem; }
            .pair-dropdown { min-width: 220px; }
            .header .price-display { font-size: 1.3rem; }
            /* Stack validation gates 2x2 on tablet */
            .validation-gates { grid-template-columns: repeat(4, 1fr); }
            /* Setup levels stay 3-col */
            .setup-levels { grid-template-columns: 1fr 1fr 1fr; }
            .schematic-levels { grid-template-columns: repeat(3, 1fr); }
            .po3-levels { grid-template-columns: repeat(3, 1fr); }
        }

        /* Phones (iPhone 13 ~390px, Samsung S22 ~360px) */
        @media (max-width: 480px) {
            body { font-size: 14px; -webkit-text-size-adjust: 100%; }
            .header {
                flex-direction: column;
                gap: 8px;
                align-items: stretch;
                padding: 8px 10px;
            }
            .header h1 {
                font-size: 1rem;
                justify-content: center;
                text-align: center;
            }
            .header > div {
                flex-direction: column;
                align-items: stretch;
                gap: 6px;
            }
            .timeframe-selector {
                flex-direction: row;
                flex-wrap: wrap;
                justify-content: space-between;
                gap: 6px;
                width: 100%;
            }
            .timeframe-selector > div {
                flex: 1;
                min-width: 0;
            }
            .pair-search-wrapper { display: block; width: 100%; }
            .pair-search {
                width: 100% !important;
                min-width: unset;
                font-size: 0.9rem;
                padding: 8px 12px;
            }
            .tf-dropdown {
                width: 100%;
                font-size: 0.85rem;
                padding: 8px 28px 8px 12px;
            }
            .pair-dropdown {
                min-width: unset;
                width: 100%;
                max-height: 300px;
            }
            .header .price-display {
                font-size: 1.5rem;
                text-align: center;
                width: 100%;
            }
            .refresh-btn {
                width: 100%;
                padding: 10px 12px;
                font-size: 0.85rem;
            }
            .main-container {
                grid-template-columns: 1fr;
                height: auto;
                gap: 8px;
                padding: 8px;
            }
            .chart-section {
                height: 45vh;
                min-height: 280px;
                border-radius: 6px;
            }
            .chart-controls {
                overflow-x: auto;
                -webkit-overflow-scrolling: touch;
                flex-wrap: nowrap;
                padding: 4px 6px;
            }
            .chart-ctrl-btn {
                padding: 6px 10px;
                font-size: 0.7rem;
                white-space: nowrap;
                flex-shrink: 0;
            }
            .metrics-panel {
                gap: 8px;
                padding-right: 0;
                max-height: none;
                overflow-y: visible;
            }
            .metric-card {
                padding: 10px;
                border-radius: 6px;
            }
            .metric-card h3 { font-size: 0.8rem; }
            .metric-row { padding: 3px 0; font-size: 0.78rem; }
            .tct-lecture { font-size: 0.6rem; }

            /* Touch-friendly tap targets */
            .pair-dropdown-item { padding: 10px 12px; font-size: 0.85rem; }
            .top5-item { padding: 10px; }
            .forming-link { padding: 8px 10px; font-size: 0.8rem; }
            .scan-btn { padding: 8px 12px; font-size: 0.75rem; }

            /* Validation gates: 2 per row on phone */
            .validation-gates {
                grid-template-columns: repeat(2, 1fr);
                gap: 4px;
            }
            .gate { padding: 8px 4px; font-size: 0.65rem; }

            /* Setup levels stack on phone */
            .setup-levels { grid-template-columns: 1fr 1fr 1fr; gap: 4px; }
            .setup-level-box { padding: 8px 4px; }

            /* Risk input 1-col on phone */
            .risk-input-group { grid-template-columns: 1fr; }
            .risk-input.full-width { grid-column: 1; }
            .risk-input input, .risk-input select { padding: 8px 10px; font-size: 0.85rem; }
            .calc-btn { padding: 10px 12px; font-size: 0.85rem; }

            /* Risk tabs scrollable */
            .risk-tabs { flex-wrap: wrap; }
            .risk-tab { padding: 6px 4px; font-size: 0.6rem; }

            /* Schematic levels stack */
            .schematic-levels { grid-template-columns: repeat(3, 1fr); gap: 4px; }
            .po3-levels { grid-template-columns: repeat(3, 1fr); gap: 4px; }

            /* Streak bar smaller */
            .streak-bar-container { height: 60px; }

            /* Compound table scroll */
            .compound-table { font-size: 0.65rem; }

            /* Top5 panel */
            .top5-panel, .forming-panel, .setup-panel {
                padding: 8px 10px;
            }
            .top5-levels { font-size: 0.55rem; }
            .top5-tags { gap: 3px; }
            .top5-tag { font-size: 0.5rem; }

            /* Zone items */
            .zone-item { padding: 8px; font-size: 0.75rem; }

            /* Range bar */
            .range-bar { height: 36px; }
        }

        /* Very small phones (under 360px) */
        @media (max-width: 360px) {
            .header h1 { font-size: 0.9rem; }
            .header .price-display { font-size: 1.3rem; }
            .chart-section { height: 40vh; min-height: 240px; }
            .setup-levels { grid-template-columns: 1fr; gap: 3px; }
            .schematic-levels { grid-template-columns: 1fr 1fr; }
            .po3-levels { grid-template-columns: 1fr 1fr; }
        }

        /* Desktop large screens (Asus ROG Zephyrus 16", desktop 1920px) */
        @media (min-width: 1440px) {
            .main-container {
                grid-template-columns: 1fr 360px;
                max-width: 1800px;
                margin: 0 auto;
            }
            .header {
                max-width: 1800px;
                margin: 0 auto;
            }
        }

        /* Touch device optimizations */
        @media (hover: none) and (pointer: coarse) {
            .chart-ctrl-btn { padding: 6px 12px; min-height: 36px; }
            .pair-dropdown-item { padding: 12px; min-height: 44px; }
            .tf-btn { padding: 8px 12px; min-height: 36px; }
            .top5-item { padding: 12px; }
            .forming-link { padding: 10px 12px; min-height: 44px; }
            .scan-btn { min-height: 40px; }
            .refresh-btn { min-height: 40px; }
            .calc-btn { min-height: 44px; }
            .risk-tab { min-height: 36px; }
        }

        /* Safe area insets for iPhone notch / dynamic island */
        @supports (padding: max(0px)) {
            .header {
                padding-left: max(15px, env(safe-area-inset-left));
                padding-right: max(15px, env(safe-area-inset-right));
                padding-top: max(10px, env(safe-area-inset-top));
            }
            .main-container {
                padding-left: max(10px, env(safe-area-inset-left));
                padding-right: max(10px, env(safe-area-inset-right));
                padding-bottom: max(10px, env(safe-area-inset-bottom));
            }
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>HPB-TCT Dashboard <span class="symbol" id="headerSymbol">BTCUSDT</span></h1>
        <div style="display: flex; align-items: center; gap: 15px;">
            <div class="timeframe-selector">
                <div>
                    <span class="tf-label">Pair</span>
                    <div class="pair-search-wrapper">
                        <input type="text" id="pairSearch" class="pair-search" value="BTCUSDT" autocomplete="off" spellcheck="false">
                        <div class="pair-dropdown" id="pairDropdown"></div>
                    </div>
                </div>
                <div>
                    <span class="tf-label">Timeframe</span>
                    <select id="tfDropdown" class="tf-dropdown">
                        <optgroup label="Low Timeframes (LTF)">
                            <option value="1m">1m</option>
                            <option value="5m">5m</option>
                            <option value="15m">15m</option>
                            <option value="30m">30m</option>
                        </optgroup>
                        <optgroup label="Mid Timeframes">
                            <option value="1h">1H</option>
                            <option value="2h">2H</option>
                            <option value="4h" selected>4H</option>
                            <option value="8h">8H</option>
                        </optgroup>
                        <optgroup label="High Timeframes (HTF)">
                            <option value="1d">1D</option>
                            <option value="1W">1W</option>
                            <option value="1M">1M</option>
                        </optgroup>
                    </select>
                </div>
            </div>
            <div class="price-display" id="currentPrice">--</div>
            <button class="refresh-btn" onclick="refreshData()">Refresh</button>
        </div>
    </div>

    <div class="main-container">
        <div class="chart-section">
            <div class="chart-controls">
                <button class="chart-ctrl-btn" onclick="chartFit()" title="Fit all candles in view">Fit</button>
                <button class="chart-ctrl-btn" onclick="chartScalePrice()" title="Scale price axis to visible candles only">Scale Price</button>
                <button class="chart-ctrl-btn" onclick="chartZoomIn()" title="Zoom in">Zoom In</button>
                <button class="chart-ctrl-btn" onclick="chartZoomOut()" title="Zoom out">Zoom Out</button>
                <button class="chart-ctrl-btn" onclick="chartReset()" title="Reset chart to default view">Reset</button>
            </div>
            <div id="chart"></div>
        </div>

        <div class="metrics-panel">
            <!-- Top 5 Ranges (Range Probability Scanner) -->
            <div class="top5-panel" id="top5Panel">
                <h3>Top 5 Ranges <span class="top5-scanner-status" id="scannerStatus">Initializing...</span></h3>
                <div id="top5Content">
                    <div class="top5-empty">Range scanner starting — scanning pairs using detect_ranges logic...</div>
                </div>
            </div>

            <!-- Market Structure (TCT Lecture 1 – 6-Candle Rule, Level 1/2/3, BOS, EOF) -->
            <div class="metric-card">
                <h3>Market Structure <span class="badge badge-neutral" id="trendBadge">--</span></h3>
                <div class="tct-lecture">TCT Lecture 1 &mdash; 6-Candle Rule &bull; Level 1/2/3 &bull; BOS &bull; EOF</div>
                <div class="metric-row">
                    <span class="label">HTF Trend (4H)</span>
                    <span class="value" id="htfTrend">--</span>
                </div>
                <div class="metric-row">
                    <span class="label">LTF Trend (15m)</span>
                    <span class="value" id="ltfTrend">--</span>
                </div>
                <div class="metric-row">
                    <span class="label">HTF EOF</span>
                    <span class="value" id="htfEOF" style="font-size:0.65rem;">--</span>
                </div>
                <div class="metric-row">
                    <span class="label">LTF EOF</span>
                    <span class="value" id="ltfEOF" style="font-size:0.65rem;">--</span>
                </div>
                <div id="msLevelsContent" style="margin-top:6px;"></div>
                <div class="metric-row">
                    <span class="label">MS Highs / Lows</span>
                    <span class="value" id="msPivotCounts">--</span>
                </div>
                <div id="bosEventsContent" style="margin-top:6px;"></div>
                <div id="dominoContent" style="margin-top:6px;"></div>
            </div>

            <!-- Active Range -->
            <div class="metric-card">
                <h3>Active Range <span class="badge" id="zoneBadge">--</span></h3>
                <div class="tct-lecture">TCT Lecture 2 - Ranges</div>
                <div id="rangeQualityTag" style="margin-bottom:6px;"></div>
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
                <div class="metric-row">
                    <span class="label">Range Size</span>
                    <span class="value" id="rangeSize">--</span>
                </div>
                <div class="metric-row">
                    <span class="label">DL+ / DL&minus;</span>
                    <span class="value" id="rangeDL" style="font-size:0.65rem;">--</span>
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
                <div id="fibContent"></div>
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
                    <span class="label">Demand OBs</span>
                    <span class="value bullish" id="demandOBCount">--</span>
                </div>
                <div class="metric-row">
                    <span class="label">Supply OBs</span>
                    <span class="value bearish" id="supplyOBCount">--</span>
                </div>
                <div class="metric-row">
                    <span class="label">Structure S/D</span>
                    <span class="value" id="structureSDCount">--</span>
                </div>
                <div class="metric-row">
                    <span class="label">Mitigated</span>
                    <span class="value" id="mitigatedCount" style="color:#888;">--</span>
                </div>
                <div id="topScoredZones" style="margin-top:6px;"></div>
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
                    <span class="label">&emsp;Primary / Internal</span>
                    <span class="value" id="bslBreakdown" style="font-size:0.7rem;">--</span>
                </div>
                <div class="metric-row">
                    <span class="label">SSL Pools (Below)</span>
                    <span class="value bullish" id="sslCount">--</span>
                </div>
                <div class="metric-row">
                    <span class="label">&emsp;Primary / Internal</span>
                    <span class="value" id="sslBreakdown" style="font-size:0.7rem;">--</span>
                </div>
                <div class="metric-row">
                    <span class="label">Equal Highs</span>
                    <span class="value" id="eqHighs" style="color:#9c27b0;">--</span>
                </div>
                <div class="metric-row">
                    <span class="label">Equal Lows</span>
                    <span class="value" id="eqLows" style="color:#9c27b0;">--</span>
                </div>
                <div id="liqCurvesContent" style="margin-top:6px;"></div>
                <div id="liqTargetsContent" style="margin-top:4px;"></div>
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
                <div style="display:flex;gap:6px;margin:6px 0;">
                    <button class="scan-btn" id="scanTCTBtn" onclick="scanTCTSchematics()" title="Scan live price action for TCT schematics on HTF/MTF/LTF">
                        <span id="scanTCTIcon">&#x1F50D;</span> Scan Price Action
                    </button>
                </div>
                <div id="schematicsContent">
                    <div class="metric-row">
                        <span class="label">Loading...</span>
                    </div>
                </div>
            </div>

            <!-- PO3 Schematics (Lecture 8) -->
            <div class="metric-card po3-section">
                <h3>PO3 Schematics <span class="badge badge-neutral" id="po3Badge">--</span></h3>
                <div class="tct-lecture">Lecture 8 — Power of Three: Range &rarr; Manipulation &rarr; Expansion</div>
                <div style="display:flex;gap:6px;margin:6px 0;">
                    <button class="scan-btn" id="scanPO3Btn" onclick="scanPO3Schematics()" title="Scan live price action for PO3 schematics on HTF/MTF/LTF">
                        <span id="scanPO3Icon">&#x1F50D;</span> Scan Price Action
                    </button>
                </div>
                <div id="po3Content">
                    <div class="metric-row">
                        <span class="label">Loading...</span>
                    </div>
                </div>
            </div>

            <!-- Forming TCT Models (derived from current pair analysis) -->
            <div class="forming-panel" id="formingPanel">
                <h3>Forming Models <span class="forming-count" id="formingCount">--</span></h3>
                <div id="formingContent">
                    <div class="forming-empty">Select a pair to analyze forming schematics...</div>
                </div>
            </div>

            <!-- Highest Probability Setup (derived from all sections above) -->
            <div class="setup-panel" id="setupPanel">
                <h3>Highest Probability Setup <span class="setup-direction none" id="setupDirection">--</span></h3>
                <div id="setupContent">
                    <div class="metric-row"><span class="label">Select a pair to analyze...</span></div>
                </div>
                <div class="setup-confidence">
                    <div class="setup-confidence-fill" id="setupConfidence" style="width: 0%; background: #ffc107;"></div>
                </div>
            </div>

            <!-- Risk Management (Lecture 7) -->
            <div class="metric-card risk-section">
                <h3>Risk Management <span class="badge" style="background:rgba(255,193,7,0.2);color:#ffc107;">L7</span></h3>
                <div class="tct-lecture">TCT Lecture 7 — "The Most Important Lecture"</div>

                <div class="risk-tabs">
                    <button class="risk-tab active" data-tab="calculator" onclick="switchRiskTab('calculator')">Calculator</button>
                    <button class="risk-tab" data-tab="equity" onclick="switchRiskTab('equity')">Equity Sim</button>
                    <button class="risk-tab" data-tab="compound" onclick="switchRiskTab('compound')">Compound</button>
                </div>

                <!-- Tab 1: Position Size Calculator -->
                <div class="risk-tab-content active" id="tab-calculator">
                    <div class="risk-input-group">
                        <div class="risk-input">
                            <label>Account Balance ($)</label>
                            <input type="number" id="riskBalance" value="10000" min="1" step="100">
                        </div>
                        <div class="risk-input">
                            <label>Risk Per Trade (%)</label>
                            <input type="number" id="riskPct" value="1" min="0.1" max="3" step="0.1">
                        </div>
                        <div class="risk-input">
                            <label>Stop-Loss Size (%)</label>
                            <input type="number" id="riskSL" value="0.26" min="0.01" max="50" step="0.01">
                        </div>
                        <div class="risk-input">
                            <label>Risk:Reward</label>
                            <input type="number" id="riskRR" value="3" min="0.5" max="20" step="0.1">
                        </div>
                        <div class="risk-input">
                            <label>Market</label>
                            <select id="riskMarket" onchange="toggleGoldPrice()">
                                <option value="crypto">Crypto</option>
                                <option value="forex">Forex</option>
                                <option value="gold">Gold</option>
                            </select>
                        </div>
                        <div class="risk-input">
                            <label>Leverage (x)</label>
                            <input type="number" id="riskLeverage" value="10" min="1" max="200" step="1">
                        </div>
                        <div class="risk-input" id="goldPriceGroup" style="display:none;">
                            <label>Gold Price ($)</label>
                            <input type="number" id="goldPrice" value="2000" min="100" step="10">
                        </div>
                    </div>
                    <button class="calc-btn" onclick="calculateRisk()">CALCULATE POSITION</button>

                    <div class="risk-results" id="riskResults">
                        <!-- Position Sizing -->
                        <div class="risk-result-card">
                            <h4>Position Sizing</h4>
                            <div class="result-row">
                                <span class="r-label">Risk Amount</span>
                                <span class="r-value loss" id="resRiskAmt">--</span>
                            </div>
                            <div class="result-row">
                                <span class="r-label">Position Size</span>
                                <span class="r-value highlight" id="resPosSize">--</span>
                            </div>
                            <div class="result-row" id="resLotsRow" style="display:none;">
                                <span class="r-label">Lots</span>
                                <span class="r-value highlight" id="resLots">--</span>
                            </div>
                        </div>

                        <!-- Leverage & Margin -->
                        <div class="risk-result-card">
                            <h4>Leverage & Margin</h4>
                            <div class="result-row">
                                <span class="r-label">Min Leverage Needed</span>
                                <span class="r-value info" id="resMinLev">--</span>
                            </div>
                            <div class="result-row">
                                <span class="r-label">Used Margin</span>
                                <span class="r-value" id="resMargin">--</span>
                            </div>
                            <div class="result-row">
                                <span class="r-label">Free Margin</span>
                                <span class="r-value profit" id="resFreeMargin">--</span>
                            </div>
                        </div>

                        <!-- Trade Outcome -->
                        <div class="risk-result-card">
                            <h4>Trade Outcome</h4>
                            <div class="result-row">
                                <span class="r-label">Loss at Stop</span>
                                <span class="r-value loss" id="resLoss">--</span>
                            </div>
                            <div class="result-row">
                                <span class="r-label">Profit at Target</span>
                                <span class="r-value profit" id="resProfit">--</span>
                            </div>
                            <div class="result-row">
                                <span class="r-label">Account Gain</span>
                                <span class="r-value profit" id="resGainPct">--</span>
                            </div>
                        </div>

                        <!-- Streak Simulation -->
                        <div class="risk-result-card">
                            <h4>Worst-Case Streak (6L then 3W)</h4>
                            <div class="streak-bar-container" id="streakBars"></div>
                            <div class="streak-result" id="streakResult"></div>
                        </div>

                        <div class="risk-warning">
                            <strong>TCT Rules:</strong> Risk 1-3% per trade. Always use ISOLATED margin. Ensure liquidation price stays outside your stop-loss. Leverage does NOT determine risk — position size does.
                        </div>
                    </div>
                </div>

                <!-- Tab 2: Equity Simulator (Trader A vs B) -->
                <div class="risk-tab-content" id="tab-equity">
                    <div class="risk-input-group">
                        <div class="risk-input">
                            <label>Starting Balance ($)</label>
                            <input type="number" id="eqBalance" value="10000" min="100" step="100">
                        </div>
                        <div class="risk-input">
                            <label>Num Trades</label>
                            <input type="number" id="eqTrades" value="50" min="10" max="200" step="5">
                        </div>
                        <div class="risk-input">
                            <label>Trader A Risk (%)</label>
                            <input type="number" id="eqRiskA" value="1" min="0.5" max="3" step="0.5">
                        </div>
                        <div class="risk-input">
                            <label>Trader B Risk (%)</label>
                            <input type="number" id="eqRiskB" value="10" min="3" max="50" step="1">
                        </div>
                    </div>
                    <button class="calc-btn" onclick="simulateEquity()">SIMULATE EQUITY CURVES</button>
                    <div class="equity-chart-container" id="equityChartContainer" style="display:none;">
                        <canvas id="equityCanvas" class="equity-canvas"></canvas>
                        <div class="equity-legend">
                            <span><span class="legend-dot trader-a"></span> Trader A (disciplined)</span>
                            <span><span class="legend-dot trader-b"></span> Trader B (over-risking)</span>
                        </div>
                        <div class="result-row" style="margin-top:8px;">
                            <span class="r-label">Trader A Final</span>
                            <span class="r-value profit" id="eqFinalA">--</span>
                        </div>
                        <div class="result-row">
                            <span class="r-label">Trader B Final</span>
                            <span class="r-value loss" id="eqFinalB">--</span>
                        </div>
                    </div>
                    <div class="risk-warning" style="margin-top:8px;">
                        <strong>Trader A</strong> risks small, grows steadily. <strong>Trader B</strong> over-risks, gets "spikes of hope" but always reverts to net zero. Trading is a longevity game.
                    </div>
                </div>

                <!-- Tab 3: Compounding Projections -->
                <div class="risk-tab-content" id="tab-compound">
                    <div class="risk-input-group">
                        <div class="risk-input">
                            <label>Starting Capital ($)</label>
                            <input type="number" id="compBalance" value="10000" min="100" step="100">
                        </div>
                        <div class="risk-input">
                            <label>Weekly Gain (%)</label>
                            <input type="number" id="compWeekly" value="5" min="1" max="20" step="0.5">
                        </div>
                        <div class="risk-input">
                            <label>Trading Weeks/Year</label>
                            <input type="number" id="compWeeks" value="35" min="20" max="52" step="1">
                        </div>
                        <div class="risk-input">
                            <label>Years</label>
                            <input type="number" id="compYears" value="3" min="1" max="10" step="1">
                        </div>
                    </div>
                    <button class="calc-btn" onclick="calculateCompounding()">PROJECT GROWTH</button>
                    <div id="compoundResults" style="display:none;">
                        <table class="compound-table">
                            <thead>
                                <tr><th>Year</th><th>Balance</th><th>Growth</th></tr>
                            </thead>
                            <tbody id="compoundBody"></tbody>
                        </table>
                        <div class="risk-warning" style="margin-top:8px;">
                            <strong>5% per week</strong> = ~1% per day. Achievable with just 2 setups/week at 2.5 R:R. The power of compounding turns small, consistent gains into life-changing wealth.
                        </div>
                    </div>
                </div>
            </div>

            <!-- Trade Execution (Lecture 9) -->
            <div class="metric-card exec-section">
                <h3>Trade Execution <span class="badge" style="background:rgba(0,188,212,0.2);color:#00bcd4;">L9</span></h3>
                <div class="tct-lecture">TCT Lecture 9 — Exchange Settings, Leverage Safety & Execution Plan</div>

                <div class="exec-tabs">
                    <button class="exec-tab active" data-tab="exec-planner" onclick="switchExecTab('exec-planner')">Planner</button>
                    <button class="exec-tab" data-tab="exec-leverage" onclick="switchExecTab('exec-leverage')">Leverage</button>
                    <button class="exec-tab" data-tab="exec-capital" onclick="switchExecTab('exec-capital')">Capital</button>
                </div>

                <!-- Tab 1: Execution Planner -->
                <div class="exec-tab-content active" id="tab-exec-planner">
                    <div class="exec-input-group">
                        <div class="exec-input">
                            <label>Account Balance ($)</label>
                            <input type="number" id="execBalance" value="10000" min="1" step="100">
                        </div>
                        <div class="exec-input">
                            <label>Risk Per Trade (%)</label>
                            <input type="number" id="execRiskPct" value="1" min="0.1" max="3" step="0.1">
                        </div>
                        <div class="exec-input">
                            <label>Entry Price ($)</label>
                            <input type="number" id="execEntry" value="100000" min="0.01" step="0.01">
                        </div>
                        <div class="exec-input">
                            <label>Stop-Loss Price ($)</label>
                            <input type="number" id="execSL" value="99500" min="0.01" step="0.01">
                        </div>
                        <div class="exec-input">
                            <label>Take-Profit Price ($)</label>
                            <input type="number" id="execTP" value="101500" min="0.01" step="0.01">
                        </div>
                        <div class="exec-input">
                            <label>TP2 Price (optional)</label>
                            <input type="number" id="execTP2" value="" min="0.01" step="0.01" placeholder="Auto">
                        </div>
                        <div class="exec-input">
                            <label>Direction</label>
                            <select id="execDirection">
                                <option value="long">LONG</option>
                                <option value="short">SHORT</option>
                            </select>
                        </div>
                        <div class="exec-input">
                            <label>Leverage (x)</label>
                            <input type="number" id="execLeverage" value="10" min="1" max="400" step="1">
                        </div>
                    </div>
                    <button class="exec-btn" onclick="generateExecutionPlan()">GENERATE EXECUTION PLAN</button>

                    <div class="exec-results" id="execResults">
                        <!-- Exchange Settings -->
                        <div class="exec-result-card">
                            <h4>Exchange Settings</h4>
                            <div class="result-row">
                                <span class="r-label">Margin Mode</span>
                                <span class="r-value highlight">ISOLATED</span>
                            </div>
                            <div class="result-row">
                                <span class="r-label">Order Type</span>
                                <span class="r-value">MARKET (95%)</span>
                            </div>
                            <div class="result-row">
                                <span class="r-label">Quantity Mode</span>
                                <span class="r-value">By Quantity (USDT)</span>
                            </div>
                            <div class="result-row">
                                <span class="r-label">Direction</span>
                                <span class="r-value highlight" id="execResDirection">--</span>
                            </div>
                        </div>

                        <!-- Position Sizing -->
                        <div class="exec-result-card">
                            <h4>Position Sizing</h4>
                            <div class="result-row">
                                <span class="r-label">Risk Amount</span>
                                <span class="r-value loss" id="execResRisk">--</span>
                            </div>
                            <div class="result-row">
                                <span class="r-label">SL Distance</span>
                                <span class="r-value" id="execResSLPct">--</span>
                            </div>
                            <div class="result-row">
                                <span class="r-label">Position Size</span>
                                <span class="r-value highlight" id="execResPos">--</span>
                            </div>
                        </div>

                        <!-- Leverage Safety -->
                        <div class="exec-result-card">
                            <h4>Leverage Safety Check</h4>
                            <div class="result-row">
                                <span class="r-label">Selected Leverage</span>
                                <span class="r-value" id="execResLev">--</span>
                            </div>
                            <div class="result-row">
                                <span class="r-label">Margin Required</span>
                                <span class="r-value" id="execResMargin">--</span>
                            </div>
                            <div class="result-row">
                                <span class="r-label">Liquidation Price</span>
                                <span class="r-value" id="execResLiq">--</span>
                            </div>
                            <div class="result-row">
                                <span class="r-label">SL vs Liq Gap</span>
                                <span class="r-value" id="execResGap">--</span>
                            </div>
                            <div class="result-row">
                                <span class="r-label">Safety Status</span>
                                <span class="r-value" id="execResSafety">--</span>
                            </div>
                            <div class="result-row">
                                <span class="r-label">Max Safe Leverage</span>
                                <span class="r-value highlight" id="execResMaxLev">--</span>
                            </div>
                        </div>

                        <!-- Trade Outcome -->
                        <div class="exec-result-card">
                            <h4>Trade Outcome</h4>
                            <div class="result-row">
                                <span class="r-label">Risk : Reward</span>
                                <span class="r-value highlight" id="execResRR">--</span>
                            </div>
                            <div class="result-row">
                                <span class="r-label">Loss at SL</span>
                                <span class="r-value loss" id="execResLoss">--</span>
                            </div>
                            <div class="result-row">
                                <span class="r-label">Profit at TP</span>
                                <span class="r-value profit" id="execResProfit">--</span>
                            </div>
                            <div class="result-row">
                                <span class="r-label">Account Gain</span>
                                <span class="r-value profit" id="execResGain">--</span>
                            </div>
                        </div>

                        <!-- Partial TPs -->
                        <div class="exec-result-card">
                            <h4>Partial Take Profits</h4>
                            <div id="execTPLevels"></div>
                            <div class="result-row" style="border-top:1px solid #2d2d44;margin-top:4px;padding-top:4px;">
                                <span class="r-label">Total Expected Profit</span>
                                <span class="r-value profit" id="execResTotalTP">--</span>
                            </div>
                        </div>

                        <!-- Execution Checklist -->
                        <div class="exec-result-card">
                            <h4>Execution Checklist</h4>
                            <ul class="exec-checklist" id="execChecklist"></ul>
                        </div>

                        <div class="exec-info">
                            <strong>TCT Rule:</strong> Leverage only changes margin-to-position ratio, NOT risk. Same position = same risk regardless of leverage. But liquidation must ALWAYS be outside your stop-loss.
                        </div>
                    </div>
                </div>

                <!-- Tab 2: Leverage Comparison -->
                <div class="exec-tab-content" id="tab-exec-leverage">
                    <div class="exec-info" style="margin-top:0;margin-bottom:10px;">
                        Same position size across different leverages. Risk stays constant — only margin changes. Fill in the planner first, then view comparison here.
                    </div>
                    <div id="leverageCompare">
                        <table class="lev-compare-table">
                            <thead>
                                <tr><th>Leverage</th><th>Margin</th><th>Liq Price</th><th>Gap to SL</th><th>Status</th></tr>
                            </thead>
                            <tbody id="levCompareBody">
                                <tr><td colspan="5" style="text-align:center;color:#666;">Generate plan first</td></tr>
                            </tbody>
                        </table>
                    </div>
                </div>

                <!-- Tab 3: Capital Allocation -->
                <div class="exec-tab-content" id="tab-exec-capital">
                    <div class="exec-input-group">
                        <div class="exec-input">
                            <label>Total Capital ($)</label>
                            <input type="number" id="execCapital" value="50000" min="100" step="100">
                        </div>
                        <div class="exec-input">
                            <label>Exchange Allocation (%)</label>
                            <input type="number" id="execExchangePct" value="50" min="10" max="100" step="5">
                        </div>
                    </div>
                    <button class="exec-btn" onclick="calculateCapital()">CALCULATE ALLOCATION</button>
                    <div class="exec-results" id="capitalResults">
                        <div class="exec-result-card">
                            <h4>Capital Management</h4>
                            <div class="result-row">
                                <span class="r-label">Total Capital</span>
                                <span class="r-value" id="capTotal">--</span>
                            </div>
                            <div class="result-row">
                                <span class="r-label">On Exchange</span>
                                <span class="r-value highlight" id="capExchange">--</span>
                            </div>
                            <div class="result-row">
                                <span class="r-label">Off Exchange (Cold Storage)</span>
                                <span class="r-value" id="capOff">--</span>
                            </div>
                            <div class="result-row">
                                <span class="r-label">Recommendation</span>
                                <span class="r-value" id="capRec">--</span>
                            </div>
                        </div>
                        <div class="exec-info">
                            <strong>TCT Lecture 9:</strong> Never put all your money on one exchange. Keep 50-70% in cold storage or spread across wallets. Only risk what you can afford on-exchange.
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        let chart, candleSeries, lineSeries = [];
        let additionalSeries = []; // For liquidity curves
        let currentTimeframe = '4h';
        let currentSymbol = 'BTCUSDT';
        let coinList = { categories: {}, all: [] };
        let isLoading = false;
        let refreshGeneration = 0; // Incremented on each refresh request to abort stale ones
        let lastCandles = []; // Store candles for index-to-time mapping

        // HTF context cache: stores fetched HTF data per symbol so timeframe changes reuse it
        let htfCache = {
            symbol: null,
            rangesData: null,
            zonesData: null,
            liqData: null,
            valData: null,
            schematicsData: null,
            po3Data: null,
        };

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

        // Chart control functions
        function chartFit() {
            if (chart) chart.timeScale().fitContent();
        }
        function chartZoomIn() {
            if (!chart) return;
            const ts = chart.timeScale();
            const range = ts.getVisibleLogicalRange();
            if (range) {
                const mid = (range.from + range.to) / 2;
                const half = (range.to - range.from) / 4;
                ts.setVisibleLogicalRange({ from: mid - half, to: mid + half });
            }
        }
        function chartZoomOut() {
            if (!chart) return;
            const ts = chart.timeScale();
            const range = ts.getVisibleLogicalRange();
            if (range) {
                const mid = (range.from + range.to) / 2;
                const half = (range.to - range.from);
                ts.setVisibleLogicalRange({ from: mid - half, to: mid + half });
            }
        }
        function chartScalePrice() {
            if (!chart || !lastCandles || lastCandles.length === 0) return;
            // Override autoscale to only use candle OHLC data, ignoring price lines/overlays
            candleSeries.applyOptions({
                autoscaleInfoProvider: () => {
                    // Get the visible time range
                    const visRange = chart.timeScale().getVisibleLogicalRange();
                    if (!visRange) return null;
                    const fromIdx = Math.max(0, Math.floor(visRange.from));
                    const toIdx = Math.min(lastCandles.length - 1, Math.ceil(visRange.to));
                    let minPrice = Infinity, maxPrice = -Infinity;
                    for (let i = fromIdx; i <= toIdx; i++) {
                        const c = lastCandles[i];
                        if (c) {
                            if (c.low < minPrice) minPrice = c.low;
                            if (c.high > maxPrice) maxPrice = c.high;
                        }
                    }
                    if (minPrice === Infinity) return null;
                    const pad = (maxPrice - minPrice) * 0.05;
                    return {
                        priceRange: {
                            minValue: minPrice - pad,
                            maxValue: maxPrice + pad,
                        },
                    };
                }
            });
            // Trigger re-scale
            chart.priceScale('right').applyOptions({ autoScale: true });
        }
        function chartReset() {
            if (!chart || !lastCandles || lastCandles.length === 0) return;
            chart.timeScale().fitContent();
            // Remove custom autoscale override and restore default behavior
            candleSeries.applyOptions({ autoscaleInfoProvider: undefined });
            chart.priceScale('right').applyOptions({ autoScale: true });
        }

        // Fetch candle data from server (avoids CORS issues)
        async function fetchCandles(interval = '4h', limit = 100, symbol = null) {
            const sym = symbol || currentSymbol;
            try {
                const data = await fetchWithRetry(
                    `/api/candles?interval=${interval}&limit=${limit}&symbol=${sym}`,
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

        // Smart price formatter: adapts decimal places to price magnitude
        function fmtPrice(p) {
            if (p == null || isNaN(p)) return '--';
            if (p >= 10000) return '$' + p.toLocaleString(undefined, {maximumFractionDigits: 0});
            if (p >= 100)   return '$' + p.toLocaleString(undefined, {maximumFractionDigits: 2});
            if (p >= 1)     return '$' + p.toFixed(2);
            if (p >= 0.01)  return '$' + p.toFixed(4);
            return '$' + p.toPrecision(4);
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

        // Clear all price lines, additional series, and chart markers
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

            // Clear tap markers
            try { candleSeries.setMarkers([]); } catch(e) {}
        }

        // Fetch and display all TCT data
        async function refreshData() {
            // Abort-aware refresh: each call gets a generation number.
            // If a newer refresh starts, stale fetches are discarded.
            const thisGen = ++refreshGeneration;
            const isStale = () => refreshGeneration !== thisGen;

            // If another refresh is already running, it will detect staleness
            // via its own generation check and stop. Allow this one to proceed.
            isLoading = true;

            // Capture symbol at start to prevent race conditions when user switches
            // pairs during async API calls. All fetches use this snapshot so data
            // stays consistent even if currentSymbol changes mid-refresh.
            const targetSymbol = currentSymbol;
            const targetTimeframe = currentTimeframe;

            // Determine if this is a new pair (full fetch) or just a timeframe change (reuse HTF cache)
            const isNewPair = (htfCache.symbol !== targetSymbol);

            // Show loading states for all sections
            setLoading('trendBadge', true);
            setLoading('zoneBadge', true);
            setLoading('zoneCount', true);
            setLoading('liqCount', true);
            setLoading('actionBadge', true);
            setLoading('schematicsBadge', true);
            setLoading('po3Badge', true);
            document.getElementById('setupDirection').textContent = '...';
            document.getElementById('setupDirection').className = 'setup-direction none';
            document.getElementById('setupContent').innerHTML = '<div class="metric-row"><span class="label">Running analysis pipeline...</span></div>';
            document.getElementById('setupConfidence').style.width = '0%';
            document.getElementById('formingCount').textContent = '...';
            document.getElementById('formingContent').innerHTML = '<div class="forming-empty">Analyzing pair...</div>';

            // Reset sidebar to clear stale data from previous pair
            if (isNewPair) {
                // Market Structure
                document.getElementById('htfTrend').textContent = '--';
                document.getElementById('ltfTrend').textContent = '--';
                document.getElementById('htfEOF').textContent = '--';
                document.getElementById('ltfEOF').textContent = '--';
                document.getElementById('msPivotCounts').textContent = '--';
                document.getElementById('msLevelsContent').innerHTML = '';
                document.getElementById('bosEventsContent').innerHTML = '';
                document.getElementById('dominoContent').innerHTML = '';
                // Active Range
                document.getElementById('rangeHigh').textContent = '--';
                document.getElementById('rangeEq').textContent = '--';
                document.getElementById('rangeLow').textContent = '--';
                document.getElementById('rangeSize').textContent = '--';
                document.getElementById('rangeDL').textContent = '--';
                document.getElementById('rangeQualityTag').innerHTML = '';
                document.getElementById('tradingBias').textContent = '--';
                document.getElementById('tradingBias').className = 'value';
                document.getElementById('biasStrength').textContent = '--';
                document.getElementById('fibContent').innerHTML = '';
                document.getElementById('rangeViz').style.display = 'none';
                document.getElementById('rangeHighLabel').textContent = '--';
                document.getElementById('rangeLowLabel').textContent = '--';
                // Deviations
                document.getElementById('wickDevs').textContent = '0';
                document.getElementById('candleDevs').textContent = '0';
                document.getElementById('dlDevs').textContent = '0';
                document.getElementById('devBadge').textContent = '0';
                // S&D Zones
                document.getElementById('demandOBCount').textContent = '--';
                document.getElementById('supplyOBCount').textContent = '--';
                document.getElementById('structureSDCount').textContent = '--';
                document.getElementById('mitigatedCount').textContent = '--';
                document.getElementById('topScoredZones').innerHTML = '';
                document.getElementById('topZones').innerHTML = '';
                // Liquidity
                document.getElementById('bslBreakdown').textContent = '--';
                document.getElementById('sslBreakdown').textContent = '--';
                document.getElementById('liqCurvesContent').innerHTML = '';
                document.getElementById('liqTargetsContent').innerHTML = '';
                document.getElementById('liqPools').innerHTML = '';
                // Schematics
                document.getElementById('schematicsContent').innerHTML = '<div class="metric-row"><span class="label">Loading...</span></div>';
                document.getElementById('po3Content').innerHTML = '<div class="metric-row"><span class="label">Loading...</span></div>';
            }

            // ─── STEP 1: Fetch candles and update chart ───
            lastCandles = await fetchCandles(targetTimeframe, getCandleLimit(targetTimeframe), targetSymbol);

            // If a newer refresh has started, abort this stale one
            if (isStale()) { isLoading = false; return; }

            if (lastCandles.length > 0) {
                candleSeries.setData(lastCandles);
                // Auto-fit chart to bring new pair's candles into view
                chart.timeScale().fitContent();
                chart.priceScale('right').applyOptions({ autoScale: true });
                const lastPrice = lastCandles[lastCandles.length - 1].close;
                document.getElementById('currentPrice').textContent = fmtPrice(lastPrice);
            } else {
                // Candle fetch failed - clear chart and show error price
                candleSeries.setData([]);
                document.getElementById('currentPrice').textContent = 'No data';
                document.getElementById('currentPrice').style.color = '#ff4444';
                setTimeout(() => { document.getElementById('currentPrice').style.color = ''; }, 3000);
            }
            clearPriceLines();

            // For HTF data: fetch fresh if new pair, reuse cache if just timeframe change
            let rangesData = null, zonesData = null, liqData = null, valData = null, schematicsData = null, po3Data = null;

            if (isNewPair) {
                // ─── STEP 2: Fetch ranges, zones, liquidity in PARALLEL ───
                const [rangesResult, zonesResult, liqResult] = await Promise.allSettled([
                    fetchWithRetry(`/api/ranges?symbol=${targetSymbol}`, {}, 3, 25000),
                    fetchWithRetry(`/api/zones?symbol=${targetSymbol}`, {}, 3, 25000),
                    fetchWithRetry(`/api/liquidity?symbol=${targetSymbol}`, {}, 3, 25000),
                ]);

                if (isStale()) { isLoading = false; return; }

                // Process ranges
                rangesData = rangesResult.status === 'fulfilled' ? rangesResult.value : null;
                if (rangesData && !rangesData.error) {
                    try { updateRangesUI(rangesData, lastCandles); } catch(e) { console.error('Ranges render error:', e); }
                } else { setError('trendBadge'); setError('zoneBadge'); }

                // Process zones
                zonesData = zonesResult.status === 'fulfilled' ? zonesResult.value : null;
                if (zonesData && !zonesData.error) {
                    try { updateZonesUI(zonesData); } catch(e) { console.error('Zones render error:', e); }
                } else { setError('zoneCount'); }

                // Process liquidity
                liqData = liqResult.status === 'fulfilled' ? liqResult.value : null;
                if (liqData && !liqData.error) {
                    try { updateLiquidityUI(liqData, lastCandles); } catch(e) { console.error('Liq render error:', e); }
                } else { setError('liqCount'); }

                if (isStale()) { isLoading = false; return; }

                // ─── STEP 3: Fetch schematics, PO3, validation in PARALLEL ───
                const [schResult, po3Result, valResult] = await Promise.allSettled([
                    fetchWithRetry(`/api/schematics?symbol=${targetSymbol}&timeframe=${targetTimeframe}`, {}, 3, 30000),
                    fetchWithRetry(`/api/po3?symbol=${targetSymbol}&timeframe=${targetTimeframe}`, {}, 3, 30000),
                    fetchWithRetry(`/api/validate?symbol=${targetSymbol}`, {}, 3, 25000),
                ]);

                if (isStale()) { isLoading = false; return; }

                // Process schematics
                schematicsData = schResult.status === 'fulfilled' ? schResult.value : null;
                if (schematicsData && !schematicsData.error) {
                    try { updateSchematicsUI(schematicsData); } catch(e) { console.error('Schematics render error:', e); }
                } else { setError('schematicsBadge'); }

                // Process PO3
                po3Data = po3Result.status === 'fulfilled' ? po3Result.value : null;
                if (po3Data && !po3Data.error) {
                    try { updatePO3UI(po3Data); } catch(e) { console.error('PO3 render error:', e); }
                } else { setError('po3Badge'); }

                // Process validation
                valData = valResult.status === 'fulfilled' ? valResult.value : null;
                if (valData && !valData.error) {
                    try { updateValidationUI(valData); } catch(e) { console.error('Validation render error:', e); }
                } else { setError('actionBadge'); }

                // Cache all HTF data for this symbol
                if (!isStale()) {
                    htfCache = {
                        symbol: targetSymbol,
                        rangesData, zonesData, liqData, valData, schematicsData, po3Data,
                    };
                }

            } else {
                // ─── TIMEFRAME CHANGE: Reuse cached ranges/zones/liq/val, re-fetch schematics/PO3 ───
                rangesData = htfCache.rangesData;
                zonesData = htfCache.zonesData;
                liqData = htfCache.liqData;
                valData = htfCache.valData;

                // Re-render cached data with new timeframe candles
                if (rangesData && !rangesData.error) {
                    try { updateRangesUI(rangesData, lastCandles); } catch(e) { console.error('Ranges render error:', e); }
                } else { setError('trendBadge'); setError('zoneBadge'); }
                if (zonesData && !zonesData.error) {
                    try { updateZonesUI(zonesData); } catch(e) { console.error('Zones render error:', e); }
                } else { setError('zoneCount'); }
                if (liqData && !liqData.error) {
                    try { updateLiquidityUI(liqData, lastCandles); } catch(e) { console.error('Liq render error:', e); }
                } else { setError('liqCount'); }
                if (valData && !valData.error) {
                    try { updateValidationUI(valData); } catch(e) { console.error('Validation render error:', e); }
                } else { setError('actionBadge'); }

                if (isStale()) { isLoading = false; return; }

                // Re-fetch schematics and PO3 with new timeframe context in parallel
                const [schResult2, po3Result2] = await Promise.allSettled([
                    fetchWithRetry(`/api/schematics?symbol=${targetSymbol}&timeframe=${targetTimeframe}`, {}, 3, 30000),
                    fetchWithRetry(`/api/po3?symbol=${targetSymbol}&timeframe=${targetTimeframe}`, {}, 3, 30000),
                ]);

                if (isStale()) { isLoading = false; return; }

                schematicsData = schResult2.status === 'fulfilled' ? schResult2.value : null;
                if (schematicsData && !schematicsData.error) {
                    try { updateSchematicsUI(schematicsData); } catch(e) { console.error('Schematics render error:', e); }
                } else { setError('schematicsBadge'); }

                po3Data = po3Result2.status === 'fulfilled' ? po3Result2.value : null;
                if (po3Data && !po3Data.error) {
                    try { updatePO3UI(po3Data); } catch(e) { console.error('PO3 render error:', e); }
                } else { setError('po3Badge'); }

                // Update cache with new schematics/PO3 data
                if (!isStale()) {
                    htfCache.schematicsData = schematicsData;
                    htfCache.po3Data = po3Data;
                }
            }

            // If stale at this point, discard final rendering
            if (isStale()) { isLoading = false; return; }

            // ─── STEP 4: Forming Models (derived from current pair's schematic data) ───
            try { deriveFormingModels(schematicsData, po3Data); }
            catch(e) { console.error('Forming models error:', e); }

            // ─── STEP 5: Highest Probability Setup (uses all data from pipeline) ───
            try {
                const bestSetup = analyzeHighestProbabilitySetup(rangesData, zonesData, liqData, schematicsData, po3Data, valData, lastCandles);
                renderSetupPanel(bestSetup);
                try { drawTCTModelOverlays(bestSetup, lastCandles); }
                catch(e) { console.error('Overlay draw error:', e); }
            } catch(e) {
                console.error('Setup analysis error:', e);
                // Show error in setup panel instead of blank
                document.getElementById('setupDirection').textContent = 'ERR';
                document.getElementById('setupDirection').className = 'setup-direction none';
                document.getElementById('setupContent').innerHTML = '<div class="metric-row"><span class="label" style="color:#ff4444;">Analysis error: ' + e.message + '</span></div>';
            }

            isLoading = false;
        }

        // Derive Forming Models from current pair's HTF+LTF schematic data
        function deriveFormingModels(schematicsData, po3Data) {
            const countEl = document.getElementById('formingCount');
            const contentEl = document.getElementById('formingContent');

            const models = [];

            // Collect TCT schematics across all timeframes
            if (schematicsData) {
                const tfGroups = [
                    { key: 'htf_schematics', label: 'HTF' },
                    { key: 'mtf_schematics', label: 'MTF' },
                    { key: 'ltf_schematics', label: 'LTF' },
                ];
                tfGroups.forEach(({ key, label }) => {
                    const group = schematicsData[key];
                    if (!group || !group.schematics) return;
                    group.schematics.forEach(s => {
                        models.push({
                            source: 'TCT',
                            tf_label: label,
                            timeframe: group.timeframe || label,
                            type: (s.schematic_type || 'unknown').replace(/_/g, ' '),
                            direction: s.direction || 'unknown',
                            is_confirmed: !!s.is_confirmed,
                            quality: s.quality_score || 0,
                            rr: s.risk_reward || 0,
                            has_entry: !!(s.entry && s.entry.price),
                        });
                    });
                });
            }

            // Collect PO3 schematics across all timeframes
            if (po3Data) {
                const tfGroups = [
                    { key: 'htf_po3', label: 'HTF' },
                    { key: 'mtf_po3', label: 'MTF' },
                    { key: 'ltf_po3', label: 'LTF' },
                ];
                tfGroups.forEach(({ key, label }) => {
                    const group = po3Data[key];
                    if (!group || !group.schematics) return;
                    group.schematics.forEach(p => {
                        models.push({
                            source: 'PO3',
                            tf_label: label,
                            timeframe: group.timeframe || label,
                            type: 'PO3 ' + (p.phase || 'range'),
                            direction: p.direction || 'unknown',
                            is_confirmed: !!p.has_expansion,
                            quality: p.quality_score || 0,
                            rr: p.risk_reward || 0,
                            has_entry: !!(p.entry && p.entry.price),
                        });
                    });
                });
            }

            const formingOnly = models.filter(m => !m.is_confirmed);
            const confirmedOnly = models.filter(m => m.is_confirmed);
            countEl.textContent = formingOnly.length + ' forming / ' + confirmedOnly.length + ' confirmed';

            if (models.length === 0) {
                // No schematic/PO3 models — derive from range + candle context
                let emptyHtml = '<div class="forming-empty" style="color:#888;font-size:0.7rem;padding:6px 0;">No TCT/PO3 schematics detected.</div>';

                // Analyze what could be forming from available candle data
                if (lastCandles && lastCandles.length >= 20) {
                    const recent = lastCandles.slice(-30);
                    const rHigh = Math.max(...recent.map(c => c.high));
                    const rLow = Math.min(...recent.map(c => c.low));
                    const rSize = rHigh - rLow;
                    const price = lastCandles[lastCandles.length - 1].close;
                    const posInRange = rSize > 0 ? (price - rLow) / rSize : 0.5;
                    const sma10 = recent.slice(-10).reduce((s,c) => s + c.close, 0) / 10;
                    const sma20 = recent.slice(-20).reduce((s,c) => s + c.close, 0) / Math.min(recent.length, 20);
                    const trendUp = sma10 > sma20;

                    // Count consolidation (small body candles)
                    const last10 = recent.slice(-10);
                    const consolidating = last10.filter(c => Math.abs(c.close - c.open) / (c.high - c.low + 0.0001) < 0.4).length;
                    const isConsolidating = consolidating >= 5;

                    emptyHtml += '<div style="background:#1a1a2e;border-radius:6px;padding:8px;margin-top:4px;">';
                    emptyHtml += '<div style="font-size:0.7rem;font-weight:600;color:#00d4ff;margin-bottom:4px;">What\'s Forming</div>';

                    if (isConsolidating) {
                        const model = trendUp ? 'Re-Accumulation' : 'Re-Distribution';
                        const modelColor = trendUp ? '#00ff88' : '#ff4444';
                        emptyHtml += '<div style="display:flex;justify-content:space-between;font-size:0.7rem;padding:2px 0;">';
                        emptyHtml += '<span style="color:#888;">Pattern</span><span style="color:' + modelColor + ';font-weight:600;">' + model + ' forming</span></div>';
                        emptyHtml += '<div style="display:flex;justify-content:space-between;font-size:0.65rem;padding:2px 0;">';
                        emptyHtml += '<span style="color:#888;">Phase</span><span style="color:#ffc107;">Consolidation / Range</span></div>';
                        emptyHtml += '<div style="display:flex;justify-content:space-between;font-size:0.65rem;padding:2px 0;">';
                        emptyHtml += '<span style="color:#888;">Look for</span><span style="color:#e0e0e0;">EQ tap → deviation → expansion</span></div>';
                    } else if (posInRange < 0.3) {
                        emptyHtml += '<div style="display:flex;justify-content:space-between;font-size:0.7rem;padding:2px 0;">';
                        emptyHtml += '<span style="color:#888;">Pattern</span><span style="color:#00ff88;font-weight:600;">Accumulation watch</span></div>';
                        emptyHtml += '<div style="display:flex;justify-content:space-between;font-size:0.65rem;padding:2px 0;">';
                        emptyHtml += '<span style="color:#888;">Phase</span><span style="color:#ffc107;">Near range low / discount</span></div>';
                        emptyHtml += '<div style="display:flex;justify-content:space-between;font-size:0.65rem;padding:2px 0;">';
                        emptyHtml += '<span style="color:#888;">Look for</span><span style="color:#e0e0e0;">Spring / SSL sweep → BOS up</span></div>';
                    } else if (posInRange > 0.7) {
                        emptyHtml += '<div style="display:flex;justify-content:space-between;font-size:0.7rem;padding:2px 0;">';
                        emptyHtml += '<span style="color:#888;">Pattern</span><span style="color:#ff4444;font-weight:600;">Distribution watch</span></div>';
                        emptyHtml += '<div style="display:flex;justify-content:space-between;font-size:0.65rem;padding:2px 0;">';
                        emptyHtml += '<span style="color:#888;">Phase</span><span style="color:#ffc107;">Near range high / premium</span></div>';
                        emptyHtml += '<div style="display:flex;justify-content:space-between;font-size:0.65rem;padding:2px 0;">';
                        emptyHtml += '<span style="color:#888;">Look for</span><span style="color:#e0e0e0;">UTAD / BSL sweep → BOS down</span></div>';
                    } else {
                        emptyHtml += '<div style="display:flex;justify-content:space-between;font-size:0.7rem;padding:2px 0;">';
                        emptyHtml += '<span style="color:#888;">Pattern</span><span style="color:#ffc107;font-weight:600;">Ranging / EQ test</span></div>';
                        emptyHtml += '<div style="display:flex;justify-content:space-between;font-size:0.65rem;padding:2px 0;">';
                        emptyHtml += '<span style="color:#888;">Phase</span><span style="color:#ffc107;">Mid-range / equilibrium</span></div>';
                        emptyHtml += '<div style="display:flex;justify-content:space-between;font-size:0.65rem;padding:2px 0;">';
                        emptyHtml += '<span style="color:#888;">Look for</span><span style="color:#e0e0e0;">Directional BOS from EQ</span></div>';
                    }

                    emptyHtml += '<div style="display:flex;justify-content:space-between;font-size:0.65rem;padding:2px 0;margin-top:2px;">';
                    emptyHtml += '<span style="color:#888;">Range</span><span style="color:#e0e0e0;">' + fmtPrice(rLow) + ' - ' + fmtPrice(rHigh) + '</span></div>';
                    emptyHtml += '<div style="display:flex;justify-content:space-between;font-size:0.65rem;padding:2px 0;">';
                    emptyHtml += '<span style="color:#888;">Pos in range</span><span style="color:#e0e0e0;">' + (posInRange * 100).toFixed(0) + '%</span></div>';
                    emptyHtml += '</div>';

                    countEl.textContent = '0 (analyzing)';
                }

                contentEl.innerHTML = emptyHtml;
                return;
            }

            // Sort: forming first, then by quality descending
            models.sort((a, b) => {
                if (a.is_confirmed !== b.is_confirmed) return a.is_confirmed ? 1 : -1;
                return b.quality - a.quality;
            });

            let html = '';
            models.forEach(m => {
                const dirCls = m.direction === 'bullish' ? 'bullish' : m.direction === 'bearish' ? 'bearish' : '';
                const statusCls = m.is_confirmed ? 'confirmed' : 'forming';
                const typeCls = m.direction === 'bullish' ? 'accum' : 'dist';

                html += '<div class="forming-link ' + dirCls + '">';
                html += '<span class="pair-name">' + m.source + '</span>';
                html += '<span class="model-type ' + typeCls + '">' + m.type + '</span>';
                html += '<span class="tf-badge">' + m.timeframe.toUpperCase() + '</span>';
                html += '<span class="status-dot ' + statusCls + '" title="' + (m.is_confirmed ? 'Confirmed' : 'Forming') + '"></span>';
                html += '</div>';
            });

            contentEl.innerHTML = html;
        }

        function updateRangesUI(data, candles = []) {
            if (data.error) return;

            const ms = data.market_structure || {};

            // ═══ MARKET STRUCTURE CARD ═══

            // HTF / LTF trends
            const htfTrend = ms.htf_trend || 'neutral';
            const ltfTrend = ms.ltf_trend || 'neutral';
            document.getElementById('htfTrend').textContent = htfTrend.toUpperCase();
            document.getElementById('htfTrend').className = 'value ' + (htfTrend === 'bullish' ? 'bullish' : htfTrend === 'bearish' ? 'bearish' : '');
            document.getElementById('ltfTrend').textContent = ltfTrend.toUpperCase();
            document.getElementById('ltfTrend').className = 'value ' + (ltfTrend === 'bullish' ? 'bullish' : ltfTrend === 'bearish' ? 'bearish' : '');

            const trendBadge = document.getElementById('trendBadge');
            trendBadge.textContent = htfTrend.toUpperCase();
            trendBadge.className = 'badge badge-' + (htfTrend === 'bullish' ? 'bullish' : htfTrend === 'bearish' ? 'bearish' : 'neutral');

            // Expectational Order Flow (EOF)
            const htfEof = ms.htf_eof || {};
            const ltfEof = ms.ltf_eof || {};
            const htfEofEl = document.getElementById('htfEOF');
            const ltfEofEl = document.getElementById('ltfEOF');
            const eofLabels = {
                'higher_low_for_higher_high': 'HL \u2192 HH',
                'lower_high_for_lower_low': 'LH \u2192 LL',
                'undetermined': '--'
            };
            const htfEofText = eofLabels[htfEof.expectation] || htfEof.expectation || '--';
            const ltfEofText = eofLabels[ltfEof.expectation] || ltfEof.expectation || '--';
            htfEofEl.textContent = htfEofText + (htfEof.trend_shift ? ' (SHIFT)' : '');
            htfEofEl.className = 'value ' + (htfEof.bias === 'bullish' ? 'bullish' : htfEof.bias === 'bearish' ? 'bearish' : '');
            if (htfEof.trend_shift) htfEofEl.style.color = '#ffc107';
            else htfEofEl.style.color = '';
            ltfEofEl.textContent = ltfEofText + (ltfEof.trend_shift ? ' (SHIFT)' : '');
            ltfEofEl.className = 'value ' + (ltfEof.bias === 'bullish' ? 'bullish' : ltfEof.bias === 'bearish' ? 'bearish' : '');
            if (ltfEof.trend_shift) ltfEofEl.style.color = '#ffc107';
            else ltfEofEl.style.color = '';

            // Level 1/2/3 summary
            const htfLevels = ms.htf_levels || {};
            const levelsEl = document.getElementById('msLevelsContent');
            let lvlHtml = '';
            const lvlColors = { 'level_1': '#e0e0e0', 'level_2': '#ff4444', 'level_3': '#00d4ff' };
            const lvlLabels = { 'level_1': 'L1 (Primary)', 'level_2': 'L2 (Counter)', 'level_3': 'L3 (Refined)' };
            ['level_1', 'level_2', 'level_3'].forEach(lvl => {
                const ld = htfLevels[lvl] || {};
                const lvlTrend = ld.trend || '--';
                const bosCount = (ld.bos || []).length;
                if (lvlTrend && lvlTrend !== '--' && lvlTrend !== 'neutral' && lvlTrend !== '') {
                    lvlHtml += '<div style="display:flex;justify-content:space-between;align-items:center;padding:2px 0;font-size:0.65rem;">';
                    lvlHtml += '<span style="color:' + lvlColors[lvl] + ';font-weight:600;">' + lvlLabels[lvl] + '</span>';
                    lvlHtml += '<span style="color:' + (lvlTrend === 'bullish' ? '#00ff88' : lvlTrend === 'bearish' ? '#ff4444' : '#888') + ';">';
                    lvlHtml += lvlTrend.toUpperCase();
                    if (bosCount > 0) lvlHtml += ' <span style="color:#ffc107;">(' + bosCount + ' BOS)</span>';
                    lvlHtml += '</span></div>';
                }
            });
            levelsEl.innerHTML = lvlHtml;

            // MS Highs / Lows count
            const msHighs = ms.htf_ms_highs || [];
            const msLows = ms.htf_ms_lows || [];
            document.getElementById('msPivotCounts').textContent = msHighs.length + ' H / ' + msLows.length + ' L';

            // BOS events on chart + sidebar list
            const htfBosEvents = ms.htf_bos_events || [];
            htfBosEvents.forEach(bos => {
                if (bos.broken_level) {
                    const bosColor = bos.type === 'bullish' ? '#00ff88' : '#ff4444';
                    const bosLabel = bos.type === 'bullish' ? 'BOS \u2191' : 'BOS \u2193';
                    lineSeries.push(addPriceLine(bos.broken_level, bosColor, bosLabel, 2, 1));
                }
            });

            // BOS events sidebar list (last 5, matching /market-structure page)
            let bosHtml = '';
            if (htfBosEvents.length > 0) {
                bosHtml += '<div style="font-size:0.7rem;font-weight:600;color:#ffc107;margin-bottom:4px;">BOS Events (' + htfBosEvents.length + ')</div>';
                htfBosEvents.slice(-5).reverse().forEach(b => {
                    const dir = b.type === 'bullish' ? '\u25B2' : '\u25BC';
                    const dirColor = b.type === 'bullish' ? '#00ff88' : '#ff4444';
                    const qualCls = b.quality === 'good' ? 'bullish' : b.quality === 'bad' ? 'bearish' : 'warning';
                    const qualBg = b.quality === 'good' ? 'rgba(0,255,136,0.15)' : b.quality === 'bad' ? 'rgba(255,68,68,0.15)' : 'rgba(255,193,7,0.15)';
                    bosHtml += '<div style="display:flex;justify-content:space-between;align-items:center;padding:2px 0;font-size:0.65rem;">';
                    bosHtml += '<span style="color:' + dirColor + ';">' + dir + ' ' + b.type + '</span>';
                    bosHtml += '<span>' + fmtPrice(b.broken_level || 0) + '</span>';
                    if (b.quality) bosHtml += '<span style="padding:1px 5px;border-radius:3px;background:' + qualBg + ';color:' + dirColor + ';font-size:0.6rem;">' + b.quality + '</span>';
                    bosHtml += '</div>';
                });
            }
            document.getElementById('bosEventsContent').innerHTML = bosHtml;

            // Domino Effect (from /market-structure page logic)
            // Note: domino data is not in the /api/ranges response, but we can derive from level data
            let dominoHtml = '';
            const l1 = htfLevels.level_1 || {};
            const l2 = htfLevels.level_2 || {};
            const l3 = htfLevels.level_3 || {};
            if (l1.trend && l1.trend !== 'neutral') {
                const l2HasBos = (l2.bos || []).length > 0;
                const l3HasBos = (l3.bos || []).length > 0;
                let stage = 'L1 rotation';
                let conf = 'low';
                if (l3HasBos) { stage = 'L3 confirmed'; conf = 'high'; }
                else if (l2HasBos) { stage = 'L2 confirmed, awaiting L3'; conf = 'moderate'; }
                else { stage = 'Awaiting L2 rotation'; conf = 'low'; }
                const confColor = conf === 'high' ? '#00ff88' : conf === 'moderate' ? '#ffc107' : '#888';
                const confBg = conf === 'high' ? 'rgba(0,255,136,0.15)' : conf === 'moderate' ? 'rgba(255,193,7,0.15)' : 'rgba(136,136,136,0.15)';
                dominoHtml += '<div style="font-size:0.7rem;font-weight:600;color:#00d4ff;margin-bottom:4px;">Domino Effect</div>';
                dominoHtml += '<div style="display:flex;justify-content:space-between;align-items:center;padding:2px 0;font-size:0.65rem;">';
                dominoHtml += '<span style="color:#888;">Stage</span><span style="color:#e0e0e0;">' + stage + '</span>';
                dominoHtml += '</div>';
                dominoHtml += '<div style="display:flex;justify-content:space-between;align-items:center;padding:2px 0;font-size:0.65rem;">';
                dominoHtml += '<span style="color:#888;">Confidence</span><span style="padding:1px 5px;border-radius:3px;background:' + confBg + ';color:' + confColor + ';">' + conf + '</span>';
                dominoHtml += '</div>';
            }
            document.getElementById('dominoContent').innerHTML = dominoHtml;

            // ═══ ACTIVE RANGE CARD ═══

            const activeRange = data.htf_ranges?.active_range || data.ltf_ranges?.active_range;
            if (activeRange) {
                const high = activeRange.range_high || activeRange.high;
                const low = activeRange.range_low || activeRange.low;
                const eq = activeRange.equilibrium || ((high + low) / 2);
                const rangeSize = activeRange.range_size || (high && low ? high - low : 0);
                const dlHigh = activeRange.deviation_limit_high;
                const dlLow = activeRange.deviation_limit_low;
                const isFallback = activeRange.is_fallback;
                const isConfirmed = activeRange.is_confirmed;

                // Quality tag (matching /ranges page)
                let qualHtml = '';
                if (isFallback) {
                    qualHtml = '<span style="padding:2px 8px;border-radius:3px;background:rgba(136,136,136,0.15);color:#888;font-size:0.65rem;">Fallback (not TCT validated)</span>';
                } else if (isConfirmed) {
                    qualHtml = '<span style="padding:2px 8px;border-radius:3px;background:rgba(0,255,136,0.15);color:#00ff88;font-size:0.65rem;">Confirmed</span>';
                }
                document.getElementById('rangeQualityTag').innerHTML = qualHtml;

                document.getElementById('rangeHigh').textContent = fmtPrice(high);
                document.getElementById('rangeEq').textContent = fmtPrice(eq);
                document.getElementById('rangeLow').textContent = fmtPrice(low);

                // Range size
                document.getElementById('rangeSize').textContent = fmtPrice(rangeSize);

                // DL+ / DL-
                if (dlHigh && dlLow) {
                    document.getElementById('rangeDL').textContent = fmtPrice(dlHigh) + ' / ' + fmtPrice(dlLow);
                }

                // Chart lines — Range High/Low/EQ
                if (high) lineSeries.push(addPriceLine(high, '#ff4444', 'RANGE HIGH (1.0)', 0, 2));
                if (eq) lineSeries.push(addPriceLine(eq, '#ffc107', 'EQ (0.5)', 1, 2));
                if (low) lineSeries.push(addPriceLine(low, '#00ff88', 'RANGE LOW (0.0)', 0, 2));

                // Fibonacci retracement levels on chart
                if (high && low) {
                    const fib236 = high - (rangeSize * 0.236);
                    const fib382 = high - (rangeSize * 0.382);
                    const fib618 = high - (rangeSize * 0.618);
                    const fib786 = high - (rangeSize * 0.786);
                    lineSeries.push(addPriceLine(fib236, '#b388ff', '0.236', 2, 1));
                    lineSeries.push(addPriceLine(fib382, '#7c4dff', '0.382', 2, 1));
                    lineSeries.push(addPriceLine(fib618, '#7c4dff', '0.618', 2, 1));
                    lineSeries.push(addPriceLine(fib786, '#b388ff', '0.786', 2, 1));
                }

                // DL lines on chart (dotted gray, matching /ranges page)
                if (dlHigh) lineSeries.push(addPriceLine(dlHigh, '#6c757d', 'DL+', 2, 1));
                if (dlLow) lineSeries.push(addPriceLine(dlLow, '#6c757d', 'DL-', 2, 1));

                // Shaded range zones on chart
                if (high && low && candles.length > 0) {
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

                // Range viz sidebar
                document.getElementById('rangeViz').style.display = 'block';
                document.getElementById('rangeHighLabel').textContent = fmtPrice(high);
                document.getElementById('rangeLowLabel').textContent = fmtPrice(low);

                const currentPrice = data.current_price;
                if (currentPrice && high && low) {
                    const pct = ((high - currentPrice) / (high - low)) * 100;
                    const marker = document.getElementById('priceMarker');
                    marker.style.top = Math.min(100, Math.max(0, pct)) + '%';
                    marker.style.left = '50%';
                }

                // === Fibonacci Retracement Sidebar ===
                if (high && low) {
                    const fib236 = high - (rangeSize * 0.236);
                    const fib382 = high - (rangeSize * 0.382);
                    const fib618 = high - (rangeSize * 0.618);
                    const fib786 = high - (rangeSize * 0.786);
                    const fmt = (v) => '$' + v.toLocaleString(undefined, {maximumFractionDigits: 2});

                    let fibHtml = '<div style="font-size:0.7rem;font-weight:600;color:#b388ff;margin-bottom:4px;margin-top:8px;">Fib Retracement</div>';
                    const fibLevels = [
                        { label: '1.0 (High)', price: high, color: '#ff4444' },
                        { label: '0.786', price: fib786, color: '#b388ff' },
                        { label: '0.618', price: fib618, color: '#7c4dff' },
                        { label: '0.5 (EQ)', price: eq, color: '#ffc107' },
                        { label: '0.382', price: fib382, color: '#7c4dff' },
                        { label: '0.236', price: fib236, color: '#b388ff' },
                        { label: '0.0 (Low)', price: low, color: '#00ff88' },
                    ];
                    fibLevels.forEach(f => {
                        fibHtml += '<div style="display:flex;justify-content:space-between;padding:1px 0;font-size:0.65rem;">';
                        fibHtml += '<span style="color:' + f.color + ';">' + f.label + '</span>';
                        fibHtml += '<span style="color:#e0e0e0;">' + fmt(f.price) + '</span>';
                        fibHtml += '</div>';
                    });

                    // Tap detection: count how many times price touched range high/low/EQ
                    if (candles.length > 0) {
                        const tolerance = rangeSize * 0.01;
                        let highTaps = 0, lowTaps = 0, eqTaps = 0;
                        let highTapTimes = [], lowTapTimes = [], eqTapTimes = [];
                        let lastHighTap = -3, lastLowTap = -3, lastEqTap = -3;
                        candles.forEach((c, idx) => {
                            if (Math.abs(c.high - high) <= tolerance && idx - lastHighTap > 2) {
                                highTaps++; lastHighTap = idx;
                                highTapTimes.push(idx);
                            }
                            if (Math.abs(c.low - low) <= tolerance && idx - lastLowTap > 2) {
                                lowTaps++; lastLowTap = idx;
                                lowTapTimes.push(idx);
                            }
                            if ((c.low <= eq + tolerance && c.high >= eq - tolerance) && idx - lastEqTap > 2) {
                                eqTaps++; lastEqTap = idx;
                                eqTapTimes.push(idx);
                            }
                        });

                        fibHtml += '<div style="font-size:0.7rem;font-weight:600;color:#00d4ff;margin-top:8px;margin-bottom:4px;">Range Taps</div>';
                        fibHtml += '<div style="display:flex;justify-content:space-between;padding:1px 0;font-size:0.65rem;">';
                        fibHtml += '<span style="color:#ff4444;">High Taps</span><span style="color:#e0e0e0;">' + highTaps + '</span></div>';
                        fibHtml += '<div style="display:flex;justify-content:space-between;padding:1px 0;font-size:0.65rem;">';
                        fibHtml += '<span style="color:#ffc107;">EQ Taps</span><span style="color:#e0e0e0;">' + eqTaps + '</span></div>';
                        fibHtml += '<div style="display:flex;justify-content:space-between;padding:1px 0;font-size:0.65rem;">';
                        fibHtml += '<span style="color:#00ff88;">Low Taps</span><span style="color:#e0e0e0;">' + lowTaps + '</span></div>';

                        // Determine model forming based on taps and trend
                        const htfTrendVal = ms.htf_trend || 'neutral';
                        let modelType = 'Undetermined';
                        let modelColor = '#888';
                        if (highTaps >= 2 && lowTaps >= 1 && eqTaps >= 1) {
                            if (htfTrendVal === 'bearish') { modelType = 'Distribution'; modelColor = '#ff4444'; }
                            else if (htfTrendVal === 'bullish') { modelType = 'Re-Accumulation'; modelColor = '#00d4ff'; }
                            else { modelType = 'Distribution (poss.)'; modelColor = '#ffc107'; }
                        } else if (lowTaps >= 2 && highTaps >= 1 && eqTaps >= 1) {
                            if (htfTrendVal === 'bullish') { modelType = 'Accumulation'; modelColor = '#00ff88'; }
                            else if (htfTrendVal === 'bearish') { modelType = 'Re-Distribution'; modelColor = '#e040fb'; }
                            else { modelType = 'Accumulation (poss.)'; modelColor = '#ffc107'; }
                        } else if (eqTaps >= 2) {
                            modelType = 'Consolidation'; modelColor = '#ffc107';
                        } else if (highTaps + lowTaps + eqTaps < 3) {
                            modelType = 'Forming (early)'; modelColor = '#888';
                        }

                        fibHtml += '<div style="font-size:0.7rem;font-weight:600;color:#e040fb;margin-top:8px;margin-bottom:4px;">Model Forming</div>';
                        fibHtml += '<div style="display:flex;justify-content:space-between;padding:2px 0;font-size:0.7rem;">';
                        fibHtml += '<span style="color:#888;">Type</span>';
                        fibHtml += '<span style="padding:2px 8px;border-radius:3px;background:rgba(255,255,255,0.05);color:' + modelColor + ';font-weight:600;">' + modelType + '</span>';
                        fibHtml += '</div>';

                        // Add tap markers on chart
                        const markers = [];
                        highTapTimes.slice(0, 5).forEach((idx, i) => {
                            if (idx < candles.length) {
                                markers.push({ time: candles[idx].time, position: 'aboveBar', color: '#ff4444', shape: 'arrowDown', text: 'T' + (i+1) });
                            }
                        });
                        lowTapTimes.slice(0, 5).forEach((idx, i) => {
                            if (idx < candles.length) {
                                markers.push({ time: candles[idx].time, position: 'belowBar', color: '#00ff88', shape: 'arrowUp', text: 'T' + (i+1) });
                            }
                        });
                        if (markers.length > 0) {
                            markers.sort((a, b) => a.time - b.time);
                            candleSeries.setMarkers(markers);
                        }
                    }

                    document.getElementById('fibContent').innerHTML = fibHtml;
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

            // ═══ DEVIATIONS CARD ═══
            const htfDevs = data.deviations?.htf_deviations || {};
            const ltfDevs = data.deviations?.ltf_deviations || {};
            const totalDevs = (htfDevs.total_deviations || 0) + (ltfDevs.total_deviations || 0);

            document.getElementById('devBadge').textContent = totalDevs;
            document.getElementById('wickDevs').textContent = (htfDevs.wick_deviations?.length || 0) + (ltfDevs.wick_deviations?.length || 0);
            document.getElementById('candleDevs').textContent = (htfDevs.candle_close_deviations?.length || 0) + (ltfDevs.candle_close_deviations?.length || 0);
            document.getElementById('dlDevs').textContent = (htfDevs.dl_deviations?.length || 0) + (ltfDevs.dl_deviations?.length || 0);
        }

        function updateZonesUI(data) {
            if (data.error) return;

            // Count OBs and structure zones separately (matching /supply-demand page)
            const allOBs = data.htf_zones?.order_blocks || [];
            const allStructure = data.htf_zones?.structure_zones || [];
            const demandOBs = allOBs.filter(z => z.type === 'bullish' && !z.mitigated);
            const supplyOBs = allOBs.filter(z => z.type === 'bearish' && !z.mitigated);
            const activeStructDemand = allStructure.filter(z => z.type === 'demand' && !z.mitigated);
            const activeStructSupply = allStructure.filter(z => z.type === 'supply' && !z.mitigated);
            const mitigatedTotal = allOBs.filter(z => z.mitigated).length + allStructure.filter(z => z.mitigated).length;
            const totalActive = demandOBs.length + supplyOBs.length + activeStructDemand.length + activeStructSupply.length;

            document.getElementById('demandOBCount').textContent = demandOBs.length;
            document.getElementById('supplyOBCount').textContent = supplyOBs.length;
            document.getElementById('structureSDCount').textContent = (activeStructDemand.length + activeStructSupply.length) + ' (' + activeStructDemand.length + 'D / ' + activeStructSupply.length + 'S)';
            document.getElementById('mitigatedCount').textContent = mitigatedTotal;
            document.getElementById('zoneCount').textContent = totalActive;

            // Top scored zones (matching /supply-demand page)
            const topScored = data.htf_zones?.top_3_high_quality || data.htf_zones?.top_3_all || [];
            let scoredHtml = '';
            if (topScored.length > 0) {
                scoredHtml += '<div style="font-size:0.7rem;font-weight:600;color:#00d4ff;margin-bottom:4px;">Top Scored</div>';
                topScored.slice(0, 4).forEach(z => {
                    const zoneType = z.type || (z.top > data.current_price ? 'supply' : 'demand');
                    const isDemand = zoneType === 'demand' || z.type === 'bullish';
                    const zClass = z.zone_class || (z.fvg ? 'order_block' : 'structure');
                    const label = (isDemand ? 'Demand' : 'Supply') + ' ' + (zClass === 'order_block' ? 'OB' : 'SS');
                    const strength = z.strength || 0;
                    const strengthColor = strength >= 70 ? '#00ff88' : strength >= 40 ? '#ffc107' : '#ff4444';
                    const locType = z.location_type || '';
                    const hasLiqSweep = z.pivot_quality?.has_liquidity_sweep;

                    scoredHtml += '<div style="background:#1a1a2e;border-radius:4px;padding:5px 7px;margin-bottom:3px;border-left:3px solid ' + (isDemand ? '#00ff88' : '#ff4444') + ';font-size:0.65rem;">';
                    scoredHtml += '<div style="display:flex;justify-content:space-between;align-items:center;">';
                    scoredHtml += '<span style="font-weight:600;color:' + (isDemand ? '#00ff88' : '#ff4444') + ';">' + label + '</span>';
                    scoredHtml += '<span style="padding:1px 5px;border-radius:3px;background:rgba(0,255,136,0.15);color:' + strengthColor + ';font-size:0.6rem;">' + strength + '%</span>';
                    scoredHtml += '</div>';
                    scoredHtml += '<div style="color:#888;margin-top:2px;">' + fmtPrice(z.top || 0) + ' - ' + fmtPrice(z.bottom || 0) + '</div>';
                    if (locType || hasLiqSweep) {
                        scoredHtml += '<div style="display:flex;gap:4px;margin-top:2px;">';
                        if (locType) scoredHtml += '<span style="padding:1px 4px;border-radius:2px;background:rgba(0,212,255,0.15);color:#00d4ff;font-size:0.55rem;">' + locType + '</span>';
                        if (hasLiqSweep) scoredHtml += '<span style="padding:1px 4px;border-radius:2px;background:rgba(255,193,7,0.15);color:#ffc107;font-size:0.55rem;">Liq Sweep</span>';
                        scoredHtml += '</div>';
                    }
                    scoredHtml += '</div>';
                });
            }
            document.getElementById('topScoredZones').innerHTML = scoredHtml;

            // Zone list on chart
            const topZonesEl = document.getElementById('topZones');
            topZonesEl.innerHTML = '';

            const topZones = data.htf_zones?.top_3_high_quality || data.htf_zones?.top_3_all || [];
            topZones.slice(0, 3).forEach(zone => {
                const zoneType = zone.type || (zone.top > data.current_price ? 'supply' : 'demand');
                const isDemand = zoneType === 'demand' || zone.type === 'bullish';

                // Add zone to chart
                if (zone.top && zone.bottom) {
                    lineSeries.push(addPriceLine(zone.top, isDemand ? '#00ff88' : '#ff4444', isDemand ? 'D' : 'S', 2));
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

            // Primary / Internal breakdown (matching /liquidity page)
            const bslPrimary = bslPools.filter(p => p.is_primary).length;
            const bslInternal = bslPools.length - bslPrimary;
            const sslPrimary = sslPools.filter(p => p.is_primary).length;
            const sslInternal = sslPools.length - sslPrimary;
            document.getElementById('bslBreakdown').textContent = bslPrimary + ' / ' + bslInternal;
            document.getElementById('sslBreakdown').textContent = sslPrimary + ' / ' + sslInternal;

            // Display top liquidity pools with details
            const liqPoolsEl = document.getElementById('liqPools');
            let poolHtml = '';

            // Top 3 BSL pools with strength + distance
            bslPools.slice(0, 3).forEach(pool => {
                const isPrimary = pool.is_primary;
                const isEqual = pool.is_equal;
                const strength = pool.strength ? Math.round(pool.strength * 100) : 0;
                const dist = pool.distance_from_price ? pool.distance_from_price.toFixed(1) : '?';
                poolHtml += '<div class="zone-item bsl" style="flex-direction:column;align-items:flex-start;gap:2px;">';
                poolHtml += '<div style="display:flex;justify-content:space-between;width:100%;align-items:center;">';
                poolHtml += '<span style="font-weight:600;">BSL ' + fmtPrice(pool.price || 0) + '</span>';
                poolHtml += '<span style="display:flex;gap:3px;">';
                if (isPrimary) poolHtml += '<span style="padding:1px 4px;border-radius:2px;background:rgba(33,150,243,0.15);color:#2196f3;font-size:0.55rem;">PRIMARY</span>';
                else poolHtml += '<span style="padding:1px 4px;border-radius:2px;background:rgba(136,136,136,0.15);color:#888;font-size:0.55rem;">INTERNAL</span>';
                if (isEqual) poolHtml += '<span style="padding:1px 4px;border-radius:2px;background:rgba(156,39,176,0.15);color:#9c27b0;font-size:0.55rem;">EQ</span>';
                poolHtml += '</span></div>';
                poolHtml += '<div style="color:#888;font-size:0.6rem;">Str: ' + strength + '% | Dist: ' + dist + '%</div>';
                poolHtml += '</div>';

                if (pool.price) {
                    const lineColor = isPrimary ? '#ff6b6b' : 'rgba(255,107,107,0.5)';
                    const lineStyle = isPrimary ? 0 : 2;
                    lineSeries.push(addPriceLine(pool.price, lineColor, 'BSL', lineStyle, isPrimary ? 2 : 1));
                }
            });

            // Top 3 SSL pools with strength + distance
            sslPools.slice(0, 3).forEach(pool => {
                const isPrimary = pool.is_primary;
                const isEqual = pool.is_equal;
                const strength = pool.strength ? Math.round(pool.strength * 100) : 0;
                const dist = pool.distance_from_price ? pool.distance_from_price.toFixed(1) : '?';
                poolHtml += '<div class="zone-item ssl" style="flex-direction:column;align-items:flex-start;gap:2px;">';
                poolHtml += '<div style="display:flex;justify-content:space-between;width:100%;align-items:center;">';
                poolHtml += '<span style="font-weight:600;">SSL ' + fmtPrice(pool.price || 0) + '</span>';
                poolHtml += '<span style="display:flex;gap:3px;">';
                if (isPrimary) poolHtml += '<span style="padding:1px 4px;border-radius:2px;background:rgba(255,152,0,0.15);color:#ff9800;font-size:0.55rem;">PRIMARY</span>';
                else poolHtml += '<span style="padding:1px 4px;border-radius:2px;background:rgba(136,136,136,0.15);color:#888;font-size:0.55rem;">INTERNAL</span>';
                if (isEqual) poolHtml += '<span style="padding:1px 4px;border-radius:2px;background:rgba(156,39,176,0.15);color:#9c27b0;font-size:0.55rem;">EQ</span>';
                poolHtml += '</span></div>';
                poolHtml += '<div style="color:#888;font-size:0.6rem;">Str: ' + strength + '% | Dist: ' + dist + '%</div>';
                poolHtml += '</div>';

                if (pool.price) {
                    const lineColor = isPrimary ? '#4ecdc4' : 'rgba(78,205,196,0.5)';
                    const lineStyle = isPrimary ? 0 : 2;
                    lineSeries.push(addPriceLine(pool.price, lineColor, 'SSL', lineStyle, isPrimary ? 2 : 1));
                }
            });

            liqPoolsEl.innerHTML = poolHtml;

            // Liquidity curves summary (matching /liquidity page)
            const sellSideCurves = data.htf_curves?.sell_side_curves || [];
            const buySideCurves = data.htf_curves?.buy_side_curves || [];
            let curvesHtml = '';
            if (sellSideCurves.length > 0 || buySideCurves.length > 0) {
                curvesHtml += '<div style="font-size:0.7rem;font-weight:600;color:#9c27b0;margin-bottom:4px;">Curves</div>';
                sellSideCurves.slice(0, 2).forEach((c, i) => {
                    const q = c.quality || 'WEAK';
                    const qColor = q === 'EXCELLENT' ? '#00ff88' : q === 'GOOD' ? '#ffc107' : '#888';
                    curvesHtml += '<div style="display:flex;justify-content:space-between;font-size:0.6rem;padding:1px 0;">';
                    curvesHtml += '<span style="color:#ff6b6b;">Sell #' + (i + 1) + ' (' + (c.tap_count || 0) + ' taps)</span>';
                    curvesHtml += '<span style="color:' + qColor + ';">' + q + '</span>';
                    curvesHtml += '</div>';
                });
                buySideCurves.slice(0, 2).forEach((c, i) => {
                    const q = c.quality || 'WEAK';
                    const qColor = q === 'EXCELLENT' ? '#00ff88' : q === 'GOOD' ? '#ffc107' : '#888';
                    curvesHtml += '<div style="display:flex;justify-content:space-between;font-size:0.6rem;padding:1px 0;">';
                    curvesHtml += '<span style="color:#4ecdc4;">Buy #' + (i + 1) + ' (' + (c.tap_count || 0) + ' taps)</span>';
                    curvesHtml += '<span style="color:' + qColor + ';">' + q + '</span>';
                    curvesHtml += '</div>';
                });
            }
            document.getElementById('liqCurvesContent').innerHTML = curvesHtml;

            // Extreme liquidity targets (matching /liquidity page)
            const sellTargets = data.extreme_targets?.htf?.sell_side_targets || [];
            const buyTargets = data.extreme_targets?.htf?.buy_side_targets || [];
            let targetsHtml = '';
            if (sellTargets.length > 0 || buyTargets.length > 0) {
                targetsHtml += '<div style="font-size:0.7rem;font-weight:600;color:#4caf50;margin-bottom:4px;">Extreme Targets</div>';
                [...sellTargets, ...buyTargets].slice(0, 3).forEach(t => {
                    const isSwept = t.is_swept;
                    const label = t.type === 'distribution' ? 'Dist' : 'Accum';
                    const color = isSwept ? '#ff4444' : '#4caf50';
                    const statusTag = isSwept ? 'SWEPT' : 'ACTIVE';
                    targetsHtml += '<div style="display:flex;justify-content:space-between;font-size:0.6rem;padding:2px 0;">';
                    targetsHtml += '<span style="color:' + color + ';">' + label + ': ' + fmtPrice(t.target_price || 0) + '</span>';
                    targetsHtml += '<span style="padding:1px 4px;border-radius:2px;background:' + (isSwept ? 'rgba(255,68,68,0.15)' : 'rgba(76,175,80,0.15)') + ';color:' + color + ';font-size:0.55rem;' + (isSwept ? 'text-decoration:line-through;' : '') + '">' + statusTag + '</span>';
                    targetsHtml += '</div>';
                });
            }
            document.getElementById('liqTargetsContent').innerHTML = targetsHtml;

            // === DRAW LIQUIDITY CURVES ON CHART ===
            // Draw best quality sell-side curve
            if (sellSideCurves.length > 0 && candles.length > 0) {
                const bestSellCurve = [...sellSideCurves].sort((a, b) =>
                    (b.quality || b.tap_count || 0) - (a.quality || a.tap_count || 0)
                )[0];
                if (bestSellCurve && bestSellCurve.taps && bestSellCurve.taps.length >= 2) {
                    addLiquidityCurve(bestSellCurve, candles, true);
                }
            }

            // Draw best quality buy-side curve
            if (buySideCurves.length > 0 && candles.length > 0) {
                const bestBuyCurve = [...buySideCurves].sort((a, b) =>
                    (b.quality || b.tap_count || 0) - (a.quality || a.tap_count || 0)
                )[0];
                if (bestBuyCurve && bestBuyCurve.taps && bestBuyCurve.taps.length >= 2) {
                    addLiquidityCurve(bestBuyCurve, candles, false);
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

        function buildSchematicCardHTML(s, tf) {
            const isAccum = s.direction === 'bullish';
            const typeClass = isAccum ? 'accumulation' : 'distribution';
            const schType = s.schematic_type || '';
            let typeLabel = schType.replace(/_/g, ' ').toUpperCase() || (isAccum ? 'ACCUMULATION' : 'DISTRIBUTION');
            const quality = Math.round((s.quality_score || 0) * 100);
            const entry = s.entry?.price;
            const stop = s.stop_loss?.price;
            const target = s.target?.price;
            const rr = s.risk_reward;
            const isSafe = s.entry?.is_safe !== false;
            const isConfirmed = s.is_confirmed;
            const statusLabel = isConfirmed ? 'CONFIRMED' : 'FORMING';
            const statusColor = isConfirmed ? '#00ff88' : '#ffc107';
            const statusBg = isConfirmed ? 'rgba(0,255,136,0.2)' : 'rgba(255,193,7,0.2)';

            // Lecture 5B enhancements
            const enhancements = s.lecture_5b_enhancements || {};
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

            // Tap status
            const tap1 = s.tap1;
            const tap2 = s.tap2;
            const tap3 = s.tap3;
            let tapStatus = '';
            if (tap3) tapStatus = 'Tap3 ' + (isConfirmed ? 'confirmed' : 'forming');
            else if (tap2) tapStatus = 'Tap2 hit, awaiting Tap3';
            else if (tap1) tapStatus = 'Tap1 hit';

            const chartUrl = '/schematic-chart?symbol=' + currentSymbol + '&timeframe=' + tf;

            let html = '<a href="' + chartUrl + '" target="_blank" rel="noopener" class="schematic-link" title="Open schematic chart in new tab">';
            html += '<div class="schematic-item ' + typeClass + '">';
            html += '<div class="schematic-header">';
            html += '<span class="schematic-type">' + typeLabel + '</span>';
            html += '<span style="display:flex;gap:4px;align-items:center;">';
            html += '<span style="padding:2px 6px;border-radius:3px;background:' + statusBg + ';color:' + statusColor + ';font-size:0.6rem;font-weight:600;">' + statusLabel + '</span>';
            html += '<span class="schematic-quality">' + quality + '%</span>';
            html += '</span>';
            html += '</div>';

            // Direction badge
            html += '<div style="display:flex;justify-content:space-between;align-items:center;margin:4px 0;font-size:0.65rem;">';
            html += '<span style="color:' + (isAccum ? '#00ff88' : '#ff4444') + ';font-weight:600;">' + (isAccum ? 'LONG' : 'SHORT') + '</span>';
            if (tapStatus) html += '<span style="color:#888;">' + tapStatus + '</span>';
            html += '</div>';

            if (entry && stop && target) {
                html += '<div class="schematic-levels">';
                html += '<div class="level-box entry"><span class="level-label">ENTRY</span><span class="level-price">' + fmtPrice(entry) + '</span></div>';
                html += '<div class="level-box stop"><span class="level-label">STOP</span><span class="level-price">' + fmtPrice(stop) + '</span></div>';
                html += '<div class="level-box target"><span class="level-label">TARGET</span><span class="level-price">' + fmtPrice(target) + '</span></div>';
                html += '</div>';
            }

            html += '<div class="schematic-meta">';
            if (rr) html += '<span class="rr">R:R ' + rr.toFixed(1) + '</span>';
            html += '<span class="' + (isSafe ? 'safe' : 'unsafe') + '">' + (isSafe ? 'Safe Entry' : 'Caution: S/D Zone') + '</span>';
            if (has6CR) html += '<span class="safe">6CR Valid</span>';
            if (hasTrendline) html += '<span class="safe">TL Confluence</span>';
            html += '</div>';

            // Lecture 6 advanced indicators
            if (hasConversion || hasDualDev || hasWOV || hasM1toM2 || followBias) {
                html += '<div class="schematic-meta l6-indicators" style="margin-top:4px;border-top:1px solid #333;padding-top:4px;">';
                if (hasConversion) html += '<span style="color:#ff9800;">Converted</span>';
                if (hasDualDev) html += '<span style="color:#e91e63;">Dual Dev</span>';
                if (hasWOV) html += '<span style="color:#00bcd4;">WOV Entry</span>';
                if (hasM1toM2) html += '<span style="color:#9c27b0;">M1&rarr;M2</span>';
                if (followBias && followBias !== 'neutral') html += '<span style="color:#8bc34a;">' + followBias + '</span>';
                if (enhancedTarget) html += '<span style="color:#ffc107;">Ext: ' + fmtPrice(enhancedTarget) + '</span>';
                html += '</div>';
            }
            html += '<div class="view-chart-hint">View chart &rarr;</div>';
            html += '</div></a>';
            return html;
        }

        function buildTFGroupHTML(label, tf, schematics, buildCardFn) {
            if (!schematics || schematics.length === 0) {
                return '<div class="tf-group"><div class="tf-group-header"><span class="tf-label">' + label + ' (' + tf + ')</span><span class="tf-count">No schematics</span></div></div>';
            }
            let html = '<div class="tf-group">';
            html += '<div class="tf-group-header"><span class="tf-label">' + label + ' (' + tf + ')</span><span class="tf-count">' + schematics.length + ' found</span></div>';
            schematics.slice(0, 3).forEach(s => {
                html += buildCardFn(s, tf);
            });
            html += '</div>';
            return html;
        }

        function updateSchematicsUI(data) {
            const contentEl = document.getElementById('schematicsContent');
            const badgeEl = document.getElementById('schematicsBadge');

            // Reset scan button state
            const scanBtn = document.getElementById('scanTCTBtn');
            if (scanBtn) { scanBtn.disabled = false; scanBtn.classList.remove('scanning'); document.getElementById('scanTCTIcon').textContent = '\\u{1F50D}'; }

            if (data.error) {
                contentEl.innerHTML = '<div class="metric-row"><span class="label">Error loading schematics</span></div>';
                badgeEl.textContent = 'ERR';
                badgeEl.className = 'badge badge-bearish';
                return;
            }

            // Get schematics for all 3 timeframes (use timeframes from response)
            const htfTF = data.htf_schematics?.timeframe || 'HTF';
            const mtfTF = data.mtf_schematics?.timeframe || 'MTF';
            const ltfTF = data.ltf_schematics?.timeframe || 'LTF';
            const htfSchematics = data.htf_schematics?.schematics || [];
            const mtfSchematics = data.mtf_schematics?.schematics || [];
            const ltfSchematics = data.ltf_schematics?.schematics || [];
            const allSchematics = [...htfSchematics, ...mtfSchematics, ...ltfSchematics];

            if (allSchematics.length === 0) {
                contentEl.innerHTML = '<div class="metric-row"><span class="label">No active schematics detected</span></div>' +
                    '<div class="metric-row" style="margin-top:4px;"><span class="label" style="font-size:0.6rem;color:#666;">Scanned HTF (' + htfTF + '), MTF (' + mtfTF + '), LTF (' + ltfTF + ')</span></div>';
                badgeEl.textContent = '0';
                badgeEl.className = 'badge badge-neutral';
                return;
            }

            // Update badge
            const totalCount = allSchematics.length;
            const hasAccum = allSchematics.some(s => s.direction === 'bullish');
            const hasDist = allSchematics.some(s => s.direction === 'bearish');
            badgeEl.textContent = totalCount;
            badgeEl.className = 'badge badge-' + (hasAccum && !hasDist ? 'bullish' : hasDist && !hasAccum ? 'bearish' : 'neutral');

            // Build grouped HTML using actual timeframes from response
            let html = '';
            html += buildTFGroupHTML('HTF', htfTF, htfSchematics, buildSchematicCardHTML);
            html += buildTFGroupHTML('MTF', mtfTF, mtfSchematics, buildSchematicCardHTML);
            html += buildTFGroupHTML('LTF', ltfTF, ltfSchematics, buildSchematicCardHTML);

            contentEl.innerHTML = html;
        }

        // Scan TCT schematics on demand
        async function scanTCTSchematics() {
            const btn = document.getElementById('scanTCTBtn');
            const icon = document.getElementById('scanTCTIcon');
            btn.disabled = true;
            btn.classList.add('scanning');
            icon.textContent = '\\u23F3';

            const contentEl = document.getElementById('schematicsContent');
            contentEl.innerHTML = '<div class="metric-row"><span class="label" style="color:#ffc107;">Scanning schematics for ' + currentSymbol + ' (' + currentTimeframe + ')...</span></div>';

            try {
                const data = await fetchWithRetry('/api/schematics?symbol=' + currentSymbol + '&timeframe=' + currentTimeframe, {}, 3, 30000);
                updateSchematicsUI(data);
            } catch (e) {
                contentEl.innerHTML = '<div class="metric-row"><span class="label" style="color:#ff4444;">Scan failed: ' + e.message + '</span></div>';
                btn.disabled = false;
                btn.classList.remove('scanning');
                icon.textContent = '\\u{1F50D}';
            }
        }

        // Timeframe dropdown selector
        document.getElementById('tfDropdown').addEventListener('change', async (e) => {
            currentTimeframe = e.target.value;
            await refreshData();
        });

        // ===== PAIR SELECTOR =====
        const pairSearchEl = document.getElementById('pairSearch');
        const pairDropdownEl = document.getElementById('pairDropdown');
        let pairDropdownOpen = false;

        // Load coin list from server
        async function loadCoinList() {
            try {
                const data = await fetchWithRetry('/api/coin-list', {}, 2, 10000);
                if (data && data.all) {
                    coinList = data;
                    console.log('Loaded', data.total, 'trading pairs');
                }
            } catch (e) {
                console.error('Failed to load coin list:', e);
            }
        }

        // Render the pair dropdown
        function renderPairDropdown(filter = '') {
            const f = filter.toUpperCase();
            let html = '';

            const categoryLabels = {
                'majors': 'Majors',
                'defi': 'DeFi',
                'layer_1_2': 'Layer 1 / Layer 2',
                'meme': 'Meme',
                'ai': 'AI & Compute'
            };

            // If filtering, show flat list
            if (f.length > 0) {
                const matches = coinList.all.filter(p => p.includes(f)).slice(0, 40);
                if (matches.length === 0) {
                    html = '<div class="pair-dropdown-item" style="color:#666;cursor:default;">No matches</div>';
                } else {
                    matches.forEach(pair => {
                        const base = pair.replace('USDT', '');
                        const isActive = pair === currentSymbol ? ' active' : '';
                        html += '<div class="pair-dropdown-item' + isActive + '" data-pair="' + pair + '">';
                        html += '<span class="pair-base">' + base + '</span>';
                        html += '<span class="pair-quote">/ USDT</span>';
                        html += '</div>';
                    });
                }
            } else {
                // Show categorized
                for (const [cat, label] of Object.entries(categoryLabels)) {
                    const pairs = coinList.categories[cat] || [];
                    if (pairs.length > 0) {
                        html += '<div class="pair-dropdown-group">';
                        html += '<div class="pair-dropdown-group-label">' + label + '</div>';
                        pairs.forEach(pair => {
                            const base = pair.replace('USDT', '');
                            const isActive = pair === currentSymbol ? ' active' : '';
                            html += '<div class="pair-dropdown-item' + isActive + '" data-pair="' + pair + '">';
                            html += '<span class="pair-base">' + base + '</span>';
                            html += '<span class="pair-quote">/ USDT</span>';
                            html += '</div>';
                        });
                        html += '</div>';
                    }
                }
                // Also show "All Pairs" section for remaining
                html += '<div class="pair-dropdown-group">';
                html += '<div class="pair-dropdown-group-label">All Pairs (' + coinList.all.length + ')</div>';
                coinList.all.slice(0, 30).forEach(pair => {
                    const base = pair.replace('USDT', '');
                    const isActive = pair === currentSymbol ? ' active' : '';
                    html += '<div class="pair-dropdown-item' + isActive + '" data-pair="' + pair + '">';
                    html += '<span class="pair-base">' + base + '</span>';
                    html += '<span class="pair-quote">/ USDT</span>';
                    html += '</div>';
                });
                if (coinList.all.length > 30) {
                    html += '<div class="pair-dropdown-item" style="color:#555;cursor:default;font-style:italic;">Type to search ' + (coinList.all.length - 30) + ' more...</div>';
                }
                html += '</div>';
            }

            pairDropdownEl.innerHTML = html;

            // Add click handlers
            pairDropdownEl.querySelectorAll('.pair-dropdown-item[data-pair]').forEach(item => {
                item.addEventListener('click', () => {
                    selectPair(item.dataset.pair);
                });
            });
        }

        async function selectPair(pair) {
            currentSymbol = pair;
            pairSearchEl.value = pair;
            document.getElementById('headerSymbol').textContent = pair;
            closePairDropdown();
            await refreshData();
        }

        function openPairDropdown() {
            if (!pairDropdownOpen) {
                pairDropdownOpen = true;
                pairDropdownEl.classList.add('open');
                renderPairDropdown(pairSearchEl.value === currentSymbol ? '' : pairSearchEl.value);
            }
        }

        function closePairDropdown() {
            pairDropdownOpen = false;
            pairDropdownEl.classList.remove('open');
        }

        pairSearchEl.addEventListener('focus', () => {
            pairSearchEl.select();
            openPairDropdown();
        });

        pairSearchEl.addEventListener('input', () => {
            openPairDropdown();
            renderPairDropdown(pairSearchEl.value);
        });

        pairSearchEl.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                const val = pairSearchEl.value.toUpperCase().trim();
                // Check if valid pair
                if (coinList.all.includes(val)) {
                    selectPair(val);
                } else if (coinList.all.includes(val + 'USDT')) {
                    selectPair(val + 'USDT');
                } else {
                    // Try first match
                    const match = coinList.all.find(p => p.includes(val));
                    if (match) selectPair(match);
                }
                e.preventDefault();
            } else if (e.key === 'Escape') {
                pairSearchEl.value = currentSymbol;
                closePairDropdown();
                pairSearchEl.blur();
            }
        });

        // Close dropdown when clicking outside
        document.addEventListener('click', (e) => {
            if (!e.target.closest('.pair-search-wrapper')) {
                if (pairDropdownOpen) {
                    pairSearchEl.value = currentSymbol;
                    closePairDropdown();
                }
            }
        });

        // Load coin list on startup
        loadCoinList();

        // Map timeframe to optimal candle count
        function getCandleLimit(tf) {
            const limits = {
                '1m': 200, '5m': 200, '15m': 200, '30m': 150,
                '1h': 150, '2h': 120, '4h': 100, '8h': 80,
                '1d': 120, '1W': 100, '1M': 60
            };
            return limits[tf] || 100;
        }

        // Get the HTF timeframe corresponding to any selected timeframe
        function getHTFForTimeframe(tf) {
            const htfMap = {
                '1m': '15m', '5m': '1h', '15m': '4h', '30m': '4h',
                '1h': '4h', '2h': '1d', '4h': '1d', '8h': '1d',
                '1d': '1W', '1W': '1M', '1M': '1M'
            };
            return htfMap[tf] || '1d';
        }

        // ===== TCT MODEL SCHEMATIC OVERLAYS =====
        // Based on TCT model schematics: parabolic curves, S/R zones, SL trails, targets

        // Draw a parabolic curve (ascending lows or descending highs) on chart
        function drawParabolicCurve(candles, direction = 'bullish') {
            if (!candles || candles.length < 10) return;

            const isBullish = direction === 'bullish';
            const pivotPoints = [];

            // Find swing points for the curve
            for (let i = 3; i < candles.length - 3; i++) {
                if (isBullish) {
                    // Ascending lows - find swing lows
                    const isSwingLow = candles[i].low <= candles[i-1].low && candles[i].low <= candles[i-2].low &&
                                       candles[i].low <= candles[i+1].low && candles[i].low <= candles[i+2].low;
                    if (isSwingLow) {
                        pivotPoints.push({ time: candles[i].time, value: candles[i].low, idx: i });
                    }
                } else {
                    // Descending highs - find swing highs
                    const isSwingHigh = candles[i].high >= candles[i-1].high && candles[i].high >= candles[i-2].high &&
                                        candles[i].high >= candles[i+1].high && candles[i].high >= candles[i+2].high;
                    if (isSwingHigh) {
                        pivotPoints.push({ time: candles[i].time, value: candles[i].high, idx: i });
                    }
                }
            }

            if (pivotPoints.length < 3) return;

            // Filter for ascending (bullish) or descending (bearish) pattern
            const filteredPivots = [];
            for (let i = 0; i < pivotPoints.length; i++) {
                if (filteredPivots.length === 0) {
                    filteredPivots.push(pivotPoints[i]);
                } else {
                    const last = filteredPivots[filteredPivots.length - 1];
                    if (isBullish && pivotPoints[i].value >= last.value) {
                        filteredPivots.push(pivotPoints[i]);
                    } else if (!isBullish && pivotPoints[i].value <= last.value) {
                        filteredPivots.push(pivotPoints[i]);
                    }
                }
            }

            if (filteredPivots.length < 3) return;

            // Interpolate a smooth parabolic curve between pivot points
            const curveData = [];
            for (let i = 0; i < filteredPivots.length - 1; i++) {
                const p1 = filteredPivots[i];
                const p2 = filteredPivots[i + 1];
                const startIdx = p1.idx;
                const endIdx = p2.idx;

                for (let j = startIdx; j <= endIdx; j++) {
                    const t = (j - startIdx) / (endIdx - startIdx);
                    // Parabolic interpolation (quadratic ease)
                    const eased = isBullish ? t * t : 1 - (1 - t) * (1 - t);
                    const value = p1.value + (p2.value - p1.value) * eased;
                    curveData.push({ time: candles[j].time, value: value });
                }
            }

            // Extend curve forward (projection)
            const lastPivot = filteredPivots[filteredPivots.length - 1];
            const secondLast = filteredPivots[filteredPivots.length - 2];
            const slope = (lastPivot.value - secondLast.value) / (lastPivot.idx - secondLast.idx);

            for (let j = lastPivot.idx + 1; j < candles.length; j++) {
                const accel = isBullish ? 1.15 : 0.85;
                const projValue = lastPivot.value + slope * (j - lastPivot.idx) * accel;
                curveData.push({ time: candles[j].time, value: projValue });
            }

            if (curveData.length >= 2) {
                // Deduplicate by time
                const seen = new Set();
                const uniqueData = curveData.filter(d => {
                    if (seen.has(d.time)) return false;
                    seen.add(d.time);
                    return true;
                });

                const color = isBullish ? 'rgba(0, 255, 136, 0.6)' : 'rgba(255, 68, 68, 0.6)';
                const curveSeries = chart.addLineSeries({
                    color: color,
                    lineWidth: 2,
                    lineStyle: LightweightCharts.LineStyle.Dotted,
                    crosshairMarkerVisible: false,
                    priceLineVisible: false,
                    lastValueVisible: false,
                    priceScaleId: 'right',
                });
                curveSeries.setData(uniqueData);
                additionalSeries.push(curveSeries);
            }
        }

        // Draw shaded resistance/support zone (like the pink/red zones in TradingView screenshots)
        function drawZoneBox(high, low, candles, type = 'resistance', startIdx = 0) {
            if (!candles || candles.length === 0 || !high || !low) return;

            const start = Math.max(0, startIdx);
            const startTime = candles[start].time;
            const endTime = candles[candles.length - 1].time;

            let topColor, bottomColor, lineColor;
            if (type === 'resistance' || type === 'supply') {
                topColor = 'rgba(255, 107, 107, 0.25)';
                bottomColor = 'rgba(255, 107, 107, 0.08)';
                lineColor = 'rgba(255, 107, 107, 0.6)';
            } else if (type === 'target' || type === 'demand') {
                topColor = 'rgba(130, 130, 255, 0.20)';
                bottomColor = 'rgba(130, 130, 255, 0.06)';
                lineColor = 'rgba(130, 130, 255, 0.5)';
            } else {
                topColor = 'rgba(255, 193, 7, 0.15)';
                bottomColor = 'rgba(255, 193, 7, 0.05)';
                lineColor = 'rgba(255, 193, 7, 0.4)';
            }

            // Upper boundary line
            const upperData = [];
            const lowerData = [];
            for (let i = start; i < candles.length; i++) {
                upperData.push({ time: candles[i].time, value: high });
                lowerData.push({ time: candles[i].time, value: low });
            }

            if (upperData.length < 2) return;

            // Upper line
            const upperSeries = chart.addLineSeries({
                color: lineColor,
                lineWidth: 1,
                lineStyle: LightweightCharts.LineStyle.Solid,
                crosshairMarkerVisible: false,
                priceLineVisible: false,
                lastValueVisible: false,
                priceScaleId: 'right',
            });
            upperSeries.setData(upperData);
            additionalSeries.push(upperSeries);

            // Lower line
            const lowerSeries = chart.addLineSeries({
                color: lineColor,
                lineWidth: 1,
                lineStyle: LightweightCharts.LineStyle.Solid,
                crosshairMarkerVisible: false,
                priceLineVisible: false,
                lastValueVisible: false,
                priceScaleId: 'right',
            });
            lowerSeries.setData(lowerData);
            additionalSeries.push(lowerSeries);

            // Fill area (using area series trick)
            const fillSeries = chart.addAreaSeries({
                topColor: topColor,
                bottomColor: bottomColor,
                lineColor: 'rgba(0,0,0,0)',
                lineWidth: 0,
                priceScaleId: 'right',
                lastValueVisible: false,
                priceLineVisible: false,
            });
            fillSeries.setData(upperData);
            additionalSeries.push(fillSeries);
        }

        // Draw stop loss trail line
        function drawSLTrail(price, candles, startIdx = 0, label = 'SL Trail') {
            if (!candles || candles.length === 0 || !price) return;

            const start = Math.max(0, startIdx);
            const trailData = [];
            for (let i = start; i < candles.length; i++) {
                trailData.push({ time: candles[i].time, value: price });
            }

            if (trailData.length < 2) return;

            const trailSeries = chart.addLineSeries({
                color: 'rgba(255, 107, 182, 0.7)',
                lineWidth: 2,
                lineStyle: LightweightCharts.LineStyle.Dashed,
                crosshairMarkerVisible: false,
                priceLineVisible: true,
                lastValueVisible: true,
                priceScaleId: 'right',
                title: label,
            });
            trailSeries.setData(trailData);
            additionalSeries.push(trailSeries);
        }

        // Draw key horizontal level (like the thick black lines in the screenshots)
        function drawKeyLevel(price, candles, color = '#ffffff', label = '', lineWidth = 2) {
            if (!candles || candles.length === 0 || !price) return;

            const levelData = candles.map(c => ({ time: c.time, value: price }));
            const levelSeries = chart.addLineSeries({
                color: color,
                lineWidth: lineWidth,
                lineStyle: LightweightCharts.LineStyle.Solid,
                crosshairMarkerVisible: false,
                priceLineVisible: false,
                lastValueVisible: false,
                priceScaleId: 'right',
            });
            levelSeries.setData(levelData);
            additionalSeries.push(levelSeries);

            if (label) {
                lineSeries.push(addPriceLine(price, color, label, 0, lineWidth));
            }
        }

        // Add trade entry/exit markers to chart
        function addTradeMarkers(candles, entry, sl, tp, direction = 'long') {
            if (!candles || candles.length < 5 || !entry) return;

            const markers = [];
            const lastIdx = candles.length - 1;

            // Find candle closest to entry price
            let entryIdx = lastIdx - 3;
            let minDiff = Infinity;
            for (let i = Math.max(0, lastIdx - 20); i <= lastIdx; i++) {
                const diff = Math.abs(candles[i].close - entry);
                if (diff < minDiff) {
                    minDiff = diff;
                    entryIdx = i;
                }
            }

            const isLong = direction === 'long';

            markers.push({
                time: candles[entryIdx].time,
                position: isLong ? 'belowBar' : 'aboveBar',
                color: '#00d4ff',
                shape: isLong ? 'arrowUp' : 'arrowDown',
                text: 'ENTRY'
            });

            if (sl) {
                // SL marker at a nearby candle
                const slIdx = Math.min(lastIdx, entryIdx + 1);
                markers.push({
                    time: candles[slIdx].time,
                    position: isLong ? 'belowBar' : 'aboveBar',
                    color: '#ff4444',
                    shape: 'circle',
                    text: 'SL'
                });
            }

            if (tp) {
                // TP marker
                const tpIdx = Math.min(lastIdx, entryIdx + 2);
                markers.push({
                    time: candles[tpIdx].time,
                    position: isLong ? 'aboveBar' : 'belowBar',
                    color: '#00ff88',
                    shape: 'circle',
                    text: 'TP'
                });
            }

            if (markers.length > 0) {
                markers.sort((a, b) => a.time - b.time);
                candleSeries.setMarkers(markers);
            }
        }

        // ===== HIGHEST PROBABILITY SETUP ANALYSIS =====

        function analyzeHighestProbabilitySetup(rangesData, zonesData, liqData, schematicsData, po3Data, valData, candles) {
            const setup = {
                direction: 'none',
                confidence: 0,
                entry: null,
                stop: null,
                target: null,
                rr: null,
                source: null,
                tags: [],
                resistanceZones: [],
                supportZones: [],
                keyLevels: [],
                slTrail: null,
            };

            // Collect all candidate setups with scores
            const candidates = [];

            // 1. Check TCT Schematics (highest priority - Lecture 5/6)
            if (schematicsData) {
                const htfS = schematicsData.htf_schematics?.schematics || [];
                const mtfS = schematicsData.mtf_schematics?.schematics || [];
                const ltfS = schematicsData.ltf_schematics?.schematics || [];
                [...htfS, ...mtfS, ...ltfS].forEach(s => {
                    if (s.entry?.price && s.stop_loss?.price && s.target?.price) {
                        const quality = s.quality_score || 0;
                        const rr = s.risk_reward || 0;
                        const isConfirmed = s.is_confirmed ? 1 : 0;
                        const has6CR = s.lecture_5b_enhancements?.htf_validation?.all_taps_valid_6cr ? 1 : 0;
                        const hasTL = s.lecture_5b_enhancements?.has_trendline_confluence ? 1 : 0;
                        const l6Score = (s.lecture_6_enhancements?.has_conversion ? 0.1 : 0) +
                                       (s.lecture_6_enhancements?.has_dual_deviation ? 0.1 : 0) +
                                       (s.lecture_6_enhancements?.has_wov_opportunity ? 0.05 : 0);

                        const score = (quality * 40) + (Math.min(rr, 5) * 8) + (isConfirmed * 15) + (has6CR * 10) + (hasTL * 5) + (l6Score * 100);

                        candidates.push({
                            score,
                            direction: s.direction === 'bullish' ? 'long' : 'short',
                            entry: s.entry.price,
                            stop: s.stop_loss.price,
                            target: s.lecture_6_enhancements?.enhanced_target || s.target.price,
                            rr: rr,
                            isConfirmed: !!isConfirmed,
                            source: 'TCT Schematic (' + (s.schematic_type || 'unknown').replace(/_/g, ' ') + ')',
                            tags: [
                                isConfirmed ? { text: 'CONFIRMED', cls: 'good' } : { text: 'FORMING', cls: 'warn' },
                                quality >= 0.7 ? { text: 'HQ ' + Math.round(quality * 100) + '%', cls: 'good' } : { text: 'Q ' + Math.round(quality * 100) + '%', cls: 'warn' },
                                has6CR ? { text: '6CR Valid', cls: 'good' } : null,
                                hasTL ? { text: 'TL Confluence', cls: 'good' } : null,
                                s.lecture_6_enhancements?.has_conversion ? { text: 'Converted', cls: 'good' } : null,
                                rr >= 3 ? { text: 'R:R ' + rr.toFixed(1), cls: 'good' } : { text: 'R:R ' + rr.toFixed(1), cls: 'warn' },
                            ].filter(Boolean),
                        });
                    }
                });
            }

            // 2. Check PO3 Schematics (Lecture 8)
            if (po3Data) {
                const htfP = po3Data.htf_po3?.schematics || [];
                const mtfP = po3Data.mtf_po3?.schematics || [];
                const ltfP = po3Data.ltf_po3?.schematics || [];
                [...htfP, ...mtfP, ...ltfP].forEach(p => {
                    if (p.entry?.price && p.stop_loss?.price && p.target?.price) {
                        const quality = p.quality_score || 0;
                        const rr = p.risk_reward || 0;
                        const hasExpansion = p.has_expansion ? 1 : 0;
                        const hasTCTModel = p.tct_model?.detected ? 1 : 0;

                        const score = (quality * 35) + (Math.min(rr, 5) * 7) + (hasExpansion * 12) + (hasTCTModel * 10);

                        candidates.push({
                            score,
                            direction: p.direction === 'bullish' ? 'long' : 'short',
                            entry: p.entry.price,
                            stop: p.stop_loss.price,
                            target: p.target.price,
                            rr: rr,
                            source: 'PO3 (' + (p.phase || 'range').replace(/_/g, ' ') + ')',
                            tags: [
                                { text: 'PO3', cls: 'good' },
                                quality >= 0.6 ? { text: Math.round(quality * 100) + '%', cls: 'good' } : { text: Math.round(quality * 100) + '%', cls: 'warn' },
                                hasExpansion ? { text: 'Expanding', cls: 'good' } : null,
                                hasTCTModel ? { text: 'TCT Model', cls: 'good' } : null,
                                rr >= 3 ? { text: 'R:R ' + rr.toFixed(1), cls: 'good' } : { text: 'R:R ' + rr.toFixed(1), cls: 'warn' },
                            ].filter(Boolean),
                        });
                    }
                });
            }

            // 3. Check gate validation for directional bias
            let gateBias = 'none';
            if (valData && valData.gates) {
                const action = valData.Action || '';
                if (action.includes('LONG') || action.includes('VALID')) gateBias = 'long';
                else if (action.includes('SHORT')) gateBias = 'short';
            }

            // 4. Use range data for S/R zones overlay
            if (rangesData) {
                const activeRange = rangesData.htf_ranges?.active_range || rangesData.ltf_ranges?.active_range;
                if (activeRange) {
                    const high = activeRange.range_high || activeRange.high;
                    const low = activeRange.range_low || activeRange.low;
                    if (high) setup.keyLevels.push({ price: high, label: 'Range High', color: '#ff4444' });
                    if (low) setup.keyLevels.push({ price: low, label: 'Range Low', color: '#00ff88' });
                }
            }

            // 5. Collect zone data for overlays
            if (zonesData) {
                const topZones = zonesData.htf_zones?.top_3_high_quality || zonesData.htf_zones?.top_3_all || [];
                topZones.forEach(z => {
                    if (z.type === 'supply' || z.top > (rangesData?.current_price || 0)) {
                        setup.resistanceZones.push({ high: z.top, low: z.bottom });
                    } else {
                        setup.supportZones.push({ high: z.top, low: z.bottom });
                    }
                });
            }

            // Pick best candidate
            if (candidates.length > 0) {
                // Boost candidates that match gate bias
                candidates.forEach(c => {
                    if (gateBias !== 'none' && c.direction === gateBias) {
                        c.score *= 1.3;
                    }
                });

                candidates.sort((a, b) => b.score - a.score);
                const best = candidates[0];

                setup.direction = best.direction;
                setup.confidence = Math.min(100, Math.round(best.score));
                setup.entry = best.entry;
                setup.stop = best.stop;
                setup.target = best.target;
                setup.rr = best.rr;
                setup.source = best.source;
                setup.tags = best.tags;
                setup.isConfirmed = best.isConfirmed || false;

                // Calculate SL trail (midpoint between entry and target)
                if (setup.entry && setup.target) {
                    setup.slTrail = setup.entry + (setup.target - setup.entry) * 0.33;
                }
            } else {
                // No schematic/PO3 candidates — build a forming setup from range + candle analysis
                const formingSetup = analyzeFormingSetupFromData(rangesData, zonesData, liqData, candles, gateBias);
                setup.direction = formingSetup.direction;
                setup.confidence = formingSetup.confidence;
                setup.entry = formingSetup.entry;
                setup.stop = formingSetup.stop;
                setup.target = formingSetup.target;
                setup.rr = formingSetup.rr;
                setup.source = formingSetup.source;
                setup.tags = formingSetup.tags;
                setup.isConfirmed = false;
            }

            return setup;
        }

        // Fallback: analyze forming setups from range/zone/candle data when no schematics exist
        function analyzeFormingSetupFromData(rangesData, zonesData, liqData, candles, gateBias) {
            const result = { direction: 'none', confidence: 0, entry: null, stop: null, target: null, rr: null, source: null, tags: [] };

            if (!candles || candles.length < 20) {
                result.tags = [{ text: 'Insufficient Data', cls: 'warn' }];
                return result;
            }

            const lastPrice = candles[candles.length - 1].close;
            const recent20 = candles.slice(-20);
            const recent50 = candles.slice(-50);
            const recentHigh = Math.max(...recent50.map(c => c.high));
            const recentLow = Math.min(...recent50.map(c => c.low));
            const rangeSize = recentHigh - recentLow;
            const eq = (recentHigh + recentLow) / 2;

            // Determine trend from candle data
            const sma20 = recent20.reduce((s, c) => s + c.close, 0) / recent20.length;
            const firstHalf = recent20.slice(0, 10);
            const secondHalf = recent20.slice(-10);
            const smaFirst = firstHalf.reduce((s, c) => s + c.close, 0) / firstHalf.length;
            const smaSecond = secondHalf.reduce((s, c) => s + c.close, 0) / secondHalf.length;
            const trendUp = smaSecond > smaFirst;
            const trendStrength = Math.abs(smaSecond - smaFirst) / smaFirst * 100;

            // Check premium vs discount
            const inPremium = lastPrice > eq;
            const inDiscount = lastPrice < eq;
            const posInRange = (lastPrice - recentLow) / rangeSize;

            // Use range data if available
            let activeRange = null;
            if (rangesData) {
                activeRange = rangesData.htf_ranges?.active_range || rangesData.ltf_ranges?.active_range;
            }

            // Look at market structure from range data
            const msTrend = rangesData?.market_structure?.htf_trend || (trendUp ? 'bullish' : 'bearish');

            // Find nearest demand/supply zones
            let nearestDemand = null, nearestSupply = null;
            if (zonesData) {
                const allOBs = zonesData.htf_zones?.order_blocks || [];
                const allStruct = zonesData.htf_zones?.structure_zones || [];
                const allZones = [...allOBs, ...allStruct].filter(z => !z.mitigated);
                const demandZones = allZones.filter(z => z.type === 'bullish' || z.type === 'demand');
                const supplyZones = allZones.filter(z => z.type === 'bearish' || z.type === 'supply');

                demandZones.forEach(z => {
                    const mid = ((z.top || z.high || 0) + (z.bottom || z.low || 0)) / 2;
                    if (mid < lastPrice && (!nearestDemand || mid > nearestDemand.mid)) {
                        nearestDemand = { top: z.top || z.high, bottom: z.bottom || z.low, mid };
                    }
                });
                supplyZones.forEach(z => {
                    const mid = ((z.top || z.high || 0) + (z.bottom || z.low || 0)) / 2;
                    if (mid > lastPrice && (!nearestSupply || mid < nearestSupply.mid)) {
                        nearestSupply = { top: z.top || z.high, bottom: z.bottom || z.low, mid };
                    }
                });
            }

            // Find nearest liquidity
            let nearestBSL = null, nearestSSL = null;
            if (liqData) {
                const bslPools = liqData.htf_liquidity?.bsl_pools || [];
                const sslPools = liqData.htf_liquidity?.ssl_pools || [];
                if (bslPools.length > 0) nearestBSL = bslPools[0].price;
                if (sslPools.length > 0) nearestSSL = sslPools[0].price;
            }

            // Build forming setup
            let score = 0;
            const tags = [];

            if (activeRange) {
                const rHigh = activeRange.range_high || activeRange.high;
                const rLow = activeRange.range_low || activeRange.low;
                const rEq = activeRange.equilibrium || (rHigh + rLow) / 2;
                const rSize = rHigh - rLow;

                if (inDiscount && msTrend === 'bullish') {
                    // Bullish forming: in discount of range with bullish trend
                    result.direction = 'long';
                    result.entry = nearestDemand ? nearestDemand.top : rLow + rSize * 0.236;
                    result.stop = nearestDemand ? nearestDemand.bottom - rSize * 0.02 : rLow - rSize * 0.05;
                    result.target = nearestSupply ? nearestSupply.bottom : rHigh;
                    result.source = 'Range Discount + ' + msTrend + ' trend';
                    score = 35;
                    tags.push({ text: 'FORMING', cls: 'warn' });
                    tags.push({ text: 'Discount Zone', cls: 'good' });
                    tags.push({ text: 'Bullish Trend', cls: 'good' });
                    if (nearestDemand) { tags.push({ text: 'Near Demand OB', cls: 'good' }); score += 15; }
                    if (nearestSSL && nearestSSL < lastPrice) { tags.push({ text: 'SSL Below', cls: 'good' }); score += 10; }
                } else if (inPremium && msTrend === 'bearish') {
                    // Bearish forming: in premium of range with bearish trend
                    result.direction = 'short';
                    result.entry = nearestSupply ? nearestSupply.bottom : rHigh - rSize * 0.236;
                    result.stop = nearestSupply ? nearestSupply.top + rSize * 0.02 : rHigh + rSize * 0.05;
                    result.target = nearestDemand ? nearestDemand.top : rLow;
                    result.source = 'Range Premium + ' + msTrend + ' trend';
                    score = 35;
                    tags.push({ text: 'FORMING', cls: 'warn' });
                    tags.push({ text: 'Premium Zone', cls: 'good' });
                    tags.push({ text: 'Bearish Trend', cls: 'good' });
                    if (nearestSupply) { tags.push({ text: 'Near Supply OB', cls: 'good' }); score += 15; }
                    if (nearestBSL && nearestBSL > lastPrice) { tags.push({ text: 'BSL Above', cls: 'good' }); score += 10; }
                } else if (inDiscount) {
                    // In discount but trend unclear — watch for bullish reversal
                    result.direction = 'long';
                    result.entry = rLow + rSize * 0.236;
                    result.stop = rLow - rSize * 0.05;
                    result.target = rEq;
                    result.source = 'Discount bounce watch';
                    score = 20;
                    tags.push({ text: 'FORMING', cls: 'warn' });
                    tags.push({ text: 'Discount Zone', cls: 'good' });
                    tags.push({ text: 'Watch for HL', cls: 'warn' });
                } else if (inPremium) {
                    // In premium but trend unclear — watch for bearish reversal
                    result.direction = 'short';
                    result.entry = rHigh - rSize * 0.236;
                    result.stop = rHigh + rSize * 0.05;
                    result.target = rEq;
                    result.source = 'Premium rejection watch';
                    score = 20;
                    tags.push({ text: 'FORMING', cls: 'warn' });
                    tags.push({ text: 'Premium Zone', cls: 'warn' });
                    tags.push({ text: 'Watch for LH', cls: 'warn' });
                }
            } else {
                // No range data — use pure candle analysis
                if (trendUp && posInRange < 0.5) {
                    result.direction = 'long';
                    result.entry = recentLow + rangeSize * 0.382;
                    result.stop = recentLow - rangeSize * 0.05;
                    result.target = recentHigh;
                    result.source = 'Candle trend analysis';
                    score = 15;
                    tags.push({ text: 'FORMING', cls: 'warn' });
                    tags.push({ text: 'Uptrend', cls: 'good' });
                } else if (!trendUp && posInRange > 0.5) {
                    result.direction = 'short';
                    result.entry = recentHigh - rangeSize * 0.382;
                    result.stop = recentHigh + rangeSize * 0.05;
                    result.target = recentLow;
                    result.source = 'Candle trend analysis';
                    score = 15;
                    tags.push({ text: 'FORMING', cls: 'warn' });
                    tags.push({ text: 'Downtrend', cls: 'warn' });
                } else {
                    result.source = 'Awaiting structure';
                    tags.push({ text: 'NO SETUP', cls: 'warn' });
                    tags.push({ text: 'Range: ' + fmtPrice(recentLow) + ' - ' + fmtPrice(recentHigh), cls: '' });
                    tags.push({ text: trendUp ? 'Bias: Bullish' : 'Bias: Bearish', cls: 'warn' });
                }
            }

            // Apply gate bias boost
            if (gateBias !== 'none' && result.direction === gateBias) {
                score += 10;
                tags.push({ text: 'Gate Aligned', cls: 'good' });
            } else if (gateBias !== 'none') {
                tags.push({ text: 'Gate: ' + gateBias.toUpperCase(), cls: 'warn' });
            }

            // Calculate R:R
            if (result.entry && result.stop && result.target) {
                const risk = Math.abs(result.entry - result.stop);
                const reward = Math.abs(result.target - result.entry);
                result.rr = risk > 0 ? reward / risk : 0;
                if (result.rr >= 3) score += 10;
            }

            result.confidence = Math.min(100, score);
            result.tags = tags;
            return result;
        }

        // Render the highest probability setup in sidebar
        function renderSetupPanel(setup) {
            const dirEl = document.getElementById('setupDirection');
            dirEl.textContent = setup.direction === 'long' ? 'LONG' : setup.direction === 'short' ? 'SHORT' : 'NO SETUP';
            dirEl.className = 'setup-direction ' + setup.direction;

            const contentEl = document.getElementById('setupContent');
            let html = '';

            if (setup.entry && setup.stop && setup.target) {
                // Status banner (forming vs confirmed)
                const isConfirmed = setup.isConfirmed;
                const statusLabel = isConfirmed ? 'CONFIRMED - Active Trade Setup' : 'FORMING - Watch for Confirmation';
                const statusColor = isConfirmed ? '#00ff88' : '#ffc107';
                const statusBg = isConfirmed ? 'rgba(0,255,136,0.15)' : 'rgba(255,193,7,0.15)';
                html += '<div style="text-align:center;padding:4px 8px;border-radius:4px;background:' + statusBg + ';color:' + statusColor + ';font-size:0.7rem;font-weight:600;margin-bottom:8px;">' + statusLabel + '</div>';

                html += '<div class="setup-levels">';
                html += '<div class="setup-level-box entry"><span class="setup-level-label">ENTRY</span><span class="setup-level-price">' + fmtPrice(setup.entry) + '</span></div>';
                html += '<div class="setup-level-box sl"><span class="setup-level-label">STOP</span><span class="setup-level-price">' + fmtPrice(setup.stop) + '</span></div>';
                html += '<div class="setup-level-box tp"><span class="setup-level-label">TARGET</span><span class="setup-level-price">' + fmtPrice(setup.target) + '</span></div>';
                html += '</div>';

                if (setup.rr) {
                    html += '<div class="metric-row"><span class="label">Risk : Reward</span><span class="value" style="color:#ffc107;font-weight:bold;">' + setup.rr.toFixed(1) + 'R</span></div>';
                }
                if (setup.source) {
                    html += '<div class="metric-row"><span class="label">Source</span><span class="value" style="font-size:0.7rem;">' + setup.source + '</span></div>';
                }
            }

            if (setup.tags.length > 0) {
                html += '<div class="setup-meta">';
                setup.tags.forEach(tag => {
                    html += '<span class="setup-tag ' + tag.cls + '">' + tag.text + '</span>';
                });
                html += '</div>';
            }

            contentEl.innerHTML = html || '<div class="metric-row"><span class="label">No high probability setup on this timeframe</span></div>';

            // Confidence bar
            const confEl = document.getElementById('setupConfidence');
            const confPct = Math.min(100, setup.confidence);
            confEl.style.width = confPct + '%';
            if (confPct >= 70) confEl.style.background = '#00ff88';
            else if (confPct >= 40) confEl.style.background = '#ffc107';
            else confEl.style.background = '#ff4444';
        }

        // Draw all TCT model overlays on chart based on setup data
        function drawTCTModelOverlays(setup, candles) {
            if (!candles || candles.length < 10) return;

            // 1. Draw parabolic curve (like the cup-shaped curves in the screenshots)
            if (setup.direction === 'long') {
                drawParabolicCurve(candles, 'bullish');
            } else if (setup.direction === 'short') {
                drawParabolicCurve(candles, 'bearish');
            } else {
                // Draw both to show market structure
                drawParabolicCurve(candles, 'bullish');
            }

            // 2. Draw resistance zones (pink/red shaded areas from screenshots)
            setup.resistanceZones.forEach(z => {
                drawZoneBox(z.high, z.low, candles, 'resistance');
            });

            // 3. Draw support/target zones (blue/purple shaded areas)
            setup.supportZones.forEach(z => {
                drawZoneBox(z.high, z.low, candles, 'target');
            });

            // 4. Draw key levels (thick horizontal lines like in screenshots)
            setup.keyLevels.forEach(level => {
                drawKeyLevel(level.price, candles, level.color, level.label);
            });

            // 5. Draw entry/stop/target on chart if we have a setup
            if (setup.entry && setup.stop && setup.target) {
                const isForming = !setup.isConfirmed;
                const entryLabel = (isForming ? 'Entry (forming)' : 'Entry') + ' ' + fmtPrice(setup.entry);
                const slLabel = 'Stop ' + fmtPrice(setup.stop);
                const tpLabel = 'Target ' + fmtPrice(setup.target);

                // Entry level (cyan, prominent)
                lineSeries.push(addPriceLine(setup.entry, '#00d4ff', entryLabel, 0, 2));
                drawKeyLevel(setup.entry, candles, '#00d4ff', '', 2);

                // Stop level (red dashed)
                lineSeries.push(addPriceLine(setup.stop, '#ff4444', slLabel, 2, 1));
                const slData = candles.map(c => ({ time: c.time, value: setup.stop }));
                const slSeries = chart.addLineSeries({
                    color: 'rgba(255, 68, 68, 0.8)',
                    lineWidth: 1,
                    lineStyle: LightweightCharts.LineStyle.Dashed,
                    crosshairMarkerVisible: false,
                    priceLineVisible: false,
                    lastValueVisible: false,
                    priceScaleId: 'right',
                });
                slSeries.setData(slData);
                additionalSeries.push(slSeries);

                // Target level (green, prominent)
                lineSeries.push(addPriceLine(setup.target, '#00ff88', tpLabel, 0, 2));

                // Target zone (shaded blue/purple area around target)
                const targetRange = Math.abs(setup.target - setup.entry) * 0.05;
                drawZoneBox(setup.target + targetRange, setup.target - targetRange, candles, 'target', Math.floor(candles.length * 0.5));

                // SL Trail line
                if (setup.slTrail) {
                    drawSLTrail(setup.slTrail, candles, Math.floor(candles.length * 0.6), 'SL Trail');
                }

                // Add trade markers
                addTradeMarkers(candles, setup.entry, setup.stop, setup.target, setup.direction);
            }
        }

        // ===== PO3 SCHEMATICS FUNCTIONS (TCT Lecture 8) =====

        function buildPO3CardHTML(p, tf) {
            const isBull = p.direction === 'bullish';
            const dirClass = isBull ? 'bullish' : 'bearish';
            const typeLabel = isBull ? 'BULLISH PO3' : 'BEARISH PO3';
            const quality = Math.round((p.quality_score || 0) * 100);
            const phase = p.phase || 'range';
            const entry = p.entry?.price;
            const stop = p.stop_loss?.price;
            const target = p.target?.price;
            const rr = p.risk_reward;

            const chartUrl = '/po3-chart?symbol=' + currentSymbol + '&timeframe=' + tf;

            let html = '<a href="' + chartUrl + '" target="_blank" rel="noopener" class="schematic-link" title="Open PO3 chart in new tab">';
            html += '<div class="po3-item ' + dirClass + '">';

            // Header: type + phase + quality
            html += '<div class="po3-header">';
            html += '<span class="po3-type">' + typeLabel + '</span>';
            html += '<span class="po3-phase ' + phase.replace(' ', '_') + '">' + phase.replace('_', ' ') + '</span>';
            html += '<span class="po3-quality">' + quality + '%</span>';
            html += '</div>';

            // Entry/Stop/Target levels
            if (entry && stop && target) {
                html += '<div class="po3-levels">';
                html += '<div class="po3-level-box entry"><span class="po3-level-label">ENTRY</span><span class="po3-level-price">' + fmtPrice(entry) + '</span></div>';
                html += '<div class="po3-level-box stop"><span class="po3-level-label">STOP</span><span class="po3-level-price">' + fmtPrice(stop) + '</span></div>';
                html += '<div class="po3-level-box target"><span class="po3-level-label">TARGET</span><span class="po3-level-price">' + fmtPrice(target) + '</span></div>';
                html += '</div>';
            }

            // Manipulation depth bar
            const manipInfo = p.manipulation || {};
            const devPct = manipInfo.deviation_pct || 0;
            const dl2Pct = 30;
            const barWidth = Math.min(100, (devPct / dl2Pct) * 100);

            html += '<div class="po3-manip-bar">';
            html += '<div class="bar-track">';
            html += '<div class="bar-fill ' + dirClass + '" style="width:' + barWidth + '%"></div>';
            html += '</div>';
            html += '<div class="po3-manip-labels">';
            html += '<span>Dev: ' + devPct.toFixed(1) + '%</span>';
            html += '<span>DL2: ' + dl2Pct + '%</span>';
            html += '</div>';
            html += '</div>';

            // Meta tags
            html += '<div class="po3-meta">';
            if (rr) html += '<span class="rr">R:R ' + rr.toFixed(1) + '</span>';
            if (p.tct_model?.detected) html += '<span class="tct-model">TCT ' + (p.tct_model.type || 'Model') + '</span>';
            if (p.exception) {
                const excLabel = p.exception === 'exception_1_two_tap' ? '2-Tap' : p.exception === 'exception_2_internal_tct' ? 'Internal TCT' : p.exception;
                html += '<span class="exception">' + excLabel + '</span>';
            }
            if (p.has_compression) html += '<span class="compression">Compressed</span>';
            if (p.has_liquidity_both_sides) html += '<span class="liq-both">Dual Liq</span>';
            if (p.has_expansion) html += '<span style="color:#00ff88;">Expanding</span>';
            html += '</div>';

            // Range info
            if (p.range) {
                html += '<div style="margin-top:4px;font-size:0.6rem;color:#555;">';
                html += 'Range: ' + fmtPrice(p.range.low || 0);
                html += ' &mdash; ' + fmtPrice(p.range.high || 0);
                html += ' (' + (p.range.size_pct || 0) + '%)';
                html += '</div>';
            }

            html += '<div class="view-chart-hint">View PO3 chart &rarr;</div>';
            html += '</div></a>';
            return html;
        }

        function updatePO3UI(data) {
            const contentEl = document.getElementById('po3Content');
            const badgeEl = document.getElementById('po3Badge');

            // Reset scan button state
            const scanBtn = document.getElementById('scanPO3Btn');
            if (scanBtn) { scanBtn.disabled = false; scanBtn.classList.remove('scanning'); document.getElementById('scanPO3Icon').textContent = '\\u{1F50D}'; }

            if (data.error) {
                contentEl.innerHTML = '<div class="metric-row"><span class="label">Error loading PO3</span></div>';
                badgeEl.textContent = 'ERR';
                badgeEl.className = 'badge badge-bearish';
                return;
            }

            // Get PO3 for all 3 timeframes (use timeframes from response)
            const htfPO3TF = data.htf_po3?.timeframe || 'HTF';
            const mtfPO3TF = data.mtf_po3?.timeframe || 'MTF';
            const ltfPO3TF = data.ltf_po3?.timeframe || 'LTF';
            const htfPO3 = data.htf_po3?.schematics || [];
            const mtfPO3 = data.mtf_po3?.schematics || [];
            const ltfPO3 = data.ltf_po3?.schematics || [];
            const allPO3 = [...htfPO3, ...mtfPO3, ...ltfPO3];

            if (allPO3.length === 0) {
                contentEl.innerHTML = '<div class="metric-row"><span class="label">No active PO3 schematics detected</span></div>' +
                    '<div class="metric-row" style="margin-top:4px;"><span class="label" style="font-size:0.6rem;color:#666;">PO3 = Range &rarr; Manipulation &rarr; Expansion</span></div>' +
                    '<div class="metric-row"><span class="label" style="font-size:0.6rem;color:#666;">Scanned HTF (' + htfPO3TF + '), MTF (' + mtfPO3TF + '), LTF (' + ltfPO3TF + ')</span></div>';
                badgeEl.textContent = '0';
                badgeEl.className = 'badge badge-neutral';
                return;
            }

            // Update badge
            const totalCount = allPO3.length;
            const hasBull = allPO3.some(p => p.direction === 'bullish');
            const hasBear = allPO3.some(p => p.direction === 'bearish');
            badgeEl.textContent = totalCount;
            badgeEl.className = 'badge badge-' + (hasBull && !hasBear ? 'bullish' : hasBear && !hasBull ? 'bearish' : 'neutral');

            // Build grouped HTML using actual timeframes from response
            let html = '';
            html += buildTFGroupHTML('HTF', htfPO3TF, htfPO3, buildPO3CardHTML);
            html += buildTFGroupHTML('MTF', mtfPO3TF, mtfPO3, buildPO3CardHTML);
            html += buildTFGroupHTML('LTF', ltfPO3TF, ltfPO3, buildPO3CardHTML);

            contentEl.innerHTML = html;
        }

        // Scan PO3 schematics on demand
        async function scanPO3Schematics() {
            const btn = document.getElementById('scanPO3Btn');
            const icon = document.getElementById('scanPO3Icon');
            btn.disabled = true;
            btn.classList.add('scanning');
            icon.textContent = '\\u23F3';

            const contentEl = document.getElementById('po3Content');
            contentEl.innerHTML = '<div class="metric-row"><span class="label" style="color:#ffc107;">Scanning PO3 for ' + currentSymbol + ' (' + currentTimeframe + ')...</span></div>';

            try {
                const data = await fetchWithRetry('/api/po3?symbol=' + currentSymbol + '&timeframe=' + currentTimeframe, {}, 3, 30000);
                updatePO3UI(data);
            } catch (e) {
                contentEl.innerHTML = '<div class="metric-row"><span class="label" style="color:#ff4444;">Scan failed: ' + e.message + '</span></div>';
                btn.disabled = false;
                btn.classList.remove('scanning');
                icon.textContent = '\\u{1F50D}';
            }
        }

        // ===== RISK MANAGEMENT FUNCTIONS (TCT Lecture 7) =====

        function switchRiskTab(tabName) {
            document.querySelectorAll('.risk-tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.risk-tab-content').forEach(c => c.classList.remove('active'));
            document.querySelector(`.risk-tab[data-tab="${tabName}"]`).classList.add('active');
            document.getElementById('tab-' + tabName).classList.add('active');
        }

        function toggleGoldPrice() {
            const market = document.getElementById('riskMarket').value;
            document.getElementById('goldPriceGroup').style.display = market === 'gold' ? 'flex' : 'none';
        }

        function fmt(n) {
            return '$' + n.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2});
        }

        function calculateRisk() {
            const balance = parseFloat(document.getElementById('riskBalance').value) || 10000;
            const riskPct = parseFloat(document.getElementById('riskPct').value) || 1;
            const slPct = parseFloat(document.getElementById('riskSL').value) || 0.26;
            const rr = parseFloat(document.getElementById('riskRR').value) || 3;
            const market = document.getElementById('riskMarket').value;
            const leverage = parseFloat(document.getElementById('riskLeverage').value) || 10;
            const goldPrice = parseFloat(document.getElementById('goldPrice').value) || 2000;

            if (slPct <= 0) return;

            // Core TCT Lecture 7 formula: Position Size = (Risk $ / SL%) x 100
            const riskAmount = balance * (riskPct / 100);
            const positionSize = (riskAmount / slPct) * 100;

            // Leverage
            const minLeverage = positionSize / balance;
            const usedMargin = positionSize / leverage;
            const freeMargin = balance - usedMargin;

            // Trade outcome
            const lossAtSL = riskAmount;
            const profitAtTP = riskAmount * rr;
            const gainPct = (profitAtTP / balance) * 100;

            // Update UI
            document.getElementById('resRiskAmt').textContent = '-' + fmt(riskAmount);
            document.getElementById('resPosSize').textContent = fmt(positionSize);
            document.getElementById('resMinLev').textContent = minLeverage.toFixed(2) + 'x';
            document.getElementById('resMargin').textContent = fmt(usedMargin);
            document.getElementById('resFreeMargin').textContent = fmt(freeMargin);
            document.getElementById('resLoss').textContent = '-' + fmt(lossAtSL);
            document.getElementById('resProfit').textContent = '+' + fmt(profitAtTP);
            document.getElementById('resGainPct').textContent = '+' + gainPct.toFixed(2) + '%';

            // Lots display for forex/gold
            const lotsRow = document.getElementById('resLotsRow');
            if (market === 'forex') {
                lotsRow.style.display = 'flex';
                document.getElementById('resLots').textContent = (positionSize / 100000).toFixed(4) + ' lots';
            } else if (market === 'gold') {
                lotsRow.style.display = 'flex';
                const lotVal = goldPrice * 100;
                document.getElementById('resLots').textContent = (positionSize / lotVal).toFixed(4) + ' lots';
            } else {
                lotsRow.style.display = 'none';
            }

            // Streak simulation (6 losses then 3 wins)
            const lossMult = 1 - (riskPct / 100);
            const winMult = 1 + (riskPct * rr / 100);
            let streakBal = balance;
            const trades = [];

            for (let i = 0; i < 6; i++) {
                streakBal *= lossMult;
                trades.push({balance: streakBal, result: 'loss'});
            }
            for (let i = 0; i < 3; i++) {
                streakBal *= winMult;
                trades.push({balance: streakBal, result: 'win'});
            }

            // Draw streak bars
            const barsEl = document.getElementById('streakBars');
            const minBal = Math.min(...trades.map(t => t.balance));
            const maxBal = Math.max(balance, ...trades.map(t => t.balance));
            const range = maxBal - minBal;

            barsEl.innerHTML = '';
            trades.forEach((t, i) => {
                const pct = range > 0 ? ((t.balance - minBal) / range) * 100 : 50;
                const bar = document.createElement('div');
                bar.className = 'streak-bar ' + (t.result === 'loss' ? 'loss-bar' : 'win-bar');
                bar.style.height = Math.max(8, pct) + '%';
                bar.innerHTML = '<span class="bar-label">T' + (i+1) + '</span>';
                barsEl.appendChild(bar);
            });

            const netPct = ((streakBal - balance) / balance) * 100;
            const resultEl = document.getElementById('streakResult');
            resultEl.className = 'streak-result ' + (netPct >= 0 ? 'positive' : 'negative');
            resultEl.textContent = 'After 6L + 3W: ' + fmt(streakBal) + ' (' + (netPct >= 0 ? '+' : '') + netPct.toFixed(2) + '%)';

            document.getElementById('riskResults').classList.add('active');
        }

        function simulateEquity() {
            const balance = parseFloat(document.getElementById('eqBalance').value) || 10000;
            const numTrades = parseInt(document.getElementById('eqTrades').value) || 50;
            const riskA = parseFloat(document.getElementById('eqRiskA').value) || 1;
            const riskB = parseFloat(document.getElementById('eqRiskB').value) || 10;
            const winRate = 0.70; // TCT 70% win rate
            const avgRR = 2.3;   // TCT average R:R

            // Simulate Trader A (disciplined, 1% risk)
            let balA = balance;
            const curveA = [balA];
            // Simulate Trader B (over-risking, 10% risk)
            let balB = balance;
            const curveB = [balB];

            // Use seeded pseudo-random for consistency
            let seed = 42;
            function seededRandom() {
                seed = (seed * 16807) % 2147483647;
                return (seed - 1) / 2147483646;
            }

            for (let i = 0; i < numTrades; i++) {
                const rand = seededRandom();
                const isWin = rand < winRate;

                if (isWin) {
                    balA *= (1 + (riskA * avgRR / 100));
                    balB *= (1 + (riskB * avgRR / 100));
                } else {
                    balA *= (1 - riskA / 100);
                    balB *= (1 - riskB / 100);
                }
                // Trader B occasionally revenge trades (extra loss)
                if (!isWin && seededRandom() < 0.4) {
                    balB *= (1 - riskB * 1.5 / 100);
                }
                curveA.push(balA);
                curveB.push(Math.max(0, balB));
            }

            // Draw equity chart on canvas
            const container = document.getElementById('equityChartContainer');
            container.style.display = 'block';
            const canvas = document.getElementById('equityCanvas');
            const ctx = canvas.getContext('2d');

            // Set actual pixel dimensions
            canvas.width = canvas.clientWidth * 2;
            canvas.height = canvas.clientHeight * 2;
            ctx.scale(2, 2);

            const w = canvas.clientWidth;
            const h = canvas.clientHeight;
            const padding = {top: 10, right: 10, bottom: 20, left: 50};

            ctx.clearRect(0, 0, w, h);

            // Find min/max across both curves
            const allVals = [...curveA, ...curveB];
            const minVal = Math.min(...allVals) * 0.95;
            const maxVal = Math.max(...allVals) * 1.05;

            const chartW = w - padding.left - padding.right;
            const chartH = h - padding.top - padding.bottom;

            function toX(i) { return padding.left + (i / numTrades) * chartW; }
            function toY(v) { return padding.top + chartH - ((v - minVal) / (maxVal - minVal)) * chartH; }

            // Grid lines
            ctx.strokeStyle = '#1e1e2d';
            ctx.lineWidth = 0.5;
            for (let i = 0; i <= 4; i++) {
                const y = padding.top + (chartH / 4) * i;
                ctx.beginPath();
                ctx.moveTo(padding.left, y);
                ctx.lineTo(w - padding.right, y);
                ctx.stroke();

                const val = maxVal - ((maxVal - minVal) / 4) * i;
                ctx.fillStyle = '#666';
                ctx.font = '9px sans-serif';
                ctx.textAlign = 'right';
                ctx.fillText('$' + Math.round(val).toLocaleString(), padding.left - 4, y + 3);
            }

            // Starting balance line
            ctx.strokeStyle = '#2d2d44';
            ctx.lineWidth = 1;
            ctx.setLineDash([4, 4]);
            ctx.beginPath();
            ctx.moveTo(padding.left, toY(balance));
            ctx.lineTo(w - padding.right, toY(balance));
            ctx.stroke();
            ctx.setLineDash([]);

            // Draw Trader B first (behind)
            ctx.strokeStyle = '#ff4444';
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            curveB.forEach((v, i) => {
                if (i === 0) ctx.moveTo(toX(i), toY(v));
                else ctx.lineTo(toX(i), toY(v));
            });
            ctx.stroke();

            // Draw Trader A (on top)
            ctx.strokeStyle = '#00ff88';
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            curveA.forEach((v, i) => {
                if (i === 0) ctx.moveTo(toX(i), toY(v));
                else ctx.lineTo(toX(i), toY(v));
            });
            ctx.stroke();

            // Update final values
            document.getElementById('eqFinalA').textContent = fmt(balA);
            document.getElementById('eqFinalB').textContent = fmt(Math.max(0, balB));
        }

        function calculateCompounding() {
            const balance = parseFloat(document.getElementById('compBalance').value) || 10000;
            const weeklyPct = parseFloat(document.getElementById('compWeekly').value) || 5;
            const weeksPerYear = parseInt(document.getElementById('compWeeks').value) || 35;
            const years = parseInt(document.getElementById('compYears').value) || 3;

            const rate = weeklyPct / 100;
            const body = document.getElementById('compoundBody');
            body.innerHTML = '';

            // Add starting row
            let row = document.createElement('tr');
            row.innerHTML = '<td>Start</td><td>' + fmt(balance) + '</td><td>--</td>';
            body.appendChild(row);

            let bal = balance;
            for (let y = 1; y <= years; y++) {
                const prevBal = bal;
                bal = bal * Math.pow(1 + rate, weeksPerYear);
                const growth = ((bal - prevBal) / prevBal) * 100;

                row = document.createElement('tr');
                row.innerHTML = '<td>Year ' + y + '</td><td class="compound-growth">' + fmt(bal) + '</td><td class="compound-growth">+' + growth.toFixed(0) + '%</td>';
                body.appendChild(row);
            }

            document.getElementById('compoundResults').style.display = 'block';
        }

        // ========== Trade Execution (Lecture 9) Functions ==========
        function switchExecTab(tabName) {
            document.querySelectorAll('.exec-tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.exec-tab-content').forEach(c => c.classList.remove('active'));
            document.querySelector(`.exec-tab[data-tab="${tabName}"]`).classList.add('active');
            document.getElementById('tab-' + tabName).classList.add('active');
        }

        async function generateExecutionPlan() {
            const balance = parseFloat(document.getElementById('execBalance').value) || 10000;
            const riskPct = parseFloat(document.getElementById('execRiskPct').value) || 1;
            const entry = parseFloat(document.getElementById('execEntry').value) || 100000;
            const sl = parseFloat(document.getElementById('execSL').value) || 99500;
            const tp = parseFloat(document.getElementById('execTP').value) || 101500;
            const tp2 = parseFloat(document.getElementById('execTP2').value) || 0;
            const direction = document.getElementById('execDirection').value;
            const leverage = parseFloat(document.getElementById('execLeverage').value) || 10;

            let url = `/api/trade-execution?account_balance=${balance}&risk_pct=${riskPct}&entry_price=${entry}&stop_loss_price=${sl}&take_profit_price=${tp}&direction=${direction}&leverage=${leverage}`;
            if (tp2 > 0) url += `&tp2_price=${tp2}`;

            try {
                const data = await fetchWithRetry(url, {}, 2, 15000);
                if (data.error) {
                    alert('Error: ' + data.error);
                    return;
                }
                updateExecutionUI(data);
            } catch (e) {
                alert('Execution plan error: ' + e.message);
            }
        }

        function updateExecutionUI(data) {
            const ps = data.position_sizing || {};
            const la = data.leverage_analysis || {};
            const to = data.trade_outcome || {};
            const pt = data.partial_take_profits || {};
            const safety = la.safety || {};

            // Direction
            document.getElementById('execResDirection').textContent = (ps.entry_price ? data.execution_plan?.direction || '--' : '--');

            // Position sizing
            document.getElementById('execResRisk').textContent = '-' + fmt(ps.risk_amount || 0);
            document.getElementById('execResSLPct').textContent = (ps.stop_loss_pct || 0).toFixed(4) + '%';
            document.getElementById('execResPos').textContent = fmt(ps.position_size || 0);

            // Leverage safety
            document.getElementById('execResLev').textContent = (la.selected_leverage || 0) + 'x';
            document.getElementById('execResMargin').textContent = fmt(la.margin_required || 0);
            document.getElementById('execResLiq').textContent = fmt(la.liquidation_price || 0);

            const gapEl = document.getElementById('execResGap');
            gapEl.textContent = fmt(Math.abs(safety.gap || 0)) + ' (' + (safety.gap_pct || 0).toFixed(4) + '%)';

            const safetyEl = document.getElementById('execResSafety');
            if (safety.is_safe) {
                safetyEl.textContent = 'SAFE';
                safetyEl.className = 'r-value safe';
            } else {
                safetyEl.textContent = 'DANGER';
                safetyEl.className = 'r-value danger';
            }

            document.getElementById('execResMaxLev').textContent = (la.max_safe_leverage || 0) + 'x';

            // Trade outcome
            document.getElementById('execResRR').textContent = '1 : ' + (to.risk_reward || 0);
            document.getElementById('execResLoss').textContent = '-' + fmt(to.loss_at_sl || 0);
            document.getElementById('execResProfit').textContent = '+' + fmt(to.profit_at_tp || 0);
            document.getElementById('execResGain').textContent = '+' + (to.tp_account_gain_pct || 0) + '%';

            // Partial TPs
            const tpContainer = document.getElementById('execTPLevels');
            tpContainer.innerHTML = '';
            if (pt.tp_levels) {
                pt.tp_levels.forEach((tp, i) => {
                    tpContainer.innerHTML += `
                        <div class="result-row">
                            <span class="r-label">TP${i+1} @ ${fmt(tp.price)} (${tp.close_pct}%)</span>
                            <span class="r-value profit">+${fmt(tp.profit)}</span>
                        </div>`;
                });
            }
            document.getElementById('execResTotalTP').textContent = '+' + fmt(pt.total_profit || 0);

            // Checklist
            const checkEl = document.getElementById('execChecklist');
            checkEl.innerHTML = '';
            if (data.execution_checklist) {
                data.execution_checklist.forEach(item => {
                    if (item) {
                        const li = document.createElement('li');
                        li.textContent = item;
                        checkEl.appendChild(li);
                    }
                });
            }

            // Leverage comparison table
            const levBody = document.getElementById('levCompareBody');
            levBody.innerHTML = '';
            if (la.comparison) {
                la.comparison.forEach(c => {
                    const row = document.createElement('tr');
                    row.className = c.is_safe ? 'safe-row' : 'danger-row';
                    row.innerHTML = `<td>${c.leverage}x</td><td>${fmt(c.margin)}</td><td>${fmt(c.liquidation_price)}</td><td>${fmt(Math.abs(c.gap_to_sl))}</td><td>${c.is_safe ? 'SAFE' : 'DANGER'}</td>`;
                    levBody.appendChild(row);
                });
            }

            document.getElementById('execResults').classList.add('active');
        }

        function calculateCapital() {
            const capital = parseFloat(document.getElementById('execCapital').value) || 50000;
            const pct = parseFloat(document.getElementById('execExchangePct').value) || 50;

            const onExchange = capital * (pct / 100);
            const offExchange = capital - onExchange;

            document.getElementById('capTotal').textContent = fmt(capital);
            document.getElementById('capExchange').textContent = fmt(onExchange);
            document.getElementById('capOff').textContent = fmt(offExchange);
            document.getElementById('capRec').textContent = pct <= 50
                ? 'Good — conservative allocation'
                : 'Caution — consider reducing to 30-50%';
            document.getElementById('capRec').className = 'r-value ' + (pct <= 50 ? 'safe' : 'danger');
            document.getElementById('capitalResults').classList.add('active');
        }

        // ===== TOP 5 SETUPS PANEL =====

        async function fetchTop5Setups() {
            try {
                const data = await fetchWithRetry('/api/top-setups', {}, 2, 15000);
                if (data) {
                    renderTop5(data.top_setups || [], data.scanner_status || {});
                }
            } catch (e) {
                console.error('Failed to fetch top 5 setups:', e);
            }
        }

        function renderTop5(setups, status) {
            // Update scanner status badge
            const statusEl = document.getElementById('scannerStatus');
            if (status.is_scanning) {
                statusEl.textContent = 'Scanning ' + (status.pairs_scanned || 0) + '/' + (status.total_pairs || 0) + '...';
                statusEl.className = 'top5-scanner-status scanning';
            } else if (status.last_scan) {
                const ago = Math.round((Date.now() - new Date(status.last_scan + 'Z').getTime()) / 60000);
                const agoText = ago < 60 ? ago + 'm ago' : Math.round(ago / 60) + 'h ago';
                statusEl.textContent = agoText + ' | ' + (status.scan_duration_sec || 0) + 's';
                statusEl.className = 'top5-scanner-status';
            }

            const contentEl = document.getElementById('top5Content');

            if (!setups || setups.length === 0) {
                if (status.is_scanning) {
                    contentEl.innerHTML = '<div class="top5-empty">Scanning ' + (status.total_pairs || 0) + ' pairs for ranges (detect_ranges)...</div>';
                } else {
                    contentEl.innerHTML = '<div class="top5-empty">Waiting for first range scan to complete...</div>';
                }
                return;
            }

            let html = '';
            setups.forEach((s, i) => {
                const basePair = s.symbol.replace('USDT', '');

                // Format prices compactly
                const fmt = (p) => {
                    if (p == null || isNaN(p)) return '--';
                    if (p >= 10000) return '$' + p.toLocaleString(undefined, {maximumFractionDigits: 0});
                    if (p >= 100)   return '$' + p.toLocaleString(undefined, {maximumFractionDigits: 2});
                    if (p >= 1)     return '$' + p.toFixed(2);
                    if (p >= 0.01)  return '$' + p.toFixed(4);
                    return '$' + p.toPrecision(4);
                };

                // Quality color mapping (matches /ranges page)
                const qualityColor = s.quality === 'good' ? '#00ff88' : s.quality === 'moderate' ? '#ffc107' : '#dc3545';
                const qualityBg = s.quality === 'good' ? 'rgba(0,255,136,0.15)' : s.quality === 'moderate' ? 'rgba(255,193,7,0.15)' : 'rgba(220,53,69,0.15)';

                // Price position relative to range
                const priceInRange = s.current_price >= s.range_low && s.current_price <= s.range_high;
                const priceZone = s.current_price > s.range_eq ? 'Premium' : 'Discount';
                const priceZoneColor = s.current_price > s.range_eq ? '#dc3545' : '#00ff88';

                html += '<div class="top5-item qualified" data-symbol="' + s.symbol + '" data-tf="1d">';
                html += '<div class="top5-header">';
                html += '<span><span class="top5-rank">#' + (i + 1) + '</span><span class="top5-pair">' + basePair + '</span></span>';
                html += '<span style="display:flex;gap:4px;align-items:center;">';
                html += '<span class="top5-tf" style="color:' + qualityColor + ';background:' + qualityBg + ';">' + (s.quality || 'good') + '</span>';
                html += '<span class="top5-tf">' + (s.timeframe || '1d').toUpperCase() + '</span>';
                html += '</span>';
                html += '</div>';

                // Score line with EQ touched indicator
                html += '<div class="top5-rps">Score: <span class="rps-value">' + s.RPS + '</span> / 10';
                if (s.eq_touched) {
                    html += ' <span style="color:#ffc107;font-size:0.6rem;" title="Equilibrium touched — range confirmed per TCT rules">EQ &#10003;</span>';
                }
                html += '</div>';

                // Range levels (High / EQ / Low)
                html += '<div class="top5-levels">';
                html += '<span>High: ' + fmt(s.range_high) + '</span>';
                html += '<span class="eq-highlight">EQ: ' + fmt(s.range_eq) + '</span>';
                html += '<span>Low: ' + fmt(s.range_low) + '</span>';
                html += '</div>';

                // DL levels + current price
                html += '<div class="top5-levels" style="margin-top:2px;">';
                html += '<span style="color:#6c757d;">DL+: ' + fmt(s.upper_dl) + '</span>';
                html += '<span style="color:' + priceZoneColor + ';">Price: ' + fmt(s.current_price) + '</span>';
                html += '<span style="color:#6c757d;">DL-: ' + fmt(s.lower_dl) + '</span>';
                html += '</div>';

                // Tags
                html += '<div class="top5-tags">';
                html += '<span class="top5-tag duration">' + (s.candles || 0) + ' candles</span>';
                if (s.trend_before && s.trend_before !== 'neutral') {
                    const trendColor = s.trend_before === 'up' ? '#00ff88' : '#dc3545';
                    html += '<span class="top5-tag" style="color:' + trendColor + ';background:rgba(255,255,255,0.05);">Pre: ' + s.trend_before + '</span>';
                }
                if (priceInRange) {
                    html += '<span class="top5-tag" style="color:' + priceZoneColor + ';background:rgba(255,255,255,0.05);">' + priceZone + '</span>';
                } else {
                    html += '<span class="top5-tag" style="color:#dc3545;background:rgba(220,53,69,0.1);">Outside</span>';
                }
                html += '<span class="top5-tag" style="color:#e040fb;background:rgba(224,64,251,0.1);">' + s.RPS + '/10</span>';
                html += '</div>';

                html += '</div>';
            });

            contentEl.innerHTML = html;

            // Click handler — switch pair + timeframe to view the setup
            contentEl.querySelectorAll('.top5-item').forEach(item => {
                item.addEventListener('click', async () => {
                    const sym = item.dataset.symbol;
                    const tf = item.dataset.tf;

                    // Update pair selector
                    currentSymbol = sym;
                    document.getElementById('pairSearch').value = sym;
                    document.getElementById('headerSymbol').textContent = sym;

                    // Update timeframe dropdown
                    const tfSelect = document.getElementById('tfDropdown');
                    if (tfSelect.querySelector('option[value="' + tf + '"]')) {
                        tfSelect.value = tf;
                        currentTimeframe = tf;
                    }

                    await refreshData();
                });
            });
        }

        // (Forming models are now derived from current pair's schematic data in deriveFormingModels())

        // Initialize
        initChart();
        refreshData();
        fetchTop5Setups();

        // Auto-refresh every 30 seconds (forming models derived within refreshData)
        setInterval(refreshData, 30000);

        // Refresh top 5 every 60 seconds (lightweight — reads cached results)
        setInterval(fetchTop5Setups, 60000);
    </script>
</body>
</html>
    """
    return HTMLResponse(content=html_content)

@app.get("/api/zones")
async def detect_zones(symbol: Optional[str] = None):
    """
    Detect and score Supply/Demand zones using TCT Mentorship methodology.

    Returns:
        - Order Blocks (with FVG requirement)
        - Structure Supply/Demand zones
        - Location-based scoring (pivot, range discount/premium, deviation)
        - Top zones sorted by strength
    """
    try:
        sym = resolve_symbol(symbol)
        # Fetch candles
        htf_df = await fetch_mexc_candles(sym, "4h", 100)
        ltf_df = await fetch_mexc_candles(sym, "15m", 200)

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
            "symbol": sym,
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
async def detect_ranges(symbol: Optional[str] = None):
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
        sym = resolve_symbol(symbol)
        # Fetch candles
        htf_df = await fetch_mexc_candles(sym, "4h", 100)
        ltf_df = await fetch_mexc_candles(sym, "15m", 200)

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
            "symbol": sym,
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
                "ltf_trend": ltf_pivots.get("trend", "neutral"),
                "htf_eof": htf_pivots.get("eof", {}),
                "ltf_eof": ltf_pivots.get("eof", {}),
                "htf_bos_events": htf_pivots.get("bos_events", []),
                "ltf_bos_events": ltf_pivots.get("bos_events", []),
                "htf_ms_highs": htf_pivots.get("ms_highs", []),
                "htf_ms_lows": htf_pivots.get("ms_lows", []),
                "ltf_ms_highs": ltf_pivots.get("ms_highs", []),
                "ltf_ms_lows": ltf_pivots.get("ms_lows", []),
                "htf_levels": htf_pivots.get("levels", {}),
                "ltf_levels": ltf_pivots.get("levels", {}),
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
                "total_nested_ltf_ranges": nested_ranges.get("total_nested_ltf_ranges", 0),
                "htf_eof_bias": htf_pivots.get("eof", {}).get("bias", "neutral"),
                "ltf_eof_bias": ltf_pivots.get("eof", {}).get("bias", "neutral"),
                "htf_trend_shift": htf_pivots.get("eof", {}).get("trend_shift", False),
                "ltf_trend_shift": ltf_pivots.get("eof", {}).get("trend_shift", False),
            },
            "tct_concepts": {
                "six_candle_rule": "2 consecutive bullish + 2 consecutive bearish candles (inside bars excluded)",
                "market_structure_high": "Highest point between two consecutive lows, confirmed when 2nd low is touched",
                "market_structure_low": "Lowest point between two consecutive highs, confirmed when 2nd high is touched",
                "bos": "Break of Structure = candle CLOSE above MS high (bullish) or below MS low (bearish)",
                "level_1": "Primary trend direction - most important structure pool",
                "level_2": "Always opposite direction of Level 1",
                "level_3": "Refined structure of most recent Level 2 expansion - domino effect entry",
                "eof": "Expectational Order Flow: Bullish trend → expect HL for HH; opposite BOS = trend shift",
                "deviation_limit": "30% of range size - threshold for deviation vs break",
                "premium_zone": "Above equilibrium (0.5) to range high",
                "discount_zone": "Below equilibrium (0.5) to range low",
            }
        })
        return JSONResponse(response_data)

    except Exception as e:
        logger.error(f"[RANGES_ERROR] {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/api/liquidity")
async def detect_liquidity(symbol: Optional[str] = None):
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
        sym = resolve_symbol(symbol)
        # Fetch candles
        htf_df = await fetch_mexc_candles(sym, "4h", 100)
        ltf_df = await fetch_mexc_candles(sym, "15m", 200)

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
            "symbol": sym,
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
async def validate_current_setup(symbol: Optional[str] = None):
    """Complete 7-gate validation endpoint"""
    try:
        sym = resolve_symbol(symbol)
        htf_df = await fetch_mexc_candles(sym, "4h", 100)
        ltf_df = await fetch_mexc_candles(sym, "15m", 100)

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
            "symbol": sym
        }

        result = validate_gates(context)
        result.pop("htf_candles", None)
        result.pop("ltf_candles", None)

        return JSONResponse(result)
    except Exception as e:
        logger.error(f"[VALIDATE_ERROR] {e}")
        return JSONResponse({"error": str(e), "Action": "NO_TRADE"}, status_code=500)

@app.get("/api/schematics")
async def get_tct_schematics(symbol: Optional[str] = None, timeframe: Optional[str] = None):
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
    Accepts optional timeframe parameter to scan relative HTF/MTF/LTF windows.
    """
    try:
        sym = resolve_symbol(symbol)

        # Map selected timeframe to appropriate HTF/MTF/LTF scanning windows
        tf_map = {
            "1m":  {"htf": "15m", "mtf": "5m",  "ltf": "1m"},
            "5m":  {"htf": "1h",  "mtf": "15m", "ltf": "5m"},
            "15m": {"htf": "4h",  "mtf": "1h",  "ltf": "15m"},
            "30m": {"htf": "4h",  "mtf": "1h",  "ltf": "30m"},
            "1h":  {"htf": "4h",  "mtf": "1h",  "ltf": "15m"},
            "2h":  {"htf": "1d",  "mtf": "4h",  "ltf": "1h"},
            "4h":  {"htf": "1d",  "mtf": "4h",  "ltf": "1h"},
            "8h":  {"htf": "1d",  "mtf": "8h",  "ltf": "4h"},
            "1d":  {"htf": "1W",  "mtf": "1d",  "ltf": "4h"},
            "1W":  {"htf": "1M",  "mtf": "1W",  "ltf": "1d"},
            "1M":  {"htf": "1M",  "mtf": "1W",  "ltf": "1d"},
        }
        tf_windows = tf_map.get(timeframe, {"htf": "4h", "mtf": "1h", "ltf": "15m"})

        # Multi-timeframe scanning based on selected timeframe context
        timeframes = {"htf": tf_windows["htf"], "mtf": tf_windows["mtf"], "ltf": tf_windows["ltf"]}
        dfs = {}
        for tf_key, tf_val in timeframes.items():
            df = await fetch_mexc_candles(sym, tf_val, 200)
            if df is not None and len(df) >= 30:
                dfs[tf_key] = df

        if not dfs:
            return JSONResponse({"error": "Failed to fetch candle data"}, status_code=500)

        # Use finest available timeframe for current price (can't use `or` on DataFrames)
        price_df = next((dfs[k] for k in ("ltf", "mtf", "htf") if k in dfs and not dfs[k].empty), None)
        if price_df is None or price_df.empty:
            return JSONResponse({"error": "No valid candle data available"}, status_code=500)
        current_price = float(price_df.iloc[-1]["close"])

        # Detect ranges for each timeframe (convert to list for range detection)
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

        # Filter and sort schematics by quality
        def filter_active_schematics(schematics, current_price):
            """Filter to schematics that are still valid for trading"""
            active = []
            for s in schematics:
                if not isinstance(s, dict):
                    continue
                entry = s.get('entry', {}).get('price')
                target = s.get('target', {}).get('price')
                stop = s.get('stop_loss', {}).get('price')

                if entry and target and stop:
                    if s.get('direction') == 'bullish':
                        if current_price < target and current_price > stop:
                            active.append(s)
                    elif s.get('direction') == 'bearish':
                        if current_price > target and current_price < stop:
                            active.append(s)
                else:
                    active.append(s)
            return sorted(active, key=lambda x: x.get('quality_score', 0), reverse=True)

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

        # Scan each timeframe for schematics
        tf_results = {}
        for tf_key, tf_val in timeframes.items():
            if tf_key not in dfs:
                tf_results[tf_key] = {"timeframe": tf_val, "schematics": [], "summary": summarize_schematics([])}
                continue

            df = dfs[tf_key]
            candles_list = df_to_candles(df)
            detected_range = await detect_best_range(candles_list)
            range_list = [detected_range] if detected_range and not isinstance(detected_range, list) else (detected_range or [])

            schematics_result = detect_tct_schematics(df, range_list)
            all_schematics = (
                schematics_result.get("accumulation_schematics", []) +
                schematics_result.get("distribution_schematics", [])
            )

            # Tag each schematic with its timeframe
            for s in all_schematics:
                if isinstance(s, dict):
                    s["timeframe"] = tf_val

            active = filter_active_schematics(all_schematics, current_price)
            tf_results[tf_key] = {
                "timeframe": tf_val,
                "schematics": active[:5],
                "summary": summarize_schematics(active)
            }

        # Convert numpy types to native Python types for JSON serialization
        response_data = convert_numpy_types({
            "symbol": sym,
            "current_price": current_price,
            "methodology": "TCT Mentorship Lecture 5A + 5B + 6 - Advanced TCT Schematics",
            "htf_schematics": tf_results.get("htf", {"timeframe": tf_windows["htf"], "schematics": [], "summary": summarize_schematics([])}),
            "mtf_schematics": tf_results.get("mtf", {"timeframe": tf_windows["mtf"], "schematics": [], "summary": summarize_schematics([])}),
            "ltf_schematics": tf_results.get("ltf", {"timeframe": tf_windows["ltf"], "schematics": [], "summary": summarize_schematics([])}),
            "trading_rules": {
                "model_1": "Two successive deviations - Tap2 below Tap1, Tap3 below Tap2 (accumulation) or above (distribution)",
                "model_2": "One deviation then higher low/lower high - must grab extreme liquidity OR mitigate extreme demand/supply",
                "entry": "Wait for BOS confirmation from lowest/highest point between Tap2 and Tap3",
                "stop_loss": "Below Tap3 for longs, above Tap3 for shorts",
                "target": "Opposite range extreme (Wyckoff high for longs, Wyckoff low for shorts)",
                "six_candle_rule": "Each tap pivot must pass 6-candle rule for valid schematic on that timeframe"
            },
            "summary": {
                "total_htf_schematics": len(tf_results.get("htf", {}).get("schematics", [])),
                "total_mtf_schematics": len(tf_results.get("mtf", {}).get("schematics", [])),
                "total_ltf_schematics": len(tf_results.get("ltf", {}).get("schematics", [])),
            }
        })
        return JSONResponse(response_data)

    except Exception as e:
        logger.error(f"[SCHEMATICS_ERROR] {e}")
        import traceback
        logger.error(traceback.format_exc())
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/api/po3")
async def get_po3_schematics(symbol: Optional[str] = None, timeframe: Optional[str] = None):
    """
    PO3 (Power of Three) Schematics endpoint — TCT Lecture 8

    Detects PO3 manipulation plays: Range → Manipulation → Expansion.
    - Bullish PO3: Range → breakdown below low → accumulation → expansion up
    - Bearish PO3: Range → breakout above high → distribution → expansion down

    Requirements:
    - Manipulation must stay inside DL2 (30% of range)
    - Must contain TCT model (accumulation/distribution) in manipulation phase
    - Good RTZ quality in the range

    Also detects Exception 1 (2-tap) and Exception 2 (Internal TCT).
    Accepts optional timeframe parameter to scan relative HTF/MTF/LTF windows.
    """
    try:
        sym = resolve_symbol(symbol)

        # Map selected timeframe to appropriate HTF/MTF/LTF scanning windows
        tf_map = {
            "1m":  {"htf": "15m", "mtf": "5m",  "ltf": "1m"},
            "5m":  {"htf": "1h",  "mtf": "15m", "ltf": "5m"},
            "15m": {"htf": "4h",  "mtf": "1h",  "ltf": "15m"},
            "30m": {"htf": "4h",  "mtf": "1h",  "ltf": "30m"},
            "1h":  {"htf": "4h",  "mtf": "1h",  "ltf": "15m"},
            "2h":  {"htf": "1d",  "mtf": "4h",  "ltf": "1h"},
            "4h":  {"htf": "1d",  "mtf": "4h",  "ltf": "1h"},
            "8h":  {"htf": "1d",  "mtf": "8h",  "ltf": "4h"},
            "1d":  {"htf": "1W",  "mtf": "1d",  "ltf": "4h"},
            "1W":  {"htf": "1M",  "mtf": "1W",  "ltf": "1d"},
            "1M":  {"htf": "1M",  "mtf": "1W",  "ltf": "1d"},
        }
        tf_windows = tf_map.get(timeframe, {"htf": "4h", "mtf": "1h", "ltf": "15m"})

        # Multi-timeframe scanning based on selected timeframe context
        timeframes = {"htf": tf_windows["htf"], "mtf": tf_windows["mtf"], "ltf": tf_windows["ltf"]}
        dfs = {}
        for tf_key, tf_val in timeframes.items():
            df = await fetch_mexc_candles(sym, tf_val, 200)
            if df is not None and len(df) >= 50:
                dfs[tf_key] = df

        if not dfs:
            return JSONResponse({"error": "Failed to fetch candle data"}, status_code=500)

        price_df = next((dfs[k] for k in ("ltf", "mtf", "htf") if k in dfs and not dfs[k].empty), None)
        if price_df is None or price_df.empty:
            return JSONResponse({"error": "No valid candle data available"}, status_code=500)
        current_price = float(price_df.iloc[-1]["close"])

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

        def filter_active_po3(po3_list, current_price):
            active = []
            for p in po3_list:
                if not isinstance(p, dict):
                    continue
                entry = p.get("entry", {}).get("price")
                target = p.get("target", {}).get("price")
                stop = p.get("stop_loss", {}).get("price")
                if entry and target and stop:
                    if p.get("direction") == "bullish":
                        if current_price < target and current_price > stop:
                            active.append(p)
                    elif p.get("direction") == "bearish":
                        if current_price > target and current_price < stop:
                            active.append(p)
                else:
                    active.append(p)
            return sorted(active, key=lambda x: x.get("quality_score", 0), reverse=True)

        def summarize_po3(po3_list):
            return {
                "total": len(po3_list),
                "bullish": sum(1 for p in po3_list if p.get("direction") == "bullish"),
                "bearish": sum(1 for p in po3_list if p.get("direction") == "bearish"),
                "in_expansion": sum(1 for p in po3_list if p.get("phase") == "expansion"),
                "in_manipulation": sum(1 for p in po3_list if p.get("phase") in ("manipulation", "manipulation_complete")),
            }

        # Scan each timeframe for PO3 schematics
        tf_results = {}
        for tf_key, tf_val in timeframes.items():
            if tf_key not in dfs:
                tf_results[tf_key] = {"timeframe": tf_val, "schematics": [], "summary": summarize_po3([])}
                continue

            df = dfs[tf_key]
            candles_list = df_to_candles(df)
            detected_range = await detect_best_range(candles_list)
            range_list = [detected_range] if detected_range and not isinstance(detected_range, list) else (detected_range or [])

            po3_result = detect_po3_schematics(df, range_list)
            all_po3 = po3_result.get("bullish_po3", []) + po3_result.get("bearish_po3", [])

            # Tag each PO3 with its timeframe
            for p in all_po3:
                if isinstance(p, dict):
                    p["timeframe"] = tf_val

            active = filter_active_po3(all_po3, current_price)
            tf_results[tf_key] = {
                "timeframe": tf_val,
                "schematics": active[:5],
                "summary": summarize_po3(active)
            }

        response_data = convert_numpy_types({
            "symbol": sym,
            "current_price": current_price,
            "methodology": "TCT Mentorship Lecture 8 - PO3 Schematics (Power of Three)",
            "htf_po3": tf_results.get("htf", {"timeframe": tf_windows["htf"], "schematics": [], "summary": summarize_po3([])}),
            "mtf_po3": tf_results.get("mtf", {"timeframe": tf_windows["mtf"], "schematics": [], "summary": summarize_po3([])}),
            "ltf_po3": tf_results.get("ltf", {"timeframe": tf_windows["ltf"], "schematics": [], "summary": summarize_po3([])}),
            "po3_rules": {
                "range": "4H+ range with good RTZ quality and compression",
                "manipulation": "Breakout/breakdown must stay inside DL2 (30% of range)",
                "tct_model": "Must contain TCT accumulation (bullish) or distribution (bearish) in manipulation phase",
                "expansion": "Target extends to opposite range extreme for maximum R:R",
                "exception_1": "2-tap: Single deviation with excellent RTZ — skip third tap",
                "exception_2": "Internal TCT: Model forms in manipulation without sweeping range extreme",
                "key_rule": "When a TCT model fails, look for potential PO3 setup"
            },
            "summary": {
                "total_htf_po3": len(tf_results.get("htf", {}).get("schematics", [])),
                "total_mtf_po3": len(tf_results.get("mtf", {}).get("schematics", [])),
                "total_ltf_po3": len(tf_results.get("ltf", {}).get("schematics", [])),
            }
        })
        return JSONResponse(response_data)

    except Exception as e:
        logger.error(f"[PO3_ERROR] {e}")
        import traceback
        logger.error(traceback.format_exc())
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/trade-execution")
async def trade_execution_endpoint(
    account_balance: float = 10000,
    risk_pct: float = 1.0,
    entry_price: float = 100000,
    stop_loss_price: float = 99500,
    take_profit_price: float = 101500,
    direction: str = "long",
    leverage: float = 10,
    tp2_price: float = None,
    symbol: str = None
):
    """
    TCT Lecture 9 — Trade Execution Planner.

    Generates a complete execution plan including position sizing,
    leverage safety analysis (liquidation vs SL), partial take profits,
    and capital management — all following TCT methodology.

    Now integrates TCT Lecture 1 market structure when symbol is provided:
    - Validates trade direction against HTF trend/EOF bias
    - Checks for BOS confirmation in trade direction
    - Adds confluence scoring to execution plan

    Key Rules:
    - Always ISOLATED margin mode
    - Order by quantity (USDT)
    - Market orders 95% of the time (BOS entries)
    - Liquidation price must ALWAYS be outside stop-loss zone
    - Trail SL to breakeven, then increase leverage + reduce margin
    """
    try:
        # Optionally fetch market structure for confluence
        ms_context = None
        if symbol:
            try:
                sym = resolve_symbol(symbol)
                htf_df = await fetch_mexc_candles(sym, "4h", 100)
                if htf_df is not None and len(htf_df) >= 6:
                    ms = MarketStructure()
                    htf_pivots = ms.detect_pivots(htf_df)
                    ms_context = {
                        "trend": htf_pivots.get("trend"),
                        "eof": htf_pivots.get("eof"),
                        "bos_events": htf_pivots.get("bos_events", []),
                        "levels": htf_pivots.get("levels"),
                    }
            except Exception as ms_err:
                logger.warning(f"[TRADE_EXEC] MS fetch failed: {ms_err}")

        plan = generate_execution_plan(
            account_balance=account_balance,
            risk_pct=risk_pct,
            entry_price=entry_price,
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price,
            direction=direction,
            leverage=leverage,
            tp2_price=tp2_price,
            market_structure=ms_context
        )

        if "error" in plan:
            return JSONResponse({"error": plan["error"]}, status_code=400)

        return JSONResponse(convert_numpy_types(plan))

    except Exception as e:
        logger.error(f"[TRADE_EXECUTION_ERROR] {e}")
        import traceback
        logger.error(traceback.format_exc())
        return JSONResponse({"error": str(e)}, status_code=500)


# ================================================================
# MARKET STRUCTURE DEBUG PAGE
# ================================================================
# Standalone page for debugging and verifying the market structure
# algorithm.  Shows candlestick chart with 6CR pivot points,
# MSH/MSL, BOS events, wick/SFP, Level 1/2/3 hierarchy,
# and multi-timeframe (LTF/MTF/HTF) market structure overlays.
# ================================================================

# Multi-timeframe mapping: for any selected TF, define its LTF and HTF
_MS_TF_MAP = {
    "1m":  {"ltf": "1m",  "mtf": "5m",  "htf": "15m"},
    "5m":  {"ltf": "1m",  "mtf": "5m",  "htf": "15m"},
    "15m": {"ltf": "5m",  "mtf": "15m", "htf": "1h"},
    "30m": {"ltf": "15m", "mtf": "30m", "htf": "4h"},
    "1h":  {"ltf": "15m", "mtf": "1h",  "htf": "4h"},
    "4h":  {"ltf": "1h",  "mtf": "4h",  "htf": "1d"},
    "1d":  {"ltf": "4h",  "mtf": "1d",  "htf": "1d"},
    "1W":  {"ltf": "1d",  "mtf": "1W",  "htf": "1W"},
    "1M":  {"ltf": "1W",  "mtf": "1M",  "htf": "1M"},
}


@app.get("/api/market-structure")
async def get_market_structure_data(symbol: str = "BTCUSDT", timeframe: str = "4h", limit: int = 300):
    """
    Market structure analysis endpoint.
    Returns candles + pivot/BOS/wick/level data for selected TF,
    plus MTF and HTF structure overlays.
    """
    from market_structure import MarketStructure as MS

    try:
        sym = resolve_symbol(symbol)
        # Convert PERP pair format (e.g. BRETT_USDT_PERP) to MEXC spot symbol (BRETTUSDT)
        if "_PERP" in sym:
            sym = sym.replace("_PERP", "").replace("_", "")
        tf_map = _MS_TF_MAP.get(timeframe, {"ltf": timeframe, "mtf": timeframe, "htf": timeframe})

        # Fetch candles for selected TF (primary) + HTF + LTF in parallel
        primary_task = fetch_mexc_candles(sym, timeframe, min(limit, 500))
        htf_task = fetch_mexc_candles(sym, tf_map["htf"], min(limit, 500)) if tf_map["htf"] != timeframe else None
        ltf_task = fetch_mexc_candles(sym, tf_map["ltf"], min(limit, 500)) if tf_map["ltf"] != timeframe else None

        tasks = [primary_task]
        if htf_task:
            tasks.append(htf_task)
        if ltf_task:
            tasks.append(ltf_task)

        results = await asyncio.gather(*tasks)

        df_primary = results[0]
        df_htf = results[1] if htf_task else df_primary
        df_ltf = results[2 if htf_task else 1] if ltf_task else df_primary

        if df_primary is None:
            return JSONResponse({"error": "Failed to fetch candles from MEXC"}, status_code=500)

        # Run market structure analysis on each TF
        ms_primary = MS.detect_pivots(df_primary)
        ms_htf = MS.detect_pivots(df_htf) if df_htf is not None and df_htf is not df_primary else None
        ms_ltf = MS.detect_pivots(df_ltf) if df_ltf is not None and df_ltf is not df_primary else None

        # Format candles for chart
        candles = []
        for _, row in df_primary.iterrows():
            candles.append({
                "time": int(row["open_time"].timestamp()),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
            })

        def _format_ms(ms_data, tf_label, df):
            """Format MS data for JSON output."""
            if not ms_data:
                return None
            # Convert candle times for pivot points
            def _time_at(idx):
                if df is not None and 0 <= idx < len(df):
                    return int(df.iloc[idx]["open_time"].timestamp())
                return None

            pivots = []
            for h in ms_data.get("ms_highs", []):
                pivots.append({
                    "type": "high",
                    "price": h["price"],
                    "idx": h["idx"],
                    "time": _time_at(h["idx"]),
                    "confirmed_at": _time_at(h.get("confirmed_at_idx", h["idx"])),
                })
            for l in ms_data.get("ms_lows", []):
                pivots.append({
                    "type": "low",
                    "price": l["price"],
                    "idx": l["idx"],
                    "time": _time_at(l["idx"]),
                    "confirmed_at": _time_at(l.get("confirmed_at_idx", l["idx"])),
                })
            pivots.sort(key=lambda x: x["idx"])

            bos = []
            for b in ms_data.get("bos_events", []):
                bos.append({
                    "type": b["type"],
                    "price": b["bos_price"],
                    "idx": b["bos_idx"],
                    "time": _time_at(b["bos_idx"]),
                    "broken_level": b["broken_level"],
                    "broken_level_idx": b.get("broken_level_idx"),
                    "broken_level_time": _time_at(b.get("broken_level_idx", 0)),
                    "quality": b.get("quality", "moderate"),
                })

            wicks = []
            for w in ms_data.get("wicks", []):
                wicks.append({
                    "type": w["type"],
                    "wick_price": w["wick_price"],
                    "level_price": w["level_price"],
                    "idx": w["wick_idx"],
                    "time": _time_at(w["wick_idx"]),
                    "scenario": w.get("scenario", "undetermined"),
                    "weakens": w.get("weakens", ""),
                })

            raw_highs = []
            for h in ms_data.get("pivot_highs_6cr", ms_data.get("highs", [])):
                raw_highs.append({
                    "price": h["price"],
                    "idx": h["idx"],
                    "time": _time_at(h["idx"]),
                })

            raw_lows = []
            for l in ms_data.get("pivot_lows_6cr", ms_data.get("lows", [])):
                raw_lows.append({
                    "price": l["price"],
                    "idx": l["idx"],
                    "time": _time_at(l["idx"]),
                })

            return {
                "timeframe": tf_label,
                "trend": ms_data.get("trend", "neutral"),
                "eof": ms_data.get("eof", {}),
                "pivots": pivots,
                "raw_highs": raw_highs,
                "raw_lows": raw_lows,
                "bos_events": bos,
                "wicks": wicks,
                "domino_effect": ms_data.get("domino_effect", {}),
                "trend_shifts": ms_data.get("trend_shifts", []),
                "choch_events": ms_data.get("choch_events", []),
                "rtz": ms_data.get("rtz", {}),
                "levels": ms_data.get("levels", {}),
            }

        primary_result = _format_ms(ms_primary, timeframe, df_primary)
        htf_result = _format_ms(ms_htf, tf_map["htf"], df_htf) if ms_htf else None
        ltf_result = _format_ms(ms_ltf, tf_map["ltf"], df_ltf) if ms_ltf else None

        response = {
            "symbol": sym,
            "timeframe": timeframe,
            "tf_map": tf_map,
            "candles": candles,
            "structure": primary_result,
            "htf_structure": htf_result,
            "ltf_structure": ltf_result,
        }

        return JSONResponse(convert_numpy_types(response))

    except Exception as e:
        logger.error(f"[MS_API_ERROR] {e}")
        import traceback
        logger.error(traceback.format_exc())
        return JSONResponse({"error": str(e)}, status_code=500)


# ---------- Pairs list endpoint for dropdown ----------

@app.get("/api/ms-pairs")
async def get_ms_pairs():
    """Return pairs from mexc_all_pairs.txt for the market structure page dropdown."""
    pairs_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mexc_all_pairs.txt")
    try:
        with open(pairs_path, "r") as f:
            pairs = [line.strip() for line in f if line.strip()]
        return {"pairs": pairs}
    except FileNotFoundError:
        return {"pairs": COIN_LIST}


@app.get("/market-structure", response_class=HTMLResponse)
async def market_structure_page():
    """
    Standalone market structure debug page.
    Candlestick chart with 6CR pivots, MSH/MSL, BOS labels,
    Level 1/2/3 hierarchy lines (HTF=black, MTF=red, LTF=blue).
    """
    html = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
<title>Market Structure — HPB-TCT</title>
<script src="https://unpkg.com/lightweight-charts@4.1.0/dist/lightweight-charts.standalone.production.js"></script>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#fff;color:#333;font-family:'Segoe UI',sans-serif;overflow:hidden}
.header{display:flex;align-items:center;gap:10px;padding:6px 12px;background:#f8f9fa;border-bottom:1px solid #dee2e6;flex-wrap:wrap}
.header h1{font-size:.95rem;color:#333;white-space:nowrap;font-weight:600}
.header h1 .sub{color:#888;font-weight:400;font-size:.75rem;margin-left:5px}
.ctrl{display:flex;align-items:center;gap:5px}
.ctrl label{font-size:.7rem;color:#666}
.ctrl select{background:#fff;color:#333;border:1px solid #ced4da;border-radius:4px;padding:4px 8px;font-size:.75rem;min-width:100px}
.ctrl select:focus{outline:none;border-color:#007bff}
.ctrl button{background:#007bff;color:#fff;border:none;border-radius:4px;padding:5px 12px;font-size:.75rem;cursor:pointer;font-weight:500}
.ctrl button:hover{background:#0056b3}
.ctrl button:disabled{opacity:.5;cursor:not-allowed}
.chart-btns{display:flex;gap:4px;margin-left:8px}
.chart-btns button{background:#6c757d;padding:4px 8px;font-size:.65rem}
.chart-btns button:hover{background:#545b62}
.back-link{color:#666;text-decoration:none;font-size:.7rem;padding:3px 8px;border:1px solid #ced4da;border-radius:4px;margin-left:auto}
.back-link:hover{color:#333;border-color:#adb5bd}
.main-area{display:flex;height:calc(100vh - 80px)}
.chart-wrap{flex:1;position:relative;background:#fff}
.chart-container{width:100%;height:100%}
.info-panel{width:260px;background:#f8f9fa;border-left:1px solid #dee2e6;overflow-y:auto;padding:10px;font-size:.7rem}
.info-panel h3{color:#333;font-size:.8rem;margin:10px 0 6px;border-bottom:1px solid #dee2e6;padding-bottom:4px;font-weight:600}
.info-panel h3:first-child{margin-top:0}
.info-row{display:flex;justify-content:space-between;padding:2px 0;color:#666}
.info-row .val{color:#333;font-weight:600}
.info-row .val.bullish{color:#28a745}
.info-row .val.bearish{color:#dc3545}
.info-row .val.ranging{color:#ffc107}
.tag{display:inline-block;font-size:.6rem;padding:1px 5px;border-radius:3px;margin-left:4px}
.tag.good{background:rgba(40,167,69,.15);color:#28a745}
.tag.bad{background:rgba(220,53,69,.15);color:#dc3545}
.tag.moderate{background:rgba(255,193,7,.15);color:#856404}
.bos-list{margin:4px 0}
.bos-item{padding:3px 6px;margin:2px 0;border-radius:3px;background:rgba(0,0,0,.03);font-size:.65rem}
.legend{display:flex;gap:14px;padding:5px 12px;background:#f8f9fa;border-top:1px solid #dee2e6;font-size:.65rem;color:#666;flex-wrap:wrap}
.legend-item{display:flex;align-items:center;gap:4px}
.legend-line{width:16px;height:2px}
.loading-overlay{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;background:rgba(255,255,255,.9);z-index:100;font-size:.85rem;color:#666}
.loading-overlay.hidden{display:none}
.spinner{width:18px;height:18px;border:2px solid #dee2e6;border-top-color:#007bff;border-radius:50%;animation:spin .6s linear infinite;margin-right:8px}
@keyframes spin{to{transform:rotate(360deg)}}
@media(max-width:768px){
    .info-panel{display:none}
    .header{padding:4px 8px}
    .ctrl select{min-width:80px;font-size:.7rem;padding:3px 5px}
    .ctrl button{padding:4px 8px;font-size:.7rem}
    .chart-btns button{padding:3px 6px}
}
</style>
</head>
<body>

<div class="header">
    <h1>Market Structure <span class="sub">Debug</span></h1>
    <div class="ctrl">
        <label>Pair</label>
        <select id="pairSelect"><option>Loading...</option></select>
    </div>
    <div class="ctrl">
        <label>TF</label>
        <select id="tfSelect">
            <option value="1m">1m</option>
            <option value="5m">5m</option>
            <option value="15m">15m</option>
            <option value="30m">30m</option>
            <option value="1h">1h</option>
            <option value="4h" selected>4h</option>
            <option value="1d">1D</option>
            <option value="1W">1W</option>
            <option value="1M">1M</option>
        </select>
    </div>
    <div class="ctrl">
        <button id="loadBtn" onclick="loadData()">Load</button>
    </div>
    <div class="chart-btns">
        <button onclick="resetChart()">Reset</button>
        <button onclick="fitChart()">Fit</button>
        <button onclick="scalePrice()">Scale Price</button>
        <button onclick="zoomIn()">+</button>
        <button onclick="zoomOut()">−</button>
        <button onclick="testChart()" style="background:#6c757d;color:#fff;border:none;border-radius:4px;padding:4px 10px;font-size:.72rem;cursor:pointer;margin-left:10px" title="Draw 5 hardcoded candles to verify chart renders">Test</button>
    </div>
    <a class="back-link" href="/dashboard">← Back</a>
</div>

<div class="main-area">
    <div class="chart-wrap">
        <div id="chart" class="chart-container"></div>
        <div id="loadingOverlay" class="loading-overlay hidden">
            <div class="spinner"></div> Loading...
        </div>
    </div>
    <div id="infoPanel" class="info-panel">
        <h3>Select pair &amp; timeframe</h3>
        <div class="info-row"><span>Then click <b>Load</b></span></div>
    </div>
</div>

<div class="legend">
    <div class="legend-item"><div class="legend-line" style="background:#000"></div> HTF (black)</div>
    <div class="legend-item"><div class="legend-line" style="background:#dc3545"></div> MTF (red)</div>
    <div class="legend-item"><div class="legend-line" style="background:#007bff"></div> LTF (blue)</div>
    <div class="legend-item"><div class="legend-line" style="background:#ffc107;height:1px;border-top:1px dashed #ffc107"></div> BOS</div>
</div>

<script>
let chart, candleSeries;
let msLineSeries = [];
const BASE = '';
let chartReady = false;

function initChart() {
    if (typeof LightweightCharts === 'undefined') {
        throw new Error('LightweightCharts library not loaded');
    }
    const el = document.getElementById('chart');
    // Guard against zero-size container (can happen on slow renders)
    const w = el.clientWidth || el.offsetWidth || 800;
    const h = el.clientHeight || el.offsetHeight || 500;
    chart = LightweightCharts.createChart(el, {
        width: w,
        height: h,
        layout: { background: { type: 'solid', color: '#ffffff' }, textColor: '#333', fontSize: 11 },
        grid: { vertLines: { color: '#f0f0f0' }, horzLines: { color: '#f0f0f0' } },
        crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
        rightPriceScale: { borderColor: '#dee2e6' },
        timeScale: { borderColor: '#dee2e6', timeVisible: true, secondsVisible: false },
        handleScroll: { mouseWheel: true, pressedMouseMove: true, horzTouchDrag: true, vertTouchDrag: true },
        handleScale: { axisPressedMouseMove: true, mouseWheel: true, pinch: true },
    });
    candleSeries = chart.addCandlestickSeries({
        upColor: '#26a69a', downColor: '#ef5350',
        borderUpColor: '#26a69a', borderDownColor: '#ef5350',
        wickUpColor: '#26a69a', wickDownColor: '#ef5350',
    });
    window.addEventListener('resize', () => {
        chart.applyOptions({ width: el.clientWidth, height: el.clientHeight });
    });
    chartReady = true;
}

// Chart control functions
function resetChart() {
    if (!chartReady) return;
    chart.timeScale().resetTimeScale();
    chart.priceScale('right').applyOptions({ autoScale: true });
}
function fitChart() {
    if (!chartReady) return;
    chart.timeScale().fitContent();
}
function zoomIn() {
    if (!chartReady) return;
    const visibleRange = chart.timeScale().getVisibleLogicalRange();
    if (visibleRange) {
        const center = (visibleRange.from + visibleRange.to) / 2;
        const halfRange = (visibleRange.to - visibleRange.from) / 4;
        chart.timeScale().setVisibleLogicalRange({ from: center - halfRange, to: center + halfRange });
    }
}
function zoomOut() {
    if (!chartReady) return;
    const visibleRange = chart.timeScale().getVisibleLogicalRange();
    if (visibleRange) {
        const center = (visibleRange.from + visibleRange.to) / 2;
        const halfRange = (visibleRange.to - visibleRange.from);
        chart.timeScale().setVisibleLogicalRange({ from: center - halfRange, to: center + halfRange });
    }
}
let lastLoadedCandles = [];
function scalePrice() {
    if (!chartReady || !lastLoadedCandles || lastLoadedCandles.length === 0) return;
    // Use default auto-scaling — it handles multiple series correctly
    chart.priceScale('right').applyOptions({ autoScale: true });
}
function fmtPrice(v) {
    if (v == null || isNaN(v)) return '—';
    const a = Math.abs(v);
    if (a === 0) return '0';
    if (a >= 1) return v.toFixed(Math.min(4, Math.max(2, 4 - Math.floor(Math.log10(a)))));
    const decimals = Math.max(4, -Math.floor(Math.log10(a)) + 3);
    return v.toFixed(Math.min(decimals, 12));
}

// Test function: draws hardcoded candles to verify chart renders
function testChart() {
    if (!chartReady) { alert('Chart not ready'); return; }
    clearOverlays();
    const now = Math.floor(Date.now() / 1000);
    const day = 86400;
    const testData = [];
    for (let i = 0; i < 30; i++) {
        const o = 50000 + Math.random() * 5000;
        const c = o + (Math.random() - 0.5) * 2000;
        testData.push({ time: now - (30 - i) * day, open: o, high: Math.max(o, c) + Math.random() * 500, low: Math.min(o, c) - Math.random() * 500, close: c });
    }
    candleSeries.setData(testData);
    chart.timeScale().fitContent();
    chart.priceScale('right').applyOptions({ autoScale: true });
    document.getElementById('infoPanel').innerHTML = '<h3>Test Chart</h3><div class="info-row">30 random candles drawn. If you see them, the chart works.</div>';
}

async function loadPairs() {
    const MAX_RETRIES = 3;
    for (let attempt = 1; attempt <= MAX_RETRIES; attempt++) {
        try {
            const r = await fetch(BASE + '/api/ms-pairs');
            if (!r.ok) throw new Error(`HTTP ${r.status}`);
            const d = await r.json();
            const sel = document.getElementById('pairSelect');
            sel.innerHTML = '';
            (d.pairs || []).forEach(p => {
                const o = document.createElement('option');
                o.value = p; o.textContent = p;
                if (p === 'BTC_USDT_PERP') o.selected = true;
                sel.appendChild(o);
            });
            return; // success
        } catch(e) {
            console.warn(`loadPairs attempt ${attempt}/${MAX_RETRIES} failed:`, e);
            if (attempt < MAX_RETRIES) await new Promise(r => setTimeout(r, 1000 * attempt));
        }
    }
    // Fallback: at least put a usable default so Load works
    const sel = document.getElementById('pairSelect');
    sel.innerHTML = '<option value="BTCUSDT" selected>BTCUSDT</option><option value="ETHUSDT">ETHUSDT</option><option value="SOLUSDT">SOLUSDT</option>';
}

function clearOverlays() {
    msLineSeries.forEach(s => { try { chart.removeSeries(s); } catch(e){} });
    msLineSeries = [];
    try { candleSeries.setMarkers([]); } catch(e) {}
}

// Fetch with timeout helper
function fetchWithTimeout(url, timeoutMs) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    return fetch(url, { signal: controller.signal }).finally(() => clearTimeout(timer));
}

let loadAttempt = 0;

async function loadData(retryCount) {
    // Guard: if called from event listener, retryCount is an Event object — treat as 0
    const attempt = (typeof retryCount === 'number') ? retryCount : 0;

    const sym = document.getElementById('pairSelect').value;
    const tf = document.getElementById('tfSelect').value;
    if (!sym || sym === 'Loading...') {
        document.getElementById('infoPanel').innerHTML = '<h3>Pairs still loading...</h3><div class="info-row">Please wait and try again</div>';
        return;
    }
    if (!chartReady) {
        document.getElementById('infoPanel').innerHTML = '<h3 style="color:#dc3545">Chart not ready</h3><div class="info-row">Reload the page (Ctrl+Shift+R)</div>';
        return;
    }

    const btn = document.getElementById('loadBtn');
    const overlay = document.getElementById('loadingOverlay');

    btn.disabled = true;
    overlay.classList.remove('hidden');
    if (attempt === 0) clearOverlays();

    try {
        const r = await fetchWithTimeout(
            BASE + `/api/market-structure?symbol=${sym}&timeframe=${tf}&limit=300`,
            30000  // 30s timeout for Render cold starts
        );
        if (!r.ok) {
            const errBody = await r.json().catch(() => ({}));
            throw new Error(errBody.error || `HTTP ${r.status}`);
        }
        const data = await r.json();

        if (!data.candles || data.candles.length === 0) {
            throw new Error(`No candle data returned for ${sym} (${tf})`);
        }

        lastLoadedCandles = data.candles;
        console.log('[MS] Candles loaded:', lastLoadedCandles.length, 'first:', JSON.stringify(lastLoadedCandles[0]), 'last:', JSON.stringify(lastLoadedCandles[lastLoadedCandles.length-1]));

        // Detect price precision for small-priced coins (e.g. PEPE)
        if (lastLoadedCandles.length > 0) {
            const samplePrice = Math.abs(lastLoadedCandles[lastLoadedCandles.length - 1].close);
            let prec = 2;
            if (samplePrice > 0 && samplePrice < 1) prec = Math.max(4, -Math.floor(Math.log10(samplePrice)) + 3);
            else if (samplePrice >= 1 && samplePrice < 10) prec = 4;
            else if (samplePrice >= 10 && samplePrice < 1000) prec = 2;
            prec = Math.min(prec, 12);
            candleSeries.applyOptions({ priceFormat: { type: 'price', precision: prec, minMove: Math.pow(10, -prec) } });
        }
        // Set candle data BEFORE adding any line series
        candleSeries.setData(lastLoadedCandles);
        console.log('[MS] setData done, candles:', lastLoadedCandles.length);

        // Draw structure in order: HTF first (back), then MTF, then LTF (front)
        if (data.htf_structure) {
            drawStructure(data.htf_structure, data.candles, 'htf');
        }
        if (data.structure) {
            drawStructure(data.structure, data.candles, 'mtf');
        }
        if (data.ltf_structure) {
            drawStructure(data.ltf_structure, data.candles, 'ltf');
        }
        console.log('[MS] Structure drawn, line series count:', msLineSeries.length);

        updateInfoPanel(data);

        // Fit chart to show all candle data
        chart.timeScale().fitContent();
        scalePrice();
        console.log('[MS] fitContent done, visible range:', JSON.stringify(chart.timeScale().getVisibleLogicalRange()));

        // Fallback: re-fit after a short delay to handle any rendering lag
        setTimeout(() => {
            try {
                chart.timeScale().fitContent();
                chart.priceScale('right').applyOptions({ autoScale: true });
                console.log('[MS] Fallback fitContent, range:', JSON.stringify(chart.timeScale().getVisibleLogicalRange()));
            } catch(e2) { console.error('[MS] Fallback fitContent error:', e2); }
        }, 200);

    } catch(e) {
        console.error('Load error:', e);
        // Auto-retry once on timeout or network error
        if (attempt < 1 && (e.name === 'AbortError' || e.message.includes('fetch'))) {
            console.log('Retrying after transient error...');
            overlay.querySelector('.spinner').insertAdjacentHTML('afterend', '<span> Retrying...</span>');
            await new Promise(r => setTimeout(r, 2000));
            return loadData(attempt + 1);
        }
        const msg = e.name === 'AbortError' ? 'Request timed out — server may be starting up. Try again in a few seconds.' : e.message;
        document.getElementById('infoPanel').innerHTML = `<h3 style="color:#dc3545">Error</h3><div class="info-row" style="flex-direction:column"><div>${msg}</div><div style="margin-top:6px"><button onclick="loadData()" style="background:#007bff;color:#fff;border:none;border-radius:4px;padding:4px 12px;font-size:.75rem;cursor:pointer">Retry</button></div></div>`;
    } finally {
        btn.disabled = false;
        overlay.classList.add('hidden');
    }
}

// Draw structure with color based on TF type: HTF=black, MTF=red, LTF=blue
function drawStructure(st, candles, tfType) {
    if (!st) return;

    // Color scheme: HTF=black, MTF=red, LTF=blue
    const colors = {
        htf: { line: '#000000', lineWidth: 2.5 },
        mtf: { line: '#dc3545', lineWidth: 2 },
        ltf: { line: '#007bff', lineWidth: 1.5 },
    };
    const c = colors[tfType] || colors.mtf;

    const primaryStart = candles[0]?.time || 0;
    const primaryEnd = candles[candles.length - 1]?.time || Infinity;

    // Filter pivots to primary candle range, skip null/NaN values
    const pivots = (st.pivots || [])
        .filter(p => p.time && p.price != null && isFinite(p.price) && p.time >= primaryStart && p.time <= primaryEnd)
        .sort((a,b) => a.idx - b.idx);

    // Draw zigzag line through pivot points
    if (pivots.length >= 2) {
        // CRITICAL: Lightweight Charts v4.1 crashes with "Value is null" if a
        // line series has duplicate timestamps. Deduplicate by keeping only the
        // last entry per timestamp (pivot high+low on same candle).
        const deduped = [];
        const seenTimes = new Set();
        for (let i = pivots.length - 1; i >= 0; i--) {
            if (!seenTimes.has(pivots[i].time)) {
                seenTimes.add(pivots[i].time);
                deduped.unshift({ time: pivots[i].time, value: pivots[i].price });
            }
        }
        if (deduped.length >= 2) {
            try {
                const zz = chart.addLineSeries({
                    color: c.line,
                    lineWidth: c.lineWidth,
                    lineStyle: 0,
                    crosshairMarkerVisible: false,
                    lastValueVisible: false,
                    priceLineVisible: false,
                });
                zz.setData(deduped);
                msLineSeries.push(zz);
            } catch(e) {
                console.error('[MS] Zigzag line error (' + tfType + '):', e);
            }
        }
    }

    // Only draw BOS markers for MTF (primary structure)
    if (tfType === 'mtf') {
        const markers = [];

        (st.bos_events || []).forEach(b => {
            if (!b.time || b.time < primaryStart || b.time > primaryEnd) return;
            markers.push({
                time: b.time,
                position: b.type === 'bullish' ? 'aboveBar' : 'belowBar',
                color: '#ffc107',
                shape: 'arrowDown',
                size: 1.5,
                text: 'BOS',
            });

            // Horizontal BOS line — both times must be valid and different
            if (b.broken_level_time && b.broken_level && isFinite(b.broken_level)
                && b.broken_level_time !== b.time) {
                try {
                    const bosLine = chart.addLineSeries({
                        color: 'rgba(255,193,7,.6)',
                        lineWidth: 1,
                        lineStyle: LightweightCharts.LineStyle.Dashed,
                        crosshairMarkerVisible: false,
                        lastValueVisible: false,
                        priceLineVisible: false,
                    });
                    bosLine.setData([
                        { time: b.broken_level_time, value: b.broken_level },
                        { time: b.time, value: b.broken_level },
                    ]);
                    msLineSeries.push(bosLine);
                } catch(e) {
                    console.error('[MS] BOS line error:', e);
                }
            }
        });

        if (markers.length > 0) {
            // Deduplicate markers by time (only one marker per candle)
            const markerMap = new Map();
            markers.forEach(m => { if (!markerMap.has(m.time)) markerMap.set(m.time, m); });
            const uniqueMarkers = Array.from(markerMap.values()).sort((a,b) => a.time - b.time);
            try { candleSeries.setMarkers(uniqueMarkers); } catch(e) {
                console.error('[MS] Markers error:', e);
            }
        }
    }
}

function updateInfoPanel(data) {
    const panel = document.getElementById('infoPanel');
    const st = data.structure || {};
    const sym = data.symbol || '—';
    const tf = data.timeframe || '—';

    let html = `<h3>${sym} · ${tf.toUpperCase()}</h3>`;

    const trend = st.trend || 'neutral';
    const tc = trend === 'bullish' ? 'bullish' : trend === 'bearish' ? 'bearish' : 'ranging';
    html += `<div class="info-row"><span>Trend</span><span class="val ${tc}">${trend.toUpperCase()}</span></div>`;

    const eof = st.eof || {};
    if (eof.expectation) {
        html += `<div class="info-row"><span>EOF</span><span class="val">${eof.expectation.replace(/_/g,' ')}</span></div>`;
        if (eof.trend_shift) html += `<div class="info-row"><span>Trend Shift</span><span class="val bearish">YES</span></div>`;
    }

    const highs = (st.pivots || []).filter(p => p.type === 'high');
    const lows = (st.pivots || []).filter(p => p.type === 'low');
    html += `<div class="info-row"><span>MS Highs</span><span class="val">${highs.length}</span></div>`;
    html += `<div class="info-row"><span>MS Lows</span><span class="val">${lows.length}</span></div>`;

    const bos = st.bos_events || [];
    if (bos.length > 0) {
        html += `<h3>BOS Events (${bos.length})</h3>`;
        html += `<div class="bos-list">`;
        bos.slice(-5).forEach(b => {
            const dir = b.type === 'bullish' ? '▲' : '▼';
            const clr = b.type === 'bullish' ? '#28a745' : '#dc3545';
            html += `<div class="bos-item">
                <span style="color:${clr}">${dir} ${b.type}</span>
                @ ${b.price ? fmtPrice(b.price) : '—'}
                <span class="tag ${b.quality || 'moderate'}">${b.quality || '—'}</span>
            </div>`;
        });
        html += `</div>`;
    }

    const domino = st.domino_effect || {};
    if (domino.stage) {
        html += `<h3>Domino Effect</h3>`;
        html += `<div class="info-row"><span>Stage</span><span class="val">${domino.stage.replace(/_/g,' ')}</span></div>`;
        html += `<div class="info-row"><span>Confidence</span><span class="val"><span class="tag ${domino.confidence === 'high' ? 'good' : domino.confidence === 'low' ? 'bad' : 'moderate'}">${domino.confidence}</span></span></div>`;
    }

    const tfMap = data.tf_map || {};
    html += `<h3>Timeframes</h3>`;
    html += `<div class="info-row"><span>HTF</span><span class="val">${tfMap.htf || '—'} (black)</span></div>`;
    html += `<div class="info-row"><span>MTF</span><span class="val">${tfMap.mtf || '—'} (red)</span></div>`;
    html += `<div class="info-row"><span>LTF</span><span class="val">${tfMap.ltf || '—'} (blue)</span></div>`;

    if (data.htf_structure) {
        const ht = data.htf_structure.trend || 'neutral';
        const htc = ht === 'bullish' ? 'bullish' : ht === 'bearish' ? 'bearish' : 'ranging';
        html += `<div class="info-row"><span>HTF Trend</span><span class="val ${htc}">${ht.toUpperCase()}</span></div>`;
    }
    if (data.ltf_structure) {
        const lt = data.ltf_structure.trend || 'neutral';
        const ltc = lt === 'bullish' ? 'bullish' : lt === 'bearish' ? 'bearish' : 'ranging';
        html += `<div class="info-row"><span>LTF Trend</span><span class="val ${ltc}">${lt.toUpperCase()}</span></div>`;
    }

    panel.innerHTML = html;
}

// --- Initialization ---
try {
    initChart();
} catch(e) {
    console.error('Chart init failed:', e);
    document.getElementById('infoPanel').innerHTML =
        '<h3 style="color:#dc3545">Chart Failed to Load</h3>' +
        '<div class="info-row" style="flex-direction:column">' +
        '<div>The charting library did not load. This usually means the CDN was slow or blocked.</div>' +
        '<div style="margin-top:8px"><button onclick="location.reload()" style="background:#007bff;color:#fff;border:none;border-radius:4px;padding:6px 16px;font-size:.8rem;cursor:pointer">Reload Page</button></div></div>';
}

document.getElementById('pairSelect').addEventListener('change', () => loadData());
document.getElementById('tfSelect').addEventListener('change', () => loadData());

// Auto-load: fetch pairs then immediately load chart data
loadPairs().then(() => {
    if (chartReady) loadData();
}).catch(() => {
    if (chartReady) loadData();  // fallback pairs were set, still try
});
</script>
</body>
</html>"""

    return HTMLResponse(content=html, headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"})


# ================================================================
# RANGES DEBUG PAGE
# ================================================================

@app.get("/api/ranges-pairs")
async def get_ranges_pairs():
    """Return pairs from mexc_all_pairs.txt for the ranges page dropdown."""
    pairs_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mexc_all_pairs.txt")
    try:
        with open(pairs_path, "r") as f:
            pairs = [line.strip() for line in f if line.strip()]
        return {"pairs": pairs}
    except FileNotFoundError:
        return {"pairs": COIN_LIST}


def detect_ranges(df: pd.DataFrame, lookback: int = 50):
    """
    Detect TCT-style ranges (consolidation zones) in price data.

    TCT Range Rules from PDFs:
    - Range = price moving sideways (consolidation)
    - Range confirmed when price touches equilibrium (0.5 fib)
    - Deviation limit (DL) = 30% of range size
    """
    if df is None or df.empty or len(df) < 10:
        return []

    ranges = []
    df = df.copy()
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values

    # Convert times to Unix timestamps for JSON serialization
    # Use the same approach as market-structure endpoint
    try:
        times = [int(row["open_time"].timestamp()) for _, row in df.iterrows()]
    except Exception:
        times = list(range(len(df)))

    n = len(df)

    # Use sliding window approach to find consolidation ranges
    # A range is where price moves sideways within a bounded area
    window_sizes = [20, 40, 60]  # Look for ranges of different sizes

    for window_size in window_sizes:
        if n < window_size:
            continue

        for start in range(0, n - window_size, window_size // 2):
            end = min(start + window_size, n - 1)

            window_highs = highs[start:end + 1]
            window_lows = lows[start:end + 1]

            range_high = float(max(window_highs))
            range_low = float(min(window_lows))
            range_size = range_high - range_low

            # Skip if range is too small (less than 0.5% of price)
            if range_size < range_low * 0.005:
                continue

            # Calculate the average true range to check if it's consolidating
            avg_candle_size = sum(window_highs - window_lows) / len(window_highs)

            # A good range has small candles relative to the range size
            # (price is consolidating, not trending aggressively)
            consolidation_ratio = avg_candle_size / range_size if range_size > 0 else 1

            # Skip if candles are too large (trending, not ranging)
            if consolidation_ratio > 0.4:
                continue

            # Check how many times price touched the boundaries
            touch_threshold_high = range_high - (range_size * 0.1)
            touch_threshold_low = range_low + (range_size * 0.1)

            touches_high = sum(1 for h in window_highs if h >= touch_threshold_high)
            touches_low = sum(1 for l in window_lows if l <= touch_threshold_low)

            # Need at least some touches on both sides for a valid range
            if touches_high < 2 or touches_low < 2:
                continue

            # Calculate equilibrium (0.5 fib) and deviation limits
            equilibrium = range_low + (range_size * 0.5)
            deviation_limit = range_size * 0.30

            # Check if price touched equilibrium
            eq_touched = any(window_lows[i] <= equilibrium <= window_highs[i]
                           for i in range(len(window_highs)))

            # Assess quality
            quality = "good" if (touches_high >= 3 and touches_low >= 3 and consolidation_ratio < 0.25) else \
                      "moderate" if (touches_high >= 2 and touches_low >= 2) else "bad"

            # Determine trend before range
            if start >= 10:
                pre_close = closes[start - 10]
                start_close = closes[start]
                trend_before = "up" if start_close > pre_close else "down"
            else:
                trend_before = "neutral"

            ranges.append({
                'start_time': times[start],
                'end_time': times[end],
                'start_idx': int(start),
                'end_idx': int(end),
                'high': range_high,
                'low': range_low,
                'equilibrium': float(equilibrium),
                'deviation_limit': float(deviation_limit),
                'upper_dl': float(range_high + deviation_limit),
                'lower_dl': float(range_low - deviation_limit),
                'range_size': float(range_size),
                'eq_touched': eq_touched,
                'quality': quality,
                'candles': int(end - start),
                'trend_before': trend_before,
            })

    # Remove overlapping ranges, keeping the better quality ones
    if ranges:
        ranges.sort(key=lambda r: (r['quality'] == 'good', r['quality'] == 'moderate', r['candles']), reverse=True)
        filtered = []
        for r in ranges:
            # Check if this range overlaps significantly with any already selected range
            dominated = False
            for f in filtered:
                overlap_start = max(r['start_idx'], f['start_idx'])
                overlap_end = min(r['end_idx'], f['end_idx'])
                if overlap_end > overlap_start:
                    overlap_pct = (overlap_end - overlap_start) / (r['end_idx'] - r['start_idx'])
                    if overlap_pct > 0.5:
                        dominated = True
                        break
            if not dominated:
                filtered.append(r)
        ranges = filtered[:10]  # Keep top 10 ranges

    return ranges


@app.get("/api/ranges-debug")
async def get_ranges_debug_data(symbol: str = "BTC_USDT_PERP", timeframe: str = "4h", limit: int = 300):
    """
    API endpoint for ranges debug page.
    Returns candle data with detected TCT ranges.
    """
    try:
        # Convert symbol format like market-structure does
        sym = resolve_symbol(symbol)
        if "_PERP" in sym:
            sym = sym.replace("_PERP", "").replace("_", "")

        # HTF/LTF mapping
        htf_ltf_map = {
            "1m": {"htf": "15m", "ltf": "1m"},
            "5m": {"htf": "1h", "ltf": "1m"},
            "15m": {"htf": "4h", "ltf": "5m"},
            "30m": {"htf": "4h", "ltf": "15m"},
            "1h": {"htf": "4h", "ltf": "15m"},
            "4h": {"htf": "1d", "ltf": "1h"},
            "8h": {"htf": "1d", "ltf": "4h"},
            "1d": {"htf": "1W", "ltf": "4h"},
            "1D": {"htf": "1W", "ltf": "4h"},
            "1W": {"htf": "1M", "ltf": "1d"},
        }
        tf_config = htf_ltf_map.get(timeframe, {"htf": "1d", "ltf": "1h"})

        # Fetch candle data for primary timeframe
        df_primary = await fetch_mexc_candles(sym, timeframe, limit)
        if df_primary is None or df_primary.empty:
            return JSONResponse({"error": f"No data for {symbol} ({timeframe}). '{sym}' may not exist on MEXC spot. Try a different pair."}, status_code=404)

        # Format candles for chart (same pattern as market-structure)
        candles = []
        for _, row in df_primary.iterrows():
            candles.append({
                "time": int(row["open_time"].timestamp()),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
            })

        # Detect ranges on primary timeframe
        primary_ranges = detect_ranges(df_primary, lookback=limit)

        # Fetch and detect ranges on HTF
        htf_tf = tf_config["htf"]
        df_htf = await fetch_mexc_candles(sym, htf_tf, limit)
        htf_ranges = detect_ranges(df_htf, lookback=limit) if df_htf is not None and not df_htf.empty else []

        # Fetch and detect ranges on LTF
        ltf_tf = tf_config["ltf"]
        df_ltf = await fetch_mexc_candles(sym, ltf_tf, limit * 2)
        ltf_ranges = detect_ranges(df_ltf, lookback=limit * 2) if df_ltf is not None and not df_ltf.empty else []

        response = {
            "symbol": symbol,
            "timeframe": timeframe,
            "tf_config": {
                "htf": htf_tf,
                "mtf": timeframe,
                "ltf": ltf_tf,
            },
            "candles": candles,
            "ranges": primary_ranges,
            "htf_ranges": htf_ranges,
            "ltf_ranges": ltf_ranges,
            "debug": {
                "sym_resolved": sym,
                "primary_candles": len(candles),
                "htf_candles": len(df_htf) if df_htf is not None else 0,
                "ltf_candles": len(df_ltf) if df_ltf is not None else 0,
            }
        }

        return JSONResponse(convert_numpy_types(response))

    except Exception as e:
        logger.error(f"[RANGES_API_ERROR] {e}")
        import traceback
        tb = traceback.format_exc()
        logger.error(tb)
        return JSONResponse({"error": str(e), "traceback": tb}, status_code=500)


@app.get("/ranges", response_class=HTMLResponse)
async def ranges_page():
    """
    Standalone ranges debug page for testing TCT range detection algorithm.
    Displays candlestick chart with detected ranges following TCT model rules.
    """
    html = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
<title>Ranges — HPB-TCT</title>
<script src="https://unpkg.com/lightweight-charts@4.1.0/dist/lightweight-charts.standalone.production.js"></script>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#fff;color:#333;font-family:'Segoe UI',sans-serif;overflow:hidden}
.header{display:flex;align-items:center;gap:10px;padding:6px 12px;background:#f8f9fa;border-bottom:1px solid #dee2e6;flex-wrap:wrap}
.header h1{font-size:.95rem;color:#333;white-space:nowrap;font-weight:600}
.header h1 .sub{color:#888;font-weight:400;font-size:.75rem;margin-left:5px}
.ctrl{display:flex;align-items:center;gap:5px}
.ctrl label{font-size:.7rem;color:#666}
.ctrl select{background:#fff;color:#333;border:1px solid #ced4da;border-radius:4px;padding:4px 8px;font-size:.75rem;min-width:100px}
.ctrl select:focus{outline:none;border-color:#007bff}
.ctrl button{background:#007bff;color:#fff;border:none;border-radius:4px;padding:5px 12px;font-size:.75rem;cursor:pointer;font-weight:500}
.ctrl button:hover{background:#0056b3}
.ctrl button:disabled{opacity:.5;cursor:not-allowed}
.chart-btns{display:flex;gap:4px;margin-left:8px}
.chart-btns button{background:#6c757d;padding:4px 8px;font-size:.65rem}
.chart-btns button:hover{background:#545b62}
.back-link{color:#666;text-decoration:none;font-size:.7rem;padding:3px 8px;border:1px solid #ced4da;border-radius:4px;margin-left:auto}
.back-link:hover{color:#333;border-color:#adb5bd}
.main-area{display:flex;height:calc(100vh - 80px)}
.chart-wrap{flex:1;position:relative;background:#fff}
.chart-container{width:100%;height:100%}
.info-panel{width:280px;background:#f8f9fa;border-left:1px solid #dee2e6;overflow-y:auto;padding:10px;font-size:.7rem}
.info-panel h3{color:#333;font-size:.8rem;margin:10px 0 6px;border-bottom:1px solid #dee2e6;padding-bottom:4px;font-weight:600}
.info-panel h3:first-child{margin-top:0}
.info-row{display:flex;justify-content:space-between;padding:2px 0;color:#666}
.info-row .val{color:#333;font-weight:600}
.info-row .val.bullish{color:#28a745}
.info-row .val.bearish{color:#dc3545}
.info-row .val.ranging{color:#ffc107}
.tag{display:inline-block;font-size:.6rem;padding:1px 5px;border-radius:3px;margin-left:4px}
.tag.good{background:rgba(40,167,69,.15);color:#28a745}
.tag.bad{background:rgba(220,53,69,.15);color:#dc3545}
.tag.moderate{background:rgba(255,193,7,.15);color:#856404}
.range-list{margin:4px 0}
.range-item{padding:4px 6px;margin:3px 0;border-radius:3px;background:rgba(0,0,0,.03);font-size:.65rem;border-left:3px solid #007bff}
.range-item.htf{border-left-color:#000}
.range-item.ltf{border-left-color:#dc3545}
.legend{display:flex;gap:14px;padding:5px 12px;background:#f8f9fa;border-top:1px solid #dee2e6;font-size:.65rem;color:#666;flex-wrap:wrap}
.legend-item{display:flex;align-items:center;gap:4px}
.legend-box{width:12px;height:12px;border:2px solid;background:transparent}
.loading-overlay{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;background:rgba(255,255,255,.9);z-index:100;font-size:.85rem;color:#666}
.loading-overlay.hidden{display:none}
.spinner{width:18px;height:18px;border:2px solid #dee2e6;border-top-color:#007bff;border-radius:50%;animation:spin .6s linear infinite;margin-right:8px}
@keyframes spin{to{transform:rotate(360deg)}}
@media(max-width:768px){
    .info-panel{display:none}
    .header{padding:4px 8px}
    .ctrl select{min-width:80px;font-size:.7rem;padding:3px 5px}
    .ctrl button{padding:4px 8px;font-size:.7rem}
    .chart-btns button{padding:3px 6px}
}
</style>
</head>
<body>

<div class="header">
    <h1>Ranges <span class="sub">Debug</span></h1>
    <div class="ctrl">
        <label>Pair</label>
        <select id="pairSelect"><option>Loading...</option></select>
    </div>
    <div class="ctrl">
        <label>TF</label>
        <select id="tfSelect">
            <option value="1m">1m</option>
            <option value="5m">5m</option>
            <option value="15m">15m</option>
            <option value="30m">30m</option>
            <option value="1h">1h</option>
            <option value="4h" selected>4h</option>
            <option value="8h">8h</option>
            <option value="1d">1D</option>
            <option value="1W">1W</option>
        </select>
    </div>
    <div class="ctrl">
        <button id="loadBtn" onclick="loadData()">Load</button>
    </div>
    <div class="chart-btns">
        <button onclick="resetChart()">Reset</button>
        <button onclick="fitChart()">Fit</button>
        <button onclick="scalePrice()">Scale Price</button>
        <button onclick="zoomIn()">+</button>
        <button onclick="zoomOut()">−</button>
    </div>
    <a class="back-link" href="/dashboard">← Back</a>
</div>

<div class="main-area">
    <div class="chart-wrap">
        <div id="chart" class="chart-container"></div>
        <div id="loadingOverlay" class="loading-overlay hidden">
            <div class="spinner"></div> Loading...
        </div>
    </div>
    <div id="infoPanel" class="info-panel">
        <h3>Select pair &amp; timeframe</h3>
        <div class="info-row"><span>Then click <b>Load</b></span></div>
    </div>
</div>

<div class="legend">
    <div class="legend-item"><div class="legend-box" style="border-color:#000"></div> HTF Range (black)</div>
    <div class="legend-item"><div class="legend-box" style="border-color:#dc3545"></div> LTF Range (red)</div>
    <div class="legend-item"><div style="width:16px;height:1px;border-top:1px dashed #ffc107"></div> Equilibrium (0.5)</div>
    <div class="legend-item"><div style="width:16px;height:1px;border-top:1px dotted #6c757d"></div> Deviation Limit</div>
</div>

<script>
let chart, candleSeries;
let rangeSeries = [];
const BASE = '';

function initChart() {
    const el = document.getElementById('chart');
    chart = LightweightCharts.createChart(el, {
        width: el.clientWidth,
        height: el.clientHeight,
        layout: { background: { type: 'solid', color: '#ffffff' }, textColor: '#333', fontSize: 11 },
        grid: { vertLines: { color: '#f0f0f0' }, horzLines: { color: '#f0f0f0' } },
        crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
        rightPriceScale: { borderColor: '#dee2e6' },
        timeScale: { borderColor: '#dee2e6', timeVisible: true, secondsVisible: false },
        handleScroll: { mouseWheel: true, pressedMouseMove: true, horzTouchDrag: true, vertTouchDrag: true },
        handleScale: { axisPressedMouseMove: true, mouseWheel: true, pinch: true },
    });
    candleSeries = chart.addCandlestickSeries({
        upColor: '#26a69a', downColor: '#ef5350',
        borderUpColor: '#26a69a', borderDownColor: '#ef5350',
        wickUpColor: '#26a69a', wickDownColor: '#ef5350',
    });
    window.addEventListener('resize', () => {
        chart.applyOptions({ width: el.clientWidth, height: el.clientHeight });
    });
}

function resetChart() {
    candleSeries.applyOptions({ autoscaleInfoProvider: undefined });
    chart.timeScale().resetTimeScale();
    chart.priceScale('right').applyOptions({ autoScale: true });
}
function fitChart() {
    chart.timeScale().fitContent();
}
function zoomIn() {
    const visibleRange = chart.timeScale().getVisibleLogicalRange();
    if (visibleRange) {
        const center = (visibleRange.from + visibleRange.to) / 2;
        const halfRange = (visibleRange.to - visibleRange.from) / 4;
        chart.timeScale().setVisibleLogicalRange({ from: center - halfRange, to: center + halfRange });
    }
}
function zoomOut() {
    const visibleRange = chart.timeScale().getVisibleLogicalRange();
    if (visibleRange) {
        const center = (visibleRange.from + visibleRange.to) / 2;
        const halfRange = (visibleRange.to - visibleRange.from);
        chart.timeScale().setVisibleLogicalRange({ from: center - halfRange, to: center + halfRange });
    }
}
let lastLoadedCandles = [];
function scalePrice() {
    if (!lastLoadedCandles || lastLoadedCandles.length === 0) return;
    candleSeries.applyOptions({
        autoscaleInfoProvider: () => {
            const visRange = chart.timeScale().getVisibleLogicalRange();
            if (!visRange) return null;
            const fromIdx = Math.max(0, Math.floor(visRange.from));
            const toIdx = Math.min(lastLoadedCandles.length - 1, Math.ceil(visRange.to));
            let minP = Infinity, maxP = -Infinity;
            for (let i = fromIdx; i <= toIdx; i++) {
                const c = lastLoadedCandles[i];
                if (c) { if (c.low < minP) minP = c.low; if (c.high > maxP) maxP = c.high; }
            }
            if (minP === Infinity) return null;
            const pad = (maxP - minP) * 0.05;
            return { priceRange: { minValue: minP - pad, maxValue: maxP + pad } };
        }
    });
    chart.priceScale('right').applyOptions({ autoScale: true });
}
function fmtPrice(v) {
    if (v == null || isNaN(v)) return '—';
    const a = Math.abs(v);
    if (a === 0) return '0';
    if (a >= 1) return v.toFixed(Math.min(4, Math.max(2, 4 - Math.floor(Math.log10(a)))));
    // For values < 1, show enough decimals to get significant digits
    const decimals = Math.max(4, -Math.floor(Math.log10(a)) + 3);
    return v.toFixed(Math.min(decimals, 12));
}

async function loadPairs() {
    try {
        const r = await fetch(BASE + '/api/ranges-pairs');
        const d = await r.json();
        const sel = document.getElementById('pairSelect');
        sel.innerHTML = '';
        (d.pairs || []).forEach(p => {
            const o = document.createElement('option');
            o.value = p; o.textContent = p;
            if (p === 'BTC_USDT_PERP') o.selected = true;
            sel.appendChild(o);
        });
    } catch(e) { console.error('Failed to load pairs', e); }
}

function clearOverlays() {
    rangeSeries.forEach(s => { try { chart.removeSeries(s); } catch(e){} });
    rangeSeries = [];
    candleSeries.setMarkers([]);
}

// Draw a range box using line series
function drawRange(range, type, candles) {
    // Color scheme: HTF=black, LTF=red
    const colors = {
        htf: { border: '#000000', fill: 'rgba(0,0,0,0.05)', eq: '#000000', dl: '#666666', lineWidth: 2 },
        ltf: { border: '#dc3545', fill: 'rgba(220,53,69,0.05)', eq: '#dc3545', dl: '#dc3545', lineWidth: 1.5 },
        mtf: { border: '#007bff', fill: 'rgba(0,123,255,0.05)', eq: '#007bff', dl: '#007bff', lineWidth: 1.5 },
    };
    const c = colors[type] || colors.mtf;

    // Find the time range within visible candles
    const startTime = range.start_time;
    const endTime = range.end_time;

    // Filter to candles we have
    const firstCandle = candles[0]?.time || 0;
    const lastCandle = candles[candles.length - 1]?.time || Infinity;

    // Only draw if range overlaps with our candle data
    if (endTime < firstCandle || startTime > lastCandle) return;

    const effectiveStart = Math.max(startTime, firstCandle);
    const effectiveEnd = Math.min(endTime, lastCandle);

    // Top line of range
    const topLine = chart.addLineSeries({
        color: c.border,
        lineWidth: c.lineWidth,
        lineStyle: 0,
        crosshairMarkerVisible: false,
        lastValueVisible: false,
        priceLineVisible: false,
    });
    topLine.setData([
        { time: effectiveStart, value: range.high },
        { time: effectiveEnd, value: range.high },
    ]);
    rangeSeries.push(topLine);

    // Bottom line of range
    const bottomLine = chart.addLineSeries({
        color: c.border,
        lineWidth: c.lineWidth,
        lineStyle: 0,
        crosshairMarkerVisible: false,
        lastValueVisible: false,
        priceLineVisible: false,
    });
    bottomLine.setData([
        { time: effectiveStart, value: range.low },
        { time: effectiveEnd, value: range.low },
    ]);
    rangeSeries.push(bottomLine);

    // Equilibrium line (0.5 fib) - dashed
    const eqLine = chart.addLineSeries({
        color: 'rgba(255,193,7,0.8)',
        lineWidth: 1,
        lineStyle: LightweightCharts.LineStyle.Dashed,
        crosshairMarkerVisible: false,
        lastValueVisible: false,
        priceLineVisible: false,
    });
    eqLine.setData([
        { time: effectiveStart, value: range.equilibrium },
        { time: effectiveEnd, value: range.equilibrium },
    ]);
    rangeSeries.push(eqLine);

    // Deviation limit lines - dotted (optional, only for good/moderate ranges)
    if (range.quality !== 'bad') {
        // Upper deviation limit
        const upperDL = chart.addLineSeries({
            color: 'rgba(108,117,125,0.5)',
            lineWidth: 1,
            lineStyle: LightweightCharts.LineStyle.Dotted,
            crosshairMarkerVisible: false,
            lastValueVisible: false,
            priceLineVisible: false,
        });
        upperDL.setData([
            { time: effectiveStart, value: range.upper_dl },
            { time: effectiveEnd, value: range.upper_dl },
        ]);
        rangeSeries.push(upperDL);

        // Lower deviation limit
        const lowerDL = chart.addLineSeries({
            color: 'rgba(108,117,125,0.5)',
            lineWidth: 1,
            lineStyle: LightweightCharts.LineStyle.Dotted,
            crosshairMarkerVisible: false,
            lastValueVisible: false,
            priceLineVisible: false,
        });
        lowerDL.setData([
            { time: effectiveStart, value: range.lower_dl },
            { time: effectiveEnd, value: range.lower_dl },
        ]);
        rangeSeries.push(lowerDL);
    }
}

async function loadData() {
    const sym = document.getElementById('pairSelect').value;
    const tf = document.getElementById('tfSelect').value;
    const btn = document.getElementById('loadBtn');
    const overlay = document.getElementById('loadingOverlay');

    btn.disabled = true;
    overlay.classList.remove('hidden');
    clearOverlays();

    try {
        const r = await fetch(BASE + `/api/ranges-debug?symbol=${sym}&timeframe=${tf}&limit=300`);
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        const data = await r.json();

        lastLoadedCandles = data.candles || [];
        // Detect price precision for small-priced coins (e.g. PEPE)
        if (lastLoadedCandles.length > 0) {
            const samplePrice = Math.abs(lastLoadedCandles[lastLoadedCandles.length - 1].close);
            let prec = 2;
            if (samplePrice > 0 && samplePrice < 1) prec = Math.max(4, -Math.floor(Math.log10(samplePrice)) + 3);
            else if (samplePrice >= 1 && samplePrice < 10) prec = 4;
            else if (samplePrice >= 10 && samplePrice < 1000) prec = 2;
            prec = Math.min(prec, 12);
            candleSeries.applyOptions({ priceFormat: { type: 'price', precision: prec, minMove: Math.pow(10, -prec) } });
        }
        candleSeries.applyOptions({ autoscaleInfoProvider: undefined });
        candleSeries.setData(lastLoadedCandles);

        // Draw ranges in order: HTF first (back), then LTF (front)
        // HTF ranges in black
        (data.htf_ranges || []).forEach(range => {
            drawRange(range, 'htf', data.candles);
        });

        // LTF ranges in red
        (data.ltf_ranges || []).forEach(range => {
            drawRange(range, 'ltf', data.candles);
        });

        updateInfoPanel(data);
        chart.timeScale().fitContent();
        // Auto-scale price to candle data
        scalePrice();

    } catch(e) {
        console.error('Load error:', e);
        document.getElementById('infoPanel').innerHTML = `<h3 style="color:#dc3545">Error</h3><div class="info-row">${e.message}</div>`;
    } finally {
        btn.disabled = false;
        overlay.classList.add('hidden');
    }
}

function updateInfoPanel(data) {
    const panel = document.getElementById('infoPanel');
    const sym = data.symbol || '—';
    const tf = data.timeframe || '—';
    const tfConfig = data.tf_config || {};

    let html = `<h3>${sym} · ${tf.toUpperCase()}</h3>`;

    // Timeframe configuration
    html += `<div class="info-row"><span>HTF</span><span class="val">${tfConfig.htf || '—'} (black)</span></div>`;
    html += `<div class="info-row"><span>MTF</span><span class="val">${tfConfig.mtf || '—'}</span></div>`;
    html += `<div class="info-row"><span>LTF</span><span class="val">${tfConfig.ltf || '—'} (red)</span></div>`;

    // HTF Ranges
    const htfRanges = data.htf_ranges || [];
    html += `<h3>HTF Ranges (${htfRanges.length})</h3>`;
    if (htfRanges.length > 0) {
        html += `<div class="range-list">`;
        htfRanges.slice(-5).forEach((r, i) => {
            html += `<div class="range-item htf">
                <div><b>Range ${i+1}</b> <span class="tag ${r.quality}">${r.quality}</span></div>
                <div>High: ${fmtPrice(r.high)}</div>
                <div>Low: ${fmtPrice(r.low)}</div>
                <div>EQ: ${fmtPrice(r.equilibrium)} ${r.eq_touched ? '✓' : ''}</div>
                <div>Size: ${fmtPrice(r.range_size)} (${r.candles} candles)</div>
                <div>DL: ±${fmtPrice(r.deviation_limit)}</div>
            </div>`;
        });
        html += `</div>`;
    } else {
        html += `<div class="info-row"><span>No HTF ranges detected</span></div>`;
    }

    // LTF Ranges
    const ltfRanges = data.ltf_ranges || [];
    html += `<h3>LTF Ranges (${ltfRanges.length})</h3>`;
    if (ltfRanges.length > 0) {
        html += `<div class="range-list">`;
        ltfRanges.slice(-5).forEach((r, i) => {
            html += `<div class="range-item ltf">
                <div><b>Range ${i+1}</b> <span class="tag ${r.quality}">${r.quality}</span></div>
                <div>High: ${fmtPrice(r.high)}</div>
                <div>Low: ${fmtPrice(r.low)}</div>
                <div>EQ: ${fmtPrice(r.equilibrium)} ${r.eq_touched ? '✓' : ''}</div>
                <div>Size: ${fmtPrice(r.range_size)} (${r.candles} candles)</div>
            </div>`;
        });
        html += `</div>`;
    } else {
        html += `<div class="info-row"><span>No LTF ranges detected</span></div>`;
    }

    // TCT Range Rules reference
    html += `<h3>TCT Range Rules</h3>`;
    html += `<div class="info-row" style="flex-direction:column;gap:2px">
        <div>• Range = sideways consolidation</div>
        <div>• EQ = 0.5 fib (equilibrium)</div>
        <div>• DL = 30% deviation limit</div>
        <div>• Good: slow action, liquidity</div>
        <div>• Bad: V-shaped, aggressive</div>
    </div>`;

    panel.innerHTML = html;
}

initChart();
loadPairs();
document.getElementById('pairSelect').addEventListener('change', loadData);
document.getElementById('tfSelect').addEventListener('change', loadData);
</script>
</body>
</html>"""

    return HTMLResponse(content=html)


# ================================================================
# SUPPLY & DEMAND DEBUG PAGE  (TCT Mentorship – Lecture 3)
# ================================================================
# Two types of zones:
#   1. Order Blocks (OB/OBIF) – single candle S/D with FVG requirement
#   2. Structure Supply/Demand (SS SD) – multi-candle S/D with FVG
# Both require Fair Value Gap (inefficiency) to be valid.
# ================================================================

@app.get("/api/sd-pairs")
async def get_sd_pairs():
    """Return pairs from mexc_all_pairs.txt for the supply-demand page dropdown."""
    pairs_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mexc_all_pairs.txt")
    try:
        with open(pairs_path, "r") as f:
            pairs = [line.strip() for line in f if line.strip()]
        return {"pairs": pairs}
    except FileNotFoundError:
        return {"pairs": COIN_LIST}


def _detect_supply_demand_zones(df: pd.DataFrame) -> Dict:
    """
    Full supply/demand detection pipeline for a single timeframe.

    Steps (per TCT Lecture 3):
      1. Run market structure (6CR pivots, MSH/MSL, BOS, trend)
      2. Detect FVGs (inefficiencies)
      3. Detect Order Blocks (OBIF – single candle with FVG)
      4. Detect Structure S/D (SS SD – multi-candle with FVG)
      5. Check mitigation status for each zone
      6. Score zones by location (pivot, range, deviation)

    Returns dict with all zone data + market structure context.
    """
    from market_structure import MarketStructure as MS

    if df is None or df.empty or len(df) < 10:
        return {
            "order_blocks": [], "structure_zones": [],
            "fvgs": {"bullish_fvgs": [], "bearish_fvgs": []},
            "pivots": {}, "trend": "neutral",
        }

    # Step 1: Market structure
    ms_data = MS.detect_pivots(df)
    trend = ms_data.get("trend", "neutral")

    # Step 2: Detect FVGs
    fvgs = FairValueGap.detect_fvgs(df)

    # Step 3: Detect Order Blocks (OBIF)
    obs = OrderBlock.detect_order_blocks(df, fvgs)

    # Step 4: Detect Structure S/D zones
    ss_sd = StructureSupplyDemand.detect_structure_zones(df, fvgs, ms_data)

    # Step 5: Check mitigation on all zones
    all_order_blocks = []
    for ob in obs.get("bullish_obs", []):
        ob_refined = ZoneRefinement.refine_zone_after_mitigation(ob, df)
        idx = ob.get("idx", 0)
        # Mark as mitigated if price returned to zone after formation
        if idx + 3 < len(df):
            post_candles = df.iloc[idx + 3:]
            entered = post_candles[
                (post_candles["low"] <= ob["top"]) &
                (post_candles["high"] >= ob["bottom"])
            ]
            if len(entered) > 0:
                ob_refined["mitigated"] = True
                ob_refined["mitigation_idx"] = int(entered.index[0])
        all_order_blocks.append(ob_refined)

    for ob in obs.get("bearish_obs", []):
        ob_refined = ZoneRefinement.refine_zone_after_mitigation(ob, df)
        idx = ob.get("idx", 0)
        if idx + 3 < len(df):
            post_candles = df.iloc[idx + 3:]
            entered = post_candles[
                (post_candles["low"] <= ob["top"]) &
                (post_candles["high"] >= ob["bottom"])
            ]
            if len(entered) > 0:
                ob_refined["mitigated"] = True
                ob_refined["mitigation_idx"] = int(entered.index[0])
        all_order_blocks.append(ob_refined)

    all_structure_zones = []
    for zone in ss_sd.get("demand_zones", []):
        zone_refined = ZoneRefinement.refine_zone_after_mitigation(zone, df)
        end_idx = zone.get("end_idx", 0)
        if end_idx + 3 < len(df):
            post_candles = df.iloc[end_idx + 3:]
            entered = post_candles[
                (post_candles["low"] <= zone["top"]) &
                (post_candles["high"] >= zone["bottom"])
            ]
            if len(entered) > 0:
                zone_refined["mitigated"] = True
                zone_refined["mitigation_idx"] = int(entered.index[0])
        all_structure_zones.append(zone_refined)

    for zone in ss_sd.get("supply_zones", []):
        zone_refined = ZoneRefinement.refine_zone_after_mitigation(zone, df)
        end_idx = zone.get("end_idx", 0)
        if end_idx + 3 < len(df):
            post_candles = df.iloc[end_idx + 3:]
            entered = post_candles[
                (post_candles["low"] <= zone["top"]) &
                (post_candles["high"] >= zone["bottom"])
            ]
            if len(entered) > 0:
                zone_refined["mitigated"] = True
                zone_refined["mitigation_idx"] = int(entered.index[0])
        all_structure_zones.append(zone_refined)

    # Step 6: Score zones by location
    # Find best range for scoring context
    ranges = detect_ranges(df, lookback=len(df))
    best_range = ranges[0] if ranges else None
    current_price = float(df.iloc[-1]["close"])

    combined_zones = {
        "bullish_obs": [z for z in all_order_blocks if z.get("type") == "bullish"],
        "bearish_obs": [z for z in all_order_blocks if z.get("type") == "bearish"],
        "demand_zones": [z for z in all_structure_zones if z.get("type") == "demand"],
        "supply_zones": [z for z in all_structure_zones if z.get("type") == "supply"],
    }

    scored_zones = ZoneScoring.score_zones(combined_zones, ms_data, best_range, current_price)

    # Enhance each scored zone with pivot quality
    for zone in scored_zones:
        pq = PivotQualityScorer.score_pivot_quality(zone, ms_data, df)
        zone["pivot_quality"] = pq

    return {
        "order_blocks": all_order_blocks,
        "structure_zones": all_structure_zones,
        "scored_zones": scored_zones,
        "fvgs": fvgs,
        "ms_data": ms_data,
        "trend": trend,
        "ranges": ranges,
    }


@app.get("/api/supply-demand")
async def get_supply_demand_data(symbol: str = "BTC_USDT_PERP", timeframe: str = "4h", limit: int = 300):
    """
    API endpoint for Supply & Demand debug page.
    Returns candles + OBIFs + Structure S/D + FVGs + scoring for MTF/HTF/LTF.
    """
    try:
        sym = resolve_symbol(symbol)
        if "_PERP" in sym:
            sym = sym.replace("_PERP", "").replace("_", "")

        # Use same TF hierarchy as market-structure page
        tf_map = _MS_TF_MAP.get(timeframe, {"ltf": timeframe, "mtf": timeframe, "htf": timeframe})

        # Fetch candles for primary + HTF + LTF in parallel
        primary_task = fetch_mexc_candles(sym, timeframe, min(limit, 500))
        htf_task = fetch_mexc_candles(sym, tf_map["htf"], min(limit, 500)) if tf_map["htf"] != timeframe else None
        ltf_task = fetch_mexc_candles(sym, tf_map["ltf"], min(limit * 2, 800)) if tf_map["ltf"] != timeframe else None

        tasks = [primary_task]
        if htf_task:
            tasks.append(htf_task)
        if ltf_task:
            tasks.append(ltf_task)

        results = await asyncio.gather(*tasks)

        df_primary = results[0]
        idx = 1
        df_htf = results[idx] if htf_task else None
        if htf_task:
            idx += 1
        df_ltf = results[idx] if ltf_task else None

        if df_primary is None or df_primary.empty:
            return JSONResponse(
                {"error": f"No data for {symbol} ({timeframe}). '{sym}' may not exist on MEXC spot."},
                status_code=404,
            )

        # Format candles for chart
        candles = []
        for _, row in df_primary.iterrows():
            candles.append({
                "time": int(row["open_time"].timestamp()),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
            })

        # Helper: convert zone times using a dataframe
        def _add_times(zones, df_ref):
            """Attach Unix timestamps to zone idx fields."""
            for z in zones:
                for key in ("idx", "start_idx", "end_idx", "mitigation_idx"):
                    val = z.get(key)
                    if val is not None and 0 <= val < len(df_ref):
                        z[key + "_time"] = int(df_ref.iloc[val]["open_time"].timestamp())
                # Also convert FVG idx
                fvg = z.get("fvg")
                if fvg and "idx" in fvg:
                    fidx = fvg["idx"]
                    if 0 <= fidx < len(df_ref):
                        fvg["time"] = int(df_ref.iloc[fidx]["open_time"].timestamp())

        # Run detection on primary TF
        primary_sd = _detect_supply_demand_zones(df_primary)
        _add_times(primary_sd["order_blocks"], df_primary)
        _add_times(primary_sd["structure_zones"], df_primary)
        _add_times(primary_sd.get("scored_zones", []), df_primary)

        # Add times to FVGs
        for fvg in primary_sd["fvgs"].get("bullish_fvgs", []):
            fidx = fvg.get("idx", 0)
            if 0 <= fidx < len(df_primary):
                fvg["time"] = int(df_primary.iloc[fidx]["open_time"].timestamp())
        for fvg in primary_sd["fvgs"].get("bearish_fvgs", []):
            fidx = fvg.get("idx", 0)
            if 0 <= fidx < len(df_primary):
                fvg["time"] = int(df_primary.iloc[fidx]["open_time"].timestamp())

        # Run detection on HTF
        htf_sd = None
        if df_htf is not None and not df_htf.empty:
            htf_sd = _detect_supply_demand_zones(df_htf)
            _add_times(htf_sd["order_blocks"], df_htf)
            _add_times(htf_sd["structure_zones"], df_htf)

        # Run detection on LTF
        ltf_sd = None
        if df_ltf is not None and not df_ltf.empty:
            ltf_sd = _detect_supply_demand_zones(df_ltf)
            _add_times(ltf_sd["order_blocks"], df_ltf)
            _add_times(ltf_sd["structure_zones"], df_ltf)

        # Format market structure pivots for chart overlay
        ms_data = primary_sd.get("ms_data", {})
        ms_pivots = []
        for h in ms_data.get("ms_highs", []):
            idx_val = h["idx"]
            if 0 <= idx_val < len(df_primary):
                ms_pivots.append({
                    "type": "high", "price": h["price"], "idx": idx_val,
                    "time": int(df_primary.iloc[idx_val]["open_time"].timestamp()),
                })
        for l in ms_data.get("ms_lows", []):
            idx_val = l["idx"]
            if 0 <= idx_val < len(df_primary):
                ms_pivots.append({
                    "type": "low", "price": l["price"], "idx": idx_val,
                    "time": int(df_primary.iloc[idx_val]["open_time"].timestamp()),
                })
        ms_pivots.sort(key=lambda x: x["idx"])

        bos_events = []
        for b in ms_data.get("bos_events", []):
            bidx = b.get("bos_idx", 0)
            if 0 <= bidx < len(df_primary):
                bos_events.append({
                    "type": b["type"],
                    "price": b["bos_price"],
                    "idx": bidx,
                    "time": int(df_primary.iloc[bidx]["open_time"].timestamp()),
                    "broken_level": b["broken_level"],
                    "quality": b.get("quality", "moderate"),
                })

        response = {
            "symbol": symbol,
            "timeframe": timeframe,
            "tf_map": tf_map,
            "candles": candles,
            "trend": primary_sd["trend"],
            "ms_pivots": ms_pivots,
            "bos_events": bos_events,
            "order_blocks": primary_sd["order_blocks"],
            "structure_zones": primary_sd["structure_zones"],
            "scored_zones": primary_sd.get("scored_zones", []),
            "fvgs": primary_sd["fvgs"],
            "ranges": primary_sd.get("ranges", []),
            "htf_zones": {
                "order_blocks": htf_sd["order_blocks"] if htf_sd else [],
                "structure_zones": htf_sd["structure_zones"] if htf_sd else [],
                "trend": htf_sd["trend"] if htf_sd else "neutral",
            },
            "ltf_zones": {
                "order_blocks": ltf_sd["order_blocks"] if ltf_sd else [],
                "structure_zones": ltf_sd["structure_zones"] if ltf_sd else [],
                "trend": ltf_sd["trend"] if ltf_sd else "neutral",
            },
            "debug": {
                "sym_resolved": sym,
                "primary_candles": len(candles),
                "htf_candles": len(df_htf) if df_htf is not None else 0,
                "ltf_candles": len(df_ltf) if df_ltf is not None else 0,
                "bullish_fvgs": len(primary_sd["fvgs"].get("bullish_fvgs", [])),
                "bearish_fvgs": len(primary_sd["fvgs"].get("bearish_fvgs", [])),
                "bullish_obs": len([z for z in primary_sd["order_blocks"] if z.get("type") == "bullish"]),
                "bearish_obs": len([z for z in primary_sd["order_blocks"] if z.get("type") == "bearish"]),
                "demand_ss": len([z for z in primary_sd["structure_zones"] if z.get("type") == "demand"]),
                "supply_ss": len([z for z in primary_sd["structure_zones"] if z.get("type") == "supply"]),
            },
        }

        return JSONResponse(convert_numpy_types(response))

    except Exception as e:
        logger.error(f"[SD_API_ERROR] {e}")
        import traceback
        tb = traceback.format_exc()
        logger.error(tb)
        return JSONResponse({"error": str(e), "traceback": tb}, status_code=500)


@app.get("/supply-demand", response_class=HTMLResponse)
async def supply_demand_page():
    """
    Standalone Supply & Demand debug page (TCT Lecture 3).
    Shows OBIFs and Structure S/D zones on candlestick chart
    with FVG markers, mitigation status, and zone scoring.
    """
    html = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
<title>Supply & Demand — HPB-TCT</title>
<script src="https://unpkg.com/lightweight-charts@4.1.0/dist/lightweight-charts.standalone.production.js"></script>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#fff;color:#333;font-family:'Segoe UI',sans-serif;overflow:hidden}
.header{display:flex;align-items:center;gap:10px;padding:6px 12px;background:#f8f9fa;border-bottom:1px solid #dee2e6;flex-wrap:wrap}
.header h1{font-size:.95rem;color:#333;white-space:nowrap;font-weight:600}
.header h1 .sub{color:#888;font-weight:400;font-size:.75rem;margin-left:5px}
.ctrl{display:flex;align-items:center;gap:5px}
.ctrl label{font-size:.7rem;color:#666}
.ctrl select{background:#fff;color:#333;border:1px solid #ced4da;border-radius:4px;padding:4px 8px;font-size:.75rem;min-width:100px}
.ctrl select:focus{outline:none;border-color:#007bff}
.ctrl button{background:#007bff;color:#fff;border:none;border-radius:4px;padding:5px 12px;font-size:.75rem;cursor:pointer;font-weight:500}
.ctrl button:hover{background:#0056b3}
.ctrl button:disabled{opacity:.5;cursor:not-allowed}
.ctrl .toggle-group{display:flex;gap:2px}
.ctrl .toggle-btn{background:#e9ecef;color:#666;border:1px solid #ced4da;border-radius:4px;padding:4px 8px;font-size:.65rem;cursor:pointer;transition:all .15s}
.ctrl .toggle-btn.active{background:#007bff;color:#fff;border-color:#007bff}
.chart-btns{display:flex;gap:4px;margin-left:8px}
.chart-btns button{background:#6c757d;padding:4px 8px;font-size:.65rem}
.chart-btns button:hover{background:#545b62}
.back-link{color:#666;text-decoration:none;font-size:.7rem;padding:3px 8px;border:1px solid #ced4da;border-radius:4px;margin-left:auto}
.back-link:hover{color:#333;border-color:#adb5bd}
.main-area{display:flex;height:calc(100vh - 80px)}
.chart-wrap{flex:1;position:relative;background:#fff}
.chart-container{width:100%;height:100%}
.info-panel{width:290px;background:#f8f9fa;border-left:1px solid #dee2e6;overflow-y:auto;padding:10px;font-size:.7rem}
.info-panel h3{color:#333;font-size:.8rem;margin:10px 0 6px;border-bottom:1px solid #dee2e6;padding-bottom:4px;font-weight:600}
.info-panel h3:first-child{margin-top:0}
.info-row{display:flex;justify-content:space-between;padding:2px 0;color:#666}
.info-row .val{color:#333;font-weight:600}
.info-row .val.bullish{color:#28a745}
.info-row .val.bearish{color:#dc3545}
.info-row .val.ranging{color:#ffc107}
.tag{display:inline-block;font-size:.6rem;padding:1px 5px;border-radius:3px;margin-left:4px}
.tag.good{background:rgba(40,167,69,.15);color:#28a745}
.tag.bad{background:rgba(220,53,69,.15);color:#dc3545}
.tag.moderate{background:rgba(255,193,7,.15);color:#856404}
.tag.confirmed{background:rgba(0,123,255,.15);color:#007bff}
.tag.mitigated{background:rgba(108,117,125,.15);color:#6c757d;text-decoration:line-through}
.tag.demand{background:rgba(38,166,154,.15);color:#26a69a}
.tag.supply{background:rgba(239,83,80,.15);color:#ef5350}
.zone-list{margin:4px 0}
.zone-item{padding:5px 6px;margin:3px 0;border-radius:3px;font-size:.65rem;border-left:3px solid #007bff}
.zone-item.demand-ob{background:rgba(38,166,154,.06);border-left-color:#26a69a}
.zone-item.supply-ob{background:rgba(239,83,80,.06);border-left-color:#ef5350}
.zone-item.demand-ss{background:rgba(38,166,154,.04);border-left-color:#1a9688}
.zone-item.supply-ss{background:rgba(239,83,80,.04);border-left-color:#d32f2f}
.zone-item.mitigated{opacity:.5}
.zone-item .zone-header{display:flex;justify-content:space-between;align-items:center;margin-bottom:2px}
.zone-item .zone-label{font-weight:600;font-size:.7rem}
.legend{display:flex;gap:14px;padding:5px 12px;background:#f8f9fa;border-top:1px solid #dee2e6;font-size:.65rem;color:#666;flex-wrap:wrap}
.legend-item{display:flex;align-items:center;gap:4px}
.legend-box{width:14px;height:10px}
.legend-line{width:16px;height:2px}
.loading-overlay{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;background:rgba(255,255,255,.9);z-index:100;font-size:.85rem;color:#666}
.loading-overlay.hidden{display:none}
.spinner{width:18px;height:18px;border:2px solid #dee2e6;border-top-color:#007bff;border-radius:50%;animation:spin .6s linear infinite;margin-right:8px}
@keyframes spin{to{transform:rotate(360deg)}}
@media(max-width:768px){
    .info-panel{display:none}
    .header{padding:4px 8px}
    .ctrl select{min-width:80px;font-size:.7rem;padding:3px 5px}
    .ctrl button{padding:4px 8px;font-size:.7rem}
    .chart-btns button{padding:3px 6px}
}
</style>
</head>
<body>

<div class="header">
    <h1>Supply &amp; Demand <span class="sub">Debug</span></h1>
    <div class="ctrl">
        <label>Pair</label>
        <select id="pairSelect"><option>Loading...</option></select>
    </div>
    <div class="ctrl">
        <label>TF</label>
        <select id="tfSelect">
            <option value="1m">1m</option>
            <option value="5m">5m</option>
            <option value="15m">15m</option>
            <option value="30m">30m</option>
            <option value="1h">1h</option>
            <option value="4h" selected>4h</option>
            <option value="1d">1D</option>
            <option value="1W">1W</option>
            <option value="1M">1M</option>
        </select>
    </div>
    <div class="ctrl">
        <label>Show</label>
        <div class="toggle-group">
            <button class="toggle-btn active" id="togOB" onclick="toggleLayer('ob')">OB</button>
            <button class="toggle-btn active" id="togSS" onclick="toggleLayer('ss')">SS/SD</button>
            <button class="toggle-btn" id="togFVG" onclick="toggleLayer('fvg')">FVG</button>
            <button class="toggle-btn" id="togMS" onclick="toggleLayer('ms')">MS</button>
            <button class="toggle-btn" id="togHTF" onclick="toggleLayer('htf')">HTF</button>
            <button class="toggle-btn" id="togMit" onclick="toggleLayer('mit')">Mitigated</button>
        </div>
    </div>
    <div class="ctrl">
        <button id="loadBtn" onclick="loadData()">Load</button>
    </div>
    <div class="chart-btns">
        <button onclick="resetChart()">Reset</button>
        <button onclick="fitChart()">Fit</button>
        <button onclick="scalePrice()">Scale Price</button>
        <button onclick="zoomIn()">+</button>
        <button onclick="zoomOut()">&minus;</button>
    </div>
    <a class="back-link" href="/dashboard">&larr; Back</a>
</div>

<div class="main-area">
    <div class="chart-wrap">
        <div id="chart" class="chart-container"></div>
        <div id="loadingOverlay" class="loading-overlay hidden">
            <div class="spinner"></div> Loading...
        </div>
    </div>
    <div id="infoPanel" class="info-panel">
        <h3>Select pair &amp; timeframe</h3>
        <div class="info-row"><span>Then click <b>Load</b></span></div>
    </div>
</div>

<div class="legend">
    <div class="legend-item"><div class="legend-box" style="background:rgba(38,166,154,.18);border:1.5px solid rgba(38,166,154,.6)"></div> Demand OB</div>
    <div class="legend-item"><div class="legend-box" style="background:rgba(239,83,80,.18);border:1.5px solid rgba(239,83,80,.6)"></div> Supply OB</div>
    <div class="legend-item"><div class="legend-box" style="background:rgba(38,166,154,.10);border:1.5px dashed rgba(38,166,154,.45)"></div> Str Demand</div>
    <div class="legend-item"><div class="legend-box" style="background:rgba(239,83,80,.10);border:1.5px dashed rgba(239,83,80,.45)"></div> Str Supply</div>
    <div class="legend-item"><div style="width:16px;height:4px;background:rgba(33,150,243,.25)"></div> Bullish FVG</div>
    <div class="legend-item"><div style="width:16px;height:4px;background:rgba(255,152,0,.25)"></div> Bearish FVG</div>
</div>

<script>
let chart, candleSeries;
let overlays = [];
const BASE = '';

// Toggle state — FVG + MS off by default for a clean initial view
let layers = { ob: true, ss: true, fvg: false, ms: false, htf: false, mit: false };
let lastData = null;

// Max zones drawn per category to avoid clutter
const MAX_OB_DRAWN = 6;
const MAX_SS_DRAWN = 6;

function toggleLayer(key) {
    layers[key] = !layers[key];
    const idMap = { ob:'togOB', ss:'togSS', fvg:'togFVG', ms:'togMS', htf:'togHTF', mit:'togMit' };
    const el = document.getElementById(idMap[key]);
    if (el) el.classList.toggle('active', layers[key]);
    if (lastData) redraw(lastData);
}

function initChart() {
    const el = document.getElementById('chart');
    chart = LightweightCharts.createChart(el, {
        width: el.clientWidth,
        height: el.clientHeight,
        layout: { background: { type: 'solid', color: '#ffffff' }, textColor: '#333', fontSize: 11 },
        grid: { vertLines: { color: '#f0f0f0' }, horzLines: { color: '#f0f0f0' } },
        crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
        rightPriceScale: { borderColor: '#dee2e6' },
        timeScale: { borderColor: '#dee2e6', timeVisible: true, secondsVisible: false },
        handleScroll: { mouseWheel: true, pressedMouseMove: true, horzTouchDrag: true, vertTouchDrag: true },
        handleScale: { axisPressedMouseMove: true, mouseWheel: true, pinch: true },
    });
    candleSeries = chart.addCandlestickSeries({
        upColor: '#26a69a', downColor: '#ef5350',
        borderUpColor: '#26a69a', borderDownColor: '#ef5350',
        wickUpColor: '#26a69a', wickDownColor: '#ef5350',
    });
    window.addEventListener('resize', () => {
        chart.applyOptions({ width: el.clientWidth, height: el.clientHeight });
    });
}

function resetChart() {
    candleSeries.applyOptions({ autoscaleInfoProvider: undefined });
    chart.timeScale().resetTimeScale();
    chart.priceScale('right').applyOptions({ autoScale: true });
}
function fitChart() { chart.timeScale().fitContent(); }
function zoomIn() {
    const vr = chart.timeScale().getVisibleLogicalRange();
    if (vr) {
        const c = (vr.from + vr.to) / 2, h = (vr.to - vr.from) / 4;
        chart.timeScale().setVisibleLogicalRange({ from: c - h, to: c + h });
    }
}
function zoomOut() {
    const vr = chart.timeScale().getVisibleLogicalRange();
    if (vr) {
        const c = (vr.from + vr.to) / 2, h = (vr.to - vr.from);
        chart.timeScale().setVisibleLogicalRange({ from: c - h, to: c + h });
    }
}
let lastLoadedCandles = [];
function scalePrice() {
    if (!lastLoadedCandles || lastLoadedCandles.length === 0) return;
    candleSeries.applyOptions({
        autoscaleInfoProvider: () => {
            const vr = chart.timeScale().getVisibleLogicalRange();
            if (!vr) return null;
            const fi = Math.max(0, Math.floor(vr.from));
            const ti = Math.min(lastLoadedCandles.length - 1, Math.ceil(vr.to));
            let minP = Infinity, maxP = -Infinity;
            for (let i = fi; i <= ti; i++) {
                const c = lastLoadedCandles[i];
                if (c) { if (c.low < minP) minP = c.low; if (c.high > maxP) maxP = c.high; }
            }
            if (minP === Infinity) return null;
            const pad = (maxP - minP) * 0.05;
            return { priceRange: { minValue: minP - pad, maxValue: maxP + pad } };
        }
    });
    chart.priceScale('right').applyOptions({ autoScale: true });
}
function fmtPrice(v) {
    if (v == null || isNaN(v)) return '\u2014';
    const a = Math.abs(v);
    if (a === 0) return '0';
    if (a >= 1) return v.toFixed(Math.min(4, Math.max(2, 4 - Math.floor(Math.log10(a)))));
    const d = Math.max(4, -Math.floor(Math.log10(a)) + 3);
    return v.toFixed(Math.min(d, 12));
}

async function loadPairs() {
    try {
        const r = await fetch(BASE + '/api/sd-pairs');
        const d = await r.json();
        const sel = document.getElementById('pairSelect');
        sel.innerHTML = '';
        (d.pairs || []).forEach(p => {
            const o = document.createElement('option');
            o.value = p; o.textContent = p;
            if (p === 'BTC_USDT_PERP') o.selected = true;
            sel.appendChild(o);
        });
    } catch(e) { console.error('Failed to load pairs', e); }
}

function clearOverlays() {
    overlays.forEach(s => { try { chart.removeSeries(s); } catch(e){} });
    overlays = [];
    candleSeries.setMarkers([]);
}

/* --- Zone drawing helpers --- */

// Draws a zone as top line + bottom line + subtle fill lines between them
function drawZone(startTime, endTime, top, bottom, borderColor, fillColor, lineWidth, lineStyle, candles) {
    const first = candles[0]?.time || 0;
    const last = candles[candles.length - 1]?.time || Infinity;
    const s = Math.max(startTime || first, first);
    const e = endTime ? Math.min(endTime, last) : last;
    if (s >= last) return;

    // Top border
    const topL = chart.addLineSeries({
        color: borderColor, lineWidth: lineWidth, lineStyle: lineStyle,
        crosshairMarkerVisible: false, lastValueVisible: false, priceLineVisible: false,
    });
    topL.setData([{ time: s, value: top }, { time: e, value: top }]);
    overlays.push(topL);

    // Bottom border
    const botL = chart.addLineSeries({
        color: borderColor, lineWidth: lineWidth, lineStyle: lineStyle,
        crosshairMarkerVisible: false, lastValueVisible: false, priceLineVisible: false,
    });
    botL.setData([{ time: s, value: bottom }, { time: e, value: bottom }]);
    overlays.push(botL);

    // Fill: 3 thin lines evenly spaced inside the zone to simulate shading
    const gap = top - bottom;
    if (gap > 0) {
        for (let f = 0.25; f <= 0.75; f += 0.25) {
            const fl = chart.addLineSeries({
                color: fillColor, lineWidth: 1, lineStyle: 0,
                crosshairMarkerVisible: false, lastValueVisible: false, priceLineVisible: false,
            });
            fl.setData([{ time: s, value: bottom + gap * f }, { time: e, value: bottom + gap * f }]);
            overlays.push(fl);
        }
    }
}

// Draw a thin FVG band (top + bottom edge, short extension)
function drawFVGBand(fvg, type, candles) {
    const first = candles[0]?.time || 0;
    const last = candles[candles.length - 1]?.time || Infinity;
    const fTime = fvg.time || first;
    if (fTime < first || fTime > last) return;

    const color = type === 'bullish' ? 'rgba(33,150,243,0.3)' : 'rgba(255,152,0,0.3)';
    // Extend FVG a limited distance (~60 candles) rather than to chart end
    let interval = 14400;
    if (candles.length >= 2) interval = candles[1].time - candles[0].time;
    const fvgEnd = Math.min(fTime + interval * 60, last);

    const tl = chart.addLineSeries({
        color: color, lineWidth: 1, lineStyle: 0,
        crosshairMarkerVisible: false, lastValueVisible: false, priceLineVisible: false,
    });
    tl.setData([{ time: fTime, value: fvg.top }, { time: fvgEnd, value: fvg.top }]);
    overlays.push(tl);

    const bl = chart.addLineSeries({
        color: color, lineWidth: 1, lineStyle: 0,
        crosshairMarkerVisible: false, lastValueVisible: false, priceLineVisible: false,
    });
    bl.setData([{ time: fTime, value: fvg.bottom }, { time: fvgEnd, value: fvg.bottom }]);
    overlays.push(bl);
}

/* --- Main redraw --- */

function redraw(data) {
    clearOverlays();
    const candles = data.candles || [];
    if (candles.length === 0) return;

    const first = candles[0].time;
    const last = candles[candles.length - 1].time;
    const markers = [];

    // Candle interval for zone extension
    let candleInterval = 14400;
    if (candles.length >= 2) candleInterval = candles[1].time - candles[0].time;
    // Unmitigated zones extend 50 candles past formation (not to chart edge)
    const ZONE_EXTEND = candleInterval * 50;

    // MS zigzag — off by default, opt-in via toggle
    if (layers.ms) {
        const pivots = (data.ms_pivots || []).filter(p => p.time >= first && p.time <= last).sort((a,b) => a.idx - b.idx);
        if (pivots.length >= 2) {
            const zz = chart.addLineSeries({
                color: 'rgba(0,0,0,0.2)', lineWidth: 1, lineStyle: 0,
                crosshairMarkerVisible: false, lastValueVisible: false, priceLineVisible: false,
            });
            zz.setData(pivots.map(p => ({ time: p.time, value: p.price })));
            overlays.push(zz);
        }
    }

    // -- Order Blocks (single candle) --
    if (layers.ob) {
        const obActive = (data.order_blocks || []).filter(z => !z.mitigated);
        const obMit = layers.mit ? (data.order_blocks || []).filter(z => z.mitigated) : [];
        const obDraw = obActive.slice(-MAX_OB_DRAWN);

        obDraw.forEach(ob => {
            const isDemand = ob.type === 'bullish';
            const border = isDemand ? 'rgba(38,166,154,0.65)' : 'rgba(239,83,80,0.65)';
            const fill   = isDemand ? 'rgba(38,166,154,0.08)' : 'rgba(239,83,80,0.08)';
            const sTime = ob.idx_time || first;
            const eTime = Math.min(sTime + ZONE_EXTEND, last);
            drawZone(sTime, eTime, ob.top, ob.bottom, border, fill, 2, 0, candles);
            markers.push({
                time: sTime,
                position: isDemand ? 'belowBar' : 'aboveBar',
                color: isDemand ? '#26a69a' : '#ef5350',
                shape: isDemand ? 'arrowUp' : 'arrowDown',
                size: 1, text: isDemand ? 'Demand OB' : 'Supply OB',
            });
        });

        // Mitigated OBs: faded dotted lines, end at mitigation point, no fill
        obMit.slice(-4).forEach(ob => {
            const isDemand = ob.type === 'bullish';
            const border = isDemand ? 'rgba(38,166,154,0.2)' : 'rgba(239,83,80,0.2)';
            const sTime = ob.idx_time || first;
            const eTime = ob.mitigation_idx_time || last;
            const tl = chart.addLineSeries({
                color: border, lineWidth: 1, lineStyle: LightweightCharts.LineStyle.Dotted,
                crosshairMarkerVisible: false, lastValueVisible: false, priceLineVisible: false,
            });
            tl.setData([{ time: Math.max(sTime, first), value: ob.top }, { time: Math.min(eTime, last), value: ob.top }]);
            overlays.push(tl);
            const bl = chart.addLineSeries({
                color: border, lineWidth: 1, lineStyle: LightweightCharts.LineStyle.Dotted,
                crosshairMarkerVisible: false, lastValueVisible: false, priceLineVisible: false,
            });
            bl.setData([{ time: Math.max(sTime, first), value: ob.bottom }, { time: Math.min(eTime, last), value: ob.bottom }]);
            overlays.push(bl);
        });
    }

    // -- Structure S/D zones (multi-candle) --
    if (layers.ss) {
        const ssActive = (data.structure_zones || []).filter(z => !z.mitigated);
        const ssMit = layers.mit ? (data.structure_zones || []).filter(z => z.mitigated) : [];
        const ssDraw = ssActive.slice(-MAX_SS_DRAWN);

        ssDraw.forEach(zone => {
            const isDemand = zone.type === 'demand';
            const border = isDemand ? 'rgba(38,166,154,0.45)' : 'rgba(239,83,80,0.45)';
            const fill   = isDemand ? 'rgba(38,166,154,0.05)' : 'rgba(239,83,80,0.05)';
            const sTime = zone.start_idx_time || first;
            const eTime = Math.min(sTime + ZONE_EXTEND, last);
            drawZone(sTime, eTime, zone.top, zone.bottom, border, fill, 1.5,
                LightweightCharts.LineStyle.Dashed, candles);
            markers.push({
                time: sTime,
                position: isDemand ? 'belowBar' : 'aboveBar',
                color: isDemand ? '#1a9688' : '#d32f2f',
                shape: isDemand ? 'arrowUp' : 'arrowDown',
                size: 0.8, text: isDemand ? 'Str Demand' : 'Str Supply',
            });
        });

        ssMit.slice(-4).forEach(zone => {
            const isDemand = zone.type === 'demand';
            const border = isDemand ? 'rgba(38,166,154,0.15)' : 'rgba(239,83,80,0.15)';
            const sTime = zone.start_idx_time || first;
            const eTime = zone.mitigation_idx_time || last;
            const tl = chart.addLineSeries({
                color: border, lineWidth: 1, lineStyle: LightweightCharts.LineStyle.Dotted,
                crosshairMarkerVisible: false, lastValueVisible: false, priceLineVisible: false,
            });
            tl.setData([{ time: Math.max(sTime, first), value: zone.top }, { time: Math.min(eTime, last), value: zone.top }]);
            overlays.push(tl);
            const bl = chart.addLineSeries({
                color: border, lineWidth: 1, lineStyle: LightweightCharts.LineStyle.Dotted,
                crosshairMarkerVisible: false, lastValueVisible: false, priceLineVisible: false,
            });
            bl.setData([{ time: Math.max(sTime, first), value: zone.bottom }, { time: Math.min(eTime, last), value: zone.bottom }]);
            overlays.push(bl);
        });
    }

    // -- HTF zones (wider, transparent, behind) --
    if (layers.htf) {
        const htfAll = [
            ...(data.htf_zones?.order_blocks || []).map(z => ({...z, _class: 'ob'})),
            ...(data.htf_zones?.structure_zones || []).map(z => ({...z, _class: 'ss'})),
        ].filter(z => !z.mitigated || layers.mit);

        htfAll.slice(-6).forEach(zone => {
            const isDemand = zone.type === 'bullish' || zone.type === 'demand';
            const border = isDemand ? 'rgba(38,166,154,0.25)' : 'rgba(239,83,80,0.25)';
            const fill   = isDemand ? 'rgba(38,166,154,0.04)' : 'rgba(239,83,80,0.04)';
            const sTime = zone.idx_time || zone.start_idx_time || first;
            const style = zone._class === 'ob' ? 0 : LightweightCharts.LineStyle.Dashed;
            drawZone(sTime, last, zone.top, zone.bottom, border, fill, 2.5, style, candles);
        });
    }

    // -- FVGs: only those attached to drawn zones, as short bands --
    if (layers.fvg) {
        const drawnFvgIdx = new Set();
        (data.order_blocks || []).filter(z => !z.mitigated).slice(-MAX_OB_DRAWN).forEach(z => {
            if (z.fvg?.idx != null) drawnFvgIdx.add(z.fvg.idx);
        });
        (data.structure_zones || []).filter(z => !z.mitigated).slice(-MAX_SS_DRAWN).forEach(z => {
            if (z.fvg?.idx != null) drawnFvgIdx.add(z.fvg.idx);
        });

        (data.fvgs?.bullish_fvgs || []).forEach(fvg => {
            if (drawnFvgIdx.has(fvg.idx)) drawFVGBand(fvg, 'bullish', candles);
        });
        (data.fvgs?.bearish_fvgs || []).forEach(fvg => {
            if (drawnFvgIdx.has(fvg.idx)) drawFVGBand(fvg, 'bearish', candles);
        });
    }

    // -- Markers: deduplicate, one per time+position slot --
    if (markers.length > 0) {
        const seen = new Set();
        const unique = markers.filter(m => {
            const key = m.time + '_' + m.position;
            if (seen.has(key)) return false;
            seen.add(key);
            return true;
        });
        unique.sort((a, b) => a.time - b.time);
        candleSeries.setMarkers(unique);
    }
}

async function loadData() {
    const sym = document.getElementById('pairSelect').value;
    const tf = document.getElementById('tfSelect').value;
    const btn = document.getElementById('loadBtn');
    const overlay = document.getElementById('loadingOverlay');

    btn.disabled = true;
    overlay.classList.remove('hidden');
    clearOverlays();

    try {
        const r = await fetch(BASE + `/api/supply-demand?symbol=${sym}&timeframe=${tf}&limit=300`);
        if (!r.ok) {
            const errData = await r.json().catch(() => ({}));
            throw new Error(errData.error || `HTTP ${r.status}`);
        }
        const data = await r.json();
        lastData = data;

        lastLoadedCandles = data.candles || [];
        if (lastLoadedCandles.length > 0) {
            const sp = Math.abs(lastLoadedCandles[lastLoadedCandles.length - 1].close);
            let prec = 2;
            if (sp > 0 && sp < 1) prec = Math.max(4, -Math.floor(Math.log10(sp)) + 3);
            else if (sp >= 1 && sp < 10) prec = 4;
            else if (sp >= 10 && sp < 1000) prec = 2;
            prec = Math.min(prec, 12);
            candleSeries.applyOptions({ priceFormat: { type: 'price', precision: prec, minMove: Math.pow(10, -prec) } });
        }
        candleSeries.applyOptions({ autoscaleInfoProvider: undefined });
        candleSeries.setData(lastLoadedCandles);

        redraw(data);
        updateInfoPanel(data);
        chart.timeScale().fitContent();
        scalePrice();

    } catch(e) {
        console.error('Load error:', e);
        document.getElementById('infoPanel').innerHTML =
            `<h3 style="color:#dc3545">Error</h3><div class="info-row">${e.message}</div>`;
    } finally {
        btn.disabled = false;
        overlay.classList.add('hidden');
    }
}

function updateInfoPanel(data) {
    const panel = document.getElementById('infoPanel');
    const sym = data.symbol || '\u2014';
    const tf = data.timeframe || '\u2014';
    const tfMap = data.tf_map || {};
    const dbg = data.debug || {};

    let html = `<h3>${sym} \u00b7 ${tf.toUpperCase()}</h3>`;

    const trend = data.trend || 'neutral';
    const tc = trend === 'bullish' ? 'bullish' : trend === 'bearish' ? 'bearish' : 'ranging';
    html += `<div class="info-row"><span>Trend</span><span class="val ${tc}">${trend.toUpperCase()}</span></div>`;

    html += `<div class="info-row"><span>HTF</span><span class="val">${tfMap.htf || '\u2014'}</span></div>`;
    html += `<div class="info-row"><span>MTF</span><span class="val">${tfMap.mtf || tf}</span></div>`;
    html += `<div class="info-row"><span>LTF</span><span class="val">${tfMap.ltf || '\u2014'}</span></div>`;

    html += `<h3>Detection Summary</h3>`;
    html += `<div class="info-row"><span>Bullish FVGs</span><span class="val">${dbg.bullish_fvgs || 0}</span></div>`;
    html += `<div class="info-row"><span>Bearish FVGs</span><span class="val">${dbg.bearish_fvgs || 0}</span></div>`;
    html += `<div class="info-row"><span>Demand OBs</span><span class="val bullish">${dbg.bullish_obs || 0}</span></div>`;
    html += `<div class="info-row"><span>Supply OBs</span><span class="val bearish">${dbg.bearish_obs || 0}</span></div>`;
    html += `<div class="info-row"><span>Structure Demand</span><span class="val bullish">${dbg.demand_ss || 0}</span></div>`;
    html += `<div class="info-row"><span>Structure Supply</span><span class="val bearish">${dbg.supply_ss || 0}</span></div>`;

    const obs = data.order_blocks || [];
    const activeOBs = obs.filter(z => !z.mitigated);
    const mitOBs = obs.filter(z => z.mitigated);
    html += `<h3>Order Blocks (${activeOBs.length} active, ${mitOBs.length} mit)</h3>`;
    if (activeOBs.length > 0) {
        html += `<div class="zone-list">`;
        activeOBs.slice(-8).forEach(ob => {
            const isDemand = ob.type === 'bullish';
            const cls = isDemand ? 'demand-ob' : 'supply-ob';
            const label = isDemand ? 'Demand OBIF' : 'Supply OBIF';
            const pq = ob.pivot_quality || {};
            const qlabel = pq.quality_label || '';
            html += `<div class="zone-item ${cls}">
                <div class="zone-header">
                    <span class="zone-label">${label}</span>
                    <span class="tag ${isDemand ? 'demand' : 'supply'}">${isDemand ? 'BUY' : 'SELL'}</span>
                </div>
                <div>Top: ${fmtPrice(ob.top)} | Bottom: ${fmtPrice(ob.bottom)}</div>
                <div>FVG gap: ${fmtPrice(ob.fvg?.gap_size || 0)}</div>
                ${ob.location_type ? '<div>Location: ' + ob.location_type + ' <span class="tag good">score ' + (ob.strength || 0).toFixed(0) + '</span></div>' : ''}
                ${qlabel ? '<div>Quality: ' + qlabel.replace(/_/g,' ') + '</div>' : ''}
            </div>`;
        });
        html += `</div>`;
    }

    const szones = data.structure_zones || [];
    const activeSZ = szones.filter(z => !z.mitigated);
    const mitSZ = szones.filter(z => z.mitigated);
    html += `<h3>Structure S/D (${activeSZ.length} active, ${mitSZ.length} mit)</h3>`;
    if (activeSZ.length > 0) {
        html += `<div class="zone-list">`;
        activeSZ.slice(-8).forEach(zone => {
            const isDemand = zone.type === 'demand';
            const cls = isDemand ? 'demand-ss' : 'supply-ss';
            const label = isDemand ? 'Structure Demand' : 'Structure Supply';
            html += `<div class="zone-item ${cls}">
                <div class="zone-header">
                    <span class="zone-label">${label}</span>
                    <span class="tag ${isDemand ? 'demand' : 'supply'}">${zone.candle_count || '?'} candles</span>
                </div>
                <div>Top: ${fmtPrice(zone.top)} | Bottom: ${fmtPrice(zone.bottom)}</div>
                <div>MS: <span class="tag ${zone.ms_quality === 'confirmed' ? 'confirmed' : 'moderate'}">${zone.ms_quality || 'inferred'}</span>
                ${zone.has_bos_support ? '<span class="tag good">BOS</span>' : ''}
                ${zone.eof_aligned ? '<span class="tag good">EOF</span>' : ''}</div>
            </div>`;
        });
        html += `</div>`;
    }

    const scored = data.scored_zones || [];
    if (scored.length > 0) {
        html += `<h3>Top Scored Zones</h3>`;
        html += `<div class="zone-list">`;
        scored.slice(0, 5).forEach(z => {
            const isDemand = z.type === 'bullish' || z.type === 'demand';
            const zClass = z.zone_class || 'unknown';
            const cls = isDemand ? 'demand-ob' : 'supply-ob';
            html += `<div class="zone-item ${cls}">
                <div class="zone-header">
                    <span class="zone-label">${isDemand ? 'Demand' : 'Supply'} (${zClass})</span>
                    <span class="tag good">${(z.strength || 0).toFixed(0)}%</span>
                </div>
                <div>${fmtPrice(z.top)} - ${fmtPrice(z.bottom)}</div>
                <div>Location: ${z.location_type || 'none'}${z.pivot_quality?.has_liquidity_sweep ? ' | Liq Sweep' : ''}</div>
            </div>`;
        });
        html += `</div>`;
    }

    const htfZones = data.htf_zones || {};
    const htfOBCount = (htfZones.order_blocks || []).length;
    const htfSSCount = (htfZones.structure_zones || []).length;
    html += `<h3>HTF Context (${tfMap.htf || '\u2014'})</h3>`;
    html += `<div class="info-row"><span>HTF Trend</span><span class="val ${htfZones.trend === 'bullish' ? 'bullish' : htfZones.trend === 'bearish' ? 'bearish' : 'ranging'}">${(htfZones.trend || 'neutral').toUpperCase()}</span></div>`;
    html += `<div class="info-row"><span>HTF OBs</span><span class="val">${htfOBCount}</span></div>`;
    html += `<div class="info-row"><span>HTF Str S/D</span><span class="val">${htfSSCount}</span></div>`;

    html += `<h3>TCT S/D Rules</h3>`;
    html += `<div class="info-row" style="flex-direction:column;gap:2px">
        <div>\u2022 OB = last opposing candle before expansion</div>
        <div>\u2022 MUST have FVG (inefficiency) to be valid</div>
        <div>\u2022 SS/SD = multi-candle zone with FVG</div>
        <div>\u2022 3 locations: pivot, in-range, deviation</div>
        <div>\u2022 Mitigated zone = no longer valid</div>
        <div>\u2022 HTF block + LTF true inefficiency = best</div>
    </div>`;

    panel.innerHTML = html;
}

initChart();
loadPairs();
document.getElementById('pairSelect').addEventListener('change', loadData);
document.getElementById('tfSelect').addEventListener('change', loadData);
</script>
</body>
</html>"""

    return HTMLResponse(content=html)


# ================================================================
# LIQUIDITY DEBUG PAGE (TCT Lecture 4)
# ================================================================

@app.get("/api/liq-pairs")
async def get_liq_pairs():
    """Return pairs for the liquidity page dropdown."""
    pairs_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mexc_all_pairs.txt")
    try:
        with open(pairs_path, "r") as f:
            pairs = [line.strip() for line in f if line.strip()]
        return {"pairs": pairs}
    except FileNotFoundError:
        return {"pairs": COIN_LIST}


@app.get("/api/liquidity-chart")
async def get_liquidity_chart_data(symbol: str = "BTC_USDT_PERP", timeframe: str = "4h", limit: int = 300):
    """
    API endpoint for Liquidity debug page (TCT Lecture 4).
    Returns candles + BSL/SSL pools + liquidity curves + extreme targets + voids
    with Unix timestamps for chart rendering.
    """
    try:
        sym = resolve_symbol(symbol)
        if "_PERP" in sym:
            sym = sym.replace("_PERP", "").replace("_", "")

        tf_map = _MS_TF_MAP.get(timeframe, {"ltf": timeframe, "mtf": timeframe, "htf": timeframe})

        df_primary = await fetch_mexc_candles(sym, timeframe, min(limit, 500))

        if df_primary is None or df_primary.empty:
            return JSONResponse(
                {"error": f"No data for {symbol} ({timeframe}). '{sym}' may not exist on MEXC spot."},
                status_code=404,
            )

        # Format candles for chart
        candles = []
        for _, row in df_primary.iterrows():
            candles.append({
                "time": int(row["open_time"].timestamp()),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
            })

        current_price = float(df_primary.iloc[-1]["close"])

        # Market structure detection
        from market_structure import MarketStructure as MS
        ms_data = MS.detect_pivots(df_primary)
        trend = ms_data.get("trend", "neutral")

        # Detect FVGs
        fvgs = FairValueGap.detect_fvgs(df_primary)

        # Detect Order Blocks and Structure S/D (needed for void detection)
        obs = OrderBlock.detect_order_blocks(df_primary, fvgs)
        ss_sd = StructureSupplyDemand.detect_structure_zones(df_primary, fvgs, ms_data)

        all_zones = (
            obs.get("bullish_obs", []) +
            obs.get("bearish_obs", []) +
            ss_sd.get("demand_zones", []) +
            ss_sd.get("supply_zones", [])
        )

        # Liquidity detection
        liq_detector = LiquidityDetector()
        liq_pools = liq_detector.detect_liquidity_pools(df_primary, ms_data, current_price)

        # Liquidity curves
        curve_gen = LiquidityCurveGenerator()
        liq_curves = curve_gen.generate_curves(liq_pools, df_primary)

        # Extreme liquidity targets
        target_det = ExtremeLiquidityTarget()
        liq_targets = target_det.identify_extreme_targets(liq_curves, df_primary)

        # Check if targets are swept
        for target in liq_targets.get("sell_side_targets", []) + liq_targets.get("buy_side_targets", []):
            tp = target["target_price"]
            if target["type"] == "distribution":
                target["is_swept"] = current_price < tp
            else:
                target["is_swept"] = current_price > tp

        # Liquidity voids
        void_det = LiquidityVoidDetector()
        liq_voids = void_det.detect_voids(all_zones, df_primary, current_price)

        # Helper: convert idx to Unix timestamps
        def _pool_with_time(pool):
            idx = pool.get("idx", 0)
            if 0 <= idx < len(df_primary):
                pool["time"] = int(df_primary.iloc[idx]["open_time"].timestamp())
            return pool

        bsl_pools = [_pool_with_time(p) for p in liq_pools.get("bsl_pools", [])]
        ssl_pools = [_pool_with_time(p) for p in liq_pools.get("ssl_pools", [])]

        # Convert curve tap indices to timestamps
        def _curves_with_times(curves):
            for curve in curves:
                for tap in curve.get("taps", []):
                    idx = tap.get("idx", 0)
                    if 0 <= idx < len(df_primary):
                        tap["time"] = int(df_primary.iloc[idx]["open_time"].timestamp())
            return curves

        sell_curves = _curves_with_times(liq_curves.get("sell_side_curves", []))
        buy_curves = _curves_with_times(liq_curves.get("buy_side_curves", []))

        # Convert extreme target indices to timestamps
        def _targets_with_times(targets):
            for t in targets:
                idx = t.get("target_idx", 0)
                if 0 <= idx < len(df_primary):
                    t["time"] = int(df_primary.iloc[idx]["open_time"].timestamp())
            return targets

        sell_targets = _targets_with_times(liq_targets.get("sell_side_targets", []))
        buy_targets = _targets_with_times(liq_targets.get("buy_side_targets", []))

        # Format MS pivots for chart
        ms_pivots = []
        for h in ms_data.get("ms_highs", []):
            idx_val = h["idx"]
            if 0 <= idx_val < len(df_primary):
                ms_pivots.append({
                    "type": "high", "price": h["price"], "idx": idx_val,
                    "time": int(df_primary.iloc[idx_val]["open_time"].timestamp()),
                })
        for l in ms_data.get("ms_lows", []):
            idx_val = l["idx"]
            if 0 <= idx_val < len(df_primary):
                ms_pivots.append({
                    "type": "low", "price": l["price"], "idx": idx_val,
                    "time": int(df_primary.iloc[idx_val]["open_time"].timestamp()),
                })
        ms_pivots.sort(key=lambda x: x["idx"])

        # Equal highs/lows
        equal_highs = liq_pools.get("equal_highs", [])
        equal_lows = liq_pools.get("equal_lows", [])

        response = {
            "symbol": symbol,
            "timeframe": timeframe,
            "tf_map": tf_map,
            "candles": candles,
            "current_price": current_price,
            "trend": trend,
            "ms_pivots": ms_pivots,
            "bsl_pools": bsl_pools,
            "ssl_pools": ssl_pools,
            "equal_highs": equal_highs,
            "equal_lows": equal_lows,
            "sell_side_curves": sell_curves,
            "buy_side_curves": buy_curves,
            "sell_side_targets": sell_targets,
            "buy_side_targets": buy_targets,
            "voids": liq_voids.get("voids", []),
            "summary": {
                "total_bsl": len(bsl_pools),
                "total_ssl": len(ssl_pools),
                "primary_highs": liq_pools.get("primary_highs_count", 0),
                "primary_lows": liq_pools.get("primary_lows_count", 0),
                "internal_highs": liq_pools.get("internal_highs_count", 0),
                "internal_lows": liq_pools.get("internal_lows_count", 0),
                "equal_highs": len(equal_highs),
                "equal_lows": len(equal_lows),
                "sell_curves": len(sell_curves),
                "buy_curves": len(buy_curves),
                "sell_targets": len(sell_targets),
                "buy_targets": len(buy_targets),
                "voids": liq_voids.get("total_voids", 0),
            },
        }

        return JSONResponse(convert_numpy_types(response))

    except Exception as e:
        logger.error(f"[LIQ_CHART_ERROR] {e}")
        import traceback
        tb = traceback.format_exc()
        logger.error(tb)
        return JSONResponse({"error": str(e), "traceback": tb}, status_code=500)


@app.get("/liquidity", response_class=HTMLResponse)
async def liquidity_page():
    """
    Standalone Liquidity debug page (TCT Lecture 4).
    Shows BSL/SSL pools, liquidity curves, extreme targets, and voids
    on a candlestick chart for debugging the liquidity detection algo.
    """
    html = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
<title>Liquidity — HPB-TCT</title>
<script src="https://unpkg.com/lightweight-charts@4.1.0/dist/lightweight-charts.standalone.production.js"></script>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#fff;color:#333;font-family:'Segoe UI',sans-serif;overflow:hidden}
.header{display:flex;align-items:center;gap:10px;padding:6px 12px;background:#f8f9fa;border-bottom:1px solid #dee2e6;flex-wrap:wrap}
.header h1{font-size:.95rem;color:#333;white-space:nowrap;font-weight:600}
.header h1 .sub{color:#888;font-weight:400;font-size:.75rem;margin-left:5px}
.ctrl{display:flex;align-items:center;gap:5px}
.ctrl label{font-size:.7rem;color:#666}
.ctrl select{background:#fff;color:#333;border:1px solid #ced4da;border-radius:4px;padding:4px 8px;font-size:.75rem;min-width:100px}
.ctrl select:focus{outline:none;border-color:#007bff}
.ctrl button{background:#007bff;color:#fff;border:none;border-radius:4px;padding:5px 12px;font-size:.75rem;cursor:pointer;font-weight:500}
.ctrl button:hover{background:#0056b3}
.ctrl button:disabled{opacity:.5;cursor:not-allowed}
.ctrl .toggle-group{display:flex;gap:2px}
.ctrl .toggle-btn{background:#e9ecef;color:#666;border:1px solid #ced4da;border-radius:4px;padding:4px 8px;font-size:.65rem;cursor:pointer;transition:all .15s}
.ctrl .toggle-btn.active{background:#007bff;color:#fff;border-color:#007bff}
.chart-btns{display:flex;gap:4px;margin-left:8px}
.chart-btns button{background:#6c757d;padding:4px 8px;font-size:.65rem}
.chart-btns button:hover{background:#545b62}
.back-link{color:#666;text-decoration:none;font-size:.7rem;padding:3px 8px;border:1px solid #ced4da;border-radius:4px;margin-left:auto}
.back-link:hover{color:#333;border-color:#adb5bd}
.main-area{display:flex;height:calc(100vh - 80px)}
.chart-wrap{flex:1;position:relative;background:#fff}
.chart-container{width:100%;height:100%}
.info-panel{width:290px;background:#f8f9fa;border-left:1px solid #dee2e6;overflow-y:auto;padding:10px;font-size:.7rem}
.info-panel h3{color:#333;font-size:.8rem;margin:10px 0 6px;border-bottom:1px solid #dee2e6;padding-bottom:4px;font-weight:600}
.info-panel h3:first-child{margin-top:0}
.info-row{display:flex;justify-content:space-between;padding:2px 0;color:#666}
.info-row .val{color:#333;font-weight:600}
.info-row .val.bullish{color:#28a745}
.info-row .val.bearish{color:#dc3545}
.info-row .val.ranging{color:#ffc107}
.tag{display:inline-block;font-size:.6rem;padding:1px 5px;border-radius:3px;margin-left:4px}
.tag.good{background:rgba(40,167,69,.15);color:#28a745}
.tag.bad{background:rgba(220,53,69,.15);color:#dc3545}
.tag.moderate{background:rgba(255,193,7,.15);color:#856404}
.tag.primary{background:rgba(0,123,255,.15);color:#007bff}
.tag.internal{background:rgba(108,117,125,.15);color:#6c757d}
.tag.equal{background:rgba(156,39,176,.15);color:#9c27b0}
.tag.swept{background:rgba(220,53,69,.15);color:#dc3545;text-decoration:line-through}
.pool-list{margin:4px 0}
.pool-item{padding:5px 6px;margin:3px 0;border-radius:3px;font-size:.65rem;border-left:3px solid #007bff}
.pool-item.bsl{background:rgba(33,150,243,.06);border-left-color:#2196f3}
.pool-item.ssl{background:rgba(255,152,0,.06);border-left-color:#ff9800}
.pool-item.curve{background:rgba(156,39,176,.06);border-left-color:#9c27b0}
.pool-item.target{background:rgba(76,175,80,.06);border-left-color:#4caf50}
.pool-item.void{background:rgba(158,158,158,.06);border-left-color:#9e9e9e}
.pool-item .pool-header{display:flex;justify-content:space-between;align-items:center;margin-bottom:2px}
.pool-item .pool-label{font-weight:600;font-size:.7rem}
.legend{display:flex;gap:14px;padding:5px 12px;background:#f8f9fa;border-top:1px solid #dee2e6;font-size:.65rem;color:#666;flex-wrap:wrap}
.legend-item{display:flex;align-items:center;gap:4px}
.legend-box{width:14px;height:10px}
.legend-line{width:16px;height:2px}
.loading-overlay{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;background:rgba(255,255,255,.9);z-index:100;font-size:.85rem;color:#666}
.loading-overlay.hidden{display:none}
.spinner{width:18px;height:18px;border:2px solid #dee2e6;border-top-color:#007bff;border-radius:50%;animation:spin .6s linear infinite;margin-right:8px}
@keyframes spin{to{transform:rotate(360deg)}}
@media(max-width:768px){
    .info-panel{display:none}
    .header{padding:4px 8px}
    .ctrl select{min-width:80px;font-size:.7rem;padding:3px 5px}
    .ctrl button{padding:4px 8px;font-size:.7rem}
    .chart-btns button{padding:3px 6px}
}
</style>
</head>
<body>

<div class="header">
    <h1>Liquidity <span class="sub">Debug &middot; TCT Lecture 4</span></h1>
    <div class="ctrl">
        <label>Pair</label>
        <select id="pairSelect"><option>Loading...</option></select>
    </div>
    <div class="ctrl">
        <label>TF</label>
        <select id="tfSelect">
            <option value="1m">1m</option>
            <option value="5m">5m</option>
            <option value="15m">15m</option>
            <option value="30m">30m</option>
            <option value="1h">1h</option>
            <option value="4h" selected>4h</option>
            <option value="1d">1D</option>
            <option value="1W">1W</option>
            <option value="1M">1M</option>
        </select>
    </div>
    <div class="ctrl">
        <label>Show</label>
        <div class="toggle-group">
            <button class="toggle-btn active" id="togBSL" onclick="toggleLayer('bsl')">BSL</button>
            <button class="toggle-btn active" id="togSSL" onclick="toggleLayer('ssl')">SSL</button>
            <button class="toggle-btn active" id="togCurves" onclick="toggleLayer('curves')">Curves</button>
            <button class="toggle-btn active" id="togTargets" onclick="toggleLayer('targets')">Targets</button>
            <button class="toggle-btn" id="togVoids" onclick="toggleLayer('voids')">Voids</button>
            <button class="toggle-btn" id="togMS" onclick="toggleLayer('ms')">MS</button>
            <button class="toggle-btn" id="togEQ" onclick="toggleLayer('eq')">EQ H/L</button>
        </div>
    </div>
    <div class="ctrl">
        <button id="loadBtn" onclick="loadData()">Load</button>
    </div>
    <div class="chart-btns">
        <button onclick="resetChart()">Reset</button>
        <button onclick="fitChart()">Fit</button>
        <button onclick="scalePrice()">Scale Price</button>
        <button onclick="zoomIn()">+</button>
        <button onclick="zoomOut()">&minus;</button>
    </div>
    <a class="back-link" href="/dashboard">&larr; Back</a>
</div>

<div class="main-area">
    <div class="chart-wrap">
        <div id="chart" class="chart-container"></div>
        <div id="loadingOverlay" class="loading-overlay hidden">
            <div class="spinner"></div> Loading...
        </div>
    </div>
    <div id="infoPanel" class="info-panel">
        <h3>Select pair &amp; timeframe</h3>
        <div class="info-row"><span>Then click <b>Load</b></span></div>
    </div>
</div>

<div class="legend">
    <div class="legend-item"><div class="legend-line" style="background:#2196f3"></div> BSL Primary</div>
    <div class="legend-item"><div class="legend-line" style="background:rgba(33,150,243,.4)"></div> BSL Internal</div>
    <div class="legend-item"><div class="legend-line" style="background:#ff9800"></div> SSL Primary</div>
    <div class="legend-item"><div class="legend-line" style="background:rgba(255,152,0,.4)"></div> SSL Internal</div>
    <div class="legend-item"><div class="legend-line" style="background:#9c27b0;height:3px"></div> Liquidity Curve</div>
    <div class="legend-item"><div class="legend-box" style="background:rgba(76,175,80,.3);border:1px solid #4caf50"></div> Extreme Target</div>
    <div class="legend-item"><div class="legend-box" style="background:rgba(158,158,158,.15);border:1px dashed #9e9e9e"></div> Void</div>
    <div class="legend-item"><div class="legend-line" style="background:#9c27b0;height:3px;border-top:1px dashed #9c27b0"></div> Equal H/L</div>
</div>

<script>
let chart, candleSeries;
let overlays = [];
const BASE = '';

let layers = { bsl: true, ssl: true, curves: true, targets: true, voids: false, ms: false, eq: false };
let lastData = null;

function toggleLayer(key) {
    layers[key] = !layers[key];
    const idMap = { bsl:'togBSL', ssl:'togSSL', curves:'togCurves', targets:'togTargets', voids:'togVoids', ms:'togMS', eq:'togEQ' };
    const el = document.getElementById(idMap[key]);
    if (el) el.classList.toggle('active', layers[key]);
    if (lastData) redraw(lastData);
}

function initChart() {
    const el = document.getElementById('chart');
    chart = LightweightCharts.createChart(el, {
        width: el.clientWidth,
        height: el.clientHeight,
        layout: { background: { type: 'solid', color: '#ffffff' }, textColor: '#333', fontSize: 11 },
        grid: { vertLines: { color: '#f0f0f0' }, horzLines: { color: '#f0f0f0' } },
        crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
        rightPriceScale: { borderColor: '#dee2e6' },
        timeScale: { borderColor: '#dee2e6', timeVisible: true, secondsVisible: false },
        handleScroll: { mouseWheel: true, pressedMouseMove: true, horzTouchDrag: true, vertTouchDrag: true },
        handleScale: { axisPressedMouseMove: true, mouseWheel: true, pinch: true },
    });
    candleSeries = chart.addCandlestickSeries({
        upColor: '#26a69a', downColor: '#ef5350',
        borderUpColor: '#26a69a', borderDownColor: '#ef5350',
        wickUpColor: '#26a69a', wickDownColor: '#ef5350',
    });
    window.addEventListener('resize', () => {
        chart.applyOptions({ width: el.clientWidth, height: el.clientHeight });
    });
}

function resetChart() {
    candleSeries.applyOptions({ autoscaleInfoProvider: undefined });
    chart.timeScale().resetTimeScale();
    chart.priceScale('right').applyOptions({ autoScale: true });
}
function fitChart() { chart.timeScale().fitContent(); }
function zoomIn() {
    const vr = chart.timeScale().getVisibleLogicalRange();
    if (vr) {
        const c = (vr.from + vr.to) / 2, h = (vr.to - vr.from) / 4;
        chart.timeScale().setVisibleLogicalRange({ from: c - h, to: c + h });
    }
}
function zoomOut() {
    const vr = chart.timeScale().getVisibleLogicalRange();
    if (vr) {
        const c = (vr.from + vr.to) / 2, h = (vr.to - vr.from);
        chart.timeScale().setVisibleLogicalRange({ from: c - h, to: c + h });
    }
}
let lastLoadedCandles = [];
function scalePrice() {
    if (!lastLoadedCandles || lastLoadedCandles.length === 0) return;
    candleSeries.applyOptions({
        autoscaleInfoProvider: () => {
            const vr = chart.timeScale().getVisibleLogicalRange();
            if (!vr) return null;
            const fi = Math.max(0, Math.floor(vr.from));
            const ti = Math.min(lastLoadedCandles.length - 1, Math.ceil(vr.to));
            let minP = Infinity, maxP = -Infinity;
            for (let i = fi; i <= ti; i++) {
                const c = lastLoadedCandles[i];
                if (c) { if (c.low < minP) minP = c.low; if (c.high > maxP) maxP = c.high; }
            }
            if (minP === Infinity) return null;
            const pad = (maxP - minP) * 0.05;
            return { priceRange: { minValue: minP - pad, maxValue: maxP + pad } };
        }
    });
    chart.priceScale('right').applyOptions({ autoScale: true });
}
function fmtPrice(v) {
    if (v == null || isNaN(v)) return '\u2014';
    const a = Math.abs(v);
    if (a === 0) return '0';
    if (a >= 1) return v.toFixed(Math.min(4, Math.max(2, 4 - Math.floor(Math.log10(a)))));
    const d = Math.max(4, -Math.floor(Math.log10(a)) + 3);
    return v.toFixed(Math.min(d, 12));
}

async function loadPairs() {
    try {
        const r = await fetch(BASE + '/api/liq-pairs');
        const d = await r.json();
        const sel = document.getElementById('pairSelect');
        sel.innerHTML = '';
        (d.pairs || []).forEach(p => {
            const o = document.createElement('option');
            o.value = p; o.textContent = p;
            if (p === 'BTC_USDT_PERP') o.selected = true;
            sel.appendChild(o);
        });
    } catch(e) { console.error('Failed to load pairs', e); }
}

function clearOverlays() {
    overlays.forEach(s => { try { chart.removeSeries(s); } catch(e){} });
    overlays = [];
    candleSeries.setMarkers([]);
}

/* --- Draw a horizontal liquidity pool line with $$$ label --- */
function drawPoolLine(pool, candles, isBSL) {
    const first = candles[0]?.time || 0;
    const last = candles[candles.length - 1]?.time || Infinity;
    const poolTime = pool.time || first;
    if (poolTime > last) return;

    const isPrimary = pool.is_primary;
    const isEqual = pool.is_equal;

    let color, lineWidth, lineStyle;
    if (isBSL) {
        color = isPrimary ? 'rgba(33,150,243,0.7)' : 'rgba(33,150,243,0.35)';
        lineWidth = isPrimary ? 2 : 1;
    } else {
        color = isPrimary ? 'rgba(255,152,0,0.7)' : 'rgba(255,152,0,0.35)';
        lineWidth = isPrimary ? 2 : 1;
    }
    lineStyle = isPrimary ? 0 : LightweightCharts.LineStyle.Dashed;

    // Extend pool line from its time to chart end
    const pl = chart.addLineSeries({
        color: color, lineWidth: lineWidth, lineStyle: lineStyle,
        crosshairMarkerVisible: false, lastValueVisible: false, priceLineVisible: false,
    });
    pl.setData([{ time: poolTime, value: pool.price }, { time: last, value: pool.price }]);
    overlays.push(pl);
}

/* --- Draw a liquidity curve connecting taps --- */
function drawCurve(curve, candles, isSellSide) {
    const first = candles[0]?.time || 0;
    const last = candles[candles.length - 1]?.time || Infinity;
    const taps = (curve.taps || []).filter(t => t.time && t.time >= first && t.time <= last);
    if (taps.length < 2) return;

    const color = isSellSide ? 'rgba(156,39,176,0.7)' : 'rgba(156,39,176,0.7)';

    const cl = chart.addLineSeries({
        color: color, lineWidth: 2, lineStyle: 0,
        crosshairMarkerVisible: false, lastValueVisible: false, priceLineVisible: false,
    });
    cl.setData(taps.map(t => ({ time: t.time, value: t.price })));
    overlays.push(cl);
}

/* --- Draw extreme liquidity target as a highlighted zone --- */
function drawTarget(target, candles) {
    const first = candles[0]?.time || 0;
    const last = candles[candles.length - 1]?.time || Infinity;
    const tTime = target.time || first;
    if (tTime > last) return;

    const color = target.is_swept ? 'rgba(220,53,69,0.5)' : 'rgba(76,175,80,0.6)';
    const lineStyle = target.is_swept ? LightweightCharts.LineStyle.Dotted : LightweightCharts.LineStyle.Solid;

    // Draw target price as a highlighted line
    const tl = chart.addLineSeries({
        color: color, lineWidth: 2, lineStyle: lineStyle,
        crosshairMarkerVisible: false, lastValueVisible: false, priceLineVisible: false,
    });
    tl.setData([{ time: tTime, value: target.target_price }, { time: last, value: target.target_price }]);
    overlays.push(tl);
}

/* --- Draw a liquidity void region --- */
function drawVoid(liqVoid, candles) {
    const first = candles[0]?.time || 0;
    const last = candles[candles.length - 1]?.time || Infinity;

    const borderColor = 'rgba(158,158,158,0.3)';
    const fillColor = 'rgba(158,158,158,0.08)';

    // Top border
    const tl = chart.addLineSeries({
        color: borderColor, lineWidth: 1, lineStyle: LightweightCharts.LineStyle.Dashed,
        crosshairMarkerVisible: false, lastValueVisible: false, priceLineVisible: false,
    });
    tl.setData([{ time: first, value: liqVoid.high }, { time: last, value: liqVoid.high }]);
    overlays.push(tl);

    // Bottom border
    const bl = chart.addLineSeries({
        color: borderColor, lineWidth: 1, lineStyle: LightweightCharts.LineStyle.Dashed,
        crosshairMarkerVisible: false, lastValueVisible: false, priceLineVisible: false,
    });
    bl.setData([{ time: first, value: liqVoid.low }, { time: last, value: liqVoid.low }]);
    overlays.push(bl);

    // Fill lines
    const gap = liqVoid.high - liqVoid.low;
    if (gap > 0) {
        for (let f = 0.25; f <= 0.75; f += 0.25) {
            const fl = chart.addLineSeries({
                color: fillColor, lineWidth: 1, lineStyle: 0,
                crosshairMarkerVisible: false, lastValueVisible: false, priceLineVisible: false,
            });
            fl.setData([{ time: first, value: liqVoid.low + gap * f }, { time: last, value: liqVoid.low + gap * f }]);
            overlays.push(fl);
        }
    }
}

/* --- Draw equal high/low level --- */
function drawEqualLevel(price, candles, isHigh) {
    const first = candles[0]?.time || 0;
    const last = candles[candles.length - 1]?.time || Infinity;

    const color = 'rgba(156,39,176,0.6)';
    const el = chart.addLineSeries({
        color: color, lineWidth: 2, lineStyle: LightweightCharts.LineStyle.SparseDotted,
        crosshairMarkerVisible: false, lastValueVisible: false, priceLineVisible: false,
    });
    el.setData([{ time: first, value: price }, { time: last, value: price }]);
    overlays.push(el);
}

/* --- Main redraw --- */
function redraw(data) {
    clearOverlays();
    const candles = data.candles || [];
    if (candles.length === 0) return;

    const first = candles[0].time;
    const last = candles[candles.length - 1].time;
    const markers = [];

    // Candle interval
    let candleInterval = 14400;
    if (candles.length >= 2) candleInterval = candles[1].time - candles[0].time;

    // MS zigzag (off by default)
    if (layers.ms) {
        const pivots = (data.ms_pivots || []).filter(p => p.time >= first && p.time <= last).sort((a,b) => a.idx - b.idx);
        if (pivots.length >= 2) {
            const zz = chart.addLineSeries({
                color: 'rgba(0,0,0,0.2)', lineWidth: 1, lineStyle: 0,
                crosshairMarkerVisible: false, lastValueVisible: false, priceLineVisible: false,
            });
            zz.setData(pivots.map(p => ({ time: p.time, value: p.price })));
            overlays.push(zz);
        }
    }

    // BSL pools (Buy-Side Liquidity above price)
    if (layers.bsl) {
        (data.bsl_pools || []).forEach(pool => {
            drawPoolLine(pool, candles, true);
            // $$$ marker at pool location
            if (pool.time && pool.time >= first && pool.time <= last) {
                markers.push({
                    time: pool.time,
                    position: 'aboveBar',
                    color: pool.is_primary ? '#2196f3' : 'rgba(33,150,243,0.5)',
                    shape: 'circle',
                    size: pool.is_primary ? 1 : 0.5,
                    text: pool.is_equal ? '$$$\u2261' : '$$$',
                });
            }
        });
    }

    // SSL pools (Sell-Side Liquidity below price)
    if (layers.ssl) {
        (data.ssl_pools || []).forEach(pool => {
            drawPoolLine(pool, candles, false);
            // $$$ marker at pool location
            if (pool.time && pool.time >= first && pool.time <= last) {
                markers.push({
                    time: pool.time,
                    position: 'belowBar',
                    color: pool.is_primary ? '#ff9800' : 'rgba(255,152,0,0.5)',
                    shape: 'circle',
                    size: pool.is_primary ? 1 : 0.5,
                    text: pool.is_equal ? '$$$\u2261' : '$$$',
                });
            }
        });
    }

    // Liquidity curves
    if (layers.curves) {
        (data.sell_side_curves || []).forEach(curve => {
            drawCurve(curve, candles, true);
            // Mark taps
            (curve.taps || []).forEach((tap, i) => {
                if (tap.time && tap.time >= first && tap.time <= last) {
                    markers.push({
                        time: tap.time,
                        position: 'aboveBar',
                        color: '#9c27b0',
                        shape: 'arrowDown',
                        size: 0.8,
                        text: (i === 0 ? '1st' : i === 1 ? '2nd' : (i+1)+'th') + ' tap',
                    });
                }
            });
        });
        (data.buy_side_curves || []).forEach(curve => {
            drawCurve(curve, candles, false);
            (curve.taps || []).forEach((tap, i) => {
                if (tap.time && tap.time >= first && tap.time <= last) {
                    markers.push({
                        time: tap.time,
                        position: 'belowBar',
                        color: '#9c27b0',
                        shape: 'arrowUp',
                        size: 0.8,
                        text: (i === 0 ? '1st' : i === 1 ? '2nd' : (i+1)+'th') + ' tap',
                    });
                }
            });
        });
    }

    // Extreme liquidity targets
    if (layers.targets) {
        (data.sell_side_targets || []).forEach(t => {
            drawTarget(t, candles);
            if (t.time && t.time >= first && t.time <= last) {
                markers.push({
                    time: t.time,
                    position: 'aboveBar',
                    color: t.is_swept ? '#dc3545' : '#4caf50',
                    shape: 'square',
                    size: 1,
                    text: t.is_swept ? 'ELT swept' : 'ELT',
                });
            }
        });
        (data.buy_side_targets || []).forEach(t => {
            drawTarget(t, candles);
            if (t.time && t.time >= first && t.time <= last) {
                markers.push({
                    time: t.time,
                    position: 'belowBar',
                    color: t.is_swept ? '#dc3545' : '#4caf50',
                    shape: 'square',
                    size: 1,
                    text: t.is_swept ? 'ELT swept' : 'ELT',
                });
            }
        });
    }

    // Liquidity voids
    if (layers.voids) {
        (data.voids || []).forEach(v => drawVoid(v, candles));
    }

    // Equal highs/lows
    if (layers.eq) {
        (data.equal_highs || []).forEach(price => drawEqualLevel(price, candles, true));
        (data.equal_lows || []).forEach(price => drawEqualLevel(price, candles, false));
    }

    // Deduplicate markers (one per time+position slot)
    if (markers.length > 0) {
        const seen = new Set();
        const unique = markers.filter(m => {
            const key = m.time + '_' + m.position;
            if (seen.has(key)) return false;
            seen.add(key);
            return true;
        });
        unique.sort((a, b) => a.time - b.time);
        candleSeries.setMarkers(unique);
    }
}

async function loadData() {
    const sym = document.getElementById('pairSelect').value;
    const tf = document.getElementById('tfSelect').value;
    const btn = document.getElementById('loadBtn');
    const overlay = document.getElementById('loadingOverlay');

    btn.disabled = true;
    overlay.classList.remove('hidden');
    clearOverlays();

    try {
        const r = await fetch(BASE + `/api/liquidity-chart?symbol=${sym}&timeframe=${tf}&limit=300`);
        if (!r.ok) {
            const errData = await r.json().catch(() => ({}));
            throw new Error(errData.error || `HTTP ${r.status}`);
        }
        const data = await r.json();
        lastData = data;

        lastLoadedCandles = data.candles || [];
        if (lastLoadedCandles.length > 0) {
            const sp = Math.abs(lastLoadedCandles[lastLoadedCandles.length - 1].close);
            let prec = 2;
            if (sp > 0 && sp < 1) prec = Math.max(4, -Math.floor(Math.log10(sp)) + 3);
            else if (sp >= 1 && sp < 10) prec = 4;
            else if (sp >= 10 && sp < 1000) prec = 2;
            prec = Math.min(prec, 12);
            candleSeries.applyOptions({ priceFormat: { type: 'price', precision: prec, minMove: Math.pow(10, -prec) } });
        }
        candleSeries.applyOptions({ autoscaleInfoProvider: undefined });
        candleSeries.setData(lastLoadedCandles);

        redraw(data);
        updateInfoPanel(data);
        chart.timeScale().fitContent();
        scalePrice();

    } catch(e) {
        console.error('Load error:', e);
        document.getElementById('infoPanel').innerHTML =
            `<h3 style="color:#dc3545">Error</h3><div class="info-row">${e.message}</div>`;
    } finally {
        btn.disabled = false;
        overlay.classList.add('hidden');
    }
}

function updateInfoPanel(data) {
    const panel = document.getElementById('infoPanel');
    const sym = data.symbol || '\u2014';
    const tf = data.timeframe || '\u2014';
    const summary = data.summary || {};

    let html = `<h3>${sym} \u00b7 ${tf.toUpperCase()}</h3>`;

    const trend = data.trend || 'neutral';
    const tc = trend === 'bullish' ? 'bullish' : trend === 'bearish' ? 'bearish' : 'ranging';
    html += `<div class="info-row"><span>Trend</span><span class="val ${tc}">${trend.toUpperCase()}</span></div>`;
    html += `<div class="info-row"><span>Price</span><span class="val">${fmtPrice(data.current_price)}</span></div>`;

    // BSL/SSL counts
    html += `<h3>Liquidity Pools</h3>`;
    html += `<div class="info-row"><span>BSL Pools</span><span class="val" style="color:#2196f3">${summary.total_bsl || 0}</span></div>`;
    html += `<div class="info-row"><span>\u2003Primary Highs</span><span class="val">${summary.primary_highs || 0}</span></div>`;
    html += `<div class="info-row"><span>\u2003Internal Highs</span><span class="val">${summary.internal_highs || 0}</span></div>`;
    html += `<div class="info-row"><span>SSL Pools</span><span class="val" style="color:#ff9800">${summary.total_ssl || 0}</span></div>`;
    html += `<div class="info-row"><span>\u2003Primary Lows</span><span class="val">${summary.primary_lows || 0}</span></div>`;
    html += `<div class="info-row"><span>\u2003Internal Lows</span><span class="val">${summary.internal_lows || 0}</span></div>`;
    html += `<div class="info-row"><span>Equal Highs</span><span class="val" style="color:#9c27b0">${summary.equal_highs || 0}</span></div>`;
    html += `<div class="info-row"><span>Equal Lows</span><span class="val" style="color:#9c27b0">${summary.equal_lows || 0}</span></div>`;

    // BSL pool details
    const bslPools = data.bsl_pools || [];
    if (bslPools.length > 0) {
        html += `<h3>BSL Pools (Buy-Side)</h3>`;
        html += `<div class="pool-list">`;
        bslPools.slice(0, 10).forEach(pool => {
            const cls = pool.is_primary ? 'primary' : 'internal';
            const eqTag = pool.is_equal ? '<span class="tag equal">EQUAL</span>' : '';
            const msTag = pool.is_confirmed_ms ? '<span class="tag primary">MS</span>' : '';
            html += `<div class="pool-item bsl">
                <div class="pool-header">
                    <span class="pool-label">${fmtPrice(pool.price)}</span>
                    <span class="tag ${cls}">${pool.is_primary ? 'PRIMARY' : 'INTERNAL'}</span>
                    ${eqTag}${msTag}
                </div>
                <div>Strength: ${(pool.strength * 100).toFixed(0)}% | Dist: ${pool.distance_from_price?.toFixed(2) || 0}%</div>
            </div>`;
        });
        html += `</div>`;
    }

    // SSL pool details
    const sslPools = data.ssl_pools || [];
    if (sslPools.length > 0) {
        html += `<h3>SSL Pools (Sell-Side)</h3>`;
        html += `<div class="pool-list">`;
        sslPools.slice(0, 10).forEach(pool => {
            const cls = pool.is_primary ? 'primary' : 'internal';
            const eqTag = pool.is_equal ? '<span class="tag equal">EQUAL</span>' : '';
            const msTag = pool.is_confirmed_ms ? '<span class="tag primary">MS</span>' : '';
            html += `<div class="pool-item ssl">
                <div class="pool-header">
                    <span class="pool-label">${fmtPrice(pool.price)}</span>
                    <span class="tag ${cls}">${pool.is_primary ? 'PRIMARY' : 'INTERNAL'}</span>
                    ${eqTag}${msTag}
                </div>
                <div>Strength: ${(pool.strength * 100).toFixed(0)}% | Dist: ${pool.distance_from_price?.toFixed(2) || 0}%</div>
            </div>`;
        });
        html += `</div>`;
    }

    // Curves
    const sellCurves = data.sell_side_curves || [];
    const buyCurves = data.buy_side_curves || [];
    if (sellCurves.length > 0 || buyCurves.length > 0) {
        html += `<h3>Liquidity Curves</h3>`;
        html += `<div class="info-row"><span>Sell-Side (Dist)</span><span class="val">${sellCurves.length}</span></div>`;
        html += `<div class="info-row"><span>Buy-Side (Accum)</span><span class="val">${buyCurves.length}</span></div>`;
        html += `<div class="pool-list">`;
        sellCurves.forEach((c, i) => {
            html += `<div class="pool-item curve">
                <div class="pool-header">
                    <span class="pool-label">Sell-Side #${i+1}</span>
                    <span class="tag ${c.quality === 'EXCELLENT' ? 'good' : 'moderate'}">${c.quality} (${c.tap_count} taps)</span>
                </div>
                <div>Range: ${fmtPrice(c.highest_price)} \u2192 ${fmtPrice(c.lowest_price)}</div>
            </div>`;
        });
        buyCurves.forEach((c, i) => {
            html += `<div class="pool-item curve">
                <div class="pool-header">
                    <span class="pool-label">Buy-Side #${i+1}</span>
                    <span class="tag ${c.quality === 'EXCELLENT' ? 'good' : 'moderate'}">${c.quality} (${c.tap_count} taps)</span>
                </div>
                <div>Range: ${fmtPrice(c.lowest_price)} \u2192 ${fmtPrice(c.highest_price)}</div>
            </div>`;
        });
        html += `</div>`;
    }

    // Extreme targets
    const sellTargets = data.sell_side_targets || [];
    const buyTargets = data.buy_side_targets || [];
    if (sellTargets.length > 0 || buyTargets.length > 0) {
        html += `<h3>Extreme Liquidity Targets</h3>`;
        html += `<div class="pool-list">`;
        sellTargets.forEach(t => {
            html += `<div class="pool-item target">
                <div class="pool-header">
                    <span class="pool-label">Dist ELT: ${fmtPrice(t.target_price)}</span>
                    <span class="tag ${t.is_swept ? 'bad' : 'good'}">${t.is_swept ? 'SWEPT' : 'ACTIVE'}</span>
                </div>
                <div>2nd tap: ${fmtPrice(t.second_tap_price)}</div>
            </div>`;
        });
        buyTargets.forEach(t => {
            html += `<div class="pool-item target">
                <div class="pool-header">
                    <span class="pool-label">Accum ELT: ${fmtPrice(t.target_price)}</span>
                    <span class="tag ${t.is_swept ? 'bad' : 'good'}">${t.is_swept ? 'SWEPT' : 'ACTIVE'}</span>
                </div>
                <div>2nd tap: ${fmtPrice(t.second_tap_price)}</div>
            </div>`;
        });
        html += `</div>`;
    }

    // Voids
    const voids = data.voids || [];
    if (voids.length > 0) {
        html += `<h3>Liquidity Voids</h3>`;
        html += `<div class="pool-list">`;
        voids.forEach(v => {
            html += `<div class="pool-item void">
                <div class="pool-header">
                    <span class="pool-label">${fmtPrice(v.low)} - ${fmtPrice(v.high)}</span>
                    <span class="tag ${v.quality === 'LARGE' ? 'bad' : 'moderate'}">${v.quality}</span>
                </div>
                <div>Size: ${v.size_percent?.toFixed(1) || 0}% | Dist: ${v.distance_from_price?.toFixed(2) || 0}%</div>
            </div>`;
        });
        html += `</div>`;
    }

    // TCT Liquidity rules reference
    html += `<h3>TCT Liquidity Rules</h3>`;
    html += `<div class="info-row" style="flex-direction:column;gap:2px">
        <div>\u2022 BSL = above highs (shorts' stop losses)</div>
        <div>\u2022 SSL = below lows (longs' stop losses)</div>
        <div>\u2022 Primary = confirmed MS pivots (strongest)</div>
        <div>\u2022 Internal = 6CR pivots between primaries</div>
        <div>\u2022 Equal H/L = amazing liquidity targets</div>
        <div>\u2022 Curve = 3+ taps, no S/D backing levels</div>
        <div>\u2022 ELT = 1st higher low / lower high after 2nd tap</div>
        <div>\u2022 Void = no S/D zones, price moves freely</div>
        <div>\u2022 Markets move from liquidity to liquidity</div>
    </div>`;

    panel.innerHTML = html;
}

initChart();
loadPairs();
document.getElementById('pairSelect').addEventListener('change', loadData);
document.getElementById('tfSelect').addEventListener('change', loadData);
</script>
</body>
</html>"""

    return HTMLResponse(content=html)


# ================================================================
# ENTRY POINT
# ================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server_mexc:app", host="0.0.0.0", port=PORT, reload=True)
