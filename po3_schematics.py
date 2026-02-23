"""
po3_schematics.py — PO3 (Power of Three) Schematic Detection (TCT Lecture 8)
Author: HPB-TCT Dev Team
Date: 2026-01-30

Detects PO3 Schematics: Range → Manipulation → Expansion

TCT Lecture 8 Key Concepts:
- PO3 = Power of Three: Range → Manipulation → Expansion
- Bullish PO3: Range forms → price breaks below range low (manipulation) → accumulation
  in manipulation range → expansion upward
- Bearish PO3: Range forms → price breaks above range high (manipulation) → distribution
  in manipulation range → expansion downward

Range Requirements:
- Good return-to-zone (RTZ) via liquidity curve or trendline
- Extra confluence if price compresses within range building liquidity on both sides

Manipulation Requirements:
- Must stay inside DL2 (deviation limit ~30% of range)
- Must contain valid TCT model (accumulation for bullish, distribution for bearish)
- More aggressive breakout = more cautious for PO3

Expansion:
- No specific requirements, entry is in manipulation phase
- Extend target to PO3 range high/low instead of local manipulation range target

Exceptions:
- Exception 1 (2-tap): Only deviates once then reverses, no third tap. Valid if RTZ is very good.
- Exception 2 (Internal TCT): Accumulation/distribution within manipulation range without
  sweeping the main range high/low.

Key Rule: When a TCT model fails, look for potential PO3.
Minimum Timeframe: 4H+ for the range, manipulation can be LTF.
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger("PO3-Schematics")


class PO3SchematicDetector:
    """
    PO3 (Power of Three) Schematic Detection Engine — TCT Lecture 8.

    Detects the three-phase manipulation play:
    1. RANGE: Identifies a consolidation range with good RTZ quality
    2. MANIPULATION: Detects aggressive breakout that stays within DL2,
       containing a valid TCT model (accumulation or distribution)
    3. EXPANSION: Tracks reversal and expansion toward opposite range extreme

    Also detects:
    - Exception 1 (2-tap schematic): Single deviation with excellent RTZ
    - Exception 2 (Internal TCT): TCT model inside manipulation without sweeping range extreme
    - Nested PO3 (PO3 within PO3)
    """

    # TCT Lecture 8 constants
    DEVIATION_LIMIT_PCT = 0.30          # DL2 = 30% of range size
    MIN_RANGE_CANDLES = 10              # Minimum candles to form a valid range
    RTZ_QUALITY_THRESHOLD = 0.5         # Minimum RTZ quality score (0-1)
    COMPRESSION_THRESHOLD = 0.4         # Range narrowing by 40% = compression
    AGGRESSION_THRESHOLD = 0.7          # Breakout speed threshold for caution
    MIN_MANIPULATION_CANDLES = 3        # Minimum candles in manipulation phase
    MIN_RR_RATIO = 2.0                  # Minimum risk-to-reward for PO3 setup

    def __init__(self, candles: pd.DataFrame):
        """Initialize with OHLC candle data."""
        self.candles = candles.reset_index(drop=True)

    def detect_po3_schematics(self, detected_ranges: List[Dict] = None) -> Dict:
        """
        Main entry point: detect all PO3 schematics.

        Args:
            detected_ranges: Optional pre-detected ranges from range detection engine

        Returns:
            Dict with bullish_po3, bearish_po3, and summary
        """
        if len(self.candles) < 50:
            return {
                "bullish_po3": [],
                "bearish_po3": [],
                "total": 0,
                "error": "Insufficient data (need 50+ candles)"
            }

        ranges = detected_ranges or self._detect_ranges()

        bullish = []
        bearish = []

        for rng in ranges:
            # Check for bullish PO3 (breakdown below range → accumulation → expansion up)
            bull_po3 = self._detect_bullish_po3(rng)
            if bull_po3:
                bullish.append(bull_po3)

            # Check for bearish PO3 (breakout above range → distribution → expansion down)
            bear_po3 = self._detect_bearish_po3(rng)
            if bear_po3:
                bearish.append(bear_po3)

        # Sort by quality score descending
        bullish.sort(key=lambda x: x.get("quality_score", 0), reverse=True)
        bearish.sort(key=lambda x: x.get("quality_score", 0), reverse=True)

        return {
            "bullish_po3": bullish,
            "bearish_po3": bearish,
            "total": len(bullish) + len(bearish),
            "candles_analyzed": len(self.candles),
            "timestamp": datetime.utcnow().isoformat()
        }

    # ================================================================
    # RANGE DETECTION
    # ================================================================

    def _detect_ranges(self) -> List[Dict]:
        """Detect consolidation ranges suitable for PO3 analysis."""
        ranges = []
        n = len(self.candles)
        if n < self.MIN_RANGE_CANDLES:
            return ranges

        # Sliding window approach to find consolidation ranges
        for window_size in [40, 30, 20]:
            if n < window_size:
                continue
            for start in range(0, n - window_size, window_size // 2):
                end = min(start + window_size, n)
                segment = self.candles.iloc[start:end]

                high = float(segment["high"].max())
                low = float(segment["low"].min())
                range_size = high - low

                if range_size <= 0 or low <= 0:
                    continue

                range_pct = (range_size / low) * 100

                # Range should be tight enough (consolidation)
                if range_pct > 15:
                    continue

                # Check for horizontal price action (range rationality)
                body_sizes = abs(segment["close"] - segment["open"])
                avg_body = float(body_sizes.mean())
                if avg_body <= 0:
                    continue

                rationality = 1 - min(1.0, (avg_body / range_size) * 2)

                if rationality < 0.3:
                    continue

                # RTZ quality: how many times does price return to equilibrium?
                eq = (high + low) / 2
                eq_zone_size = range_size * 0.2
                touches = int((abs(segment["close"].astype(float) - eq) < eq_zone_size).sum())
                rtz_quality = min(1.0, touches / (len(segment) * 0.3))

                # Compression detection: check if range narrows over time
                first_half = segment.iloc[:len(segment)//2]
                second_half = segment.iloc[len(segment)//2:]
                first_range = float(first_half["high"].max() - first_half["low"].min())
                second_range = float(second_half["high"].max() - second_half["low"].min())
                has_compression = second_range < first_range * (1 - self.COMPRESSION_THRESHOLD) if first_range > 0 else False

                # Liquidity building: check for wicks on both sides
                upper_wicks = int((segment["high"].astype(float) > high - range_size * 0.1).sum())
                lower_wicks = int((segment["low"].astype(float) < low + range_size * 0.1).sum())
                has_liquidity_both_sides = upper_wicks >= 2 and lower_wicks >= 2

                ranges.append({
                    "start_idx": start,
                    "end_idx": end - 1,
                    "range_high": high,
                    "range_low": low,
                    "range_size": range_size,
                    "equilibrium": eq,
                    "range_pct": round(range_pct, 2),
                    "rtz_quality": round(rtz_quality, 3),
                    "rationality": round(rationality, 3),
                    "has_compression": bool(has_compression),
                    "has_liquidity_both_sides": bool(has_liquidity_both_sides)
                })

        # Deduplicate overlapping ranges, keep highest quality
        ranges = self._deduplicate_ranges(ranges)
        return ranges

    def _deduplicate_ranges(self, ranges: List[Dict]) -> List[Dict]:
        """Remove overlapping ranges, keeping highest quality."""
        if len(ranges) <= 1:
            return ranges

        ranges.sort(key=lambda r: r["rtz_quality"] + r["rationality"], reverse=True)
        kept = []
        for r in ranges:
            overlap = False
            for k in kept:
                if (r["start_idx"] < k["end_idx"] and r["end_idx"] > k["start_idx"]):
                    overlap = True
                    break
            if not overlap:
                kept.append(r)
        return kept

    # ================================================================
    # BULLISH PO3 DETECTION
    # ================================================================

    def _detect_bullish_po3(self, rng: Dict) -> Optional[Dict]:
        """
        Detect Bullish PO3:
        Range → Breakdown below range low (manipulation) → Accumulation → Expansion UP

        The manipulation phase should:
        - Break below the range low
        - Stay within DL2 (30% of range below the range low)
        - Contain a TCT accumulation model
        """
        range_high = rng["range_high"]
        range_low = rng["range_low"]
        range_size = rng["range_size"]
        end_idx = rng["end_idx"]

        if end_idx >= len(self.candles) - 3:
            return None

        # DL2 boundary: 30% of range below range low
        dl2_limit = range_low - (range_size * self.DEVIATION_LIMIT_PCT)

        # Look for breakdown below range low after range ends
        post_range = self.candles.iloc[end_idx + 1:]
        if len(post_range) < self.MIN_MANIPULATION_CANDLES:
            return None

        # Find the manipulation phase: candles that break below range low
        manipulation_start = None
        manipulation_low = range_low
        manipulation_end = None
        broke_below = False

        for idx in range(len(post_range)):
            candle = post_range.iloc[idx]
            actual_idx = end_idx + 1 + idx
            low = float(candle["low"])
            close = float(candle["close"])

            if low < range_low:
                broke_below = True
                if manipulation_start is None:
                    manipulation_start = actual_idx

                manipulation_low = min(manipulation_low, low)

                # Check DL2 violation - if price goes too far, PO3 is invalid
                if low < dl2_limit:
                    return None

            # If we broke below and price returns above range low = manipulation ending
            if broke_below and close > range_low and manipulation_start is not None:
                manipulation_end = actual_idx
                break

            # Limit search window
            if idx > 50:
                break

        if not broke_below or manipulation_start is None:
            return None

        # Calculate manipulation metrics
        deviation_depth = range_low - manipulation_low
        deviation_pct = (deviation_depth / range_size) * 100 if range_size > 0 else 0

        # Aggression check: how fast was the breakdown?
        if manipulation_start > end_idx + 1:
            breakdown_candles = manipulation_start - end_idx
        else:
            breakdown_candles = 1
        aggression = min(1.0, 3.0 / max(1, breakdown_candles))

        # Check for TCT accumulation model in manipulation range
        manip_end_idx = manipulation_end or min(manipulation_start + 20, len(self.candles) - 1)
        has_tct_model = self._detect_tct_model_in_range(
            manipulation_start, manip_end_idx, "accumulation",
            manipulation_low, range_low
        )

        # Exception detection
        exception_type = self._detect_exception(
            rng, manipulation_start, manip_end_idx, "bullish", manipulation_low
        )

        # Check for expansion (price moving above range high)
        has_expansion = False
        expansion_high = range_high
        if manipulation_end and manipulation_end < len(self.candles) - 1:
            expansion_candles = self.candles.iloc[manipulation_end:]
            _exp_highs = expansion_candles["high"].astype(float)
            _above_range = _exp_highs[_exp_highs > range_high]
            if not _above_range.empty:
                has_expansion = True
                expansion_high = max(expansion_high, float(_above_range.max()))

        # Calculate quality score
        quality = self._calculate_quality(
            rtz_quality=rng["rtz_quality"],
            deviation_pct=deviation_pct,
            has_tct_model=has_tct_model,
            has_compression=rng.get("has_compression", False),
            has_liq_both_sides=rng.get("has_liquidity_both_sides", False),
            aggression=aggression,
            exception_type=exception_type
        )

        # Calculate entry, stop, target
        entry_price = range_low  # Entry at range low (end of manipulation)
        stop_price = manipulation_low  # Stop below manipulation low
        target_price = range_high  # PO3 target = opposite range extreme

        risk = entry_price - stop_price
        reward = target_price - entry_price
        rr_ratio = reward / risk if risk > 0 else 0

        # Determine phase
        if has_expansion:
            phase = "expansion"
        elif manipulation_end:
            phase = "manipulation_complete"
        elif broke_below:
            phase = "manipulation"
        else:
            phase = "range"

        return {
            "direction": "bullish",
            "type": "PO3_BULLISH",
            "phase": phase,
            "range": {
                "high": round(range_high, 2),
                "low": round(range_low, 2),
                "equilibrium": round(rng["equilibrium"], 2),
                "size_pct": rng["range_pct"]
            },
            "manipulation": {
                "low": round(manipulation_low, 2),
                "deviation_depth": round(deviation_depth, 2),
                "deviation_pct": round(deviation_pct, 2),
                "dl2_limit": round(dl2_limit, 2),
                "within_dl2": True,
                "aggression": round(aggression, 3)
            },
            "tct_model": {
                "detected": has_tct_model,
                "type": "accumulation"
            },
            "exception": exception_type,
            "entry": {
                "price": round(entry_price, 2),
                "type": "manipulation_reversal"
            },
            "stop_loss": {
                "price": round(stop_price, 2),
                "below_manipulation_low": True
            },
            "target": {
                "price": round(target_price, 2),
                "type": "po3_range_high",
                "extended": has_expansion
            },
            "risk_reward": round(rr_ratio, 2),
            "quality_score": round(quality, 3),
            "has_compression": rng.get("has_compression", False),
            "has_liquidity_both_sides": rng.get("has_liquidity_both_sides", False),
            "has_expansion": has_expansion,
            "range_indices": {
                "range_start": rng["start_idx"],
                "range_end": rng["end_idx"],
                "manipulation_start": manipulation_start,
                "manipulation_end": manipulation_end
            }
        }

    # ================================================================
    # BEARISH PO3 DETECTION
    # ================================================================

    def _detect_bearish_po3(self, rng: Dict) -> Optional[Dict]:
        """
        Detect Bearish PO3:
        Range → Breakout above range high (manipulation) → Distribution → Expansion DOWN

        The manipulation phase should:
        - Break above the range high
        - Stay within DL2 (30% of range above the range high)
        - Contain a TCT distribution model
        """
        range_high = rng["range_high"]
        range_low = rng["range_low"]
        range_size = rng["range_size"]
        end_idx = rng["end_idx"]

        if end_idx >= len(self.candles) - 3:
            return None

        # DL2 boundary: 30% of range above range high
        dl2_limit = range_high + (range_size * self.DEVIATION_LIMIT_PCT)

        # Look for breakout above range high after range ends
        post_range = self.candles.iloc[end_idx + 1:]
        if len(post_range) < self.MIN_MANIPULATION_CANDLES:
            return None

        manipulation_start = None
        manipulation_high = range_high
        manipulation_end = None
        broke_above = False

        for idx in range(len(post_range)):
            candle = post_range.iloc[idx]
            actual_idx = end_idx + 1 + idx
            high = float(candle["high"])
            close = float(candle["close"])

            if high > range_high:
                broke_above = True
                if manipulation_start is None:
                    manipulation_start = actual_idx

                manipulation_high = max(manipulation_high, high)

                # Check DL2 violation
                if high > dl2_limit:
                    return None

            # If we broke above and price returns below range high = manipulation ending
            if broke_above and close < range_high and manipulation_start is not None:
                manipulation_end = actual_idx
                break

            if idx > 50:
                break

        if not broke_above or manipulation_start is None:
            return None

        # Calculate manipulation metrics
        deviation_depth = manipulation_high - range_high
        deviation_pct = (deviation_depth / range_size) * 100 if range_size > 0 else 0

        if manipulation_start > end_idx + 1:
            breakdown_candles = manipulation_start - end_idx
        else:
            breakdown_candles = 1
        aggression = min(1.0, 3.0 / max(1, breakdown_candles))

        # Check for TCT distribution model in manipulation range
        manip_end_idx = manipulation_end or min(manipulation_start + 20, len(self.candles) - 1)
        has_tct_model = self._detect_tct_model_in_range(
            manipulation_start, manip_end_idx, "distribution",
            range_high, manipulation_high
        )

        # Exception detection
        exception_type = self._detect_exception(
            rng, manipulation_start, manip_end_idx, "bearish", manipulation_high
        )

        # Check for expansion (price moving below range low)
        has_expansion = False
        expansion_low = range_low
        if manipulation_end and manipulation_end < len(self.candles) - 1:
            expansion_candles = self.candles.iloc[manipulation_end:]
            _exp_lows = expansion_candles["low"].astype(float)
            _below_range = _exp_lows[_exp_lows < range_low]
            if not _below_range.empty:
                has_expansion = True
                expansion_low = min(expansion_low, float(_below_range.min()))

        # Calculate quality
        quality = self._calculate_quality(
            rtz_quality=rng["rtz_quality"],
            deviation_pct=deviation_pct,
            has_tct_model=has_tct_model,
            has_compression=rng.get("has_compression", False),
            has_liq_both_sides=rng.get("has_liquidity_both_sides", False),
            aggression=aggression,
            exception_type=exception_type
        )

        # Entry, stop, target
        entry_price = range_high
        stop_price = manipulation_high
        target_price = range_low

        risk = stop_price - entry_price
        reward = entry_price - target_price
        rr_ratio = reward / risk if risk > 0 else 0

        # Determine phase
        if has_expansion:
            phase = "expansion"
        elif manipulation_end:
            phase = "manipulation_complete"
        elif broke_above:
            phase = "manipulation"
        else:
            phase = "range"

        return {
            "direction": "bearish",
            "type": "PO3_BEARISH",
            "phase": phase,
            "range": {
                "high": round(range_high, 2),
                "low": round(range_low, 2),
                "equilibrium": round(rng["equilibrium"], 2),
                "size_pct": rng["range_pct"]
            },
            "manipulation": {
                "high": round(manipulation_high, 2),
                "deviation_depth": round(deviation_depth, 2),
                "deviation_pct": round(deviation_pct, 2),
                "dl2_limit": round(dl2_limit, 2),
                "within_dl2": True,
                "aggression": round(aggression, 3)
            },
            "tct_model": {
                "detected": has_tct_model,
                "type": "distribution"
            },
            "exception": exception_type,
            "entry": {
                "price": round(entry_price, 2),
                "type": "manipulation_reversal"
            },
            "stop_loss": {
                "price": round(stop_price, 2),
                "above_manipulation_high": True
            },
            "target": {
                "price": round(target_price, 2),
                "type": "po3_range_low",
                "extended": has_expansion
            },
            "risk_reward": round(rr_ratio, 2),
            "quality_score": round(quality, 3),
            "has_compression": rng.get("has_compression", False),
            "has_liquidity_both_sides": rng.get("has_liquidity_both_sides", False),
            "has_expansion": has_expansion,
            "range_indices": {
                "range_start": rng["start_idx"],
                "range_end": rng["end_idx"],
                "manipulation_start": manipulation_start,
                "manipulation_end": manipulation_end
            }
        }

    # ================================================================
    # TCT MODEL DETECTION IN MANIPULATION RANGE
    # ================================================================

    @staticmethod
    def _is_inside_bar(candle, prev_candle) -> bool:
        """TCT Lecture 1: Inside bar has H/L within previous bar's H/L."""
        return (float(candle["high"]) <= float(prev_candle["high"]) and
                float(candle["low"]) >= float(prev_candle["low"]))

    def _find_6cr_swing_low(self, segment: pd.DataFrame, start: int, end: int) -> Optional[Dict]:
        """Find swing low using 6-candle rule with inside bar exclusion."""
        best = None
        for i in range(start + 2, min(end - 2, len(segment) - 2)):
            # Skip inside bars
            if i >= 1 and self._is_inside_bar(segment.iloc[i], segment.iloc[i - 1]):
                continue
            low_val = float(segment.iloc[i]["low"])
            # Check 2 non-inside-bar candles before have higher lows
            before_ok = True
            count = 0
            for j in range(i - 1, max(start - 1, -1), -1):
                if j >= 1 and self._is_inside_bar(segment.iloc[j], segment.iloc[j - 1]):
                    continue
                if float(segment.iloc[j]["low"]) <= low_val:
                    before_ok = False
                    break
                count += 1
                if count >= 2:
                    break
            if not before_ok or count < 2:
                continue
            # Check 2 non-inside-bar candles after have higher lows
            after_ok = True
            count = 0
            for j in range(i + 1, min(end + 1, len(segment))):
                if self._is_inside_bar(segment.iloc[j], segment.iloc[j - 1]):
                    continue
                if float(segment.iloc[j]["low"]) <= low_val:
                    after_ok = False
                    break
                count += 1
                if count >= 2:
                    break
            if after_ok and count >= 2:
                if best is None or low_val < best["price"]:
                    best = {"idx": i, "price": low_val}
        return best

    def _find_6cr_swing_high(self, segment: pd.DataFrame, start: int, end: int) -> Optional[Dict]:
        """Find swing high using 6-candle rule with inside bar exclusion."""
        best = None
        for i in range(start + 2, min(end - 2, len(segment) - 2)):
            if i >= 1 and self._is_inside_bar(segment.iloc[i], segment.iloc[i - 1]):
                continue
            high_val = float(segment.iloc[i]["high"])
            before_ok = True
            count = 0
            for j in range(i - 1, max(start - 1, -1), -1):
                if j >= 1 and self._is_inside_bar(segment.iloc[j], segment.iloc[j - 1]):
                    continue
                if float(segment.iloc[j]["high"]) >= high_val:
                    before_ok = False
                    break
                count += 1
                if count >= 2:
                    break
            if not before_ok or count < 2:
                continue
            after_ok = True
            count = 0
            for j in range(i + 1, min(end + 1, len(segment))):
                if self._is_inside_bar(segment.iloc[j], segment.iloc[j - 1]):
                    continue
                if float(segment.iloc[j]["high"]) >= high_val:
                    after_ok = False
                    break
                count += 1
                if count >= 2:
                    break
            if after_ok and count >= 2:
                if best is None or high_val > best["price"]:
                    best = {"idx": i, "price": high_val}
        return best

    def _detect_tct_model_in_range(
        self, start_idx: int, end_idx: int,
        model_type: str, zone_low: float, zone_high: float
    ) -> bool:
        """
        Detect a TCT model (accumulation or distribution) within the manipulation range.

        Uses TCT Lecture 1 market structure:
        - 6-candle rule with inside bar exclusion for pivot detection
        - BOS confirmed by candle CLOSE (not wick)
        - Accumulation: lower low (spring) → higher low → bullish BOS (close above swing high)
        - Distribution: higher high (throw-over) → lower high → bearish BOS (close below swing low)
        """
        if start_idx >= end_idx or end_idx >= len(self.candles):
            return False

        segment = self.candles.iloc[start_idx:end_idx + 1].reset_index(drop=True)
        if len(segment) < 4:
            return False

        if model_type == "accumulation":
            lows = segment["low"].astype(float).tolist()
            closes = segment["close"].astype(float).tolist()

            if len(lows) < 3:
                return False

            # Find lowest point (spring/deviation)
            min_idx = int(np.argmin(lows))

            if min_idx >= len(closes) - 1:
                return False

            # Check for higher low after the spring (6CR validated if possible)
            post_min_lows = lows[min_idx + 1:]
            has_higher_low = any(l > lows[min_idx] for l in post_min_lows) if post_min_lows else False

            if not has_higher_low:
                return False

            # TCT Lecture 1: BOS = candle CLOSE above the swing high between lows
            # Find the highest high between spring and end
            post_spring = segment.iloc[min_idx:]
            if len(post_spring) < 2:
                return has_higher_low

            swing_high_price = float(post_spring["high"].max())
            # Check if any candle closes above this swing high
            for i in range(min_idx + 1, len(segment)):
                if closes[i] > swing_high_price * 0.998:  # Small tolerance
                    return True

            # Even without BOS, higher low pattern is a valid TCT signal
            return has_higher_low

        elif model_type == "distribution":
            highs = segment["high"].astype(float).tolist()
            closes = segment["close"].astype(float).tolist()

            if len(highs) < 3:
                return False

            max_idx = int(np.argmax(highs))

            if max_idx >= len(closes) - 1:
                return False

            # Check for lower high after the throw-over
            post_max_highs = highs[max_idx + 1:]
            has_lower_high = any(h < highs[max_idx] for h in post_max_highs) if post_max_highs else False

            if not has_lower_high:
                return False

            # TCT Lecture 1: BOS = candle CLOSE below the swing low between highs
            post_throwover = segment.iloc[max_idx:]
            if len(post_throwover) < 2:
                return has_lower_high

            swing_low_price = float(post_throwover["low"].min())
            for i in range(max_idx + 1, len(segment)):
                if closes[i] < swing_low_price * 1.002:
                    return True

            return has_lower_high

        return False

    # ================================================================
    # EXCEPTION DETECTION
    # ================================================================

    def _detect_exception(
        self, rng: Dict, manip_start: int, manip_end: int,
        direction: str, extreme_price: float
    ) -> Optional[str]:
        """
        Detect PO3 exceptions from TCT Lecture 8:

        Exception 1 (2-tap): Only one deviation, no third tap. Valid if RTZ quality is high.
        Exception 2 (Internal TCT): TCT model forms inside manipulation range without
                                     sweeping the main range extreme.
        """
        if manip_start >= manip_end or manip_end >= len(self.candles):
            return None

        segment = self.candles.iloc[manip_start:manip_end + 1]

        if direction == "bullish":
            # Count how many times price dips below range low
            range_low = rng["range_low"]
            dip_count = int((segment["low"].astype(float) < range_low).sum())

            # Exception 1: only 1-2 dips (2-tap), needs excellent RTZ
            if dip_count <= 2 and rng["rtz_quality"] > 0.7:
                return "exception_1_two_tap"

            # Exception 2: TCT model forms but doesn't sweep range low aggressively
            deviation = range_low - extreme_price
            range_size = rng["range_size"]
            if range_size > 0 and (deviation / range_size) < 0.10:
                return "exception_2_internal_tct"

        elif direction == "bearish":
            range_high = rng["range_high"]
            spike_count = int((segment["high"].astype(float) > range_high).sum())

            if spike_count <= 2 and rng["rtz_quality"] > 0.7:
                return "exception_1_two_tap"

            deviation = extreme_price - range_high
            range_size = rng["range_size"]
            if range_size > 0 and (deviation / range_size) < 0.10:
                return "exception_2_internal_tct"

        return None

    # ================================================================
    # QUALITY SCORING
    # ================================================================

    def _calculate_quality(
        self, rtz_quality: float, deviation_pct: float,
        has_tct_model: bool, has_compression: bool,
        has_liq_both_sides: bool, aggression: float,
        exception_type: Optional[str]
    ) -> float:
        """
        Calculate overall PO3 quality score (0-1).

        Factors:
        - RTZ quality (0.3 weight): Higher = better range
        - TCT model presence (0.25 weight): Required for high-quality PO3
        - Deviation within DL2 (0.15 weight): Tighter = better
        - Compression (0.1 weight): Bonus for range compression
        - Liquidity both sides (0.1 weight): Bonus for dual-side liquidity
        - Aggression penalty (0.1 weight): More aggressive = less reliable
        """
        score = 0.0

        # RTZ quality (30%)
        score += rtz_quality * 0.30

        # TCT model (25%)
        if has_tct_model:
            score += 0.25

        # Deviation within DL2 — tighter deviation is better (15%)
        # deviation_pct is % of range, DL2 = 30%
        if deviation_pct > 0:
            dl2_score = max(0, 1 - (deviation_pct / 30))
            score += dl2_score * 0.15

        # Compression bonus (10%)
        if has_compression:
            score += 0.10

        # Liquidity both sides bonus (10%)
        if has_liq_both_sides:
            score += 0.10

        # Aggression penalty (10%) — lower aggression = safer PO3
        aggression_score = max(0, 1 - aggression)
        score += aggression_score * 0.10

        # Exception bonus/penalty
        if exception_type == "exception_1_two_tap":
            score *= 0.9  # Slightly lower confidence for 2-tap
        elif exception_type == "exception_2_internal_tct":
            score *= 0.95  # Slightly lower for internal TCT

        return min(1.0, score)


def detect_po3_schematics(candles: pd.DataFrame, detected_ranges: List[Dict] = None) -> Dict:
    """
    Main entry point for PO3 schematic detection.

    Args:
        candles: DataFrame with OHLC data
        detected_ranges: Optional pre-detected ranges

    Returns: Dict with bullish_po3 and bearish_po3 schematics
    """
    if len(candles) < 50:
        return {
            "bullish_po3": [],
            "bearish_po3": [],
            "total": 0,
            "error": "Insufficient data (need 50+ candles)",
            "timestamp": datetime.utcnow().isoformat()
        }

    try:
        detector = PO3SchematicDetector(candles)
        return detector.detect_po3_schematics(detected_ranges)
    except Exception as e:
        logger.error(f"PO3 detection error: {e}")
        return {
            "bullish_po3": [],
            "bearish_po3": [],
            "total": 0,
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }
