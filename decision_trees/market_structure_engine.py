"""
market_structure_engine.py

Institution-grade market structure engine for TCT / HPM architecture.

Provides:

L1 structure detection (HTF trend)
L2 counter structure detection
L3 execution structure confirmation
liquidity pool mapping
sweep detection
pivot detection
range integrity gate

Compatible with pandas OHLC dataframe:
columns required: open, high, low, close
"""

import warnings
from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd


# ============================================================
# Helper Data Structures
# ============================================================

@dataclass
class StructureResult:
    trend: str
    hh: int
    hl: int
    lh: int
    ll: int


@dataclass
class SweepResult:
    swept: bool
    classification: str
    returned_inside: bool
    sweep_count: int


@dataclass
class StructuralAcceptance:
    """Result of post-sweep structural confirmation via L1/L2."""
    confirmed: Optional[bool]   # True=confirmed, False=rejected, None=unavailable
    reason: str
    l1_trend: Optional[str] = None


# ============================================================
# Main Engine
# ============================================================

class MarketStructureEngine:

    def __init__(self) -> None:
        pass

    # ========================================================
    # Pivot Detection
    # ========================================================

    def detect_pivots(
        self, df: pd.DataFrame, window: int = 3
    ) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:

        highs = df["high"].values
        lows = df["low"].values

        pivot_highs: list[tuple[int, float]] = []
        pivot_lows: list[tuple[int, float]] = []

        for i in range(window, len(df) - window):

            high_slice = highs[i - window:i + window + 1]
            low_slice = lows[i - window:i + window + 1]

            # Tie-break: only the first candle of a plateau is a pivot.
            # Require highs[i] to be strictly greater than every candle
            # in the left half of the window so equal adjacent highs only
            # produce one pivot (the leftmost occurrence).
            if highs[i] == max(high_slice) and highs[i] > max(highs[i - window:i]):
                pivot_highs.append((i, float(highs[i])))

            if lows[i] == min(low_slice) and lows[i] < min(lows[i - window:i]):
                pivot_lows.append((i, float(lows[i])))

        return pivot_highs, pivot_lows


    # ========================================================
    # L1 Structure Detection (HTF)
    # ========================================================

    def detect_l1_structure(self, df: pd.DataFrame) -> StructureResult:

        pivot_highs, pivot_lows = self.detect_pivots(df)

        hh = hl = lh = ll = 0

        for i in range(1, len(pivot_highs)):
            if pivot_highs[i][1] > pivot_highs[i-1][1]:
                hh += 1
            else:
                lh += 1

        for i in range(1, len(pivot_lows)):
            if pivot_lows[i][1] > pivot_lows[i-1][1]:
                hl += 1
            else:
                ll += 1

        if hh >= 2 and hl >= 2:
            trend = "bullish"
        elif lh >= 2 and ll >= 2:
            trend = "bearish"
        else:
            trend = "neutral"

        return StructureResult(trend, hh, hl, lh, ll)


    # ========================================================
    # L2 Counter Structure Detection
    # ========================================================

    def detect_l2_structure(self, df: pd.DataFrame, htf_bias: str):

        recent = df.tail(30)

        highs = recent["high"].values
        lows = recent["low"].values

        lh = ll = hh = hl = 0

        for i in range(1, len(highs)):
            if highs[i] < highs[i-1]:
                lh += 1
            if highs[i] > highs[i-1]:
                hh += 1

        for i in range(1, len(lows)):
            if lows[i] < lows[i-1]:
                ll += 1
            if lows[i] > lows[i-1]:
                hl += 1

        if htf_bias == "bullish":
            exists = lh >= 2 and ll >= 2
        elif htf_bias == "bearish":
            exists = hh >= 2 and hl >= 2
        else:
            exists = False

        return {
            "exists": exists,
            "lh": lh,
            "ll": ll,
            "hh": hh,
            "hl": hl
        }


    # ========================================================
    # L3 Execution Structure (Micro BOS)
    # ========================================================

    def detect_l3_structure(self, df: pd.DataFrame, direction: str,
                            relaxed_bos_tolerance: float = 0.0) -> bool:
        """Detect L3 execution structure (compression + micro-BOS).

        Args:
            relaxed_bos_tolerance: When > 0, accept micro-BOS if close is
                within this fraction of the breakout level (e.g. 0.0015 = 0.15%).
                Only used for ETH-only live override. Default 0.0 = strict.

        Also populates self.last_l3_trace for diagnostic instrumentation.
        """
        self.last_l3_trace = {
            "direction": direction,
            "candles_available": len(df) if df is not None else 0,
            "compression_count": 0,
            "compression_required": 3,
            "compression_pass": False,
            "micro_bos_pass": False,
            "micro_bos_relaxed_pass": False,
            "bos_distance_pct": 0.0,
            "l3_pass": False,
            "first_failed": None,
            "relaxed_bos_tolerance": relaxed_bos_tolerance,
        }

        if df is None or len(df) < 10:
            self.last_l3_trace["first_failed"] = "insufficient_candles"
            return False

        recent = df.tail(10)
        highs = recent["high"].values
        lows = recent["low"].values
        closes = recent["close"].values

        compression = 0

        if direction == "bullish":
            for i in range(1, len(lows)):
                if lows[i] > lows[i - 1]:
                    compression += 1
            prev_high = max(highs[:-1])
            broke_structure = closes[-1] > prev_high
            # Distance from breakout level (negative = hasn't broken yet)
            bos_dist = (closes[-1] - prev_high) / prev_high if prev_high > 0 else 0
            relaxed_pass = closes[-1] >= prev_high * (1 - relaxed_bos_tolerance) if relaxed_bos_tolerance > 0 else False
        else:
            for i in range(1, len(highs)):
                if highs[i] < highs[i - 1]:
                    compression += 1
            prev_low = min(lows[:-1])
            broke_structure = closes[-1] < prev_low
            bos_dist = (prev_low - closes[-1]) / prev_low if prev_low > 0 else 0
            relaxed_pass = closes[-1] <= prev_low * (1 + relaxed_bos_tolerance) if relaxed_bos_tolerance > 0 else False

        comp_pass = compression >= 3
        self.last_l3_trace["compression_count"] = compression
        self.last_l3_trace["compression_pass"] = comp_pass
        self.last_l3_trace["micro_bos_pass"] = broke_structure
        self.last_l3_trace["micro_bos_relaxed_pass"] = relaxed_pass
        self.last_l3_trace["bos_distance_pct"] = round(bos_dist * 100, 4)

        # Strict pass
        if comp_pass and broke_structure:
            self.last_l3_trace["l3_pass"] = True
            return True

        # Relaxed pass (only when tolerance is set and compression passes)
        if comp_pass and relaxed_pass and relaxed_bos_tolerance > 0:
            self.last_l3_trace["l3_pass"] = True
            self.last_l3_trace["micro_bos_pass"] = True  # mark as passed via relaxed rule
            return True

        # Record which failed
        if not comp_pass and not broke_structure:
            self.last_l3_trace["first_failed"] = "both"
        elif not comp_pass:
            self.last_l3_trace["first_failed"] = "compression"
        else:
            self.last_l3_trace["first_failed"] = "micro_bos"

        return False

    # ========================================================
    # Liquidity Pool Detection
    # ========================================================

    def detect_liquidity_pools(self, df: pd.DataFrame) -> dict[str, list[float]]:

        highs = df["high"].values
        lows = df["low"].values

        pools = {
            "equal_highs": [],
            "equal_lows": [],
            "swing_highs": [],
            "swing_lows": []
        }

        for i in range(2, len(df) - 2):

            if highs[i] > highs[i-1] and highs[i] > highs[i+1]:
                pools["swing_highs"].append(highs[i])

            if lows[i] < lows[i-1] and lows[i] < lows[i+1]:
                pools["swing_lows"].append(lows[i])

            if abs(highs[i] - highs[i-1]) < highs[i]*0.001:
                pools["equal_highs"].append(highs[i])

            if abs(lows[i] - lows[i-1]) < lows[i]*0.001:
                pools["equal_lows"].append(lows[i])

        return pools


    # ========================================================
    # Liquidity Sweep Detection
    # ========================================================

    def detect_sweep(self, df, range_high, range_low, direction):

        if df is None or len(df) < 10:
            return SweepResult(swept=False, classification="no_sweep",
                               returned_inside=False, sweep_count=0)

        recent = df.tail(20)

        highs = recent["high"].values
        lows = recent["low"].values
        closes = recent["close"].values

        sweep_count = 0
        returned = False

        # =========================
        # BULLISH (sell-side sweep)
        # =========================
        if direction == "bullish":

            for i in range(len(lows)):
                if lows[i] < range_low:

                    sweep_count = 1

                    if i < len(closes) - 1:
                        if closes[i + 1] >= range_low:
                            returned = True
                        else:
                            returned = False
                    else:
                        returned = False

                    break

        # =========================
        # BEARISH (buy-side sweep)
        # =========================
        else:

            for i in range(len(highs)):
                if highs[i] > range_high:

                    sweep_count = 1

                    if i < len(closes) - 1:
                        if closes[i + 1] <= range_high:
                            returned = True
                        else:
                            returned = False
                    else:
                        returned = False

                    break

        # =========================
        # CLASSIFICATION
        # =========================
        swept = sweep_count > 0

        if swept and returned:
            classification = "liquidity_grab"
        elif swept:
            classification = "true_break"
        else:
            classification = "no_sweep"

        return SweepResult(
            swept=swept,
            classification=classification,
            returned_inside=returned,
            sweep_count=sweep_count
        )


    # ========================================================
    # Post-Sweep Structural Acceptance (L1/L2)
    # ========================================================

    def confirm_structure_after_sweep(
        self,
        df: Optional[pd.DataFrame],
        sweep_side: str,
        htf_bias: Optional[str] = None,
    ) -> StructuralAcceptance:
        """Validate acceptance back inside range via L1 structure after a sweep.

        Args:
            df: OHLC DataFrame for structure detection (None → unavailable)
            sweep_side: "buy_side" or "sell_side" — which side was swept
            htf_bias: optional HTF bias for L2 counter-structure check

        Returns:
            StructuralAcceptance with confirmed=True/False/None
        """
        # TODO: htf_bias reserved for future L2 counter-structure validation.
        # Planned use: verify that post-sweep structure aligns or conflicts
        # with HTF directional bias (e.g., confirming reversal vs continuation).

        if df is None or len(df) < 10:
            return StructuralAcceptance(
                confirmed=None,
                reason="Insufficient data for structural confirmation",
            )

        l1 = self.detect_l1_structure(df)

        # Buy-side swept → expect bearish structure (LH+LL) for short acceptance
        # Sell-side swept → expect bullish structure (HH+HL) for long acceptance
        if sweep_side == "buy_side":
            confirmed = l1.trend == "bearish"
            expected = "bearish"
        elif sweep_side == "sell_side":
            confirmed = l1.trend == "bullish"
            expected = "bullish"
        else:
            return StructuralAcceptance(
                confirmed=None,
                reason=f"Unknown sweep_side: {sweep_side}",
                l1_trend=l1.trend,
            )

        if confirmed:
            reason = f"L1 {l1.trend} structure confirms {expected} acceptance"
        elif l1.trend == "neutral":
            # Neutral is not confirmation — treat as unavailable (pending)
            return StructuralAcceptance(
                confirmed=None,
                reason=f"L1 neutral — structural confirmation pending (expected {expected})",
                l1_trend=l1.trend,
            )
        else:
            reason = f"L1 {l1.trend} contradicts expected {expected} acceptance"

        return StructuralAcceptance(
            confirmed=confirmed,
            reason=reason,
            l1_trend=l1.trend,
        )

    # ========================================================
    # Range Integrity Gate
    # ========================================================

    def range_integrity_gate(self,
                             range_high: float,
                             range_low: float,
                             current_price: float,
                             threshold: float = 0.10):
        """EQ proximity check — returns True always (no longer blocks).

        Equilibrium proximity is handled as a confidence penalty in
        _compute_rig_payload (decision_tree_bridge.py), not as a gate.
        Kept for API compatibility — always passes.
        """
        warnings.warn(
            "range_integrity_gate is deprecated — use RIG confidence penalty instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return True
