"""
phemex_tct_algo.py — 6-Gate TCT Signal Pipeline for Phemex

Implements the full TCT entry qualification sequence as 6 sequential gate
functions, one per lecture layer. Each gate is a pure function: it takes
read-only DataFrames and returns a GateResult. Mutation of the input
DataFrames is a bug — gates that need to annotate data call .copy() explicitly
at that point.

Pipeline sequence:
  Gate 1: Market Structure  (Lecture 1) — trend, BOS, RTZ on LTF
  Gate 2: Ranges            (Lecture 2) — valid consolidation on HTF
  Gate 3: Supply & Demand   (Lecture 3) — OB/demand zone aligned with bias
  Gate 4: Liquidity         (Lecture 4) — tap/sweep at zone level on LTF
  Gate 5: TCT Schematics    (Lecture 5) — Model 1/2 pattern on LTF
  Gate 6: Advanced TCT      (Lecture 6) — final confluence + R:R signal

The pipeline short-circuits on the first failing gate. Gate 6 emits the
final LONG / SHORT / NO_TRADE signal.

Usage:
    rules = load_tct_rules()         # once at startup
    candles = phemex_feed.fetch_all()
    result = run_pipeline(candles["4h"], candles["1h"], candles["15m"], rules)
    if result.signal in ("LONG", "SHORT"):
        ...
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from market_structure import MarketStructure
from tct_schematics import detect_tct_schematics
from tct_pdf_rules import TCTRuleSet, LayerRules

logger = logging.getLogger("TCT-Algo")


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class GateResult:
    layer: int
    name: str
    passed: bool
    data: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return f"GateResult(layer={self.layer}, name={self.name!r}, status={status})"


@dataclass
class PipelineResult:
    """Final output of the 6-gate pipeline."""

    signal: str          # "LONG", "SHORT", or "NO_TRADE"
    confidence: float    # 0.0–1.0
    entry: float
    stop: float
    target: float
    rr: float            # realized R:R ratio
    gate_results: list[GateResult] = field(default_factory=list)
    blocking_gate: Optional[int] = None   # layer number of first failing gate

    @property
    def is_trade(self) -> bool:
        return self.signal in ("LONG", "SHORT")


_NO_TRADE = PipelineResult(
    signal="NO_TRADE",
    confidence=0.0,
    entry=0.0,
    stop=0.0,
    target=0.0,
    rr=0.0,
)


# ---------------------------------------------------------------------------
# Gate thresholds (deterministic — never adapt)
# ---------------------------------------------------------------------------

# Gate 2: minimum fraction of candles needed at range extremes
RANGE_LOOKBACK = 50          # HTF candles to evaluate for range
RANGE_MAX_SIZE_PCT = 0.05    # Range must be < 5% of mid-price
RANGE_MIN_TOUCHES = 2        # Each side needs >= 2 touches

# Gate 3: impulse detection — need this many candles in same direction
SD_IMPULSE_CANDLES = 3
SD_IMPULSE_PCT = 0.005       # Each impulse candle moves >= 0.5% of price

# Gate 4: zone proximity tolerance
LIQUIDITY_ZONE_TOLERANCE_PCT = 0.003   # Within 0.3% of zone counts as a tap

# Gate 5: minimum schematic quality
SCHEMATIC_MIN_QUALITY = 0.6

# Gate 6: minimum R:R and confidence for a valid signal
FINAL_MIN_RR = 2.0
FINAL_MIN_CONFIDENCE = 0.65


# ---------------------------------------------------------------------------
# Gate 1: Market Structure (Lecture 1)
# ---------------------------------------------------------------------------

def _gate_1_market_structure(
    ltf: pd.DataFrame,
    rules: LayerRules,  # noqa: ARG001 — reserved for rule-guided filtering
) -> GateResult:
    """
    Layer 1: Confirm a clear directional trend with at least one BOS and a
    valid RTZ on the LTF timeframe.

    Pass conditions:
    - Trend is 'bullish' or 'bearish' (not 'neutral')
    - At least one BOS event detected
    - RTZ is valid (clean pullback zone for entry)
    """
    ms = MarketStructure.detect_pivots(ltf)
    trend = ms["trend"]
    bos_events = ms["bos_events"]
    rtz = ms["rtz"]
    eof = ms["eof"]

    passed = (
        trend in ("bullish", "bearish")
        and len(bos_events) > 0
        and rtz.get("valid", False)
    )

    logger.debug(
        "Gate 1 | trend=%s bos=%d rtz_valid=%s → %s",
        trend, len(bos_events), rtz.get("valid"), "PASS" if passed else "FAIL",
    )

    return GateResult(
        layer=1,
        name="Market Structure",
        passed=passed,
        data={
            "trend": trend,
            "bos_count": len(bos_events),
            "bos_events": bos_events,
            "rtz": rtz,
            "eof": eof,
        },
    )


# ---------------------------------------------------------------------------
# Gate 2: Ranges (Lecture 2)
# ---------------------------------------------------------------------------

def _detect_range(htf: pd.DataFrame) -> dict:
    """
    Detect a valid consolidation range in the most recent HTF candles.

    A valid range requires:
    - At least RANGE_LOOKBACK candles
    - Total high-low size < RANGE_MAX_SIZE_PCT of mid-price
    - At least RANGE_MIN_TOUCHES at each extreme

    Returns dict with keys: valid, range_high, range_low, range_size_pct,
    high_touches, low_touches.
    """
    n = min(RANGE_LOOKBACK, len(htf))
    if n < 10:
        return {"valid": False, "range_high": 0.0, "range_low": 0.0}

    # Work on a slice — no copy needed since we only read
    window = htf.iloc[-n:]
    rng_high = float(window["high"].max())
    rng_low = float(window["low"].min())
    mid = (rng_high + rng_low) / 2.0

    if mid == 0:
        return {"valid": False, "range_high": 0.0, "range_low": 0.0}

    size_pct = (rng_high - rng_low) / mid

    # Count touches: candle high within tolerance of range high, etc.
    tol = mid * LIQUIDITY_ZONE_TOLERANCE_PCT
    high_touches = int((window["high"] >= rng_high - tol).sum())
    low_touches = int((window["low"] <= rng_low + tol).sum())

    valid = (
        size_pct < RANGE_MAX_SIZE_PCT
        and high_touches >= RANGE_MIN_TOUCHES
        and low_touches >= RANGE_MIN_TOUCHES
    )

    return {
        "valid": valid,
        "range_high": rng_high,
        "range_low": rng_low,
        "range_size_pct": round(size_pct * 100, 2),
        "high_touches": high_touches,
        "low_touches": low_touches,
    }


def _gate_2_ranges(
    htf: pd.DataFrame,
    rules: LayerRules,  # noqa: ARG001
) -> GateResult:
    """
    Layer 2: Confirm a valid consolidation range exists on the HTF timeframe.

    Pass conditions: see _detect_range().
    """
    rng = _detect_range(htf)

    logger.debug(
        "Gate 2 | valid=%s size=%.2f%% high_touches=%s low_touches=%s → %s",
        rng.get("valid"),
        rng.get("range_size_pct", 0),
        rng.get("high_touches"),
        rng.get("low_touches"),
        "PASS" if rng.get("valid") else "FAIL",
    )

    return GateResult(
        layer=2,
        name="Ranges",
        passed=bool(rng.get("valid", False)),
        data=rng,
    )


# ---------------------------------------------------------------------------
# Gate 3: Supply & Demand (Lecture 3)
# ---------------------------------------------------------------------------

def _find_zone(mtf: pd.DataFrame, bias: str) -> Optional[dict]:
    """
    Find the most recent supply or demand zone on the MTF.

    For bullish bias (demand zone): last bearish candle before SD_IMPULSE_CANDLES
    consecutive bullish candles each moving >= SD_IMPULSE_PCT.

    For bearish bias (supply zone): last bullish candle before SD_IMPULSE_CANDLES
    consecutive bearish candles each moving >= SD_IMPULSE_PCT.
    """
    closes = mtf["close"].values
    opens = mtf["open"].values
    highs = mtf["high"].values
    lows = mtf["low"].values
    n = len(mtf)

    bullish = bias == "bullish"
    zone_type = "demand" if bullish else "supply"

    for i in range(n - SD_IMPULSE_CANDLES - 1, 0, -1):
        # OB candle must be opposite to the impulse direction
        ob_is_bearish = closes[i] < opens[i]
        if bullish and not ob_is_bearish:
            continue
        if not bullish and ob_is_bearish:
            continue

        # Subsequent candles must form an impulse in the bias direction
        impulse_ok = True
        for j in range(i + 1, i + 1 + SD_IMPULSE_CANDLES):
            if j >= n:
                impulse_ok = False
                break
            if bullish:
                candle_pct = (closes[j] - opens[j]) / (opens[j] + 1e-9)
                if closes[j] <= opens[j] or candle_pct < SD_IMPULSE_PCT:
                    impulse_ok = False
                    break
            else:
                candle_pct = (opens[j] - closes[j]) / (opens[j] + 1e-9)
                if closes[j] >= opens[j] or candle_pct < SD_IMPULSE_PCT:
                    impulse_ok = False
                    break

        if impulse_ok:
            return {
                "zone_type": zone_type,
                "zone_high": float(highs[i]),
                "zone_low": float(lows[i]),
                "candle_idx": i,
            }

    return None


def _gate_3_supply_demand(
    mtf: pd.DataFrame,
    bias: str,
    rules: LayerRules,  # noqa: ARG001
) -> GateResult:
    """
    Layer 3: Detect a supply or demand zone on the MTF aligned with the bias.

    For bullish bias: find a demand zone (last bearish OB before bullish impulse).
    For bearish bias: find a supply zone (last bullish OB before bearish impulse).

    Pass conditions: a valid zone was found matching the bias.
    """
    zone: Optional[dict] = None

    if bias in ("bullish", "bearish"):
        zone = _find_zone(mtf, bias)

    passed = zone is not None

    logger.debug(
        "Gate 3 | bias=%s zone=%s → %s",
        bias,
        zone.get("zone_type") if zone else "None",
        "PASS" if passed else "FAIL",
    )

    return GateResult(
        layer=3,
        name="Supply & Demand",
        passed=passed,
        data=zone if zone else {},
    )


# ---------------------------------------------------------------------------
# Gate 4: Liquidity (Lecture 4)
# ---------------------------------------------------------------------------

def _gate_4_liquidity(
    ltf: pd.DataFrame,
    zone: dict,
    bias: str,
    rules: LayerRules,  # noqa: ARG001
) -> GateResult:
    """
    Layer 4: Confirm that price has tapped (wicked into) the supply/demand zone
    in the LTF candles.

    For demand zones (bullish bias): a tap is any LTF candle whose low is at or
    below zone_high + tolerance — price entered the zone.
    For supply zones (bearish bias): a tap is any LTF candle whose high is at or
    above zone_low - tolerance — price entered the zone.

    Pass conditions:
    - At least one tap detected within the last 50 LTF candles
    - The most recent tap is within the last 30 candles (not stale)
    """
    if not zone:
        return GateResult(layer=4, name="Liquidity", passed=False,
                          data={"error": "no zone from Gate 3"})

    zone_high = zone["zone_high"]
    zone_low = zone["zone_low"]
    mid = (zone_high + zone_low) / 2.0
    tol = mid * LIQUIDITY_ZONE_TOLERANCE_PCT

    window = ltf.iloc[-50:]
    tap_indices = []

    if bias == "bullish":
        # Price dipped into demand zone: low <= zone_high + tol
        for idx, row in enumerate(window.itertuples(index=False)):
            if row.low <= zone_high + tol:
                tap_indices.append(idx)
    else:
        # Price pushed into supply zone: high >= zone_low - tol
        for idx, row in enumerate(window.itertuples(index=False)):
            if row.high >= zone_low - tol:
                tap_indices.append(idx)

    passed = len(tap_indices) > 0 and (len(window) - 1 - tap_indices[-1]) <= 30

    tap_price: Optional[float] = None
    if tap_indices:
        last = tap_indices[-1]
        row = window.iloc[last]
        tap_price = float(row["low"] if bias == "bullish" else row["high"])

    logger.debug(
        "Gate 4 | bias=%s taps=%d last_tap_age=%s → %s",
        bias,
        len(tap_indices),
        (len(window) - 1 - tap_indices[-1]) if tap_indices else "N/A",
        "PASS" if passed else "FAIL",
    )

    return GateResult(
        layer=4,
        name="Liquidity",
        passed=passed,
        data={
            "tap_count": len(tap_indices),
            "tap_price": tap_price,
            "tap_indices": tap_indices,
        },
    )


# ---------------------------------------------------------------------------
# Gate 5: TCT Schematics (Lecture 5)
# ---------------------------------------------------------------------------

def _gate_5_schematics(
    ltf: pd.DataFrame,
    bias: str,
    rules: LayerRules,  # noqa: ARG001
) -> GateResult:
    """
    Layer 5: Confirm a valid TCT Model 1 or Model 2 schematic on the LTF,
    aligned with the directional bias.

    Uses detect_tct_schematics() from tct_schematics.py — the existing
    Lecture 5A/5B implementation.

    Pass conditions:
    - At least one valid schematic of the correct type
    - Best schematic quality >= SCHEMATIC_MIN_QUALITY
    """
    result = detect_tct_schematics(ltf)

    if bias == "bullish":
        schematics = result.get("accumulation_schematics", [])
    else:
        schematics = result.get("distribution_schematics", [])

    # Filter to schematics that are complete (have a BOS or are validated)
    valid = [
        s for s in schematics
        if s.get("quality_score", 0.0) >= SCHEMATIC_MIN_QUALITY
    ]

    passed = len(valid) > 0
    best_score = max((s.get("quality_score", 0.0) for s in valid), default=0.0)

    logger.debug(
        "Gate 5 | bias=%s schematics=%d valid=%d best_score=%.2f → %s",
        bias, len(schematics), len(valid), best_score,
        "PASS" if passed else "FAIL",
    )

    return GateResult(
        layer=5,
        name="TCT Schematics",
        passed=passed,
        data={
            "total_schematics": len(schematics),
            "valid_count": len(valid),
            "best_score": round(best_score, 3),
            "schematics": valid[:3],  # cap payload — don't flood logs
        },
    )


# ---------------------------------------------------------------------------
# Gate 6: Advanced TCT — Final Confluence + Signal (Lecture 6)
# ---------------------------------------------------------------------------

def _compute_confidence(gate_results: list[GateResult]) -> float:
    """
    Compute a confidence score from gate results 1–5.

    Each gate contributes equally. Partial passes (gate data quality) are not
    used here — this is a simple passed/total ratio with a floor.
    """
    passed = sum(1 for g in gate_results if g.passed)
    return round(passed / len(gate_results), 3) if gate_results else 0.0


def _gate_6_advanced_tct(
    gate_results: list[GateResult],
    ltf: pd.DataFrame,
    zone: dict,
    bias: str,
    rules: LayerRules,  # noqa: ARG001
) -> GateResult:
    """
    Layer 6: Final confluence check — compute entry, stop, target, R:R.

    Entry: last close price on LTF.
    Stop:  zone_low (for LONG) or zone_high (for SHORT).
    Target: entry ± (entry - stop) × FINAL_MIN_RR.

    Pass conditions:
    - R:R >= FINAL_MIN_RR
    - Confidence >= FINAL_MIN_CONFIDENCE

    Signal: "LONG" if bullish, "SHORT" if bearish, "NO_TRADE" otherwise.
    """
    confidence = _compute_confidence(gate_results)
    last_close = float(ltf.iloc[-1]["close"])

    entry = last_close
    if bias == "bullish":
        stop = zone.get("zone_low", 0.0)
        # Stop must be below entry for a long — if the zone is above price,
        # the trade geometry is invalid.
        risk = entry - stop
    else:
        stop = zone.get("zone_high", 0.0)
        # Stop must be above entry for a short
        risk = stop - entry

    # Validate trade geometry before computing target
    stop_valid = risk > 0

    if stop_valid:
        target = (entry + risk * FINAL_MIN_RR) if bias == "bullish" else (entry - risk * FINAL_MIN_RR)
        # Round to 4 dp to avoid floating-point artifacts (e.g. 1.9999999999 instead of 2.0)
        rr = round(abs(target - entry) / risk, 4)
    else:
        target = entry
        rr = 0.0

    passed = stop_valid and rr >= FINAL_MIN_RR and confidence >= FINAL_MIN_CONFIDENCE

    logger.debug(
        "Gate 6 | bias=%s entry=%.2f stop=%.2f target=%.2f rr=%.2f conf=%.2f → %s",
        bias, entry, stop, target, rr, confidence,
        "PASS" if passed else "FAIL",
    )

    return GateResult(
        layer=6,
        name="Advanced TCT",
        passed=passed,
        data={
            "entry": round(entry, 4),
            "stop": round(stop, 4),
            "target": round(target, 4),
            "rr": round(rr, 2),
            "confidence": confidence,
            "signal": ("LONG" if bias == "bullish" else "SHORT") if passed else "NO_TRADE",
        },
    )


# ---------------------------------------------------------------------------
# Public pipeline entry point
# ---------------------------------------------------------------------------

def run_pipeline(
    htf: pd.DataFrame,
    mtf: pd.DataFrame,
    ltf: pd.DataFrame,
    rules: TCTRuleSet,
) -> PipelineResult:
    """
    Run the full 6-gate TCT signal pipeline.

    The pipeline is sequential and short-circuits on the first failing gate.
    Gates 1–6 must all pass for a LONG or SHORT signal to be emitted.

    Args:
        htf: Read-only DataFrame of HTF (4h) candles from phemex_feed.
        mtf: Read-only DataFrame of MTF (1h) candles from phemex_feed.
        ltf: Read-only DataFrame of LTF (15m) candles from phemex_feed.
        rules: TCTRuleSet loaded at startup by tct_pdf_rules.load_tct_rules().

    Returns:
        PipelineResult with signal, confidence, entry/stop/target, and all
        gate results for logging and audit.
    """
    gate_results: list[GateResult] = []

    def _bail(gate: GateResult) -> PipelineResult:
        gate_results.append(gate)
        r = PipelineResult(
            signal="NO_TRADE",
            confidence=0.0,
            entry=0.0,
            stop=0.0,
            target=0.0,
            rr=0.0,
            gate_results=gate_results,
            blocking_gate=gate.layer,
        )
        logger.info(
            "Pipeline blocked at Gate %d (%s). Signal: NO_TRADE.",
            gate.layer, gate.name,
        )
        return r

    # ── Gate 1: Market Structure ──────────────────────────────────────────
    g1 = _gate_1_market_structure(ltf, rules.get(1))
    if not g1.passed:
        return _bail(g1)
    gate_results.append(g1)

    bias: str = g1.data["trend"]  # "bullish" or "bearish"

    # ── Gate 2: Ranges ────────────────────────────────────────────────────
    g2 = _gate_2_ranges(htf, rules.get(2))
    if not g2.passed:
        return _bail(g2)
    gate_results.append(g2)

    # ── Gate 3: Supply & Demand ───────────────────────────────────────────
    g3 = _gate_3_supply_demand(mtf, bias, rules.get(3))
    if not g3.passed:
        return _bail(g3)
    gate_results.append(g3)

    zone = g3.data  # {zone_type, zone_high, zone_low, candle_idx}

    # ── Gate 4: Liquidity ─────────────────────────────────────────────────
    g4 = _gate_4_liquidity(ltf, zone, bias, rules.get(4))
    if not g4.passed:
        return _bail(g4)
    gate_results.append(g4)

    # ── Gate 5: TCT Schematics ────────────────────────────────────────────
    g5 = _gate_5_schematics(ltf, bias, rules.get(5))
    if not g5.passed:
        return _bail(g5)
    gate_results.append(g5)

    # ── Gate 6: Advanced TCT — Final Signal ───────────────────────────────
    g6 = _gate_6_advanced_tct(gate_results, ltf, zone, bias, rules.get(6))
    gate_results.append(g6)

    if not g6.passed:
        return PipelineResult(
            signal="NO_TRADE",
            confidence=g6.data.get("confidence", 0.0),
            entry=g6.data.get("entry", 0.0),
            stop=g6.data.get("stop", 0.0),
            target=g6.data.get("target", 0.0),
            rr=g6.data.get("rr", 0.0),
            gate_results=gate_results,
            blocking_gate=6,
        )

    signal = g6.data["signal"]
    result = PipelineResult(
        signal=signal,
        confidence=g6.data["confidence"],
        entry=g6.data["entry"],
        stop=g6.data["stop"],
        target=g6.data["target"],
        rr=g6.data["rr"],
        gate_results=gate_results,
        blocking_gate=None,
    )

    logger.info(
        "Pipeline complete. Signal=%s entry=%.4f stop=%.4f target=%.4f "
        "rr=%.2f confidence=%.2f",
        signal, result.entry, result.stop, result.target,
        result.rr, result.confidence,
    )

    return result
