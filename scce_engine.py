"""
scce_engine.py — Structural Context Continuity Engine (SCCE)
=============================================================
Persistent structural memory for TCT setup development tracking.

Tracks evolving range formation, tap progression, and BOS state across
scan cycles so the engine builds continuous awareness instead of
recomputing everything from scratch.

SHADOW MODE ONLY: does not control live entries. Observational layer.

Feature flags:
  SCCE_ENABLED     = true  → runs and records
  SCCE_SHADOW_MODE = true  → does NOT alter entry decisions
"""

import logging
import os
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Feature flags ────────────────────────────────────────────────
SCCE_ENABLED = os.getenv("SCCE_ENABLED", "true").lower() == "true"
SCCE_SHADOW_MODE = os.getenv("SCCE_SHADOW_MODE", "true").lower() == "true"

# ── Staleness thresholds ─────────────────────────────────────────
MAX_AGE_BARS = 50       # candidate goes stale after this many bars without update
MAX_CANDIDATES = 10     # per symbol, keep top N candidates


# ═══════════════════════════════════════════════════════════════════
# SCCE Candidate State
# ═══════════════════════════════════════════════════════════════════

def _new_candidate(symbol: str, timeframe: str, model_family: str,
                   range_high: float, range_low: float) -> Dict:
    """Create a fresh SCCE candidate."""
    ts = datetime.now(timezone.utc).isoformat()
    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "candidate_id": f"{symbol}_{timeframe}_{int(time.time())}",
        "model_family": model_family,

        "range_high": range_high,
        "range_low": range_low,
        "equilibrium": round((range_high + range_low) / 2, 4) if range_high and range_low else None,
        "range_valid": range_high > range_low if range_high and range_low else False,

        "tap1_detected": False, "tap1_price": None, "tap1_time": None,
        "tap2_detected": False, "tap2_price": None, "tap2_time": None,
        "tap3_detected": False, "tap3_price": None, "tap3_time": None,

        "compression_active": False,
        "bos_pending": False,
        "bos_detected": False,
        "bos_price": None,
        "bos_time": None,

        "phase": "seed",
        "last_updated": ts,
        "created_at": ts,
        "age_bars": 0,
        "stale": False,
        "invalidation_reason": None,
    }


def _advance_phase(candidate: Dict) -> str:
    """Determine the current phase from tap/BOS state."""
    if candidate.get("invalidation_reason"):
        return "invalidated"
    if candidate.get("bos_detected"):
        return "qualified"
    if candidate.get("tap3_detected") and candidate.get("bos_pending"):
        return "bos_pending"
    if candidate.get("tap3_detected"):
        return "tap3"
    if candidate.get("tap2_detected"):
        return "tap2"
    if candidate.get("tap1_detected"):
        return "tap1"
    return "seed"


# ═══════════════════════════════════════════════════════════════════
# SCCE Engine
# ═══════════════════════════════════════════════════════════════════

class SCCEEngine:
    """Structural Context Continuity Engine — shadow-mode structural memory."""

    def __init__(self):
        self.candidates: Dict[str, List[Dict]] = {}  # key: "SYMBOL_TF"
        self.event_history: List[Dict] = []  # rolling event log
        self._max_events = 50

    def _key(self, symbol: str, tf: str) -> str:
        return f"{symbol}_{tf}"

    def _log_event(self, symbol: str, tf: str, event: str, detail: str = ""):
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": symbol,
            "tf": tf,
            "event": event,
            "detail": detail,
        }
        self.event_history.append(entry)
        if len(self.event_history) > self._max_events:
            self.event_history = self.event_history[-self._max_events:]
        logger.debug("[SCCE] %s %s: %s %s", symbol, tf, event, detail)

    def update_from_schematics(self, symbol: str, tf: str,
                                schematics: List[Dict],
                                current_price: float):
        """Ingest detected schematics and update SCCE state.

        Called once per symbol/tf per scan cycle.
        """
        if not SCCE_ENABLED:
            return

        key = self._key(symbol, tf)
        existing = self.candidates.get(key, [])

        # Age existing candidates
        for c in existing:
            c["age_bars"] += 1
            if c["age_bars"] > MAX_AGE_BARS and not c.get("stale"):
                c["stale"] = True
                c["invalidation_reason"] = "aged_stale"
                c["phase"] = "invalidated"
                self._log_event(symbol, tf, "invalidated", "aged stale")

        # Process incoming schematics
        for s in schematics:
            if not isinstance(s, dict):
                continue

            direction = s.get("direction", "unknown")
            model = s.get("model", "unknown")
            rng = s.get("range") or {}
            r_high = rng.get("high")
            r_low = rng.get("low")

            # Determine model family
            if "accumulation" in model.lower() or direction == "bullish":
                family = "accumulation"
            elif "distribution" in model.lower() or direction == "bearish":
                family = "distribution"
            elif "continuation" in model.lower():
                family = "continuation"
            else:
                family = "unknown"

            # Try to match to existing candidate by range overlap
            matched = None
            for c in existing:
                if c.get("phase") == "invalidated":
                    continue
                if c.get("model_family") != family:
                    continue
                # Range overlap check
                if (c.get("range_high") and r_high and c.get("range_low") and r_low
                        and abs(c["range_high"] - r_high) / max(r_high, 1) < 0.02
                        and abs(c["range_low"] - r_low) / max(r_low, 1) < 0.02):
                    matched = c
                    break

            if matched:
                # Update existing candidate
                matched["age_bars"] = 0
                matched["stale"] = False
                matched["last_updated"] = datetime.now(timezone.utc).isoformat()
            else:
                # Seed new candidate
                c = _new_candidate(symbol, tf, family, r_high, r_low)
                existing.append(c)
                self._log_event(symbol, tf, "candidate_seeded", f"{family} range=[{r_low},{r_high}]")

            target = matched or existing[-1]

            # Update tap progression
            tap1 = s.get("tap1")
            tap2 = s.get("tap2")
            tap3 = s.get("tap3")

            if tap1 and isinstance(tap1, dict) and not target["tap1_detected"]:
                target["tap1_detected"] = True
                target["tap1_price"] = tap1.get("price")
                target["tap1_time"] = datetime.now(timezone.utc).isoformat()
                self._log_event(symbol, tf, "tap1_detected", f"price={tap1.get('price')}")

            if tap2 and isinstance(tap2, dict) and not target["tap2_detected"]:
                target["tap2_detected"] = True
                target["tap2_price"] = tap2.get("price")
                target["tap2_time"] = datetime.now(timezone.utc).isoformat()
                self._log_event(symbol, tf, "tap2_detected", f"price={tap2.get('price')}")

            if tap3 and isinstance(tap3, dict) and not target["tap3_detected"]:
                target["tap3_detected"] = True
                target["tap3_price"] = tap3.get("price")
                target["tap3_time"] = datetime.now(timezone.utc).isoformat()
                target["bos_pending"] = True
                self._log_event(symbol, tf, "tap3_detected", f"price={tap3.get('price')}")

            # BOS detection
            bos = s.get("bos_confirmation") or {}
            if bos.get("bos_price") and not target["bos_detected"]:
                target["bos_detected"] = True
                target["bos_price"] = bos.get("bos_price")
                target["bos_time"] = datetime.now(timezone.utc).isoformat()
                target["bos_pending"] = False
                self._log_event(symbol, tf, "bos_detected", f"price={bos.get('bos_price')}")

            # Update phase
            target["phase"] = _advance_phase(target)

        # Prune: keep only active + recent candidates
        active = [c for c in existing if c.get("phase") != "invalidated"]
        invalidated = [c for c in existing if c.get("phase") == "invalidated"]
        existing = active[-MAX_CANDIDATES:] + invalidated[-3:]

        self.candidates[key] = existing

    def get_active_candidates(self, symbol: str = None) -> List[Dict]:
        """Return all non-invalidated candidates, optionally filtered by symbol."""
        result = []
        for key, cands in self.candidates.items():
            for c in cands:
                if c.get("phase") != "invalidated":
                    if symbol is None or c.get("symbol") == symbol:
                        result.append(c)
        return result

    def get_snapshot(self) -> Dict:
        """Return full SCCE state for API/dashboard."""
        active = self.get_active_candidates()
        return {
            "enabled": SCCE_ENABLED,
            "shadow_mode": SCCE_SHADOW_MODE,
            "total_candidates": sum(len(v) for v in self.candidates.values()),
            "active_candidates": len(active),
            "top_candidates": sorted(active, key=lambda c: c.get("age_bars", 999))[:10],
            "event_history": self.event_history[-20:],
            "context_readiness": {
                "scce_ready": len(active) > 0,
            },
        }


# ── Module-level singleton ───────────────────────────────────────
_scce_instance: Optional[SCCEEngine] = None


def get_scce() -> SCCEEngine:
    """Get or create the SCCE singleton."""
    global _scce_instance
    if _scce_instance is None:
        _scce_instance = SCCEEngine()
        logger.info("[SCCE] Engine initialized (enabled=%s shadow=%s)", SCCE_ENABLED, SCCE_SHADOW_MODE)
    return _scce_instance
