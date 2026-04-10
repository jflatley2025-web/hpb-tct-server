"""
top_range_snapshot.py — Daily Top 5 Range Snapshot (Annotation Only)
=====================================================================
Scans symbol universe for high-quality consolidation ranges, ranks by
quality score, persists top 5 to JSON for annotation/tagging use.

ZERO execution impact. Write-only persistence. Fail-open design.
No feedback into trading pipeline. No imports from execution modules.

Reuses existing range detection + scoring from server_mexc:
  - detect_ranges()  — sliding-window consolidation detection
  - score_range_quality()  — multi-factor 0-10 scoring
  - fetch_mexc_candles()  — async MEXC data fetcher
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.join(_BASE_DIR, "data")
_SNAPSHOT_PATH = os.path.join(_DATA_DIR, "top_range_snapshot.json")

# Pair sources
_LARS_PAIRS_PATH = os.path.join(_BASE_DIR, "mexc_lars_pairs.txt")
_PRIMARY_PAIRS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

# Scanner config
_RANGE_MIN_RPS = 6.5
_TOP_N = 5
_SCAN_INTERVAL_SEC = 12 * 60 * 60  # 12 hours
_RATE_LIMIT_DELAY = 0.5  # seconds between API calls


def _load_lars_pairs() -> List[str]:
    """Load symbol list from mexc_lars_pairs.txt.

    Returns MEXC API format (e.g., HYPEUSDT from HYPE_USDT_PERP).
    """
    try:
        with open(_LARS_PAIRS_PATH, "r") as f:
            raw = [line.strip() for line in f if line.strip()]
        # Convert HYPE_USDT_PERP -> HYPEUSDT
        converted = []
        for pair in raw:
            # Strip _PERP suffix, then collapse _USDT_ to USDT
            clean = pair.replace("_PERP", "").replace("_USDT", "USDT").replace("_", "")
            if clean and clean not in converted:
                converted.append(clean)
        return converted
    except Exception as e:
        logger.warning("[TOP_RANGE] Failed to load lars pairs: %s", e)
        return []


def _build_scan_universe() -> List[str]:
    """Build full scan list: Lars pairs + primary pairs, deduplicated."""
    lars = _load_lars_pairs()
    universe = list(lars)
    for p in _PRIMARY_PAIRS:
        if p not in universe:
            universe.append(p)
    return universe


async def _scan_single_pair(symbol: str) -> Optional[Dict]:
    """Scan one pair for best range. Reuses server_mexc detection + scoring.

    Returns qualified setup dict or None.
    """
    try:
        from server_mexc import fetch_mexc_candles, detect_ranges, score_range_quality

        df = await fetch_mexc_candles(symbol, "1d", 200)
        if df is None or len(df) < 30:
            return None

        # Pair age filter
        earliest = df["open_time"].iloc[0]
        latest = df["open_time"].iloc[-1]
        age_days = (latest - earliest).total_seconds() / 86400
        if age_days < 180:
            return None

        # Run detection in thread pool (CPU-bound)
        loop = asyncio.get_event_loop()
        all_ranges = await loop.run_in_executor(None, detect_ranges, df, len(df))
        if not all_ranges:
            return None

        # Score and pick best
        best_range = None
        best_score = 0.0
        for r in all_ranges:
            rps = score_range_quality(r)
            if rps > best_score:
                best_score = rps
                best_range = r

        if not best_range or best_score < _RANGE_MIN_RPS:
            return None

        return {
            "symbol": symbol,
            "timeframe": "1d",
            "range_high": round(best_range["high"], 6),
            "range_low": round(best_range["low"], 6),
            "range_eq": round(best_range["equilibrium"], 6),
            "range_strength": best_score,
            "quality": best_range["quality"],
            "eq_touched": best_range["eq_touched"],
            "candles": best_range["candles"],
            "detected_at": datetime.now(timezone.utc).isoformat(),
        }

    except Exception as e:
        logger.debug("[TOP_RANGE] Error scanning %s: %s", symbol, e)
        return None


def _persist_snapshot(data: dict) -> bool:
    """Write snapshot to JSON atomically. Returns True on success."""
    try:
        os.makedirs(_DATA_DIR, exist_ok=True)
        tmp_path = _SNAPSHOT_PATH + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        os.replace(tmp_path, _SNAPSHOT_PATH)
        return True
    except Exception as e:
        logger.warning("[TOP_RANGE] Failed to persist snapshot: %s", e)
        return False


def load_top_range_snapshot() -> List[Dict]:
    """Load top ranges from persisted JSON.

    Returns empty list on any failure. Never raises. Never blocks.
    """
    try:
        if not os.path.exists(_SNAPSHOT_PATH):
            return []
        with open(_SNAPSHOT_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        ranges = data.get("top_ranges", [])
        if not isinstance(ranges, list):
            return []
        return ranges
    except Exception:
        return []


def get_snapshot_metadata() -> Dict:
    """Return snapshot metadata (age, loaded status) for telemetry.

    Never raises.
    """
    try:
        if not os.path.exists(_SNAPSHOT_PATH):
            return {"loaded": False, "age_sec": None, "generated_at": None}
        mtime = os.path.getmtime(_SNAPSHOT_PATH)
        age = time.time() - mtime
        with open(_SNAPSHOT_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {
            "loaded": True,
            "age_sec": round(age),
            "generated_at": data.get("generated_at"),
            "pairs_scanned": data.get("pairs_scanned", 0),
            "qualified_count": data.get("qualified_count", 0),
            "top_count": len(data.get("top_ranges", [])),
        }
    except Exception:
        return {"loaded": False, "age_sec": None, "generated_at": None}


async def run_top_range_snapshot() -> Dict:
    """Run full Top 5 range scan. Persists result to JSON.

    Uses scan coordination lock if available (non-blocking).
    If lock is busy, skips this cycle.

    Returns snapshot dict (also persisted).
    """
    # Check scan lock (non-blocking) — skip if 5B engine is mid-scan
    try:
        from server_mexc import _get_scan_lock
        lock = _get_scan_lock()
        if lock.locked():
            logger.info("[TOP_RANGE] Scan lock busy — skipping this cycle")
            return {"skipped": True, "reason": "scan_lock_busy"}
    except Exception:
        pass  # No lock available — proceed without it

    universe = _build_scan_universe()
    if not universe:
        logger.warning("[TOP_RANGE] Empty scan universe — skipping")
        return {"skipped": True, "reason": "empty_universe"}

    logger.info("[TOP_RANGE] Starting snapshot scan of %d pairs", len(universe))
    start = time.time()

    qualified = []
    errors = 0

    for i, symbol in enumerate(universe):
        try:
            result = await _scan_single_pair(symbol)
            if result:
                qualified.append(result)
                errors = 0  # Reset on success
        except Exception as e:
            logger.debug("[TOP_RANGE] Exception for %s: %s", symbol, e)
            errors += 1

        # Adaptive rate limiting
        delay = _RATE_LIMIT_DELAY
        if errors > 5:
            delay = min(_RATE_LIMIT_DELAY + (errors * 0.3), 5.0)
        await asyncio.sleep(delay)

    # Sort by range_strength (RPS), take top N
    qualified.sort(key=lambda x: x.get("range_strength", 0), reverse=True)
    top_ranges = qualified[:_TOP_N]

    duration = time.time() - start

    snapshot = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scan_duration_sec": round(duration, 1),
        "pairs_scanned": len(universe),
        "qualified_count": len(qualified),
        "top_ranges": top_ranges,
    }

    persisted = _persist_snapshot(snapshot)

    symbols_str = [f"{r['symbol']}={r['range_strength']}" for r in top_ranges]
    logger.info(
        "[TOP_RANGE] Snapshot complete in %.0fs — %d pairs, %d qualified, "
        "top %d: %s (persisted=%s)",
        duration, len(universe), len(qualified), len(top_ranges),
        symbols_str, persisted,
    )

    return snapshot


async def top_range_snapshot_loop():
    """Background loop that runs the snapshot every 12 hours.

    Waits 90s on startup to avoid competing with 5B engine first scan.
    """
    await asyncio.sleep(90)

    while True:
        try:
            await run_top_range_snapshot()
        except Exception as e:
            logger.error("[TOP_RANGE] Snapshot loop error: %s", e)

        await asyncio.sleep(_SCAN_INTERVAL_SEC)
