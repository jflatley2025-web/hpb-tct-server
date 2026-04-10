"""
orchestrator.py — Single public interface for CCS v2 intelligence.

Reads JSONL, builds indices, computes all 4 metrics.
Pure read-only. No side effects. No engine imports.
"""

from __future__ import annotations

from datetime import datetime, timezone

from ccs_intelligence import CCS_INTELLIGENCE_ENABLED
from ccs_intelligence.reader import resolve_file, read_events
from ccs_intelligence.parser import filter_and_normalize, build_indices
from ccs_intelligence.metrics import (
    compute_bos_stats,
    compute_candidate_funnel,
    compute_range_tap_density,
    compute_tap3_bos_latency,
    compute_po3_confluence,
)


def compute_ccs_metrics(
    symbol: str | None = None,
    date_str: str | None = None,
    max_lines: int = 10_000,
) -> dict:
    """Compute all CCS v2 metrics for a given symbol and date.

    Args:
        symbol: Filter to specific symbol (e.g. "BTCUSDT"). None = all symbols.
        date_str: Date string "YYYY-MM-DD". None = today UTC.
        max_lines: Max JSONL lines to read.

    Returns:
        Full intelligence response dict. Empty results on any failure.
    """
    if not CCS_INTELLIGENCE_ENABLED:
        return {"status": "disabled"}

    try:
        return _compute(symbol, date_str, max_lines)
    except Exception:
        return _empty_response(date_str)


def get_health_summary(date_str: str | None = None) -> dict | None:
    """Compact 3-field summary for live_health extension.

    Returns None if disabled or on any failure.
    """
    if not CCS_INTELLIGENCE_ENABLED:
        return None

    try:
        filepath = resolve_file(date_str)
        if filepath is None:
            return {"events_today": 0, "bos_success_rate_all": None, "symbols_tracked": 0}

        events, _, _ = read_events(filepath, max_lines=10_000)
        valid, _ = filter_and_normalize(events)

        # Quick BOS stats across all symbols
        indices = build_indices(valid)
        bos = compute_bos_stats(indices)

        symbols = set(e.get("symbol") for e in valid if e.get("symbol"))

        return {
            "events_today": len(valid),
            "bos_success_rate_all": bos.get("success_rate"),
            "symbols_tracked": len(symbols),
        }
    except Exception:
        return None


def _compute(
    symbol: str | None, date_str: str | None, max_lines: int
) -> dict:
    """Internal: full metric computation."""
    effective_date = date_str or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    filepath = resolve_file(effective_date)

    if filepath is None:
        return _empty_response(effective_date)

    events, malformed, file_size = read_events(filepath, max_lines)

    if not events:
        resp = _empty_response(effective_date)
        resp["malformed_lines"] = malformed
        resp["file_size_bytes"] = file_size
        return resp

    # Determine symbols to process
    all_symbols = sorted(set(e.get("symbol") for e in events if e.get("symbol")))
    target_symbols = [symbol] if symbol else all_symbols

    symbols_out: dict[str, dict] = {}
    total_orphans = 0

    for sym in target_symbols:
        valid, orphans = filter_and_normalize(events, sym)
        total_orphans += orphans
        if not valid:
            continue

        indices = build_indices(valid)
        symbols_out[sym] = {
            "bos": compute_bos_stats(indices),
            "funnel": compute_candidate_funnel(indices),
            "range": compute_range_tap_density(indices),
            "latency": compute_tap3_bos_latency(indices),
            "po3": compute_po3_confluence(indices),
        }

    return {
        "date": effective_date,
        "total_events": len(events),
        "malformed_lines": malformed,
        "orphan_events": total_orphans,
        "file_size_bytes": file_size,
        "symbols": symbols_out,
    }


def _empty_response(date_str: str | None = None) -> dict:
    effective = date_str or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return {
        "date": effective,
        "total_events": 0,
        "malformed_lines": 0,
        "orphan_events": 0,
        "file_size_bytes": 0,
        "symbols": {},
    }
