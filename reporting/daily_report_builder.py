"""
reporting/daily_report_builder.py — HPB Daily Report Builder
=============================================================
Generates one structured daily report from live engine telemetry.

Outputs:
  logs/daily/YYYY-MM-DD.json
  logs/daily/YYYY-MM-DD.md

Usage:
  python -m reporting.daily_report_builder            # generate today's report
  python -m reporting.daily_report_builder --date 2026-04-07   # backfill a date

Safety guarantees:
  - NEVER imports the trader at module level (avoids starting the engine)
  - ALL telemetry pulls are wrapped in try/except with safe defaults
  - Never writes to stdout in a way that could corrupt scan loop output
  - Non-blocking: runs in < 1 second, suitable for cron or background thread

Data sources:
  engine_version.get_version_info()         → version / build / uptime
  schematics_5b_trader.get_5b_trader().*   → live health, funnels, trade state
  scce_engine.get_scce().get_snapshot()    → SCCE structural state
  reporting.task_registry.TASK_REGISTRY    → active / completed / queued tasks
"""

import logging
import os
import sys
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
_DEFAULT_NOTES: List[str] = []  # auto-populated from event history


# ═════════════════════════════════════════════════════════════════════════════
# Telemetry collectors — each isolated, never raises
# ═════════════════════════════════════════════════════════════════════════════

def _collect_version() -> dict:
    """Pull version / build / uptime from engine_version module."""
    try:
        from engine_version import get_version_info
        info = get_version_info()
        return {
            "engine_version": info.get("engine_version", "n/a"),
            "git_commit": info.get("git_commit", "n/a"),
            "build_timestamp": info.get("build_timestamp", "n/a"),
            "environment": info.get("environment", "n/a"),
            "uptime_seconds": info.get("uptime_seconds"),
            "uptime_human": info.get("uptime_human", "n/a"),
            "process_start_time": info.get("process_start_time"),
        }
    except Exception as e:
        logger.debug("[DailyReport] version collect failed: %s", e)
        return {
            "engine_version": "n/a", "git_commit": "n/a",
            "build_timestamp": "n/a", "environment": "n/a",
            "uptime_seconds": None, "uptime_human": "n/a",
            "process_start_time": None,
        }


def _collect_live_health() -> dict:
    """Pull live execution health from running trader (if available)."""
    empty = {
        "signals_seen": 0,
        "after_rig": 0,
        "after_v2_gate": 0,
        "order_attempts": 0,
        "orders_submitted": 0,
        "orders_rejected": 0,
        "last_signal_time": None,
        "last_order_attempt_time": None,
        "top_block_reasons": {},
        "l3_near_miss": {},
        "scce_l3_cross": {},
        "neutral_htf": {},
        "shadow_candidates": [],
        "symbol_funnels": {},
        "scan_mode": "unknown",
        "trading_symbols": [],
        "scce": {},
        "current_trade_open": False,
    }
    try:
        from schematics_5b_trader import get_5b_trader
        trader = get_5b_trader()
        h = trader.get_live_health()

        # Merge scan_while_trade_open diagnostic
        swto = h.get("scan_while_trade_open") or {}
        h["current_trade_open"] = swto.get("current_trade_open", False)

        # Merge ETH context debug for scan mode / symbols
        eth_ctx = h.get("eth_context_debug") or {}
        h["scan_mode"] = eth_ctx.get("scan_mode", "unknown")
        h["trading_symbols"] = eth_ctx.get("trading_symbols", [])

        return {**empty, **h}
    except Exception as e:
        logger.debug("[DailyReport] live_health collect failed: %s", e)
        return empty


def _collect_open_trade(health: dict) -> Optional[dict]:
    """Build open_trade_summary from trader state (if trade is open)."""
    try:
        from schematics_5b_trader import get_5b_trader
        trader = get_5b_trader()
        trade = trader.state.current_trade
        if not trade:
            return None

        entry = trade.get("entry_price") or trade.get("entry")
        symbol = trade.get("symbol", "unknown")
        current_price = None
        try:
            from schematics_5b_trader import fetch_live_price
            current_price = fetch_live_price(symbol)
        except Exception:
            pass

        pnl_pct = None
        if entry and current_price:
            direction = trade.get("direction", "bullish")
            if direction == "bullish":
                pnl_pct = round((current_price - entry) / entry * 100, 3)
            else:
                pnl_pct = round((entry - current_price) / entry * 100, 3)

        return {
            "symbol": symbol,
            "model": trade.get("model", "n/a"),
            "timeframe": trade.get("timeframe", "n/a"),
            "direction": trade.get("direction", "n/a"),
            "entry_price": entry,
            "current_price": current_price,
            "pnl_pct": pnl_pct,
            "tp1_hit": trade.get("tp1_hit", False),
            "tp2_hit": trade.get("tp2_hit", False),
            "stop_hit": trade.get("stop_hit", False),
            "opened_at": trade.get("opened_at") or trade.get("entry_time"),
        }
    except Exception as e:
        logger.debug("[DailyReport] open_trade collect failed: %s", e)
        return None


def _collect_scce(health: dict) -> dict:
    """Pull SCCE snapshot (prefers already-fetched health, falls back to direct import)."""
    try:
        # Prefer SCCE snapshot already embedded in health (avoids double import)
        scce_h = health.get("scce") or {}
        if scce_h.get("enabled") is not False and scce_h:
            result = {
                "enabled": scce_h.get("enabled", False),
                "shadow_mode": scce_h.get("shadow_mode", True),
                "total_candidates": scce_h.get("total_candidates", 0),
                "active_candidates": scce_h.get("active_candidates", 0),
                "top_candidates": scce_h.get("top_candidates", [])[:5],
                "event_history": scce_h.get("event_history", [])[-5:],
            }
        else:
            from scce_engine import get_scce, SCCE_ENABLED
            if not SCCE_ENABLED:
                return {"enabled": False, "shadow_mode": True,
                        "total_candidates": 0, "active_candidates": 0,
                        "top_candidates": [], "event_history": []}
            snap = get_scce().get_snapshot()
            result = {
                "enabled": snap.get("enabled", False),
                "shadow_mode": snap.get("shadow_mode", True),
                "total_candidates": snap.get("total_candidates", 0),
                "active_candidates": snap.get("active_candidates", 0),
                "top_candidates": snap.get("top_candidates", [])[:5],
                "event_history": snap.get("event_history", [])[-5:],
            }

        # Attach L3 cross-table from health
        result["scce_l3_cross"] = health.get("scce_l3_cross") or {}
        return result

    except Exception as e:
        logger.debug("[DailyReport] scce collect failed: %s", e)
        return {"enabled": False, "shadow_mode": True,
                "total_candidates": 0, "active_candidates": 0,
                "top_candidates": [], "event_history": [],
                "scce_l3_cross": {}}


def _collect_bottlenecks(health: dict) -> List[dict]:
    """Extract top block reasons as sorted bottleneck list."""
    try:
        raw = health.get("top_block_reasons") or {}
        if not raw:
            return []
        return [
            {"reason": k, "count": v}
            for k, v in sorted(raw.items(), key=lambda x: -x[1])
        ][:10]
    except Exception:
        return []


def _collect_important_events(health: dict, scce: dict) -> List[str]:
    """
    Surface notable events as human-readable strings.
    Sources: SCCE event history, L3 near-miss counts, neutral HTF activity,
             shadow candidates.
    """
    events = []
    try:
        # SCCE events
        for ev in (scce.get("event_history") or [])[-5:]:
            ts = ev.get("timestamp", "?")[:16]
            events.append(f"[SCCE {ts}] {ev.get('symbol')} {ev.get('tf')}: "
                          f"{ev.get('event')} {ev.get('detail', '')}")

        # L3 near-miss summary
        nm = health.get("l3_near_miss") or {}
        total_nm = sum(v for k, v in nm.items() if isinstance(v, int))
        if total_nm:
            within_tight = nm.get("within_0_10_pct", 0) + nm.get("within_0_15_pct", 0)
            events.append(
                f"[L3] {total_nm} L3 failures observed; "
                f"{within_tight} within 0.15% of breakout (near-miss candidates)"
            )

        # SCCE × L3 cross-table
        cross = health.get("scce_l3_cross") or {}
        mature_misses = cross.get("tap3", 0) + cross.get("bos_pending", 0) + cross.get("qualified", 0)
        if mature_misses:
            events.append(
                f"[SCCE×L3] {mature_misses} L3 failures on mature SCCE candidates "
                f"(tap3={cross.get('tap3', 0)}, bos_pending={cross.get('bos_pending', 0)}, "
                f"qualified={cross.get('qualified', 0)}) — tolerance adjustment candidate"
            )

        # Neutral HTF
        n_htf = health.get("neutral_htf") or {}
        if n_htf.get("allowed_with_penalty", 0) > 0:
            events.append(
                f"[HTF] Neutral HTF passthrough: {n_htf['allowed_with_penalty']} "
                f"allowed with penalty (was hard-blocked before fix)"
            )

        # Shadow candidates
        shadow = health.get("shadow_candidates") or []
        if shadow:
            latest = shadow[-1]
            events.append(
                f"[SHADOW] Latest shadow candidate: {latest.get('symbol')} "
                f"{latest.get('model')} score={latest.get('score')} "
                f"— blocked only by open trade"
            )

    except Exception as e:
        logger.debug("[DailyReport] events collect failed: %s", e)

    return events


# ═════════════════════════════════════════════════════════════════════════════
# Main builder
# ═════════════════════════════════════════════════════════════════════════════

def build_daily_report(date: str = None, notes: List[str] = None) -> dict:
    """
    Build and return a complete daily report dict.

    Args:
        date:  ISO date string 'YYYY-MM-DD'. Defaults to today (UTC).
        notes: Optional list of manual notes to append.

    Returns:
        Fully populated report dict. All missing fields default to null/0/[].
        Never raises — failures produce partial reports with logged warnings.
    """
    if date is None:
        date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    generated_at = datetime.now(timezone.utc).isoformat()

    # ── Collect from all sources ──────────────────────────────────────
    version = _collect_version()
    health = _collect_live_health()
    open_trade = _collect_open_trade(health)
    scce = _collect_scce(health)
    bottlenecks = _collect_bottlenecks(health)
    important_events = _collect_important_events(health, scce)

    # ── Task registry ──────────────────────────────────────────────────
    try:
        from reporting.task_registry import TASK_REGISTRY
        active_tasks = TASK_REGISTRY.get("active", [])
        completed_tasks = TASK_REGISTRY.get("completed", [])
        queued_tasks = TASK_REGISTRY.get("queued", [])
    except Exception as e:
        logger.debug("[DailyReport] task_registry load failed: %s", e)
        active_tasks, completed_tasks, queued_tasks = [], [], []

    # ── Symbol funnels ─────────────────────────────────────────────────
    funnels = health.get("symbol_funnels") or {}

    # ── Execution summary ──────────────────────────────────────────────
    total_detected = sum(f.get("schematics_detected", 0) for f in funnels.values())
    total_confirmed = sum(f.get("confirmed", 0) for f in funnels.values())
    total_after_l3 = sum(f.get("after_l3", 0) for f in funnels.values())
    total_qualified = sum(f.get("qualified", 0) for f in funnels.values())

    execution_summary = {
        "schematics_detected": total_detected,
        "confirmed_schematics": total_confirmed,
        "qualified_setups": total_qualified,
        "order_attempts": health.get("order_attempts", 0),
        "orders_submitted": health.get("orders_submitted", 0),
        "orders_rejected": health.get("orders_rejected", 0),
        "signals_seen": health.get("signals_seen", 0),
        "after_rig": health.get("after_rig", 0),
        "after_v2_gate": health.get("after_v2_gate", 0),
    }

    # ── Live status ────────────────────────────────────────────────────
    live_status = {
        "scan_mode": health.get("scan_mode", "unknown"),
        "active_symbols": health.get("trading_symbols", []),
        "open_trade": open_trade is not None,
        "open_trade_symbol": open_trade["symbol"] if open_trade else None,
        "open_trade_pnl_pct": open_trade["pnl_pct"] if open_trade else None,
        "scanner_healthy": health.get("scanner_healthy", None),
        "runtime_errors": health.get("runtime_errors", None),
        "uptime_seconds": version.get("uptime_seconds"),
    }

    # ── Assemble report ────────────────────────────────────────────────
    report = {
        "date": date,
        "generated_at": generated_at,

        # Version
        "engine_version": version["engine_version"],
        "git_commit": version["git_commit"],
        "build_timestamp": version["build_timestamp"],
        "environment": version["environment"],
        "uptime_human": version["uptime_human"],
        "uptime_seconds": version["uptime_seconds"],
        "process_start_time": version["process_start_time"],

        # Tasks
        "active_tasks": active_tasks,
        "completed_tasks": completed_tasks,
        "queued_tasks": queued_tasks,

        # Trade
        "open_trade_summary": open_trade,

        # Status
        "live_status": live_status,

        # Execution
        "execution_summary": execution_summary,
        "symbol_funnels": funnels,
        "bottlenecks": bottlenecks,

        # SCCE
        "scce_summary": scce,

        # Events & notes
        "important_events": important_events,
        "notes": notes or _DEFAULT_NOTES,
    }

    return report


# ═════════════════════════════════════════════════════════════════════════════
# CLI entry point — python -m reporting.daily_report_builder
# ═════════════════════════════════════════════════════════════════════════════

def _main():
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="HPB Daily Report Builder")
    parser.add_argument("--date", default=None,
                        help="Date to generate report for (YYYY-MM-DD). Defaults to today UTC.")
    parser.add_argument("--note", action="append", default=[],
                        help="Manual note to include (can repeat). e.g. --note 'SOL trade closed'")
    args = parser.parse_args()

    t0 = time.monotonic()
    report = build_daily_report(date=args.date, notes=args.note or None)

    from reporting.report_exporters import export_report
    outputs = export_report(report)

    elapsed = round(time.monotonic() - t0, 2)
    print(f"\nDone. Daily report generated in {elapsed}s")
    for path in outputs:
        print(f"  -> {path}")
    print()


if __name__ == "__main__":
    _main()
