"""
reporting/report_exporters.py — Report Export Boundary
========================================================
Exporter interface for daily reports.

Currently implemented:
  - LocalJSONExporter  → logs/daily/YYYY-MM-DD.json
  - LocalMarkdownExporter → logs/daily/YYYY-MM-DD.md

Future exporters (not yet implemented):
  - GoogleDocsExporter.export(report)
  - NotionExporter.export(report)
  - SlackExporter.export(report)

All exporters must conform to the same interface:
  exporter.export(report: dict) -> str   (returns output path or URL)

Adding a new exporter never requires modifying daily_report_builder.py.
Just register it in ACTIVE_EXPORTERS.
"""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Output directory ──────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent
_DAILY_LOG_DIR = _REPO_ROOT / "logs" / "daily"


def _ensure_daily_dir() -> Path:
    _DAILY_LOG_DIR.mkdir(parents=True, exist_ok=True)
    return _DAILY_LOG_DIR


# ═════════════════════════════════════════════════════════════════════════════
# Local JSON Exporter
# ═════════════════════════════════════════════════════════════════════════════

class LocalJSONExporter:
    """Write daily report as JSON to logs/daily/YYYY-MM-DD.json."""

    def export(self, report: dict) -> str:
        out_dir = _ensure_daily_dir()
        date_str = report.get("date", datetime.now(timezone.utc).strftime("%Y-%m-%d"))
        path = out_dir / f"{date_str}.json"

        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, default=str)
            logger.info("[DailyReport] JSON written → %s", path)
            return str(path)
        except Exception as e:
            logger.error("[DailyReport] JSON export failed: %s", e)
            raise


# ═════════════════════════════════════════════════════════════════════════════
# Local Markdown Exporter
# ═════════════════════════════════════════════════════════════════════════════

class LocalMarkdownExporter:
    """Write daily report as Markdown to logs/daily/YYYY-MM-DD.md."""

    def export(self, report: dict) -> str:
        out_dir = _ensure_daily_dir()
        date_str = report.get("date", datetime.now(timezone.utc).strftime("%Y-%m-%d"))
        path = out_dir / f"{date_str}.md"

        try:
            md = _render_markdown(report)
            with open(path, "w", encoding="utf-8") as f:
                f.write(md)
            logger.info("[DailyReport] Markdown written → %s", path)
            return str(path)
        except Exception as e:
            logger.error("[DailyReport] Markdown export failed: %s", e)
            raise


# ═════════════════════════════════════════════════════════════════════════════
# Future Exporter Stub — Google Docs
# ═════════════════════════════════════════════════════════════════════════════

class GoogleDocsExporter:
    """
    Stub for future Google Docs integration.

    When implemented, call:
        exporter = GoogleDocsExporter(credentials_path="...", doc_id="...")
        exporter.export(report)
    """

    def __init__(self, credentials_path: str = None, doc_id: str = None):
        self._credentials_path = credentials_path
        self._doc_id = doc_id

    def export(self, report: dict) -> str:
        raise NotImplementedError(
            "GoogleDocsExporter not yet implemented. "
            "Add google-auth and google-api-python-client to requirements.txt, "
            "then implement _push_to_gdocs(report) here."
        )


# ═════════════════════════════════════════════════════════════════════════════
# Active exporter registry — add exporters here to activate them
# ═════════════════════════════════════════════════════════════════════════════

ACTIVE_EXPORTERS = [
    LocalJSONExporter(),
    LocalMarkdownExporter(),
    # GoogleDocsExporter(),  ← add when ready
]


def export_report(report: dict) -> List[str]:
    """
    Run all active exporters for a report dict.

    Returns list of output paths / URLs from each exporter.
    Non-blocking: failures are logged and skipped, never raised.
    """
    outputs = []
    for exporter in ACTIVE_EXPORTERS:
        try:
            result = exporter.export(report)
            outputs.append(result)
        except Exception as e:
            logger.error("[DailyReport] Exporter %s failed: %s", type(exporter).__name__, e)
    return outputs


# ═════════════════════════════════════════════════════════════════════════════
# Markdown renderer
# ═════════════════════════════════════════════════════════════════════════════

def _render_markdown(r: dict) -> str:
    date = r.get("date", "unknown")
    lines = [f"# HPB Daily Report — {date}", ""]

    # Engine Version
    lines += ["## Engine Version", ""]
    lines.append(f"- **Version:** {r.get('engine_version', 'n/a')}")
    lines.append(f"- **Git commit:** `{r.get('git_commit', 'n/a')}`")
    lines.append(f"- **Build timestamp:** {r.get('build_timestamp', 'n/a')}")
    lines.append(f"- **Environment:** {r.get('environment', 'n/a')}")
    lines.append(f"- **Uptime:** {r.get('uptime_human', 'n/a')}")
    lines.append("")

    # Active Tasks
    lines += ["## Active Tasks", ""]
    for t in r.get("active_tasks") or []:
        lines.append(f"- {t}")
    lines.append("")

    # Completed Tasks
    lines += ["## Completed Tasks (cumulative)", ""]
    for t in r.get("completed_tasks") or []:
        lines.append(f"- {t}")
    lines.append("")

    # Queued Tasks
    if r.get("queued_tasks"):
        lines += ["## Queued Tasks", ""]
        for t in r["queued_tasks"]:
            lines.append(f"- {t}")
        lines.append("")

    # Open Trade Summary
    ots = r.get("open_trade_summary")
    lines += ["## Live Trade", ""]
    if ots:
        lines.append(f"- **Symbol:** {ots.get('symbol', 'n/a')}")
        lines.append(f"- **Model:** {ots.get('model', 'n/a')}")
        lines.append(f"- **Timeframe:** {ots.get('timeframe', 'n/a')}")
        lines.append(f"- **Entry:** {ots.get('entry_price', 'n/a')}")
        lines.append(f"- **Current price:** {ots.get('current_price', 'n/a')}")
        pnl = ots.get("pnl_pct")
        lines.append(f"- **PnL:** {f'{pnl:+.2f}%' if pnl is not None else 'n/a'}")
        lines.append(f"- **TP1 hit:** {ots.get('tp1_hit', 'n/a')}")
        lines.append(f"- **TP2 hit:** {ots.get('tp2_hit', 'n/a')}")
        lines.append(f"- **SL hit:** {ots.get('stop_hit', 'n/a')}")
    else:
        lines.append("- No open trade")
    lines.append("")

    # Live Status
    ls = r.get("live_status") or {}
    lines += ["## Live Status", ""]
    lines.append(f"- **Scan mode:** {ls.get('scan_mode', 'n/a')}")
    lines.append(f"- **Active symbols:** {', '.join(ls.get('active_symbols') or []) or 'n/a'}")
    lines.append(f"- **Scanner healthy:** {ls.get('scanner_healthy', 'n/a')}")
    lines.append(f"- **Runtime errors:** {ls.get('runtime_errors', 'n/a')}")
    lines.append(f"- **Uptime seconds:** {ls.get('uptime_seconds', 'n/a')}")
    lines.append("")

    # Execution Summary
    es = r.get("execution_summary") or {}
    lines += ["## Execution Summary", ""]
    lines.append(f"- Schematics detected: **{es.get('schematics_detected', 0)}**")
    lines.append(f"- Confirmed schematics: **{es.get('confirmed_schematics', 0)}**")
    lines.append(f"- Qualified setups: **{es.get('qualified_setups', 0)}**")
    lines.append(f"- Order attempts: **{es.get('order_attempts', 0)}**")
    lines.append(f"- Orders submitted: **{es.get('orders_submitted', 0)}**")
    lines.append(f"- Orders rejected: **{es.get('orders_rejected', 0)}**")
    lines.append("")

    # Symbol Funnels
    funnels = r.get("symbol_funnels") or {}
    if funnels:
        lines += ["## Symbol Funnels", ""]
        for sym, f in funnels.items():
            lines.append(f"### {sym}")
            lines.append(f"- Detected: {f.get('schematics_detected', 0)}")
            lines.append(f"- Confirmed: {f.get('confirmed', 0)}")
            lines.append(f"- After L3: {f.get('after_l3', 0)}")
            lines.append(f"- Qualified: {f.get('qualified', 0)}")
            lines.append(f"- Order attempts: {f.get('order_attempts', 0)}")
            lines.append(f"- Submitted: {f.get('orders_submitted', 0)}")
            top_blocks = f.get("top_block_reasons") or {}
            if top_blocks:
                top = sorted(top_blocks.items(), key=lambda x: -x[1])[:3]
                lines.append(f"- Top blocks: {', '.join(f'{k}({v})' for k, v in top)}")
            lines.append("")

    # Bottlenecks
    bottlenecks = r.get("bottlenecks") or []
    lines += ["## Bottlenecks", ""]
    if bottlenecks:
        for b in bottlenecks:
            lines.append(f"- **{b.get('reason', '?')}**: {b.get('count', 0)}")
    else:
        lines.append("- No bottleneck data available")
    lines.append("")

    # SCCE Summary
    scce = r.get("scce_summary") or {}
    lines += ["## SCCE Summary", ""]
    lines.append(f"- Enabled: {scce.get('enabled', False)}")
    lines.append(f"- Shadow mode: {scce.get('shadow_mode', True)}")
    lines.append(f"- Total candidates: {scce.get('total_candidates', 0)}")
    lines.append(f"- Active candidates: {scce.get('active_candidates', 0)}")
    scce_l3 = scce.get("scce_l3_cross") or {}
    if any(v for k, v in scce_l3.items() if k != "examples" and isinstance(v, int)):
        lines.append("- **L3 × SCCE phase cross-table:**")
        for phase in ("seed", "tap1", "tap2", "tap3", "bos_pending", "qualified", "no_match"):
            ct = scce_l3.get(phase, 0)
            if ct:
                lines.append(f"  - {phase}: {ct}")
    lines.append("")

    # Important Events
    lines += ["## Important Events", ""]
    for e in r.get("important_events") or []:
        lines.append(f"- {e}")
    if not r.get("important_events"):
        lines.append("- None recorded")
    lines.append("")

    # Notes
    if r.get("notes"):
        lines += ["## Notes", ""]
        for n in r["notes"]:
            lines.append(f"- {n}")
        lines.append("")

    lines.append(f"---")
    lines.append(f"*Generated: {r.get('generated_at', 'n/a')} | Engine: {r.get('engine_version', 'n/a')}*")
    lines.append("")

    return "\n".join(lines)
