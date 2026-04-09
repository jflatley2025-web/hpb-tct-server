"""
ccs_writer.py — Context Continuity Store v1 (Write-Only)
=========================================================
Append-only JSONL event writer for structural context persistence.

ZERO execution impact. All writes are fail-safe (try/except/pass).
No reads. No feedback into pipeline. No external dependencies.
"""

import json
import os
import time
from datetime import datetime, timezone

_LOGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")

CCS_ENABLED = os.getenv("CCS_ENABLED", "true").lower() == "true"


def _today_file() -> str:
    """Resolve today's JSONL filename."""
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return os.path.join(_LOGS_DIR, f"context_events_{date_str}.jsonl")


_event_counter = 0

def _make_event_id() -> str:
    global _event_counter
    _event_counter += 1
    ts = int(time.time())
    return f"ccs_{ts}_{_event_counter:04d}"


def emit_event(
    symbol: str,
    cycle_id: str,
    stage: str,
    event_type: str,
    payload: dict,
    refs: dict | None = None,
) -> None:
    """Append a single CCS event to today's JSONL file.

    Fails silently. Never blocks. Never raises.
    """
    if not CCS_ENABLED:
        return
    try:
        envelope = {
            "event_id": _make_event_id(),
            "ts": datetime.now(timezone.utc).isoformat(),
            "symbol": symbol,
            "cycle_id": cycle_id,
            "stage": stage,
            "event_type": event_type,
            "payload": payload,
            "refs": {
                "candidate_id": None,
                "range_id": None,
                "structure_id": None,
                "bos_attempt_id": None,
                **(refs or {}),
            },
        }
        line = json.dumps(envelope, separators=(",", ":"), default=str)
        with open(_today_file(), "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass
