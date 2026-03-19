"""
tct_snapshot.py — TCT Decision Audit Layer
============================================
In-memory snapshot store that captures gate-level decisions from all
trading pipelines (Phemex 6-Gate, 5B Schematics, 5A TensorTrade).

Each snapshot records the EXACT values used during a scan cycle —
never recomputed or approximated. Missing components are stored as None.

Thread-safe for concurrent FastAPI access via a simple threading.Lock.
No external dependencies.
"""

from __future__ import annotations

import threading
from collections import deque
from datetime import datetime, timezone
from typing import Optional


MAX_HISTORY = 100


class TCTSnapshotStore:
    """
    Stores the latest TCT decision snapshot and a rolling history.

    Thread-safe via a simple lock. All reads and writes acquire _lock.
    """

    def __init__(self, max_history: int = MAX_HISTORY) -> None:
        self._lock = threading.Lock()
        self._latest: Optional[dict] = None
        self._history: deque[dict] = deque(maxlen=max_history)

    def update(self, snapshot: dict) -> None:
        """
        Store a new snapshot. Automatically timestamps it if not already set.

        Args:
            snapshot: Dict containing gate-level decision data.
                      Must reflect EXACT values from execution — never
                      recomputed or duplicated logic.
        """
        if "timestamp" not in snapshot:
            snapshot["timestamp"] = datetime.now(timezone.utc).isoformat()

        with self._lock:
            self._latest = snapshot
            self._history.append(snapshot)

    def get_latest(self) -> Optional[dict]:
        """Return the most recent snapshot, or None if no scans have run."""
        with self._lock:
            return self._latest

    def get_history(self, limit: int = MAX_HISTORY) -> list[dict]:
        """
        Return recent snapshots, newest first.

        Args:
            limit: Max number of snapshots to return.
        """
        with self._lock:
            items = list(self._history)
        # Newest first
        items.reverse()
        return items[:limit]


# ---------------------------------------------------------------------------
# Global singleton — imported by traders and server_mexc
# ---------------------------------------------------------------------------
tct_store = TCTSnapshotStore()
