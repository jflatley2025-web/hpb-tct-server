"""Annotation service — CRUD for trade tags and notes.

Uses a dedicated `annotations` table in the existing SQLite store.
The table is created on first access (additive, no migration needed).
"""
from __future__ import annotations

import json
import sqlite3
from typing import Optional

from local_chart_overlay.annotations.models import TradeAnnotations
from local_chart_overlay.annotations.normalization import normalize_tag, normalize_tags


class AnnotationService:
    """Manages tags and notes for trades.

    Works with an existing SQLite connection (from SqliteStore).
    Creates its own table on first use.
    """

    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn
        self._ensure_table()

    def _ensure_table(self):
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS annotations (
                trade_id    INTEGER PRIMARY KEY REFERENCES trades(id),
                tags_json   TEXT DEFAULT '[]',
                notes       TEXT
            )
        """)
        self._conn.commit()

    def get(self, trade_id: int) -> TradeAnnotations:
        """Get annotations for a trade. Returns empty if none exist."""
        cur = self._conn.execute(
            "SELECT tags_json, notes FROM annotations WHERE trade_id = ?",
            (trade_id,),
        )
        row = cur.fetchone()
        if not row:
            return TradeAnnotations.empty(trade_id)

        tags = json.loads(row[0]) if row[0] else []
        notes = row[1]
        return TradeAnnotations(trade_id=trade_id, tags=tags, notes=notes)

    def add_tags(self, trade_id: int, new_tags: list[str]) -> TradeAnnotations:
        """Add tags to a trade (normalizes, deduplicates)."""
        current = self.get(trade_id)
        normalized = normalize_tags(new_tags)
        merged = list(current.tags)  # preserve order
        seen = set(merged)
        for tag in normalized:
            if tag not in seen:
                merged.append(tag)
                seen.add(tag)
        return self._save(trade_id, merged, current.notes)

    def remove_tags(self, trade_id: int, tags_to_remove: list[str]) -> TradeAnnotations:
        """Remove tags from a trade."""
        current = self.get(trade_id)
        # Normalize removal targets for matching
        remove_set = {normalize_tag(t) for t in tags_to_remove if normalize_tag(t)}
        remaining = [t for t in current.tags if t not in remove_set]
        return self._save(trade_id, remaining, current.notes)

    def set_note(self, trade_id: int, note: str) -> TradeAnnotations:
        """Set the note for a trade (replaces existing)."""
        current = self.get(trade_id)
        return self._save(trade_id, current.tags, note.strip() if note else None)

    def clear_note(self, trade_id: int) -> TradeAnnotations:
        """Clear the note for a trade."""
        current = self.get(trade_id)
        return self._save(trade_id, current.tags, None)

    def _save(
        self, trade_id: int, tags: list[str], notes: Optional[str],
    ) -> TradeAnnotations:
        """Upsert annotation row."""
        self._conn.execute(
            """INSERT INTO annotations (trade_id, tags_json, notes)
               VALUES (?, ?, ?)
               ON CONFLICT(trade_id) DO UPDATE SET
                 tags_json = excluded.tags_json,
                 notes = excluded.notes
            """,
            (trade_id, json.dumps(tags), notes),
        )
        self._conn.commit()
        return TradeAnnotations(trade_id=trade_id, tags=tags, notes=notes)
