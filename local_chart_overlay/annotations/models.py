"""Annotation data models."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TradeAnnotations:
    """Tags and notes attached to a trade."""
    trade_id: int
    tags: list[str] = field(default_factory=list)
    notes: Optional[str] = None

    @property
    def has_tags(self) -> bool:
        return len(self.tags) > 0

    @property
    def has_notes(self) -> bool:
        return self.notes is not None and self.notes.strip() != ""

    @property
    def tags_csv(self) -> str:
        """Comma-separated tag string for display."""
        return ", ".join(self.tags)

    def to_dict(self) -> dict:
        return {
            "trade_id": self.trade_id,
            "tags": self.tags,
            "notes": self.notes,
        }

    @staticmethod
    def empty(trade_id: int) -> TradeAnnotations:
        return TradeAnnotations(trade_id=trade_id)
