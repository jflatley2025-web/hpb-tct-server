"""Base adapter interface for trade data ingestion."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from local_chart_overlay.models.trade import TradeRecord
from local_chart_overlay.models.schematic import FrozenSchematic


class BaseAdapter(ABC):
    """Abstract base for all ingestion adapters.

    Each adapter reads from a specific source format and produces
    normalized (TradeRecord, Optional[FrozenSchematic]) pairs.
    """

    @abstractmethod
    def extract(self) -> list[tuple[TradeRecord, Optional[FrozenSchematic]]]:
        """Extract trades from the source.

        Returns:
            List of (trade, optional_schematic) tuples.
            Schematic is None when the source doesn't contain schematic data.
        """
        ...

    @property
    @abstractmethod
    def source_type(self) -> str:
        """Identifier for this source type (e.g., 'json', 'csv')."""
        ...
