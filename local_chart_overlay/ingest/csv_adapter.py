"""CSV adapter — reads generic CSV trade reports with configurable column mapping."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from local_chart_overlay.ingest.base import BaseAdapter
from local_chart_overlay.models.trade import TradeRecord
from local_chart_overlay.models.schematic import FrozenSchematic


# Default column mapping: internal_field -> csv_column_name
DEFAULT_MAPPING = {
    "symbol": "symbol",
    "direction": "direction",
    "timeframe": "timeframe",
    "model": "model",
    "entry_price": "entry_price",
    "stop_price": "stop_price",
    "target_price": "target_price",
    "tp1_price": "tp1_price",
    "opened_at": "opened_at",
    "closed_at": "closed_at",
    "exit_price": "exit_price",
    "pnl_pct": "pnl_pct",
    "pnl_dollars": "pnl_dollars",
    "is_win": "is_win",
    "exit_reason": "exit_reason",
    "entry_score": "entry_score",
    "rr": "rr",
    "leverage": "leverage",
    "htf_bias": "htf_bias",
}

REQUIRED_FIELDS = {"symbol", "entry_price", "stop_price", "target_price", "opened_at"}


def _parse_dt(s: Optional[str]) -> Optional[datetime]:
    if not s or not s.strip():
        return None
    s = s.strip()
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        # Try common formats
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%m/%d/%Y %H:%M:%S"):
            try:
                dt = datetime.strptime(s, fmt)
                break
            except ValueError:
                continue
        else:
            raise ValueError(f"Cannot parse datetime: {s}")
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _parse_float(s: Optional[str]) -> Optional[float]:
    if s is None or (isinstance(s, str) and not s.strip()):
        return None
    return float(s)


def _parse_int(s: Optional[str]) -> Optional[int]:
    if s is None or (isinstance(s, str) and not s.strip()):
        return None
    return int(float(s))


def _parse_bool(s: Optional[str]) -> Optional[bool]:
    if s is None or (isinstance(s, str) and not s.strip()):
        return None
    return s.strip().lower() in ("true", "1", "yes", "win")


class CsvAdapter(BaseAdapter):
    """Reads a CSV trade report with configurable column mapping.

    Args:
        file_path: Path to the CSV file.
        column_mapping: Dict mapping internal field names to CSV column names.
                       Defaults to DEFAULT_MAPPING.
    """

    def __init__(
        self,
        file_path: str | Path,
        column_mapping: Optional[dict[str, str]] = None,
    ):
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.file_path}")
        self.mapping = column_mapping or DEFAULT_MAPPING

    @property
    def source_type(self) -> str:
        return "csv"

    def _get_val(self, row: dict, field: str) -> Optional[str]:
        """Get a value from the CSV row using the column mapping."""
        col_name = self.mapping.get(field)
        if not col_name:
            return None
        return row.get(col_name)

    def extract(self) -> list[tuple[TradeRecord, Optional[FrozenSchematic]]]:
        results = []
        filename = self.file_path.name

        with open(self.file_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            available_cols = set(reader.fieldnames or [])

            # Validate required columns exist
            missing = []
            for field in REQUIRED_FIELDS:
                col = self.mapping.get(field, field)
                if col not in available_cols:
                    missing.append(f"{field} (mapped to '{col}')")
            if missing:
                raise ValueError(
                    f"CSV missing required columns: {', '.join(missing)}. "
                    f"Available: {sorted(available_cols)}"
                )

            for row_num, row in enumerate(reader, start=1):
                # Normalize direction
                direction_raw = self._get_val(row, "direction") or ""
                direction = direction_raw.strip().lower()
                if direction in ("long", "buy", "bullish"):
                    direction = "bullish"
                elif direction in ("short", "sell", "bearish"):
                    direction = "bearish"

                trade = TradeRecord(
                    source_id=f"csv:{filename}:{row_num}",
                    source_type="csv",
                    symbol=(self._get_val(row, "symbol") or "").strip().upper(),
                    timeframe=self._get_val(row, "timeframe"),
                    direction=direction,
                    model=self._get_val(row, "model"),
                    entry_price=_parse_float(self._get_val(row, "entry_price")),
                    stop_price=_parse_float(self._get_val(row, "stop_price")),
                    target_price=_parse_float(self._get_val(row, "target_price")),
                    tp1_price=_parse_float(self._get_val(row, "tp1_price")),
                    opened_at=_parse_dt(self._get_val(row, "opened_at")),
                    closed_at=_parse_dt(self._get_val(row, "closed_at")),
                    exit_price=_parse_float(self._get_val(row, "exit_price")),
                    pnl_pct=_parse_float(self._get_val(row, "pnl_pct")),
                    pnl_dollars=_parse_float(self._get_val(row, "pnl_dollars")),
                    is_win=_parse_bool(self._get_val(row, "is_win")),
                    exit_reason=self._get_val(row, "exit_reason"),
                    entry_score=_parse_int(self._get_val(row, "entry_score")),
                    rr=_parse_float(self._get_val(row, "rr")),
                    leverage=_parse_int(self._get_val(row, "leverage")),
                    htf_bias=self._get_val(row, "htf_bias"),
                )
                results.append((trade, None))

        return results
