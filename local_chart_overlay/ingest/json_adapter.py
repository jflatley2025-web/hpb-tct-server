"""JSON adapter — reads schematics_5b_trade_log.json format."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from local_chart_overlay.ingest.base import BaseAdapter
from local_chart_overlay.models.trade import TradeRecord
from local_chart_overlay.models.schematic import FrozenSchematic


def _parse_iso(s: Optional[str]) -> Optional[datetime]:
    """Parse ISO 8601 timestamp, ensure UTC."""
    if not s:
        return None
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


class JsonAdapter(BaseAdapter):
    """Reads the schematics_5b_trade_log.json format.

    Expected structure:
    {
        "trade_history": [
            {
                "id": 1,
                "symbol": "BTCUSDT",
                "direction": "bearish",
                ...
            }
        ]
    }

    No schematic data is available in this format.
    """

    def __init__(self, file_path: str | Path):
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"Trade log not found: {self.file_path}")

    @property
    def source_type(self) -> str:
        return "json"

    def extract(self) -> list[tuple[TradeRecord, Optional[FrozenSchematic]]]:
        with open(self.file_path, "r") as f:
            data = json.load(f)

        trades_raw = data.get("trade_history", [])
        results = []

        for t in trades_raw:
            # Skip open/non-closed trades
            if t.get("status") != "closed":
                continue

            trade = TradeRecord(
                source_id=f"json:{t['id']}",
                source_type="json",
                symbol=t["symbol"],
                timeframe=t.get("timeframe"),
                direction=t["direction"],
                model=t.get("model"),
                entry_price=float(t["entry_price"]),
                stop_price=float(t["stop_price"]),
                target_price=float(t["target_price"]),
                tp1_price=float(t["tp1_price"]) if t.get("tp1_price") else None,
                opened_at=_parse_iso(t["opened_at"]),
                closed_at=_parse_iso(t.get("closed_at")),
                exit_price=float(t["exit_price"]) if t.get("exit_price") else None,
                pnl_pct=t.get("pnl_pct"),
                pnl_dollars=t.get("pnl_dollars"),
                is_win=t.get("is_win"),
                exit_reason=t.get("exit_reason"),
                entry_score=t.get("entry_score"),
                rr=t.get("rr"),
                leverage=t.get("leverage"),
                position_size=t.get("position_size"),
                risk_amount=t.get("risk_amount"),
                entry_reasons=t.get("entry_reasons"),
                htf_bias=t.get("htf_bias"),
            )
            # No schematic data in this format
            results.append((trade, None))

        return results
