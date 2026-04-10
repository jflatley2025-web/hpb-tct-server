"""Manifest builder — machine-readable metadata for replay share packages."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Optional


def build_manifest(
    trade_ids: list[int],
    symbol: str,
    timeframe: str,
    side: str,
    files: list[str],
    has_confirmed_schematic: bool = False,
    has_suggested_schematic: bool = False,
    has_accuracy_report: bool = False,
    model: Optional[str] = None,
    tags: Optional[list[str]] = None,
    notes: Optional[str] = None,
) -> str:
    """Build a manifest.json string for a replay share package.

    Args:
        trade_ids: List of trade IDs in this package.
        symbol: Trading symbol (e.g. "BTCUSDT").
        timeframe: Chart timeframe (e.g. "1h").
        side: "long" or "short".
        files: List of filenames in the package.
        has_confirmed_schematic: Whether confirmed schematic is included.
        has_suggested_schematic: Whether suggested schematic overlay is included.
        has_accuracy_report: Whether accuracy comparison is included.
        model: Model label (e.g. "Model_1").

    Returns:
        JSON string.
    """
    manifest = {
        "package_type": "replay_share",
        "version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "trade_ids": trade_ids,
        "trade_count": len(trade_ids),
        "symbol": symbol,
        "timeframe": timeframe,
        "side": side,
        "model": model,
        "files": files,
        "has_confirmed_schematic": has_confirmed_schematic,
        "has_suggested_schematic": has_suggested_schematic,
        "has_accuracy_report": has_accuracy_report,
        "tags": tags or [],
        "notes": notes,
    }
    return json.dumps(manifest, indent=2)
