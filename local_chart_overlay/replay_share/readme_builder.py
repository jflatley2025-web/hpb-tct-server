"""README builder — human-readable instructions for replay share packages."""
from __future__ import annotations

from typing import Optional


def build_readme(
    trade_ids: list[int],
    symbol: str,
    timeframe: str,
    side: str,
    model: Optional[str] = None,
    has_confirmed: bool = False,
    has_suggested: bool = False,
    has_accuracy: bool = False,
    generated_at: str = "",
    tags: Optional[list[str]] = None,
    notes: Optional[str] = None,
) -> str:
    """Build a README.txt for a replay share package."""
    ids_str = ", ".join(str(i) for i in trade_ids)
    confirmed_str = "yes" if has_confirmed else "no"
    suggested_str = "yes" if has_suggested else "no"
    accuracy_str = "yes" if has_accuracy else "no"
    tags_str = ", ".join(tags) if tags else "(none)"
    notes_str = notes if notes else "(none)"

    return f"""Trade Replay Package
====================

Contents:
  replay.html        - Interactive trade replay (open in browser)
  replay_data.json   - Replay payload data
  overlay.pine       - TradingView Pine Script overlay
  open_chart.html    - One-click Pine Script copier for TradingView
  manifest.json      - Package metadata

How to review the trade:
  1. Open replay.html in your browser
  2. Use arrow keys or buttons to step through stages:
     Pre-Context -> Tap 1 -> Tap 2 -> Tap 3 -> BOS -> Entry -> Outcome
  3. Toggle confirmed/suggested anchors with checkboxes
  4. Review score breakdowns and accuracy in the side panel

How to load into TradingView:
  1. Open open_chart.html in your browser
  2. Click "Copy Pine Script"
  3. Open TradingView (or click the link in the page)
  4. Open the Pine Editor (bottom panel)
  5. Paste the script (Ctrl+V / Cmd+V)
  6. Click "Add to chart"

Trade details:
  Trade ID(s):         {ids_str}
  Symbol:              {symbol}
  Timeframe:           {timeframe}
  Side:                {side}
  Model:               {model or 'n/a'}
  Confirmed schematic: {confirmed_str}
  Suggested overlay:   {suggested_str}
  Accuracy report:     {accuracy_str}
  Tags:                {tags_str}
  Notes:               {notes_str}
  Generated:           {generated_at}

Technical notes:
  - The Pine overlay uses fixed timestamp anchors (xloc.bar_time)
  - Switching timeframes in TradingView will NOT break the overlay
  - No database or server is required to view these files
"""
