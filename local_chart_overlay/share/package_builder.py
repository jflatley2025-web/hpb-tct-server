"""Package builder — creates self-contained shareable folders."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from local_chart_overlay.models.render_payload import RenderPayload
from local_chart_overlay.rendering.pine_generator import PineGenerator
from local_chart_overlay.share.html_template import render_html


class PackageBuilder:
    """Builds a share package folder containing Pine Script + HTML launcher.

    Output structure:
        share/<label>/
            overlay.pine
            open_chart.html
            README.txt
    """

    def __init__(self, output_dir: str | Path = "exports/share"):
        self.output_dir = Path(output_dir)
        self._pine_gen = PineGenerator()

    def build_single(
        self,
        payload: RenderPayload,
        label: Optional[str] = None,
    ) -> Path:
        """Build a share package for one trade."""
        label = label or payload.pine_label
        return self._build(
            payloads=[payload],
            label=label,
            symbol=payload.trade.symbol,
            timeframe=payload.trade.timeframe or "",
        )

    def build_batch(
        self,
        payloads: list[RenderPayload],
        label: str,
    ) -> Path:
        """Build a share package for multiple trades."""
        symbols = {p.trade.symbol for p in payloads}
        timeframes = {p.trade.timeframe for p in payloads if p.trade.timeframe}
        symbol = next(iter(symbols)) if len(symbols) == 1 else "MULTI"
        timeframe = next(iter(timeframes)) if len(timeframes) == 1 else "mixed"

        return self._build(
            payloads=payloads,
            label=label,
            symbol=symbol,
            timeframe=timeframe,
        )

    def _build(
        self,
        payloads: list[RenderPayload],
        label: str,
        symbol: str,
        timeframe: str,
    ) -> Path:
        """Core build logic — creates the folder and all files."""
        safe_label = label.replace("/", "_").replace(" ", "_").replace(":", "_")
        pkg_dir = self.output_dir / safe_label
        pkg_dir.mkdir(parents=True, exist_ok=True)

        generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        # 1. Generate Pine Script
        pine_path = pkg_dir / "overlay.pine"
        if len(payloads) == 1:
            out = self._pine_gen.generate_single(payloads[0], pkg_dir)
            # Rename to standard name
            if out.name != "overlay.pine":
                out.rename(pine_path)
            pine_content = pine_path.read_text(encoding="utf-8")
        else:
            out = self._pine_gen.generate_batch(payloads, pkg_dir, chunk_num=1, label=label)
            if out.name != "overlay.pine":
                out.rename(pine_path)
            pine_content = pine_path.read_text(encoding="utf-8")

        # Clean up any extra .pine files the generator may have created
        for f in pkg_dir.glob("tct_overlay_*.pine"):
            f.unlink()

        # 2. Generate HTML launcher
        html_content = render_html(
            pine_script=pine_content,
            title=f"TCT Overlay — {label}",
            symbol=symbol,
            timeframe=timeframe,
            trade_count=len(payloads),
            generated_at=generated_at,
        )
        (pkg_dir / "open_chart.html").write_text(html_content, encoding="utf-8")

        # 3. Generate README
        readme = _build_readme(
            symbol=symbol,
            timeframe=timeframe,
            trade_count=len(payloads),
            generated_at=generated_at,
            trade_ids=[p.trade_id for p in payloads],
        )
        (pkg_dir / "README.txt").write_text(readme, encoding="utf-8")

        return pkg_dir


def _build_readme(
    symbol: str,
    timeframe: str,
    trade_count: int,
    generated_at: str,
    trade_ids: list[int],
) -> str:
    ids_str = ", ".join(str(i) for i in trade_ids)
    return f"""TCT Overlay — Share Package
===========================

How to use:
  1. Open open_chart.html in your browser
  2. Click "Copy Pine Script"
  3. Open TradingView (or click the link in the page)
  4. Open the Pine Editor (bottom panel)
  5. Paste the script (Ctrl+V / Cmd+V)
  6. Click "Add to chart"

Details:
  Symbol:     {symbol}
  Timeframe:  {timeframe}
  Trades:     {trade_count} (IDs: {ids_str})
  Generated:  {generated_at}

Notes:
  - The overlay uses fixed timestamp anchors (xloc.bar_time)
  - Switching timeframes will NOT break the overlay
  - Use the "Trade #" input to select which trade to view
"""
