"""Replay share package builder — bundles replay + Pine into one portable folder.

Reuses existing components:
  - ReplayBuilder (replay/replay_builder.py) for replay payload assembly
  - render_replay_html (replay/replay_template.py) for replay HTML
  - PineGenerator (rendering/pine_generator.py) for Pine Script
  - render_html (share/html_template.py) for Pine HTML launcher
  - manifest_builder, readme_builder for package metadata
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from local_chart_overlay.models.trade import TradeRecord
from local_chart_overlay.models.schematic import FrozenSchematic
from local_chart_overlay.models.render_payload import RenderPayload
from local_chart_overlay.rendering.pine_generator import PineGenerator
from local_chart_overlay.replay.replay_builder import (
    ReplayBuilder, build_suggested_anchors, build_comparisons_from_schematics,
)
from local_chart_overlay.replay.replay_models import AnchorPoint, AccuracySummary
from local_chart_overlay.replay.replay_template import render_replay_html
from local_chart_overlay.share.html_template import render_html as render_pine_html
from local_chart_overlay.replay_share.manifest_builder import build_manifest
from local_chart_overlay.replay_share.readme_builder import build_readme


class ReplayShareBuilder:
    """Builds a portable replay + Pine share package for one trade.

    Orchestrates existing builders — does not duplicate any logic.
    """

    def __init__(self, output_dir: str | Path = "exports/replay_share"):
        self.output_dir = Path(output_dir)
        self._pine_gen = PineGenerator()
        self._replay_builder = ReplayBuilder()

    def build(
        self,
        trade_id: int,
        trade: TradeRecord,
        confirmed: Optional[FrozenSchematic],
        candles_df: Optional[pd.DataFrame],
        suggested_anchors: Optional[list[AnchorPoint]] = None,
        comparisons: Optional[list] = None,
        accuracy: Optional[AccuracySummary] = None,
        label: Optional[str] = None,
        tags: Optional[list[str]] = None,
        notes: Optional[str] = None,
    ) -> Path:
        """Build a complete shareable replay package for one trade.

        Args:
            trade_id: Internal trade ID.
            trade: Normalized trade record.
            confirmed: Frozen schematic (may be None).
            candles_df: OHLCV DataFrame (may be None).
            suggested_anchors: Pre-built suggested anchor list.
            comparisons: Pre-built anchor comparisons.
            accuracy: Pre-built accuracy summary.
            label: Folder label override.

        Returns:
            Path to the created package folder.
        """
        safe_label = label or f"trade_{trade_id}"
        safe_label = safe_label.replace("/", "_").replace(" ", "_").replace(":", "_")
        pkg_dir = self.output_dir / safe_label
        pkg_dir.mkdir(parents=True, exist_ok=True)

        generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        symbol = trade.symbol
        timeframe = trade.timeframe or "n/a"
        side = trade.side

        # ── 1. Replay HTML + JSON (reuse ReplayBuilder + template) ────
        payload = self._replay_builder.build(
            trade_id=trade_id,
            trade=trade,
            confirmed=confirmed,
            candles_df=candles_df,
            suggested_anchors=suggested_anchors,
            comparisons=comparisons,
            accuracy=accuracy,
            tags=tags,
            notes=notes,
        )
        payload_json = payload.to_json()

        title = f"Trade #{trade_id} — {symbol} {timeframe} {trade.direction}"
        replay_html = render_replay_html(
            payload_json, title=title, pine_script=payload.pine_script,
        )
        (pkg_dir / "replay.html").write_text(replay_html, encoding="utf-8")
        (pkg_dir / "replay_data.json").write_text(payload_json, encoding="utf-8")

        # ── 2. Pine Script (reuse PineGenerator) ──────────────────────
        render_payload = RenderPayload(
            trade_id=trade_id, trade=trade, schematic=confirmed,
        )
        pine_path = self._pine_gen.generate_single(render_payload, pkg_dir)
        # Rename to standard name
        target_pine = pkg_dir / "overlay.pine"
        if pine_path.name != "overlay.pine":
            if target_pine.exists():
                target_pine.unlink()
            pine_path.rename(target_pine)
        pine_content = target_pine.read_text(encoding="utf-8")

        # Clean up any extra generated .pine files
        for f in pkg_dir.glob("tct_overlay_*.pine"):
            f.unlink()

        # ── 3. Pine HTML launcher (reuse share html_template) ─────────
        pine_html = render_pine_html(
            pine_script=pine_content,
            title=f"Pine Overlay — {symbol} {timeframe}",
            symbol=symbol,
            timeframe=timeframe,
            trade_count=1,
            generated_at=generated_at,
        )
        (pkg_dir / "open_chart.html").write_text(pine_html, encoding="utf-8")

        # ── 4. README ─────────────────────────────────────────────────
        readme = build_readme(
            trade_ids=[trade_id],
            symbol=symbol,
            timeframe=timeframe,
            side=side,
            model=trade.model,
            has_confirmed=confirmed is not None,
            has_suggested=bool(suggested_anchors),
            has_accuracy=accuracy is not None,
            generated_at=generated_at,
            tags=tags,
            notes=notes,
        )
        (pkg_dir / "README.txt").write_text(readme, encoding="utf-8")

        # ── 5. Manifest ───────────────────────────────────────────────
        files = ["replay.html", "replay_data.json", "overlay.pine",
                 "open_chart.html", "README.txt", "manifest.json"]
        manifest = build_manifest(
            trade_ids=[trade_id],
            symbol=symbol,
            timeframe=timeframe,
            side=side,
            files=files,
            has_confirmed_schematic=confirmed is not None,
            has_suggested_schematic=bool(suggested_anchors),
            has_accuracy_report=accuracy is not None,
            model=trade.model,
            tags=tags,
            notes=notes,
        )
        (pkg_dir / "manifest.json").write_text(manifest, encoding="utf-8")

        return pkg_dir
