"""Index builder — scans packages and generates index.html."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from local_chart_overlay.replay_index.scanner import scan_packages
from local_chart_overlay.replay_index.models import ReplayIndexSummary
from local_chart_overlay.replay_index.index_template import render_index_html


class IndexBuilder:
    """Scans a replay-share directory and builds an index.html."""

    def build(
        self,
        input_dir: str | Path,
        output_path: Optional[str | Path] = None,
        recursive: bool = False,
    ) -> Path:
        """Scan packages and generate index.html.

        Args:
            input_dir: Directory containing replay share packages.
            output_path: Where to write index.html.
                         Defaults to <input_dir>/index.html.
            recursive: Scan subdirectories recursively.

        Returns:
            Path to the generated index.html.
        """
        input_dir = Path(input_dir)
        if output_path:
            out = Path(output_path)
        else:
            out = input_dir / "index.html"

        # index.html sits in the same dir as packages, so relative_to = parent of index
        relative_to = out.parent

        entries = scan_packages(
            input_dir, relative_to=relative_to, recursive=recursive,
        )
        summary = ReplayIndexSummary.from_entries(entries)
        generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        html = render_index_html(
            entries=entries,
            summary=summary,
            generated_at=generated_at,
            input_dir_label=str(input_dir),
        )

        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(html, encoding="utf-8")
        return out
