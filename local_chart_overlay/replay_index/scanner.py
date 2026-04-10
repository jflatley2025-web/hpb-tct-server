"""Scanner — discovers replay share packages by finding manifest.json files."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from local_chart_overlay.replay_index.models import (
    ReplayPackageEntry, ReplayPackageFiles,
)

logger = logging.getLogger(__name__)

EXPECTED_FILES = {
    "replay.html": "has_replay_html",
    "replay_data.json": "has_replay_data",
    "overlay.pine": "has_overlay_pine",
    "open_chart.html": "has_open_chart",
    "README.txt": "has_readme",
    "manifest.json": "has_manifest",
}


def scan_packages(
    input_dir: Path,
    relative_to: Optional[Path] = None,
    recursive: bool = False,
) -> list[ReplayPackageEntry]:
    """Scan a directory for replay share packages.

    Discovery rule: a directory is a package if it contains manifest.json.

    Args:
        input_dir: Root directory to scan.
        relative_to: Base path for computing relative links.
                     Defaults to input_dir.
        recursive: If True, scan subdirectories recursively.

    Returns:
        List of discovered ReplayPackageEntry objects, sorted by package_name.
    """
    input_dir = Path(input_dir)
    if not input_dir.exists():
        logger.warning(f"Input directory does not exist: {input_dir}")
        return []

    relative_to = relative_to or input_dir

    entries = []

    if recursive:
        manifest_files = sorted(input_dir.rglob("manifest.json"))
    else:
        # Check direct children only
        manifest_files = sorted(
            p for p in input_dir.iterdir()
            if p.is_dir() and (p / "manifest.json").exists()
        )
        manifest_files = [p / "manifest.json" for p in manifest_files]

    for manifest_path in manifest_files:
        pkg_dir = manifest_path.parent
        entry = _parse_package(pkg_dir, relative_to)
        if entry:
            entries.append(entry)

    return sorted(entries, key=lambda e: e.package_name)


def _parse_package(
    pkg_dir: Path,
    relative_to: Path,
) -> Optional[ReplayPackageEntry]:
    """Parse one package directory into a ReplayPackageEntry."""
    manifest_path = pkg_dir / "manifest.json"
    if not manifest_path.exists():
        return None

    # Load manifest
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Skipping {pkg_dir.name}: invalid manifest.json ({e})")
        return None

    if not isinstance(manifest, dict):
        logger.warning(f"Skipping {pkg_dir.name}: manifest is not a JSON object")
        return None

    # Detect files
    files = ReplayPackageFiles()
    for filename, attr in EXPECTED_FILES.items():
        setattr(files, attr, (pkg_dir / filename).exists())

    # Compute relative path (use forward slashes for HTML links)
    try:
        rel = pkg_dir.relative_to(relative_to)
        relative_dir = str(rel).replace("\\", "/")
    except ValueError:
        relative_dir = pkg_dir.name

    return ReplayPackageEntry(
        package_name=pkg_dir.name,
        relative_dir=relative_dir,
        trade_ids=manifest.get("trade_ids", []),
        trade_count=manifest.get("trade_count", 0),
        symbol=manifest.get("symbol", ""),
        timeframe=manifest.get("timeframe", ""),
        side=manifest.get("side", ""),
        model=manifest.get("model"),
        created_at=manifest.get("created_at", ""),
        has_confirmed_schematic=manifest.get("has_confirmed_schematic", False),
        has_suggested_schematic=manifest.get("has_suggested_schematic", False),
        has_accuracy_report=manifest.get("has_accuracy_report", False),
        tags=manifest.get("tags", []),
        notes=manifest.get("notes"),
        files=files,
    )
