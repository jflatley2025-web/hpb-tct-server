"""Index models — structured types for replay package discovery."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ReplayPackageFiles:
    """Presence flags for expected files in a replay package."""
    has_replay_html: bool = False
    has_replay_data: bool = False
    has_overlay_pine: bool = False
    has_open_chart: bool = False
    has_readme: bool = False
    has_manifest: bool = False

    @property
    def complete(self) -> bool:
        return all([
            self.has_replay_html, self.has_overlay_pine,
            self.has_open_chart, self.has_readme, self.has_manifest,
        ])

    @property
    def file_count(self) -> int:
        return sum([
            self.has_replay_html, self.has_replay_data,
            self.has_overlay_pine, self.has_open_chart,
            self.has_readme, self.has_manifest,
        ])


@dataclass
class ReplayPackageEntry:
    """One discovered replay package with metadata from manifest.json."""
    package_name: str           # folder name
    relative_dir: str           # relative path from index.html location
    trade_ids: list[int] = field(default_factory=list)
    trade_count: int = 0
    symbol: str = ""
    timeframe: str = ""
    side: str = ""
    model: Optional[str] = None
    created_at: str = ""
    has_confirmed_schematic: bool = False
    has_suggested_schematic: bool = False
    has_accuracy_report: bool = False
    tags: list[str] = field(default_factory=list)
    notes: Optional[str] = None
    files: ReplayPackageFiles = field(default_factory=ReplayPackageFiles)

    @property
    def replay_link(self) -> Optional[str]:
        if self.files.has_replay_html:
            return f"{self.relative_dir}/replay.html"
        return None

    @property
    def chart_link(self) -> Optional[str]:
        if self.files.has_open_chart:
            return f"{self.relative_dir}/open_chart.html"
        return None

    @property
    def readme_link(self) -> Optional[str]:
        if self.files.has_readme:
            return f"{self.relative_dir}/README.txt"
        return None


@dataclass
class ReplayIndexSummary:
    """Aggregate summary for the index page header."""
    total_packages: int = 0
    unique_symbols: list[str] = field(default_factory=list)
    unique_timeframes: list[str] = field(default_factory=list)
    with_confirmed: int = 0
    with_suggested: int = 0
    with_accuracy: int = 0
    total_trades: int = 0

    @staticmethod
    def from_entries(entries: list[ReplayPackageEntry]) -> ReplayIndexSummary:
        symbols = sorted(set(e.symbol for e in entries if e.symbol))
        timeframes = sorted(set(e.timeframe for e in entries if e.timeframe))
        return ReplayIndexSummary(
            total_packages=len(entries),
            unique_symbols=symbols,
            unique_timeframes=timeframes,
            with_confirmed=sum(1 for e in entries if e.has_confirmed_schematic),
            with_suggested=sum(1 for e in entries if e.has_suggested_schematic),
            with_accuracy=sum(1 for e in entries if e.has_accuracy_report),
            total_trades=sum(e.trade_count for e in entries),
        )
