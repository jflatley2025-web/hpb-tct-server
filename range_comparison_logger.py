"""
range_comparison_logger.py — Range Engine Structural Diff Logger
Logs differences between L1 and L2 range detection to JSONL for
offline analysis during migration.
"""

import json
import logging
import os
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger("RangeComparisonLogger")

_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_LOG_PATH = os.path.join(_DIR, "logs", "range_engine_comparison.jsonl")


class RangeComparisonLogger:
    """Append-only JSONL logger for range engine comparison."""

    def __init__(self, log_path: str = DEFAULT_LOG_PATH):
        self._log_path = log_path
        dir_path = os.path.dirname(self._log_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

    def log_comparison(
        self,
        symbol: str,
        session: Optional[str],
        engine_used: str,
        l1_ranges: list[dict],
        l2_ranges: list[dict],
        deviation_detected: bool = False,
        liquidity_sweep_detected: bool = False,
    ):
        """Log a structural comparison between L1 and L2 range detection."""
        # Sort by range_size descending so index 0 is always the true best
        sorted_l1 = sorted(l1_ranges, key=lambda r: r.get("range_size", 0), reverse=True)
        sorted_l2 = sorted(l2_ranges, key=lambda r: r.get("range_size", 0), reverse=True)
        best_l1 = sorted_l1[0] if sorted_l1 else {}
        best_l2 = sorted_l2[0] if sorted_l2 else {}

        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": symbol,
            "session": session,
            "engine_used": engine_used,
            "L1_range_high": best_l1.get("range_high"),
            "L2_range_high": best_l2.get("range_high"),
            "L1_range_low": best_l1.get("range_low"),
            "L2_range_low": best_l2.get("range_low"),
            "L1_count": len(l1_ranges),
            "L2_count": len(l2_ranges),
            "range_duration": self._calc_duration(best_l2 or best_l1),
            "deviation_detected": deviation_detected,
            "liquidity_sweep_detected": liquidity_sweep_detected,
        }

        try:
            with open(self._log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except OSError as e:
            logger.warning(f"Failed to write comparison log: {e}")

    @staticmethod
    def _calc_duration(range_data: dict) -> Optional[int]:
        """Calculate range duration in candles."""
        high_idx = range_data.get("range_high_idx")
        low_idx = range_data.get("range_low_idx")
        if high_idx is not None and low_idx is not None:
            return abs(low_idx - high_idx)
        return None
