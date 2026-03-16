"""
range_comparison_logger.py — JSONL logger for range engine comparison data.

Logs range detection results (L1, L2, or combined) to a JSONL file for
offline analysis and backtesting comparison.
"""

import json
import os
import logging
from datetime import datetime
from typing import Dict, Optional

logger = logging.getLogger("RangeComparisonLogger")

DEFAULT_LOG_PATH = "logs/range_engine_comparison.jsonl"


class RangeComparisonLogger:
    """
    Append-only JSONL logger for range engine comparison results.

    Parameters
    ----------
    log_path : str
        Path to the JSONL log file.  Defaults to
        ``logs/range_engine_comparison.jsonl``.
    """

    def __init__(self, log_path: str = DEFAULT_LOG_PATH) -> None:
        self._log_path = log_path
        dir_path = os.path.dirname(self._log_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

    def log(
        self,
        symbol: str,
        timeframe: str,
        engine: str,
        ranges: list,
        *,
        metadata: Optional[Dict] = None,
    ) -> None:
        """
        Append one comparison entry to the log file.

        Parameters
        ----------
        symbol : str
            Trading pair (e.g. "BTCUSDT").
        timeframe : str
            Candle timeframe (e.g. "4h", "1h").
        engine : str
            Engine identifier (e.g. "L1", "L2", "legacy").
        ranges : list
            Detected range dicts.
        metadata : dict, optional
            Extra key-value pairs to include in the entry.
        """
        entry: Dict = {
            "timestamp": datetime.utcnow().isoformat(),
            "symbol": symbol,
            "timeframe": timeframe,
            "engine": engine,
            "range_count": len(ranges),
            "ranges": ranges,
        }
        if metadata:
            entry["metadata"] = metadata

        try:
            with open(self._log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, default=str) + "\n")
        except OSError as e:
            logger.warning("Failed to write comparison log: %s", e)
