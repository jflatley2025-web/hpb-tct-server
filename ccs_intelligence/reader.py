"""
reader.py — Streaming JSONL reader for CCS event files.

Reads logs/context_events_YYYY-MM-DD.jsonl files.
Handles missing files, malformed lines, and oversized files.
"""

import json
import os
from datetime import datetime, timezone

_LOGS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs"
)

_MAX_FILE_BYTES = 50 * 1024 * 1024  # 50 MB hard cap


def resolve_file(date_str: str | None = None) -> str | None:
    """Return path to CCS JSONL file for given date, or None if missing."""
    if date_str is None:
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    path = os.path.join(_LOGS_DIR, f"context_events_{date_str}.jsonl")
    return path if os.path.isfile(path) else None


def read_events(
    filepath: str, max_lines: int = 10_000
) -> tuple[list[dict], int, int]:
    """Parse a CCS JSONL file into validated event dicts.

    Args:
        filepath: Absolute path to JSONL file.
        max_lines: Maximum lines to read (tail if file exceeds).

    Returns:
        (events, malformed_count, file_size_bytes)

    If file exceeds _MAX_FILE_BYTES, reads only the last max_lines lines.
    """
    try:
        file_size = os.path.getsize(filepath)
    except OSError:
        return [], 0, 0

    if file_size == 0:
        return [], 0, 0

    lines = _read_lines(filepath, file_size, max_lines)

    events = []
    malformed = 0
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if not line.endswith("}"):
            malformed += 1
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                events.append(obj)
            else:
                malformed += 1
        except (json.JSONDecodeError, ValueError):
            malformed += 1

    return events, malformed, file_size


def _read_lines(filepath: str, file_size: int, max_lines: int) -> list[str]:
    """Read lines from file. Tail-reads if file is large."""
    if file_size > _MAX_FILE_BYTES:
        return _tail_read(filepath, max_lines)
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            all_lines = f.readlines()
        if len(all_lines) > max_lines:
            return all_lines[-max_lines:]
        return all_lines
    except OSError:
        return []


def _tail_read(filepath: str, max_lines: int) -> list[str]:
    """Read last N lines from a large file efficiently."""
    lines = []
    try:
        with open(filepath, "rb") as f:
            f.seek(0, 2)
            pos = f.tell()
            buf = b""
            while pos > 0 and len(lines) < max_lines + 1:
                chunk = min(8192, pos)
                pos -= chunk
                f.seek(pos)
                buf = f.read(chunk) + buf
                lines = buf.split(b"\n")
            return [l.decode("utf-8", errors="replace") for l in lines[-max_lines:]]
    except OSError:
        return []
