"""Tag normalization — consistent, safe tag formatting.

Convention:
  - Whitespace trimmed
  - Internal spaces replaced with underscore
  - Preserves case (A+ stays A+, NY_Open stays NY_Open)
  - Empty/whitespace-only tags rejected
  - Special characters preserved (A+, #1, etc.)
  - Max length enforced (64 chars)
"""
from __future__ import annotations

import re

MAX_TAG_LENGTH = 64


def normalize_tag(raw: str) -> str | None:
    """Normalize a single tag string.

    Returns:
        Normalized tag string, or None if invalid/empty.
    """
    if not raw or not isinstance(raw, str):
        return None

    tag = raw.strip()
    if not tag:
        return None

    # Collapse internal whitespace to underscore
    tag = re.sub(r"\s+", "_", tag)

    # Enforce max length
    if len(tag) > MAX_TAG_LENGTH:
        tag = tag[:MAX_TAG_LENGTH]

    return tag


def normalize_tags(raw_tags: list[str]) -> list[str]:
    """Normalize a list of tags, removing duplicates and invalids.

    Preserves insertion order after dedup.
    """
    seen: set[str] = set()
    result: list[str] = []
    for raw in raw_tags:
        tag = normalize_tag(raw)
        if tag and tag not in seen:
            seen.add(tag)
            result.append(tag)
    return result
