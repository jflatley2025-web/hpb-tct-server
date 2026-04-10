"""
parser.py — Event validation, normalization, and index building.

Validates envelope structure, filters by symbol, builds lightweight
indices for metric computation.
"""

from __future__ import annotations

_REQUIRED_KEYS = {"event_id", "ts", "symbol", "stage", "event_type", "payload", "refs"}

_VALID_STAGES = {"SCCE", "RANGE", "TAP", "BOS", "PO3", "TOP_RANGE"}

_STAGE_TYPES = {
    "SCCE": {"SCCE_CANDIDATE_CREATED", "SCCE_CANDIDATE_UPDATED", "SCCE_CANDIDATE_INVALIDATED"},
    "RANGE": {"RANGE_CREATED", "RANGE_UPDATED", "RANGE_INVALIDATED"},
    "TAP": {"TAP_PROGRESS_UPDATED"},
    "BOS": {"BOS_ATTEMPTED", "BOS_CONFIRMED", "BOS_FAILED"},
    "PO3": {"PO3_CONFLUENCE_TAGGED"},
    "TOP_RANGE": {"TOP_RANGE_CONTEXT_TAGGED"},
}


def validate_event(event: dict) -> bool:
    """Check if event has valid envelope structure and stage/type pairing."""
    if not _REQUIRED_KEYS.issubset(event.keys()):
        return False
    stage = event.get("stage")
    etype = event.get("event_type")
    if stage not in _VALID_STAGES:
        return False
    if etype not in _STAGE_TYPES.get(stage, set()):
        return False
    if not isinstance(event.get("payload"), dict):
        return False
    if not isinstance(event.get("refs"), dict):
        return False
    return True


def filter_and_normalize(
    events: list[dict], symbol: str | None = None
) -> tuple[list[dict], int]:
    """Filter events by symbol, validate, and sort by timestamp.

    Returns:
        (valid_events_sorted_by_ts, orphan_count)
    """
    valid = []
    orphans = 0
    for e in events:
        if not validate_event(e):
            orphans += 1
            continue
        if symbol and e.get("symbol") != symbol:
            continue
        valid.append(e)

    valid.sort(key=lambda e: e.get("ts", ""))
    return valid, orphans


def build_indices(events: list[dict]) -> dict:
    """Build lightweight lookup indices from validated events.

    Returns dict with:
        bos_attempts: {bos_attempt_id: {attempted: event, outcome: event|None}}
        candidates: {candidate_id: [events...]}
        ranges: {range_id: [events...]}
        tap3_events: [events sorted by ts]
        bos_timeline: [BOS_ATTEMPTED events sorted by ts]
    """
    bos_attempts: dict[str, dict] = {}
    candidates: dict[str, list[dict]] = {}
    ranges: dict[str, list[dict]] = {}
    tap3_events: list[dict] = []
    bos_timeline: list[dict] = []
    po3_events: list[dict] = []
    top_range_events: list[dict] = []

    seen_event_ids: set[str] = set()

    for e in events:
        eid = e.get("event_id", "")
        if eid in seen_event_ids:
            continue  # duplicate — first occurrence wins
        seen_event_ids.add(eid)

        etype = e["event_type"]
        refs = e.get("refs") or {}

        # BOS index
        if etype in ("BOS_ATTEMPTED", "BOS_CONFIRMED", "BOS_FAILED"):
            bid = refs.get("bos_attempt_id")
            if bid:
                if bid not in bos_attempts:
                    bos_attempts[bid] = {"attempted": None, "outcome": None}
                if etype == "BOS_ATTEMPTED":
                    if bos_attempts[bid]["attempted"] is None:
                        bos_attempts[bid]["attempted"] = e
                        bos_timeline.append(e)
                else:
                    if bos_attempts[bid]["outcome"] is None:
                        bos_attempts[bid]["outcome"] = e

        # Candidate index
        if etype in ("SCCE_CANDIDATE_CREATED", "SCCE_CANDIDATE_UPDATED",
                      "SCCE_CANDIDATE_INVALIDATED"):
            cid = refs.get("candidate_id")
            if cid:
                candidates.setdefault(cid, []).append(e)

        # Range index
        if etype in ("RANGE_CREATED", "RANGE_UPDATED", "RANGE_INVALIDATED"):
            rid = refs.get("range_id")
            if rid:
                ranges.setdefault(rid, []).append(e)

        # PO3 index
        if etype == "PO3_CONFLUENCE_TAGGED":
            po3_events.append(e)

        # Top Range index
        if etype == "TOP_RANGE_CONTEXT_TAGGED":
            top_range_events.append(e)

        # TAP index
        if etype == "TAP_PROGRESS_UPDATED":
            cid = refs.get("candidate_id")
            if cid:
                candidates.setdefault(cid, []).append(e)
            rid = refs.get("range_id")
            if rid:
                ranges.setdefault(rid, []).append(e)
            if e.get("payload", {}).get("tap_number") == 3:
                tap3_events.append(e)

    return {
        "bos_attempts": bos_attempts,
        "candidates": candidates,
        "ranges": ranges,
        "tap3_events": tap3_events,
        "bos_timeline": bos_timeline,
        "po3_events": po3_events,
        "top_range_events": top_range_events,
    }
