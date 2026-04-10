"""
metrics.py — Four metric calculators for CCS v2.

Each function takes pre-built indices and returns a flat dict.
Pure functions. No I/O. No side effects.
"""

from __future__ import annotations
from statistics import median
from datetime import datetime, timezone


# ── A) BOS Attempt Success Rate ──────────────────────────────────

def compute_bos_stats(indices: dict) -> dict:
    """Compute BOS attempt success rate and failure breakdown."""
    attempts = indices.get("bos_attempts", {})

    confirmed = 0
    failed = 0
    pending = 0
    compression_fail = 0
    micro_bos_fail = 0
    unknown_fail = 0
    df_post_confirmed: list[int] = []
    df_post_failed: list[int] = []

    for bid, entry in attempts.items():
        att = entry.get("attempted")
        out = entry.get("outcome")

        if att is None:
            continue  # no attempted event — skip

        payload_att = att.get("payload") or {}
        df_post = payload_att.get("df_post_len")

        if out is None:
            pending += 1
            continue

        out_type = out.get("event_type")
        if out_type == "BOS_CONFIRMED":
            confirmed += 1
            if isinstance(df_post, (int, float)):
                df_post_confirmed.append(int(df_post))
        elif out_type == "BOS_FAILED":
            failed += 1
            if isinstance(df_post, (int, float)):
                df_post_failed.append(int(df_post))
            # Failure classification
            p = out.get("payload") or {}
            comp_ok = p.get("compression_ok")
            mbos_ok = p.get("micro_bos_ok")
            if comp_ok is False:
                compression_fail += 1
            elif mbos_ok is False:
                micro_bos_fail += 1
            else:
                unknown_fail += 1

    denom = confirmed + failed
    return {
        "bos_attempted": confirmed + failed + pending,
        "bos_confirmed": confirmed,
        "bos_failed": failed,
        "bos_pending": pending,
        "success_rate": round(confirmed / denom, 4) if denom > 0 else None,
        "failure_breakdown": {
            "compression_fail": compression_fail,
            "micro_bos_fail": micro_bos_fail,
            "unknown_fail": unknown_fail,
        },
        "avg_df_post_len_confirmed": (
            round(sum(df_post_confirmed) / len(df_post_confirmed), 1)
            if df_post_confirmed else None
        ),
        "avg_df_post_len_failed": (
            round(sum(df_post_failed) / len(df_post_failed), 1)
            if df_post_failed else None
        ),
    }


# ── B) Candidate Funnel ──────────────────────────────────────────

_PHASE_ORDER = {"seed": 0, "tap1": 1, "tap2": 2, "tap3": 3, "qualified": 4}


def compute_candidate_funnel(indices: dict) -> dict:
    """Compute SCCE candidate progression funnel."""
    candidates = indices.get("candidates", {})
    bos_attempts = indices.get("bos_attempts", {})

    # Collect candidate_ids that have a BOS_CONFIRMED outcome
    confirmed_cids: set[str] = set()
    for bid, entry in bos_attempts.items():
        out = entry.get("outcome")
        if out and out.get("event_type") == "BOS_CONFIRMED":
            cid = (out.get("refs") or {}).get("candidate_id")
            if cid:
                confirmed_cids.add(cid)

    funnel = {"seed": 0, "tap1": 0, "tap2": 0, "tap3": 0, "qualified": 0}
    invalidated = 0
    active = 0
    orphan = 0
    total_created = 0

    for cid, events in candidates.items():
        # Must have a CREATED event
        has_created = any(
            e["event_type"] == "SCCE_CANDIDATE_CREATED" for e in events
        )
        if not has_created:
            orphan += 1
            continue

        total_created += 1

        # Determine highest phase
        highest = 0  # seed
        is_invalidated = False
        inv_phase = "seed"

        for e in events:
            etype = e["event_type"]
            if etype == "TAP_PROGRESS_UPDATED":
                tn = (e.get("payload") or {}).get("tap_number", 0)
                tap_phase = f"tap{tn}" if 1 <= tn <= 3 else None
                if tap_phase and _PHASE_ORDER.get(tap_phase, 0) > highest:
                    highest = _PHASE_ORDER[tap_phase]
            elif etype == "SCCE_CANDIDATE_INVALIDATED":
                is_invalidated = True
                inv_phase = (e.get("payload") or {}).get(
                    "phase_at_invalidation", "seed"
                )

        if cid in confirmed_cids:
            highest = max(highest, _PHASE_ORDER["qualified"])

        # Cumulative funnel — count at every phase up to highest
        phase_names = ["seed", "tap1", "tap2", "tap3", "qualified"]
        for i, pname in enumerate(phase_names):
            if i <= highest:
                funnel[pname] += 1

        if highest >= _PHASE_ORDER["qualified"]:
            pass  # completed
        elif is_invalidated:
            invalidated += 1
        else:
            active += 1

    return {
        "total_created": total_created,
        "funnel": funnel,
        "invalidated": invalidated,
        "active": active,
        "orphan_events": orphan,
    }


# ── C) Range Tap Density ─────────────────────────────────────────

def compute_range_tap_density(indices: dict) -> dict:
    """Evaluate tap density per range and BOS association."""
    ranges = indices.get("ranges", {})
    bos_attempts = indices.get("bos_attempts", {})

    # Collect range_ids (== candidate_ids) with BOS_CONFIRMED
    confirmed_rids: set[str] = set()
    for bid, entry in bos_attempts.items():
        out = entry.get("outcome")
        if out and out.get("event_type") == "BOS_CONFIRMED":
            rid = (out.get("refs") or {}).get("candidate_id")
            if rid:
                confirmed_rids.add(rid)

    by_tap_count = {"0": 0, "1": 0, "2": 0, "3": 0}
    tap3_with_bos = 0
    tap3_without_bos = 0
    total_ranges = 0
    orphan = 0

    for rid, events in ranges.items():
        has_created = any(
            e["event_type"] == "RANGE_CREATED" for e in events
        )
        if not has_created:
            orphan += 1
            continue

        total_ranges += 1

        # Count distinct tap numbers for this range
        tap_nums: set[int] = set()
        for e in events:
            if e["event_type"] == "TAP_PROGRESS_UPDATED":
                tn = (e.get("payload") or {}).get("tap_number")
                if isinstance(tn, int) and 1 <= tn <= 3:
                    tap_nums.add(tn)

        count = len(tap_nums)
        bucket = str(min(count, 3))
        by_tap_count[bucket] += 1

        if count >= 3:
            if rid in confirmed_rids:
                tap3_with_bos += 1
            else:
                tap3_without_bos += 1

    return {
        "total_ranges": total_ranges,
        "by_tap_count": by_tap_count,
        "tap3_ranges_with_bos_confirmed": tap3_with_bos,
        "tap3_ranges_without_bos_confirmed": tap3_without_bos,
    }


# ── D) Tap3 → BOS Latency ────────────────────────────────────────

def compute_tap3_bos_latency(indices: dict) -> dict:
    """Measure time from Tap3 detection to first BOS attempt."""
    tap3_events = indices.get("tap3_events", [])
    bos_timeline = indices.get("bos_timeline", [])

    if not tap3_events:
        return {
            "tap3_events": 0,
            "tap3_with_bos_attempt": 0,
            "tap3_awaiting_bos": 0,
            "median_tap3_to_attempt_sec": None,
            "sample_size": 0,
        }

    # Parse timestamps
    tap3_ts = []
    for e in tap3_events:
        t = _parse_ts(e.get("ts"))
        if t is not None:
            tap3_ts.append(t)

    bos_ts = []
    for e in bos_timeline:
        t = _parse_ts(e.get("ts"))
        if t is not None:
            bos_ts.append(t)

    tap3_ts.sort()
    bos_ts.sort()

    # For each BOS attempt, assign to most recent preceding tap3
    latencies: list[float] = []
    matched_tap3: set[int] = set()  # indices into tap3_ts

    for bt in bos_ts:
        # Find most recent tap3 before this BOS
        best_idx = None
        for i in range(len(tap3_ts) - 1, -1, -1):
            if tap3_ts[i] < bt:
                best_idx = i
                break
        if best_idx is not None:
            delta = (bt - tap3_ts[best_idx]).total_seconds()
            if delta > 0:
                latencies.append(delta)
                matched_tap3.add(best_idx)

    tap3_with = len(matched_tap3)
    tap3_without = len(tap3_ts) - tap3_with

    return {
        "tap3_events": len(tap3_ts),
        "tap3_with_bos_attempt": tap3_with,
        "tap3_awaiting_bos": tap3_without,
        "median_tap3_to_attempt_sec": (
            round(median(latencies), 1) if len(latencies) >= 3 else None
        ),
        "sample_size": len(latencies),
    }


# ── E) PO3 Confluence ───────────────────────────────────────────

def compute_po3_confluence(indices: dict) -> dict:
    """Compute PO3 confluence distribution from persisted CCS events.

    Pure read of po3_events index. No side effects.
    """
    po3_events = indices.get("po3_events", [])

    if not po3_events:
        return {
            "po3_events": 0,
            "by_phase": {},
            "by_direction": {},
            "avg_quality": None,
            "avg_range_overlap_pct": None,
        }

    by_phase: dict[str, int] = {}
    by_direction: dict[str, int] = {}
    qualities: list[float] = []
    overlaps: list[float] = []

    for e in po3_events:
        p = e.get("payload") or {}

        phase = p.get("po3_phase", "unknown")
        by_phase[phase] = by_phase.get(phase, 0) + 1

        direction = p.get("po3_direction", "unknown")
        by_direction[direction] = by_direction.get(direction, 0) + 1

        q = p.get("po3_quality")
        if isinstance(q, (int, float)):
            qualities.append(float(q))

        o = p.get("po3_range_overlap_pct")
        if isinstance(o, (int, float)):
            overlaps.append(float(o))

    return {
        "po3_events": len(po3_events),
        "by_phase": by_phase,
        "by_direction": by_direction,
        "avg_quality": round(sum(qualities) / len(qualities), 4) if qualities else None,
        "avg_range_overlap_pct": round(sum(overlaps) / len(overlaps), 1) if overlaps else None,
    }


# ── F) Top Range Correlation ────────────────────────────────────

def compute_top_range_correlation(indices: dict) -> dict:
    """Compute in-range vs out-of-range distribution from CCS events.

    Pure read of top_range_events index. No side effects.
    """
    events = indices.get("top_range_events", [])

    if not events:
        return {
            "tagged_events": 0,
            "in_range_count": 0,
            "out_of_range_count": 0,
            "in_range_pct": None,
            "avg_range_strength_in": None,
        }

    in_range = 0
    out_range = 0
    strengths: list[float] = []

    for e in events:
        p = e.get("payload") or {}
        if p.get("in_top_range"):
            in_range += 1
            s = p.get("range_strength")
            if isinstance(s, (int, float)):
                strengths.append(float(s))
        else:
            out_range += 1

    total = in_range + out_range
    return {
        "tagged_events": len(events),
        "in_range_count": in_range,
        "out_of_range_count": out_range,
        "in_range_pct": round(in_range / total * 100, 1) if total > 0 else None,
        "avg_range_strength_in": round(sum(strengths) / len(strengths), 2) if strengths else None,
    }


def _parse_ts(ts_str: str | None) -> datetime | None:
    """Parse ISO timestamp string to datetime. Returns None on failure."""
    if not ts_str or not isinstance(ts_str, str):
        return None
    try:
        return datetime.fromisoformat(ts_str)
    except (ValueError, TypeError):
        return None
