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


# ── G) Structural Integrity Audit ──────────────────────────────

_DIR_MAP = {"accumulation": "bullish", "distribution": "bearish"}


def _check_tap_chronology(tap_events: list[dict]) -> list[tuple]:
    """Verify tap1.ts < tap2.ts < tap3.ts."""
    violations: list[tuple] = []
    by_num: dict[int, str] = {}
    for e in tap_events:
        tn = (e.get("payload") or {}).get("tap_number")
        ts = e.get("ts")
        if isinstance(tn, int) and 1 <= tn <= 3 and ts:
            if tn not in by_num:
                by_num[tn] = ts

    ordered = sorted(by_num.items())
    for i in range(1, len(ordered)):
        prev_num, prev_ts = ordered[i - 1]
        curr_num, curr_ts = ordered[i]
        if curr_ts < prev_ts:
            violations.append((
                "tap_chronology", "error",
                f"tap{curr_num} ({curr_ts}) before tap{prev_num} ({prev_ts})"
            ))
    return violations


def _check_range_validity(range_event: dict | None) -> list[tuple]:
    """Verify range_high > range_low."""
    if range_event is None:
        return []
    p = range_event.get("payload") or {}
    rh = p.get("range_high")
    rl = p.get("range_low")
    if rh is None or rl is None:
        return []
    if not isinstance(rh, (int, float)) or not isinstance(rl, (int, float)):
        return []
    if rh <= rl:
        return [("range_inversion", "error", f"range_high ({rh}) <= range_low ({rl})")]
    return []


def _check_tap_range_consistency(
    tap_events: list[dict], range_high: float, range_low: float, direction: str
) -> list[tuple]:
    """Verify taps align with expected boundary for direction."""
    violations: list[tuple] = []
    eq = (range_high + range_low) / 2
    rng_size = range_high - range_low
    if rng_size <= 0:
        return []

    for e in tap_events:
        p = e.get("payload") or {}
        price = p.get("price")
        tn = p.get("tap_number")
        if not isinstance(price, (int, float)) or not isinstance(tn, int):
            continue

        if direction == "bullish":
            # Taps should be near range_low (demand zone)
            if price > range_high + rng_size * 0.1:
                violations.append((
                    "tap_range_inconsistency", "warning",
                    f"tap{tn} price {price} far above range_high {range_high} for bullish"
                ))
        elif direction == "bearish":
            # Taps should be near range_high (supply zone)
            if price < range_low - rng_size * 0.1:
                violations.append((
                    "tap_range_inconsistency", "warning",
                    f"tap{tn} price {price} far below range_low {range_low} for bearish"
                ))
    return violations


def _check_bos_timing(bos_ts: str | None, last_tap_ts: str | None) -> list[tuple]:
    """Verify BOS occurs after last tap."""
    if bos_ts is None or last_tap_ts is None:
        return []
    bos_dt = _parse_ts(bos_ts)
    tap_dt = _parse_ts(last_tap_ts)
    if bos_dt is None or tap_dt is None:
        return []
    if bos_dt < tap_dt:
        return [("bos_before_last_tap", "warning",
                 f"BOS ({bos_ts}) before last tap ({last_tap_ts})")]
    return []


def _check_bos_price_direction(
    bos_price: float, range_high: float, range_low: float, direction: str
) -> list[tuple]:
    """Verify BOS price is on correct side of range for direction."""
    if direction == "bullish" and bos_price < range_low:
        return [("bos_price_wrong_side", "error",
                 f"bullish BOS price {bos_price} below range_low {range_low}")]
    if direction == "bearish" and bos_price > range_high:
        return [("bos_price_wrong_side", "error",
                 f"bearish BOS price {bos_price} above range_high {range_high}")]
    return []


def _check_phase_progression(candidate_events: list[dict]) -> list[tuple]:
    """Verify phase transitions are monotonically increasing."""
    violations: list[tuple] = []
    for e in candidate_events:
        if e.get("event_type") != "SCCE_CANDIDATE_UPDATED":
            continue
        p = e.get("payload") or {}
        pb = p.get("phase_before", "")
        pa = p.get("phase_after", "")
        order_b = _PHASE_ORDER.get(pb, -1)
        order_a = _PHASE_ORDER.get(pa, -1)
        if order_b >= 0 and order_a >= 0 and order_a < order_b:
            violations.append((
                "phase_regression", "warning",
                f"phase regressed from {pb} to {pa}"
            ))
    return violations


def compute_structure_integrity(indices: dict) -> dict:
    """Validate structural integrity of SCCE candidates from CCS events.

    Runs 6 rules against data available in JSONL indices.
    Pure function. No I/O. No side effects.
    """
    candidates = indices.get("candidates", {})
    ranges = indices.get("ranges", {})
    bos_attempts = indices.get("bos_attempts", {})

    # Pre-build BOS outcome lookup: candidate_id → BOS_CONFIRMED event
    bos_by_cid: dict[str, dict] = {}
    for bid, entry in bos_attempts.items():
        out = entry.get("outcome")
        if out and out.get("event_type") == "BOS_CONFIRMED":
            cid = (out.get("refs") or {}).get("candidate_id")
            if cid:
                bos_by_cid[cid] = out

    all_rules = [
        "tap_chronology", "range_inversion", "tap_range_inconsistency",
        "bos_before_last_tap", "bos_price_wrong_side", "phase_regression",
    ]
    rule_counts = {r: 0 for r in all_rules}
    rule_examples: dict[str, list[str]] = {r: [] for r in all_rules}
    sev_counts = {"error": 0, "warning": 0}

    valid_count = 0
    warning_count = 0
    invalid_count = 0
    valid_bos = 0
    invalid_bos = 0
    warning_bos = 0
    audited = 0
    invalid_bos_details: list[dict] = []
    # Time bucket accumulators: hour_key → {invalid, warning, total}
    _hour_buckets: dict[str, dict] = {}

    for cid, events in candidates.items():
        # Must have a CREATED event to audit
        created = None
        for e in events:
            if e.get("event_type") == "SCCE_CANDIDATE_CREATED":
                created = e
                break
        if created is None:
            continue

        audited += 1
        violations: list[tuple] = []

        # Extract components
        tap_events = [e for e in events if e.get("event_type") == "TAP_PROGRESS_UPDATED"]
        model_family = (created.get("payload") or {}).get("model_family", "unknown")
        direction = _DIR_MAP.get(model_family)

        # Get range data (range_id == candidate_id)
        range_evts = ranges.get(cid, [])
        range_created = None
        for e in range_evts:
            if e.get("event_type") == "RANGE_CREATED":
                range_created = e
                break

        rh, rl = None, None
        if range_created:
            p = range_created.get("payload") or {}
            rh = p.get("range_high")
            rl = p.get("range_low")

        # Get BOS outcome for this candidate
        bos_event = bos_by_cid.get(cid)

        # ── Run rules ──

        # 1. Tap chronology
        if len(tap_events) >= 2:
            violations.extend(_check_tap_chronology(tap_events))

        # 2. Range validity
        violations.extend(_check_range_validity(range_created))

        # 3. Tap-range consistency (needs direction + range)
        if direction and rh is not None and rl is not None and tap_events:
            violations.extend(
                _check_tap_range_consistency(tap_events, rh, rl, direction)
            )

        # 4. BOS timing (needs BOS + taps)
        if bos_event and tap_events:
            last_tap_ts = max(
                (e.get("ts", "") for e in tap_events), default=None
            )
            violations.extend(_check_bos_timing(bos_event.get("ts"), last_tap_ts))

        # 5. BOS price direction (needs BOS + range + direction)
        if bos_event and direction and rh is not None and rl is not None:
            bp = (bos_event.get("payload") or {}).get("bos_price")
            if isinstance(bp, (int, float)):
                violations.extend(
                    _check_bos_price_direction(bp, rh, rl, direction)
                )

        # 6. Phase progression
        violations.extend(_check_phase_progression(events))

        # ── Classify ──
        has_error = any(v[1] == "error" for v in violations)
        has_warning = any(v[1] == "warning" for v in violations)
        has_bos = cid in bos_by_cid

        if has_error:
            invalid_count += 1
            if has_bos:
                invalid_bos += 1
                _cp = created.get("payload") or {}
                invalid_bos_details.append({
                    "candidate_id": cid,
                    "violated_rules": [v[0] for v in violations if v[1] == "error"],
                    "symbol": created.get("symbol", "unknown"),
                    "timeframe": _cp.get("timeframe", ""),
                    "ts": created.get("ts", ""),
                })
        elif has_warning:
            warning_count += 1
            if has_bos:
                warning_bos += 1
        else:
            valid_count += 1
            if has_bos:
                valid_bos += 1

        # ── Time bucket ──
        _created_ts = created.get("ts", "")
        if len(_created_ts) >= 13:
            _hk = _created_ts[:13]  # "2026-04-10T10"
            if _hk not in _hour_buckets:
                _hour_buckets[_hk] = {"invalid": 0, "warning": 0, "total": 0}
            _hour_buckets[_hk]["total"] += 1
            if has_error:
                _hour_buckets[_hk]["invalid"] += 1
            elif has_warning:
                _hour_buckets[_hk]["warning"] += 1

        # ── Aggregate ──
        for rule, sev, _msg in violations:
            if rule in rule_counts:
                rule_counts[rule] += 1
                if len(rule_examples[rule]) < 5:
                    rule_examples[rule].append(cid)
            if sev in sev_counts:
                sev_counts[sev] += 1

    # ── Rule percentage share ──
    total_violations = sum(rule_counts.values())
    rule_pct = {
        r: round(c / total_violations * 100, 1) if total_violations > 0 else 0.0
        for r, c in rule_counts.items()
    }

    # ── Time buckets ──
    time_buckets = {}
    for hk, b in sorted(_hour_buckets.items()):
        t = b["total"]
        time_buckets[hk] = {
            "candidates": t,
            "invalid_rate": round(b["invalid"] / t, 4) if t > 0 else 0.0,
            "warning_rate": round(b["warning"] / t, 4) if t > 0 else 0.0,
        }

    return {
        "candidates_audited": audited,
        "valid": valid_count,
        "warning": warning_count,
        "invalid": invalid_count,
        "valid_pct": round(valid_count / audited * 100, 1) if audited > 0 else None,
        "invalid_pct": round(invalid_count / audited * 100, 1) if audited > 0 else None,
        "by_rule": rule_counts,
        "by_severity": sev_counts,
        "rule_pct": rule_pct,
        "valid_with_bos_confirmed": valid_bos,
        "invalid_with_bos_confirmed": invalid_bos,
        "warning_with_bos_confirmed": warning_bos,
        "invalid_bos_details": invalid_bos_details,
        "time_buckets": time_buckets,
        "examples": {r: ids for r, ids in rule_examples.items() if ids},
    }


def _parse_ts(ts_str: str | None) -> datetime | None:
    """Parse ISO timestamp string to datetime. Returns None on failure."""
    if not ts_str or not isinstance(ts_str, str):
        return None
    try:
        return datetime.fromisoformat(ts_str)
    except (ValueError, TypeError):
        return None
