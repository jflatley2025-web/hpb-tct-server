# Plan: Jack's TCT Mode + Auto-scan Fix

## Scope

1. **Fix auto-scan hang** (already partially addressed — reduced LTF limits + stagger; this plan confirms no further structural change is needed there)
2. **Rename existing evaluator** → "Claude's TCT Mode"
3. **Create "Jack's TCT Mode"** with a separate evaluator (`jack_tct_evaluator.py`)
4. **Mode-switching dropdown** on the `/schematics-5B` page
5. **Mode-aware debug panel** showing Jack's 5-tree data when that mode is active

---

## Architecture Review

### Issue 1 — Where does the mode live?

**Problem**: The page needs to persist which mode is selected (Claude vs Jack) across page reloads and 60s scan cycles, but not necessarily across server restarts.

**Options:**

**A (Recommended) — In-memory on the trader singleton, persisted to the existing `schematics_5b_trade_log.json`**
- Add `"trading_mode": "claude"` field to the trade log JSON
- `Schematics5BTrader` reads it on startup and exposes a `set_mode()/get_mode()` method
- A new `POST /api/schematics-5b-trader/mode` endpoint writes through to trader and disk
- Implementation effort: Low
- Risk: Minimal — appending one field to an existing JSON
- Impact: No change to existing trade history or state fields
- Maintenance: Low — naturally co-located with state

**B — Separate `mode.json` config file**
- Clean separation of config vs state
- Implementation effort: Low
- Risk: One more file to manage / back up
- Maintenance: Low but slightly more surface area

**C — In-memory only (reset to default on restart)**
- Simplest; no disk I/O for mode changes
- Risk: Mode resets to "claude" on every deploy/restart
- Maintenance: Lowest

**Recommendation: Option A.** Persisting inside the existing JSON is the DRY choice — no new files, no new I/O paths, and survives deploy restarts.

---

### Issue 2 — How do the two evaluators co-exist?

**Problem**: `Schematics5BTrader` currently holds a single `DecisionTreeEvaluator`. We need it to call Jack's logic when in Jack mode.

**Options:**

**A (Recommended) — Two separate evaluator classes, trader switches at call time**
- `jack_tct_evaluator.py` exports `JackTCTEvaluator` with the same `evaluate_schematic(...)` interface
- `Schematics5BTrader._get_evaluator()` returns the correct one based on `self._mode`
- No change to the `evaluate_schematic(...)` call site in `_scan_and_trade_locked()`
- Implementation effort: Medium (new evaluator file)
- Risk: Low — clean interface, no touching of Claude's evaluator
- Impact: Additive only; Claude mode unchanged

**B — Single evaluator with a mode flag**
- Pass `mode` into `DecisionTreeEvaluator` constructor; one class, two paths
- Implementation effort: Medium
- Risk: Medium — Claude's 6-tree logic entangled with Jack's 5-tree logic
- Maintenance: Higher; changes to one path could break the other

**C — Subclass: `JackTCTEvaluator(DecisionTreeEvaluator)`**
- Override `evaluate_schematic`; reuse helpers
- Implementation effort: Low
- Risk: Moderate — inheritance creates tight coupling to Claude's internals

**Recommendation: Option A.** Separate files are the cleanest isolation. Both evaluators satisfy the same interface (`evaluate_schematic` → same return dict format) so `Schematics5BTrader` doesn't need to know the difference.

---

### Issue 3 — What does `jack_tct_evaluator.py` implement?

Jack's 5 trees map directly to data already available from `detect_tct_schematics()` + `market_structure.py` + the existing `_detect_*` helpers in `decision_tree_bridge.py`. **No new data sources needed.**

| Tree | Input data | Logic |
|------|-----------|-------|
| 1 — Market Structure | 4H candle df | Run `market_structure.py` to get confirmed MSH + MSL; check direction; gate on valid structure |
| 2 — Ranges | Schematic `range` field + candle df | Compute DL2, Range High, EQ, Range Low; check price position (premium/discount); validate horizontal range |
| 3 — Supply & Demand | Schematic taps + candle df | Detect OB + FVG near each tap; classify premium (supply) vs discount (demand) vs extreme zones |
| 4 — Liquidity | 4H candle df + schematic | Connect swing highs (bearish) or swing lows (bullish) with a trendline; check if price is touching/near it |
| 5 — TCT 5A | Schematic dict directly | Model 1/2, tap1/2/3 prices, BOS confirmed or not, BOS price, highest valid TF |

**Return format (same as Claude's evaluator):**
```python
{
    "score": int,          # 0-100
    "pass": bool,          # score >= 50
    "direction": str,
    "model": str,
    "rr": float,
    "required_score": 50,
    "reasons": List[str],
    "tree_results": {      # one key per tree, for debug panel
        "market_structure": {...},
        "ranges": {...},
        "supply_demand": {...},
        "liquidity": {...},
        "tct_5a": {...},
    }
}
```

**Scoring (Jack's mode):**
- Tree 1 Market Structure pass → 20 pts (hard gate if no valid structure)
- Tree 2 Ranges pass (price in correct zone) → 20 pts
- Tree 3 S/D + FVG → 20 pts (hard gate if no FVG)
- Tree 4 Liquidity curve present → 15 pts
- Tree 5 TCT 5A schematic confirmed → 25 pts (hard gate if not confirmed)
- Threshold: 50 pts (same as Claude's mode)

---

### Issue 4 — Trendline / liquidity curve detection (Tree 4)

Jack's Tree 4 asks: "Is there a trend line or liquidity curve forming?"

**Options:**

**A (Recommended) — Simple swing-point trendline check**
- For bullish setups: connect last 2 confirmed swing lows (`_is_swing_low()` from `tct_schematics.py`)
- For bearish setups: connect last 2 confirmed swing highs (`_is_swing_high()`)
- If the slope is consistent and price is within ~1% of that line near tap1/tap2 → YES
- Implementation effort: Low (reuses existing swing-point helpers)
- Risk: Low — conservative, matches TCT trendline definition

**B — Linear regression over recent swing points**
- More robust but requires scipy or numpy computation
- Implementation effort: Medium
- Risk: Medium — adds computation, harder to explain to traders

**C — Always return YES / skip this tree**
- Zero implementation effort
- Risk: High — meaningless tree, misleading pass/fail

**Recommendation: Option A.** Simple, deterministic, matches TCT methodology.

---

### Issue 5 — Debug panel rendering for Jack's mode

The current debug panel renders Claude's 7-category tree accordion. Jack's mode has 5 different categories.

**Options:**

**A (Recommended) — Mode-aware rendering in existing `renderDecisionTrees()` JS function**
- `d.trading_mode` returned in debug endpoint
- `renderDecisionTrees()` branches on `d.trading_mode === 'jack'` to render Jack's 5 trees with the correct live-scan fields
- Implementation effort: Medium (new JS rendering block)
- Risk: Low — additive, existing rendering untouched for Claude mode

**B — Separate API endpoint `/api/schematics-5b-trader/jack-debug`**
- Cleaner separation but requires two separate polling calls
- Implementation effort: Medium
- Risk: Low but unnecessary complexity

**Recommendation: Option A.** Same endpoint, mode-aware rendering — DRY, single polling call.

---

## Files to Create / Modify

| File | Action | What changes |
|------|--------|-------------|
| `jack_tct_evaluator.py` | **CREATE** | `JackTCTEvaluator` class with 5-tree pipeline |
| `schematics_5b_trader.py` | **MODIFY** | Add `_mode` field, `get_mode()`/`set_mode()`, load/save mode in JSON, `_get_evaluator()` helper |
| `server_mexc.py` | **MODIFY** | Add `GET/POST /api/schematics-5b-trader/mode` endpoints; include `trading_mode` in debug response |
| `server_mexc.py` (schematics-5B page HTML/JS) | **MODIFY** | Add mode dropdown in header; mode-aware `renderDecisionTrees()` for Jack's 5 trees; live scan data display per tree |

---

## Auto-scan Fix Status

The hang was caused by `fetch_candles_sync` fetching `5m×1000` + `1m×1000` candles (83h of 5m data) over synchronous `requests` in a background thread. Already committed:
- LTF limits reduced: 1000 → 200 (5m) / 100 (1m)
- Startup stagger reduced: 55s → 5s
- Diagnostic timing logs added to pinpoint any remaining slowness

No additional structural fix needed here — the next deploy's logs will confirm timing per phase.

---

## Questions Before Proceeding

**Q1 (Issue 1 — Mode persistence):** Does **Option A** (persisted inside `schematics_5b_trade_log.json`) work for you, or do you prefer **Option C** (in-memory only, resets on restart)?

**Q2 (Issue 3 — Scoring):** The proposed scoring for Jack's mode is:
- Tree 1 Market Structure: 20 pts (hard gate)
- Tree 2 Ranges: 20 pts
- Tree 3 S/D + FVG: 20 pts (hard gate)
- Tree 4 Liquidity curve: 15 pts
- Tree 5 TCT 5A confirmed: 25 pts (hard gate)

Do these weights and hard gates look right to you, or do you want different emphasis?

**Q3 (Issue 4 — Tree 4 Liquidity):** Are you OK with the **swing-point trendline approach** (Option A), or do you have a specific definition of "trendline or liquidity curve forming" you'd like to use?

**Q4 — Scan behavior in Jack's mode:** When Jack's mode is active, should it:
- **(a)** Still scan all MTF timeframes (1d, 4h, 1h, 30m) and pick the best TF
- **(b)** Only scan the 4H timeframe (since all of Jack's trees reference 4H)

Please answer each question — I'll implement after your approval.
