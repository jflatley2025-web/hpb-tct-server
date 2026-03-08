# Plan: Replace Schematics-5B Trading Bot with 6-Tree Decision Pipeline

## Summary

Replace the hardcoded 50-point scoring system in `schematics_5b_trader.py` with a full 6-tree decision pipeline that chains all decision tree files from `decision_trees/`. BTCUSDT only. Preserve all execution infrastructure.

---

## Architecture: What Changes vs What Stays

### STAYS (untouched)
- `Schematics5BTradeState` — state persistence, save/load, GitHub sync
- `_enter_trade()` — position sizing, margin, leverage, TP1 calc
- `_manage_open_trade()` — TP1 partial close, SL-to-BE, target/stop monitoring
- `_close_trade()` / `_take_partial_profit()` — PnL calc, balance update
- `_is_duplicate_setup()` — cooldown + price tolerance
- Telegram notifications — `_notify_5b_entry/tp1/exit`
- HTF bias detection — `_get_htf_bias()` (used as one input to the pipeline)
- LTF BOS refinement — `refine_schematic_bos_with_ltf()`
- HTF upgrade cascade — `_find_htf_upgrade()`
- Background scan loop in `server_mexc.py`
- Dashboard HTML layout (update debug section only)

### REPLACED
- `Schematics5BEvaluator` class → new `DecisionTreeEvaluator` class
- The 6 hardcoded gates (BOS, staleness, quality, R:R, HTF, model) → 6-tree pipeline
- `_scan_single_symbol()` scanning logic → new method that builds tree inputs from candles

### NEW
- `decision_tree_bridge.py` — module that maps raw candle data + schematic dicts → decision tree input dataclasses
- Import all 6 evaluate functions from `decision_trees/`

---

## Pipeline Design

Each scan cycle runs this pipeline for BTCUSDT across MTF timeframes:

```
Step 1: Fetch candles (1d, 4h, 1h, 30m) + LTF (5m, 1m)  [EXISTING]
Step 2: detect_tct_schematics() on each TF               [EXISTING]
Step 3: For each detected schematic, run the 6-tree pipeline:

  Tree 1 — Ranges (evaluate_range_setup)
    Input: Derive trend, 6-candle rule, EQ touch, horizontal check from candles
    Gate: range_valid must be True, trade_bias != WAIT

  Tree 2 — Supply/Demand (evaluate_sd_zone)
    Input: Derive zone type, FVG, mitigation from candle structure
    Gate: fvg_valid must be True, failed_at_phase must be None

  Tree 3 — Liquidity (evaluate_liquidity_setup)
    Input: Derive pool type, sweep side, DL2 breach, acceptance from candles
    Gate: sweep_classification must be LIQUIDITY_GRAB, accepted_back_inside True

  Tree 4 — TCT 5A (evaluate_tct_schematic)
    Input: Map schematic dict fields → TCTSchematicInputs
    Gate: status must be VALID_ENTRY, trade_bias != WAIT

  Tree 5 — TCT 5B (evaluate_5b_schematic)
    Input: Map schematic + 5A result → TCT5BInputs
    Gate: status must be VALID_ENTRY, trade_bias != WAIT

  Tree 6 — Advanced (evaluate_schematic_flip + evaluate_wyckoff_in_wyckoff)
    Input: If active trade exists → check flip; if tap3 zone → check W-in-W
    Effect: Can upgrade conviction or trigger flip exit

Step 4: Score surviving setups (all 5 gates passed)
  - Composite score from tree outputs (conviction, path quality, zone priority, R:R)
  - Best setup wins

Step 5: Enter trade via existing _enter_trade()           [EXISTING]
```

---

## New File: `decision_tree_bridge.py`

This is the core new code. It contains functions that translate raw market data into each tree's input dataclass:

### Functions:

1. `build_range_inputs(df, schematic) → RangeInputs`
   - Analyze candle highs/lows for HH/HL or LH/LL pattern
   - Check 6-candle rule (at least 6 candles touching both range boundaries)
   - Check EQ touch (price crossed midpoint of range)
   - Determine if range looks horizontal (range height < threshold relative to price)
   - Check deviation characteristics (wick only, close outside, etc.)

2. `build_sd_inputs(df, schematic, range_eval) → SDZoneInputs`
   - Identify order blocks near schematic tap zones
   - Check FVG (3-candle pattern where C1 wick doesn't overlap C3 wick)
   - Determine mitigation status from subsequent price action
   - Check multi-TF confluence from HTF data

3. `build_liquidity_inputs(df, schematic, range_eval) → LiquidityInputs`
   - Identify pool type from schematic range structure
   - Determine sweep side from tap2/tap3 deviation direction
   - Check DL2 breach (30% extension beyond range extreme)
   - Check acceptance back inside range
   - Assess path quality (clean move vs choppy)

4. `build_5a_inputs(schematic, range_eval) → TCTSchematicInputs`
   - Direct mapping from schematic dict fields
   - Range confirmed from tree 1 result
   - Model type, tap validation, BOS status from schematic

5. `build_5b_inputs(schematic, eval_5a, range_eval) → TCT5BInputs`
   - Rationality from range_eval
   - Tap spacing from schematic tap distances
   - R:R from schematic entry/stop/target
   - LTF BOS availability from refinement results

6. `build_flip_inputs(active_trade, schematic) → FlipInputs` (only when trade is open)

7. `build_wiw_inputs(schematic, local_schematic) → WyckoffInWyckoffInputs` (when nested pattern detected)

8. `compute_composite_score(range_eval, sd_eval, liq_eval, eval_5a, eval_5b) → dict`
   - Weighted composite from all tree outputs
   - Returns score (0–100), pass/fail, and reasons list
   - Compatible with existing _enter_trade() expectations

### DecisionTreeEvaluator class:
- `evaluate(df, schematic, htf_bias, current_price, ltf_dfs=None) → dict`
  - Orchestrates the full pipeline: build inputs → run trees → compute score
  - Returns `{"score": int, "pass": bool, "reasons": list, "rr": float, "tree_results": dict}`
  - `tree_results` contains per-tree outputs for debug endpoint

---

## Changes to `schematics_5b_trader.py`

### Replace `Schematics5BEvaluator`
- Delete the class entirely
- Replace with: `from decision_tree_bridge import DecisionTreeEvaluator`

### Modify `_scan_single_symbol()`
- Remove multi-symbol support (BTCUSDT only, no top_5_pairs)
- After detecting schematics on each TF, call `DecisionTreeEvaluator.evaluate(df, schematic, htf_bias)` which runs the full 6-tree pipeline
- Keep HTF cascade and LTF BOS refinement (run BEFORE tree evaluation so trees get refined data)

### Modify `scan_and_trade()`
- Remove `top_5_pairs` parameter
- Hardcode symbol = "BTCUSDT"
- Rest of flow stays the same

### Modify `_manage_open_trade()`
- Add Tree 6 flip detection: after fetching current price, also run flip check
- If flip detected → close current trade + enter new one

---

## Changes to `server_mexc.py`

### Background loop
- Remove top_5_pairs passing (just call `scan_and_trade()` with no args)

### Debug endpoint
- Update `/api/schematics-5b-trader/debug` to include tree evaluation phases:
  ```json
  {
    "tree_pipeline": {
      "ranges": {"passed": true, "trade_bias": "LONG", ...},
      "supply_demand": {"passed": true, "zone_priority": "EXTREME", ...},
      "liquidity": {"passed": true, "sweep": "LIQUIDITY_GRAB", ...},
      "tct_5a": {"passed": true, "status": "VALID_ENTRY", ...},
      "tct_5b": {"passed": true, "entry_tf": "RED_MID", ...},
      "advanced": {"flip": null, "wiw": null, "escalation": null}
    }
  }
  ```

### Dashboard HTML
- Update debug section to render tree pipeline results
- Add phase-by-phase pass/fail indicators

---

## File Changes Summary

| File | Action |
|---|---|
| `decision_tree_bridge.py` | **NEW** — bridge layer mapping candles → tree inputs |
| `schematics_5b_trader.py` | **MODIFY** — replace evaluator, simplify to BTCUSDT only |
| `server_mexc.py` | **MODIFY** — update debug endpoint + remove top_5_pairs from loop |
| `decision_trees/*.py` | **NO CHANGE** — used as-is via imports |

---

## Risk Mitigation

1. **Backward compatibility**: The bridge produces evaluation dicts with the same shape as current evaluator (`score`, `pass`, `reasons`, `rr`) so `_enter_trade()` works unchanged
2. **Fallback**: If any tree raises an exception, the pipeline catches it and fails the setup (no trade entered on error)
3. **Testing**: Add unit tests for the bridge functions with mock candle data
4. **No live trading risk**: This is paper trading ($5000 simulated balance)

---

## Execution Order

1. Create `decision_tree_bridge.py` with all builder functions + DecisionTreeEvaluator
2. Modify `schematics_5b_trader.py` — swap evaluator, remove multi-symbol, add flip detection
3. Modify `server_mexc.py` — update debug endpoint, simplify background loop
4. Update dashboard debug section HTML
5. Add/update unit tests
6. Run tests and verify
