# HPB–TCT Trading System — Reverse Engineering Analysis

**Date:** 2026-03-18
**Scope:** Complete system reverse-engineering, execution flow mapping, TCT framework alignment, and cleanup planning.

---

## 1. FILE MAP

### Data Ingestion
| File | Description | Key Functions/Classes | Used in Execution |
|------|-------------|----------------------|-------------------|
| `phemex_feed.py` | OHLCV feed with TTL cache; fetches from MEXC API | `fetch_candles()`, `fetch_all()`, `clear_cache()` | **YES** — primary data source for Phemex trader |
| `range_scanner.py` | Async MEXC range finder with self-healing retries | `MEXCRangeScanner`, `fetch_klines()` | **NO** — test-only, not imported by any execution path |
| `risk_model.py` | Multi-TF risk model fetching OKX data (async) | `fetch_live_prices()`, `compute_risk_profile()` | **NO** — test-only, not imported by any execution path |
| `High_Probability_Model_v17/build_live_context.py` | Fetches OKX BTC candles | `okx_get_candles()`, `build_live_context()` | **NO** — module entirely unused |

### Indicators / Structure
| File | Description | Key Functions/Classes | Used in Execution |
|------|-------------|----------------------|-------------------|
| `market_structure.py` (49.8KB) | TCT Lecture 1: pivot detection (6-candle rule), BOS, CHoCH, RTZ, trend | `MarketStructure`, `evaluate_rtz()` | **YES** — Gate 1 of phemex_tct_algo |
| `tct_schematics.py` (48.2KB) | TCT Lecture 5A/5B: Model 1/2 accumulation/distribution detection | `detect_tct_schematics()`, `TCTSchematicDetector` | **YES** — Gates 3/5 + 5A/5B traders |
| `tct_model_detector.py` | v21.2 model detection (liquidity curves, tap structure) | `detect_tct_models()`, `LiquidityCurveDetector`, `TCTModelDetector` | **NO** — test-only |
| `po3_schematics.py` | PO3 (Power of Three): Range→Manipulation→Expansion | `PO3SchematicDetector`, `detect_po3_schematics()` | **YES** — served via `/api/po3-data` endpoint |
| `pivot_cache.py` | Centralized pivot engine (compute once, share) | `PivotCache` | **YES** — consumed by range engines, tct_schematics |
| `session_manipulation.py` | MSCE session detection + confidence multipliers | `get_active_session()`, `apply_session_multiplier()` | **YES** — via range_engine_controller and tct_schematics |

### Range Detection
| File | Description | Key Functions/Classes | Used in Execution |
|------|-------------|----------------------|-------------------|
| `range_engine_controller.py` | Feature flag controller for L1/L2 dual engine | `RangeEngineController` | **YES** — called by tct_schematics |
| `range_engine_l1.py` | Legacy L1 range engine (swing pivot-based) | `RangeEngineL1` | **YES** — via controller |
| `range_engine_l2.py` | L2 counter-structure range engine (TCT-correct) | `RangeEngineL2` | **YES** — via controller |
| `range_utils.py` | Shared equilibrium-touch confirmation | `check_equilibrium_touch()` | **YES** — L1, L2, tct_schematics, server |
| `range_comparison_logger.py` | JSONL logger for L1 vs L2 structural diffs | `RangeComparisonLogger` | **YES** — via controller in "compare" mode |

### Strategy Logic
| File | Description | Key Functions/Classes | Used in Execution |
|------|-------------|----------------------|-------------------|
| `phemex_tct_algo.py` | 6-Gate TCT signal pipeline (core decision engine) | `run_pipeline()`, `_gate_1` through `_gate_6`, `PipelineResult` | **YES** — core pipeline |
| `tct_pdf_rules.py` | ChromaDB RAG rule extraction from lecture PDFs | `load_tct_rules()`, `TCTRuleSet` | **YES** — rules loaded at startup |
| `decision_tree_bridge.py` | Bridge between raw candles and decision trees | `_detect_trend()`, `_six_candle_rule()` | **YES** — via jack_tct_evaluator |
| `jack_tct_evaluator.py` | 5-tree scoring pipeline (50pt threshold, 3 hard gates) | 5 tree evaluators | **YES** — used by schematics_5b_trader |
| `hpb_rig_validator.py` | Range Integrity Gate (blocks counter-bias trades) | `range_integrity_validator()` | **NO** — test-only, NOT wired into any execution path |
| `High_Probability_Model_v17/validate_gates.py` | HPB 1A–1D gate validation (**PLACEHOLDER with random.uniform**) | `validate_gates()` | **NO** — module entirely unused |

### Execution / Orders
| File | Description | Key Functions/Classes | Used in Execution |
|------|-------------|----------------------|-------------------|
| `trade_execution.py` | Position sizing, leverage, liquidation, partial TPs | `generate_execution_plan()`, `calculate_position_size()`, `find_max_safe_leverage()` | **YES** — schematics_5b_trader, server |
| `phemex_tct_trader.py` | Phemex paper trading engine (singleton) | `PhemexTCTTrader`, `scan()`, `_open_trade()`, `_close_trade()` | **YES** — entry point |
| `5A_tct_trader.py` | TCT 5A simulated trader (fixed threshold 50) | Paper trading engine | **YES** — via tensor_tct_trader shim |
| `schematics_5b_trader.py` | 5B simulated trader (threshold 60, cascade LTF) | Paper trading engine | **YES** — via server_mexc |
| `tensor_tct_trader.py` | Shim: re-exports 5A_tct_trader (Python naming workaround) | `sys.modules` replacement | **YES** — import bridge |

### Config
| File | Description | Key Functions/Classes | Used in Execution |
|------|-------------|----------------------|-------------------|
| `config/risk_profile.json` | Risk profile (shadow mode, 1% risk, $5k max) | N/A | **UNCERTAIN** — exists but no code reads it |
| `bias_memory.json` | Last bias state (neutral) | N/A | **UNCERTAIN** — may be read by 5A trader |
| `render.yaml` | Render deployment config (2 services) | N/A | **YES** — defines startup commands |
| `High_Probability_Model_v17_RIG.config` | RIG config (text/pseudo-code, not parsed) | N/A | **NO** — documentation only |
| `trading_bot_logic_map.json` | Logic map documentation (v1.2) | N/A | **UNCERTAIN** — may be read at runtime |

### Utilities
| File | Description | Key Functions/Classes | Used in Execution |
|------|-------------|----------------------|-------------------|
| `github_storage.py` | GitHub-based trade log persistence | `fetch_trade_log()`, `push_trade_log()` | **YES** — 5A + Phemex traders |
| `telegram_notifications.py` | Fire-and-forget Telegram alerts | `send_message()`, `notify_trade_entered()` | **YES** — Phemex trader |
| `telegram_bot.py` | Telegram bot command handler (standalone worker) | `main()`, command handlers | **YES** — standalone entry point |

### Server
| File | Description | Key Functions/Classes | Used in Execution |
|------|-------------|----------------------|-------------------|
| `server_mexc.py` (18,354 lines) | FastAPI server: 43+ endpoints, all dashboards | `app`, all `/api/*` routes | **YES** — primary entry point |

### Decision Trees (all under `decision_trees/`)
| File | Used in Execution |
|------|-------------------|
| `market_structure_engine.py` | **YES** — via range_engine_l2 |
| `ranges_decision_tree.py` | **YES** — via decision_tree_bridge |
| `supply_demand_decision_tree.py` | **YES** — via decision_tree_bridge |
| `liquidity_decision_tree.py` | **YES** — via decision_tree_bridge |
| `tct_5a_schematics_decision_tree.py` | **YES** — via decision_tree_bridge |
| `tct_5b_schematics_real_examples_decision_tree.py` | **YES** — via decision_tree_bridge |
| `tct_6_advanced_schematics_decision_tree.py` | **YES** — via decision_tree_bridge |

---

## 2. EXECUTION FLOW

### Entry Point: `server_mexc.py` → `uvicorn server_mexc:app`

There are **three parallel trading engines**, all served through `server_mexc.py`:

### Pipeline A: Phemex TCT Trader (6-Gate Pipeline)

```
Step 1 → server_mexc.py      → GET /api/phemex-tct/scan      → triggers scan
Step 2 → phemex_tct_trader.py → PhemexTCTTrader.scan()        → orchestrates full cycle
Step 3 → phemex_tct_trader.py → _ensure_rules()               → lazy-load TCT rules (once)
Step 4 → tct_pdf_rules.py     → load_tct_rules()              → ChromaDB RAG extraction from PDFs
Step 5 → phemex_feed.py        → fetch_all()                   → fetch LTF(15m)/MTF(1h)/HTF(4h) candles from MEXC
Step 6 → phemex_tct_trader.py → _check_position()             → check SL/TP on existing position
Step 7 → phemex_tct_algo.py   → run_pipeline(htf, mtf, ltf)   → execute 6-gate pipeline
  Step 7a → _gate_1_market_structure(ltf)    → trend + BOS + RTZ via MarketStructure
  Step 7b → _gate_2_ranges(htf)              → consolidation range on HTF
  Step 7c → _gate_3_supply_demand(mtf)       → S&D zone detection (OB/FVG)
  Step 7d → _gate_4_liquidity(ltf)           → liquidity tap confirmation
  Step 7e → _gate_5_schematics(ltf)          → TCT Model 1/2 detection
  Step 7f → _gate_6_advanced_tct()           → confluence + R:R ≥ 2.0 + confidence ≥ 0.65
Step 8 → phemex_tct_trader.py → _open_trade(result)           → open simulated position
Step 9 → telegram_notifications.py → notify_trade_entered()   → fire-and-forget alert
Step 10 → github_storage.py   → push_phemex_log()             → persist state to GitHub
```

### Pipeline B: Schematics 5B Trader (9-Phase + 5-Tree)

```
Step 1 → server_mexc.py          → GET /api/schematics-5b-trader/scan → triggers scan
Step 2 → schematics_5b_trader.py → scan()                             → orchestrate
Step 3 → (MEXC API via requests)  → fetch candles                      → HTF(1d)/MTF(4h,1h,30m)/LTF(5m,1m)
Step 4 → tct_schematics.py       → detect_tct_schematics()            → schematic detection
  Step 4a → pivot_cache.py       → PivotCache                         → compute pivots once
  Step 4b → range_engine_controller.py → detect_ranges()              → L1/L2 range detection
  Step 4c → session_manipulation.py → apply_session_multiplier()      → MSCE adjustment
Step 5 → jack_tct_evaluator.py   → evaluate (5 trees)                 → 100pt scoring
  Step 5a → decision_tree_bridge.py → route to 5 decision trees       → MS/Range/SD/Liq/TCT5A
Step 6 → trade_execution.py      → generate_execution_plan()          → position sizing + leverage
Step 7 → schematics_5b_trader.py → open trade if score ≥ 60           → simulated entry
Step 8 → telegram_notifications  → notify                             → alert
Step 9 → github_storage.py       → push                               → persist
```

### Pipeline C: 5A TCT Trader (Simpler Pipeline)

```
Step 1 → server_mexc.py         → GET /api/tensor-trade/scan  → triggers scan
Step 2 → tensor_tct_trader.py   → (shim to 5A_tct_trader.py)  → redirect
Step 3 → 5A_tct_trader.py       → scan()                      → fetch + detect
Step 4 → tct_schematics.py      → detect_tct_schematics()     → schematic detection
Step 5 → 5A_tct_trader.py       → qualify trade (threshold 50) → entry decision
Step 6 → github_storage.py      → push_trade_log()            → persist
```

---

## 3. TRADE ANALYSIS

### PART A — Real Trade Reconstruction

**Status: NO TRADE LOGS AVAILABLE**

No `phemex_trade_log.json` or `tensor_trade_log.json` files exist in the repository. These are stored on Render's persistent storage (`/opt/render/project/chroma_db`) and synced to the GitHub `data` branch. Without access to either, real trade reconstruction is **not possible from this codebase alone**.

**To reconstruct a real trade, you would need to:**
1. Pull logs from GitHub `data` branch: `git fetch origin data && git checkout origin/data -- phemex_trade_log.json`
2. Or SSH into the Render instance and read from `/opt/render/project/chroma_db/`

**What the logs would contain** (based on code in `phemex_tct_trader.py`):
- Entry price, stop loss, target
- Direction (LONG/SHORT)
- Gate results (6 gates pass/fail + data)
- Confidence score
- R:R ratio
- Outcome (WIN/LOSS/OPEN)
- Balance before/after

### PART B — Simulated Trade Flow (Phemex Pipeline)

Walking through `run_pipeline()` in `phemex_tct_algo.py` with hypothetical data:

```
1. FETCH: phemex_feed.fetch_all()
   → HTF: 100 x 4h candles from MEXC
   → MTF: 150 x 1h candles
   → LTF: 200 x 15m candles

2. GATE 1 — Market Structure (LTF 15m):
   → MarketStructure(ltf).analyze()
   → Checks: trend direction, confirmed BOS (body-close), RTZ quality
   → PASS if: trend confirmed AND BOS exists AND RTZ quality ≥ threshold
   → FAIL → pipeline short-circuits, returns no signal

3. GATE 2 — Ranges (HTF 4h):
   → _detect_range(htf): lookback=50, max_size=5%, min_touches=2/side
   → Checks: valid range with equilibrium touch
   → PASS if: range detected with ≥ 2 touches each side
   → FAIL → short-circuit

4. GATE 3 — Supply & Demand (MTF 1h):
   → _find_demand_zone(mtf) or _find_supply_zone(mtf) based on bias
   → Checks: impulse candle (3-candle pattern) forming OB
   → PASS if: aligned zone found in direction of bias

5. GATE 4 — Liquidity (LTF 15m):
   → Checks: price tapped into zone (within 0.3% tolerance)
   → PASS if: liquidity interaction confirmed

6. GATE 5 — Schematics (LTF 15m):
   → detect_tct_schematics(ltf): Model 1 or Model 2 pattern
   → Checks: quality ≥ 0.6
   → PASS if: valid schematic with sufficient quality

7. GATE 6 — Advanced TCT (Final):
   → _compute_confidence(): weighted score from all gates
   → Checks: R:R ≥ 2.0, confidence ≥ 0.65
   → PASS if: both thresholds met
   → OUTPUT: PipelineResult with entry, stop, target, direction, confidence
```

### PART C — Real vs Simulated Comparison

**Critical Discrepancies Identified:**

| Aspect | Phemex Pipeline (A) | 5B Pipeline (B) | 5A Pipeline (C) |
|--------|---------------------|------------------|------------------|
| **Entry threshold** | R:R ≥ 2.0 + confidence ≥ 0.65 | Score ≥ 60 (100pt scale) | Score ≥ 50 (fixed) |
| **Timeframes** | 4h/1h/15m | 1d/4h/1h/30m/5m/1m | 1d/4h/1h/30m |
| **RIG validation** | **NOT IMPLEMENTED** | **NOT IMPLEMENTED** | **NOT IMPLEMENTED** |
| **MSCE session** | **NOT IMPLEMENTED** in pipeline | Applied via range_engine_controller | **NOT IMPLEMENTED** |
| **Decision trees** | Not used (inline gates) | 5-tree scoring | Not used |
| **Range engine** | Inline `_detect_range()` | Via tct_schematics → controller → L1/L2 | Via tct_schematics |
| **Rules source** | ChromaDB PDF extraction | Hardcoded thresholds | Hardcoded thresholds |
| **BOS cascade** | Single TF (15m) | Cascade 5m → 1m refinement | Single TF |

**Key inconsistencies:**
1. **Phemex pipeline uses its own inline range detection** (`_detect_range()` in `phemex_tct_algo.py`) that is DIFFERENT from the L1/L2 engine used by 5B. The Phemex range detection is simpler and may produce different results.
2. **RIG is completely unwired** — `hpb_rig_validator.py` exists with full logic but is imported by ZERO execution paths. Counter-bias trades are NOT blocked.
3. **Session multipliers differ** — `session_manipulation.py` uses Asia=1.05, London=1.10, NY=1.20, but `trading_bot_logic_map.json` documents Asia=0.85, London=1.10, NY=1.15. These are **contradictory**.
4. **`config/risk_profile.json` appears orphaned** — no code reads it; risk parameters are hardcoded in each trader.
5. **The 5A trader has a LOWER entry threshold (50 vs 60)** which means it takes MORE trades than 5B.

---

## 4. DECISION POINTS

| # | File | Function | Purpose |
|---|------|----------|---------|
| 1 | `phemex_tct_algo.py` | `_gate_1_market_structure()` | Confirm trend + BOS + RTZ on LTF |
| 2 | `phemex_tct_algo.py` | `_gate_2_ranges()` | Validate HTF range exists |
| 3 | `phemex_tct_algo.py` | `_gate_3_supply_demand()` | S&D zone aligned with bias |
| 4 | `phemex_tct_algo.py` | `_gate_4_liquidity()` | Liquidity tap in zone confirmed |
| 5 | `phemex_tct_algo.py` | `_gate_5_schematics()` | TCT Model 1/2 quality ≥ 0.6 |
| 6 | `phemex_tct_algo.py` | `_gate_6_advanced_tct()` | R:R ≥ 2.0 AND confidence ≥ 0.65 |
| 7 | `phemex_tct_trader.py` | `_check_position()` | SL/TP hit check on open position |
| 8 | `phemex_tct_trader.py` | `_open_trade()` | Position sizing + trade entry |
| 9 | `jack_tct_evaluator.py` | Tree 1 (MS) | MSH/MSL presence (hard gate) |
| 10 | `jack_tct_evaluator.py` | Tree 3 (S&D) | OB/FVG near tap zones (hard gate) |
| 11 | `jack_tct_evaluator.py` | Tree 5 (TCT 5A) | BOS + model validity (hard gate) |
| 12 | `jack_tct_evaluator.py` | Threshold | Combined score ≥ 50 |
| 13 | `schematics_5b_trader.py` | Entry logic | Score ≥ 60, R:R ≥ 1.5, no duplicate |
| 14 | `5A_tct_trader.py` | Entry logic | Score ≥ 50, quality ≥ 0.70, R:R ≥ 1.5 |
| 15 | `tct_schematics.py` | Schematic detection | Model 1/2 tap structure + BOS confirm |
| 16 | `range_engine_controller.py` | `detect_ranges()` | L1 vs L2 engine selection |
| 17 | `range_engine_l2.py` | Min duration | 24-hour minimum range filter |
| 18 | `session_manipulation.py` | `apply_session_multiplier()` | Session-based confidence adjustment |
| 19 | `market_structure.py` | BOS classification | Good vs Bad BOS (distance-based) |
| 20 | `market_structure.py` | CHoCH detection | Domino-effect trend reversal |
| 21 | `decision_tree_bridge.py` | R:R filter | R:R ≥ 1.5 (phase 8) |
| 22 | `decision_tree_bridge.py` | Confidence threshold | ≥ 60 (phase 9) |

---

## 5. TCT MAPPING

### 1A — BTC Structure / Bias

| Status | File | Function | Notes |
|--------|------|----------|-------|
| **PARTIALLY IMPLEMENTED** | `market_structure.py` | `MarketStructure.analyze()` | Computes trend, BOS, CHoCH on the candle data passed in — but this operates on the TRADING PAIR, not on BTC specifically as a macro bias anchor |
| **NOT IMPLEMENTED (placeholder)** | `High_Probability_Model_v17/validate_gates.py` | `validate_gates()` | Has 1A gate structure but uses `random.uniform()` — **FAKE** |
| **MISSING** | — | — | No dedicated BTC macro structure analysis that gates altcoin execution |

### 1B — USDT.D / Correlation Logic

| Status | File | Function | Notes |
|--------|------|----------|-------|
| **NOT IMPLEMENTED** | — | — | No USDT.D feed, no inverse correlation logic anywhere in the live codebase |
| **PLACEHOLDER ONLY** | `High_Probability_Model_v17/validate_gates.py` | Gate "1B" | `random.uniform(0.8, 0.92)` — not real |

### 1C — Altcoin Alignment

| Status | File | Function | Notes |
|--------|------|----------|-------|
| **NOT IMPLEMENTED** | — | — | No altcoin alignment check exists in any execution path |
| **DOCUMENTED** | `trading_bot_logic_map.json` | — | References min 24h volume ($10M), spread max (0.15%) — but no code implements this |
| **PLACEHOLDER ONLY** | `High_Probability_Model_v17/validate_gates.py` | Gate "1C" | Fake random score |

### RCM — Range Detection

| Status | File | Function | Notes |
|--------|------|----------|-------|
| **IMPLEMENTED** | `range_engine_controller.py` | `detect_ranges()` | Dual L1/L2 with feature flags |
| **IMPLEMENTED** | `range_engine_l1.py` | `RangeEngineL1` | Legacy swing-pivot range detection |
| **IMPLEMENTED** | `range_engine_l2.py` | `RangeEngineL2` | TCT-correct counter-structure pools |
| **IMPLEMENTED** | `range_utils.py` | `check_equilibrium_touch()` | Equilibrium confirmation |
| **ALSO** | `phemex_tct_algo.py` | `_detect_range()` | **SEPARATE inline implementation** (not using L1/L2 engines) |

### RIG — Counter-Bias Filter (Range Integrity Gate)

| Status | File | Function | Notes |
|--------|------|----------|-------|
| **IMPLEMENTED BUT NOT WIRED** | `hpb_rig_validator.py` | `range_integrity_validator()` | Full logic exists: checks session bias vs HTF bias with 24h min duration + 25% displacement threshold |
| **NOT CONNECTED** | — | — | Zero imports from any execution path. Only imported by `tests/unit/test_hpb_rig_validator.py` |
| **CRITICAL GAP** | — | — | Counter-bias trades are NOT being blocked |

### MSCE — Session Logic

| Status | File | Function | Notes |
|--------|------|----------|-------|
| **IMPLEMENTED** | `session_manipulation.py` | `get_active_session()`, `apply_session_multiplier()` | Asia/London/NY session detection with confidence multipliers |
| **PARTIALLY WIRED** | `range_engine_controller.py` | — | Session used in range engine controller |
| **PARTIALLY WIRED** | `tct_schematics.py` | — | Session multiplier applied to schematic quality |
| **NOT WIRED** | `phemex_tct_algo.py` | — | The Phemex 6-gate pipeline does NOT use session logic |
| **INCONSISTENT** | — | — | Code multipliers (Asia=1.05, London=1.10, NY=1.20) ≠ documented multipliers (Asia=0.85, London=1.10, NY=1.15) |

### 1D — Execution (BOS, POI, Triggers)

| Status | File | Function | Notes |
|--------|------|----------|-------|
| **IMPLEMENTED** | `trade_execution.py` | `generate_execution_plan()` | Position sizing, leverage, liquidation, partial TPs |
| **IMPLEMENTED** | `phemex_tct_algo.py` | `_gate_6_advanced_tct()` | Final confluence check (R:R + confidence) |
| **IMPLEMENTED** | `market_structure.py` | BOS detection | Body-close BOS with Good/Bad classification |
| **IMPLEMENTED** | `phemex_tct_algo.py` | `_gate_3_supply_demand()` | POI (OB/FVG) detection |

---

## 6. FILE USAGE CLASSIFICATION

### ACTIVE (clearly used in execution)
| File | Evidence |
|------|----------|
| `server_mexc.py` | Entry point: `uvicorn server_mexc:app` |
| `phemex_tct_trader.py` | Imported by server_mexc (lazy) |
| `phemex_tct_algo.py` | Imported by phemex_tct_trader |
| `phemex_feed.py` | Imported by phemex_tct_algo, phemex_tct_trader |
| `market_structure.py` | Imported by phemex_tct_algo, jack_tct_evaluator, server_mexc |
| `tct_schematics.py` | Imported by phemex_tct_algo, 5A_tct_trader, schematics_5b_trader, server_mexc |
| `tct_pdf_rules.py` | Imported by phemex_tct_algo, phemex_tct_trader |
| `trade_execution.py` | Imported by schematics_5b_trader, server_mexc |
| `pivot_cache.py` | Imported by range engines, tct_schematics, schematics_5b_trader |
| `range_engine_controller.py` | Imported by tct_schematics |
| `range_engine_l1.py` | Imported by controller |
| `range_engine_l2.py` | Imported by controller |
| `range_utils.py` | Imported by L1, L2, tct_schematics, server_mexc |
| `range_comparison_logger.py` | Imported by controller |
| `session_manipulation.py` | Imported by controller, tct_schematics |
| `po3_schematics.py` | Imported by server_mexc |
| `decision_tree_bridge.py` | Imported by jack_tct_evaluator, schematics_5b_trader |
| `jack_tct_evaluator.py` | Imported by schematics_5b_trader |
| `5A_tct_trader.py` | Imported via tensor_tct_trader shim |
| `schematics_5b_trader.py` | Imported by server_mexc (lazy) |
| `tensor_tct_trader.py` | Imported by schematics_5b_trader |
| `telegram_notifications.py` | Imported by phemex_tct_trader |
| `telegram_bot.py` | Standalone entry point (render.yaml worker) |
| `github_storage.py` | Imported by 5A_tct_trader, phemex_tct_trader |
| All 7 decision tree files | Imported by decision_tree_bridge, range_engine_l2 |

### UNUSED (no references found)
| File | Evidence | Risk Level |
|------|----------|------------|
| `High_Probability_Model_v17/__init__.py` | No imports from any file | **LOW** — safe to remove |
| `High_Probability_Model_v17/build_live_context.py` | No imports from any file | **LOW** — safe to remove |
| `High_Probability_Model_v17/validate_gates.py` | No imports; uses `random.uniform()` (placeholder) | **LOW** — safe to remove |
| `High_Probability_Model_v17_RIG.config` | Text pseudo-code, not parsed by any code | **LOW** — documentation artifact |
| `hpb_rig_validator.py` | Only imported by test file | **MEDIUM** — contains real RIG logic that SHOULD be wired in |
| `tct_model_detector.py` | Only imported by test file | **MEDIUM** — alternative model detection, may be future use |
| `risk_model.py` | Only imported by test file; fetches OKX (not MEXC) | **LOW** — different exchange, superseded |
| `range_scanner.py` | Only imported by test file | **MEDIUM** — async scanner, may be intended for multi-pair |

### UNCERTAIN
| File | Status | Notes |
|------|--------|-------|
| `config/risk_profile.json` | No code reads this file | May have been intended for future use; risk params are hardcoded |
| `bias_memory.json` | May be read by 5A trader at runtime | Would need runtime trace to confirm |
| `trading_bot_logic_map.json` | Documentation; may be read by server_mexc | Contains documented constants that don't match code |
| `mexc_all_pairs.txt` | May be read by server_mexc `/api/coin-list` | Would need to check server_mexc endpoint |

---

## 7. CLEANUP PLAN

**DO NOT DELETE ANYTHING. This is a staged plan only.**

### Phase 1 — Zero Risk (documentation/placeholder artifacts)
1. `High_Probability_Model_v17/` (entire directory)
   - **Evidence:** Zero imports, `validate_gates()` uses `random.uniform()` — fake
   - **Action:** `mv High_Probability_Model_v17/ _archive/High_Probability_Model_v17/`
   - **Test:** Run full test suite; verify all endpoints respond

2. `High_Probability_Model_v17_RIG.config`
   - **Evidence:** Text pseudo-code, not parsed
   - **Action:** Archive alongside above

### Phase 2 — Low Risk (superseded modules)
3. `risk_model.py`
   - **Evidence:** Fetches OKX data (system uses MEXC), only test imports
   - **Action:** Archive, keep test for reference
   - **Test:** Run tests excluding `test_risk_model.py`; verify no import errors

### Phase 3 — Medium Risk (potentially useful but currently unwired)
4. `range_scanner.py`
   - **Evidence:** Async multi-pair scanner, test-only imports
   - **Risk:** May be needed for multi-pair expansion
   - **Action:** Archive with note about intended purpose
   - **Test:** Run test suite

5. `tct_model_detector.py`
   - **Evidence:** Alternative detection engine, test-only
   - **Risk:** May overlap with or improve `tct_schematics.py`
   - **Action:** Archive with comparison notes

### Phase 4 — DO NOT REMOVE (needs wiring, not cleanup)
6. `hpb_rig_validator.py` — **DO NOT REMOVE**
   - This is a **MISSING GATE**, not dead code
   - **Action:** Wire into execution pipelines (see Issues section)

### Backup Strategy
```bash
# Before any removal:
git checkout -b archive/pre-cleanup-$(date +%Y%m%d)
git add -A && git commit -m "Pre-cleanup snapshot"
git push origin archive/pre-cleanup-$(date +%Y%m%d)

# Archive rather than delete:
mkdir -p _archive
mv <file> _archive/
git add -A && git commit -m "Archive <file>: <reason>"

# After each phase, run:
python -m pytest tests/ -v
# Verify server starts:
uvicorn server_mexc:app --host 0.0.0.0 --port 8000 &
curl http://localhost:8000/status
# Test each trader endpoint:
curl http://localhost:8000/api/phemex-tct/state
curl http://localhost:8000/api/schematics-5b-trader/state
curl http://localhost:8000/api/tensor-trade/state
```

---

## 8. ISSUES FOUND

### ISSUE 1 — RIG Gate Not Wired (CRITICAL)

**Problem:** `hpb_rig_validator.py` contains a complete Range Integrity Gate that blocks counter-bias trades when the HTF range is still intact. This is a **core TCT safety gate** but it is imported by ZERO execution paths. The bot can take counter-bias trades that should be blocked.

**Files:** `hpb_rig_validator.py:range_integrity_validator()`
**Impact:** Incorrect trades may be taken against the dominant range direction.

**Options:**
- **A (Recommended):** Wire RIG into all three pipelines (phemex_tct_algo Gate 6, jack_tct_evaluator as hard gate, 5A qualifying logic). Medium effort, high impact.
- **B:** Wire into Phemex pipeline only (it's the most mature). Low effort, partial coverage.
- **C:** Do nothing. Zero effort, ongoing risk of counter-bias trades.

### ISSUE 2 — Dual Range Detection (Inconsistency)

**Problem:** The Phemex pipeline (`phemex_tct_algo._detect_range()`) uses a completely DIFFERENT range detection algorithm than the L1/L2 engine system used by the 5B trader via `tct_schematics.py`. This means:
- Same market data → different range conclusions
- Phemex may see a range where 5B doesn't (or vice versa)

**Files:** `phemex_tct_algo.py:_detect_range()` vs `range_engine_controller.py` + `range_engine_l1.py` + `range_engine_l2.py`

**Options:**
- **A (Recommended):** Refactor Phemex pipeline to use `range_engine_controller.py`. DRY, consistent behavior across all traders.
- **B:** Keep separate but document why they differ.
- **C:** Do nothing.

### ISSUE 3 — Session Multiplier Contradiction

**Problem:** Code in `session_manipulation.py` uses multipliers that contradict the documented values in `trading_bot_logic_map.json`:

| Session | Code (`session_manipulation.py`) | Documented (`trading_bot_logic_map.json`) |
|---------|----------------------------------|------------------------------------------|
| Asia | 1.05 (BOOST) | 0.85 (PENALTY) |
| London | 1.10 | 1.10 |
| NY | 1.20 | 1.15 |

Asia session boosting confidence is **opposite** to the documented intent of penalizing lower-volume sessions.

**Files:** `session_manipulation.py:get_session_multiplier()`, `trading_bot_logic_map.json`

**Options:**
- **A (Recommended):** Align code to documented values (Asia=0.85, NY=1.15). The documentation appears to reflect the trading methodology more accurately.
- **B:** Update documentation to match code.
- **C:** Do nothing.

### ISSUE 4 — HPB Gates 1A/1B/1C Missing (Structural Gap)

**Problem:** The full HPB framework requires:
- **1A:** BTC macro structure as directional anchor
- **1B:** USDT.D inverse correlation confirmation
- **1C:** Altcoin alignment check

None of these exist in the live execution paths. The `High_Probability_Model_v17` module has placeholder logic using `random.uniform()` that is never called. The system currently trades based on the instrument's own structure only, without the HPB directional context layer.

**Files:** `High_Probability_Model_v17/validate_gates.py` (placeholder), no live implementation
**Impact:** Trades may be taken against the macro bias.

**Options:**
- **A (Recommended):** Implement 1A first (BTC structure as macro bias). Wire it as a pre-gate in all pipelines. 1B and 1C can follow.
- **B:** Implement all three as a standalone module, then wire in.
- **C:** Do nothing — accept that HPB layer is missing.

### ISSUE 5 — `config/risk_profile.json` Orphaned

**Problem:** Risk parameters are defined in `config/risk_profile.json` (risk=1%, max_exposure=$5k, mode=shadow) but each trader hardcodes its own values:
- `phemex_tct_trader.py`: `RISK_PER_TRADE_PCT = 1.0` (env var)
- `5A_tct_trader.py`: `RISK_PER_TRADE_PCT = 1.0`
- `schematics_5b_trader.py`: hardcoded in class

This means changing `risk_profile.json` has **zero effect**.

**Options:**
- **A (Recommended):** Make all traders read from `config/risk_profile.json` at startup.
- **B:** Delete the config file (it's misleading).
- **C:** Do nothing.

### ISSUE 6 — 18,354-Line Monolith (`server_mexc.py`)

**Problem:** `server_mexc.py` is an 18,354-line monolith containing:
- 43+ API endpoints
- Inline HTML for 10+ dashboard pages
- Candle fetching logic (duplicated from `phemex_feed.py`)
- ChromaDB initialization
- PDF processing
- All three trader integrations

This makes the file extremely difficult to maintain, debug, or test.

**Options:**
- **A:** Extract HTML into templates, routes into blueprints. High effort but proper solution.
- **B (Recommended):** Extract just the trader endpoints into separate router files (`routers/phemex.py`, `routers/schematics_5b.py`, etc.). Medium effort, high readability gain.
- **C:** Do nothing.

### ISSUE 7 — No Live Trade Logs in Repository

**Problem:** Trade logs are stored on Render's persistent storage and synced to the GitHub `data` branch, but are not accessible for offline analysis. This makes it impossible to:
- Review historical trade decisions
- Backtest changes against real data
- Validate that changes don't degrade performance

**Options:**
- **A (Recommended):** Add a `/api/export-trades` endpoint that returns the current trade log as JSON.
- **B:** Set up a cron job to sync logs to the `data` branch daily.
- **C:** Do nothing.

### ISSUE 8 — Phemex Pipeline Skips MSCE Session Logic

**Problem:** The 6-gate pipeline in `phemex_tct_algo.py` does NOT call `session_manipulation.py` at any point. Session timing has zero influence on Phemex trade decisions. Meanwhile, the 5B pipeline applies session multipliers through `tct_schematics.py`. This is an execution consistency gap.

**Files:** `phemex_tct_algo.py` (no session import), `session_manipulation.py` (unused by Phemex)

**Options:**
- **A (Recommended):** Add session multiplier to Gate 6 confidence calculation.
- **B:** Do nothing — accept Phemex pipeline is session-agnostic.

---

*End of Analysis*
