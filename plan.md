# Schematics-5B Implementation Plan

## Overview

Build a new **schematics-5B** automated trading simulator page:
1. **Enhanced TCT logic** from Lecture 5B (structure dissection, domino effect, reconfirmation tool)
2. **Automated simulated trading** on **BTCUSDT only** using live MEXC prices
3. **Purely deterministic evaluation** — fixed score threshold (50/100), no learning, no reward system, no adaptive adjustments
4. **GitHub persistence** via `GITHUB_TOKEN_2` env var
5. **Telegram notifications** to `TELEGRAM_CHAT_ID_3` with entry/exit details

---

## Finalized Architecture Decisions

### Decision 1: Separate Trading Engine ✅ (Option A)

Dedicated `schematics_5b_trader.py` module — clean separation from tensor trader.

### Decision 2: Single Pair Only ✅ (Option B)

**BTCUSDT only.** No multi-pair scanning, no pair selector UI.

### Decision 3: No Learning / No Reward System ✅

Explicitly excluded from the 5B trader:
- ❌ No `HPBContextualReward` / no reward_history tracking
- ❌ No `compute_model_weights()` — no learned per-model score bonuses
- ❌ No `adapt_after_loss()` / `adapt_after_win()` — no consecutive-loss adaptation
- ❌ No tightening/loosening of entry thresholds based on trade history
- ✅ Fixed 50-point pass threshold — same rules on trade #1 as trade #100
- ✅ Pure W/L trade log (no reward_value, no analysis, no solution fields)

### Decision 4: Fixed Score Threshold ✅

Entry evaluation uses a fixed 50/100 minimum score based on structural factors:
- BOS confirmation (30 pts)
- R:R quality (5-25 pts based on ratio)
- HTF bias alignment (20 pts)
- Schematic quality score bonus (10-15 pts)
- Model type bonus (3-5 pts)

No dynamic adjustments — threshold stays at 50 forever.

---

## Components to Build

### 1. `schematics_5b_trader.py` — Trading Engine (~350 lines)

Modeled after `tensor_tct_trader.py` but **stripped of all learning/reward logic**.

**`Schematics5BTradeState`** — Trade state manager
- Separate log file: `schematics_5b_trade_log.json`
- Balance tracking ($5,000 starting), trade history, W/L stats
- No reward_history, no solutions_applied, no model_weights
- GitHub restore on load (uses `GITHUB_TOKEN_2`)

**`Schematics5BEvaluator`** — Deterministic schematic evaluator
- Fixed 50-point threshold (never changes)
- Same scoring factors as tensor trader: BOS + R:R + HTF alignment + quality + model type
- No consecutive_losses tracking, no adapt_after_loss/win
- 5B-specific structural checks:
  - **Tap spacing validation** — reject if tap spacing is severely uneven
  - **Horizontal range quality** — reject steep trends disguised as ranges
  - **Structure dissection scoring** — bonus points for overlapping structure (domino effect)

**`Schematics5BTrader`** — Main trading engine
- `scan_and_trade()` — BTCUSDT only, scans MTF timeframes (1h, 15m)
- HTF gate (4h bias) with caching — same pattern as tensor trader
- Position management: entry at market, SL/TP monitoring
- Uses existing `trade_execution.py` for position sizing (1% risk, 10x leverage)
- Deduplication: cooldown to prevent re-entering same setup

**Reused from tensor trader (not reimplemented):**
- `fetch_candles_sync()` — imported from `tensor_tct_trader.py`
- `fetch_live_price()` — imported from `tensor_tct_trader.py`
- `calculate_position_size/margin/liquidation` — imported from `trade_execution.py`

**Telegram notifications** — to `TELEGRAM_CHAT_ID_3`
- Entry: direction, entry price, stop, target, R:R, score
- Exit: WIN/LOSS, entry price, exit price, P&L dollars
- Uses existing `TELEGRAM_BOT_TOKEN` for the bot, different chat ID

**GitHub persistence** — `Schematics5BGitHubStorage`
- Uses `GITHUB_TOKEN_2` env var
- Separate file on data branch: `schematics_5b_trade_log.json`
- Push on trade close + hourly sync

### 2. Server Routes in `server_mexc.py` (~150 lines)

- `GET /schematics-5B` — HTML page (the UI)
- `GET /api/schematics-5b-data` — Schematic detection data + chart candles (reuses 5A detection logic)
- `GET /api/schematics-5b-trader/state` — Live trade state JSON
- `GET /api/schematics-5b-trader/scan` — Manual scan trigger
- Background loop: `schematics_5b_auto_scan_loop()` launched at startup

No toggle-pair endpoint (single pair only).

### 3. UI Page: `/schematics-5B` (~800 lines inline HTML/JS/CSS)

Based on the tensor-trade dashboard layout (simpler than 5A — no chart needed):

**Header:**
- Title: "Schematics 5B — Simulated Trading — BTCUSDT"
- Navigation links: Schematics 5A, Tensor Trade, Dashboard
- Live price display

**Controls:**
- Manual Scan button
- Refresh button
- Auto-scan status indicator (ON, every 60s)

**Stats Row:**
- Balance / Starting Balance
- P&L ($ and %)
- Win Rate
- Total Trades / Wins / Losses

**Current Trade Panel:**
- Direction, entry price, stop, target, R:R
- Live P&L % and current price
- Entry score and reasons

**Trade History Table:**
- Last 50 trades: direction, entry/exit prices, P&L, W/L, timestamp
- Color-coded green/red for wins/losses

**Debug Panel (collapsible):**
- Last scan results per timeframe
- HTF bias status
- Evaluator scores and rejection reasons

No schematic chart, no pair selector, no learning indicators.

### 4. Tests (~150 lines)

`tests/unit/test_schematics_5b_trader.py`:
- Test fixed threshold is always 50 (no adaptation)
- Test trade entry/exit mechanics (long and short)
- Test R:R validation at market price
- Test stale BOS rejection
- Test HTF bias gate (aligned, conflicting, neutral)
- Test quality score gate
- Test deduplication cooldown
- Test state save/load round-trip

---

## Key Differences from Tensor Trader

| Feature | Tensor Trader | 5B Trader |
|---------|--------------|-----------|
| Learning | ✅ HPBContextualReward | ❌ None |
| Score threshold | Dynamic (50→60→70) | Fixed 50 |
| Consecutive loss tracking | ✅ Tightens threshold | ❌ No tracking |
| Model weights | ✅ Learned from history | ❌ Fixed defaults |
| reward_history | ✅ Tracked | ❌ Not tracked |
| adapt_after_loss/win | ✅ Generates solutions | ❌ Not present |
| Pair | BTCUSDT | BTCUSDT |
| Evaluation | Same structural checks | Same structural checks |

---

## Lecture 5B Key Rules to Encode

1. **Always check highest TF range validity** — use six candle rule across timeframes
2. **Structure dissection for better entries** — when main BOS gives bad R:R (<1.5), dissect last leg on lower TF
3. **Domino effect entry** — enter on blue (micro) structure break, ride through red to black to target
4. **Reconfirmation on breaks in supply/demand** — if BOS occurs inside a supply zone, wait for retest
5. **Model 2 requirements** — price must mitigate extreme supply/demand OR grab extreme liquidity before tap 3
6. **Tap spacing** — taps should be roughly equally spaced; reject if severely uneven
7. **Horizontal range quality** — reject ranges that look like steep trends with wicks
8. **Take-the-L-to-get-the-W** — when M2 fails, scan for M1 on the same range

---

## File Changes Summary

| File | Action | Description |
|------|--------|-------------|
| `schematics_5b_trader.py` | **CREATE** | Deterministic trading engine (no learning) |
| `server_mexc.py` | **EDIT** | Add /schematics-5B routes, API endpoints, background loop |
| `tests/unit/test_schematics_5b_trader.py` | **CREATE** | Unit tests for 5B trader |

---

## Env Vars Required

| Variable | Purpose |
|----------|---------|
| `MEXC_KEY` | MEXC API key (existing) |
| `MEXC_SECRET` | MEXC API secret (existing) |
| `GITHUB_TOKEN_2` | GitHub PAT for 5B trade log persistence |
| `GITHUB_REPO` | GitHub repo (existing, shared) |
| `TELEGRAM_BOT_TOKEN` | Telegram bot token (existing, shared) |
| `TELEGRAM_CHAT_ID_3` | Telegram chat ID for 5B notifications |

---

## Execution Order

1. Create `schematics_5b_trader.py` with deterministic trading engine
2. Add API routes and background loop to `server_mexc.py`
3. Build the schematics-5B UI page in `server_mexc.py`
4. Create unit tests
5. Run tests and verify
