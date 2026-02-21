# Schematics-5B Implementation Plan

## Overview

Build a new **schematics-5B** page that extends the existing schematics-5A page with:
1. **Enhanced TCT logic** from Lecture 5B (real-world trade examples, structure dissection, domino effect, reconfirmation tool)
2. **Automated simulated trading** using live MEXC prices with minimal lag
3. **GitHub persistence** via `GITHUB_TOKEN_2` env var
4. **Telegram notifications** to `TELEGRAM_CHAT_ID_3` with entry/exit details

---

## Architecture Decisions

### Decision 1: Separate vs. Shared Trading Engine

**Option A (Recommended): Dedicated `schematics_5b_trader.py` module**
- New standalone trader module modeled after `tensor_tct_trader.py`
- Separate trade log file (`schematics_5b_trade_log.json`)
- Uses `GITHUB_TOKEN_2` for persistence (separate from tensor trader's `GITHUB_TOKEN`)
- Sends Telegram to `TELEGRAM_CHAT_ID_3` (separate channel)
- Independent background loop in `server_mexc.py`
- Effort: Medium
- Risk: Low — no interference with existing tensor trader
- Maintenance: Self-contained, easy to reason about

**Option B: Extend `tensor_tct_trader.py` with a 5B mode**
- Add 5B-specific logic branches to existing TensorTCTTrader
- Risk: High — coupling two trading strategies in one engine makes debugging harder
- Could break the existing working tensor trader

**Recommendation: Option A** — clean separation, no risk to existing system.

### Decision 2: Scan Multi-Pair vs. Single Pair

**Option A (Recommended): Multi-pair scanning (configurable)**
- Allow user to select which pairs to auto-trade from the UI
- Default to a predefined list (majors: BTC, ETH, SOL, etc.)
- Background loop scans all enabled pairs every cycle

**Option B: Single pair only (like current tensor trader)**
- Simpler but less useful

**Recommendation: Option A** — the 5B lecture covers SOL, PEPE, Gold, EUR/USD showing the methodology works on many instruments. Multi-pair gives more trade opportunities.

### Decision 3: Entry Refinement Strategy (Key 5B Enhancement)

The biggest improvement from Lecture 5B is the **overlapping structure dissection** technique for better R:R entries. Instead of waiting for the main market structure break (which often gives bad R:R like 0.4-0.67:1), the lecture teaches:

1. Identify main structure break level (red)
2. Dissect last leg into lower-TF structure (blue)
3. Enter on blue structure break (gets entries like 3:1 to 6:1 instead of 0.4:1)
4. The "domino effect": break blue → red → black → target

**Implementation:** Add a multi-layer structure analysis that:
- Uses the existing TCT detector for macro structure
- Adds a second-pass lower-TF analysis for micro structure within the last leg
- Scores entries by R:R improvement from dissection

---

## Components to Build

### 1. `schematics_5b_trader.py` — Trading Engine (~400 lines)

New module containing:

- **`Schematics5BTradeState`** — Trade state manager (modeled after `TradeState`)
  - Separate log file: `schematics_5b_trade_log.json`
  - Balance tracking, trade history, P&L

- **`Schematics5BTrader`** — Main trading engine
  - `scan_and_trade()` loop scanning configurable pairs across timeframes
  - Enhanced TCT evaluation with 5B rules:
    - **Six candle rule validation** for highest TF range validity
    - **Structure dissection** for better entries (overlapping structure)
    - **Reconfirmation tool** — reject entries in supply/demand zones, wait for retest
    - **Tap spacing check** — taps should be roughly equally spaced (5B rule)
    - **Horizontal range quality** — reject ranges that don't look like "horizontal price action"
    - **Model 1 from Model 2 failure** — when M2 fails, look for M1 (the "take the L to get the W" pattern)
  - Position management: entry, SL, TP monitoring with live MEXC prices
  - Uses existing `trade_execution.py` for position sizing

- **`Schematics5BGitHubStorage`** — GitHub persistence
  - Uses `GITHUB_TOKEN_2` env var (NOT the existing `GITHUB_TOKEN`)
  - Separate file on data branch: `schematics_5b_trade_log.json`
  - Push on trade close + hourly sync

- **Telegram notifications** — to `TELEGRAM_CHAT_ID_3`
  - Entry notification: entry price, stop loss, target
  - Exit notification: WIN/LOSS, entry price, exit price, amount won/lost
  - Uses existing `TELEGRAM_BOT_TOKEN` for the bot, just different chat ID

### 2. Server Routes in `server_mexc.py` (~200 lines)

- `GET /schematics-5B` — HTML page (the UI)
- `GET /api/schematics-5b-data` — Schematic detection data + chart candles (like 5A)
- `GET /api/schematics-5b-trader/state` — Live trade state JSON
- `GET /api/schematics-5b-trader/scan` — Manual scan trigger
- `GET /api/schematics-5b-trader/toggle-pair` — Enable/disable pairs for auto-trade
- Background loop: `schematics_5b_auto_scan_loop()` launched at startup

### 3. UI Page: `/schematics-5B` (~800 lines inline HTML/JS/CSS)

Based on the schematics-5A layout with these changes (from screenshot analysis):

**Same as 5A:**
- Left: LightweightCharts candlestick chart with schematic overlays
- Right: Panel with schematic cards (Active/Forming/Completed)
- Top: Header with pair selector, timeframe selector, scan button
- Legend bar at bottom of chart

**New in 5B (from screenshot):**
- **Auto-Trade toggle** in header — ON/OFF switch to enable/disable automated trading
- **Trade Status panel** — shows current open trade (if any) with live P&L
- **Trade History section** — recent closed trades with W/L results
- **Balance display** — current simulated balance in header
- **Multi-pair selector** — checkboxes to enable/disable which pairs are auto-traded
- **Enhanced schematic cards** — show 5B-specific metrics:
  - Structure dissection depth (how many layers of overlapping structure found)
  - Domino chain status (blue → red → black)
  - R:R improvement from dissection
  - Horizontal range quality score
  - Tap spacing score
- **Trade execution log** — live feed of trading decisions and actions

### 4. Tests (~200 lines)

- `tests/unit/test_schematics_5b_trader.py`
  - Test trade entry/exit mechanics
  - Test 5B-specific evaluation rules (tap spacing, horizontal quality, structure dissection)
  - Test Telegram notification formatting
  - Test GitHub storage with separate token

---

## Lecture 5B Key Rules to Encode

From the transcript analysis, these are the concrete trading rules from 5B:

1. **Always check highest TF range validity** — use six candle rule across timeframes (30m → 45m → 1h → 2h → 4h)
2. **Structure dissection for better entries** — when main structure break gives bad R:R (<1.5), dissect last leg on lower TF
3. **Domino effect entry** — enter on blue (micro) structure break, ride through red to black to target
4. **Reconfirmation on breaks in supply/demand** — if BOS occurs inside a supply zone, wait for retest + lower TF break confirmation
5. **Model 2 requirements** — price must mitigate extreme supply/demand OR grab extreme liquidity before tap 3
6. **Extreme liquidity via six candle rule** — first market structure high/low from tap 2 to tap 3 using 6CR
7. **Pivot point order blocks** — OBs in pivot points of deviations are best for Model 2 setups
8. **Tap spacing** — taps should be roughly equally spaced; reject if tap 2→3 distance is much smaller than tap 1→2
9. **Horizontal range quality** — range should look like horizontal/sideways price action, not a steep trend with wicks
10. **Take-the-L-to-get-the-W** — when M2 fails, immediately scan for M1 on the same range

---

## File Changes Summary

| File | Action | Description |
|------|--------|-------------|
| `schematics_5b_trader.py` | **CREATE** | New trading engine with 5B logic |
| `server_mexc.py` | **EDIT** | Add /schematics-5B routes, API endpoints, background loop |
| `tests/unit/test_schematics_5b_trader.py` | **CREATE** | Unit tests for new trader |

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

1. Create `schematics_5b_trader.py` with the trading engine
2. Add API routes and background loop to `server_mexc.py`
3. Build the schematics-5B UI page in `server_mexc.py`
4. Create unit tests
5. Run tests and verify
