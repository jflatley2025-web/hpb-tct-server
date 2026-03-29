# Backtest Report — 2026-03-28

Generated: 2026-03-28
Symbol: BTCUSDT | Timeframe: 1h | Period: 2025-04-01 → 2026-03-26

---

## Status Summary

| Run | Engine | Status | Progress |
|-----|--------|--------|----------|
| Run 28 | v13 | **COMPLETED** (2026-03-28 04:07 UTC) | 100% |
| Run 29 | v14 | **RUNNING** | ~95.6% (last signal: 2026-03-10) |

Run 28 is the primary completed result analyzed in this report.
Run 29 (v14) is still in progress and will be the subject of the next report.

---

## Run 28 — Primary Results (Engine v13)

### Overview

| Metric | Value |
|--------|-------|
| Symbol | BTCUSDT |
| Timeframe | 1h |
| Backtest Period | 2025-04-01 → 2026-03-26 (~12 months) |
| Engine Version | v13 |
| Starting Balance | $5,000.00 |
| Final Balance | $17,170.09 |
| **Net Profit** | **$12,170.09** |
| **Total Return** | **243.4%** |
| **Annualized Return (CAGR)** | **~243%** |

### Core Performance Metrics

| Metric | Value |
|--------|-------|
| Total Trades | 112 |
| Wins | 92 |
| Losses | 20 |
| **Win Rate** | **82.14%** |
| Avg Win (dollars) | $160.02 |
| Avg Loss (dollars) | -$127.59 |
| **Profit Factor** | **5.77** |
| **Max Drawdown** | **5.72%** |
| Sharpe Ratio (annualized) | 9.97 |
| Avg R:R | 2.49 |
| Min R:R | 0.52 |
| Max R:R | 18.37 |
| TP1 Hit Rate | 94/112 (83.9%) |

### Trade Characteristics

| Metric | Value |
|--------|-------|
| Avg Trade Duration | 5.9 hours |
| Min Duration | 1.0 hour |
| Max Duration | 89.0 hours |
| Max Consecutive Wins | 16 |
| Max Consecutive Losses | 4 |
| Largest Single Win | $1,194.62 (Trade #84) |
| Largest Single Loss | -$243.43 (Trade #105) |
| Avg MFE (price units) | 2,654.57 |
| Avg MAE (price units) | -231.64 |

### Configuration

| Parameter | Value |
|-----------|-------|
| Entry Threshold | 50 |
| Min R:R | 0.5 |
| TP1 Close % | 50% |
| TP1 Level % | 50% |
| Trail Factor | 0.5 |
| Fee | 0.06% |
| Slippage | 0.05% |
| Min Pivot Confirm | 6 |
| Warmup Days | 90 |
| Min Bars Between Trades | 1 |

---

## Model Breakdown

**Legacy Taxonomy Note:**
- **Model_3** in this report corresponds to continuation variants in v14 (renamed to `Model_1_CONTINUATION` or `Model_2_CONTINUATION`). Model_3 is not a new model but a legacy label for continuation entries.
- **Model_1_from_M2_failure** represents fallback entries triggered when a Model_2 setup fails gate qualification but the structure still supports a Model_1 entry.
## Model Breakdown

> Note: In v14 taxonomy, legacy `Model_3` entries are continuation structures
> (`Model_1_CONTINUATION` / `Model_2_CONTINUATION`) depending on family context.

| Model | Trades | Wins | Win Rate | Total PnL | Avg R:R |
|-------|--------|------|----------|-----------|---------|
| Model_1 | 54 | 42 | 78% | $6,310.60 | 2.10 |
| Model_2 | 28 | 25 | 89% | $4,642.90 | 3.02 |
| Model_3 | 23 | 18 | 78% | $608.35 | 2.86 |
| Model_1_from_M2_failure | 7 | 7 | **100%** | $608.18 | 2.14 |

**Key observations:**
- Model_2 has the highest win rate (89%) and best average R:R (3.02)
- Model_1 generates the most trades (54) and the most total PnL ($6,310.60)
- `Model_1_from_M2_failure` (fallback entries when M2 setup fails) went 7-for-7 — small sample but notable
- Model_3 has decent edge but lowest PnL contribution; may be entering at lower conviction setups
---

## Direction Breakdown

| Direction | Trades | Wins | Win Rate | Total PnL |
|-----------|--------|------|----------|-----------|
| Bullish (long) | 81 | 69 | **85%** | $8,919.64 |
| Bearish (short) | 31 | 23 | 74% | $3,250.39 |

**Observation:** Bullish setups significantly outperformed bearish in both win rate and total PnL. This aligns with BTCUSDT's overall upward trend in the backtest period. Bearish still profitable but with lower conviction — expected in a trending bull market.

---

## Exit Reason Breakdown

| Exit Reason | Count | % of Trades |
|-------------|-------|-------------|
| breakeven_after_tp1 | 45 | 40.2% |
| target_hit | 29 | 25.9% |
| trailing_stop | 20 | 17.9% |
| stop_hit | 18 | 16.1% |

**Observation:** Only 16.1% of trades hit the hard stop — most exits are managed (breakeven or trail). The 40% breakeven-after-TP1 exits suggest the system is protecting capital aggressively after partial take-profit, which contributes to the low drawdown profile.

---

## Monthly Performance

| Month | Trades | Wins | Win Rate | Monthly PnL |
|-------|--------|------|----------|-------------|
| 2025-06 | 3 | 2 | 67% | -$53.10 |
| 2025-07 | 8 | 7 | 88% | $397.33 |
| 2025-08 | 1 | 0 | 0% | -$72.83 |
| 2025-09 | 17 | 15 | 88% | $412.84 |
| 2025-10 | 21 | 19 | 90% | $2,035.26 |
| 2025-11 | 18 | 13 | 72% | $2,815.60 |
| 2025-12 | 4 | 4 | 100% | $371.45 |
| 2026-01 | 21 | 18 | 86% | $3,569.63 |
| 2026-02 | 9 | 8 | 89% | $1,834.71 |
| 2026-03 | 10 | 6 | 60% | $859.14 |

**Notes:**
- Only 2 losing months (Jun/Aug 2025), both early in the warm-up window
- Best month: Jan 2026 ($3,569.63 on 21 trades)
- Mar 2026 win rate dropped to 60% — may reflect choppier market conditions
- Aug 2025 had only 1 trade (loss) — low activity month, likely fewer qualifying setups

---

## Run Comparison (Recent Completed Runs)

| Run | Engine | Trades | Win Rate | Return | Max DD | Final Balance |
|-----|--------|--------|----------|--------|--------|---------------|
| Run 27 | v11 | 133 | 81.2% | 222.4% | 4.35% | $16,117.58 |
| **Run 28** | **v13** | **112** | **82.1%** | **243.4%** | **5.72%** | **$17,170.09** |

**v13 vs v11:**
- Fewer trades (112 vs 133) — v13 is more selective
- Higher return (243% vs 222%) despite fewer trades — better quality entries
- Slightly higher max drawdown (5.72% vs 4.35%) — acceptable given improved return
- Win rate nearly identical (82.1% vs 81.2%)

---

## Run 29 — In Progress (Engine v14)

| Metric | Value |
|--------|-------|
| Status | Running |
| Progress | ~95.6% |
| Last signal processed | 2026-03-10 08:00 UTC |
| Live trade count (so far) | 131 |

Run 29 uses engine v14 and has already generated 131 trades with ~16 days of period remaining. Final results expected within the next few hours. Note: v14 is already showing higher trade count than v13 (131 vs 112 with ~4% of the period left), suggesting v14 may have looser entry filters.

---

## Risk Assessment

- **Max drawdown of 5.72%** is well within typical institutional tolerance (<15%)
- **Profit factor of 5.77** is exceptional — for every $1 lost, $5.77 was made
- **A Sharpe of ~10** is unusually high and likely reflects the low-drawdown, high-win-rate regime during a trending bull market; it should be stress-tested in sideways and bear periods
- **Only 4 max consecutive losses** — the system handles losing streaks gracefully
- **Low avg trade duration (5.9h)** minimizes overnight gap and event risk

---

## Caveats & Flags

1. **Sharpe ratio (~10) is suspiciously high** — warrants scrutiny of fee/slippage assumptions and potential look-ahead bias, even if unintentional
2. **Bearish win rate (74%)** — meaningful edge but lower than bullish; consider whether shorts should have a higher entry threshold in trending environments
3. **Aug 2025 (0% win rate, 1 trade)** — single data point but worth checking if that period had anomalous conditions that should filter entries
4. **Model_3 lower PnL** — 23 trades for $608 vs Model_2's 28 trades for $4,642. Model_3 may need a higher entry threshold or be reconsidered
5. **Run 29 (v14) has more trades at 95% completion** — need to confirm v14 changes are intentional and not a regression in selectivity