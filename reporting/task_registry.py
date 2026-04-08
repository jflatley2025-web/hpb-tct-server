"""
reporting/task_registry.py — HPB Task Registry
================================================
Maintained source-of-truth for active, completed, and queued engineering tasks.

This is a manually-edited file, not auto-generated.
Update it whenever a task state changes.

Convention:
  active   → currently in-progress or under observation
  completed → done, merged, deployed
  queued   → approved but not yet started
  paused   → deprioritized but not dropped
"""

# ── Last updated ──────────────────────────────────────────────────────────────
REGISTRY_LAST_UPDATED = "2026-04-08"

# ═════════════════════════════════════════════════════════════════════════════
TASK_REGISTRY = {
    "active": [
        "SOL live trade lifecycle monitoring (Model_1 4h, entry=81.98)",
        "SCCE shadow-mode observation window — 20-cycle L3×phase cross-table",
        "Report AV: SCCE / L3 reality check — awaiting cycle data",
        "Daily logging system build (Report AW)",
    ],

    "completed": [
        # Infrastructure
        "mexc_data.py extraction — shared MEXC utilities decoupled from 5A engine",
        "5A dead engine cleanup — 5A_tct_trader.py deleted",
        "Deploy hash-mismatch packaging fix (pip pin removed, --no-cache-dir added)",
        "VARCHAR(10) column widened to VARCHAR(20) for CONDITIONAL rig_status",

        # Engine gates
        "RIG rewrite — counter-bias range logic, displacement fix, score floor",
        "BTC anchor converted from hard block to confidence penalty (-20 pts)",
        "Neutral HTF hard-block removed — now applies 0.90 confidence penalty only",
        "Gate 1A-b HTF direction enforcement (FAIL_HTF_MODEL_DIRECTION)",
        "ROLLOUT_FRACTION set to 1.0 (full production rollout)",

        # Scan loop
        "Scan-while-trade-open (Phase 1): scan continues when trade open, entry blocked by guard",
        "Parallel symbol scan with ThreadPoolExecutor (all 3 symbols concurrent)",
        "Global scan timeout removed (was killing futures at 240s)",
        "L3 relaxed BOS tolerance passthrough bug fixed",

        # SCCE
        "scce_engine.py scaffold — phase machine: seed→tap1→tap2→tap3→bos_pending→qualified",
        "SCCE shadow integration into scan cycle (update after Phase A.5)",
        "SCCE telemetry in get_live_health() + /api/schematics-5b-trader/scce endpoint",
        "SCCE×L3 cross-telemetry (scce_l3_cross phase distribution map)",

        # Tests
        "test_decision_engine_v2.py — HTF bias gate tests (12 cases)",
        "test_neutral_htf_allowed_with_penalty test updated",
        "test_all_pairs_present fixed to use _ALL_SYMBOLS",

        # Monitoring
        "Neutral HTF passthrough telemetry (_neutral_htf counters)",
        "Shadow candidate tracking (_shadow_candidates list)",
        "Per-symbol execution funnel tracking (_symbol_funnels)",
        "L3 near-miss bucket tracking (within_0_10_pct through beyond_0_25_pct)",
        "ETH first-event markers (_eth_first_events)",
    ],

    "queued": [
        "L3 tolerance adjustment — pending Report AV cross-table data",
        "Phase 1 pair expansion: AAVEUSDT, ADAUSDT, XLMUSDT, CRVUSDT, VIRTUALUSDT",
        "Google Docs / Google Drive exporter for daily reports",
        "SCCE automatic invalidation on contradictory BOS (counter-phase detection)",
        "Per-cycle SCCE snapshot archiving (rolling 24h history)",
    ],

    "paused": [
        "Portfolio layer (USE_PORTFOLIO_LAYER=False — multi-position not activated)",
        "MoonDev paper trading integration (MOONDEV_PAPER_TRADING not live)",
    ],
}
