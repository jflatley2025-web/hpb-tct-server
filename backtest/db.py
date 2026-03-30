"""
backtest/db.py — Database Connection + Schema DDL
===================================================
Idempotent schema creation for PostgreSQL backtest tables.
Uses psycopg2-binary with raw SQL, no ORM.
"""

import os
import hashlib
import json
import logging
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Optional

import psycopg2
from psycopg2.extras import RealDictCursor

from backtest.config import DB_NAME, DB_USER, DB_PASSWORD, DB_HOST, DB_PORT

logger = logging.getLogger("backtest.db")

# ── Schema DDL ────────────────────────────────────────────────────────

SCHEMA_SQL = """
-- OHLCV candles with close_time for BOS + latency tracking
CREATE TABLE IF NOT EXISTS ohlcv_candles (
    symbol      VARCHAR(20)      NOT NULL DEFAULT 'BTCUSDT',
    timeframe   VARCHAR(5)       NOT NULL,
    open_time   TIMESTAMPTZ      NOT NULL,
    close_time  TIMESTAMPTZ,
    open        DOUBLE PRECISION NOT NULL,
    high        DOUBLE PRECISION NOT NULL,
    low         DOUBLE PRECISION NOT NULL,
    close       DOUBLE PRECISION NOT NULL,
    volume      DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (symbol, timeframe, open_time),
    CONSTRAINT chk_timeframe CHECK (timeframe IN ('1m','5m','15m','30m','1h','4h','1d'))
);
CREATE INDEX IF NOT EXISTS idx_candles_symbol_tf_time
    ON ohlcv_candles (symbol, timeframe, open_time);

-- Backtest runs with idempotency hash
CREATE TABLE IF NOT EXISTS backtest_runs (
    id               SERIAL PRIMARY KEY,
    name             VARCHAR(200),
    run_hash         VARCHAR(64),
    created_at       TIMESTAMPTZ DEFAULT NOW(),
    start_date       TIMESTAMPTZ NOT NULL,
    end_date         TIMESTAMPTZ NOT NULL,
    step_interval    VARCHAR(10) NOT NULL,
    starting_balance DOUBLE PRECISION NOT NULL,
    final_balance    DOUBLE PRECISION,
    total_trades     INTEGER DEFAULT 0,
    wins             INTEGER DEFAULT 0,
    losses           INTEGER DEFAULT 0,
    win_rate         DOUBLE PRECISION,
    max_drawdown_pct DOUBLE PRECISION,
    config_json      JSONB,
    status           VARCHAR(20) DEFAULT 'running',
    completed_at     TIMESTAMPTZ,
    error_message    TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_bt_runs_hash
    ON backtest_runs(run_hash);

-- Backtest trades with direction-aware MFE/MAE
CREATE TABLE IF NOT EXISTS backtest_trades (
    id            SERIAL PRIMARY KEY,
    run_id        INTEGER REFERENCES backtest_runs(id) ON DELETE CASCADE,
    trade_num     INTEGER NOT NULL,
    symbol        VARCHAR(20) NOT NULL,
    timeframe     VARCHAR(5) NOT NULL,
    direction     VARCHAR(10) NOT NULL,
    model         VARCHAR(50),
    entry_price   DOUBLE PRECISION NOT NULL,
    stop_price    DOUBLE PRECISION NOT NULL,
    target_price  DOUBLE PRECISION NOT NULL,
    tp1_price     DOUBLE PRECISION,
    tp1_hit       BOOLEAN DEFAULT FALSE,
    position_size DOUBLE PRECISION,
    risk_amount   DOUBLE PRECISION,
    leverage      INTEGER,
    rr            DOUBLE PRECISION,
    entry_score   INTEGER,
    entry_reasons JSONB,
    mfe           DOUBLE PRECISION,
    mae           DOUBLE PRECISION,
    opened_at     TIMESTAMPTZ NOT NULL,
    closed_at     TIMESTAMPTZ,
    exit_price    DOUBLE PRECISION,
    exit_reason   VARCHAR(30),
    pnl_pct       DOUBLE PRECISION,
    pnl_dollars   DOUBLE PRECISION,
    is_win        BOOLEAN,
    balance_after DOUBLE PRECISION
);
CREATE INDEX IF NOT EXISTS idx_bt_trades_run
    ON backtest_trades (run_id);

-- Backtest signals — full gate audit + structure snapshot + failure codes
CREATE TABLE IF NOT EXISTS backtest_signals (
    id                       SERIAL PRIMARY KEY,
    run_id                   INTEGER REFERENCES backtest_runs(id) ON DELETE CASCADE,
    signal_time              TIMESTAMPTZ NOT NULL,
    price_at_signal          DOUBLE PRECISION,
    timeframe                VARCHAR(5) NOT NULL,
    direction                VARCHAR(10),
    model                    VARCHAR(50),
    -- Gate audit (MSCE -> 1A -> 1B -> 1C -> RCM -> RIG -> 1D)
    gate_1a_bias             VARCHAR(20),
    gate_1a_pass             BOOLEAN,
    gate_1b_pass             BOOLEAN,
    gate_1c_pass             BOOLEAN,
    rcm_score                DOUBLE PRECISION,
    rcm_valid                BOOLEAN,
    range_duration_hours     DOUBLE PRECISION,
    local_displacement       DOUBLE PRECISION,
    htf_bias                 VARCHAR(20),
    msce_session             VARCHAR(20),
    msce_confidence          DOUBLE PRECISION,
    session_bias             VARCHAR(20),
    rig_status               VARCHAR(10),
    rig_reason               VARCHAR(200),
    score_1d                 INTEGER,
    -- Execution tracking
    execution_confidence     DOUBLE PRECISION,
    latency_to_entry_seconds DOUBLE PRECISION,
    -- Final decision
    final_score              INTEGER,
    final_decision           VARCHAR(10),
    skip_reason              VARCHAR(200),
    failure_code             VARCHAR(50),
    -- Structure snapshot (lightweight, for debugging)
    structure_state          JSONB,
    -- Trade params (if passed)
    entry_price              DOUBLE PRECISION,
    stop_price               DOUBLE PRECISION,
    target_price             DOUBLE PRECISION,
    rr                       DOUBLE PRECISION,
    schematic_json           JSONB
);
CREATE INDEX IF NOT EXISTS idx_bt_signals_run
    ON backtest_signals (run_id);
CREATE INDEX IF NOT EXISTS idx_bt_signals_decision
    ON backtest_signals (run_id, final_decision);
CREATE INDEX IF NOT EXISTS idx_bt_signals_lookup
    ON backtest_signals (run_id, signal_time, final_decision);
"""

# Unique index requires special handling (no IF NOT EXISTS for UNIQUE INDEX in PG < 9.5-style)
UNIQUE_INDEX_SQL = """
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_indexes
        WHERE indexname = 'idx_bt_signal_unique'
    ) THEN
        CREATE UNIQUE INDEX idx_bt_signal_unique
            ON backtest_signals (run_id, signal_time, timeframe, model);
    END IF;
END
$$;
"""


# ── Connection helpers ────────────────────────────────────────────────

def get_connection(
    dbname: Optional[str] = None,
    user: Optional[str] = None,
    password: Optional[str] = None,
    host: Optional[str] = None,
    port: Optional[int] = None,
):
    """Create a new psycopg2 connection using config or overrides."""
    return psycopg2.connect(
        dbname=dbname or os.environ.get("BACKTEST_DB_NAME", DB_NAME),
        user=user or os.environ.get("BACKTEST_DB_USER", DB_USER),
        password=password or os.environ.get("BACKTEST_DB_PASSWORD", DB_PASSWORD),
        host=host or os.environ.get("BACKTEST_DB_HOST", DB_HOST),
        port=port or int(os.environ.get("BACKTEST_DB_PORT", str(DB_PORT))),
    )


@contextmanager
def get_cursor(conn=None, commit: bool = True):
    """Yield a RealDictCursor. Auto-commits on clean exit if commit=True."""
    own_conn = conn is None
    if own_conn:
        conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            yield cur
        if commit:
            conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        if own_conn:
            conn.close()


# ── Schema management ─────────────────────────────────────────────────

def create_schema(conn=None):
    """Create all backtest tables idempotently."""
    own_conn = conn is None
    if own_conn:
        conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(SCHEMA_SQL)
            cur.execute(UNIQUE_INDEX_SQL)
        conn.commit()
        logger.info("Backtest schema created/verified successfully")
    except Exception:
        conn.rollback()
        raise
    finally:
        if own_conn:
            conn.close()


def drop_schema(conn=None):
    """Drop all backtest tables. USE WITH CAUTION."""
    own_conn = conn is None
    if own_conn:
        conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                DROP TABLE IF EXISTS backtest_signals CASCADE;
                DROP TABLE IF EXISTS backtest_trades CASCADE;
                DROP TABLE IF EXISTS backtest_runs CASCADE;
                DROP TABLE IF EXISTS ohlcv_candles CASCADE;
            """)
        conn.commit()
        logger.info("Backtest schema dropped")
    except Exception:
        conn.rollback()
        raise
    finally:
        if own_conn:
            conn.close()


# ── Run management ────────────────────────────────────────────────────

def compute_run_hash(config: dict) -> str:
    """
    Hash from run configuration + unique nonce.
    Includes a UUID nonce so identical reruns never collide on the unique index.
    The config is embedded so the hash remains traceable to its configuration.
    """
    payload = {**config, "_nonce": str(uuid.uuid4())}
    serialized = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode()).hexdigest()


def create_run(
    conn,
    name: str,
    start_date: datetime,
    end_date: datetime,
    step_interval: str,
    starting_balance: float,
    config: dict,
) -> int:
    """Insert a new backtest run. Returns the run ID."""
    run_hash = compute_run_hash(config)
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO backtest_runs
                (name, run_hash, start_date, end_date, step_interval,
                 starting_balance, config_json, status)
            VALUES (%s, %s, %s, %s, %s, %s, %s, 'running')
            RETURNING id
            """,
            (name, run_hash, start_date, end_date, step_interval,
             starting_balance, json.dumps(config, default=str)),
        )
        run_id = cur.fetchone()[0]
    conn.commit()
    logger.info(f"Created backtest run #{run_id} (hash={run_hash[:12]}...)")
    return run_id


def complete_run(conn, run_id: int, final_balance: float, total_trades: int,
                 wins: int, losses: int, max_drawdown_pct: float):
    """Mark a run as completed with final metrics."""
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0.0
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE backtest_runs SET
                final_balance = %s, total_trades = %s, wins = %s, losses = %s,
                win_rate = %s, max_drawdown_pct = %s,
                status = 'completed', completed_at = NOW()
            WHERE id = %s
            """,
            (final_balance, total_trades, wins, losses, win_rate,
             max_drawdown_pct, run_id),
        )
    conn.commit()


def fail_run(conn, run_id: int, error_message: str):
    """Mark a run as failed."""
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE backtest_runs SET
                status = 'failed', error_message = %s, completed_at = NOW()
            WHERE id = %s
            """,
            (error_message, run_id),
        )
    conn.commit()


# ── Trade logging ─────────────────────────────────────────────────────

def insert_trade(conn, run_id: int, trade: dict) -> int:
    """Insert a completed trade. Returns the trade row ID."""
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO backtest_trades
                (run_id, trade_num, symbol, timeframe, direction, model,
                 entry_price, stop_price, target_price, tp1_price, tp1_hit,
                 position_size, risk_amount, leverage, rr, entry_score,
                 entry_reasons, mfe, mae, opened_at, closed_at,
                 exit_price, exit_reason, pnl_pct, pnl_dollars, is_win,
                 balance_after)
            VALUES (
                %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s,
                %s
            )
            RETURNING id
            """,
            (
                run_id, trade.get("trade_num"), trade.get("symbol"),
                trade.get("timeframe"), trade.get("direction"), trade.get("model"),
                trade.get("entry_price"), trade.get("stop_price"),
                trade.get("target_price"), trade.get("tp1_price"),
                trade.get("tp1_hit", False),
                trade.get("position_size"), trade.get("risk_amount"),
                trade.get("leverage"), trade.get("rr"), trade.get("entry_score"),
                json.dumps(trade.get("entry_reasons", [])),
                trade.get("mfe"), trade.get("mae"),
                trade.get("opened_at"), trade.get("closed_at"),
                trade.get("exit_price"), trade.get("exit_reason"),
                trade.get("pnl_pct"), trade.get("pnl_dollars"),
                trade.get("is_win"), trade.get("balance_after"),
            ),
        )
        row_id = cur.fetchone()[0]
    conn.commit()
    return row_id


# ── Signal logging ────────────────────────────────────────────────────

def insert_signal(conn, run_id: int, signal: dict) -> int:
    """Insert a signal record (TAKE or SKIP). Returns the signal row ID."""
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO backtest_signals
                (run_id, signal_time, price_at_signal, timeframe, direction, model,
                 gate_1a_bias, gate_1a_pass, gate_1b_pass, gate_1c_pass,
                 rcm_score, rcm_valid, range_duration_hours, local_displacement,
                 htf_bias, msce_session, msce_confidence, session_bias,
                 rig_status, rig_reason, score_1d,
                 execution_confidence, latency_to_entry_seconds,
                 final_score, final_decision, skip_reason, failure_code,
                 structure_state,
                 entry_price, stop_price, target_price, rr, schematic_json)
            VALUES (
                %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s, %s,
                %s, %s,
                %s, %s, %s, %s,
                %s,
                %s, %s, %s, %s, %s
            )
            ON CONFLICT (run_id, signal_time, timeframe, model)
            DO UPDATE SET
                final_decision = EXCLUDED.final_decision,
                final_score = EXCLUDED.final_score,
                score_1d = EXCLUDED.score_1d,
                skip_reason = EXCLUDED.skip_reason,
                failure_code = EXCLUDED.failure_code,
                execution_confidence = EXCLUDED.execution_confidence,
                entry_price = EXCLUDED.entry_price,
                stop_price = EXCLUDED.stop_price,
                target_price = EXCLUDED.target_price,
                rr = EXCLUDED.rr,
                schematic_json = EXCLUDED.schematic_json
            WHERE EXCLUDED.final_decision = 'TAKE'
               OR (backtest_signals.final_decision != 'TAKE'
                   AND EXCLUDED.final_score > backtest_signals.final_score)
            RETURNING id
            """,
            (
                run_id, signal.get("signal_time"), signal.get("price_at_signal"),
                signal.get("timeframe"), signal.get("direction"), signal.get("model"),
                signal.get("gate_1a_bias"), signal.get("gate_1a_pass"),
                signal.get("gate_1b_pass"), signal.get("gate_1c_pass"),
                signal.get("rcm_score"), signal.get("rcm_valid"),
                signal.get("range_duration_hours"), signal.get("local_displacement"),
                signal.get("htf_bias"), signal.get("msce_session"),
                signal.get("msce_confidence"), signal.get("session_bias"),
                signal.get("rig_status"), signal.get("rig_reason"),
                signal.get("score_1d"),
                signal.get("execution_confidence"),
                signal.get("latency_to_entry_seconds"),
                signal.get("final_score"), signal.get("final_decision"),
                signal.get("skip_reason"), signal.get("failure_code"),
                json.dumps(signal.get("structure_state")) if signal.get("structure_state") else None,
                signal.get("entry_price"), signal.get("stop_price"),
                signal.get("target_price"), signal.get("rr"),
                json.dumps(signal.get("schematic_json")) if signal.get("schematic_json") else None,
            ),
        )
        row = cur.fetchone()
        row_id = row[0] if row else None
    conn.commit()
    return row_id


# ── Model name normalization ──────────────────────────────────────────

def normalize_model(model: Optional[str]) -> Optional[str]:
    """Map legacy model names from historical DB rows to current taxonomy.

    Historical rows written before the Run 29 rename contain "Model_3".
    "Model_3" → "Model_2_EXT" (continuation / re-accumulation logic).
    Call this wherever model names are read from DB for display, grouping,
    or analysis so legacy and current rows produce consistent labels.
    """
    if model in ("Model_3", "Model_1_CONTINUATION", "Model_2_CONTINUATION"):
        return "Model_2_EXT"
    return model


def model_family(model: Optional[str]) -> Optional[str]:
    """Return the top-level family ("Model_1" or "Model_2") for a model string."""
    if not model:
        return None
    if "Model_2" in model:
        return "Model_2"
    if "Model_1" in model:
        return "Model_1"
    return None


# ── Entrypoint ────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Creating backtest schema...")
    create_schema()
    print("Done.")