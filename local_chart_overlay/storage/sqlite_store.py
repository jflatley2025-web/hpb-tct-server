"""SQLite storage — single source of truth for trades and schematics."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from local_chart_overlay.models.trade import TradeRecord
from local_chart_overlay.models.schematic import FrozenSchematic


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS trades (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id     TEXT UNIQUE NOT NULL,
    source_type   TEXT NOT NULL,
    symbol        TEXT NOT NULL,
    timeframe     TEXT,
    direction     TEXT NOT NULL,
    model         TEXT,
    entry_price   REAL NOT NULL,
    stop_price    REAL NOT NULL,
    target_price  REAL NOT NULL,
    tp1_price     REAL,
    opened_at     TEXT NOT NULL,
    closed_at     TEXT,
    exit_price    REAL,
    pnl_pct       REAL,
    pnl_dollars   REAL,
    is_win        INTEGER,
    exit_reason   TEXT,
    entry_score   INTEGER,
    rr            REAL,
    leverage      INTEGER,
    position_size REAL,
    risk_amount   REAL,
    mfe           REAL,
    mae           REAL,
    entry_reasons TEXT,
    htf_bias      TEXT,
    metadata      TEXT,
    imported_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS schematics (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_id        INTEGER NOT NULL UNIQUE REFERENCES trades(id),
    tap1_price      REAL,
    tap1_time       TEXT,
    tap2_price      REAL,
    tap2_time       TEXT,
    tap3_price      REAL,
    tap3_time       TEXT,
    range_high_price REAL,
    range_high_time TEXT,
    range_low_price REAL,
    range_low_time  TEXT,
    bos_price       REAL,
    bos_time        TEXT,
    sweep_type      TEXT,
    model_label     TEXT,
    timeframe       TEXT,
    context_timeframe TEXT,
    execution_timeframe TEXT,
    parent_structure_id INTEGER,
    version         INTEGER DEFAULT 1,
    source          TEXT DEFAULT 'manual',
    created_at      TEXT,
    updated_at      TEXT,
    manually_edited INTEGER DEFAULT 0,
    last_edited_at  TEXT,
    notes           TEXT,
    confidence      REAL,
    tags            TEXT,
    data_source     TEXT
);

CREATE TABLE IF NOT EXISTS schematic_history (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_id        INTEGER NOT NULL REFERENCES trades(id),
    version         INTEGER NOT NULL,
    snapshot_json   TEXT NOT NULL,
    changed_at      TEXT NOT NULL,
    change_source   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS render_exports (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_ids   TEXT NOT NULL,
    exported_at TEXT NOT NULL,
    output_path TEXT NOT NULL,
    pine_version TEXT DEFAULT 'v6'
);
"""


def _iso(dt: Optional[datetime]) -> Optional[str]:
    if dt is None:
        return None
    return dt.isoformat()


def _parse_dt(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


class SqliteStore:
    """CRUD operations on the local overlay database."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._init_schema()

    def _init_schema(self):
        self._conn.executescript(SCHEMA_SQL)
        self._conn.commit()

    @property
    def annotations(self):
        """Lazy-initialized AnnotationService on the same connection."""
        if not hasattr(self, "_annotations"):
            from local_chart_overlay.annotations.service import AnnotationService
            self._annotations = AnnotationService(self._conn)
        return self._annotations

    def close(self):
        self._conn.close()

    # ── Trades ────────────────────────────────────────────────────────

    def upsert_trade(self, trade: TradeRecord) -> int:
        """Insert or update a trade by source_id. Returns internal ID."""
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            """INSERT INTO trades (
                source_id, source_type, symbol, timeframe, direction, model,
                entry_price, stop_price, target_price, tp1_price,
                opened_at, closed_at, exit_price,
                pnl_pct, pnl_dollars, is_win, exit_reason,
                entry_score, rr, leverage, position_size, risk_amount,
                mfe, mae, entry_reasons, htf_bias, metadata, imported_at
            ) VALUES (
                ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?,
                ?, ?, ?,
                ?, ?, ?, ?,
                ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?
            )
            ON CONFLICT(source_id) DO UPDATE SET
                symbol=excluded.symbol, timeframe=excluded.timeframe,
                direction=excluded.direction, model=excluded.model,
                entry_price=excluded.entry_price, stop_price=excluded.stop_price,
                target_price=excluded.target_price, tp1_price=excluded.tp1_price,
                opened_at=excluded.opened_at, closed_at=excluded.closed_at,
                exit_price=excluded.exit_price,
                pnl_pct=excluded.pnl_pct, pnl_dollars=excluded.pnl_dollars,
                is_win=excluded.is_win, exit_reason=excluded.exit_reason,
                entry_score=excluded.entry_score, rr=excluded.rr,
                leverage=excluded.leverage, position_size=excluded.position_size,
                risk_amount=excluded.risk_amount,
                mfe=excluded.mfe, mae=excluded.mae,
                entry_reasons=excluded.entry_reasons, htf_bias=excluded.htf_bias,
                metadata=excluded.metadata
            """,
            (
                trade.source_id, trade.source_type,
                trade.symbol, trade.timeframe, trade.direction, trade.model,
                trade.entry_price, trade.stop_price, trade.target_price, trade.tp1_price,
                _iso(trade.opened_at), _iso(trade.closed_at), trade.exit_price,
                trade.pnl_pct, trade.pnl_dollars,
                int(trade.is_win) if trade.is_win is not None else None,
                trade.exit_reason,
                trade.entry_score, trade.rr, trade.leverage,
                trade.position_size, trade.risk_amount,
                trade.mfe, trade.mae,
                json.dumps(trade.entry_reasons) if trade.entry_reasons else None,
                trade.htf_bias,
                json.dumps(trade.metadata) if trade.metadata else None,
                now,
            ),
        )
        self._conn.commit()
        cur = self._conn.execute(
            "SELECT id FROM trades WHERE source_id = ?", (trade.source_id,)
        )
        return cur.fetchone()["id"]

    def get_trade(self, trade_id: int) -> Optional[TradeRecord]:
        cur = self._conn.execute("SELECT * FROM trades WHERE id = ?", (trade_id,))
        row = cur.fetchone()
        if not row:
            return None
        return self._row_to_trade(row)

    def list_trades(
        self,
        symbol: Optional[str] = None,
        direction: Optional[str] = None,
        source_type: Optional[str] = None,
    ) -> list[tuple[int, TradeRecord]]:
        """Return list of (id, TradeRecord) tuples, with optional filters."""
        query = "SELECT * FROM trades WHERE 1=1"
        params: list = []
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        if direction:
            query += " AND direction = ?"
            params.append(direction)
        if source_type:
            query += " AND source_type = ?"
            params.append(source_type)
        query += " ORDER BY opened_at DESC"
        cur = self._conn.execute(query, params)
        return [(row["id"], self._row_to_trade(row)) for row in cur.fetchall()]

    def _row_to_trade(self, row: sqlite3.Row) -> TradeRecord:
        return TradeRecord(
            source_id=row["source_id"],
            source_type=row["source_type"],
            symbol=row["symbol"],
            timeframe=row["timeframe"],
            direction=row["direction"],
            model=row["model"],
            entry_price=row["entry_price"],
            stop_price=row["stop_price"],
            target_price=row["target_price"],
            tp1_price=row["tp1_price"],
            opened_at=_parse_dt(row["opened_at"]),
            closed_at=_parse_dt(row["closed_at"]),
            exit_price=row["exit_price"],
            pnl_pct=row["pnl_pct"],
            pnl_dollars=row["pnl_dollars"],
            is_win=bool(row["is_win"]) if row["is_win"] is not None else None,
            exit_reason=row["exit_reason"],
            entry_score=row["entry_score"],
            rr=row["rr"],
            leverage=row["leverage"],
            position_size=row["position_size"],
            risk_amount=row["risk_amount"],
            mfe=row["mfe"],
            mae=row["mae"],
            entry_reasons=json.loads(row["entry_reasons"]) if row["entry_reasons"] else None,
            htf_bias=row["htf_bias"],
            metadata=json.loads(row["metadata"]) if row["metadata"] else None,
        )

    # ── Schematics ────────────────────────────────────────────────────

    def upsert_schematic(self, trade_id: int, sch: FrozenSchematic):
        """Insert or replace schematic for a trade. Preserves version history."""
        now = datetime.now(timezone.utc).isoformat()

        # Check for existing schematic to determine version + archive
        existing = self._conn.execute(
            "SELECT version, id FROM schematics WHERE trade_id = ?", (trade_id,)
        ).fetchone()

        if existing:
            old_version = existing["version"]
            new_version = old_version + 1
            # Archive the old version before overwriting
            old_row = self._conn.execute(
                "SELECT * FROM schematics WHERE trade_id = ?", (trade_id,)
            ).fetchone()
            old_snapshot = {k: old_row[k] for k in old_row.keys() if k != "id"}
            self._conn.execute(
                """INSERT INTO schematic_history
                   (trade_id, version, snapshot_json, changed_at, change_source)
                   VALUES (?, ?, ?, ?, ?)""",
                (trade_id, old_version, json.dumps(old_snapshot), now, sch.source),
            )
        else:
            new_version = sch.version
            now_created = _iso(sch.created_at) or now

        created_at = _iso(sch.created_at) if sch.created_at else (
            now if not existing else None  # preserve original created_at on update
        )

        self._conn.execute(
            """INSERT INTO schematics (
                trade_id,
                tap1_price, tap1_time, tap2_price, tap2_time,
                tap3_price, tap3_time,
                range_high_price, range_high_time,
                range_low_price, range_low_time,
                bos_price, bos_time,
                sweep_type, model_label, timeframe,
                context_timeframe, execution_timeframe, parent_structure_id,
                version, source, created_at, updated_at,
                manually_edited, last_edited_at,
                notes, confidence, tags, data_source
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                      ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(trade_id) DO UPDATE SET
                tap1_price=excluded.tap1_price, tap1_time=excluded.tap1_time,
                tap2_price=excluded.tap2_price, tap2_time=excluded.tap2_time,
                tap3_price=excluded.tap3_price, tap3_time=excluded.tap3_time,
                range_high_price=excluded.range_high_price,
                range_high_time=excluded.range_high_time,
                range_low_price=excluded.range_low_price,
                range_low_time=excluded.range_low_time,
                bos_price=excluded.bos_price, bos_time=excluded.bos_time,
                sweep_type=excluded.sweep_type,
                model_label=excluded.model_label, timeframe=excluded.timeframe,
                context_timeframe=excluded.context_timeframe,
                execution_timeframe=excluded.execution_timeframe,
                parent_structure_id=excluded.parent_structure_id,
                version=excluded.version, source=excluded.source,
                updated_at=excluded.updated_at,
                manually_edited=excluded.manually_edited,
                last_edited_at=excluded.last_edited_at,
                notes=excluded.notes, confidence=excluded.confidence,
                tags=excluded.tags, data_source=excluded.data_source
            """,
            (
                trade_id,
                sch.tap1_price, _iso(sch.tap1_time),
                sch.tap2_price, _iso(sch.tap2_time),
                sch.tap3_price, _iso(sch.tap3_time),
                sch.range_high_price, _iso(sch.range_high_time),
                sch.range_low_price, _iso(sch.range_low_time),
                sch.bos_price, _iso(sch.bos_time),
                sch.sweep_type, sch.model_label, sch.timeframe,
                sch.context_timeframe, sch.execution_timeframe,
                sch.parent_structure_id,
                new_version, sch.source, created_at, now,
                int(sch.manually_edited), _iso(sch.last_edited_at),
                sch.notes, sch.confidence, sch.tags, sch.data_source,
            ),
        )
        self._conn.commit()

    def get_schematic(self, trade_id: int) -> Optional[FrozenSchematic]:
        cur = self._conn.execute(
            "SELECT * FROM schematics WHERE trade_id = ?", (trade_id,)
        )
        row = cur.fetchone()
        if not row:
            return None
        return self._row_to_schematic(row)

    def _row_to_schematic(self, row: sqlite3.Row) -> FrozenSchematic:
        return FrozenSchematic(
            tap1_price=row["tap1_price"],
            tap1_time=_parse_dt(row["tap1_time"]),
            tap2_price=row["tap2_price"],
            tap2_time=_parse_dt(row["tap2_time"]),
            tap3_price=row["tap3_price"],
            tap3_time=_parse_dt(row["tap3_time"]),
            range_high_price=row["range_high_price"],
            range_high_time=_parse_dt(row["range_high_time"]),
            range_low_price=row["range_low_price"],
            range_low_time=_parse_dt(row["range_low_time"]),
            bos_price=row["bos_price"],
            bos_time=_parse_dt(row["bos_time"]),
            sweep_type=row["sweep_type"],
            model_label=row["model_label"],
            timeframe=row["timeframe"],
            context_timeframe=row["context_timeframe"],
            execution_timeframe=row["execution_timeframe"],
            parent_structure_id=row["parent_structure_id"],
            version=row["version"] or 1,
            source=row["source"] or "manual",
            created_at=_parse_dt(row["created_at"]),
            updated_at=_parse_dt(row["updated_at"]),
            manually_edited=bool(row["manually_edited"]),
            last_edited_at=_parse_dt(row["last_edited_at"]),
            notes=row["notes"],
            confidence=row["confidence"],
            tags=row["tags"],
            data_source=row["data_source"] if "data_source" in row.keys() else None,
        )

    def get_schematic_history(self, trade_id: int) -> list[dict]:
        """Return version history for a trade's schematic."""
        cur = self._conn.execute(
            """SELECT version, changed_at, change_source, snapshot_json
               FROM schematic_history WHERE trade_id = ?
               ORDER BY version ASC""",
            (trade_id,),
        )
        return [
            {
                "version": row["version"],
                "changed_at": row["changed_at"],
                "source": row["change_source"],
                "snapshot": json.loads(row["snapshot_json"]),
            }
            for row in cur.fetchall()
        ]

    # ── Combined queries ──────────────────────────────────────────────

    def get_trades_with_schematics(
        self,
        trade_ids: Optional[list[int]] = None,
    ) -> list[tuple[int, TradeRecord, Optional[FrozenSchematic]]]:
        """Return (id, trade, schematic) triples."""
        if trade_ids:
            placeholders = ",".join("?" for _ in trade_ids)
            query = f"""
                SELECT t.*, s.id as sch_id,
                    s.tap1_price as s_tap1_price, s.tap1_time as s_tap1_time,
                    s.tap2_price as s_tap2_price, s.tap2_time as s_tap2_time,
                    s.tap3_price as s_tap3_price, s.tap3_time as s_tap3_time,
                    s.range_high_price as s_range_high_price,
                    s.range_high_time as s_range_high_time,
                    s.range_low_price as s_range_low_price,
                    s.range_low_time as s_range_low_time,
                    s.bos_price as s_bos_price, s.bos_time as s_bos_time,
                    s.sweep_type as s_sweep_type,
                    s.model_label as s_model_label,
                    s.timeframe as s_timeframe,
                    s.context_timeframe as s_context_timeframe,
                    s.execution_timeframe as s_execution_timeframe,
                    s.parent_structure_id as s_parent_structure_id,
                    s.version as s_version, s.source as s_source,
                    s.created_at as s_created_at, s.updated_at as s_updated_at,
                    s.manually_edited as s_manually_edited,
                    s.last_edited_at as s_last_edited_at,
                    s.notes as s_notes, s.confidence as s_confidence,
                    s.tags as s_tags, s.data_source as s_data_source
                FROM trades t
                LEFT JOIN schematics s ON s.trade_id = t.id
                WHERE t.id IN ({placeholders})
                ORDER BY t.opened_at DESC
            """
            cur = self._conn.execute(query, trade_ids)
        else:
            cur = self._conn.execute("""
                SELECT t.*, s.id as sch_id,
                    s.tap1_price as s_tap1_price, s.tap1_time as s_tap1_time,
                    s.tap2_price as s_tap2_price, s.tap2_time as s_tap2_time,
                    s.tap3_price as s_tap3_price, s.tap3_time as s_tap3_time,
                    s.range_high_price as s_range_high_price,
                    s.range_high_time as s_range_high_time,
                    s.range_low_price as s_range_low_price,
                    s.range_low_time as s_range_low_time,
                    s.bos_price as s_bos_price, s.bos_time as s_bos_time,
                    s.sweep_type as s_sweep_type,
                    s.model_label as s_model_label,
                    s.timeframe as s_timeframe,
                    s.context_timeframe as s_context_timeframe,
                    s.execution_timeframe as s_execution_timeframe,
                    s.parent_structure_id as s_parent_structure_id,
                    s.version as s_version, s.source as s_source,
                    s.created_at as s_created_at, s.updated_at as s_updated_at,
                    s.manually_edited as s_manually_edited,
                    s.last_edited_at as s_last_edited_at,
                    s.notes as s_notes, s.confidence as s_confidence,
                    s.tags as s_tags, s.data_source as s_data_source
                FROM trades t
                LEFT JOIN schematics s ON s.trade_id = t.id
                ORDER BY t.opened_at DESC
            """)

        results = []
        for row in cur.fetchall():
            trade = self._row_to_trade(row)
            schematic = None
            if row["sch_id"] is not None:
                schematic = FrozenSchematic(
                    tap1_price=row["s_tap1_price"],
                    tap1_time=_parse_dt(row["s_tap1_time"]),
                    tap2_price=row["s_tap2_price"],
                    tap2_time=_parse_dt(row["s_tap2_time"]),
                    tap3_price=row["s_tap3_price"],
                    tap3_time=_parse_dt(row["s_tap3_time"]),
                    range_high_price=row["s_range_high_price"],
                    range_high_time=_parse_dt(row["s_range_high_time"]),
                    range_low_price=row["s_range_low_price"],
                    range_low_time=_parse_dt(row["s_range_low_time"]),
                    bos_price=row["s_bos_price"],
                    bos_time=_parse_dt(row["s_bos_time"]),
                    sweep_type=row["s_sweep_type"],
                    model_label=row["s_model_label"],
                    timeframe=row["s_timeframe"],
                    context_timeframe=row["s_context_timeframe"],
                    execution_timeframe=row["s_execution_timeframe"],
                    parent_structure_id=row["s_parent_structure_id"],
                    version=row["s_version"] or 1,
                    source=row["s_source"] or "manual",
                    created_at=_parse_dt(row["s_created_at"]),
                    updated_at=_parse_dt(row["s_updated_at"]),
                    manually_edited=bool(row["s_manually_edited"]),
                    last_edited_at=_parse_dt(row["s_last_edited_at"]),
                    notes=row["s_notes"],
                    confidence=row["s_confidence"],
                    tags=row["s_tags"],
                    data_source=row["s_data_source"],
                )
            results.append((row["id"], trade, schematic))
        return results

    # ── Export tracking ───────────────────────────────────────────────

    def record_export(self, trade_ids: list[int], output_path: str):
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "INSERT INTO render_exports (trade_ids, exported_at, output_path) VALUES (?, ?, ?)",
            (json.dumps(trade_ids), now, output_path),
        )
        self._conn.commit()

    def trade_count(self) -> int:
        cur = self._conn.execute("SELECT COUNT(*) FROM trades")
        return cur.fetchone()[0]
