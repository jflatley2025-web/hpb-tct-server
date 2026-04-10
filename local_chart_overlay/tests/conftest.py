"""Test fixtures for local_chart_overlay."""
from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from local_chart_overlay.models.trade import TradeRecord
from local_chart_overlay.models.schematic import FrozenSchematic
from local_chart_overlay.storage.sqlite_store import SqliteStore


@pytest.fixture
def tmp_db(tmp_path):
    """Create a temporary SQLite store."""
    db_path = tmp_path / "test_overlay.db"
    store = SqliteStore(db_path)
    yield store
    store.close()


@pytest.fixture
def sample_trade() -> TradeRecord:
    return TradeRecord(
        source_id="json:1",
        source_type="json",
        symbol="BTCUSDT",
        timeframe="1h",
        direction="bearish",
        model="Model_1",
        entry_price=71044.18,
        stop_price=72036.35,
        target_price=68979.18,
        tp1_price=70011.68,
        opened_at=datetime(2026, 3, 13, 19, 1, 28, tzinfo=timezone.utc),
        closed_at=datetime(2026, 3, 15, 21, 38, 24, tzinfo=timezone.utc),
        exit_price=72187.18,
        pnl_pct=-1.61,
        pnl_dollars=-57.6,
        is_win=False,
        exit_reason="stop_hit",
        entry_score=93,
        rr=2.08,
        leverage=10,
        position_size=3580.23,
        risk_amount=50.0,
        htf_bias="bullish",
    )


@pytest.fixture
def sample_trade_win() -> TradeRecord:
    return TradeRecord(
        source_id="json:2",
        source_type="json",
        symbol="BTCUSDT",
        timeframe="30m",
        direction="bearish",
        model="Model_1",
        entry_price=74147.84,
        stop_price=74531.72,
        target_price=72850.0,
        tp1_price=73498.92,
        opened_at=datetime(2026, 3, 17, 5, 2, 11, tzinfo=timezone.utc),
        closed_at=datetime(2026, 3, 26, 15, 11, 39, tzinfo=timezone.utc),
        exit_price=69149.8,
        pnl_pct=6.74,
        pnl_dollars=643.48,
        is_win=True,
        exit_reason="target_hit",
        entry_score=91,
        rr=3.38,
        leverage=10,
    )


@pytest.fixture
def sample_schematic() -> FrozenSchematic:
    return FrozenSchematic(
        tap1_price=72000.0,
        tap1_time=datetime(2026, 3, 12, 10, 0, 0, tzinfo=timezone.utc),
        tap2_price=72200.0,
        tap2_time=datetime(2026, 3, 12, 14, 0, 0, tzinfo=timezone.utc),
        tap3_price=72100.0,
        tap3_time=datetime(2026, 3, 13, 8, 0, 0, tzinfo=timezone.utc),
        range_high_price=72500.0,
        range_high_time=datetime(2026, 3, 11, 12, 0, 0, tzinfo=timezone.utc),
        range_low_price=70500.0,
        range_low_time=datetime(2026, 3, 11, 18, 0, 0, tzinfo=timezone.utc),
        bos_price=71800.0,
        bos_time=datetime(2026, 3, 13, 16, 0, 0, tzinfo=timezone.utc),
        sweep_type="liquidity",
        model_label="Model_1",
        timeframe="1h",
    )


@pytest.fixture
def sample_json_file(tmp_path) -> Path:
    """Create a sample trade log JSON file."""
    data = {
        "balance": 5527.44,
        "starting_balance": 5000.0,
        "current_trade": None,
        "trade_history": [
            {
                "id": 1,
                "symbol": "BTCUSDT",
                "timeframe": "1h",
                "direction": "bearish",
                "model": "Model_1",
                "entry_price": 71044.18,
                "stop_price": 72036.35,
                "target_price": 68979.18,
                "tp1_price": 70011.68,
                "tp1_hit": False,
                "position_size": 3580.23,
                "risk_amount": 50.0,
                "leverage": 10,
                "rr": 2.08,
                "entry_score": 93,
                "entry_reasons": ["Range: 20/20", "Taps: 20/20"],
                "htf_bias": "bullish",
                "opened_at": "2026-03-13T19:01:28.640322+00:00",
                "status": "closed",
                "exit_price": 72187.18,
                "exit_reason": "stop_hit",
                "pnl_pct": -1.61,
                "pnl_dollars": -57.6,
                "is_win": False,
                "closed_at": "2026-03-15T21:38:24.592234+00:00",
                "balance_after": 4942.4,
            },
            {
                "id": 2,
                "symbol": "ETHUSDT",
                "timeframe": "4h",
                "direction": "bullish",
                "model": "Model_2",
                "entry_price": 2100.0,
                "stop_price": 2050.0,
                "target_price": 2250.0,
                "tp1_price": 2175.0,
                "tp1_hit": True,
                "position_size": 1000.0,
                "risk_amount": 25.0,
                "leverage": 20,
                "rr": 3.0,
                "entry_score": 88,
                "entry_reasons": ["Range: 18/20"],
                "htf_bias": "bullish",
                "opened_at": "2026-03-20T12:00:00+00:00",
                "status": "closed",
                "exit_price": 2250.0,
                "exit_reason": "target_hit",
                "pnl_pct": 7.14,
                "pnl_dollars": 71.4,
                "is_win": True,
                "closed_at": "2026-03-22T18:30:00+00:00",
                "balance_after": 5013.8,
            },
            {
                "id": 99,
                "symbol": "SOLUSDT",
                "direction": "bullish",
                "model": "Model_1",
                "entry_price": 80.0,
                "stop_price": 78.0,
                "target_price": 86.0,
                "opened_at": "2026-04-01T10:00:00+00:00",
                "status": "open",
            },
        ],
    }
    path = tmp_path / "trade_log.json"
    path.write_text(json.dumps(data))
    return path


@pytest.fixture
def sample_csv_file(tmp_path) -> Path:
    """Create a sample CSV trade report."""
    csv_content = (
        "symbol,direction,timeframe,entry_price,stop_price,target_price,"
        "opened_at,closed_at,pnl_pct,is_win,exit_reason,rr\n"
        "BTCUSDT,bearish,1h,71000.0,72000.0,69000.0,"
        "2026-03-13T19:00:00+00:00,2026-03-15T21:00:00+00:00,-1.5,false,stop_hit,2.0\n"
        "ETHUSDT,bullish,4h,2100.0,2050.0,2250.0,"
        "2026-03-20T12:00:00+00:00,2026-03-22T18:00:00+00:00,7.0,true,target_hit,3.0\n"
    )
    path = tmp_path / "trades.csv"
    path.write_text(csv_content)
    return path
