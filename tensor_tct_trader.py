"""
tensor_tct_trader.py — Backward-compatible re-export shim
==========================================================
Re-exports shared utilities from mexc_data.py for server_mexc.py
and other legacy callers that import from this module.

The 5A trading engine (TensorTCTTrader, TradeState, TCTTradeEvaluator)
has been removed.  Stubs are provided so that lazy imports in
server_mexc.py don't crash at import time — they raise at call time.
"""

# --- Shared utilities (active, used by 5B engine + server) ---
from mexc_data import (  # noqa: F401
    fetch_candles_sync,
    fetch_live_price,
    TRADE_LOG_PATH,
    TRADE_LOG_BACKUP_PATH,
    AUTO_SCAN_INTERVAL,
    STARTING_BALANCE,
    RISK_PER_TRADE_PCT,
    ENTRY_THRESHOLD,
    MIN_RR,
    MIN_QUALITY_SCORE,
    DUPLICATE_COOLDOWN_SECONDS,
    DUPLICATE_PRICE_TOLERANCE,
    MEXC_KLINES_URL,
    MEXC_TICKER_URL,
)


# --- Stubs for removed 5A engine classes ---
# server_mexc.py has lazy imports of these in route handlers.
# Raising at call time (not import time) avoids crashing the server.

class _RemovedEngineError(RuntimeError):
    def __init__(self, name: str):
        super().__init__(
            f"{name} has been removed. The 5A trading engine is retired; "
            f"use Schematics5BTrader instead."
        )


class TCTTradeEvaluator:
    """Stub — 5A evaluator removed."""
    def __init__(self, *a, **kw):
        raise _RemovedEngineError("TCTTradeEvaluator")


class TensorTCTTrader:
    """Stub — 5A trader removed."""
    def __init__(self, *a, **kw):
        raise _RemovedEngineError("TensorTCTTrader")


class TradeState:
    """Stub — 5A state removed."""
    def __init__(self, *a, **kw):
        raise _RemovedEngineError("TradeState")


def get_trader():
    """Stub — 5A singleton removed."""
    raise _RemovedEngineError("get_trader")
