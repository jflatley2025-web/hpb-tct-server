"""
backtest/config.py — Backtest Configuration & Constants
========================================================
Defaults match live Schematics 5B: 1% risk, 10x leverage, $5k balance,
60-point entry threshold.
"""

# ── Account defaults ──────────────────────────────────────────────────
STARTING_BALANCE = 5000.0
RISK_PER_TRADE_PCT = 1.0       # % of equity risked per trade
DEFAULT_LEVERAGE = 10
ENTRY_THRESHOLD = 60           # minimum 1D score to take a trade

# ── Execution realism ─────────────────────────────────────────────────
EXECUTION_SLIPPAGE_PCT = 0.0005  # 0.05% slippage per fill
FEE_PCT = 0.0006                 # 0.06% taker fee per fill (entry + exit)

# ── Structural guards ─────────────────────────────────────────────────
MIN_PIVOT_CONFIRM = 6            # candles required to confirm a pivot
MIN_BARS_BETWEEN_TRADES = 1      # cooldown: bars after trade close before re-entry
MIN_RR = 0.5                     # floor filter: block structurally invalid setups only
WARMUP_DAYS = 90                 # days of data before signals are evaluated

# ── TP1 / Trailing Stop ─────────────────────────────────────────────
TP1_POSITION_CLOSE_PCT = 0.50    # close 50% at TP1
TRAIL_FACTOR = 0.50              # trailing stop = 50% of (target - entry) distance

# ── Timeframes ────────────────────────────────────────────────────────
VALID_TIMEFRAMES = ("1m", "5m", "15m", "30m", "1h", "4h", "1d")

HTF_TIMEFRAME = "1d"
MTF_TIMEFRAMES = ["4h", "1h", "30m", "15m"]
LTF_BOS_TIMEFRAMES = ["5m", "1m"]

# Candle limits per timeframe for ingestion
CANDLE_LIMITS = {
    "1d": 200,
    "4h": 300,
    "1h": 400,
    "30m": 400,
    "15m": 400,
    "5m": 400,
    "1m": 400,
}

# ── Database ──────────────────────────────────────────────────────────
import os
DB_NAME = os.environ.get("BACKTEST_DB_NAME", "first_db_local")
DB_USER = os.environ.get("BACKTEST_DB_USER", "bulldog")
DB_PASSWORD = os.environ.get("BACKTEST_DB_PASSWORD", "")
DB_HOST = os.environ.get("BACKTEST_DB_HOST", "localhost")
DB_PORT = int(os.environ.get("BACKTEST_DB_PORT", "5432"))

# ── Ingestion ─────────────────────────────────────────────────────────
DEFAULT_SYMBOL = "BTCUSDT"
EXCHANGE_PRIORITY = ["bybit", "okx", "mexc"]  # primary → fallback
RATE_LIMIT_DELAY = 0.5          # seconds between API requests

# ── Timeframe durations (seconds) ────────────────────────────────────
TIMEFRAME_SECONDS = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "4h": 14400,
    "1d": 86400,
}


def timeframe_to_seconds(tf: str) -> int:
    """Convert timeframe string to seconds."""
    if tf not in TIMEFRAME_SECONDS:
        raise ValueError(f"Unknown timeframe: {tf}")
    return TIMEFRAME_SECONDS[tf]
