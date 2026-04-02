"""
portfolio_manager.py — Correlation-Aware Portfolio Risk Layer
=============================================================
Issue 5: Multi-symbol portfolio with shared capital pool and
correlation-adjusted exposure control.

Feature flag: USE_PORTFOLIO_LAYER (default False).
When False: can_open_trade() returns allowed=True with no restriction
so the single-symbol BTC path is byte-for-byte identical to pre-v5.

Symbols are normalised internally to their base asset
(e.g. BTCUSDT → BTC) — callers may pass either form.

Public API
----------
USE_PORTFOLIO_LAYER    — bool flag (set to True to activate)
MAX_PORTFOLIO_RISK_PCT — hard cap on total correlated risk
MAX_CONCURRENT_TRADES  — hard cap on simultaneous open positions

PortfolioPosition  — one open position (dataclass)
PortfolioState     — shared account pool (dataclass)

get_correlation(sym_a, sym_b)            -> float
adjusted_portfolio_risk(sym, risk, port) -> (float, dict)
can_open_trade(sym, risk, port)          -> dict
open_position(port, ...)                 -> PortfolioPosition
close_position(port, symbol)             -> bool
debug_snapshot(port)                     -> dict

Structured log tags
-------------------
PORTFOLIO_ALLOW  — trade fits within cap, full size
PORTFOLIO_SCALE  — trade scaled down to fill remaining headroom
PORTFOLIO_BLOCK  — no headroom; trade suppressed entirely
PORTFOLIO_OPEN   — position registered after successful order
PORTFOLIO_CLOSE  — position deregistered after trade closes
"""

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger("portfolio_manager")

# ── Feature flag ──────────────────────────────────────────────────────
# Default OFF so single-symbol path is unaffected until explicitly activated.
USE_PORTFOLIO_LAYER: bool = False

# ── Constants ─────────────────────────────────────────────────────────
MAX_PORTFOLIO_RISK_PCT: float = 2.0   # % of equity; cap on total correlated risk
MAX_CONCURRENT_TRADES: int = 2        # hard cap on simultaneous open positions


def _load_risk_config() -> None:
    """Load risk config from config/risk_profile.json (if present).

    Updates module-level MAX_PORTFOLIO_RISK_PCT and MAX_CONCURRENT_TRADES
    from the JSON file.  Silently no-ops if the file is missing or corrupt
    so hardcoded defaults remain in effect.
    """
    global MAX_PORTFOLIO_RISK_PCT, MAX_CONCURRENT_TRADES
    config_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "config", "risk_profile.json"
    )
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        if "risk_per_trade_pct" in cfg:
            # risk_per_trade_pct is per-trade; portfolio cap is separate
            pass
        if "max_open_positions" in cfg:
            MAX_CONCURRENT_TRADES = int(cfg["max_open_positions"])
        # risk_profile.json doesn't have a portfolio risk field yet;
        # MAX_PORTFOLIO_RISK_PCT stays at the hardcoded default (2.0%).
        logger.debug(
            "Risk config loaded: MAX_CONCURRENT_TRADES=%d, "
            "MAX_PORTFOLIO_RISK_PCT=%.1f%%",
            MAX_CONCURRENT_TRADES, MAX_PORTFOLIO_RISK_PCT,
        )
    except FileNotFoundError:
        logger.debug("risk_profile.json not found — using defaults")
    except Exception as e:
        logger.warning("Failed to load risk_profile.json: %s — using defaults", e)


_load_risk_config()

# ── Correlation matrix ────────────────────────────────────────────────
# Pairwise Pearson-like coefficients (monthly, crypto bull-market regime).
# Key: (A, B) with A < B alphabetically (canonical order).
# Unlisted pairs default to 0.0 (uncorrelated / no data).
_CORRELATION: dict[tuple[str, str], float] = {
    ("BTC", "ETH"): 0.85,
    ("BTC", "SOL"): 0.75,
    ("ETH", "SOL"): 0.80,
}


def _base_asset(symbol: str) -> str:
    """Strip quote-currency suffix: BTCUSDT → BTC, ETHUSDT → ETH."""
    sym = symbol.upper()
    for quote in ("USDT", "USDC", "BUSD", "USD"):
        if sym.endswith(quote):
            return sym[: -len(quote)]
    return sym


def get_correlation(sym_a: str, sym_b: str) -> float:
    """Return the correlation between two trading symbols.

    Returns 1.0 for identical symbols, 0.0 for unknown pairs.
    Symmetric: get_correlation(A, B) == get_correlation(B, A).
    """
    a = _base_asset(sym_a)
    b = _base_asset(sym_b)
    if a == b:
        return 1.0
    pair = (min(a, b), max(a, b))  # canonical alphabetical key
    return _CORRELATION.get(pair, 0.0)


# ── Data structures ───────────────────────────────────────────────────

@dataclass
class PortfolioPosition:
    """One open position registered in the shared capital pool.

    ``notional_risk`` is the dollar amount at risk (fixed at entry,
    not mark-to-market).  Used to compute total portfolio exposure.
    """
    symbol: str           # e.g. "BTCUSDT"
    direction: str        # "bullish" | "bearish"
    notional_risk: float  # $ at risk on this leg (fixed at open)
    entry_price: float
    model: str
    timeframe: str
    opened_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


@dataclass
class PortfolioState:
    """Shared account equity and open-position registry.

    ``equity`` is the account balance at the time of the last sync.
    Callers must sync it from the source-of-truth state before calling
    can_open_trade(); this class does NOT auto-maintain equity.

    Properties
    ----------
    symbol_exposure  — $ at risk per base asset
    total_risk_pct   — aggregate risk as % of current equity
    """
    equity: float
    peak_equity: float
    open_positions: list[PortfolioPosition] = field(default_factory=list)
    last_update_ts: Optional[datetime] = None

    @property
    def symbol_exposure(self) -> dict[str, float]:
        """Dollars at risk per base asset across all open positions."""
        exposure: dict[str, float] = {}
        for pos in self.open_positions:
            base = _base_asset(pos.symbol)
            exposure[base] = exposure.get(base, 0.0) + pos.notional_risk
        return exposure

    @property
    def total_risk_pct(self) -> float:
        """Total notional risk as % of current equity; 0.0 if equity ≤ 0."""
        if self.equity <= 0:
            return 0.0
        total = sum(pos.notional_risk for pos in self.open_positions)
        return (total / self.equity) * 100


# ── Core functions ────────────────────────────────────────────────────

def adjusted_portfolio_risk(
    new_symbol: str,
    new_risk: float,
    portfolio: PortfolioState,
) -> tuple[float, dict]:
    """Compute the correlation-weighted risk contribution of the new trade.

    **Intentional linear approximation** — not a full variance-covariance
    marginal risk calculation.  The adjusted risk is simply:

        adj_risk = new_risk × 1.0                              (self)
                 + new_risk × corr(new_symbol, pos.symbol)     (each open pos)

    This treats ``new_risk`` as the unit of "stress" and scales it by the
    pairwise correlation with every existing position.  It does NOT:
      - weight contributions by each position's own notional_risk
      - compute the full portfolio variance (σ_new × ρ(new, portfolio))
      - use a covariance matrix (Σ) for rigorous marginal risk contribution

    The approximation deliberately favours simplicity and determinism over
    precision: it over-estimates exposure when existing positions are large
    and under-estimates when they are small.  For quick sizing decisions at
    the trade entry stage this is acceptable; a rigorous MRC would require
    the full Σ and is deferred until the position universe grows.

    Returns:
        adjusted_risk_dollars (float)  — correlation-weighted risk added
        details (dict)                 — per-symbol breakdown for logging
    """
    details: dict[str, dict] = {}
    correlated_risk = new_risk  # self-correlation = 1.0

    for pos in portfolio.open_positions:
        corr = get_correlation(new_symbol, pos.symbol)
        contribution = new_risk * corr
        correlated_risk += contribution
        base_pos = _base_asset(pos.symbol)
        details[base_pos] = {
            "notional_risk": round(pos.notional_risk, 4),
            "correlation": corr,
            "contribution": round(contribution, 4),
        }

    return correlated_risk, details


_CORRELATION_BLOCK_THRESHOLD: float = 0.70  # block same-direction if corr ≥ this


def _check_correlated_direction_conflict(
    new_symbol: str,
    new_direction: str,
    portfolio: PortfolioState,
) -> Optional[str]:
    """Return a block reason if an open position conflicts, else None.

    A conflict exists when:
      - An existing position is on a highly correlated pair (≥ threshold)
      - AND it is in the same direction as the proposed trade

    This prevents doubling up on essentially the same directional bet
    (e.g. long BTC + long ETH when corr=0.85).
    """
    if not new_direction:
        return None  # direction not provided — skip check
    base_new = _base_asset(new_symbol)
    for pos in portfolio.open_positions:
        base_existing = _base_asset(pos.symbol)
        if base_new == base_existing:
            return (
                f"duplicate_symbol ({new_symbol} already open)"
            )
        corr = get_correlation(new_symbol, pos.symbol)
        if corr >= _CORRELATION_BLOCK_THRESHOLD and pos.direction == new_direction:
            return (
                f"correlated_same_direction "
                f"({new_symbol}<->{pos.symbol} corr={corr:.2f}, "
                f"both {new_direction})"
            )
    return None


def _log_portfolio_decision(
    action: str,
    symbol: str,
    reason: str,
    open_count: int,
    portfolio_risk_pct: float,
) -> None:
    """Emit a structured portfolio decision log entry."""
    entry = {
        "event": "portfolio_decision",
        "action": action,
        "symbol": symbol,
        "reason": reason,
        "open_positions": open_count,
        "max_allowed": MAX_CONCURRENT_TRADES,
        "portfolio_risk_pct": round(portfolio_risk_pct, 4),
    }
    if action == "BLOCK":
        logger.warning("PORTFOLIO_BLOCK | %s", entry)
    else:
        logger.info("PORTFOLIO_ALLOW | %s", entry)


def can_open_trade(
    symbol: str,
    effective_risk: float,
    portfolio: PortfolioState,
    direction: str = "",
) -> dict:
    """Check whether the portfolio can absorb a new position.

    Enforces three gates in order:
      1. Max concurrent positions (MAX_CONCURRENT_TRADES)
      2. Correlated same-direction blocking (correlation ≥ 0.70)
      3. Correlation-weighted portfolio risk cap (MAX_PORTFOLIO_RISK_PCT)

    When the risk cap would be exceeded:
      - If enough headroom remains → scale position to fill it exactly
      - If scaling < 5% of intended size → hard block

    Args:
        symbol:         Trading pair to open (e.g. "BTCUSDT").
        effective_risk: Dollar risk AFTER DD multiplier has been applied.
                        The portfolio cap is evaluated against this value.
        portfolio:      Current state.  portfolio.equity must be current
                        before this call (callers are responsible for sync).
        direction:      "bullish" or "bearish" — used for correlated-pair
                        same-direction blocking.

    Returns dict:
        allowed (bool)
        adjusted_risk (float)           — dollar risk to use (after scaling)
        scaling_factor (float)          — 1.0 = full size; < 1.0 = scaled
        adjusted_portfolio_risk (float) — projected total risk % after trade
        reason (str)
        correlation_details (dict)      — per-symbol breakdown
    """
    if not USE_PORTFOLIO_LAYER:
        # Pass-through — zero overhead when flag is off.
        return {
            "allowed": True,
            "adjusted_risk": effective_risk,
            "scaling_factor": 1.0,
            "adjusted_portfolio_risk": portfolio.total_risk_pct,
            "reason": "portfolio_layer_disabled",
            "correlation_details": {},
        }

    if portfolio.equity <= 0:
        return {
            "allowed": False,
            "adjusted_risk": 0.0,
            "scaling_factor": 0.0,
            "adjusted_portfolio_risk": 0.0,
            "reason": "zero_equity",
            "correlation_details": {},
        }

    # ── Gate 1: max concurrent positions ──────────────────────────────
    if len(portfolio.open_positions) >= MAX_CONCURRENT_TRADES:
        _log_portfolio_decision(
            "BLOCK", symbol, "max_concurrent_reached",
            len(portfolio.open_positions), portfolio.total_risk_pct,
        )
        return {
            "allowed": False,
            "adjusted_risk": effective_risk,
            "scaling_factor": 0.0,
            "adjusted_portfolio_risk": portfolio.total_risk_pct,
            "reason": (
                f"max_concurrent_trades ({len(portfolio.open_positions)}"
                f"/{MAX_CONCURRENT_TRADES})"
            ),
            "correlation_details": {},
        }

    # ── Gate 2: correlated same-direction blocking ────────────────────
    # Block if an open position on a highly correlated pair (≥ 0.70)
    # is already open in the same direction — prevents doubling up on
    # essentially the same bet.
    corr_block = _check_correlated_direction_conflict(symbol, direction, portfolio)
    if corr_block is not None:
        _log_portfolio_decision(
            "BLOCK", symbol, corr_block,
            len(portfolio.open_positions), portfolio.total_risk_pct,
        )
        return {
            "allowed": False,
            "adjusted_risk": effective_risk,
            "scaling_factor": 0.0,
            "adjusted_portfolio_risk": portfolio.total_risk_pct,
            "reason": corr_block,
            "correlation_details": {},
        }

    adj_risk, details = adjusted_portfolio_risk(symbol, effective_risk, portfolio)
    current_risk_pct = portfolio.total_risk_pct
    adj_risk_pct = (adj_risk / portfolio.equity) * 100
    projected_total_pct = current_risk_pct + adj_risk_pct

    if projected_total_pct <= MAX_PORTFOLIO_RISK_PCT:
        _log_portfolio_decision(
            "OPEN", symbol, "within_limit",
            len(portfolio.open_positions), projected_total_pct,
        )
        return {
            "allowed": True,
            "adjusted_risk": effective_risk,
            "scaling_factor": 1.0,
            "adjusted_portfolio_risk": projected_total_pct,
            "reason": "within_limit",
            "correlation_details": details,
        }

    # Over the cap — compute remaining headroom.
    # Guard: adj_risk_pct == 0 means effective_risk is zero, so the new trade
    # adds no correlated exposure; allow it in full regardless of headroom.
    headroom_pct = max(0.0, MAX_PORTFOLIO_RISK_PCT - current_risk_pct)
    if adj_risk_pct <= 0:
        scaling_factor = 1.0  # zero-risk trade: no cap impact, always full size
    else:
        scaling_factor = min(1.0, headroom_pct / adj_risk_pct)

    if scaling_factor < 0.05:
        # Less than 5% of intended size — not worth placing; hard block
        logger.warning(
            "PORTFOLIO_BLOCK | symbol=%s | adj_risk_pct=%.2f%% | "
            "current=%.2f%% | max=%.2f%% | scaling=%.3f (< 5%% — blocked) | "
            "corr_details=%s",
            symbol, adj_risk_pct, current_risk_pct,
            MAX_PORTFOLIO_RISK_PCT, scaling_factor, details,
        )
        return {
            "allowed": False,
            "adjusted_risk": effective_risk,
            "scaling_factor": 0.0,
            "adjusted_portfolio_risk": projected_total_pct,
            "reason": (
                f"portfolio_risk_exceeded "
                f"({projected_total_pct:.2f}% > {MAX_PORTFOLIO_RISK_PCT}%)"
            ),
            "correlation_details": details,
        }

    scaled_risk = effective_risk * scaling_factor
    logger.info(
        "PORTFOLIO_SCALE | symbol=%s | scaling=%.3f | "
        "risk $%.2f → $%.2f | adj_risk_pct=%.2f%% | "
        "current=%.2f%% | max=%.2f%%",
        symbol, scaling_factor,
        effective_risk, scaled_risk,
        adj_risk_pct, current_risk_pct, MAX_PORTFOLIO_RISK_PCT,
    )
    return {
        "allowed": True,
        "adjusted_risk": scaled_risk,
        "scaling_factor": scaling_factor,
        "adjusted_portfolio_risk": MAX_PORTFOLIO_RISK_PCT,
        "reason": f"scaled_to_fit ({scaling_factor:.1%} of intended)",
        "correlation_details": details,
    }


def open_position(
    portfolio: PortfolioState,
    symbol: str,
    direction: str,
    notional_risk: float,
    entry_price: float,
    model: str,
    timeframe: str,
    opened_at: Optional[datetime] = None,
) -> PortfolioPosition:
    """Register a new open position in the portfolio.

    Must be called AFTER the order has been successfully placed —
    never speculatively before the trade is confirmed.

    Duplicate prevention: if the exact same symbol is already registered,
    a warning is logged and the new entry is NOT added (caller bug).
    Different correlated symbols (e.g. BTCUSDT + ETHUSDT) are allowed —
    the correlation check in can_open_trade() handles blocking those
    before we reach this point.
    """
    for existing in portfolio.open_positions:
        if existing.symbol == symbol:
            logger.warning(
                "PORTFOLIO_OPEN | symbol=%s — duplicate symbol detected "
                "(already open with risk=$%.2f); "
                "skipping registration to prevent double-counting",
                symbol, existing.notional_risk,
            )
            return existing  # return the existing position, do not append

    pos = PortfolioPosition(
        symbol=symbol,
        direction=direction,
        notional_risk=notional_risk,
        entry_price=entry_price,
        model=model,
        timeframe=timeframe,
        opened_at=opened_at or datetime.now(timezone.utc),
    )
    portfolio.open_positions.append(pos)
    portfolio.last_update_ts = datetime.now(timezone.utc)
    logger.debug(
        "PORTFOLIO_OPEN | symbol=%s dir=%s risk=$%.2f | "
        "n_positions=%d total_risk_pct=%.2f%%",
        symbol, direction, notional_risk,
        len(portfolio.open_positions), portfolio.total_risk_pct,
    )
    return pos


def close_position(
    portfolio: PortfolioState,
    symbol: str,
) -> bool:
    """Deregister all open positions matching symbol's base asset.

    Removes every entry whose base asset matches, not just the first.
    Under normal operation there is exactly one match; removing all
    ensures the portfolio stays clean even if open_position was somehow
    called twice (the duplicate warning in open_position guards against
    that, but close_position is the safety net).

    Does NOT update portfolio.equity — callers are responsible for
    keeping it current (sync from source-of-truth state before each
    can_open_trade() call).

    Returns True if at least one matching position was found and removed.
    """
    base = _base_asset(symbol)
    before = len(portfolio.open_positions)
    portfolio.open_positions = [
        pos for pos in portfolio.open_positions
        if _base_asset(pos.symbol) != base
    ]
    removed = before - len(portfolio.open_positions)

    if removed == 0:
        logger.warning(
            "PORTFOLIO_CLOSE | symbol=%s — no matching open position found",
            symbol,
        )
        return False

    if removed > 1:
        logger.warning(
            "PORTFOLIO_CLOSE | symbol=%s — removed %d duplicate entries "
            "(expected 1); check for double open_position calls",
            symbol, removed,
        )

    portfolio.last_update_ts = datetime.now(timezone.utc)
    logger.debug(
        "PORTFOLIO_CLOSE | symbol=%s removed=%d | "
        "n_positions=%d total_risk_pct=%.2f%%",
        symbol, removed,
        len(portfolio.open_positions),
        portfolio.total_risk_pct,
    )
    return True


def debug_snapshot(portfolio: PortfolioState) -> dict:
    """Return a serialisable snapshot of the portfolio state for logging."""
    return {
        "equity": round(portfolio.equity, 2),
        "peak_equity": round(portfolio.peak_equity, 2),
        "total_risk_pct": round(portfolio.total_risk_pct, 4),
        "max_portfolio_risk_pct": MAX_PORTFOLIO_RISK_PCT,
        "max_concurrent_trades": MAX_CONCURRENT_TRADES,
        "use_portfolio_layer": USE_PORTFOLIO_LAYER,
        "n_open_positions": len(portfolio.open_positions),
        "open_positions": [
            {
                "symbol": p.symbol,
                "direction": p.direction,
                "notional_risk": round(p.notional_risk, 4),
                "entry_price": p.entry_price,
                "model": p.model,
                "timeframe": p.timeframe,
                "opened_at": p.opened_at.isoformat() if p.opened_at else None,
            }
            for p in portfolio.open_positions
        ],
        "symbol_exposure": {
            k: round(v, 4) for k, v in portfolio.symbol_exposure.items()
        },
        "last_update_ts": (
            portfolio.last_update_ts.isoformat()
            if portfolio.last_update_ts
            else None
        ),
    }
