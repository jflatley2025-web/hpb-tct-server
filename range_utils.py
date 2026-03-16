"""
range_utils.py — Shared utility functions for range detection logic.

Consolidates duplicated helpers used across tct_schematics.py,
server_mexc.py, and related modules.
"""

import pandas as pd


def check_equilibrium_touch(
    candles: pd.DataFrame,
    idx1: int,
    idx2: int,
    equilibrium: float,
    *,
    check_between: bool = True,
    post_range_candles: int = 30,
) -> bool:
    """
    Check if price touched equilibrium to confirm a range.

    TCT: "When we have a move back to the equilibrium, that's when
    the range is confirmed."

    Parameters
    ----------
    candles : pd.DataFrame
        OHLC candle data with 'high' and 'low' columns.
    idx1, idx2 : int
        Indices of the two range pivots (order does not matter).
    equilibrium : float
        The equilibrium (midpoint) price of the range.
    check_between : bool
        If True, also check candles *between* the two pivots
        (during range formation).  Default True.
    post_range_candles : int
        Number of candles after the later pivot to check for an
        equilibrium touch.  Default 30.

    Returns
    -------
    bool
        True if any candle's high-low range includes the equilibrium.
    """
    start = min(idx1, idx2)
    end = max(idx1, idx2)

    # Check between the two pivots (during range formation)
    if check_between:
        for i in range(start + 1, end):
            candle = candles.iloc[i]
            if candle["low"] <= equilibrium <= candle["high"]:
                return True

    # Check candles after the range formation (confirmation bounce)
    check_start = end + 1
    check_end = min(check_start + post_range_candles, len(candles))

    if check_start >= len(candles):
        return False

    for i in range(check_start, check_end):
        candle = candles.iloc[i]
        if candle["low"] <= equilibrium <= candle["high"]:
            return True

    return False
