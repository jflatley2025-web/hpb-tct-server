"""
trade_execution.py — Trade Execution Engine (TCT Lecture 9)
Author: HPB-TCT Dev Team
Date: 2026-01-30

Implements TCT Lecture 9 trade execution logic:

Exchange Settings:
- Always use ISOLATED margin mode (never Cross)
- Order by quantity in USDT
- Market orders 95% of the time (for BOS entries)
- Limit orders for partial take profits

Leverage Understanding:
- Leverage only changes the ratio between margin and total position size
- Same position size = same risk regardless of leverage
- Key rule: Liquidation price must ALWAYS be outside stop-loss zone
- Example: $25K pos with 25x = $1K margin, 50x = $500, 200x = $125
- Risk stays constant as long as liquidation is outside SL

Trade Management:
- Set SL based on TCT invalidation (above/below schematic tap)
- Partial take profits via limit close orders
- Trail SL to breakeven when profitable, then increase leverage + reduce margin
- Add to position when SL trails and risk diminishes

Capital Management:
- Only put 30-50% of total capital on exchange
- Spread across wallets/exchanges for safety
"""

from datetime import datetime
from typing import Dict, List, Optional
import logging

logger = logging.getLogger("Trade-Execution")


def calculate_position_size(risk_amount: float, sl_pct: float) -> float:
    """
    TCT Lecture 7+9 formula: Position Size = (Risk Amount / SL%) × 100

    Args:
        risk_amount: Dollar amount willing to risk
        sl_pct: Stop-loss size as percentage of position

    Returns:
        Total position size in USD
    """
    if sl_pct <= 0:
        raise ValueError("Stop-loss percentage must be greater than 0")
    return (risk_amount / sl_pct) * 100


def calculate_margin(position_size: float, leverage: float) -> float:
    """
    Margin = Position Size / Leverage

    Args:
        position_size: Total position size in USD
        leverage: Leverage multiplier

    Returns:
        Required margin in USD
    """
    if leverage <= 0:
        raise ValueError("Leverage must be greater than 0")
    return position_size / leverage


def calculate_liquidation_price(
    entry_price: float, leverage: float, direction: str,
    maintenance_margin_rate: float = 0.005
) -> float:
    """
    Estimate liquidation price for isolated margin.

    For LONG: liq_price = entry * (1 - 1/leverage + maintenance_margin_rate)
    For SHORT: liq_price = entry * (1 + 1/leverage - maintenance_margin_rate)

    Args:
        entry_price: Entry price
        leverage: Leverage used
        direction: "long" or "short"
        maintenance_margin_rate: Exchange maintenance margin rate (default 0.5%)

    Returns:
        Estimated liquidation price
    """
    if leverage <= 0:
        raise ValueError("Leverage must be greater than 0")

    if direction == "long":
        return entry_price * (1 - (1 / leverage) + maintenance_margin_rate)
    elif direction == "short":
        return entry_price * (1 + (1 / leverage) - maintenance_margin_rate)
    else:
        raise ValueError("Direction must be 'long' or 'short'")


def check_liquidation_safety(
    liquidation_price: float, stop_loss_price: float,
    entry_price: float, direction: str
) -> Dict:
    """
    TCT Lecture 9 KEY RULE: Liquidation price must be outside stop-loss zone.

    For LONG: liquidation must be BELOW stop-loss
    For SHORT: liquidation must be ABOVE stop-loss

    Args:
        liquidation_price: Estimated liquidation price
        stop_loss_price: Stop-loss price
        entry_price: Entry price
        direction: "long" or "short"

    Returns:
        Dict with safety status, gap, and recommendation
    """
    if direction == "long":
        is_safe = liquidation_price < stop_loss_price
        gap = stop_loss_price - liquidation_price
        gap_pct = (gap / entry_price) * 100 if entry_price > 0 else 0
    elif direction == "short":
        is_safe = liquidation_price > stop_loss_price
        gap = liquidation_price - stop_loss_price
        gap_pct = (gap / entry_price) * 100 if entry_price > 0 else 0
    else:
        raise ValueError("Direction must be 'long' or 'short'")

    if is_safe:
        if gap_pct > 2:
            recommendation = "Safe — large buffer between SL and liquidation"
        elif gap_pct > 0.5:
            recommendation = "Safe — adequate buffer, but consider lower leverage"
        else:
            recommendation = "Caution — very tight buffer, reduce leverage"
    else:
        recommendation = "DANGER — Liquidation before stop-loss! Reduce leverage immediately"

    return {
        "is_safe": is_safe,
        "liquidation_price": round(liquidation_price, 2),
        "stop_loss_price": round(stop_loss_price, 2),
        "gap": round(gap, 2),
        "gap_pct": round(gap_pct, 4),
        "recommendation": recommendation
    }


def calculate_leverage_comparison(
    position_size: float, entry_price: float,
    stop_loss_price: float, direction: str,
    leverage_options: List[float] = None
) -> List[Dict]:
    """
    Compare different leverage options for the same position size.
    TCT Lecture 9: Leverage only changes margin, not risk.

    Returns comparison showing margin, liquidation price, and safety for each leverage.
    """
    if leverage_options is None:
        leverage_options = [5, 10, 25, 50, 100, 200]

    results = []
    for lev in leverage_options:
        margin = calculate_margin(position_size, lev)
        liq_price = calculate_liquidation_price(entry_price, lev, direction)
        safety = check_liquidation_safety(liq_price, stop_loss_price, entry_price, direction)

        results.append({
            "leverage": lev,
            "margin": round(margin, 2),
            "liquidation_price": round(liq_price, 2),
            "is_safe": safety["is_safe"],
            "gap_to_sl": round(safety["gap"], 2),
            "gap_pct": safety["gap_pct"],
            "recommendation": safety["recommendation"]
        })

    return results


def calculate_partial_take_profits(
    entry_price: float, position_size: float,
    direction: str, tp_levels: List[Dict] = None
) -> Dict:
    """
    TCT Lecture 9: Partial take profits using limit close orders.

    Args:
        entry_price: Entry price
        position_size: Total position size
        direction: "long" or "short"
        tp_levels: List of {"price": float, "pct": float} for each TP level

    Returns:
        Dict with partial TP breakdown and total expected profit
    """
    if tp_levels is None:
        # Default: 50% at TP1, 50% at TP2
        if direction == "long":
            tp_levels = [
                {"price": entry_price * 1.02, "pct": 50},
                {"price": entry_price * 1.04, "pct": 50}
            ]
        else:
            tp_levels = [
                {"price": entry_price * 0.98, "pct": 50},
                {"price": entry_price * 0.96, "pct": 50}
            ]

    total_pct = sum(tp["pct"] for tp in tp_levels)
    if total_pct > 100:
        raise ValueError("Total TP percentages cannot exceed 100%")

    results = []
    total_profit = 0
    remaining_position = position_size

    for tp in tp_levels:
        tp_price = tp["price"]
        close_pct = tp["pct"]
        close_amount = position_size * (close_pct / 100)

        if direction == "long":
            price_move_pct = ((tp_price - entry_price) / entry_price) * 100
            profit = close_amount * (price_move_pct / 100)
        else:
            price_move_pct = ((entry_price - tp_price) / entry_price) * 100
            profit = close_amount * (price_move_pct / 100)

        total_profit += profit
        remaining_position -= close_amount

        results.append({
            "price": round(tp_price, 2),
            "close_pct": close_pct,
            "close_amount": round(close_amount, 2),
            "price_move_pct": round(price_move_pct, 4),
            "profit": round(profit, 2),
            "remaining_position": round(remaining_position, 2)
        })

    return {
        "tp_levels": results,
        "total_profit": round(total_profit, 2),
        "remaining_position": round(remaining_position, 2)
    }


def calculate_trailing_sl_scenario(
    entry_price: float, position_size: float,
    original_sl_price: float, new_sl_price: float,
    direction: str, risk_amount: float,
    account_balance: float
) -> Dict:
    """
    TCT Lecture 9: When SL trails and risk diminishes, can add new position.

    Calculates:
    - New risk after trailing SL
    - How much position can be added to maintain original risk
    - Break-even scenario
    """
    if direction == "long":
        original_risk_pct = ((entry_price - original_sl_price) / entry_price) * 100
        new_risk_pct = ((entry_price - new_sl_price) / entry_price) * 100
        original_risk_dollar = position_size * (original_risk_pct / 100)
        new_risk_dollar = position_size * (new_risk_pct / 100)
        is_breakeven = new_sl_price >= entry_price
    else:
        original_risk_pct = ((original_sl_price - entry_price) / entry_price) * 100
        new_risk_pct = ((new_sl_price - entry_price) / entry_price) * 100
        original_risk_dollar = position_size * (original_risk_pct / 100)
        new_risk_dollar = position_size * (new_risk_pct / 100)
        is_breakeven = new_sl_price <= entry_price

    risk_freed = max(0, original_risk_dollar - new_risk_dollar)

    # How much additional position can be opened with freed risk
    if new_risk_pct > 0:
        additional_position = calculate_position_size(risk_freed, abs(new_risk_pct))
    else:
        additional_position = 0

    return {
        "original_risk": round(original_risk_dollar, 2),
        "new_risk": round(max(0, new_risk_dollar), 2),
        "risk_freed": round(risk_freed, 2),
        "is_breakeven": is_breakeven,
        "additional_position_possible": round(additional_position, 2),
        "new_total_position": round(position_size + additional_position, 2),
        "trailing_sl": round(new_sl_price, 2),
        "margin_can_reduce": is_breakeven
    }


def calculate_capital_allocation(
    total_capital: float, exchange_pct: float = 50
) -> Dict:
    """
    TCT Lecture 9: Don't put all money on exchange.
    Recommended: 30-50% on exchange.

    Args:
        total_capital: Total trading capital
        exchange_pct: Percentage to put on exchange (30-50% recommended)
    """
    exchange_pct = max(10, min(100, exchange_pct))
    on_exchange = total_capital * (exchange_pct / 100)
    off_exchange = total_capital - on_exchange

    return {
        "total_capital": round(total_capital, 2),
        "on_exchange": round(on_exchange, 2),
        "off_exchange": round(off_exchange, 2),
        "exchange_pct": exchange_pct,
        "recommendation": (
            "Good — conservative allocation"
            if exchange_pct <= 50
            else "Caution — consider reducing exchange allocation to 30-50%"
        )
    }


def generate_execution_plan(
    account_balance: float,
    risk_pct: float,
    entry_price: float,
    stop_loss_price: float,
    take_profit_price: float,
    direction: str,
    leverage: float = 10,
    tp2_price: float = None
) -> Dict:
    """
    Generate a complete trade execution plan following TCT Lecture 9.

    Combines position sizing, leverage analysis, liquidation safety,
    partial TPs, and capital management into one comprehensive plan.
    """
    try:
        # Step 1: Calculate risk
        risk_amount = account_balance * (risk_pct / 100)

        if direction == "long":
            sl_pct = ((entry_price - stop_loss_price) / entry_price) * 100
        else:
            sl_pct = ((stop_loss_price - entry_price) / entry_price) * 100

        if sl_pct <= 0:
            return {"error": "Invalid stop-loss: must be on the opposite side of entry from target"}

        # Step 2: Position size
        position_size = calculate_position_size(risk_amount, sl_pct)

        # Step 3: Margin and liquidation
        margin = calculate_margin(position_size, leverage)
        liq_price = calculate_liquidation_price(entry_price, leverage, direction)

        # Step 4: Liquidation safety check
        safety = check_liquidation_safety(liq_price, stop_loss_price, entry_price, direction)

        # Step 5: Find max safe leverage
        max_safe_leverage = find_max_safe_leverage(
            position_size, entry_price, stop_loss_price, direction
        )

        # Step 6: Leverage comparison
        lev_comparison = calculate_leverage_comparison(
            position_size, entry_price, stop_loss_price, direction
        )

        # Step 7: R:R calculation
        if direction == "long":
            risk_distance = entry_price - stop_loss_price
            reward_distance = take_profit_price - entry_price
        else:
            risk_distance = stop_loss_price - entry_price
            reward_distance = entry_price - take_profit_price

        rr_ratio = reward_distance / risk_distance if risk_distance > 0 else 0

        # Step 8: Profit at TP
        if direction == "long":
            tp_move_pct = ((take_profit_price - entry_price) / entry_price) * 100
        else:
            tp_move_pct = ((entry_price - take_profit_price) / entry_price) * 100

        profit_at_tp = position_size * (tp_move_pct / 100)

        # Step 9: Partial TPs
        tp_levels = [{"price": take_profit_price, "pct": 50}]
        if tp2_price:
            tp_levels.append({"price": tp2_price, "pct": 50})
        else:
            # Default TP2 at midpoint
            if direction == "long":
                mid_tp = entry_price + (take_profit_price - entry_price) * 0.5
            else:
                mid_tp = entry_price - (entry_price - take_profit_price) * 0.5
            tp_levels = [
                {"price": mid_tp, "pct": 50},
                {"price": take_profit_price, "pct": 50}
            ]

        partial_tps = calculate_partial_take_profits(
            entry_price, position_size, direction, tp_levels
        )

        # Step 10: Capital allocation
        capital = calculate_capital_allocation(account_balance)

        return {
            "execution_plan": {
                "direction": direction.upper(),
                "order_type": "MARKET (95% — enter on BOS candle close)",
                "margin_mode": "ISOLATED (always)",
                "quantity_mode": "Order by Quantity (USDT)"
            },
            "position_sizing": {
                "account_balance": round(account_balance, 2),
                "risk_pct": risk_pct,
                "risk_amount": round(risk_amount, 2),
                "stop_loss_pct": round(sl_pct, 4),
                "position_size": round(position_size, 2),
                "entry_price": round(entry_price, 2),
                "stop_loss_price": round(stop_loss_price, 2),
                "take_profit_price": round(take_profit_price, 2)
            },
            "leverage_analysis": {
                "selected_leverage": leverage,
                "margin_required": round(margin, 2),
                "liquidation_price": round(liq_price, 2),
                "safety": safety,
                "max_safe_leverage": max_safe_leverage,
                "comparison": lev_comparison
            },
            "trade_outcome": {
                "risk_reward": round(rr_ratio, 2),
                "loss_at_sl": round(risk_amount, 2),
                "profit_at_tp": round(profit_at_tp, 2),
                "tp_account_gain_pct": round((profit_at_tp / account_balance) * 100, 2)
            },
            "partial_take_profits": partial_tps,
            "capital_management": capital,
            "execution_checklist": [
                "Set margin mode to ISOLATED",
                f"Set leverage to {leverage}x (max safe: {max_safe_leverage}x)",
                f"Verify liquidation price ({round(liq_price, 2)}) is {'below' if direction == 'long' else 'above'} SL ({round(stop_loss_price, 2)})",
                "Order by Quantity (USDT)",
                f"Enter {direction.upper()} with {round(position_size, 2)} USDT",
                f"Set stop-loss at {round(stop_loss_price, 2)}",
                f"Set TP1 at {round(tp_levels[0]['price'], 2)} ({tp_levels[0]['pct']}%)",
                f"Set TP2 at {round(tp_levels[1]['price'], 2)} ({tp_levels[1]['pct']}%)" if len(tp_levels) > 1 else "",
                "When profitable: trail SL to breakeven, increase leverage, reduce margin"
            ],
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Execution plan error: {e}")
        return {"error": str(e)}


def find_max_safe_leverage(
    position_size: float, entry_price: float,
    stop_loss_price: float, direction: str
) -> int:
    """Find the maximum leverage where liquidation stays outside stop-loss."""
    for lev in range(1, 401):
        liq = calculate_liquidation_price(entry_price, lev, direction)
        if direction == "long":
            if liq >= stop_loss_price:
                return max(1, lev - 1)
        else:
            if liq <= stop_loss_price:
                return max(1, lev - 1)
    return 400
