"""
broker_charges.py
-----------------
Calculates Zerodha's actual brokerage and statutory charges for trades.

Even though we use Angel One for data, we use Zerodha's brokerage
structure for realistic cost simulation in backtesting.

Zerodha Charges Reference (last verified: 2026):
- Intraday Equity: 0.03% or ₹20/order (whichever is lower)
- Delivery Equity: Free (zero brokerage)
- F&O Futures: 0.03% or ₹20/order (whichever is lower)
- F&O Options: Flat ₹20 per executed order

IMPORTANT: F&O STT rates changed April 1, 2026 (Budget 2026-27).
Verify current rates at https://zerodha.com/charges before relying on F&O backtests.
- Futures STT (sell): currently 0.01% — verify
- Options STT (sell): may be 0.0625% post-change — verify
- Options exchange txn charge: verify against current Zerodha page

Statutory charges (common across segments):
- STT (Securities Transaction Tax)
- Exchange Transaction Charges
- GST (18% on brokerage + exchange charges + SEBI)
- SEBI Turnover Fees (₹10 per crore = 0.0001%)
- Stamp Duty
"""

from typing import Dict, Tuple
import config


def calculate_charges(
    order_type: str,
    buy_price: float,
    sell_price: float,
    quantity: int,
    segment: str = "intraday_equity",
) -> Dict[str, float]:
    """
    Calculate total brokerage and statutory charges for a round-trip trade.

    Parameters
    ----------
    order_type : str
        "buy" or "sell" — the initiating order direction.
    buy_price : float
        Price at which the stock/contract was bought.
    sell_price : float
        Price at which the stock/contract was sold.
    quantity : int
        Number of shares/contracts.
    segment : str
        One of: "intraday_equity", "delivery_equity", "futures", "options".

    Returns
    -------
    dict
        Keys: brokerage, stt, exchange_charges, gst, sebi_charges,
              stamp_duty, total_charges, gross_pnl, net_pnl
    """
    if segment not in config.CHARGES_CONFIG:
        raise ValueError(
            f"Unknown segment '{segment}'. "
            f"Valid options: {list(config.CHARGES_CONFIG.keys())}"
        )

    charges = config.CHARGES_CONFIG[segment]
    buy_value = buy_price * quantity
    sell_value = sell_price * quantity
    turnover = buy_value + sell_value

    # ---- Brokerage ----
    if segment == "options":
        # Flat ₹20 per executed order (2 orders: buy + sell)
        brokerage = charges["brokerage_cap"] * 2
    elif segment == "delivery_equity":
        brokerage = 0.0  # Free delivery
    else:
        # 0.03% of trade value, capped at ₹20 per order
        buy_brokerage = min(buy_value * charges["brokerage_rate"], charges["brokerage_cap"])
        sell_brokerage = min(sell_value * charges["brokerage_rate"], charges["brokerage_cap"])
        brokerage = buy_brokerage + sell_brokerage

    # ---- STT (Securities Transaction Tax) ----
    stt_buy = buy_value * charges["stt_buy"]
    stt_sell = sell_value * charges["stt_sell"]
    stt = stt_buy + stt_sell

    # ---- Exchange Transaction Charges ----
    exchange_charges = turnover * charges["exchange_txn"]

    # ---- SEBI Turnover Fees ----
    sebi_charges = turnover * charges["sebi_rate"]

    # ---- GST (18% on brokerage + exchange charges + SEBI) ----
    gst = (brokerage + exchange_charges + sebi_charges) * charges["gst_rate"]

    # ---- Stamp Duty ----
    stamp_duty_buy = buy_value * charges["stamp_duty_buy"]
    stamp_duty_sell = sell_value * charges["stamp_duty_sell"]
    stamp_duty = stamp_duty_buy + stamp_duty_sell

    # ---- Totals ----
    total_charges = brokerage + stt + exchange_charges + gst + sebi_charges + stamp_duty

    # ---- P&L ----
    # buy_price and sell_price are already correctly assigned by the caller
    # (buy_price = entry for long, sell_price = entry for short, etc.)
    gross_pnl = (sell_price - buy_price) * quantity

    net_pnl = gross_pnl - total_charges

    return {
        "brokerage": round(brokerage, 2),
        "stt": round(stt, 2),
        "exchange_charges": round(exchange_charges, 2),
        "gst": round(gst, 2),
        "sebi_charges": round(sebi_charges, 2),
        "stamp_duty": round(stamp_duty, 2),
        "total_charges": round(total_charges, 2),
        "gross_pnl": round(gross_pnl, 2),
        "net_pnl": round(net_pnl, 2),
    }


def estimate_entry_charges(
    price: float,
    quantity: int,
    segment: str = "intraday_equity",
) -> float:
    """
    Estimate charges for an entry order only (used for position sizing).

    Parameters
    ----------
    price : float
        Entry price.
    quantity : int
        Number of shares/contracts.
    segment : str
        Trade segment.

    Returns
    -------
    float
        Estimated charges for the entry leg.
    """
    if segment not in config.CHARGES_CONFIG:
        raise ValueError(f"Unknown segment '{segment}'.")

    charges = config.CHARGES_CONFIG[segment]
    trade_value = price * quantity

    # Brokerage (entry leg only)
    if segment == "options":
        brokerage = charges["brokerage_cap"]
    elif segment == "delivery_equity":
        brokerage = 0.0
    else:
        brokerage = min(trade_value * charges["brokerage_rate"], charges["brokerage_cap"])

    stt = trade_value * charges["stt_buy"]
    exchange = trade_value * charges["exchange_txn"]
    sebi = trade_value * charges["sebi_rate"]
    gst = (brokerage + exchange + sebi) * charges["gst_rate"]
    stamp = trade_value * charges["stamp_duty_buy"]

    return round(brokerage + stt + exchange + gst + sebi + stamp, 2)


# ---- Quick test when run directly ----
if __name__ == "__main__":
    # Example: Buy 100 shares at ₹500, sell at ₹520 (intraday)
    result = calculate_charges("buy", 500, 520, 100, "intraday_equity")
    print("=== Intraday Equity Example ===")
    for k, v in result.items():
        print(f"  {k}: ₹{v:,.2f}")

    print()

    # Example: Buy 100 shares at ₹500, sell at ₹520 (delivery)
    result = calculate_charges("buy", 500, 520, 100, "delivery_equity")
    print("=== Delivery Equity Example ===")
    for k, v in result.items():
        print(f"  {k}: ₹{v:,.2f}")

    print()

    # Example: Options - Buy 1 lot (50 qty) at ₹100, sell at ₹110
    result = calculate_charges("buy", 100, 110, 50, "options")
    print("=== Options Example ===")
    for k, v in result.items():
        print(f"  {k}: ₹{v:,.2f}")

