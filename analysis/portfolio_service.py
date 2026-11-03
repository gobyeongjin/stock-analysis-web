from datetime import date
from portfolio_db import get_connection, init_db


def add_trade(ticker, stock_name, trade_type,
              trade_date, price, quantity,
              fee=0, tax=0, memo=""):

    init_db()

    ticker = ticker.strip().upper()
    stock_name = stock_name.strip()
    trade_type = trade_type.strip().upper()

    if not ticker or not stock_name:
        raise ValueError("종목코드와 종목명은 필수입니다.")

    if trade_type not in ("BUY", "SELL"):
        raise ValueError("BUY 또는 SELL만 가능합니다.")

    date.fromisoformat(trade_date)

    price = float(price)
    quantity = int(quantity)
    fee = float(fee)
    tax = float(tax)

    if price <= 0 or quantity <= 0:
        raise ValueError("가격과 수량은 0보다 커야 합니다.")

    if fee < 0 or tax < 0:
        raise ValueError("수수료와 세금은 음수일 수 없습니다.")

    with get_connection() as conn:
        # 매도 시 현재 보유 수량 확인
        if trade_type == "SELL":
            row = conn.execute("""
                SELECT COALESCE(SUM(
                    CASE
                        WHEN trade_type = 'BUY' THEN quantity
                        ELSE -quantity
                    END
                ), 0) AS shares
                FROM trades
                WHERE ticker = ?
            """, (ticker,)).fetchone()

            if quantity > row["shares"]:
                raise ValueError(
                    f"보유 수량 부족: {row['shares']}주"
                )

        cursor = conn.execute("""
            INSERT INTO trades (
                ticker, stock_name, trade_type,
                trade_date, price, quantity,
                fee, tax, memo
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            ticker, stock_name, trade_type,
            trade_date, price, quantity,
            fee, tax, memo
        ))

        return cursor.lastrowid


def get_trades():
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT *
            FROM trades
            ORDER BY trade_date DESC, id DESC
        """).fetchall()

    return [dict(row) for row in rows]


def get_holdings():
    holdings = {}

    # 거래 발생 순서대로 계산
    for trade in sorted(
        get_trades(),
        key=lambda x: (x["trade_date"], x["id"])
    ):
        ticker = trade["ticker"]

        if ticker not in holdings:
            holdings[ticker] = {
                "ticker": ticker,
                "stock_name": trade["stock_name"],
                "quantity": 0,
                "cost_basis": 0.0
            }

        h = holdings[ticker]
        qty = trade["quantity"]
        price = trade["price"]

        if trade["trade_type"] == "BUY":
            h["cost_basis"] += (
                price * qty + trade["fee"] + trade["tax"]
            )
            h["quantity"] += qty

        else:
            if qty > h["quantity"]:
                raise ValueError(
                    f"{ticker}: 거래일 기준 보유 수량 부족"
                )

            avg_cost = (
                h["cost_basis"] / h["quantity"]
            )

            h["cost_basis"] -= avg_cost * qty
            h["quantity"] -= qty

            if h["quantity"] == 0:
                h["cost_basis"] = 0.0

    result = []

    for h in holdings.values():
        if h["quantity"] > 0:
            h["avg_price"] = round(
                h["cost_basis"] / h["quantity"], 2
            )
            h["cost_basis"] = round(
                h["cost_basis"], 2
            )
            result.append(h)

    return result
