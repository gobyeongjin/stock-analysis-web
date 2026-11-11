from datetime import date
from multiuser_db import get_connection, init_db


def get_trades(user_id):
    init_db()

    with get_connection() as conn:
        rows = conn.execute("""
            SELECT id, ticker, stock_name, trade_type,
                   trade_date, price, quantity, fee, tax,
                   memo, created_at
            FROM trades
            WHERE user_id = ?
            ORDER BY trade_date DESC, id DESC
        """, (user_id,)).fetchall()

    return [dict(row) for row in rows]


def calculate_holdings(trades):
    holdings = {}

    for trade in sorted(
        trades,
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
            h["cost_basis"] += price * qty + trade["fee"] + trade["tax"]
            h["quantity"] += qty
        else:
            if qty > h["quantity"]:
                raise ValueError(f"{ticker}: 보유 수량 부족")

            avg_cost = h["cost_basis"] / h["quantity"]
            h["cost_basis"] -= avg_cost * qty
            h["quantity"] -= qty

            if h["quantity"] == 0:
                h["cost_basis"] = 0.0

    result = []

    for h in holdings.values():
        if h["quantity"] > 0:
            h["avg_price"] = round(h["cost_basis"] / h["quantity"], 2)
            h["cost_basis"] = round(h["cost_basis"], 2)
            result.append(h)

    return result


def get_holdings(user_id):
    return calculate_holdings(get_trades(user_id))


def add_trade(
    user_id, ticker, stock_name, trade_type,
    trade_date, price, quantity,
    fee=0, tax=0, memo=""
):
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
        # 검증과 저장 사이 다른 요청이 끼어들지 않도록 잠금
        conn.execute("BEGIN IMMEDIATE")

        rows = conn.execute("""
            SELECT id, ticker, stock_name, trade_type,
                   trade_date, price, quantity, fee, tax
            FROM trades
            WHERE user_id = ?
            ORDER BY trade_date, id
        """, (user_id,)).fetchall()

        existing = [dict(row) for row in rows]

        proposed = {
            "id": max([r["id"] for r in existing], default=0) + 1,
            "ticker": ticker,
            "stock_name": stock_name,
            "trade_type": trade_type,
            "trade_date": trade_date,
            "price": price,
            "quantity": quantity,
            "fee": fee,
            "tax": tax
        }

        # 보유 수량과 현금 잔액 모두 검증
        calculate_holdings(existing + [proposed])
        validate_cash_limit(existing + [proposed])

        cursor = conn.execute("""
            INSERT INTO trades (
                user_id, ticker, stock_name, trade_type,
                trade_date, price, quantity, fee, tax, memo
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            user_id, ticker, stock_name, trade_type,
            trade_date, price, quantity, fee, tax, memo
        ))

        return cursor.lastrowid



INITIAL_CAPITAL = 20_000_000


def calculate_cash_balance(trades):
    cash = float(INITIAL_CAPITAL)

    for trade in trades:
        amount = float(trade["price"]) * int(trade["quantity"])
        fee = float(trade["fee"])
        tax = float(trade["tax"])

        if trade["trade_type"] == "BUY":
            cash -= amount + fee + tax
        elif trade["trade_type"] == "SELL":
            cash += amount - fee - tax

    return round(cash, 2)


def get_cash_balance(user_id):
    trades = get_trades(user_id)
    return calculate_cash_balance(trades)



def validate_cash_limit(trades):
    """
    거래 날짜순으로 현금 잔액을 계산하고
    초기자본을 초과하는 매수를 차단한다.
    """
    cash = float(INITIAL_CAPITAL)

    ordered = sorted(
        trades,
        key=lambda x: (x["trade_date"], x["id"])
    )

    for trade in ordered:
        amount = float(trade["price"]) * int(trade["quantity"])
        fee = float(trade["fee"])
        tax = float(trade["tax"])

        if trade["trade_type"] == "BUY":
            required = amount + fee + tax

            if required > cash + 0.000001:
                raise ValueError(
                    f"매수 잔액 부족: 필요 {required:,.0f}원, "
                    f"보유 {cash:,.0f}원"
                )

            cash -= required

        elif trade["trade_type"] == "SELL":
            cash += amount - fee - tax

    return round(cash, 2)
