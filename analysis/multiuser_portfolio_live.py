from datetime import datetime

from fastapi import APIRouter, Depends

from multiuser_auth_api import get_current_user
from multiuser_portfolio_service import get_holdings, get_cash_balance, INITIAL_CAPITAL
from portfolio_live import get_prices

router = APIRouter(
    prefix="/api/my-portfolio",
    tags=["My Portfolio"],
)


@router.get("/live")
def get_my_live_portfolio(user=Depends(get_current_user)):
    # 로그인한 사용자의 보유 종목만 조회
    holdings = get_holdings(user["id"])

    symbols = list(dict.fromkeys(
        str(stock["ticker"]).split(".")[0].zfill(6)
        for stock in holdings
    ))

    prices = get_prices(symbols) if symbols else {}

    result = []
    total_cost = 0.0
    total_value = 0.0
    missing_count = 0

    for stock in holdings:
        symbol = str(stock["ticker"]).split(".")[0].zfill(6)
        quote = prices.get(symbol)

        item = dict(stock)

        if quote is None:
            item.update({
                "current_price": None,
                "price_date": None,
                "market_value": None,
                "unrealized_pnl": None,
                "return_pct": None,
            })
            missing_count += 1
        else:
            current_price = quote["price"]
            cost = float(stock["cost_basis"])
            market_value = current_price * stock["quantity"]
            pnl = market_value - cost

            item.update({
                "current_price": current_price,
                "price_date": quote["timestamp"],
                "market_value": round(market_value, 2),
                "unrealized_pnl": round(pnl, 2),
                "return_pct": round(pnl / cost * 100, 2)
                    if cost > 0 else None,
            })

            total_cost += cost
            total_value += market_value

        result.append(item)

    cash_balance = get_cash_balance(user["id"])
    complete = missing_count == 0
    pnl = total_value - total_cost

    return {
        "holdings": result,
        "summary": {
            "initial_capital": INITIAL_CAPITAL,
            "cash_balance": cash_balance,
            "total_equity": round(cash_balance + total_value, 2)
                if complete else None,
            "total_profit": round(
                cash_balance + total_value - INITIAL_CAPITAL, 2
            ) if complete else None,
            "overall_return_pct": round(
                (cash_balance + total_value - INITIAL_CAPITAL)
                / INITIAL_CAPITAL * 100, 2
            ) if complete else None,
            "stock_count": len(result),
            "missing_count": missing_count,
            "total_cost": round(total_cost, 2) if complete else None,
            "total_value": round(total_value, 2) if complete else None,
            "total_pnl": round(pnl, 2) if complete else None,
            "total_return_pct": round(pnl / total_cost * 100, 2)
                if complete and total_cost > 0 else None,
            "complete": complete,
            "updated_at": datetime.now().isoformat(),
        },
    }
