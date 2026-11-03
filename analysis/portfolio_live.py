from datetime import datetime
from pathlib import Path
import sys
import time
import threading

from fastapi import APIRouter
from portfolio_service import get_holdings

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trading.toss_readonly import TossReadOnlyAPI

router = APIRouter(prefix="/api/portfolio", tags=["Portfolio"])

CACHE_SECONDS = 60
_cache = {"timestamp": 0, "prices": {}}
_lock = threading.Lock()


def get_prices(symbols):
    now = time.monotonic()

    with _lock:
        if now - _cache["timestamp"] < CACHE_SECONDS:
            if all(symbol in _cache["prices"] for symbol in symbols):
                return {
                    symbol: _cache["prices"][symbol]
                    for symbol in symbols
                }

        api = TossReadOnlyAPI()
        data = api.get_prices(symbols)

        prices = {}

        for item in data.get("result", []):
            symbol = str(item.get("symbol", ""))
            raw_price = item.get("lastPrice")

            try:
                price = float(raw_price)
                if price <= 0:
                    continue
            except (ValueError, TypeError):
                continue

            prices[symbol] = {
                "price": price,
                "timestamp": item.get("timestamp")
            }

        _cache["prices"] = prices
        _cache["timestamp"] = time.monotonic()
        return prices


@router.get("/live")
def get_live_portfolio():
    holdings = get_holdings()

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
                "return_pct": None
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
                    if cost > 0 else None
            })

            total_cost += cost
            total_value += market_value

        result.append(item)

    pnl = total_value - total_cost
    complete = missing_count == 0

    return {
        "holdings": result,
        "summary": {
            "stock_count": len(result),
            "missing_count": missing_count,
            "total_cost": round(total_cost, 2) if complete else None,
            "total_value": round(total_value, 2) if complete else None,
            "total_pnl": round(pnl, 2) if complete else None,
            "total_return_pct": round(pnl / total_cost * 100, 2)
                if complete and total_cost > 0 else None,
            "complete": complete,
            "updated_at": datetime.now().isoformat()
        }
    }
