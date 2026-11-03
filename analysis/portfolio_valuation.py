from pathlib import Path
from functools import lru_cache

import pandas as pd
from fastapi import APIRouter

from portfolio_service import get_holdings

router = APIRouter(prefix="/api/portfolio", tags=["Portfolio"])

CSV_PATH = Path(__file__).resolve().parent / "stock_data_500_live.csv"


@lru_cache(maxsize=2)
def load_latest_prices(file_mtime):
    df = pd.read_csv(
        CSV_PATH,
        usecols=["Date", "Ticker", "Open"]
    )

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Open"] = pd.to_numeric(df["Open"], errors="coerce")
    df["Ticker"] = df["Ticker"].astype(str).str.strip().str.upper()

    df = df.dropna(subset=["Date", "Open"])
    df = df[df["Open"] > 0]

    df = df.sort_values("Date").drop_duplicates(
        subset=["Ticker"], keep="last"
    )

    return {
        row.Ticker: {
            "current_price": float(row.Open),
            "price_date": row.Date.strftime("%Y-%m-%d")
        }
        for row in df.itertuples(index=False)
    }


@router.get("/valuation")
def get_portfolio_valuation():
    holdings = get_holdings()

    if not CSV_PATH.exists():
        return {
            "holdings": [],
            "summary": {},
            "error": "주가 CSV 파일이 없습니다."
        }

    prices = load_latest_prices(CSV_PATH.stat().st_mtime_ns)

    result = []
    total_cost = 0.0
    total_value = 0.0
    missing_count = 0

    for stock in holdings:
        ticker = stock["ticker"].strip().upper()
        price_info = prices.get(ticker)
        item = dict(stock)

        if price_info is None:
            item.update({
                "current_price": None,
                "price_date": None,
                "market_value": None,
                "unrealized_pnl": None,
                "return_pct": None
            })
            missing_count += 1
        else:
            current_price = price_info["current_price"]
            market_value = current_price * stock["quantity"]
            cost = stock["cost_basis"]
            pnl = market_value - cost

            item.update({
                "current_price": current_price,
                "price_date": price_info["price_date"],
                "market_value": round(market_value, 2),
                "unrealized_pnl": round(pnl, 2),
                "return_pct": round(pnl / cost * 100, 2)
                    if cost > 0 else None
            })

            total_cost += cost
            total_value += market_value

        result.append(item)

    total_pnl = total_value - total_cost

    return {
        "holdings": result,
        "summary": {
            "stock_count": len(result),
            "priced_count": len(result) - missing_count,
            "missing_count": missing_count,
            "total_cost": round(total_cost, 2),
            "total_value": round(total_value, 2),
            "total_pnl": round(total_pnl, 2),
            "total_return_pct": round(
                total_pnl / total_cost * 100, 2
            ) if total_cost > 0 else None,
            "complete": missing_count == 0
        }
    }
