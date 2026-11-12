from datetime import datetime
import math

from fastapi import APIRouter, Depends

from multiuser_auth_api import get_current_user
from multiuser_db import get_connection
from multiuser_portfolio_service import (
    INITIAL_CAPITAL,
    calculate_cash_balance,
    calculate_holdings,
)
from multiuser_portfolio_live import get_prices

router = APIRouter(
    prefix="/api/leaderboard",
    tags=["Leaderboard"],
)


@router.get("")
def get_leaderboard(user=Depends(get_current_user)):
    # 공개에 동의하고 닉네임을 설정한 회원만 조회
    with get_connection() as conn:
        members = conn.execute("""
            SELECT id, nickname
            FROM users
            WHERE ranking_opt_in = 1
              AND nickname IS NOT NULL
              AND TRIM(nickname) != ''
            ORDER BY id
        """).fetchall()

        member_ids = [member["id"] for member in members]

        trades_by_user = {uid: [] for uid in member_ids}

        if member_ids:
            placeholders = ",".join("?" for _ in member_ids)

            rows = conn.execute(
                f"""
                SELECT id, user_id, ticker, stock_name,
                       trade_type, trade_date,
                       price, quantity, fee, tax
                FROM trades
                WHERE user_id IN ({placeholders})
                ORDER BY trade_date, id
                """,
                member_ids,
            ).fetchall()

            for row in rows:
                trade = dict(row)
                trades_by_user[trade["user_id"]].append(trade)

    # 회원별 보유 주식 및 현금 계산
    portfolios = []
    symbols = set()

    for member in members:
        uid = member["id"]
        trades = trades_by_user[uid]

        holdings = calculate_holdings(trades)
        cash = calculate_cash_balance(trades)

        portfolios.append({
            "nickname": member["nickname"],
            "holdings": holdings,
            "cash": cash,
        })

        for stock in holdings:
            symbol = str(stock["ticker"]).strip().upper()
            symbols.add(symbol)

    # 전체 회원의 종목 시세를 한 번에 조회
    prices = get_prices(sorted(symbols)) if symbols else {}

    ranking = []
    unavailable_count = 0

    for portfolio in portfolios:
        total_value = 0.0
        complete = True

        for stock in portfolio["holdings"]:
            symbol = str(stock["ticker"]).strip().upper()
            quote = prices.get(symbol)

            if quote is None:
                complete = False
                break

            try:
                current_price = float(quote["price"])
            except (TypeError, ValueError, KeyError):
                complete = False
                break

            if not math.isfinite(current_price) or current_price <= 0:
                complete = False
                break

            total_value += current_price * stock["quantity"]

        if not complete:
            unavailable_count += 1
            continue

        total_equity = portfolio["cash"] + total_value

        return_pct = (
            (total_equity / INITIAL_CAPITAL) - 1
        ) * 100

        ranking.append({
            "nickname": portfolio["nickname"],
            "total_equity": round(total_equity, 2),
            "overall_return_pct": round(return_pct, 2),
        })

    # 수익률 높은 순으로 정렬
    ranking.sort(
        key=lambda x: (
            -x["overall_return_pct"],
            x["nickname"],
        )
    )

    for index, item in enumerate(ranking, start=1):
        item["rank"] = index

    return {
        "ranking": ranking,
        "participant_count": len(members),
        "ranked_count": len(ranking),
        "unavailable_count": unavailable_count,
        "updated_at": datetime.now().isoformat(),
    }
