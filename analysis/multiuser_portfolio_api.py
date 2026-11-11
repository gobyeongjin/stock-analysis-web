from datetime import date
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from multiuser_auth_api import get_current_user
from multiuser_portfolio_service import (
    add_trade,
    get_trades,
    get_holdings,
)

router = APIRouter(
    prefix="/api/my-portfolio",
    tags=["My Portfolio"],
)


class TradeCreate(BaseModel):
    ticker: str = Field(min_length=1)
    stock_name: str = Field(min_length=1)
    trade_type: Literal["BUY", "SELL"]
    trade_date: date
    price: float = Field(gt=0)
    quantity: int = Field(gt=0)
    fee: float = Field(default=0, ge=0)
    tax: float = Field(default=0, ge=0)
    memo: str = ""


@router.get("/trades")
def list_trades(user=Depends(get_current_user)):
    return {"trades": get_trades(user["id"])}


@router.get("/holdings")
def list_holdings(user=Depends(get_current_user)):
    return {"holdings": get_holdings(user["id"])}


@router.post("/trades", status_code=201)
def create_trade(
    trade: TradeCreate,
    user=Depends(get_current_user),
):
    try:
        trade_id = add_trade(
            user_id=user["id"],
            ticker=trade.ticker,
            stock_name=trade.stock_name,
            trade_type=trade.trade_type,
            trade_date=trade.trade_date.isoformat(),
            price=trade.price,
            quantity=trade.quantity,
            fee=trade.fee,
            tax=trade.tax,
            memo=trade.memo,
        )

        return {
            "message": "거래 기록 저장 완료",
            "trade_id": trade_id,
        }

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e),
        )
