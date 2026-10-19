from dataclasses import dataclass
from decimal import Decimal
from typing import Literal


@dataclass(frozen=True)
class OrderRequest:
    symbol: str
    side: Literal["BUY", "SELL"]
    quantity: int
    price: Decimal


class TossAPI:
    """실주문 기능이 없는 초기 가상 주문 클라이언트."""

    def __init__(self):
        self.dry_run = True

    def place_order(self, order: OrderRequest) -> dict:
        if order.side not in ("BUY", "SELL"):
            raise ValueError("주문 방향은 BUY 또는 SELL이어야 합니다.")

        if not order.symbol.strip():
            raise ValueError("종목 코드가 비어 있습니다.")

        if type(order.quantity) is not int or order.quantity <= 0:
            raise ValueError("주문 수량은 양의 정수여야 합니다.")

        price = Decimal(str(order.price))
        if not price.is_finite() or price <= 0:
            raise ValueError("주문 가격은 양수여야 합니다.")

        return {
            "status": "DRY_RUN",
            "broker": "TOSS",
            "symbol": order.symbol,
            "side": order.side,
            "quantity": order.quantity,
            "price": str(price),
            "estimated_amount": str(price * order.quantity),
            "submitted": False,
        }
