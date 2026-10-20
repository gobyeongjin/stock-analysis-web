from decimal import Decimal
from math import isfinite

from trading.toss_api import TossAPI, OrderRequest


class StrategySignals:
    def __init__(self, threshold=0.65):
        self.threshold = threshold
        self.broker = TossAPI()

    def evaluate_buy(
        self,
        symbol,
        probability,
        price,
        quantity=1,
        already_held=False,
    ):
        probability = float(probability)

        if not isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("AI 확률은 0~1 사이여야 합니다.")

        if already_held:
            return {
                "status": "SKIP",
                "reason": "이미 보유한 종목",
                "symbol": symbol,
            }

        if probability < self.threshold:
            return {
                "status": "SKIP",
                "reason": "AI 확률 기준 미달",
                "symbol": symbol,
                "probability": probability,
            }

        order = OrderRequest(
            symbol=symbol,
            side="BUY",
            quantity=quantity,
            price=Decimal(str(price)),
        )

        result = self.broker.place_order(order)
        result["ai_probability"] = probability
        return result
