from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "trading"))

from kiwoom_api import KiwoomAPI

ORDER_FILE = ROOT / "trading" / "risk_orders_preview.csv"

MOCK_URL = "https://mockapi.kiwoom.com"
MAX_POSITIONS = 2
MAX_STRATEGY_CAPITAL = 1_000_000


def normalize_code(value):
    code = str(value).strip().removeprefix("A")

    if len(code) != 6 or not code.isdigit():
        raise ValueError(f"잘못된 종목코드: {value}")

    return code


def main():
    api = KiwoomAPI()

    if api.base_url != MOCK_URL:
        raise RuntimeError("모의투자 서버가 아닙니다.")

    api.get_access_token()

    deposit = api.request(
        api_id="kt00001",
        endpoint="/api/dostk/acnt",
        params={"qry_tp": "2"},
    )

    account = api.request(
        api_id="kt00018",
        endpoint="/api/dostk/acnt",
        params={
            "qry_tp": "1",
            "dmst_stex_tp": "KRX",
        },
    )

    for name, data in [
        ("예수금", deposit),
        ("보유종목", account),
    ]:
        if str(data.get("return_code", "0")) != "0":
            raise RuntimeError(
                f"{name} 조회 실패: {data.get('return_msg')}"
            )

    available_cash = int(
        str(deposit["ord_alow_amt"]).replace(",", "")
    )

    holdings = account.get(
        "acnt_evlt_remn_indv_tot", []
    )

    if not isinstance(holdings, list):
        raise RuntimeError("보유종목 응답 형식 오류")

    held_codes = {
        normalize_code(item["stk_cd"])
        for item in holdings
        if item.get("stk_cd")
    }

    if not ORDER_FILE.exists():
        raise FileNotFoundError(ORDER_FILE)

    orders = pd.read_csv(
        ORDER_FILE,
        dtype={"Ticker": str},
    )

    print("\n===== AI 주문 안전 점검 =====")
    print("서버:", api.base_url)
    print(f"키움 주문가능금액: {available_cash:,}원")
    print("계좌 보유종목:", len(held_codes))
    print("AI 매수 계획:", len(orders))

    if len(held_codes) > MAX_POSITIONS:
        raise RuntimeError("최대 보유종목 수 초과")

    if len(orders) + len(held_codes) > MAX_POSITIONS:
        raise RuntimeError("최대 2종목 제한 위반")

    if not orders.empty:
        required = {
            "Ticker",
            "Quantity",
            "Estimated_Amount",
        }

        if not required.issubset(orders.columns):
            raise RuntimeError("필수 주문 컬럼 누락")

        codes = orders["Ticker"].apply(normalize_code)

        if codes.duplicated().any():
            raise RuntimeError("중복 매수 계획 발견")

        duplicates = set(codes) & held_codes

        if duplicates:
            raise RuntimeError(
                f"이미 보유 중인 종목: {duplicates}"
            )

        quantities = pd.to_numeric(
            orders["Quantity"], errors="raise"
        )

        amounts = pd.to_numeric(
            orders["Estimated_Amount"], errors="raise"
        )

        if (
            quantities.isna().any()
            or amounts.isna().any()
            or (quantities <= 0).any()
            or (amounts <= 0).any()
            or (quantities % 1 != 0).any()
        ):
            raise RuntimeError("주문 수량 또는 금액 오류")

        total_amount = float(amounts.sum())

        if total_amount > MAX_STRATEGY_CAPITAL:
            raise RuntimeError("전략 운용금 100만 원 초과")

        if total_amount > available_cash:
            raise RuntimeError("키움 주문가능금액 초과")

        print(f"예상 주문금액: {total_amount:,.0f}원")

        print(
            "주의: 미체결 주문·체결 상태·전략 잔고는 "
            "아직 검증하지 않았습니다."
        )

    print("\n기본 점검 완료")
    print("READ ONLY — 실제 주문 전송 없음")


if __name__ == "__main__":
    main()
