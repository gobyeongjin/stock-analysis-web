import sys
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "trading"))

from kiwoom_api import KiwoomAPI
from order_guard import normalize_code, validate_buy_orders

MOCK_URL = "https://mockapi.kiwoom.com"
ORDER_FILE = ROOT / "trading" / "risk_orders_preview.csv"

MAX_POSITIONS = 2
STRATEGY_CAPITAL = 1_000_000


def to_int(value):
    return int(str(value).replace(",", "").strip())


def get_unfilled_orders(api):
    """키움 미체결 전체 조회. 실패 시 예외 발생."""
    params = {
        "all_stk_tp": "0",
        "trde_tp": "0",
        "stk_cd": "",
        "stex_tp": "0",
    }

    orders = []
    cont_yn = "N"
    next_key = ""
    seen_keys = set()

    for _ in range(20):
        headers = {
            "Content-Type": "application/json;charset=UTF-8",
            "authorization": f"Bearer {api.token}",
            "api-id": "ka10075",
            "cont-yn": cont_yn,
            "next-key": next_key,
        }

        response = requests.post(
            api.base_url + "/api/dostk/acnt",
            headers=headers,
            json=params,
            timeout=30,
        )
        response.raise_for_status()

        data = response.json()

        if str(data.get("return_code")) != "0":
            raise RuntimeError(
                f"미체결 조회 실패: {data.get('return_msg')}"
            )

        page_orders = data.get("oso")
        if not isinstance(page_orders, list):
            raise RuntimeError("미체결 응답 형식 오류")

        orders.extend(page_orders)

        cont_yn = response.headers.get(
            "cont-yn", "N"
        ).upper()
        next_key = response.headers.get(
            "next-key", ""
        )

        if cont_yn != "Y":
            return orders

        if not next_key or next_key in seen_keys:
            raise RuntimeError("미체결 연속조회 오류")

        seen_keys.add(next_key)

    raise RuntimeError("미체결 조회 페이지 제한 초과")


def main():
    api = KiwoomAPI()

    if api.base_url.rstrip("/") != MOCK_URL:
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

    unfilled = get_unfilled_orders(api)

    available_cash = to_int(deposit["ord_alow_amt"])

    holdings = account.get("acnt_evlt_remn_indv_tot")
    if not isinstance(holdings, list):
        raise RuntimeError("보유종목 응답 형식 오류")

    held_codes = {
        normalize_code(item["stk_cd"])
        for item in holdings
        if item.get("stk_cd")
    }

    pending_buy_codes = set()
    pending_buy_count = 0
    pending_sell_count = 0

    for item in unfilled:
        qty = to_int(item["oso_qty"])
        if qty <= 0:
            continue

        code = normalize_code(item["stk_cd"])
        side = str(item.get("io_tp_nm", ""))

        if "매수" in side:
            pending_buy_codes.add(code)
            pending_buy_count += 1
        elif "매도" in side:
            pending_sell_count += 1
        else:
            raise RuntimeError(
                f"매매구분 해석 불가: {side}"
            )

    if not ORDER_FILE.exists():
        raise FileNotFoundError(ORDER_FILE)

    orders = pd.read_csv(
        ORDER_FILE,
        dtype={"Ticker": str},
    )

    if orders.empty:
        planned_codes = []
        planned_amount = 0
    else:
        required = {
            "Ticker",
            "Quantity",
            "Estimated_Amount",
        }
        if not required.issubset(orders.columns):
            raise RuntimeError(
                f"주문 계획 필수 컬럼 누락: "
                f"{required - set(orders.columns)}"
            )

        planned_codes = orders["Ticker"].tolist()

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
            or (quantities % 1 != 0).any()
            or (amounts <= 0).any()
        ):
            raise RuntimeError("주문 수량·금액 오류")

        planned_amount = float(amounts.sum())

    validate_buy_orders(
        planned_codes,
        held_codes,
        pending_buy_codes,
        max_positions=MAX_POSITIONS,
    )

    if planned_amount > STRATEGY_CAPITAL:
        raise RuntimeError("전략 운용금 100만 원 초과")

    if planned_amount > available_cash:
        raise RuntimeError("키움 주문가능금액 초과")

    print("\n===== AI 모의투자 통합 안전 점검 =====")
    print(f"키움 주문가능금액: {available_cash:,}원")
    print(f"보유종목: {len(held_codes)}개")
    print(f"미체결 매수 주문: {pending_buy_count}건")
    print(f"미체결 매도 주문: {pending_sell_count}건")
    print(f"미체결 매수종목: {len(pending_buy_codes)}개")
    print(f"AI 신규 매수 계획: {len(planned_codes)}건")
    print(f"신규 매수 예상금액: {planned_amount:,.0f}원")

    print("\n기본 통합 점검 통과")
    print(
        "주의: 기존 보유원가·미체결 예약금·"
        "전략 잔고는 아직 반영되지 않았습니다."
    )
    print("READ ONLY — 주문 전송 없음")


if __name__ == "__main__":
    main()
