import os
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

sys.path.insert(0, str(ROOT / "trading"))
from kiwoom_api import KiwoomAPI

MOCK_URL = "https://mockapi.kiwoom.com"
ENDPOINT = "/api/dostk/acnt"
API_ID = "ka10075"


def to_int(value):
    return int(str(value or "0").replace(",", "").strip())


def main():
    api = KiwoomAPI()

    if api.base_url.rstrip("/") != MOCK_URL:
        raise RuntimeError("모의투자 서버가 아니므로 중단")

    api.get_access_token()
    token = api.token
    if not token:
        raise RuntimeError("토큰 발급 실패")

    headers = {
        "Content-Type": "application/json;charset=UTF-8",
        "authorization": f"Bearer {token}",
        "api-id": API_ID,
    }

    params = {
        "all_stk_tp": "0",  # 전체 종목
        "trde_tp": "0",     # 매수 + 매도
        "stk_cd": "",
        "stex_tp": "0",     # 통합
    }

    orders = []
    seen_keys = set()
    cont_yn = "N"
    next_key = ""

    for page in range(1, 21):
        headers["cont-yn"] = cont_yn
        headers["next-key"] = next_key

        response = requests.post(
            MOCK_URL + ENDPOINT,
            headers=headers,
            json=params,
            timeout=15,
        )
        response.raise_for_status()

        data = response.json()

        if str(data.get("return_code")) != "0":
            raise RuntimeError(
                f"미체결 조회 실패: {data.get('return_msg')}"
            )

        page_orders = data.get("oso")

        if not isinstance(page_orders, list):
            raise RuntimeError(
                "미체결 목록(oso) 응답 형식 확인 필요"
            )

        orders.extend(page_orders)

        new_cont_yn = response.headers.get(
            "cont-yn", "N"
        ).upper()
        new_next_key = response.headers.get(
            "next-key", ""
        )

        if new_cont_yn != "Y":
            break

        if not new_next_key:
            raise RuntimeError(
                "연속조회 키가 없어 중단"
            )

        if new_next_key in seen_keys:
            raise RuntimeError(
                "연속조회 키가 반복되어 중단"
            )

        seen_keys.add(new_next_key)
        cont_yn = "Y"
        next_key = new_next_key
    else:
        raise RuntimeError(
            "최대 조회 페이지 초과: 결과 불완전"
        )

    print("\n===== 키움 모의계좌 미체결 주문 =====")

    pending_buy_codes = set()
    pending_buy_count = 0
    pending_sell_count = 0

    for item in orders:
        qty = to_int(item.get("oso_qty"))

        if qty <= 0:
            continue

        code = str(item.get("stk_cd", "")).removeprefix("A")
        side = str(item.get("io_tp_nm", ""))

        if "매수" in side:
            pending_buy_codes.add(code)
            pending_buy_count += 1
        elif "매도" in side:
            pending_sell_count += 1
        else:
            raise RuntimeError(
                f"매매구분 확인 필요: {side}"
            )

        print(
            f"종목: {item.get('stk_nm')} ({code}) | "
            f"주문번호: {item.get('ord_no')} | "
            f"구분: {side} | "
            f"미체결: {qty}주"
        )

    print(f"\n미체결 매수 주문: {pending_buy_count}건")
    print(f"미체결 매도 주문: {pending_sell_count}건")
    print(f"미체결 매수종목: {len(pending_buy_codes)}개")
    print("READ ONLY — 주문 전송 없음")


if __name__ == "__main__":
    main()
