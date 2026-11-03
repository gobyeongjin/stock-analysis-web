import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "trading"))

from kiwoom_api import KiwoomAPI

MOCK_URL = "https://mockapi.kiwoom.com"
ENDPOINT = "/api/dostk/acnt"
API_ID = "ka10076"

OUTPUT_DIR = ROOT / "trading" / "account_snapshots"


def fetch_filled_orders(api):
    params = {
        "stk_cd": "",
        "qry_tp": "0",
        "sell_tp": "0",
        "ord_no": "",
        "stex_tp": "0",
    }

    results = []
    seen_keys = set()

    cont_yn = "N"
    next_key = ""

    for page in range(1, 21):
        headers = {
            "Content-Type": "application/json;charset=UTF-8",
            "authorization": f"Bearer {api.token}",
            "api-id": API_ID,
            "cont-yn": cont_yn,
            "next-key": next_key,
        }

        response = requests.post(
            api.base_url + ENDPOINT,
            headers=headers,
            json=params,
            timeout=30,
        )

        response.raise_for_status()
        data = response.json()

        if str(data.get("return_code")) != "0":
            raise RuntimeError(
                f"체결조회 실패: {data.get('return_msg')}"
            )

        page_rows = data.get("cntr")

        if not isinstance(page_rows, list):
            raise RuntimeError(
                "체결조회 응답 형식 오류: cntr 목록 없음"
            )

        results.extend(page_rows)

        cont_yn = response.headers.get(
            "cont-yn", "N"
        ).upper()

        next_key = response.headers.get(
            "next-key", ""
        )

        if cont_yn != "Y":
            return results

        if not next_key or next_key in seen_keys:
            raise RuntimeError(
                "체결조회 연속조회 키 오류"
            )

        seen_keys.add(next_key)

    raise RuntimeError(
        "체결조회 페이지 제한 초과: 결과 불완전"
    )


def main():
    api = KiwoomAPI()

    if api.base_url.rstrip("/") != MOCK_URL:
        raise RuntimeError(
            "모의투자 서버가 아니므로 중단"
        )

    api.get_access_token()

    rows = fetch_filled_orders(api)

    print("\n===== 키움 모의계좌 체결조회 =====")
    print(f"조회된 응답 행: {len(rows)}건")

    if rows:
        df = pd.DataFrame(rows)

        display_cols = [
            "ord_no",
            "stk_cd",
            "stk_nm",
            "io_tp_nm",
            "ord_stt",
            "ord_qty",
            "cntr_qty",
            "cntr_pric",
            "oso_qty",
            "ord_tm",
        ]

        available_cols = [
            c for c in display_cols
            if c in df.columns
        ]

        print(
            df[available_cols].to_string(index=False)
        )
    else:
        print("조회된 체결 내역이 없습니다.")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    output_file = (
        OUTPUT_DIR / f"filled_orders_{timestamp}.json"
    )

    snapshot = {
        "created_at": datetime.now().astimezone().isoformat(),
        "server": MOCK_URL,
        "api_id": API_ID,
        "query": {
            "stk_cd": "",
            "qry_tp": "0",
            "sell_tp": "0",
            "ord_no": "",
            "stex_tp": "0",
        },
        "row_count": len(rows),
        "rows": rows,
    }

    output_file.write_text(
        json.dumps(
            snapshot,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\n원본 저장: {output_file}")
    print("전략 잔고 변경 없음")
    print("READ ONLY — 주문 전송 없음")


if __name__ == "__main__":
    main()
