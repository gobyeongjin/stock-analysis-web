import os
from pathlib import Path

import requests
import csv
from dotenv import load_dotenv

# .env 파일 불러오기
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

APP_KEY = os.getenv("KIWOOM_APP_KEY")
SECRET_KEY = os.getenv("KIWOOM_SECRET_KEY")

if not APP_KEY or not SECRET_KEY:
    raise ValueError(".env에서 API 키를 읽지 못했습니다.")

BASE_URL = "https://api.kiwoom.com"


def get_access_token():
    """키움 API 접근 토큰 발급"""
    response = requests.post(
        f"{BASE_URL}/oauth2/token",
        headers={
            "Content-Type": "application/json;charset=UTF-8"
        },
        json={
            "grant_type": "client_credentials",
            "appkey": APP_KEY,
            "secretkey": SECRET_KEY,
        },
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()

    if data.get("return_code") != 0 or not data.get("token"):
        raise RuntimeError(
            f"토큰 발급 실패: {data.get('return_msg', '원인 미확인')}"
        )

    return data["token"]


def get_foreign_trend(token, stock_code="005930"):
    """주식외국인종목별매매동향 연속조회 및 CSV 저장"""

    all_rows = []
    next_key = ""
    cont_yn = "N"
    max_pages = 20  # 우선 최대 20페이지까지만 조회

    for page in range(max_pages):
        headers = {
            "Content-Type": "application/json;charset=UTF-8",
            "authorization": f"Bearer {token}",
            "api-id": "ka10008",
            "cont-yn": cont_yn,
            "next-key": next_key,
        }

        response = requests.post(
            f"{BASE_URL}/api/dostk/frgnistt",
            headers=headers,
            json={"stk_cd": stock_code},
            timeout=30,
        )

        print(f"{page + 1}페이지 HTTP 상태:", response.status_code)
        response.raise_for_status()

        data = response.json()

        if data.get("return_code", 0) != 0:
            print("조회 실패:", data.get("return_msg", "원인 미확인"))
            break

        rows = data.get("stk_frgnr", [])
        if not rows:
            print("추가 데이터가 없어 조회를 종료합니다.")
            break

        all_rows.extend(rows)
        print(f"{page + 1}페이지 수집: {len(rows)}건")

        cont_yn = response.headers.get("cont-yn", "N")
        next_key = response.headers.get("next-key", "")

        if cont_yn != "Y" or not next_key:
            break

    if not all_rows:
        print("저장할 데이터가 없습니다.")
        return

    # 날짜 기준 중복 제거
    unique_rows = {}
    for row in all_rows:
        unique_rows[row.get("dt", "")] = row

    csv_path = BASE_DIR / f"foreign_trend_{stock_code}.csv"

    fieldnames = [
        "date",
        "close_price",
        "change_qty",
        "holding_qty",
        "holding_ratio",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for date in sorted(unique_rows.keys(), reverse=True):
            row = unique_rows[date]

            def to_int(value):
                try:
                    return abs(int(str(value).replace(",", "")))
                except (ValueError, TypeError):
                    return 0

            def to_float(value):
                try:
                    return float(str(value).replace(",", "").replace("+", ""))
                except (ValueError, TypeError):
                    return 0.0

            writer.writerow({
                "date": date,
                "close_price": to_int(row.get("close_pric", "")),
                "change_qty": row.get("chg_qty", "0"),
                "holding_qty": to_int(row.get("poss_stkcnt", "")),
                "holding_ratio": to_float(row.get("wght", "")),
            })

    print(f"전체 수집 데이터: {len(all_rows)}건")
    print(f"중복 제거 후 데이터: {len(unique_rows)}건")
    print(f"CSV 저장 완료: {csv_path}")





    if not rows:
        print("데이터가 없습니다. 응답 필드와 API 명세를 확인해야 합니다.")


if __name__ == "__main__":
    token = get_access_token()
    print("토큰 발급 성공")
    get_foreign_trend(token, "005930")