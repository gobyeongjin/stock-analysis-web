
import os
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

APP_KEY = os.getenv("KIWOOM_APP_KEY")
SECRET_KEY = os.getenv("KIWOOM_SECRET_KEY")
BASE_URL = "https://api.kiwoom.com"

if not APP_KEY or not SECRET_KEY:
    raise ValueError(".env에서 API 키를 읽지 못했습니다.")


def get_access_token():
    response = requests.post(
        f"{BASE_URL}/oauth2/token",
        headers={"Content-Type": "application/json;charset=UTF-8"},
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
            f"토큰 발급 실패: {data.get('return_msg')}"
        )

    return data["token"]


def get_lending_trend(
    token,
    stock_code="005930",
    start_date="20251001",
    end_date="20261012",
):
    """종목별 대차거래 추이 조회"""

    url = f"{BASE_URL}/api/dostk/slb"
    all_rows = []
    next_key = ""
    cont_yn = "N"

    for page in range(20):
        headers = {
            "Content-Type": "application/json;charset=UTF-8",
            "authorization": f"Bearer {token}",
            "api-id": "ka20068",
            "cont-yn": cont_yn,
            "next-key": next_key,
        }

        body = {
            "stk_cd": stock_code,
            "strt_dt": start_date,
            "end_dt": end_date,
            "all_tp": "0",
        }

        response = requests.post(
            url, headers=headers, json=body, timeout=30
        )

        print(f"\n페이지 {page + 1} / HTTP {response.status_code}")
        response.raise_for_status()
        data = response.json()

        if data.get("return_code", 0) != 0:
            raise RuntimeError(
                f"API 오류: {data.get('return_msg', data)}"
            )

        rows = data.get("dbrt_trde_trnsn", [])
        print("응답 데이터 건수:", len(rows))

        if not rows:
            print("추가 데이터가 없습니다.")
            break

        all_rows.extend(rows)

        cont_yn = response.headers.get("cont-yn", "N")
        next_key = response.headers.get("next-key", "")
        print("연속조회 cont-yn:", repr(cont_yn))  
        print("다음 페이지 next-key:", repr(next_key))

        if cont_yn != "Y" or not next_key:
            break

        time.sleep(0.25)

    if not all_rows:
        print("조회된 데이터가 없습니다.")
        print("응답 JSON:", data)
        return

    df = pd.DataFrame(all_rows)

    print("\n응답 컬럼:", df.columns.tolist())
    print(df.head().to_string(index=False))

    if "dt" not in df.columns or "rmnd" not in df.columns:
        raise ValueError(
            "날짜(dt) 또는 대차잔고(rmnd) 필드를 찾지 못했습니다. "
            "위 응답 컬럼과 API 명세를 확인하세요."
        )

    df = df.drop_duplicates(subset=["dt"])
    df["dt"] = pd.to_datetime(
        df["dt"].astype(str), format="%Y%m%d", errors="coerce"
    )
    df["rmnd"] = pd.to_numeric(
        df["rmnd"].astype(str).str.replace(",", "", regex=False),
        errors="coerce",
    )

    df = df.dropna(subset=["dt", "rmnd"])
    df = df.sort_values("dt")

    # 대차잔고의 전일 대비 변화율(%)
    previous = df["rmnd"].shift(1)
    df["Lending_Balance_Change_Pct"] = (
        (df["rmnd"] - previous) / previous.where(previous != 0) * 100
    )

    output = BASE_DIR / f"lending_trend_{stock_code}.csv"
    df.to_csv(output, index=False, encoding="utf-8-sig")

    print("\n수집 건수:", len(df))
    print("조회 기간:", df["dt"].min(), "~", df["dt"].max())
    print("저장 완료:", output)


if __name__ == "__main__":
    token = get_access_token()
    print("토큰 발급 성공")

    get_lending_trend(
        token,
        stock_code="005930",
        start_date="20251001",
        end_date="20251031",
    )