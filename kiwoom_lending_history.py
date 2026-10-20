
import os
import time
import calendar
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

APP_KEY = os.getenv("KIWOOM_APP_KEY")
SECRET_KEY = os.getenv("KIWOOM_SECRET_KEY")
BASE_URL = "https://api.kiwoom.com"

STOCK_CODE = "005930"
START_YEAR = 2021
START_MONTH = 10
END_YEAR = 2026
END_MONTH = 10


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
        raise RuntimeError(f"토큰 발급 실패: {data}")

    return data["token"]


def fetch_month(token, year, month):
    last_day = calendar.monthrange(year, month)[1]
    start_date = f"{year}{month:02d}01"
    end_date = f"{year}{month:02d}{last_day:02d}"

    headers = {
        "Content-Type": "application/json;charset=UTF-8",
        "authorization": f"Bearer {token}",
        "api-id": "ka20068",
        "cont-yn": "N",
        "next-key": "",
    }

    body = {
        "stk_cd": STOCK_CODE,
        "strt_dt": start_date,
        "end_dt": end_date,
        "all_tp": "0",
    }

    response = requests.post(
        f"{BASE_URL}/api/dostk/slb",
        headers=headers,
        json=body,
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()

    if data.get("return_code", 0) != 0:
        print(f"{year}-{month:02d} 오류:", data.get("return_msg", data))
        return []

    rows = data.get("dbrt_trde_trnsn", [])
    print(f"{year}-{month:02d}: {len(rows)}건")
    return rows


def main():
    if not APP_KEY or not SECRET_KEY:
        raise ValueError(".env에서 키움 API 키를 읽지 못했습니다.")

    token = get_access_token()
    all_rows = []

    year, month = START_YEAR, START_MONTH

    while (year, month) <= (END_YEAR, END_MONTH):
        rows = fetch_month(token, year, month)
        all_rows.extend(rows)

        time.sleep(0.25)

        if month == 12:
            year += 1
            month = 1
        else:
            month += 1

    if not all_rows:
        print("수집된 데이터가 없습니다.")
        return

    df = pd.DataFrame(all_rows)

    if "dt" not in df.columns or "rmnd" not in df.columns:
        print("예상과 다른 응답 컬럼:", df.columns.tolist())
        print(df.head().to_string(index=False))
        return

    df = df.drop_duplicates(subset=["dt"], keep="last")
    df["dt"] = pd.to_datetime(
        df["dt"].astype(str), format="%Y%m%d", errors="coerce"
    )
    df["rmnd"] = pd.to_numeric(
        df["rmnd"].astype(str).str.replace(",", "", regex=False),
        errors="coerce",
    )

    df = df.dropna(subset=["dt", "rmnd"]).sort_values("dt")

    # 잔고 변화율 계산
    previous = df["rmnd"].shift(1)
    df["Lending_Balance_Change_Pct"] = (
        (df["rmnd"] - previous)
        / previous.where(previous != 0)
        * 100
    )

    output = BASE_DIR / f"lending_history_{STOCK_CODE}.csv"
    df.to_csv(output, index=False, encoding="utf-8-sig")

    print("\n===== 수집 완료 =====")
    print("데이터 건수:", len(df))
    print("시작일:", df["dt"].min())
    print("종료일:", df["dt"].max())
    print("저장 파일:", output)


if __name__ == "__main__":
    main()
