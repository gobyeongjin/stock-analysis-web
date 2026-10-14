
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

START_YEAR, START_MONTH = 2021, 10
END_YEAR, END_MONTH = 2026, 10

INPUT_CSV = BASE_DIR / "analysis" / "stock_data_500.csv"
OUTPUT_CSV = BASE_DIR / "lending_multi.csv"

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

def fetch_month(token, code, year, month):
    last_day = calendar.monthrange(year, month)[1]
    body = {
        "stk_cd": code,
        "strt_dt": f"{year}{month:02d}01",
        "end_dt": f"{year}{month:02d}{last_day:02d}",
        "all_tp": "0",
    }
    headers = {
        "Content-Type": "application/json;charset=UTF-8",
        "authorization": f"Bearer {token}",
        "api-id": "ka20068",
        "cont-yn": "N",
        "next-key": "",
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
        print(f"  오류 {code} {year}-{month:02d}: "
              f"{data.get('return_msg', data)}")
        return []

    return data.get("dbrt_trde_trnsn", [])

def main():
    if not APP_KEY or not SECRET_KEY:
        raise ValueError(".env에서 키움 API 키를 읽지 못했습니다.")

    stocks = (
        pd.read_csv(INPUT_CSV, dtype={"Ticker": str})
        [["Stock_Name", "Ticker"]]
        .drop_duplicates()
        .head(100)
    )

    token = get_access_token()
    all_rows = []

    for _, stock in stocks.iterrows():
        name = stock["Stock_Name"]
        code = str(stock["Ticker"]).split(".")[0].zfill(6)
        print(f"\n[{name} / {code}] 수집 시작")

        year, month = START_YEAR, START_MONTH
        stock_rows = []

        while (year, month) <= (END_YEAR, END_MONTH):
            try:
                rows = fetch_month(token, code, year, month)
                for row in rows:
                    row["Ticker"] = code
                    row["Stock_Name"] = name
                stock_rows.extend(rows)
                print(f"  {year}-{month:02d}: {len(rows)}건")
            except Exception as exc:
                print(f"  {year}-{month:02d} 요청 실패: {exc}")

            time.sleep(0.25)

            if month == 12:
                year += 1
                month = 1
            else:
                month += 1

        all_rows.extend(stock_rows)
        print(f"  종목 합계: {len(stock_rows)}건")

        # 종목별 수집이 끝날 때마다 중간 결과 저장
        if all_rows:
            pd.DataFrame(all_rows).drop_duplicates(
                subset=["Ticker", "dt"], keep="last"
            ).to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    if not all_rows:
        print("수집 데이터가 없습니다.")
        return

    df = pd.DataFrame(all_rows).drop_duplicates(
        subset=["Ticker", "dt"], keep="last"
    )
    df["dt"] = pd.to_datetime(
        df["dt"].astype(str), format="%Y%m%d", errors="coerce"
    )
    df["rmnd"] = pd.to_numeric(
        df["rmnd"].astype(str).str.replace(",", "", regex=False),
        errors="coerce",
    )
    df = df.dropna(subset=["dt", "rmnd"]).sort_values(["Ticker", "dt"])

    previous = df.groupby("Ticker")["rmnd"].shift(1)
    df["Lending_Balance_Change_Pct"] = (
        (df["rmnd"] - previous)
        / previous.where(previous != 0)
        * 100
    )

    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print("\n===== 전체 수집 완료 =====")
    print("종목 수:", df["Ticker"].nunique())
    print("데이터 건수:", len(df))
    print("기간:", df["dt"].min(), "~", df["dt"].max())
    print("저장 파일:", OUTPUT_CSV)

if __name__ == "__main__":
    main()
