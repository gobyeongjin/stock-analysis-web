import json
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

BASE = Path(__file__).resolve().parent

JSON_FILE = BASE / "ai_top10_recent_test.json"
CSV_FILE = BASE / "stock_data_500_recent.csv"

today = datetime.now(ZoneInfo("Asia/Seoul")).date()


def check_freshness(date, label):
    business_days = 0
    day = date

    while day < today:
        day += timedelta(days=1)
        if day.weekday() < 5:
            business_days += 1

    if date > today:
        raise ValueError(f"{label}: 미래 날짜입니다.")

    if business_days > 3:
        raise ValueError(
            f"{label}: 데이터가 오래됐습니다. "
            f"{date}, 경과 평일 {business_days}일"
        )

    return business_days


# ==========================================
# 1. AI JSON 검증
# ==========================================

data = json.loads(JSON_FILE.read_text(encoding="utf-8"))

as_of = datetime.strptime(
    data["as_of"], "%Y-%m-%d"
).date()

json_days = check_freshness(as_of, "AI 데이터")

top10 = data.get("top10", [])

if len(top10) != 10:
    raise ValueError(f"TOP10 종목 수 오류: {len(top10)}")

tickers = [stock["ticker"] for stock in top10]

if len(set(tickers)) != 10:
    raise ValueError("TOP10에 중복 종목이 있습니다.")

# ==========================================
# 2. 주가 CSV 검증
# ==========================================

if not CSV_FILE.exists():
    raise FileNotFoundError(CSV_FILE)

if CSV_FILE.stat().st_size == 0:
    raise ValueError("주가 CSV 파일이 비어 있습니다.")

df = pd.read_csv(
    CSV_FILE,
    dtype={"Ticker": str}
)

required = [
    "Date", "Open", "High", "Low",
    "Close", "Volume", "Ticker", "Stock_Name"
]

missing = [
    col for col in required
    if col not in df.columns
]

if missing:
    raise ValueError(f"CSV 필수 컬럼 누락: {missing}")

if df.empty:
    raise ValueError("주가 CSV 데이터가 없습니다.")

df["Date"] = pd.to_datetime(
    df["Date"], errors="coerce"
)

numeric_cols = [
    "Open", "High", "Low", "Close", "Volume"
]

for col in numeric_cols:
    df[col] = pd.to_numeric(
        df[col], errors="coerce"
    )

if df[required].isna().any().any():
    raise ValueError("CSV에 결측 데이터가 있습니다.")

stock_count = df["Ticker"].nunique()

if stock_count != 500:
    raise ValueError(
        f"분석 종목 수 오류: {stock_count}개"
    )

if df.duplicated(["Ticker", "Date"]).any():
    raise ValueError("중복 종목·거래일 데이터가 있습니다.")

if (df[numeric_cols] < 0).any().any():
    raise ValueError("음수 주가 또는 거래량이 있습니다.")

invalid_price = (
    (df["High"] < df["Low"])
    | (df["High"] < df["Open"])
    | (df["High"] < df["Close"])
    | (df["Low"] > df["Open"])
    | (df["Low"] > df["Close"])
)

if invalid_price.any():
    raise ValueError("OHLC 가격 범위 오류가 있습니다.")

latest_date = df["Date"].max().date()

csv_days = check_freshness(
    latest_date, "주가 CSV"
)

# AI 기준일과 CSV 거래일은 별도로 표시
print("===== AI TOP10 검증 성공 =====")
print("AI 기준일:", as_of)
print("TOP10 종목 수:", len(top10))
print("AI 데이터 경과 평일:", json_days)

print("\n===== 주가 CSV 검증 성공 =====")
print("전체 데이터:", len(df))
print("종목 수:", stock_count)
print("최신 거래일:", latest_date)
print("CSV 데이터 경과 평일:", csv_days)

print("\n전체 데이터 검증 성공!")
