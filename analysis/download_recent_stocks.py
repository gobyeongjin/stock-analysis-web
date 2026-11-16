from pathlib import Path
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import time

import pandas as pd
import yfinance as yf

BASE = Path(__file__).resolve().parent
UNIVERSE = BASE / "stock_universe_500.csv"
OUTPUT = BASE / "stock_data_500_recent.csv"

now = datetime.now(ZoneInfo("Asia/Seoul"))
today = now.date()

# 한국시간 16:30 이후에만 당일 데이터 포함
latest_allowed = (
    today if (now.hour, now.minute) >= (16, 30)
    else today - timedelta(days=1)
)

start = (latest_allowed - timedelta(days=365)).isoformat()
end = (latest_allowed + timedelta(days=1)).isoformat()

stocks = pd.read_csv(UNIVERSE, dtype={"Ticker": str})

print("수집 기간:", start, "~", end)
print("대상 종목:", len(stocks))

rows = []
failed = []

for i, item in enumerate(stocks.itertuples(index=False), 1):
    ticker = item.Ticker

    try:
        df = yf.download(
            ticker,
            start=start,
            end=end,
            auto_adjust=False,
            progress=False,
            threads=False,
        )

        if df.empty:
            failed.append(ticker)
            continue

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.reset_index()
        df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
        df["Ticker"] = ticker
        df["Stock_Name"] = item.Stock_Name

        rows.append(df)

    except Exception as e:
        failed.append(ticker)
        print(f"[오류] {ticker}: {e}")

    if i % 50 == 0:
        print(f"진행: {i}/{len(stocks)}")

    time.sleep(0.1)

if failed:
    raise RuntimeError(
        f"수집 실패 {len(failed)}개: {failed[:20]}"
    )

if not rows:
    raise RuntimeError("수집된 데이터가 없습니다.")

combined = pd.concat(rows, ignore_index=True)

combined = (
    combined.sort_values(["Ticker", "Date"])
    .drop_duplicates(["Ticker", "Date"], keep="last")
)

# ==========================================
# OHLC 데이터 품질 검증 및 정제
# ==========================================

invalid_ohlc = (
    (combined["High"] < combined["Low"])
    | (combined["High"] < combined["Open"])
    | (combined["High"] < combined["Close"])
    | (combined["Low"] > combined["Open"])
    | (combined["Low"] > combined["Close"])
)

invalid_count = int(invalid_ohlc.sum())

print(f"비정상 OHLC 데이터: {invalid_count}행")

if invalid_count:
    print(
        combined.loc[
            invalid_ohlc,
            ["Date", "Ticker", "Stock_Name",
             "Open", "High", "Low", "Close"]
        ].to_string(index=False)
    )

    combined = combined.loc[~invalid_ohlc].copy()

print("정제 후 전체 데이터:", len(combined))
print("정제 후 종목 수:", combined["Ticker"].nunique())

if combined["Ticker"].nunique() != len(stocks):
    raise RuntimeError(
        "정제 후 종목 수가 500개가 아닙니다."
    )


combined.to_csv(
    OUTPUT,
    index=False,
    encoding="utf-8-sig",
)

print("\n===== 수집 완료 =====")
print("전체 데이터:", len(combined))
print("종목 수:", combined["Ticker"].nunique())
print("최신 거래일:", combined["Date"].max().date())
print("저장 위치:", OUTPUT)
