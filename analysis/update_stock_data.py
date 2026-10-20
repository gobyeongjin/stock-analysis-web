from pathlib import Path
import time

import pandas as pd
import yfinance as yf

BASE_DIR = Path(__file__).resolve().parent

SOURCE = BASE_DIR / "stock_data_500.csv"
OUTPUT = BASE_DIR / "stock_data_500_updated.csv"

# 기존 CSV를 절대 덮어쓰지 않음
if OUTPUT.exists():
    raise FileExistsError(
        f"이미 결과 파일이 존재합니다: {OUTPUT}"
    )

df = pd.read_csv(SOURCE, dtype={"Ticker": str})
df["Date"] = pd.to_datetime(df["Date"])

required = {
    "Date", "Ticker", "Open", "High",
    "Low", "Close", "Volume"
}

missing = required - set(df.columns)

if missing:
    raise ValueError(f"기존 데이터 컬럼 누락: {missing}")

last_date = df["Date"].max()

# 마지막 저장일 이후부터 수집
start_date = (
    last_date + pd.Timedelta(days=1)
).strftime("%Y-%m-%d")

# 오늘까지 수집하기 위해 종료일은 내일로 설정
end_date = (
    pd.Timestamp.today().normalize()
    + pd.Timedelta(days=1)
).strftime("%Y-%m-%d")

tickers = sorted(df["Ticker"].dropna().unique())

print("기존 마지막 날짜:", last_date.date())
print("수집 시작일:", start_date)
print("수집 종료일(미포함):", end_date)
print("종목 수:", len(tickers))

new_data = []
failed = []

for i, ticker in enumerate(tickers, 1):
    try:
        stock = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            auto_adjust=False,
            progress=False,
            threads=False,
        )

        if stock.empty:
            failed.append(ticker)
            continue

        if isinstance(stock.columns, pd.MultiIndex):
            stock.columns = stock.columns.get_level_values(0)

        stock = stock.reset_index()

        stock["Date"] = pd.to_datetime(
            stock["Date"]
        ).dt.tz_localize(None)

        stock["Ticker"] = ticker

        if "Stock_Name" in df.columns:
            names = df.loc[
                df["Ticker"] == ticker, "Stock_Name"
            ].dropna()

            stock["Stock_Name"] = (
                names.iloc[-1] if not names.empty else ticker
            )

        new_data.append(stock)

    except Exception as e:
        failed.append(ticker)
        print(f"[오류] {ticker}: {e}")

    if i % 50 == 0:
        print(f"진행: {i}/{len(tickers)}")

    time.sleep(0.1)

if not new_data:
    raise RuntimeError(
        "신규 데이터가 없습니다. 원본은 변경하지 않았습니다."
    )

new_df = pd.concat(new_data, ignore_index=True)

new_df["Date"] = pd.to_datetime(new_df["Date"])

# 기존 데이터에 존재하는 컬럼만 유지
for col in df.columns:
    if col not in new_df.columns:
        new_df[col] = pd.NA

new_df = new_df[df.columns]

combined = pd.concat(
    [df, new_df],
    ignore_index=True
)

combined = (
    combined
    .sort_values(["Ticker", "Date"])
    .drop_duplicates(["Date", "Ticker"], keep="first")
)

combined.to_csv(
    OUTPUT,
    index=False,
    encoding="utf-8-sig"
)

print("\n========== 수집 결과 ==========")
print("기존 행:", len(df))
print("신규 수집 행:", len(new_df))
print("통합 행:", len(combined))
print("최신 날짜:", combined["Date"].max().date())
print("수집 실패/빈 응답 종목:", len(failed))
print("저장 파일:", OUTPUT)

if failed:
    print("확인 필요 종목:", ", ".join(failed[:20]))
