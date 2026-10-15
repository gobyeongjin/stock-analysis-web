from pathlib import Path
import time
import pandas as pd
from pykrx import stock

BASE = Path(__file__).resolve().parent

SOURCE = BASE / "stock_data_500.csv"
MISSING = BASE / "missing_20250919_tickers.csv"

PROGRESS = BASE / "recovered_20250919_progress.csv"
FAILED = BASE / "recovered_20250919_failed.csv"
OUTPUT = BASE / "stock_data_500_fixed.csv"

TARGET = pd.Timestamp("2025-09-19")

original = pd.read_csv(SOURCE, dtype={"Ticker": str})
original["Date"] = pd.to_datetime(original["Date"])

tickers = (
    pd.read_csv(MISSING, dtype={"Ticker": str})["Ticker"]
    .dropna()
    .unique()
    .tolist()
)

names = (
    original.dropna(subset=["Stock_Name"])
    .drop_duplicates("Ticker", keep="last")
    .set_index("Ticker")["Stock_Name"]
    .to_dict()
)

# 이전 실행에서 성공한 종목 불러오기
if PROGRESS.exists():
    saved = pd.read_csv(PROGRESS, dtype={"Ticker": str})
    saved["Date"] = pd.to_datetime(saved["Date"])
    saved = saved.drop_duplicates(["Ticker", "Date"])
    rows = saved.to_dict("records")
else:
    rows = []

completed = {r["Ticker"] for r in rows}
failures = []

print(f"복구 대상: {len(tickers)}개")
print(f"기존 복구 성공: {len(completed)}개")
print(f"남은 종목: {len(tickers) - len(completed)}개")

for i, ticker in enumerate(tickers, 1):

    if ticker in completed:
        continue

    code = ticker.split(".")[0]

    try:
        df = stock.get_market_ohlcv_by_date(
            "20250918",
            "20250922",
            code
        )

        if df.empty:
            raise ValueError("조회 결과 없음")

        target = df.loc[
            df.index.strftime("%Y-%m-%d")
            == "2025-09-19"
        ]

        if target.empty:
            raise ValueError("9월 19일 데이터 없음")

        r = target.iloc[0]

        row = {
            "Date": TARGET,
            "Ticker": ticker,
            "Stock_Name": names.get(ticker, ""),
            "Open": float(r["시가"]),
            "High": float(r["고가"]),
            "Low": float(r["저가"]),
            "Close": float(r["종가"]),
            "Volume": float(r["거래량"]),
        }

        if (
            any(pd.isna(row[c]) for c in
                ["Open", "High", "Low", "Close", "Volume"])
            or row["Low"] <= 0
            or row["Open"] <= 0
            or row["Close"] <= 0
            or row["High"] < row["Low"]
            or not (
                row["Low"] <= row["Open"] <= row["High"]
                and row["Low"] <= row["Close"] <= row["High"]
            )
        ):
            raise ValueError("OHLC 이상")

        rows.append(row)
        completed.add(ticker)

        # 성공 즉시 저장: 중단 후 재실행 가능
        pd.DataFrame(rows).to_csv(
            PROGRESS,
            index=False,
            encoding="utf-8-sig"
        )

    except Exception as e:
        failures.append({
            "Ticker": ticker,
            "Error": str(e)
        })

    if i % 30 == 0 or i == len(tickers):
        print(
            f"진행 {i}/{len(tickers)} | "
            f"성공 {len(completed)} | "
            f"이번 실행 실패 {len(failures)}",
            flush=True
        )

    time.sleep(0.3)

pd.DataFrame(
    failures,
    columns=["Ticker", "Error"]
).to_csv(
    FAILED,
    index=False,
    encoding="utf-8-sig"
)

print("\n===== 최종 복구 결과 =====")
print("성공:", len(completed))
print("실패:", len(tickers) - len(completed))

if not rows:
    raise RuntimeError("복구된 데이터가 없습니다.")

recovered = pd.DataFrame(rows)
recovered["Date"] = pd.to_datetime(recovered["Date"])

# 원본은 그대로 두고 복구된 데이터만 추가
fixed = pd.concat(
    [original, recovered],
    ignore_index=True
)

fixed = fixed.drop_duplicates(
    ["Ticker", "Date"],
    keep="first"
)

fixed = fixed.sort_values(["Ticker", "Date"])

fixed.to_csv(
    OUTPUT,
    index=False,
    encoding="utf-8-sig"
)

print(
    "9월 19일 전체 종목 수:",
    fixed.loc[
        fixed["Date"] == TARGET, "Ticker"
    ].nunique()
)

print("복구 파일:", OUTPUT)
print("진행 저장 파일:", PROGRESS)
print("실패 목록:", FAILED)
print("원본 CSV는 변경하지 않았습니다.")
