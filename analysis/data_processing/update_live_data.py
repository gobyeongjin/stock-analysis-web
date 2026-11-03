from pathlib import Path
from datetime import timedelta
import time

import numpy as np
import pandas as pd
import yfinance as yf

BASE = Path(__file__).resolve().parents[1]
SOURCE = BASE / "stock_data_500.csv"
OUTPUT = BASE / "stock_data_live.csv"

COLS = [
    "Date", "Adj Close", "Close", "High", "Low",
    "Open", "Volume", "Stock_Name", "Ticker"
]

# 최근 30일을 재조회해 기존 데이터와 가격 일관성을 점검
OVERLAP_DAYS = 30
BATCH_DELAY = 0.15

print("기존 데이터 로딩...")

old = pd.read_csv(
    SOURCE,
    dtype={"Ticker": str}
)
old["Date"] = pd.to_datetime(old["Date"])

if old.duplicated(["Ticker", "Date"]).any():
    raise ValueError("원본에 종목/날짜 중복이 있습니다.")

last_date = old["Date"].max()
start_date = last_date - timedelta(days=OVERLAP_DAYS)
end_date = pd.Timestamp.now().normalize() + timedelta(days=1)

if end_date <= last_date:
    raise ValueError("원본 최신 날짜가 현재보다 미래입니다.")

stocks = (
    old[["Ticker", "Stock_Name"]]
    .drop_duplicates("Ticker")
    .sort_values("Ticker")
)

print("기존 최신 날짜:", last_date.date())
print("재조회 시작:", start_date.date())
print("수집 종목:", len(stocks))
print("원본 데이터는 수정하지 않습니다.")

new_rows = []
failed = []
conflicts = []
checked = 0

for i, row in enumerate(stocks.itertuples(index=False), 1):
    ticker = row.Ticker
    name = row.Stock_Name

    try:
        df = yf.download(
            ticker,
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
            timeout=20
        )

        if df.empty:
            failed.append((ticker, "다운로드 결과 없음"))
            continue

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.reset_index()

        if "Date" not in df.columns:
            raise ValueError("Date 컬럼 없음")

        df["Date"] = (
            pd.to_datetime(df["Date"])
            .dt.tz_localize(None)
            .dt.normalize()
        )
        df["Stock_Name"] = name
        df["Ticker"] = ticker

        if "Adj Close" not in df.columns:
            raise ValueError("Adj Close 컬럼 없음")

        df = df[COLS].copy()

        for col in [
            "Adj Close", "Close", "High",
            "Low", "Open", "Volume"
        ]:
            df[col] = pd.to_numeric(
                df[col], errors="coerce"
            )

        # 다운로드된 데이터 자체의 품질 검사
        bad = (
            df[["Open", "High", "Low", "Close"]]
            .isna().any(axis=1)
            | (df["High"] < df["Low"])
            | (df["Low"] <= 0)
            | (df["Open"] < df["Low"])
            | (df["Open"] > df["High"])
            | (df["Close"] < df["Low"])
            | (df["Close"] > df["High"])
            | (df["Volume"] < 0)
        )

        if bad.any():
            print(
                f"{ticker}: OHLC 이상 {bad.sum()}행 제외",
                flush=True
            )
            df = df.loc[~bad].copy()

        if df.empty:
            failed.append((ticker, "정상 가격 데이터 없음"))
            continue

        # 겹치는 날짜에서 기존 가격과 새 가격 비교
        previous = old.loc[
            (old["Ticker"] == ticker)
            & (old["Date"] >= start_date),
            ["Date", "Close", "Open", "High", "Low"]
        ]

        overlap = previous.merge(
            df[["Date", "Close", "Open", "High", "Low"]],
            on="Date",
            suffixes=("_old", "_new")
        )

        # 소수점 오차는 허용, 실제 가격 차이는 확인 필요
        mismatch = np.zeros(len(overlap), dtype=bool)
        for col in ["Close", "Open", "High", "Low"]:
            mismatch |= ~np.isclose(
                overlap[f"{col}_old"].to_numpy(),
                overlap[f"{col}_new"].to_numpy(),
                rtol=1e-5,
                atol=0.01,
                equal_nan=False
            )

        if mismatch.any():
            conflicts.append(
                (ticker, name, int(mismatch.sum()))
            )
            continue

        # 기존 날짜는 보존하고 신규 날짜만 추가
        additional = df.loc[
            df["Date"] > last_date
        ].copy()

        if not additional.empty:
            new_rows.append(additional)

        checked += 1

    except Exception as e:
        failed.append((ticker, str(e)[:120]))

    if i % 50 == 0:
        print(
            f"진행: {i}/{len(stocks)} | "
            f"정상 {checked} | "
            f"실패 {len(failed)} | "
            f"가격 불일치 {len(conflicts)}",
            flush=True
        )

    time.sleep(BATCH_DELAY)

print("\n========== 수집 결과 ==========")
print("정상 확인:", checked)
print("다운로드/품질 실패:", len(failed))
print("과거 가격 불일치:", len(conflicts))

if failed:
    print("\n실패 예시:")
    for item in failed[:10]:
        print(item)

if conflicts:
    print("\n가격 불일치 예시:")
    for item in conflicts[:10]:
        print(item)

# 데이터가 일부라도 불완전하면 자동매매용 파일 저장 중단
# 종목 누락을 숨기지 않기 위한 안전장치
if failed or conflicts:
    raise SystemExit(
        "\n일부 종목 검증 실패. "
        "기존 파일은 보존했습니다. "
        "문제 확인 후 다시 실행하세요."
    )

if not new_rows:
    print("\n새로 추가할 거래 데이터가 없습니다.")
    raise SystemExit(0)

added = pd.concat(new_rows, ignore_index=True)

combined = pd.concat(
    [old[COLS], added[COLS]],
    ignore_index=True
)

combined = (
    combined.drop_duplicates(
        ["Ticker", "Date"],
        keep="first"
    )
    .sort_values(["Ticker", "Date"])
    .reset_index(drop=True)
)

# 부분 파일이 남지 않도록 임시 파일을 먼저 생성
temp_path = OUTPUT.with_suffix(".tmp.csv")
combined.to_csv(
    temp_path,
    index=False,
    encoding="utf-8-sig"
)
temp_path.replace(OUTPUT)

print("\n========== 완료 ==========")
print("추가된 행:", len(added))
print("전체 행:", len(combined))
print("최신 날짜:", combined["Date"].max().date())
print("종목 수:", combined["Ticker"].nunique())
print("저장 위치:", OUTPUT)
print("주의: 주문 전송 없음")
