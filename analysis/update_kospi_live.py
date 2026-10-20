from pathlib import Path
from datetime import timedelta
import numpy as np
import pandas as pd
import yfinance as yf

BASE = Path(__file__).resolve().parent
SOURCE = BASE / "kospi_history.csv"
OUTPUT = BASE / "kospi_history_live.csv"

old = pd.read_csv(SOURCE)
old["Date"] = pd.to_datetime(old["Date"]).dt.normalize()
old["Close"] = pd.to_numeric(old["Close"], errors="coerce")

old = (
    old[["Date", "Close"]]
    .dropna()
    .sort_values("Date")
)

if old["Date"].duplicated().any():
    raise ValueError("기존 KOSPI 날짜 중복")

last_date = old["Date"].max()
start_date = last_date - timedelta(days=30)

new = yf.download(
    "^KS11",
    start=start_date.strftime("%Y-%m-%d"),
    end=(pd.Timestamp.now().normalize() + timedelta(days=1)).strftime("%Y-%m-%d"),
    auto_adjust=False,
    progress=False,
    threads=False,
)

if new.empty:
    raise RuntimeError("KOSPI 다운로드 실패")

if isinstance(new.columns, pd.MultiIndex):
    new.columns = new.columns.get_level_values(0)

new = new.reset_index()
new["Date"] = pd.to_datetime(new["Date"]).dt.tz_localize(None).dt.normalize()
new["Close"] = pd.to_numeric(new["Close"], errors="coerce")
new = new[["Date", "Close"]].dropna()

# 기존 기간의 가격 일치 여부 검사
overlap = old.merge(new, on="Date", suffixes=("_old", "_new"))

if overlap.empty:
    raise ValueError("과거 가격 비교 구간 없음")

matched = np.isclose(
    overlap["Close_old"],
    overlap["Close_new"],
    rtol=1e-5,
    atol=0.01,
)

if not matched.all():
    raise ValueError(
        f"KOSPI 과거 가격 불일치: {(~matched).sum()}일"
    )

added = new[new["Date"] > last_date].copy()

if added.empty:
    raise ValueError("신규 KOSPI 데이터 없음")

combined = pd.concat([old, added], ignore_index=True)
combined = combined.sort_values("Date")

if combined["Date"].duplicated().any():
    raise ValueError("KOSPI 날짜 중복")

temp = OUTPUT.with_suffix(".tmp.csv")
combined.to_csv(temp, index=False, encoding="utf-8-sig")
temp.replace(OUTPUT)

print("\n===== KOSPI 업데이트 완료 =====")
print("기존 종료일:", last_date.date())
print("신규 종료일:", combined["Date"].max().date())
print("추가 거래일:", len(added))
print("전체 거래일:", len(combined))
print("저장:", OUTPUT)
