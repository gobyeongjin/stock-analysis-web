
import yfinance as yf
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

kospi = yf.download(
    "^KS11",
    start="2024-01-01",
    auto_adjust=False,
    progress=False,
)

if kospi.empty:
    raise RuntimeError("코스피 데이터를 가져오지 못했습니다.")

# yfinance의 MultiIndex 컬럼 처리
if isinstance(kospi.columns, pd.MultiIndex):
    kospi.columns = kospi.columns.get_level_values(0)

kospi = kospi.reset_index()

kospi["Date"] = pd.to_datetime(
    kospi["Date"]
).dt.tz_localize(None).dt.normalize()

kospi["Close"] = pd.to_numeric(
    kospi["Close"],
    errors="coerce"
)

kospi = kospi[
    ["Date", "Close"]
].dropna().sort_values("Date")

kospi["MA20"] = (
    kospi["Close"]
    .rolling(20, min_periods=20)
    .mean()
)

kospi["Market_OK"] = (
    kospi["Close"] > kospi["MA20"]
)

kospi.to_csv(
    BASE_DIR / "kospi_market_filter.csv",
    index=False,
    encoding="utf-8-sig",
)

print("코스피 데이터:", len(kospi))
print("시작일:", kospi["Date"].min())
print("종료일:", kospi["Date"].max())
print(kospi.tail(5).to_string(index=False))
