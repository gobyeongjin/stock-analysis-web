from pathlib import Path
import pandas as pd
import numpy as np

BASE = Path(__file__).resolve().parents[1]
RESULTS = BASE / "results"

PRICE_FILE = BASE / "stock_data_500.csv"
CANDIDATE_FILE = RESULTS / "backtest_candidates.csv"

# 데이터 불러오기
raw = pd.read_csv(PRICE_FILE, dtype={"Ticker": str})
raw["Date"] = pd.to_datetime(raw["Date"])
raw["Ticker"] = raw["Ticker"].str.strip()
raw = raw.sort_values(["Ticker", "Date"]).reset_index(drop=True)

g = raw.groupby("Ticker", sort=False)

# 신호 다음 날 시가 매수, 5거래일 보유 후 시가 매도
raw["Entry_Open"] = g["Open"].shift(-1)
raw["Exit_Open"] = g["Open"].shift(-6)

# 매수·매도 각각 0.1% 비용 가정
fee = 0.001
raw["Net_Return_Pct"] = (
    raw["Exit_Open"] * (1 - fee) /
    (raw["Entry_Open"] * (1 + fee)) - 1
) * 100

prices = raw[
    ["Date", "Ticker", "Entry_Open", "Exit_Open", "Net_Return_Pct"]
]

cand = pd.read_csv(CANDIDATE_FILE, dtype={"Ticker": str})
cand["Date"] = pd.to_datetime(cand["Date"])
cand["Ticker"] = cand["Ticker"].str.strip()
cand["AI_Score"] = pd.to_numeric(cand["AI_Score"], errors="coerce")

data = cand.merge(
    prices,
    on=["Date", "Ticker"],
    how="left",
    validate="many_to_one"
)

data = data[
    (data["AI_Score"] >= 0.80) &
    data["Net_Return_Pct"].notna() &
    data["Entry_Open"].gt(0) &
    data["Exit_Open"].gt(0)
].copy()

data["Year"] = data["Date"].dt.year

# 연도별 성과
yearly = data.groupby("Year").agg(
    Trades=("Net_Return_Pct", "size"),
    Win_Rate_Pct=("Net_Return_Pct", lambda x: (x > 0).mean() * 100),
    Average_Return_Pct=("Net_Return_Pct", "mean"),
    Median_Return_Pct=("Net_Return_Pct", "median"),
    Best_Return_Pct=("Net_Return_Pct", "max"),
    Worst_Return_Pct=("Net_Return_Pct", "min"),
).reset_index()

out = RESULTS / "yearly_actual_trade_summary.csv"
yearly.to_csv(out, index=False, encoding="utf-8-sig")

print("\n=== AI 점수 0.80 이상: 연도별 실제 매매 가정 성과 ===")
print(yearly.round(3).to_string(index=False))
print(f"\n저장 위치: {out}")
print("\n주의: 개별 신호 기준이며 보유 기간 중 거래가 겹칠 수 있습니다.")
print("연도별 결과는 미래 수익을 보장하지 않습니다.")
