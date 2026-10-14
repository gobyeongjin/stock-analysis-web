from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
RESULTS = BASE / "results"

data = pd.read_csv(
    RESULTS / "yearly_actual_trade_summary.csv"
)
# 실제 거래 상세 데이터는 원본 가격과 후보 신호에서 다시 계산
raw = pd.read_csv(BASE / "stock_data_500.csv", dtype={"Ticker": str})
raw["Date"] = pd.to_datetime(raw["Date"])
raw["Ticker"] = raw["Ticker"].str.strip()
raw = raw.sort_values(["Ticker", "Date"]).reset_index(drop=True)

g = raw.groupby("Ticker", sort=False)
raw["Entry_Open"] = g["Open"].shift(-1)
raw["Exit_Open"] = g["Open"].shift(-6)

fee = 0.001
raw["Net_Return_Pct"] = (
    raw["Exit_Open"] * (1 - fee) /
    (raw["Entry_Open"] * (1 + fee)) - 1
) * 100

cand = pd.read_csv(
    RESULTS / "backtest_candidates.csv",
    dtype={"Ticker": str}
)
cand["Date"] = pd.to_datetime(cand["Date"])
cand["Ticker"] = cand["Ticker"].str.strip()

cols = [
    "Date", "Ticker", "Open", "Close",
    "Entry_Open", "Exit_Open", "Net_Return_Pct"
]
prices = raw[cols].rename(
    columns={"Open": "Signal_Day_Open", "Close": "Signal_Day_Close"}
)

trades = cand.merge(
    prices, on=["Date", "Ticker"], how="left",
    validate="many_to_one"
)
trades = trades[trades["AI_Score"] >= 0.80].copy()
trades = trades.dropna(subset=["Net_Return_Pct"])
trades["Year"] = trades["Date"].dt.year

show_cols = [
    "Year", "Date", "Stock_Name", "Ticker", "AI_Score",
    "Signal_Day_Close", "Entry_Open", "Exit_Open",
    "Net_Return_Pct"
]

worst = trades.nsmallest(30, "Net_Return_Pct")[show_cols]
best = trades.nlargest(30, "Net_Return_Pct")[show_cols]

worst.to_csv(
    RESULTS / "extreme_worst_trades.csv",
    index=False, encoding="utf-8-sig"
)
best.to_csv(
    RESULTS / "extreme_best_trades.csv",
    index=False, encoding="utf-8-sig"
)

print("\n=== 최악의 거래 30건 ===")
print(worst.round(3).to_string(index=False))
print("\n=== 최고의 거래 30건 ===")
print(best.round(3).to_string(index=False))
print("\n저장 완료:")
print("results/extreme_worst_trades.csv")
print("results/extreme_best_trades.csv")
