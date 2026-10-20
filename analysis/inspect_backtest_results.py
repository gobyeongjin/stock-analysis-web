
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
FILE = BASE_DIR / "backtest_atr_v2_trades.csv"

df = pd.read_csv(FILE)
df["Entry_Date"] = pd.to_datetime(df["Entry_Date"])
df["Exit_Date"] = pd.to_datetime(df["Exit_Date"])

print("\n===== 전략별 거래 요약 =====")
print(
    df.groupby("Strategy").agg(
        Trades=("Net_PnL", "size"),
        Total_PnL=("Net_PnL", "sum"),
        Avg_PnL=("Net_PnL", "mean"),
        Median_PnL=("Net_PnL", "median"),
        Win_Rate=("Net_PnL", lambda s: (s > 0).mean() * 100),
        Avg_Return=("Return_Pct", "mean"),
        Best_Trade=("Return_Pct", "max"),
        Worst_Trade=("Return_Pct", "min"),
        Avg_Holding_Days=("Holding_Days", "mean"),
    ).round(2).to_string()
)

print("\n===== 스윙 ATR형 청산 사유별 =====")
swing = df[df["Strategy"] == "SWING_ATR"]

if not swing.empty:
    print(
        swing.groupby("Exit_Reason").agg(
            Trades=("Net_PnL", "size"),
            Total_PnL=("Net_PnL", "sum"),
            Avg_Return=("Return_Pct", "mean"),
            Win_Rate=("Net_PnL", lambda s: (s > 0).mean() * 100),
        ).round(2).to_string()
    )

    print("\n===== 스윙 ATR형 수익 상위 10건 =====")
    cols = [
        "Ticker", "Entry_Date", "Exit_Date", "Entry_Price",
        "Exit_Price", "Shares", "Exit_Reason",
        "Net_PnL", "Return_Pct", "Holding_Days",
    ]
    print(
        swing.nlargest(10, "Net_PnL")[cols]
        .round(2).to_string(index=False)
    )

    print("\n===== 스윙 ATR형 손실 하위 10건 =====")
    print(
        swing.nsmallest(10, "Net_PnL")[cols]
        .round(2).to_string(index=False)
    )

print("\n===== 거래 기록 일관성 =====")
print("전체 거래 수:", len(df))
print("누락된 손익:", df["Net_PnL"].isna().sum())
print("누락된 수익률:", df["Return_Pct"].isna().sum())
print("청산일이 매수일보다 빠른 거래:",
      (df["Exit_Date"] < df["Entry_Date"]).sum())
print("결과 파일:", FILE)
