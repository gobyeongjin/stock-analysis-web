
import pandas as pd

df = pd.read_csv("v4_swing_atr_trades.csv")

# ATR이 진입 가격에서 차지하는 비율
df["ATR_Pct"] = df["Entry_ATR"] / df["Entry_Price"] * 100

print("===== 종목별 성과 =====")
print(
    df.groupby("Ticker")
    .agg(
        거래수=("PnL", "size"),
        총손익=("PnL", "sum"),
        평균수익률=("Return_Pct", "mean"),
        승률=("PnL", lambda x: (x > 0).mean() * 100),
        평균_ATR비율=("ATR_Pct", "mean"),
    )
    .sort_values("총손익")
    .round(2)
    .to_string()
)

# ATR 비율별 구간 비교
bins = [0, 2, 4, 6, 8, 100]
labels = ["2% 미만", "2~4%", "4~6%", "6~8%", "8% 이상"]

df["ATR_구간"] = pd.cut(
    df["ATR_Pct"],
    bins=bins,
    labels=labels,
    right=False,
)

print("\n===== ATR 비율별 성과 =====")
print(
    df.groupby("ATR_구간", observed=False)
    .agg(
        거래수=("PnL", "size"),
        총손익=("PnL", "sum"),
        평균수익률=("Return_Pct", "mean"),
        승률=("PnL", lambda x: (x > 0).mean() * 100),
    )
    .round(2)
    .to_string()
)
