
from pathlib import Path
import pandas as pd
import numpy as np

BASE = Path(__file__).resolve().parent

FILES = {
    "Slippage_0.05": BASE / "ma40_fixed_levels_volume_dd_0.05_trades.csv",
    "Slippage_0.10": BASE / "ma40_fixed_levels_volume_dd_0.10_trades.csv",
}

def classify_exit(reason):
    reason = str(reason)
    if "손절" in reason:
        return "손절"
    if "익절" in reason:
        return "익절"
    if "최대 보유기간" in reason:
        return "기간 종료"
    return "기타"

def analyze(label, path):
    df = pd.read_csv(path)

    for col in ["PnL", "Return_Pct", "Holding_Days"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Exit_Group"] = df["Exit_Reason"].apply(classify_exit)

    summary = df.groupby("Exit_Group").agg(
        Trades=("PnL", "size"),
        Wins=("PnL", lambda x: (x > 0).sum()),
        Avg_Return_Pct=("Return_Pct", "mean"),
        Median_Return_Pct=("Return_Pct", "median"),
        Total_PnL=("PnL", "sum"),
        Avg_PnL=("PnL", "mean"),
        Avg_Holding_Days=("Holding_Days", "mean"),
    ).reset_index()

    summary.insert(0, "Scenario", label)
    summary["Win_Rate_Pct"] = (
        summary["Wins"] / summary["Trades"] * 100
    )

    wins = df.loc[df["PnL"] > 0, "PnL"]
    losses = df.loc[df["PnL"] < 0, "PnL"]

    gross_profit = wins.sum()
    gross_loss = -losses.sum()

    metrics = {
        "Scenario": label,
        "Trades": len(df),
        "Win_Rate_Pct": (df["PnL"] > 0).mean() * 100,
        "Avg_Return_Pct": df["Return_Pct"].mean(),
        "Median_Return_Pct": df["Return_Pct"].median(),
        "Total_PnL": df["PnL"].sum(),
        "Avg_Win": wins.mean(),
        "Avg_Loss": losses.mean(),
        "Payoff_Ratio": (
            wins.mean() / abs(losses.mean())
            if len(wins) and len(losses) and losses.mean() != 0
            else np.nan
        ),
        "Profit_Factor": (
            gross_profit / gross_loss
            if gross_loss > 0 else np.nan
        ),
        "Expectancy_Per_Trade": df["PnL"].mean(),
    }

    return summary, metrics

summaries = []
metrics = []

for label, path in FILES.items():
    summary, metric = analyze(label, path)
    summaries.append(summary)
    metrics.append(metric)

summary_df = pd.concat(summaries, ignore_index=True)
metrics_df = pd.DataFrame(metrics)

summary_df.to_csv(
    BASE / "exit_performance_summary.csv",
    index=False,
    encoding="utf-8-sig"
)
metrics_df.to_csv(
    BASE / "exit_performance_metrics.csv",
    index=False,
    encoding="utf-8-sig"
)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

print("\n========== 청산 유형별 성과 ==========")
print(summary_df.round(2).to_string(index=False))

print("\n========== 전체 전략 성과 ==========")
print(metrics_df.round(2).to_string(index=False))
