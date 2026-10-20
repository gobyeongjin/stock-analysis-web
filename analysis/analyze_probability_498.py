from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent

files = {
    "BASE_CAP25": "atr_risk_BASE_CAP25_trades.csv",
    "RISK_1.0": "atr_risk_RISK_1.0_trades.csv",
    "RISK_1.5": "atr_risk_RISK_1.5_trades.csv",
}

bins = [0.65, 0.75, 0.85, 0.95, np.inf]
labels = [
    "0.65~0.75",
    "0.75~0.85",
    "0.85~0.95",
    "0.95+",
]

results = []
all_trades = []

for strategy, filename in files.items():
    df = pd.read_csv(
        BASE / filename,
        dtype={"Ticker": str},
    )

    df["Probability"] = pd.to_numeric(
        df["Probability"], errors="coerce"
    )
    df["Return_Pct"] = pd.to_numeric(
        df["Return_Pct"], errors="coerce"
    )
    df["PnL"] = pd.to_numeric(
        df["PnL"], errors="coerce"
    )
    df["Entry_Cost"] = pd.to_numeric(
        df["Entry_Cost"], errors="coerce"
    )

    df["Prob_Group"] = pd.cut(
        df["Probability"],
        bins=bins,
        labels=labels,
        right=False,
    )

    df["Is_Win"] = df["PnL"] > 0
    df["Is_Stop"] = (
        df["Exit_Reason"]
        .astype(str)
        .str.contains("손절", na=False)
    )

    df["Strategy_Name"] = strategy
    all_trades.append(df)

    for group in labels:
        part = df[df["Prob_Group"] == group]

        if part.empty:
            continue

        wins = part.loc[part["PnL"] > 0, "Return_Pct"]
        losses = part.loc[part["PnL"] < 0, "Return_Pct"]

        avg_win = wins.mean() if len(wins) else np.nan
        avg_loss = losses.mean() if len(losses) else np.nan

        payoff_ratio = (
            avg_win / abs(avg_loss)
            if pd.notna(avg_win)
            and pd.notna(avg_loss)
            and avg_loss != 0
            else np.nan
        )

        gross_profit = part.loc[
            part["PnL"] > 0, "PnL"
        ].sum()

        gross_loss = abs(
            part.loc[part["PnL"] < 0, "PnL"].sum()
        )

        profit_factor = (
            gross_profit / gross_loss
            if gross_loss > 0
            else np.nan
        )

        results.append({
            "Strategy": strategy,
            "Prob_Group": group,
            "Trades": len(part),
            "Win_Rate_Pct": part["Is_Win"].mean() * 100,
            "Avg_Return_Pct": part["Return_Pct"].mean(),
            "Median_Return_Pct": part["Return_Pct"].median(),
            "Stop_Rate_Pct": part["Is_Stop"].mean() * 100,
            "Avg_Win_Pct": avg_win,
            "Avg_Loss_Pct": avg_loss,
            "Payoff_Ratio": payoff_ratio,
            "Profit_Factor": profit_factor,
            "Total_PnL": part["PnL"].sum(),
        })

summary = pd.DataFrame(results)

summary.to_csv(
    BASE / "probability_analysis_498.csv",
    index=False,
    encoding="utf-8-sig",
)

pd.concat(all_trades, ignore_index=True).to_csv(
    BASE / "probability_all_trades_498.csv",
    index=False,
    encoding="utf-8-sig",
)

print("\n===== AI 확률 구간별 매매 성과 =====")
print(
    summary[
        [
            "Strategy",
            "Prob_Group",
            "Trades",
            "Win_Rate_Pct",
            "Avg_Return_Pct",
            "Median_Return_Pct",
            "Stop_Rate_Pct",
            "Payoff_Ratio",
            "Profit_Factor",
        ]
    ].round(2).to_string(index=False)
)

print("\n===== 확률과 거래 수익률 상관관계 =====")

for strategy, filename in files.items():
    df = pd.read_csv(BASE / filename)

    corr = df["Probability"].corr(
        df["Return_Pct"],
        method="spearman",
    )

    print(f"{strategy}: Spearman = {corr:.4f}")

print("\n분석 완료!")
