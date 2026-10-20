from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parent

periods = {
    "2024Q4_2025": ("2024-10-30", "2025-12-31"),
    "2026_H1": ("2026-01-01", "2026-06-30"),
    "2026_Q3": ("2026-07-01", "2026-09-23"),
}

results = []

for strategy in ["P65", "P85"]:
    equity = pd.read_csv(
        BASE / f"threshold_{strategy}_equity.csv",
        parse_dates=["Date"]
    )

    trades = pd.read_csv(
        BASE / f"threshold_{strategy}_trades.csv",
        dtype={"Ticker": str},
        parse_dates=["Exit_Date"]
    )

    equity = equity.sort_values("Date").copy()
    equity = equity.drop_duplicates("Date", keep="last")

    for period, (start, end) in periods.items():
        start = pd.Timestamp(start)
        end = pd.Timestamp(end)

        # 기간 시작 전 마지막 평가자산을 기준점으로 사용
        before = equity[equity["Date"] < start]

        if len(before):
            start_equity = float(before.iloc[-1]["Equity"])
        else:
            start_equity = 1_000_000.0

        part = equity[
            equity["Date"].between(start, end)
        ].copy()

        if part.empty:
            continue

        end_equity = float(part.iloc[-1]["Equity"])

        # 시작 자산을 포함한 일별 MDD
        values = pd.concat([
            pd.Series([start_equity]),
            part["Equity"].reset_index(drop=True)
        ], ignore_index=True)

        drawdown = values / values.cummax() - 1

        period_trades = trades[
            trades["Exit_Date"].between(start, end)
        ].copy()

        stops = period_trades[
            period_trades["Exit_Reason"]
            .astype(str)
            .str.contains("손절", na=False)
        ]

        results.append({
            "Strategy": strategy,
            "Period": period,
            "Start_Equity": start_equity,
            "End_Equity": end_equity,
            "Return_Pct": (
                end_equity / start_equity - 1
            ) * 100,
            "MDD_Pct": drawdown.min() * 100,
            "Trades": len(period_trades),
            "Win_Rate_Pct": (
                (period_trades["PnL"] > 0).mean() * 100
                if len(period_trades) else 0
            ),
            "Stops": len(stops),
            "Stop_Rate_Pct": (
                len(stops) / len(period_trades) * 100
                if len(period_trades) else 0
            ),
        })

summary = pd.DataFrame(results)

summary.to_csv(
    BASE / "period_stability_comparison.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n===== 기간별 안정성 비교 =====")
print(
    summary[
        [
            "Strategy",
            "Period",
            "Return_Pct",
            "MDD_Pct",
            "Trades",
            "Win_Rate_Pct",
            "Stops",
            "Stop_Rate_Pct",
        ]
    ].round(2).to_string(index=False)
)

print("\n===== 기간별 P85 - P65 차이 =====")

pivot = summary.pivot(
    index="Period",
    columns="Strategy",
    values=["Return_Pct", "MDD_Pct"]
)

for period in pivot.index:
    return_diff = (
        pivot.loc[period, ("Return_Pct", "P85")]
        - pivot.loc[period, ("Return_Pct", "P65")]
    )

    mdd_diff = (
        pivot.loc[period, ("MDD_Pct", "P85")]
        - pivot.loc[period, ("MDD_Pct", "P65")]
    )

    print(
        f"{period}: "
        f"수익률 차이 {return_diff:+.2f}%p, "
        f"MDD 차이 {mdd_diff:+.2f}%p"
    )

print("\n분석 완료!")
