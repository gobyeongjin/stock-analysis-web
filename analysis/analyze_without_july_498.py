from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
INITIAL_CASH = 1_000_000
EXCLUDED_MONTH = "2026-07"

results = []
monthly_rows = []

for strategy in ["P65", "P85"]:
    df = pd.read_csv(
        BASE / f"threshold_{strategy}_equity.csv",
        parse_dates=["Date"]
    )

    df = (
        df.sort_values("Date")
        .drop_duplicates("Date", keep="last")
        .dropna(subset=["Equity"])
        .copy()
    )

    df["Month"] = df["Date"].dt.to_period("M").astype(str)

    monthly_end = df.groupby("Month")["Equity"].last()

    previous = monthly_end.shift(1)
    previous.iloc[0] = INITIAL_CASH

    monthly_return = monthly_end / previous - 1

    # 7월을 제외한 월별 수익률
    filtered = monthly_return[
        monthly_return.index != EXCLUDED_MONTH
    ]

    # 월별 수익률을 복리로 연결한 가상 자산곡선
    synthetic_equity = INITIAL_CASH * (
        1 + filtered
    ).cumprod()

    # 월말 기준 MDD (월중 하락은 반영되지 않음)
    curve = pd.concat([
        pd.Series([INITIAL_CASH], index=["START"]),
        synthetic_equity
    ])

    drawdown = curve / curve.cummax() - 1

    for month, ret in monthly_return.items():
        monthly_rows.append({
            "Strategy": strategy,
            "Month": month,
            "Monthly_Return_Pct": ret * 100
        })

    results.append({
        "Strategy": strategy,
        "Full_Return_Pct": (
            monthly_end.iloc[-1] / INITIAL_CASH - 1
        ) * 100,
        "Without_July_Return_Pct": (
            synthetic_equity.iloc[-1] / INITIAL_CASH - 1
        ) * 100,
        "July_Return_Pct": (
            monthly_return.loc[EXCLUDED_MONTH] * 100
        ),
        "Without_July_MonthEnd_MDD_Pct": (
            drawdown.min() * 100
        ),
        "Positive_Months_Without_July": int(
            (filtered > 0).sum()
        ),
        "Negative_Months_Without_July": int(
            (filtered < 0).sum()
        ),
    })

summary = pd.DataFrame(results)
monthly = pd.DataFrame(monthly_rows)

summary.to_csv(
    BASE / "without_july_comparison.csv",
    index=False,
    encoding="utf-8-sig"
)

monthly.to_csv(
    BASE / "p65_p85_monthly_returns.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n===== 2026년 7월 제외 성과 비교 =====")
print(summary.round(2).to_string(index=False))

print("\n===== 7월 제외 월별 승패 비교 =====")

pivot = monthly.pivot(
    index="Month",
    columns="Strategy",
    values="Monthly_Return_Pct"
)

pivot = pivot.drop(index=EXCLUDED_MONTH)

print(
    "P85 수익률 우위 월:",
    int((pivot["P85"] > pivot["P65"]).sum())
)

print(
    "P65 수익률 우위 월:",
    int((pivot["P65"] > pivot["P85"]).sum())
)

print("\n분석 완료!")
