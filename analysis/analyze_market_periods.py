
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

FILES = {
    "MA10": "market_ma10_equity.csv",
    "MA20": "market_ma20_equity.csv",
    "MA40": "market_ma40_equity.csv",
    "MA60": "market_ma60_equity.csv",
}

results = []
monthly_results = []

for name, filename in FILES.items():
    df = pd.read_csv(BASE_DIR / filename)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").copy()

    df = df[
    df["Date"] >= "2025-10-20"].copy()

    # 월말 평가자산
    monthly = (
        df.set_index("Date")["Equity"]
        .resample("ME")
        .last()
        .dropna()
    )

    # 첫 달 수익률 계산을 위한 초기 자산
    previous = monthly.shift(1)
    previous.iloc[0] = 10_000_000

    monthly_return = (monthly / previous - 1) * 100

    for date, value in monthly_return.items():
        monthly_results.append({
            "Strategy": name,
            "Month": date.strftime("%Y-%m"),
            "Return_Pct": round(value, 2),
        })

    # 2026년 7월 분석
    july = df[
        (df["Date"] >= "2026-07-01") &
        (df["Date"] <= "2026-07-31")
    ].copy()

    if not july.empty:
        july_start = df[
            df["Date"] < "2026-07-01"
        ]["Equity"]

        start_equity = (
            july_start.iloc[-1]
            if not july_start.empty
            else 10_000_000
        )

        end_equity = july["Equity"].iloc[-1]

        july_return = (
            end_equity / start_equity - 1
        ) * 100

        july_peak = pd.concat([
            pd.Series([start_equity]),
            july["Equity"].reset_index(drop=True)
        ]).cummax()

        july_values = pd.concat([
            pd.Series([start_equity]),
            july["Equity"].reset_index(drop=True)
        ])

        july_mdd = (
            july_values / july_peak - 1
        ).min() * 100

        results.append({
            "Strategy": name,
            "July_Return_Pct": round(july_return, 2),
            "July_MDD_Pct": round(july_mdd, 2),
        })

print("\n===== 2026년 7월 성과 =====")
print(pd.DataFrame(results).to_string(index=False))

print("\n===== 월별 수익률 =====")

monthly_df = pd.DataFrame(monthly_results)

pivot = monthly_df.pivot(
    index="Month",
    columns="Strategy",
    values="Return_Pct"
)

print(pivot.to_string())

pivot.to_csv(
    BASE_DIR / "market_monthly_returns.csv",
    encoding="utf-8-sig"
)
