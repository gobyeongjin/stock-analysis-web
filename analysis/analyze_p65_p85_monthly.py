from pathlib import Path
import pandas as pd
import numpy as np

BASE = Path(__file__).resolve().parent

strategies = ["P65", "P85"]
monthly_results = []
trade_results = []
trade_data = {}

for strategy in strategies:
    equity = pd.read_csv(
        BASE / f"threshold_{strategy}_equity.csv"
    )

    trades = pd.read_csv(
        BASE / f"threshold_{strategy}_trades.csv",
        dtype={"Ticker": str}
    )

    print(f"\n===== {strategy} 데이터 구조 =====")
    print("Equity 컬럼:", equity.columns.tolist())
    print("Trades 컬럼:", trades.columns.tolist())

    # 날짜 컬럼 확인
    date_col = next(
        (c for c in ["Date", "date"] if c in equity.columns),
        None
    )
    value_col = next(
        (c for c in ["Equity", "equity", "Total_Equity"]
         if c in equity.columns),
        None
    )

    if date_col is None or value_col is None:
        raise RuntimeError(
            f"{strategy}: Equity 날짜/평가자산 컬럼 확인 필요"
        )

    equity[date_col] = pd.to_datetime(equity[date_col])
    equity[value_col] = pd.to_numeric(
        equity[value_col], errors="coerce"
    )

    equity = (
        equity.sort_values(date_col)
        .drop_duplicates(date_col, keep="last")
        .dropna(subset=[value_col])
    )

    equity["Month"] = equity[date_col].dt.to_period("M")

    # 전체 기간의 일별 평가자산 기준 월말 수익률
    monthly_equity = equity.groupby("Month")[value_col].last()

    previous_equity = monthly_equity.shift(1)
    previous_equity.iloc[0] = 1_000_000

    monthly_returns = (
        monthly_equity / previous_equity - 1
    ) * 100

    # 각 월의 시작 자산을 포함해 월별 MDD 계산
    for month, group in equity.groupby("Month"):
        start_value = previous_equity.loc[month]

        values = pd.concat([
            pd.Series([start_value]),
            group[value_col].reset_index(drop=True)
        ], ignore_index=True)

        running_peak = values.cummax()
        drawdown = (values / running_peak - 1) * 100

        monthly_results.append({
            "Strategy": strategy,
            "Month": str(month),
            "Monthly_Return_Pct": monthly_returns.loc[month],
            "Monthly_MDD_Pct": drawdown.min(),
            "Month_End_Equity": monthly_equity.loc[month],
        })

    # 청산월 기준 거래 분석
    trades["Exit_Date"] = pd.to_datetime(trades["Exit_Date"])
    trades["Month"] = trades["Exit_Date"].dt.to_period("M")
    trades["Is_Stop"] = (
        trades["Exit_Reason"]
        .astype(str)
        .str.contains("손절", na=False)
    )

    trade_data[strategy] = trades

    for month, group in trades.groupby("Month"):
        stops = group[group["Is_Stop"]]

        trade_results.append({
            "Strategy": strategy,
            "Month": str(month),
            "Trades": len(group),
            "Stops": len(stops),
            "Stop_PnL": stops["PnL"].sum(),
            "Realized_PnL": group["PnL"].sum(),
            "Win_Rate_Pct": (
                (group["PnL"] > 0).mean() * 100
            ),
        })

monthly_df = pd.DataFrame(monthly_results)
trade_df = pd.DataFrame(trade_results)

combined = monthly_df.merge(
    trade_df,
    on=["Strategy", "Month"],
    how="left"
)

for col in ["Trades", "Stops", "Stop_PnL", "Realized_PnL"]:
    combined[col] = combined[col].fillna(0)

combined.to_csv(
    BASE / "p65_p85_monthly_comparison.csv",
    index=False,
    encoding="utf-8-sig"
)

# 월별 수익률 비교
returns = combined.pivot(
    index="Month",
    columns="Strategy",
    values="Monthly_Return_Pct"
)

returns["P85_minus_P65"] = (
    returns["P85"] - returns["P65"]
)

print("\n===== 월별 계좌 수익률 비교 (%) =====")
print(returns.round(2).to_string())

# 월별 MDD 비교
mdd = combined.pivot(
    index="Month",
    columns="Strategy",
    values="Monthly_MDD_Pct"
)

mdd["P85_minus_P65"] = mdd["P85"] - mdd["P65"]

print("\n===== 월별 MDD 비교 (%) =====")
print(mdd.round(2).to_string())

# 손절 비교
stops = combined.pivot(
    index="Month",
    columns="Strategy",
    values="Stops"
).fillna(0)

stops["P65_minus_P85"] = stops["P65"] - stops["P85"]

print("\n===== 월별 손절 횟수 비교 =====")
print(stops.astype(int).to_string())

# 전체 기간 손절 통계
print("\n===== 전체 손절 통계 =====")

for strategy in strategies:
    trades = trade_data[strategy]
    stop_trades = trades[trades["Is_Stop"]]

    print(
        f"{strategy}: "
        f"거래 {len(trades)}건, "
        f"손절 {len(stop_trades)}건, "
        f"손절률 {len(stop_trades)/len(trades)*100:.2f}%, "
        f"손절 손익 {stop_trades['PnL'].sum():,.0f}원"
    )

# 거래 식별용 키: 동일 종목의 여러 거래를 구별
key = ["Ticker", "Signal_Date", "Entry_Date"]

p65 = trade_data["P65"].copy()
p85 = trade_data["P85"].copy()

for df in [p65, p85]:
    for col in ["Signal_Date", "Entry_Date"]:
        df[col] = pd.to_datetime(df[col]).dt.normalize()

# 중복 키가 있는 경우 일대일 비교가 불가능하므로 확인
for name, df in [("P65", p65), ("P85", p85)]:
    if df.duplicated(key).any():
        raise RuntimeError(
            f"{name}: 거래 식별 키 중복 발견. 비교 로직 확인 필요."
        )

matched = p65.merge(
    p85[key],
    on=key,
    how="left",
    indicator=True
)

excluded = matched[
    matched["_merge"] == "left_only"
].copy()

excluded.to_csv(
    BASE / "p65_only_trades.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n===== P65에만 존재하는 거래 =====")
print("거래 수:", len(excluded))

if len(excluded):
    print(
        "평균 거래 수익률:",
        round(excluded["Return_Pct"].mean(), 2), "%"
    )
    print(
        "승률:",
        round((excluded["PnL"] > 0).mean() * 100, 2), "%"
    )
    print(
        "손절률:",
        round(excluded["Is_Stop"].mean() * 100, 2), "%"
    )
    print(
        "실현손익 합계:",
        round(excluded["PnL"].sum()), "원"
    )

    print("\n확률 구간별 P65 전용 거래")
    excluded["Prob_Group"] = pd.cut(
        excluded["Probability"],
        bins=[0.65, 0.75, 0.85, 0.95, np.inf],
        labels=["0.65~0.75", "0.75~0.85",
                "0.85~0.95", "0.95+"],
        right=False
    )

    print(
        excluded.groupby(
            "Prob_Group", observed=True
        ).agg(
            Trades=("PnL", "size"),
            Avg_Return_Pct=("Return_Pct", "mean"),
            Total_PnL=("PnL", "sum")
        ).round(2).to_string()
    )

print("\n분석 완료!")
