from pathlib import Path
import pandas as pd
import numpy as np
import matched_stop_loss_backtest as bt

BASE = Path(__file__).resolve().parents[1]
RESULTS = BASE / "results"
RESULTS.mkdir(exist_ok=True)

groups, date_maps, candidates = bt.load_data()

# 시나리오 1: 기존 로직
strict_groups = groups

# 시나리오 2: 급락일을 더 보수적으로 처리하는 가정
# 실제 매도 불가를 확정하는 규칙이 아니라 민감도 분석용
conservative_groups = {}
for ticker, original in groups.items():
    g = original.copy()
    g["Locked_Lower"] = (
        (g["Daily_Change"] <= -0.295)
        & (g["Low"] == g["Close"])
    )
    conservative_groups[ticker] = g

scenarios = {
    "Current": strict_groups,
    "Conservative_Crash": conservative_groups,
}

accepted = []
skips = []
busy_until = {}

# 모든 시나리오에서 공통으로 사용할 진입 집합 구성
for _, signal in candidates.iterrows():
    ticker = signal["Ticker"]
    signal_date = signal["Date"]

    if ticker not in strict_groups:
        continue

    g = strict_groups[ticker]
    signal_idx = date_maps[ticker].get(signal_date)

    if signal_idx is None:
        continue

    entry_idx = signal_idx + 1
    if entry_idx >= len(g):
        continue

    if entry_idx <= busy_until.get(ticker, -1):
        continue

    entry = g.iloc[entry_idx]
    if not bt.can_trade_open(entry):
        continue
    if entry["Locked_Upper"] or entry["Locked_Lower"]:
        continue

    if entry_idx + bt.HOLDING_DAYS >= len(g):
        continue

    # 두 시나리오의 모든 전략 결과를 계산해
    # 가장 늦은 청산 이후에만 다음 진입 허용
    outcomes = {}
    for scenario, scenario_groups in scenarios.items():
        sg = scenario_groups[ticker]
        for strategy, stop_pct in bt.STRATEGIES.items():
            outcomes[(scenario, strategy)] = bt.simulate_one(
                sg, entry_idx, stop_pct
            )

    common_exit = max(
        r["Exit_Index"] for r in outcomes.values()
    )
    busy_until[ticker] = common_exit

    entry_id = f"{ticker}_{signal_date.strftime('%Y%m%d')}"
    accepted.append(entry_id)

    for (scenario, strategy), result in outcomes.items():
        row = {
            "Scenario": scenario,
            "Strategy": strategy,
            "Entry_ID": entry_id,
            "Signal_Date": signal_date,
            "Ticker": ticker,
            "Stock_Name": signal.get("Stock_Name", ""),
            "AI_Score": signal["AI_Score"],
        }
        row.update({
            k: v for k, v in result.items()
            if k != "Exit_Index"
        })
        skips.append(row)

trades = pd.DataFrame(skips)
summaries = []

for (scenario, strategy), df in trades.groupby(
    ["Scenario", "Strategy"], sort=False
):
    closed = df[df["Net_Return_Pct"].notna()]
    r = closed["Net_Return_Pct"]

    summaries.append({
        "Scenario": scenario,
        "Strategy": strategy,
        "Common_Entry_Count": df["Entry_ID"].nunique(),
        "Closed_Trades": len(closed),
        "Unresolved": int(df["Exit_Reason"].eq("Unresolved").sum()),
        "Win_Rate_Pct": (r.gt(0).mean() * 100) if len(r) else np.nan,
        "Average_Net_Return_Pct": r.mean() if len(r) else np.nan,
        "Median_Net_Return_Pct": r.median() if len(r) else np.nan,
        "Worst_Return_Pct": r.min() if len(r) else np.nan,
        "Best_Return_Pct": r.max() if len(r) else np.nan,
        "Gap_Stop_Count": int(closed["Exit_Reason"].eq("Gap_Stop").sum()),
        "Stop_Loss_Count": int(closed["Exit_Reason"].eq("Stop_Loss").sum()),
        "Delayed_Exit_Count": int(closed["Exit_Reason"].isin([
            "Delayed_Time_Exit",
            "Delayed_Stop_After_Unfillable_Breach"
        ]).sum()),
        "Lower_Lock_Breach_Count": int(
            df["Lower_Lock_Breach"].fillna(False).sum()
        ),
    })

summary = pd.DataFrame(summaries)

summary.to_csv(
    RESULTS / "crash_scenario_comparison_summary.csv",
    index=False, encoding="utf-8-sig"
)
trades.to_csv(
    RESULTS / "crash_scenario_comparison_trades.csv",
    index=False, encoding="utf-8-sig"
)

print("공통 진입 수:", len(accepted))
print("\n=== 시나리오별 결과 ===")
print(summary.round(3).to_string(index=False))
print("\n저장 완료:")
print(RESULTS / "crash_scenario_comparison_summary.csv")
print(RESULTS / "crash_scenario_comparison_trades.csv")
print("\n주의: 보수적 시나리오는 실제 체결 가능성을 입증하지 않는 가정입니다.")
