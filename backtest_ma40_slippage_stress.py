from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

import backtest_walkforward_a_ma_dd as wf
import backtest_engine_ma_dd as ma40
import backtest_engine_ma60_dd_step as ma60

ma40.INITIAL_CASH = ma60.INITIAL_CASH = 1_000_000.0
ma40.MAX_POSITIONS = ma60.MAX_POSITIONS = 2

# 기존 파일을 수정하지 않고, 동일한 Walk-forward 신호를 사용한다.
df = wf.load_technical_universe()
dates = np.sort(df["Date"].unique())
prices_all = ma40.load_prices()
calendar = np.sort(prices_all["Date"].unique())

# MA60 엔진과 동일한 매매/리스크 로직에 MA40 필터만 적용
import types

ma40_step = types.ModuleType("ma40_step")
ma40_step.__dict__.update(ma60.__dict__)
ma40_step.run_backtest_v4 = types.FunctionType(
    ma60.run_backtest_v4.__code__,
    ma40_step.__dict__,
    name="run_backtest_v4",
    argdefs=ma60.run_backtest_v4.__defaults__,
)
ma40_step.load_ma60_filter = ma40.load_ma40_filter
ma40_step.INITIAL_CASH = 1_000_000.0
ma40_step.MAX_POSITIONS = 2

strategies = [
    ("MA40_DD_STOP_2", ma40_step, "DD_STOP_2"),
    ("MA60_DD_STOP_2", ma60, "DD_STOP_2"),
]

all_signals = []
fold_info = []


for fold, train_cutoff, test_start, test_end in wf.fold_ranges(dates):
    train = df[df["Date"] < train_cutoff]
    test = df[(df["Date"] >= test_start) & (df["Date"] <= test_end)]

    if train.empty or test.empty:
        raise ValueError(f"Fold {fold}: 학습 또는 평가 데이터가 비어 있음")

    print(f"\n[FOLD {fold}] {pd.Timestamp(test_start).date()} ~ "
          f"{pd.Timestamp(test_end).date()}", flush=True)

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(train[wf.TECH], train["Label"].astype(int))

    signals = test[["Date", "Code"]].rename(
        columns={"Code": "Ticker"}
    ).copy()
    signals["Probability"] = model.predict_proba(test[wf.TECH])[:, 1]
    signals = signals[signals["Probability"] >= wf.THRESHOLD].copy()

    end_pos = np.searchsorted(
        calendar, np.datetime64(test_end), side="right"
    )
    end_pos = min(end_pos + 4, len(calendar) - 1)
    end_price_date = calendar[end_pos]

    prices = prices_all[
        (prices_all["Date"] >= test_start)
        & (prices_all["Date"] <= end_price_date)
    ].copy()

    signals = signals.merge(
        prices[["Date", "Ticker", "ATR14"]],
        on=["Date", "Ticker"],
        how="left",
        validate="one_to_one",
    )
    signals = signals[
        signals["ATR14"].notna() & (signals["ATR14"] > 0)
    ].copy()

    signals["Size_Factor"] = 1.0
    print(f"공통 AI 신호: {len(signals):,}건", flush=True)

    all_signals.append(signals)
    fold_info.append((fold, test_start, test_end))

# 4개 폴드에서 각각 학습한 AI 신호를 하나로 결합
signals_all = pd.concat(all_signals, ignore_index=True)
signals_all = signals_all.sort_values(
    ["Date", "Ticker"]
).reset_index(drop=True)

if signals_all.duplicated(["Date", "Ticker"]).any():
    raise ValueError("폴드 간 중복 신호 발생")

first_date = min(x[1] for x in fold_info)
last_date = max(x[2] for x in fold_info)

# 마지막 폴드 종료 이후 최대 5거래일의 청산 여유
end_pos = np.searchsorted(
    calendar, np.datetime64(last_date), side="right"
)
end_pos = min(end_pos + 4, len(calendar) - 1)

prices_continuous = prices_all[
    (prices_all["Date"] >= first_date)
    & (prices_all["Date"] <= calendar[end_pos])
].copy()

print(
    f"\\n[연속 운용] {pd.Timestamp(first_date).date()} ~ "
    f"{pd.Timestamp(calendar[end_pos]).date()}"
)
print(f"전체 AI 신호: {len(signals_all):,}건")


print("\n========== MA40 슬리피지 스트레스 테스트 ==========", flush=True)

rows = []

for slip in [0.0005, 0.001, 0.002, 0.005]:
    trades, equity = ma40_step.run_backtest_v4(
        prices_continuous,
        signals_all.copy(),
        wf.STRATEGY,
        risk_mode="DD_STOP_2",
        slippage=slip,
    )

    summary = ma40_step.summarize_results(
        trades, equity, wf.STRATEGY
    )

    rows.append({
        "Slippage_Pct": slip * 100,
        "Final_Equity": summary["Final_Equity"],
        "Return_Pct": summary["Return_Pct"],
        "MDD_Pct": summary["MDD_Pct"],
        "Trades": summary["Trades"],
        "Win_Rate_Pct": summary["Win_Rate_Pct"],
    })

result = pd.DataFrame(rows)

print("\n[슬리피지별 결과]")
print(result.round(2).to_string(index=False))

output = Path("analysis/ma40_slippage_stress.csv")
if output.exists():
    raise FileExistsError(f"기존 결과 보호: {output}")

result.to_csv(output, index=False, encoding="utf-8-sig")
print(f"\n저장 완료: {output}")
