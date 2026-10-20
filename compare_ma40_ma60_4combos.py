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
    ("MA40_DD_HALF_15", ma40_step, "DD_HALF_15"),
    ("MA40_DD_STOP_2", ma40_step, "DD_STOP_2"),
    ("MA60_DD_HALF_15", ma60, "DD_HALF_15"),
    ("MA60_DD_STOP_2", ma60, "DD_STOP_2"),
]

results = []

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

    for name, engine, risk_mode in strategies:
        print(f"  실행 중: {name}", flush=True)

        trades, equity = engine.run_backtest_v4(
            prices,
            signals.copy(),
            wf.STRATEGY,
            risk_mode=risk_mode,
        )

        summary = engine.summarize_results(
            trades, equity, wf.STRATEGY
        )
        summary["Fold"] = fold
        summary["Strategy"] = name
        summary["Signal_Count"] = len(signals)
        results.append(summary)

        print(
            f"  {name}: 수익률 {summary.get('Return_Pct', 'N/A')}% "
            f"/ MDD {summary.get('MDD_Pct', 'N/A')}% "
            f"/ 거래 {len(trades)}건",
            flush=True,
        )

out = pd.DataFrame(results)
output = Path("analysis/compare_ma40_ma60_4combos.csv")
if output.exists():
    raise FileExistsError(f"기존 결과 보호: {output}")

out.to_csv(output, index=False, encoding="utf-8-sig")

print("\n========== 비교 결과 ==========")
print(out.to_string(index=False))
print(f"\n결과 저장: {output}")
