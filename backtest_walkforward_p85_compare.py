from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import backtest_walkforward_atr_compare as wf
import backtest_engine_dd_step as engine

OUT = Path("analysis")
df = wf.load_data()
prices_all = engine.load_prices()
calendar = pd.DatetimeIndex(sorted(prices_all["Date"].unique()))

start = int(len(calendar) * wf.START_FRACTION)
edges = np.linspace(start, len(calendar), wf.FOLDS + 1, dtype=int)

rows = []
all_trades = []

for i in range(wf.FOLDS):
    a, b = int(edges[i]), int(edges[i + 1])
    cutoff = calendar[a - wf.PURGE_DATES]

    eligible = df[
        (df["Date"] < cutoff)
        & df["Label"].isin([0, 1])
        & df["Label_ATR"].isin([0, 1])
    ].copy()

    dates = np.sort(eligible["Date"].unique())
    split = int(len(dates) * 0.8)
    val_start = pd.Timestamp(dates[split])
    fit_cutoff = pd.Timestamp(dates[max(0, split - 7)])

    fit = eligible[eligible["Date"] < fit_cutoff]
    val = eligible[eligible["Date"] >= val_start]

    test = df[
        (df["Date"] >= calendar[a])
        & (df["Date"] <= calendar[b - 1])
    ].copy()

    # 평가 마지막 날 이후 최대 5거래일의 청산 가격 확보
    end_pos = min(b + 4, len(calendar) - 1)
    prices = prices_all[
        (prices_all["Date"] >= calendar[a])
        & (prices_all["Date"] <= calendar[end_pos])
    ].copy()

    print(f"\n=== Fold {i+1} ===", flush=True)

    for name, label in wf.LABELS.items():
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )

        model.fit(fit[wf.TECH], fit[label].astype(int))

        val_p = model.predict_proba(val[wf.TECH])[:, 1]
        threshold = float(np.quantile(val_p, 0.85))
        test_p = model.predict_proba(test[wf.TECH])[:, 1]

        signals = test[["Date", "Code"]].rename(
            columns={"Code": "Ticker"}
        ).copy()
        signals["Probability"] = test_p

        signals = signals[
            signals["Probability"] >= threshold
        ].copy()

        raw_count = len(signals)

        signals = signals.merge(
            prices_all[["Date", "Ticker", "ATR14"]],
            on=["Date", "Ticker"],
            how="left",
            validate="one_to_one"
        )

        signals = signals[
            signals["ATR14"].notna()
            & (signals["ATR14"] > 0)
        ].copy()

        signals["Size_Factor"] = 1.0

        trades, equity = engine.run_backtest_v4(
            prices,
            signals[
                ["Date", "Ticker", "Probability",
                 "ATR14", "Size_Factor"]
            ],
            "SWING_ATR",
            risk_mode="DD_STOP_2",
            commission=0.00015,
            slippage=0.0005,
            sell_tax=0.0015
        )

        result = engine.summarize_results(
            trades, equity, "SWING_ATR"
        )

        result.update({
            "Fold": i + 1,
            "Model": name,
            "Threshold_P85": threshold,
            "Train_Rows": len(fit),
            "Validation_Rows": len(val),
            "Test_Rows": len(test),
            "Raw_Signal_Count": raw_count,
            "Signal_Count": len(signals),
            "Signal_Rate_Pct": len(signals) / len(test) * 100
        })
        rows.append(result)

        if not trades.empty:
            t = trades.copy()
            t.insert(0, "Fold", i + 1)
            t.insert(1, "Model", name)
            all_trades.append(t)

        print(
            f"{name} | P85={threshold:.4f} "
            f"| 신호={len(signals):,} "
            f"| 거래={result['Trades']} "
            f"| 수익률={result['Return_Pct']:.2f}% "
            f"| MDD={result['MDD_Pct']:.2f}%",
            flush=True
        )

summary = pd.DataFrame(rows)
summary.to_csv(
    OUT / "walkforward_p85_compare_summary.csv",
    index=False, encoding="utf-8-sig"
)

if all_trades:
    pd.concat(all_trades, ignore_index=True).to_csv(
        OUT / "walkforward_p85_compare_trades.csv",
        index=False, encoding="utf-8-sig"
    )

print("\n=== P85 Fold별 비교 ===")
print(summary[
    ["Fold", "Model", "Threshold_P85",
     "Signal_Count", "Trades",
     "Return_Pct", "MDD_Pct", "Win_Rate_Pct"]
].round(2).to_string(index=False))

overview = summary.groupby("Model").agg(
    Avg_Return_Pct=("Return_Pct", "mean"),
    Avg_MDD_Pct=("MDD_Pct", "mean"),
    Total_Trades=("Trades", "sum"),
    Total_Signals=("Signal_Count", "sum"),
    Positive_Folds=("Return_Pct", lambda x: int((x > 0).sum()))
).reset_index()

overview.to_csv(
    OUT / "walkforward_p85_compare_overview.csv",
    index=False, encoding="utf-8-sig"
)

print("\n=== P85 모델별 요약 ===")
print(overview.round(2).to_string(index=False))
print("\n기존 파일은 수정하지 않았습니다.")
