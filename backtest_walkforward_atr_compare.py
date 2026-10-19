"""Independent Walk-Forward comparison: original Label vs ATR Label v2."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import backtest_engine_dd_step as engine

ROOT = Path(__file__).resolve().parent
ANALYSIS = ROOT / "analysis"

TECH = [
    "MA20_Gap", "MA60_Gap", "MA20_MA60_Gap",
    "MA20_Change", "MA60_Change",
    "High_20D_Gap", "Low_20D_Gap",
    "RSI", "RSI_Change",
    "Volume_Ratio", "Volume_Ratio_5D",
    "Return_1D", "Return_3D", "Return_5D",
    "Return_10D", "Volatility_20D",
    "Volatility_Change", "Close_Position"
]

THRESHOLD = 0.65
PURGE_DATES = 7
FOLDS = 4
START_FRACTION = 0.60

LABELS = {
    "ORIGINAL": "Label",
    "ATR_V2": "Label_ATR"
}


def load_data():
    path = ANALYSIS / "training_data_atr_label_v2.csv"
    df = pd.read_csv(
        path, dtype={"Ticker": str}, low_memory=False
    )

    required = TECH + ["Date", "Ticker", "Label", "Label_ATR"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"필수 컬럼 누락: {missing}")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Code"] = (
        df["Ticker"].str.split(".", regex=False)
        .str[0].str.zfill(6)
    )

    for col in TECH + ["Label", "Label_ATR"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=TECH + ["Date", "Code"])

    df = df.sort_values(["Date", "Ticker"])
    df = df.drop_duplicates(
        ["Date", "Code"], keep="last"
    ).reset_index(drop=True)

    return df


def main():
    df = load_data()
    prices_all = engine.load_prices()

    calendar = pd.DatetimeIndex(
        sorted(prices_all["Date"].unique())
    )

    start = int(len(calendar) * START_FRACTION)
    edges = np.linspace(
        start, len(calendar), FOLDS + 1, dtype=int
    )

    summaries = []
    all_trades = []

    print("=== Label A/B Walk-Forward ===")
    print("전체 피처 데이터:", len(df))
    print("가격 거래일:", len(calendar))
    print("Purge:", PURGE_DATES, "거래일")
    print("Threshold:", THRESHOLD)
    print("초기 자본:", engine.INITIAL_CASH)

    for fold in range(FOLDS):
        a, b = int(edges[fold]), int(edges[fold + 1])

        if a <= PURGE_DATES:
            raise ValueError("Purge 적용 불가")

        test_start = calendar[a]
        test_end = calendar[b - 1]
        train_cutoff = calendar[a - PURGE_DATES]

        # Training labels must be observable before test starts.
        train = df[
            (df["Date"] < train_cutoff)
            & df["Label"].isin([0, 1])
            & df["Label_ATR"].isin([0, 1])
        ].copy()

        # Never filter test rows by future labels or Future_5D_Valid.
        test = df[
            (df["Date"] >= test_start)
            & (df["Date"] <= test_end)
        ].copy()

        if train.empty or test.empty:
            raise ValueError(f"Fold {fold+1}: 학습/평가 데이터 부족")

        print(
            f"\n[Fold {fold+1}] "
            f"학습 {len(train):,} / 평가 {len(test):,}"
            f" | {test_start.date()} ~ {test_end.date()}",
            flush=True
        )

        # Reserve following trading days for entries and exits.
        end_pos = min(b + 4, len(calendar) - 1)
        end_price_date = calendar[end_pos]

        prices = prices_all[
            (prices_all["Date"] >= test_start)
            & (prices_all["Date"] <= end_price_date)
        ].copy()

        for model_name, label_col in LABELS.items():
            model = RandomForestClassifier(
                n_estimators=200,
                max_depth=8,
                min_samples_leaf=10,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            )

            model.fit(
                train[TECH],
                train[label_col].astype(int)
            )

            probability = model.predict_proba(
                test[TECH]
            )[:, 1]

            signals = test[["Date", "Code"]].rename(
                columns={"Code": "Ticker"}
            ).copy()

            signals["Probability"] = probability
            signals = signals[
                signals["Probability"] >= THRESHOLD
            ].copy()

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

            print(
                f"  {model_name}: "
                f"신호 {len(signals):,}건, 백테스트 중...",
                flush=True
            )

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
                "Fold": fold + 1,
                "Model": model_name,
                "Train_Rows": len(train),
                "Test_Rows": len(test),
                "Test_Start": test_start.date(),
                "Test_End": test_end.date(),
                "Signal_Count": len(signals),
                "Signal_Rate_Pct": (
                    len(signals) / len(test) * 100
                ),
                "Threshold": THRESHOLD,
                "Purge_Dates": PURGE_DATES
            })

            summaries.append(result)

            if not trades.empty:
                trades = trades.copy()
                trades.insert(0, "Fold", fold + 1)
                trades.insert(1, "Model", model_name)
                all_trades.append(trades)

    summary = pd.DataFrame(summaries)

    summary_path = (
        ANALYSIS / "walkforward_atr_compare_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig"
    )

    if all_trades:
        pd.concat(
            all_trades, ignore_index=True
        ).to_csv(
            ANALYSIS / "walkforward_atr_compare_trades.csv",
            index=False,
            encoding="utf-8-sig"
        )

    cols = [
        "Fold", "Model", "Signal_Count",
        "Signal_Rate_Pct", "Trades",
        "Return_Pct", "MDD_Pct", "Win_Rate_Pct"
    ]

    print("\n=== Fold별 비교 ===")
    print(
        summary[cols].round(2).to_string(index=False)
    )

    overview = (
        summary.groupby("Model", as_index=False)
        .agg(
            Avg_Return_Pct=("Return_Pct", "mean"),
            Avg_MDD_Pct=("MDD_Pct", "mean"),
            Total_Trades=("Trades", "sum"),
            Total_Signals=("Signal_Count", "sum"),
            Positive_Folds=(
                "Return_Pct",
                lambda x: int((x > 0).sum())
            )
        )
    )

    overview.to_csv(
        ANALYSIS / "walkforward_atr_compare_overview.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("\n=== 모델별 요약 ===")
    print(overview.round(2).to_string(index=False))
    print("\n저장:", summary_path)
    print("주의: Fold별 평균 수익률은 연속 복리 수익률이 아닙니다.")
    print("주의: 동일 데이터에서 반복 실험한 탐색적 결과입니다.")


if __name__ == "__main__":
    main()
