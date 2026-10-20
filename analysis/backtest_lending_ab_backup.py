
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score

from backtest_atr_comparison_v4 import (
    BASE_DIR,
    TRAIN_FILE,
    FEATURES,
    LENDING,
    LAG_FEATURE,
    THRESHOLD,
    load_prices,
    normalize_ticker,
    run_backtest_v4,
    summarize_results,
)


def main():
    prices = load_prices()
    prices["Ticker"] = prices["Ticker"].map(normalize_ticker)

    df = pd.read_csv(TRAIN_FILE, dtype={"Ticker": str})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Ticker"] = df["Ticker"].map(normalize_ticker)
    df = df.sort_values(["Ticker", "Date"]).copy()

    df[LAG_FEATURE] = df.groupby("Ticker")[LENDING].shift(1)

    for col in FEATURES + [LAG_FEATURE, "Label"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "Future_5D_Valid" in df.columns:
        valid = (
            df["Future_5D_Valid"]
            .astype(str)
            .str.strip()
            .str.lower()
            .isin(["true", "1", "1.0"])
        )
        df = df[valid].copy()

    df = df.dropna(
        subset=FEATURES + [LAG_FEATURE, "Label", "Date", "Ticker"]
    )
    df = df[df["Label"].isin([0, 1])].copy()
    df["Label"] = df["Label"].astype(int)

    dates = sorted(df["Date"].unique())
    split = int(len(dates) * 0.8)
    boundary = pd.Timestamp(dates[split])
    purge_dates = set(dates[max(0, split - 5):split])

    train = df[
        (df["Date"] < boundary)
        & (~df["Date"].isin(purge_dates))
    ].copy()
    test = df[df["Date"] >= boundary].copy()

    atr = prices[["Date", "Ticker", "ATR14"]].copy()
    results = []

    print("평가 시작일:", boundary.date())

    experiments = {
        "A_PRICE_ONLY": FEATURES,
        "C_WITH_LENDING": FEATURES + [LAG_FEATURE],
    }

    for name, columns in experiments.items():
        print(f"\n===== {name} =====")

        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )

        model.fit(train[columns], train["Label"])
        probability = model.predict_proba(test[columns])[:, 1]

        ap = average_precision_score(test["Label"], probability)

        candidate = test[["Date", "Ticker"]].copy()
        candidate["Probability"] = probability
        candidate = candidate[
            candidate["Probability"] >= THRESHOLD
        ].copy()

        if name == "A_PRICE_ONLY":
    candidate = candidate[
        ~candidate["Ticker"].isin(["005930", "005935"])
    ].copy()

        candidate = candidate.merge(
            atr,
            on=["Date", "Ticker"],
            how="left",
            validate="many_to_one",
        )
        candidate = candidate.dropna(subset=["ATR14"])
        candidate = candidate[candidate["ATR14"] > 0]

        trades, equity = run_backtest_v4(
            prices,
            candidate,
            "SWING_ATR",
        )

        summary = summarize_results(trades, equity, name)
        summary["Average_Precision"] = round(ap, 4)
        summary["Candidate_Signals"] = len(candidate)
        results.append(summary)

        trades.to_csv(
            BASE_DIR / f"{name.lower()}_trades.csv",
            index=False,
            encoding="utf-8-sig",
        )
        equity.to_csv(
            BASE_DIR / f"{name.lower()}_equity.csv",
            index=False,
            encoding="utf-8-sig",
        )

        print(pd.Series(summary).to_string())

    result = pd.DataFrame(results)
    result.to_csv(
        BASE_DIR / "v4_lending_ab_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\n===== 최종 비교 =====")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
