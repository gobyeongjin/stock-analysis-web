
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score

import backtest_stop8_test as base
import backtest_market_ma as bt


BASE_DIR = Path(__file__).resolve().parent

INITIAL_CAPITAL = 10_000_000
THRESHOLD = 0.65
MARKET_MA = 40

INITIAL_TRAIN_RATIO = 0.60
N_FOLDS = 4
PURGE_DAYS = 5


def prepare_walkforward_data():
    # 기존 데이터 전처리 재사용
    # prepare_data()가 반환하는 train/test를 다시 합침
    prices, train, test = base.prepare_data()

    df = pd.concat(
        [train, test],
        ignore_index=True
    )

    df["Date"] = pd.to_datetime(
        df["Date"]
    ).dt.normalize()

    df["Ticker"] = df["Ticker"].map(
        bt.normalize_ticker
    )

    df = df.sort_values(
        ["Date", "Ticker"]
    ).reset_index(drop=True)

    return prices, df


def generate_walkforward_signals(prices, df):

    dates = pd.DatetimeIndex(
        sorted(df["Date"].unique())
    )

    n_dates = len(dates)
    initial_end = int(
        n_dates * INITIAL_TRAIN_RATIO
    )

    if initial_end <= PURGE_DAYS:
        raise ValueError("초기 학습 기간이 너무 짧습니다.")

    test_dates = dates[initial_end:]
    fold_dates = np.array_split(
        test_dates, N_FOLDS
    )

    atr = prices[
        ["Date", "Ticker", "ATR14"]
    ].copy()

    atr["Date"] = pd.to_datetime(
        atr["Date"]
    ).dt.normalize()

    all_signals = []
    fold_metrics = []

    for fold_num, current_dates in enumerate(
        fold_dates, start=1
    ):

        if len(current_dates) == 0:
            continue

        test_start = pd.Timestamp(
            current_dates[0]
        )

        test_end = pd.Timestamp(
            current_dates[-1]
        )

        # 평가 시작 이전 5거래일 제외
        start_idx = dates.get_loc(test_start)
        train_end_idx = start_idx - PURGE_DAYS

        train_dates = dates[:train_end_idx]

        train = df[
            df["Date"].isin(train_dates)
        ].copy()

        test = df[
            df["Date"].isin(current_dates)
        ].copy()

        if train.empty or test.empty:
            raise ValueError(
                f"{fold_num}차 학습 또는 평가 데이터가 없습니다."
            )

        if train["Label"].nunique() < 2:
            raise ValueError(
                f"{fold_num}차 학습 데이터에 두 클래스가 필요합니다."
            )

        print(
            f"\n===== {fold_num}차 워크포워드 ====="
        )
        print(
            "학습:",
            train["Date"].min().date(),
            "~",
            train["Date"].max().date()
        )
        print(
            "평가:",
            test_start.date(),
            "~",
            test_end.date()
        )
        print("학습 행:", len(train))
        print("평가 행:", len(test))

        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )

        model.fit(
            train[base.FEATURES],
            train["Label"]
        )

        probability = model.predict_proba(
            test[base.FEATURES]
        )[:, 1]

        ap = average_precision_score(
            test["Label"],
            probability
        )

        candidate = test[
            ["Date", "Ticker", "Volume_Ratio"]
        ].copy()

        candidate["Probability"] = probability

        candidate = candidate[
            candidate["Probability"] >= THRESHOLD
        ].copy()

        candidate = candidate[
            ~candidate["Ticker"].isin(
                base.EXCLUDED_TICKERS
            )
        ].copy()

        candidate = candidate.merge(
            atr,
            on=["Date", "Ticker"],
            how="left",
            validate="many_to_one"
        )

        candidate["ATR14"] = pd.to_numeric(
            candidate["ATR14"],
            errors="coerce"
        )

        candidate["Volume_Ratio"] = pd.to_numeric(
            candidate["Volume_Ratio"],
            errors="coerce"
        )

        candidate = candidate[
            candidate["ATR14"].notna()
            & (candidate["ATR14"] > 0)
        ].copy()

        candidate["Fold"] = fold_num

        all_signals.append(candidate)

        fold_metrics.append({
            "Fold": fold_num,
            "Train_Rows": len(train),
            "Test_Rows": len(test),
            "Test_Start": test_start,
            "Test_End": test_end,
            "Average_Precision": round(ap, 4),
            "Signals": len(candidate),
        })

        print("Average Precision:", round(ap, 4))
        print("신호 수:", len(candidate))

    signals = pd.concat(
        all_signals,
        ignore_index=True
    )

    metrics = pd.DataFrame(fold_metrics)

    return signals, metrics


def main():

    prices, df = prepare_walkforward_data()

    signals, metrics = generate_walkforward_signals(
        prices, df
    )

    # 워크포워드 평가 기간
    start_date = metrics["Test_Start"].min()
    end_date = metrics["Test_End"].max()

    # 신호 없는 거래일도 포함하도록
    # 가격 데이터는 평가 기간 전체 사용
    # 첫 평가일 다음 날 매수 가능
    backtest_prices = prices[
        (prices["Date"] >= start_date)
        & (prices["Date"] <= end_date)
    ].copy()

    strategies = {
        "A_NO_FILTER": signals.copy(),

        "B_VOLUME_1_0": signals[
            signals["Volume_Ratio"] <= 1.0
        ].copy(),
    }

    original_cash = bt.INITIAL_CASH
    original_filter = bt.USE_MARKET_FILTER
    original_ma = bt.MARKET_MA_DAYS

    results = []

    try:
        bt.INITIAL_CASH = INITIAL_CAPITAL
        bt.USE_MARKET_FILTER = True
        bt.MARKET_MA_DAYS = MARKET_MA

        for name, selected in strategies.items():

            print(f"\n===== {name} =====")
            print("전체 신호:", len(selected))

            trades, equity = bt.run_backtest_v4(
                backtest_prices,
                selected,
                "SWING_ATR"
            )

            summary = bt.summarize_results(
                trades,
                equity,
                name
            )

            summary["Signal_Count"] = len(selected)
            summary["Unique_Stocks"] = (
                selected["Ticker"].nunique()
            )

            results.append(summary)

            trades.to_csv(
                BASE_DIR / f"wf_{name}_trades.csv",
                index=False,
                encoding="utf-8-sig"
            )

            equity.to_csv(
                BASE_DIR / f"wf_{name}_equity.csv",
                index=False,
                encoding="utf-8-sig"
            )

    finally:
        bt.INITIAL_CASH = original_cash
        bt.USE_MARKET_FILTER = original_filter
        bt.MARKET_MA_DAYS = original_ma

    result = pd.DataFrame(results)

    print("\n===== 워크포워드 모델 성능 =====")
    print(metrics.to_string(index=False))

    print("\n===== 워크포워드 전략 비교 =====")
    columns = [
        "Strategy",
        "Signal_Count",
        "Unique_Stocks",
        "Return_Pct",
        "MDD_Pct",
        "Trades",
        "Win_Rate_Pct",
        "Final_Equity"
    ]

    print(result[columns].to_string(index=False))

    metrics.to_csv(
        BASE_DIR / "wf_model_metrics.csv",
        index=False,
        encoding="utf-8-sig"
    )

    result.to_csv(
        BASE_DIR / "wf_strategy_comparison.csv",
        index=False,
        encoding="utf-8-sig"
    )

    signals.to_csv(
        BASE_DIR / "wf_all_signals.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("\n워크포워드 분석 완료")


if __name__ == "__main__":
    main()
