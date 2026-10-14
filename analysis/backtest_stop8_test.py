from pathlib import Path

import pandas as pd
import backtest_atr_8pct as bt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score


# ==========================================
# 1. 설정
# ==========================================

BASE_DIR = bt.BASE_DIR
TRAIN_FILE = bt.TRAIN_FILE

FEATURES = bt.FEATURES
LENDING = bt.LENDING
LAG_FEATURE = bt.LAG_FEATURE
THRESHOLD = bt.THRESHOLD

EXCLUDED_TICKERS = ["005930", "005935"]

INITIAL_CAPITALS = [
    10_000_000,
]


# ==========================================
# 2. 학습 및 평가 데이터 준비
# ==========================================

def prepare_data():

    prices = bt.load_prices()
    prices["Ticker"] = prices["Ticker"].map(
        bt.normalize_ticker
    )

    df = pd.read_csv(
        TRAIN_FILE,
        dtype={"Ticker": str}
    )

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    df["Ticker"] = df["Ticker"].map(
        bt.normalize_ticker
    )

    df = df.sort_values(
        ["Ticker", "Date"]
    ).copy()

    # 대차잔고 하루 이전 값
    df[LAG_FEATURE] = (
        df.groupby("Ticker")[LENDING]
        .shift(1)
    )

    for col in FEATURES + [LAG_FEATURE, "Label"]:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    # 미래 5거래일 라벨이 유효한 데이터만 사용
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
        subset=(
            FEATURES
            + [LAG_FEATURE, "Label", "Date", "Ticker"]
        )
    )

    df = df[
        df["Label"].isin([0, 1])
    ].copy()

    df["Label"] = df["Label"].astype(int)

    # 날짜 기준 80:20 분할
    dates = sorted(df["Date"].unique())

    split = int(len(dates) * 0.8)
    boundary = pd.Timestamp(dates[split])

    # 학습 마지막 5거래일 제거
    purge_dates = set(
        dates[max(0, split - 5):split]
    )

    train = df[
        (df["Date"] < boundary)
        & (~df["Date"].isin(purge_dates))
    ].copy()

    test = df[
        df["Date"] >= boundary
    ].copy()

    print("평가 시작일:", boundary.date())
    print("학습 데이터:", len(train))
    print("평가 데이터:", len(test))

    return prices, train, test


# ==========================================
# 3. 모델별 매수 신호 생성
# ==========================================

def make_signals(train, test, prices):

    experiments = {
        "A_PRICE_ONLY": FEATURES,
        "C_WITH_LENDING": FEATURES + [LAG_FEATURE],
    }

    atr = prices[
        ["Date", "Ticker", "ATR14"]
    ].copy()

    signals = {}
    model_metrics = {}

    for name, columns in experiments.items():

        print(f"\n===== {name} 모델 학습 =====")

        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )

        model.fit(
            train[columns],
            train["Label"]
        )

        probability = model.predict_proba(
            test[columns]
        )[:, 1]

        ap = average_precision_score(
            test["Label"],
            probability
        )

        candidate = test[
            ["Date", "Ticker"]
        ].copy()

        candidate["Probability"] = probability

        # 0.65 이상 신호
        candidate = candidate[
            candidate["Probability"] >= THRESHOLD
        ].copy()

        total_signals = len(candidate)
        total_stocks = candidate["Ticker"].nunique()

        # 두 모델 모두 동일 종목 제외
        candidate = candidate[
            ~candidate["Ticker"].isin(
                EXCLUDED_TICKERS
            )
        ].copy()

        # ATR 연결
        candidate = candidate.merge(
            atr,
            on=["Date", "Ticker"],
            how="left",
            validate="many_to_one",
        )

        candidate = candidate.dropna(
            subset=["ATR14"]
        )

        candidate = candidate[
            candidate["ATR14"] > 0
        ].copy()

        print("전체 신호:", total_signals)
        print("전체 신호 종목:", total_stocks)
        print("제외 후 신호:", len(candidate))
        print("제외 후 종목:", candidate["Ticker"].nunique())
        print("Average Precision:", round(ap, 4))

        # 신호 데이터 별도 저장
        candidate.to_csv(
        BASE_DIR / f"stop8_{name.lower()}_signals.csv",
        index=False,
        encoding="utf-8-sig",
    )

        signals[name] = candidate

        model_metrics[name] = {
            "Average_Precision": round(ap, 4),
            "Candidate_Signals": len(candidate),
            "Signal_Stocks": candidate["Ticker"].nunique(),
        }

    return signals, model_metrics


# ==========================================
# 4. 초기 자금별 백테스트
# ==========================================

def run_capital_tests(prices, signals, model_metrics):

    results = []

    original_cash = bt.INITIAL_CASH

    try:
        for capital in INITIAL_CAPITALS:

            # 백테스트 모듈의 초기 자금 변경
            bt.INITIAL_CASH = capital

            print("\n" + "=" * 55)
            print(f"초기 자금: {capital:,}원")
            print("=" * 55)

            for name, candidate in signals.items():

                print(f"\n===== {name} =====")

                trades, equity = bt.run_backtest_v4(
                    prices,
                    candidate,
                    "SWING_ATR",
                )

                summary = bt.summarize_results(
                    trades,
                    equity,
                    name
                )

                summary["Initial_Capital"] = capital

                summary["Average_Precision"] = (
                    model_metrics[name]["Average_Precision"]
                )

                summary["Candidate_Signals"] = (
                    model_metrics[name]["Candidate_Signals"]
                )

                summary["Signal_Stocks"] = (
                    model_metrics[name]["Signal_Stocks"]
                )

                summary["Traded_Stocks"] = (
                    trades["Ticker"].nunique()
                    if not trades.empty
                    else 0
                )

                results.append(summary)

                # 기존 결과 덮어쓰기 방지
                                # 기존 결과 덮어쓰기 방지
                prefix = (
                    f"stop8_capital_{capital}_"
                    f"{name.lower()}"
                )

                trades.to_csv(
                    BASE_DIR / f"{prefix}_trades.csv",
                    index=False,
                    encoding="utf-8-sig",
                )

                equity.to_csv(
                    BASE_DIR / f"{prefix}_equity.csv",
                    index=False,
                    encoding="utf-8-sig",
                )
                print(pd.Series(summary).to_string())

    finally:
        # 실험 종료 후 초기 자금 원상복구
        bt.INITIAL_CASH = original_cash

    return pd.DataFrame(results)


# ==========================================
# 5. 메인 실행
# ==========================================

def main():

    prices, train, test = prepare_data()

    signals, model_metrics = make_signals(
        train,
        test,
        prices
    )

    result = run_capital_tests(
        prices,
        signals,
        model_metrics
    )

    result.to_csv(
    BASE_DIR / "stop8_comparison.csv",
    index=False,
    encoding="utf-8-sig",
)

    print("\n" + "=" * 70)
    print("===== 초기 자금별 최종 비교 =====")
    print("=" * 70)

    columns = [
        "Strategy",
        "Initial_Capital",
        "Final_Equity",
        "Return_Pct",
        "MDD_Pct",
        "Trades",
        "Win_Rate_Pct",
        "Traded_Stocks",
        "Candidate_Signals",
        "Average_Precision",
    ]

    print(
        result[columns].to_string(
            index=False
        )
    )

    print("\n결과 저장 완료:")
    print(BASE_DIR / "stop8_comparison.csv")


if __name__ == "__main__":
    main()