
import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

TRADES_FILE = BASE_DIR / "prob_threshold_65_trades.csv"
TRAIN_FILE = BASE_DIR / "training_data_v4_lending.csv"

OUTPUT_FILE = BASE_DIR / "trade_entry_features.csv"

FEATURES = [
    "RSI",
    "Volume_Ratio",
    "MA20_Gap",
    "MA60_Gap",
    "Return_5D",
    "Volatility_20D",
]


def normalize_ticker(series):
    return (
        series.astype(str)
        .str.split(".").str[0]
        .str.zfill(6)
    )


def main():
    # ======================================
    # 1. 실제 거래 내역 불러오기
    # ======================================

    trades = pd.read_csv(
        TRADES_FILE,
        dtype={"Ticker": str}
    )

    trades["Ticker"] = normalize_ticker(
        trades["Ticker"]
    )

    for col in [
        "Signal_Date",
        "Entry_Date",
        "Exit_Date",
    ]:
        trades[col] = pd.to_datetime(
            trades[col]
        ).dt.normalize()

    trades["PnL"] = pd.to_numeric(
        trades["PnL"], errors="coerce"
    )

    trades["Return_Pct"] = pd.to_numeric(
        trades["Return_Pct"], errors="coerce"
    )

    print("전체 거래 수:", len(trades))

    # ======================================
    # 2. 학습 데이터 불러오기
    # ======================================

    data = pd.read_csv(
        TRAIN_FILE,
        dtype={"Ticker": str}
    )

    data["Date"] = pd.to_datetime(
        data["Date"]
    ).dt.normalize()

    data["Ticker"] = normalize_ticker(
        data["Ticker"]
    )

    available_features = [
        col for col in FEATURES
        if col in data.columns
    ]

    print("사용 가능한 특징:", available_features)

    if not available_features:
        print("분석 가능한 특징 컬럼이 없습니다.")
        return

    # ======================================
    # 3. 신호 발생일 기준으로 특징 연결
    # ======================================

    feature_data = data[
        ["Date", "Ticker"] + available_features
    ].drop_duplicates(
        ["Date", "Ticker"]
    )

    result = trades.merge(
        feature_data,
        left_on=["Signal_Date", "Ticker"],
        right_on=["Date", "Ticker"],
        how="left",
        validate="many_to_one"
    )

    result = result.drop(
        columns=["Date"]
    )

    # ======================================
    # 4. 거래 결과 분류
    # ======================================

    result["Trade_Group"] = np.where(
        result["PnL"] > 0,
        "WIN",
        "LOSS"
    )

    result["Exit_Group"] = np.select(
        [
            result["Exit_Reason"].eq("손절"),
            result["Exit_Reason"].eq("익절"),
        ],
        [
            "STOP_LOSS",
            "TAKE_PROFIT",
        ],
        default="TIME_EXIT"
    )

    print("\n===== 거래 결과 분포 =====")
    print(
        result["Trade_Group"]
        .value_counts()
        .to_string()
    )

    print("\n===== 청산 유형 분포 =====")
    print(
        result["Exit_Group"]
        .value_counts()
        .to_string()
    )

    # ======================================
    # 5. 수익 / 손실 거래 특징 비교
    # ======================================

    for feature in available_features:
        result[feature] = pd.to_numeric(
            result[feature],
            errors="coerce"
        )

        summary = (
            result.groupby("Trade_Group")[feature]
            .agg(["count", "mean", "median"])
            .round(3)
        )

        print(f"\n===== {feature} 수익/손실 비교 =====")
        print(summary.to_string())

    # ======================================
    # 6. 청산 유형별 특징 비교
    # ======================================

    print("\n===== 청산 유형별 평균 특징 =====")

    exit_summary = (
        result.groupby("Exit_Group")[
            available_features
        ]
        .mean()
        .round(3)
    )

    print(exit_summary.to_string())

    # ======================================
    # 7. RSI 구간별 성과
    # ======================================

    if "RSI" in result.columns:
        result["RSI_Group"] = pd.cut(
            result["RSI"],
            bins=[-np.inf, 30, 50, 70, np.inf],
            labels=[
                "RSI_0_30",
                "RSI_30_50",
                "RSI_50_70",
                "RSI_70_PLUS",
            ]
        )

        rsi_summary = (
            result.groupby(
                "RSI_Group",
                observed=False
            )
            .agg(
                Trades=("PnL", "size"),
                Win_Rate=(
                    "PnL",
                    lambda x: (x > 0).mean() * 100
                ),
                Avg_Return=("Return_Pct", "mean"),
                Total_PnL=("PnL", "sum"),
            )
            .round(2)
        )

        print("\n===== RSI 구간별 거래 성과 =====")
        print(rsi_summary.to_string())

    # ======================================
    # 8. 종목별 손익
    # ======================================

    ticker_summary = (
        result.groupby("Ticker")
        .agg(
            Trades=("PnL", "size"),
            Win_Rate=(
                "PnL",
                lambda x: (x > 0).mean() * 100
            ),
            Total_PnL=("PnL", "sum"),
            Avg_Return=("Return_Pct", "mean"),
        )
        .sort_values("Total_PnL")
        .round(2)
    )

    print("\n===== 종목별 거래 성과 =====")
    print(ticker_summary.to_string())

    # ======================================
    # 9. 저장
    # ======================================

    result.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n분석 파일 저장:", OUTPUT_FILE.name)


if __name__ == "__main__":
    main()
