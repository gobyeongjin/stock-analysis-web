
import pandas as pd
import numpy as np
from pathlib import Path

import backtest_stop8_test as base
import backtest_market_ma as bt

BASE_DIR = Path(__file__).resolve().parent

THRESHOLD = 0.65
MARKET_MA = 40

INITIAL_CAPITAL = 10_000_000

STRATEGIES = [
    "A_NO_FILTER",
    "B_CHASE_LIMIT",
    "C_MOMENTUM",
    "D_MIDDLE_RANGE",
]


def normalize_ticker(series):
    return (
        series.astype(str)
        .str.split(".").str[0]
        .str.zfill(6)
    )


def main():

    # =====================================
    # 1. 데이터 로딩 및 AI 신호 생성
    # =====================================

    prices, train, test = base.prepare_data()

    signals, metrics = base.make_signals(
        train, test, prices
    )

    candidate = signals["A_PRICE_ONLY"].copy()

    candidate["Date"] = pd.to_datetime(
        candidate["Date"]
    ).dt.normalize()

    # 백테스트에 사용할 원래 종목코드 보존
    candidate["Original_Ticker"] = candidate["Ticker"]

    # 평가 데이터 연결용 종목코드
    candidate["Merge_Ticker"] = normalize_ticker(
        candidate["Ticker"]
    )

    candidate["Probability"] = pd.to_numeric(
        candidate["Probability"],
        errors="coerce"
    )

    candidate = candidate[
        candidate["Probability"] >= THRESHOLD
    ].copy()


    # =====================================
    # 2. 평가 데이터에서 과거 5일 수익률 연결
    # =====================================

    test = test.copy()

    test["Date"] = pd.to_datetime(
        test["Date"]
    ).dt.normalize()

    # 평가 데이터 연결용 종목코드 생성
    test["Merge_Ticker"] = normalize_ticker(
        test["Ticker"]
    )

    if "Return_5D" not in test.columns:
        raise ValueError(
            "평가 데이터에 Return_5D 컬럼이 없습니다."
        )

    features = test[
        ["Date", "Merge_Ticker", "Return_5D"]
    ].drop_duplicates(
        ["Date", "Merge_Ticker"]
    )

    candidate = candidate.merge(
        features,
        on=["Date", "Merge_Ticker"],
        how="left",
        validate="many_to_one"
    )

    # 원래 종목코드 유지
    candidate["Ticker"] = candidate["Original_Ticker"]

    candidate = candidate.drop(
        columns=["Original_Ticker", "Merge_Ticker"]
    )

    candidate["Return_5D"] = pd.to_numeric(
        candidate["Return_5D"],
        errors="coerce"
    )


    print("\n===== Return_5D 연결 확인 =====")
    print("전체 신호:", len(candidate))
    print(
        "수익률 연결 성공:",
        candidate["Return_5D"].notna().sum()
    )
    print(
        candidate["Return_5D"].describe().round(3)
    )

    if candidate["Return_5D"].isna().any():
        raise ValueError(
            "Return_5D 연결에 실패한 신호가 있습니다."
        )

    # 기존 데이터 분석 결과의 Return_5D는
    # 10%가 10.0으로 표현된 것으로 가정
    # 단위가 다른 경우 실행을 중단하고 확인
    median_abs = candidate["Return_5D"].abs().median()

    if median_abs < 0.5:
        raise ValueError(
            "Return_5D가 소수 단위일 수 있습니다. "
            "10% 기준을 0.10으로 변경해야 합니다."
        )

    # =====================================
    # 3. 전략별 신호 필터
    # =====================================

    strategies = {
        "A_NO_FILTER": candidate.copy(),

        "B_CHASE_LIMIT": candidate[
            candidate["Return_5D"] <= 10
        ].copy(),

        "C_MOMENTUM": candidate[
            candidate["Return_5D"] >= 10
        ].copy(),

        "D_MIDDLE_RANGE": candidate[
            (candidate["Return_5D"] >= 0) &
            (candidate["Return_5D"] <= 10)
        ].copy(),
    }

    # =====================================
    # 4. MA40 시장 필터 적용
    # =====================================

    original_cash = bt.INITIAL_CASH
    original_filter = bt.USE_MARKET_FILTER
    original_ma = bt.MARKET_MA_DAYS

    bt.INITIAL_CASH = INITIAL_CAPITAL
    bt.USE_MARKET_FILTER = True
    bt.MARKET_MA_DAYS = MARKET_MA

    results = []

    try:
        for name in STRATEGIES:

            selected = strategies[name].copy()

            print(f"\n===== {name} =====")
            print("신호 수:", len(selected))
            print(
                "종목 수:",
                selected["Ticker"].nunique()
            )

            # =================================
            # 5. 기존 백테스트 엔진 실행
            # =================================

            trades, equity = bt.run_backtest_v4(
                prices,
                selected,
                "SWING_ATR"
            )

            summary = bt.summarize_results(
                trades,
                equity,
                name
            )

            summary["Strategy"] = name
            summary["Signal_Count"] = len(selected)
            summary["Unique_Stocks"] = (
                selected["Ticker"].nunique()
            )

            results.append(summary)

            # =================================
            # 6. 결과 저장
            # =================================

            trades.to_csv(
                BASE_DIR / f"return5d_{name}_trades.csv",
                index=False,
                encoding="utf-8-sig"
            )

            equity.to_csv(
                BASE_DIR / f"return5d_{name}_equity.csv",
                index=False,
                encoding="utf-8-sig"
            )

    finally:
        bt.INITIAL_CASH = original_cash
        bt.USE_MARKET_FILTER = original_filter
        bt.MARKET_MA_DAYS = original_ma

    # =====================================
    # 7. 최종 비교
    # =====================================

    result = pd.DataFrame(results)

    columns = [
        "Strategy",
        "Signal_Count",
        "Unique_Stocks",
        "Return_Pct",
        "MDD_Pct",
        "Trades",
        "Win_Rate_Pct",
        "Final_Equity",
    ]

    print("\n===== 최근 5일 수익률 필터 비교 =====")

    print(
        result[columns].to_string(index=False)
    )

    result.to_csv(
        BASE_DIR / "return5d_filter_comparison.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("\n분석 완료")


if __name__ == "__main__":
    main()
