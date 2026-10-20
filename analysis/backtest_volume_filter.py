
import pandas as pd
from pathlib import Path

import backtest_stop8_test as base
import backtest_market_ma as bt

BASE_DIR = Path(__file__).resolve().parent

INITIAL_CAPITAL = 10_000_000
THRESHOLD = 0.65
MARKET_MA = 40


def normalize_ticker(series):
    return (
        series.astype(str)
        .str.split(".").str[0]
        .str.zfill(6)
    )


def main():

    # 1. 데이터 및 AI 신호 생성
    prices, train, test = base.prepare_data()

    signals, metrics = base.make_signals(
        train, test, prices
    )

    candidate = signals["A_PRICE_ONLY"].copy()

    candidate["Date"] = pd.to_datetime(
        candidate["Date"]
    ).dt.normalize()

    # 백테스트용 종목코드는 변경하지 않음
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

    # 2. 평가 데이터의 거래량 비율 연결
    test = test.copy()

    test["Date"] = pd.to_datetime(
        test["Date"]
    ).dt.normalize()

    test["Merge_Ticker"] = normalize_ticker(
        test["Ticker"]
    )

    if "Volume_Ratio" not in test.columns:
        raise ValueError(
            "평가 데이터에 Volume_Ratio 컬럼이 없습니다."
        )

    features = test[
        ["Date", "Merge_Ticker", "Volume_Ratio"]
    ].drop_duplicates(
        ["Date", "Merge_Ticker"]
    )

    candidate = candidate.merge(
        features,
        on=["Date", "Merge_Ticker"],
        how="left",
        validate="many_to_one"
    )

    candidate = candidate.drop(
        columns=["Merge_Ticker"]
    )

    candidate["Volume_Ratio"] = pd.to_numeric(
        candidate["Volume_Ratio"],
        errors="coerce"
    )

    print("\n===== 거래량 비율 연결 확인 =====")
    print("전체 신호:", len(candidate))
    print(
        "연결 성공:",
        candidate["Volume_Ratio"].notna().sum()
    )
    print(
        candidate["Volume_Ratio"].describe().round(3)
    )

    if candidate["Volume_Ratio"].isna().any():
        raise ValueError(
            "거래량 비율 연결에 실패한 신호가 있습니다."
        )

    # 3. 전략별 거래량 필터
    strategies = {
        "A_NO_FILTER": candidate.copy(),

        "B_VOLUME_1_0": candidate[
            candidate["Volume_Ratio"] <= 1.0
        ].copy(),

        "C_VOLUME_1_3": candidate[
            candidate["Volume_Ratio"] <= 1.3
        ].copy(),

        "D_VOLUME_1_5": candidate[
            candidate["Volume_Ratio"] <= 1.5
        ].copy(),

        "E_VOLUME_HIGH": candidate[
            candidate["Volume_Ratio"] >= 1.3
        ].copy(),
    }

    # 4. 기존 백테스트 설정 유지
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
            print("신호 수:", len(selected))
            print(
                "종목 수:",
                selected["Ticker"].nunique()
            )

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

            trades.to_csv(
                BASE_DIR / f"volume_{name}_trades.csv",
                index=False,
                encoding="utf-8-sig"
            )

            equity.to_csv(
                BASE_DIR / f"volume_{name}_equity.csv",
                index=False,
                encoding="utf-8-sig"
            )

    finally:
        bt.INITIAL_CASH = original_cash
        bt.USE_MARKET_FILTER = original_filter
        bt.MARKET_MA_DAYS = original_ma

    # 5. 최종 결과 비교
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

    print("\n===== 거래량 필터 백테스트 비교 =====")
    print(result[columns].to_string(index=False))

    result.to_csv(
        BASE_DIR / "volume_filter_comparison.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("\n분석 완료")


if __name__ == "__main__":
    main()
