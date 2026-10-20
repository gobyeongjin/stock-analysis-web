
import pandas as pd

import backtest_market_ma as bt
import backtest_stop8_test as base


INITIAL_CAPITAL = 10_000_000

MA_SETTINGS = [
    10,
    20,
    40,
    60,
]


def main():

    # 데이터 준비 및 A 모델 학습
    prices, train, test = base.prepare_data()

    signals, metrics = base.make_signals(
        train,
        test,
        prices,
    )

    candidate = signals["A_PRICE_ONLY"]

    results = []

    original_cash = bt.INITIAL_CASH
    original_filter = bt.USE_MARKET_FILTER
    original_ma = bt.MARKET_MA_DAYS

    try:
        bt.INITIAL_CASH = INITIAL_CAPITAL
        bt.USE_MARKET_FILTER = True

        for ma_days in MA_SETTINGS:

            bt.MARKET_MA_DAYS = ma_days

            print("\n" + "=" * 60)
            print(f"코스피 이동평균선: {ma_days}일")
            print("=" * 60)

            trades, equity = bt.run_backtest_v4(
                prices,
                candidate,
                "SWING_ATR",
            )

            summary = bt.summarize_results(
                trades,
                equity,
                "A_PRICE_ONLY",
            )

            summary["Market_MA_Days"] = ma_days
            summary["Initial_Capital"] = INITIAL_CAPITAL

            results.append(summary)

            prefix = f"market_ma{ma_days}"

            trades.to_csv(
                bt.BASE_DIR / f"{prefix}_trades.csv",
                index=False,
                encoding="utf-8-sig",
            )

            equity.to_csv(
                bt.BASE_DIR / f"{prefix}_equity.csv",
                index=False,
                encoding="utf-8-sig",
            )

            print(
                f"수익률: {summary['Return_Pct']:.2f}%"
            )
            print(
                f"MDD: {summary['MDD_Pct']:.2f}%"
            )
            print(
                f"거래 횟수: {summary['Trades']}"
            )

    finally:
        bt.INITIAL_CASH = original_cash
        bt.USE_MARKET_FILTER = original_filter
        bt.MARKET_MA_DAYS = original_ma

    result = pd.DataFrame(results)

    result.to_csv(
        bt.BASE_DIR / "market_ma_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 75)
    print("===== 코스피 이동평균선 기간별 비교 =====")
    print("=" * 75)

    print(
        result[
            [
                "Market_MA_Days",
                "Return_Pct",
                "MDD_Pct",
                "Trades",
                "Win_Rate_Pct",
                "Final_Equity",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
