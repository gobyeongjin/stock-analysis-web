
import pandas as pd

import backtest_market_filter as bt
import backtest_stop8_test as base


INITIAL_CAPITAL = 10_000_000


def main():
    prices, train, test = base.prepare_data()

    signals, metrics = base.make_signals(
        train, test, prices
    )

    candidate = signals["A_PRICE_ONLY"]

    results = []

    original_cash = bt.INITIAL_CASH
    original_filter = bt.USE_MARKET_FILTER

    try:
        bt.INITIAL_CASH = INITIAL_CAPITAL

        for name, enabled in [
            ("MARKET_OFF", False),
            ("MARKET_ON", True),
        ]:
            bt.USE_MARKET_FILTER = enabled

            print("\n" + "=" * 60)
            print("시장 필터:", name)
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

            summary["Market_Filter"] = name
            summary["Initial_Capital"] = INITIAL_CAPITAL

            results.append(summary)

            prefix = f"market_{name.lower()}"

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

            print(pd.Series(summary).to_string())

    finally:
        bt.INITIAL_CASH = original_cash
        bt.USE_MARKET_FILTER = original_filter

    result = pd.DataFrame(results)

    result.to_csv(
        bt.BASE_DIR / "market_filter_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\n===== 시장 필터 최종 비교 =====")

    print(
        result[
            [
                "Market_Filter",
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
