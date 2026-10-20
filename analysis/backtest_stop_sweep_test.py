
import pandas as pd

import backtest_stop_sweep as bt
import backtest_stop8_test as base


INITIAL_CAPITAL = 10_000_000

STOP_SETTINGS = {
    "BASE_ATR": None,
    "STOP_5": 0.05,
    "STOP_8": 0.08,
    "STOP_10": 0.10,
    "STOP_12": 0.12,
}


def main():

    # 기존 데이터 준비 및 모델 학습 함수 재사용
    prices, train, test = base.prepare_data()

    signals, metrics = base.make_signals(
        train,
        test,
        prices,
    )

    results = []

    original_cash = bt.INITIAL_CASH
    original_stop = bt.STOP_LIMIT_PCT

    try:
        bt.INITIAL_CASH = INITIAL_CAPITAL

        for setting_name, stop_pct in STOP_SETTINGS.items():

            bt.STOP_LIMIT_PCT = stop_pct

            print("\n" + "=" * 65)
            print("손절 전략:", setting_name)
            print("최대 손절 제한:", stop_pct)
            print("=" * 65)

            for model_name, candidate in signals.items():

                trades, equity = bt.run_backtest_v4(
                    prices,
                    candidate,
                    "SWING_ATR",
                )

                summary = bt.summarize_results(
                    trades,
                    equity,
                    model_name,
                )

                summary["Stop_Setting"] = setting_name
                summary["Stop_Limit_Pct"] = (
                    stop_pct * 100
                    if stop_pct is not None
                    else None
                )
                summary["Initial_Capital"] = INITIAL_CAPITAL
                summary["Average_Precision"] = (
                    metrics[model_name]["Average_Precision"]
                )

                results.append(summary)

                prefix = (
                    f"sweep_{setting_name.lower()}_"
                    f"{model_name.lower()}"
                )

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
                    f"{model_name}: "
                    f"수익률 {summary['Return_Pct']:.2f}%, "
                    f"MDD {summary['MDD_Pct']:.2f}%, "
                    f"거래 {summary['Trades']}회"
                )

    finally:
        bt.INITIAL_CASH = original_cash
        bt.STOP_LIMIT_PCT = original_stop

    result = pd.DataFrame(results)

    result.to_csv(
        bt.BASE_DIR / "stop_sweep_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 75)
    print("===== 손절 전략별 최종 비교 =====")
    print("=" * 75)

    print(
        result[
            [
                "Stop_Setting",
                "Strategy",
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
