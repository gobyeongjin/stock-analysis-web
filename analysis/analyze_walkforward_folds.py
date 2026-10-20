
import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

INITIAL_CAPITAL = 10_000_000

STRATEGIES = {
    "A_NO_FILTER": "거래량 필터 없음",
    "B_VOLUME_1_0": "거래량 1.0배 이하",
}


def calculate_mdd(values):
    values = values.astype(float)

    if values.empty:
        return np.nan

    peak = values.cummax()
    drawdown = (values / peak - 1) * 100

    return float(drawdown.min())


def main():

    metrics = pd.read_csv(
        BASE_DIR / "wf_model_metrics.csv",
        parse_dates=["Test_Start", "Test_End"]
    )

    results = []
    all_trades = []

    for strategy, description in STRATEGIES.items():

        equity = pd.read_csv(
            BASE_DIR / f"wf_{strategy}_equity.csv",
            parse_dates=["Date"]
        )

        trades = pd.read_csv(
            BASE_DIR / f"wf_{strategy}_trades.csv",
            parse_dates=[
                "Signal_Date",
                "Entry_Date",
                "Exit_Date"
            ]
        )

        equity = equity.sort_values("Date").copy()

        print(f"\n===== {description} =====")

        for _, fold in metrics.iterrows():

            fold_num = int(fold["Fold"])
            start = fold["Test_Start"]
            end = fold["Test_End"]

            # 평가 구간의 일별 자산
            period_equity = equity[
                (equity["Date"] >= start)
                & (equity["Date"] <= end)
            ].copy()

            if period_equity.empty:
                continue

            # 구간 첫날 직전의 자산을 시작 자산으로 사용
            previous = equity[
                equity["Date"] < start
            ]

            if previous.empty:
                start_equity = INITIAL_CAPITAL
            else:
                start_equity = float(
                    previous["Equity"].iloc[-1]
                )

            end_equity = float(
                period_equity["Equity"].iloc[-1]
            )

            period_return = (
                end_equity / start_equity - 1
            ) * 100

            # 첫날 직전 자산까지 포함하여
            # 구간 시작 직후의 낙폭도 계산
            mdd_values = pd.concat([
                pd.Series([start_equity]),
                period_equity["Equity"].reset_index(drop=True)
            ], ignore_index=True)

            period_mdd = calculate_mdd(mdd_values)

            # 진입일 기준으로 거래 구간 분류
            period_trades = trades[
                (trades["Entry_Date"] >= start)
                & (trades["Entry_Date"] <= end)
            ].copy()

            trade_count = len(period_trades)

            if trade_count > 0:
                win_rate = (
                    (period_trades["PnL"] > 0).mean()
                    * 100
                )
            else:
                win_rate = 0.0

            # 구간을 넘어 다음 구간에서 청산된 거래
            cross_fold = period_trades[
                period_trades["Exit_Date"] > end
            ]

            results.append({
                "Strategy": strategy,
                "Fold": fold_num,
                "Start": start.date(),
                "End": end.date(),
                "Start_Equity": round(start_equity, 2),
                "End_Equity": round(end_equity, 2),
                "Return_Pct": round(period_return, 2),
                "MDD_Pct": round(period_mdd, 2),
                "Trades": trade_count,
                "Win_Rate_Pct": round(win_rate, 2),
                "Cross_Fold_Trades": len(cross_fold),
            })

            if not period_trades.empty:
                period_trades["Fold"] = fold_num
                period_trades["Strategy_Name"] = strategy
                all_trades.append(period_trades)

    result = pd.DataFrame(results)

    print("\n===== 구간별 수익률 비교 =====")

    return_table = result.pivot(
        index="Fold",
        columns="Strategy",
        values="Return_Pct"
    )

    print(return_table.round(2).to_string())

    print("\n===== 구간별 MDD 비교 =====")

    mdd_table = result.pivot(
        index="Fold",
        columns="Strategy",
        values="MDD_Pct"
    )

    print(mdd_table.round(2).to_string())

    print("\n===== 구간별 거래 횟수 =====")

    trades_table = result.pivot(
        index="Fold",
        columns="Strategy",
        values="Trades"
    )

    print(trades_table.to_string())

    print("\n===== 전체 상세 결과 =====")
    print(result.to_string(index=False))

    result.to_csv(
        BASE_DIR / "wf_fold_comparison.csv",
        index=False,
        encoding="utf-8-sig"
    )

    if all_trades:
        pd.concat(
            all_trades,
            ignore_index=True
        ).to_csv(
            BASE_DIR / "wf_fold_trades.csv",
            index=False,
            encoding="utf-8-sig"
        )

    print("\n구간별 분석 완료")


if __name__ == "__main__":
    main()
