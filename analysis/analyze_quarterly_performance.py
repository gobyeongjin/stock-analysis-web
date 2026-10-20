
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

INITIAL_CAPITAL = 10_000_000
START_DATE = "2025-10-20"
END_DATE = "2026-09-30"

FILES = {
    "NO_FILTER": {
        "equity": "market_no_filter_equity.csv",
        "trades": "market_no_filter_trades.csv",
    },
    "MA20": {
        "equity": "market_ma20_equity.csv",
        "trades": "market_ma20_trades.csv",
    },
    "MA40": {
        "equity": "market_ma40_equity.csv",
        "trades": "market_ma40_trades.csv",
    },
}


def calculate_mdd(values):
    peak = values.cummax()
    drawdown = (values / peak - 1) * 100
    return drawdown.min()


def load_equity(path):
    df = pd.read_csv(path)

    df["Date"] = pd.to_datetime(df["Date"])
    df["Equity"] = pd.to_numeric(df["Equity"])

    df = df.sort_values("Date").copy()

    return df


def load_trades(path):
    df = pd.read_csv(path)

    df["Entry_Date"] = pd.to_datetime(df["Entry_Date"])
    df["Exit_Date"] = pd.to_datetime(df["Exit_Date"])
    df["PnL"] = pd.to_numeric(df["PnL"])

    return df


def main():
    results = []

    for strategy, paths in FILES.items():

        equity_path = BASE_DIR / paths["equity"]
        trades_path = BASE_DIR / paths["trades"]

        if not equity_path.exists():
            print(f"파일 없음: {equity_path.name}")
            continue

        if not trades_path.exists():
            print(f"파일 없음: {trades_path.name}")
            continue

        equity = load_equity(equity_path)
        trades = load_trades(trades_path)

        equity = equity[
            equity["Date"].between(START_DATE, END_DATE)
        ].copy()

        if equity.empty:
            continue

        equity["Quarter"] = (
            equity["Date"].dt.to_period("Q").astype(str)
        )

        for quarter, group in equity.groupby("Quarter"):

            group = group.sort_values("Date")

            quarter_start = group["Date"].iloc[0]
            quarter_end = group["Date"].iloc[-1]

            previous = equity[
                equity["Date"] < quarter_start
            ]

            if previous.empty:
                start_equity = INITIAL_CAPITAL
            else:
                start_equity = previous["Equity"].iloc[-1]

            end_equity = group["Equity"].iloc[-1]

            return_pct = (
                end_equity / start_equity - 1
            ) * 100

            # 분기 시작 자산을 포함해 MDD 계산
            values = pd.concat(
                [
                    pd.Series([start_equity]),
                    group["Equity"].reset_index(drop=True),
                ],
                ignore_index=True,
            )

            mdd_pct = calculate_mdd(values)

            # 분기 내 청산된 거래 기준
            quarter_trades = trades[
                trades["Exit_Date"].between(
                    quarter_start, quarter_end
                )
            ]

            trade_count = len(quarter_trades)

            if trade_count > 0:
                win_rate = (
                    (quarter_trades["PnL"] > 0).mean()
                    * 100
                )
            else:
                win_rate = 0

            results.append({
                "Strategy": strategy,
                "Quarter": quarter,
                "Return_Pct": round(return_pct, 2),
                "MDD_Pct": round(mdd_pct, 2),
                "Trades": trade_count,
                "Win_Rate_Pct": round(win_rate, 2),
                "Start_Equity": round(start_equity, 0),
                "End_Equity": round(end_equity, 0),
            })

    if not results:
        print("분석 가능한 파일이 없습니다.")
        return

    result = pd.DataFrame(results)

    print("\n===== 분기별 성과 =====")
    print(
        result[
            [
                "Strategy",
                "Quarter",
                "Return_Pct",
                "MDD_Pct",
                "Trades",
                "Win_Rate_Pct",
            ]
        ].to_string(index=False)
    )

    print("\n===== 분기별 수익률 비교 =====")
    pivot_return = result.pivot(
        index="Quarter",
        columns="Strategy",
        values="Return_Pct",
    )
    print(pivot_return.to_string())

    print("\n===== 분기별 MDD 비교 =====")
    pivot_mdd = result.pivot(
        index="Quarter",
        columns="Strategy",
        values="MDD_Pct",
    )
    print(pivot_mdd.to_string())

    result.to_csv(
        BASE_DIR / "quarterly_performance_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\nCSV 저장 완료")


if __name__ == "__main__":
    main()
