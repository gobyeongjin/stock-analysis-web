
import pandas as pd
from pathlib import Path

import backtest_market_ma as bt
import backtest_stop8_test as base

BASE_DIR = Path(__file__).resolve().parent
MA_PERIODS = [20, 40]


def main():
    prices, train, test = base.prepare_data()
    signals, _ = base.make_signals(train, test, prices)

    candidate = signals["A_PRICE_ONLY"].copy()
    candidate["Date"] = pd.to_datetime(
        candidate["Date"]
    ).dt.normalize()

    candidate["Ticker"] = (
        candidate["Ticker"].astype(str).str.zfill(6)
    )

    # 실제 거래 데이터는 기존 결과 사용
    all_results = []

    original_ma = bt.MARKET_MA_DAYS

    try:
        for ma_days in MA_PERIODS:
            bt.MARKET_MA_DAYS = ma_days

            market = pd.read_csv(
                BASE_DIR / "kospi_market_filter.csv"
            )

            market["Date"] = pd.to_datetime(
                market["Date"]
            ).dt.normalize()

            market = market.sort_values("Date").copy()

            market["Close"] = pd.to_numeric(
                market["Close"], errors="coerce"
            )

            market["Market_MA"] = (
                market["Close"]
                .rolling(ma_days, min_periods=ma_days)
                .mean()
            )

            market["Market_OK"] = (
                market["Close"] > market["Market_MA"]
            )

            # 신호일에 코스피 필터 연결
            merged = candidate.merge(
            market[
                ["Date", "Close", "Market_MA", "Market_OK"]
            ].rename(columns={"Close": "KOSPI_Close"}),
            on="Date",
            how="left",
)

            trades = pd.read_csv(
                BASE_DIR / f"market_ma{ma_days}_trades.csv",
                dtype={"Ticker": str},
            )

            trades["Ticker"] = (
                trades["Ticker"].str.zfill(6)
            )

            trades["Signal_Date"] = pd.to_datetime(
                trades["Signal_Date"]
            ).dt.normalize()

            # 같은 종목 + 같은 신호일로 실제 매수 여부 확인
            bought_keys = set(
                zip(trades["Signal_Date"], trades["Ticker"])
            )

            merged["Bought"] = [
                (date, ticker) in bought_keys
                for date, ticker in zip(
                    merged["Date"], merged["Ticker"]
                )
            ]

            merged["Reason"] = "NOT_EXECUTED_OTHER"

# 코스피 데이터가 없는 경우
            merged.loc[
                merged["KOSPI_Close"].isna(),
                "Reason"
            ] = "MARKET_DATA_MISSING"

            # 코스피 이동평균선 데이터가 부족한 경우
            merged.loc[
                merged["KOSPI_Close"].notna()
                & merged["Market_MA"].isna(),
                "Reason"
            ] = "MARKET_MA_MISSING"

            # 시장 필터에 의해 매수가 차단된 경우
            merged.loc[
                merged["Market_MA"].notna()
                & (merged["Market_OK"] == False),
                "Reason"
            ] = "MARKET_BLOCKED"

            # 실제 매수된 경우
            merged.loc[
                merged["Bought"],
                "Reason"
            ] = "BOUGHT"

            merged.loc[
                merged["Market_OK"] == False,
                "Reason"
            ] = "MARKET_BLOCKED"

            merged.loc[
                merged["Bought"],
                "Reason"
            ] = "BOUGHT"

            merged["MA_Days"] = ma_days


            all_results.append(merged)

    finally:
        bt.MARKET_MA_DAYS = original_ma

    result = pd.concat(
        all_results, ignore_index=True
    )

    print("\n===== MA별 신호 처리 현황 =====")
    summary = pd.crosstab(
        result["MA_Days"],
        result["Reason"]
    )
    print(summary.to_string())

    print("\n===== 2026년 6~7월 신호 처리 현황 =====")
    period = result[
        result["Date"].between(
            "2026-06-01", "2026-07-31"
        )
    ]
    print(
        pd.crosstab(
            period["MA_Days"],
            period["Reason"]
        ).to_string()
    )

    # 두 이동평균선의 필터 판단이 다른 신호
    pivot = result.pivot_table(
        index=["Date", "Ticker"],
        columns="MA_Days",
        values="Market_OK",
        aggfunc="first",
    )

    different = pivot[
        pivot[20] != pivot[40]
    ].reset_index()

    print("\n===== MA20과 MA40의 판단이 다른 신호 =====")
    print("신호 수:", len(different))
    print(different.to_string(index=False))

    result.to_csv(
        BASE_DIR / "market_signal_reason_detail.csv",
        index=False,
        encoding="utf-8-sig",
    )

    different.to_csv(
        BASE_DIR / "market_signal_filter_differences.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\n분석 CSV 저장 완료")


if __name__ == "__main__":
    main()
