
import pandas as pd

import backtest_market_ma as bt

CHECK_DATES = [
    "2025-09-19",
    "2026-03-09",
]

def main():

    prices = bt.load_prices()

    prices["Ticker"] = prices["Ticker"].map(
        bt.normalize_ticker
    )

    prices["Date"] = pd.to_datetime(
        prices["Date"]
    ).dt.normalize()

    trades = pd.read_csv(
        "wf_A_NO_FILTER_trades.csv",
        dtype={"Ticker": str},
        parse_dates=["Entry_Date", "Exit_Date"]
    )

    trades["Ticker"] = trades["Ticker"].map(
        bt.normalize_ticker
    )

    for date_str in CHECK_DATES:

        date = pd.Timestamp(date_str)

        print(f"\n===== {date.date()} =====")

        # 해당 날짜에 실제 보유 중인 종목
        holding = trades[
            (trades["Entry_Date"] <= date)
            & (trades["Exit_Date"] >= date)
        ].copy()

        print("보유 종목 수:", len(holding))

        for _, trade in holding.iterrows():

            ticker = trade["Ticker"]

            bar = prices[
                (prices["Date"] == date)
                & (prices["Ticker"] == ticker)
            ]

            print("\n종목:", ticker)
            print("진입일:", trade["Entry_Date"].date())
            print("청산일:", trade["Exit_Date"].date())
            print("보유 수량:", trade["Shares"])

            if bar.empty:
                print("가격 데이터 없음!")
            else:
                print(
                    "종가:",
                    bar["Close"].iloc[-1]
                )

if __name__ == "__main__":
    main()
