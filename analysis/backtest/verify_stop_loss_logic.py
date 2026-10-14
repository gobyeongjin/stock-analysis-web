from pathlib import Path
import pandas as pd
import numpy as np

BASE = Path(__file__).resolve().parents[1]
RESULTS = BASE / "results"
PRICE_FILE = BASE / "stock_data_500.csv"


def main():
    # 가장 최근에 생성된 손절 백테스트 결과 사용
    files = list(RESULTS.glob("matched_stop_loss_trades*.csv"))
    if not files:
        raise FileNotFoundError(
            "matched_stop_loss_trades*.csv 파일을 찾을 수 없습니다."
        )

    trade_file = max(files, key=lambda p: p.stat().st_mtime)
    print(f"거래 결과 파일: {trade_file}")

    trades = pd.read_csv(trade_file, dtype={"Ticker": str})
    prices = pd.read_csv(PRICE_FILE, dtype={"Ticker": str})

    for df in [trades, prices]:
        df["Ticker"] = df["Ticker"].astype("string").str.strip()
        df["Date"] = pd.to_datetime(
            df["Date"] if "Date" in df.columns
            else df["Entry_Date"], errors="coerce"
        )

    for col in [
        "Open", "High", "Low", "Close", "Volume"
    ]:
        prices[col] = pd.to_numeric(prices[col], errors="coerce")

    for col in [
        "Entry_Date", "Actual_Exit_Date", "Scheduled_Exit_Date"
    ]:
        trades[col] = pd.to_datetime(trades[col], errors="coerce")

    prices = prices.dropna(
        subset=["Date", "Ticker", "Open", "High", "Low", "Close"]
    )
    prices = prices.sort_values(["Ticker", "Date"])
    prices = prices.drop_duplicates(["Ticker", "Date"], keep="last")
    prices["Prev_Close"] = prices.groupby("Ticker")["Close"].shift(1)
    prices["Daily_Change_Pct"] = (
        prices["Close"] / prices["Prev_Close"] - 1
    ) * 100

    same_ohlc = (
        prices["Open"].eq(prices["High"])
        & prices["High"].eq(prices["Low"])
        & prices["Low"].eq(prices["Close"])
    )
    prices["Approx_Lower_Lock"] = (
        same_ohlc & (prices["Daily_Change_Pct"] <= -29.5)
    )
    prices["Zero_Volume"] = prices["Volume"].fillna(0).le(0)

    # 손절 전략 중 큰 손실 거래와 Time_Exit 거래를 우선 검사
    stop_trades = trades[
        trades["Strategy"].isin(
            ["Stop_5pct", "Stop_8pct", "Stop_10pct"]
        )
    ].copy()

    failures = stop_trades[
        (stop_trades["Net_Return_Pct"] < -10)
        | stop_trades["Exit_Reason"].isin(
            ["Time_Exit", "Delayed_Time_Exit"]
        )
    ].copy()

    detail_rows = []
    summary_rows = []

    for _, trade in failures.iterrows():
        ticker = trade["Ticker"]
        ticker_prices = prices[prices["Ticker"] == ticker].copy()

        entry_date = trade["Entry_Date"]
        exit_date = trade["Actual_Exit_Date"]
        scheduled_date = trade["Scheduled_Exit_Date"]
        stop_price = pd.to_numeric(
            pd.Series([trade["Stop_Price"]]), errors="coerce"
        ).iloc[0]

        if pd.isna(entry_date) or pd.isna(exit_date):
            continue

        window = ticker_prices[
            (ticker_prices["Date"] >= entry_date)
            & (ticker_prices["Date"] <= exit_date)
        ].copy()

        if window.empty:
            continue

        window["Low_Touched_Stop"] = (
            window["Low"] <= stop_price
            if pd.notna(stop_price) else False
        )
        window["Open_Below_Stop"] = (
            window["Open"] <= stop_price
            if pd.notna(stop_price) else False
        )
        window["Is_Scheduled_Exit_Date"] = (
            window["Date"] == scheduled_date
        )

        # 해당 날짜에 시가 매도가 가능했는지에 대한 단순 추정
        window["Can_Trade_Open"] = (
            window["Volume"].fillna(0).gt(0)
            & window["Open"].gt(0)
        )
        window["Low_Breach_Before_Scheduled"] = (
            window["Low_Touched_Stop"]
            & (window["Date"] < scheduled_date)
        )

        # 손절선에 처음 닿은 날짜
        breached = window[window["Low_Touched_Stop"]]
        first_breach_date = (
            breached.iloc[0]["Date"] if not breached.empty else pd.NaT
        )

        # 예정 청산일 이전에 손절선을 건드렸고,
        # 시가 거래가 가능했던 날이 있었는지 확인
        prior_breach = window[
            window["Low_Breach_Before_Scheduled"]
        ]
        prior_tradable_breach = prior_breach[
            prior_breach["Can_Trade_Open"]
            & ~prior_breach["Approx_Lower_Lock"]
        ]

        if (
            trade["Exit_Reason"] in ["Time_Exit", "Delayed_Time_Exit"]
            and not prior_tradable_breach.empty
        ):
            diagnosis = "예정 청산일 이전 손절 조건 확인 필요"
        elif (
            trade["Exit_Reason"] in ["Time_Exit", "Delayed_Time_Exit"]
            and not prior_breach.empty
        ):
            diagnosis = "손절선 도달했으나 거래 가능 여부 확인 필요"
        elif trade["Exit_Reason"] == "Gap_Stop":
            diagnosis = "최초 손절 발생일과 실제 청산일 비교 필요"
        else:
            diagnosis = "가격 흐름 확인"

        summary_rows.append({
            "Strategy": trade["Strategy"],
            "Ticker": ticker,
            "Stock_Name": trade.get("Stock_Name", ""),
            "Entry_ID": trade.get("Entry_ID", ""),
            "Entry_Date": entry_date,
            "Entry_Open": trade["Entry_Open"],
            "Stop_Price": stop_price,
            "Scheduled_Exit_Date": scheduled_date,
            "Actual_Exit_Date": exit_date,
            "Exit_Reason": trade["Exit_Reason"],
            "Net_Return_Pct": trade["Net_Return_Pct"],
            "First_Low_Touch_Date": first_breach_date,
            "Prior_Tradable_Breach_Count": len(prior_tradable_breach),
            "Prior_Breach_Count": len(prior_breach),
            "Gap_Calendar_Days": (
                (exit_date - first_breach_date).days
                if pd.notna(first_breach_date) else np.nan
            ),
            "Diagnosis": diagnosis,
        })

        for _, day in window.iterrows():
            row = {
                "Strategy": trade["Strategy"],
                "Ticker": ticker,
                "Stock_Name": trade.get("Stock_Name", ""),
                "Entry_ID": trade.get("Entry_ID", ""),
                "Entry_Date": entry_date,
                "Entry_Open": trade["Entry_Open"],
                "Stop_Price": stop_price,
                "Scheduled_Exit_Date": scheduled_date,
                "Actual_Exit_Date": exit_date,
                "Exit_Reason": trade["Exit_Reason"],
                "Net_Return_Pct": trade["Net_Return_Pct"],
            }
            row.update(day.to_dict())
            detail_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    details = pd.DataFrame(detail_rows)

    def unique_path(stem):
        path = RESULTS / f"{stem}.csv"
        n = 2
        while path.exists():
            path = RESULTS / f"{stem}_{n}.csv"
            n += 1
        return path

    summary_path = unique_path("stop_logic_verification_summary")
    details_path = unique_path("stop_logic_verification_daily")

    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    details.to_csv(details_path, index=False, encoding="utf-8-sig")

    print("\n=== 손절 로직 검증 요약 ===")
    if summary.empty:
        print("검사할 거래가 없습니다.")
    else:
        print(
            summary[
                [
                    "Strategy", "Ticker", "Entry_Date",
                    "Exit_Reason", "Net_Return_Pct",
                    "First_Low_Touch_Date",
                    "Prior_Tradable_Breach_Count", "Diagnosis"
                ]
            ].head(30).round(3).to_string(index=False)
        )

        print("\n=== 청산 사유별 검사 건수 ===")
        print(summary.groupby(["Strategy", "Exit_Reason"]).size().to_string())

        print("\n예정 청산일 이전에 거래 가능한 손절선 접촉이 확인된 건수:",
              int((summary["Prior_Tradable_Breach_Count"] > 0).sum()))

    print("\n저장 파일:")
    print(summary_path)
    print(details_path)
    print("\n일별 원본 가격과 손절선 접촉 여부를 함께 확인하세요.")
    print("하한가 여부는 일봉 데이터만으로 확정할 수 없습니다.")


if __name__ == "__main__":
    main()
