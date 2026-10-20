from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "stock_data_500.csv"
PERIODS = [20, 60, 120, 250]
HORIZONS = [1, 3, 5, 10]


def summarize(events, period, mode):
    row = {
        "Period_Days": period,
        "Signal_Type": mode,
        "Signal_Count": len(events),
        "Unique_Stocks": events["Ticker"].nunique(),
    }

    for n in HORIZONS:
        returns = events[f"Return_{n}D"].dropna()
        row[f"Samples_{n}D"] = len(returns)
        row[f"Mean_{n}D_Pct"] = (
            round(returns.mean() * 100, 3) if len(returns) else None
        )
        row[f"Median_{n}D_Pct"] = (
            round(returns.median() * 100, 3) if len(returns) else None
        )
        row[f"Positive_{n}D_Pct"] = (
            round((returns > 0).mean() * 100, 2) if len(returns) else None
        )
        row[f"Negative_{n}D_Pct"] = (
            round((returns < 0).mean() * 100, 2) if len(returns) else None
        )

    return row


def remove_repeated_signals(events, min_gap=10):
    """같은 종목은 신호 간격이 최소 10거래일이 되도록 첫 신호부터 유지."""
    kept = []

    for ticker, group in events.groupby("Ticker", sort=False):
        group = group.sort_values("Ticker_Row")
        last_kept_row = None

        for _, event in group.iterrows():
            current_row = int(event["Ticker_Row"])

            if (
                last_kept_row is None
                or current_row - last_kept_row >= min_gap
            ):
                kept.append(event)
                last_kept_row = current_row

    if not kept:
        return events.iloc[0:0].copy()

    return pd.DataFrame(kept).reset_index(drop=True)


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {INPUT_FILE}")

    df = pd.read_csv(
        INPUT_FILE, encoding="utf-8-sig", low_memory=False
    )

    required = {
        "Date", "Open", "Close", "Volume", "Ticker", "Stock_Name"
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"필수 컬럼 누락: {missing}")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    for col in ["Open", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Ticker"] = (
        df["Ticker"].astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.strip()
    )

    df = df.dropna(
        subset=["Date", "Open", "Close", "Volume", "Ticker"]
    )
    df = df[
        (df["Open"] > 0)
        & (df["Close"] > 0)
        & (df["Volume"] >= 0)
        & (df["Ticker"] != "")
    ].copy()

    df = (
        df.sort_values(["Ticker", "Date"])
        .drop_duplicates(["Ticker", "Date"], keep="last")
        .reset_index(drop=True)
    )

    grouped = df.groupby("Ticker", sort=False)
    df["Ticker_Row"] = grouped.cumcount()

    # 당일 거래량을 제외한 직전 20거래일 평균
    df["Prev20D_Avg_Volume"] = grouped["Volume"].transform(
        lambda s: s.shift(1).rolling(20, min_periods=20).mean()
    )
    safe_avg = df["Prev20D_Avg_Volume"].where(
        df["Prev20D_Avg_Volume"] > 0
    )
    df["Volume_Multiple"] = df["Volume"] / safe_avg

    # 신호일 종가 기준 등락률
    df["Prev_Close"] = grouped["Close"].shift(1)
    df["Daily_Return"] = df["Close"] / df["Prev_Close"] - 1

    # 신호 다음 거래일 시가 진입 가정
    df["Entry_Next_Open"] = grouped["Open"].shift(-1)

    for n in HORIZONS:
        df[f"Return_{n}D"] = (
            grouped["Close"].shift(-n) / df["Entry_Next_Open"] - 1
        )

    signal_mask = (
        (df["Volume_Multiple"] >= 2.0)
        & (df["Daily_Return"] >= -0.03)
        & (df["Daily_Return"] < 0.03)
        & df["Prev20D_Avg_Volume"].notna()
    )
    all_signals = df.loc[signal_mask].copy()

    dates = sorted(df["Date"].dropna().unique())
    latest_date = pd.Timestamp(dates[-1])
    summary_rows = []
    candidate_rows = []

    for days in PERIODS:
        start_date = pd.Timestamp(dates[max(0, len(dates) - days)])
        window = all_signals[all_signals["Date"] >= start_date].copy()

        # 기간별로 독립적으로 중복 신호를 제거
        dedup = remove_repeated_signals(window, min_gap=10)

        summary_rows.append(summarize(window, days, "원본"))
        summary_rows.append(summarize(dedup, days, "10거래일 간격"))

        for mode, frame in [("원본", window), ("10거래일 간격", dedup)]:
            if frame.empty:
                continue
            saved = frame.copy()
            saved["Period_Days"] = days
            saved["Signal_Type"] = mode
            saved["Daily_Return_Pct"] = saved["Daily_Return"] * 100
            saved["Volume_Multiple"] = saved["Volume_Multiple"].round(2)

            for n in HORIZONS:
                saved[f"Return_{n}D_Pct"] = saved[f"Return_{n}D"] * 100

            candidate_rows.append(saved)

        print(
            f"{days}일 기간 처리 완료: "
            f"원본 {len(window):,}건 -> 중복 제거 {len(dedup):,}건"
        )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(
        BASE_DIR / "volume_dedup_comparison_v4.csv",
        index=False, encoding="utf-8-sig"
    )

    if candidate_rows:
        candidates = pd.concat(candidate_rows, ignore_index=True)
        columns = [
            "Period_Days", "Signal_Type", "Date", "Stock_Name", "Ticker",
            "Close", "Volume", "Prev20D_Avg_Volume", "Volume_Multiple",
            "Daily_Return_Pct", "Entry_Next_Open",
            "Return_1D_Pct", "Return_3D_Pct",
            "Return_5D_Pct", "Return_10D_Pct",
        ]
        candidates[columns].sort_values(
            ["Period_Days", "Signal_Type", "Date"],
            ascending=[True, True, False]
        ).to_csv(
            BASE_DIR / "volume_dedup_events_v4.csv",
            index=False, encoding="utf-8-sig"
        )

    display_cols = [
        "Period_Days", "Signal_Type", "Signal_Count", "Unique_Stocks",
        "Samples_5D", "Mean_5D_Pct", "Median_5D_Pct",
        "Positive_5D_Pct", "Negative_5D_Pct",
        "Samples_10D", "Mean_10D_Pct", "Median_10D_Pct",
        "Positive_10D_Pct", "Negative_10D_Pct",
    ]

    print("\n" + "=" * 85)
    print("V4 반복 신호 제거 성과 비교")
    print("=" * 85)
    print(f"데이터 기간: {df['Date'].min().date()} ~ {latest_date.date()}")
    print("조건: 거래량 2배 이상, 당일 등락률 -3% 이상 +3% 미만")
    print("중복 제거: 같은 종목의 신호를 최소 10개 종목 거래행 간격으로 유지")
    print("\n[기간별 비교]")
    print(summary[display_cols].to_string(index=False))
    print("\n생성 파일:")
    print("1. volume_dedup_comparison_v4.csv")
    print("2. volume_dedup_events_v4.csv")
    print("주의: 거래비용·세금·슬리피지는 미반영입니다.")
    print("같은 종목은 10거래일 간격으로 제한했지만, 서로 다른 종목의 신호는 같은 날 발생할 수 있습니다.")


if __name__ == "__main__":
    main()
