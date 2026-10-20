from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "stock_data_500.csv"
PERIODS = [20, 60, 120, 250]
HORIZONS = [1, 3, 5, 10]


def remove_repeated_signals(events, min_gap=11):
    """종목별로 첫 신호를 유지하고 이후 10개 거래행의 신호를 제외."""
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


def summarize(events, period, condition, signal_type):
    result = {
        "Period_Days": period,
        "Condition": condition,
        "Signal_Type": signal_type,
        "Signal_Count": len(events),
        "Unique_Stocks": events["Ticker"].nunique(),
    }

    for n in HORIZONS:
        returns = events[f"Return_{n}D"].dropna()

        result[f"Samples_{n}D"] = len(returns)
        result[f"Mean_{n}D_Pct"] = (
            round(returns.mean() * 100, 3) if len(returns) else None
        )
        result[f"Median_{n}D_Pct"] = (
            round(returns.median() * 100, 3) if len(returns) else None
        )
        result[f"Positive_{n}D_Pct"] = (
            round((returns > 0).mean() * 100, 2) if len(returns) else None
        )
        result[f"Negative_{n}D_Pct"] = (
            round((returns < 0).mean() * 100, 2) if len(returns) else None
        )
        result[f"Hit_5Pct_{n}D_Pct"] = (
            round((returns >= 0.05).mean() * 100, 2)
            if len(returns) else None
        )

    return result


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"입력 파일이 없습니다: {INPUT_FILE}")

    df = pd.read_csv(
        INPUT_FILE, encoding="utf-8-sig", low_memory=False
    )

    required = {
        "Date", "Open", "High", "Close", "Volume", "Ticker", "Stock_Name"
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"필수 컬럼 누락: {missing}")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    for col in ["Open", "High", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Ticker"] = (
        df["Ticker"].astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.strip()
    )

    df = df.dropna(
        subset=["Date", "Open", "High", "Close", "Volume", "Ticker"]
    )
    df = df[
        (df["Open"] > 0)
        & (df["High"] > 0)
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

    # 공통 A 조건의 데이터
    df["Prev20D_Avg_Volume"] = grouped["Volume"].transform(
        lambda s: s.shift(1).rolling(20, min_periods=20).mean()
    )
    df["Volume_Multiple"] = (
        df["Volume"] / df["Prev20D_Avg_Volume"].where(
            df["Prev20D_Avg_Volume"] > 0
        )
    )
    df["Prev_Close"] = grouped["Close"].shift(1)
    df["Daily_Return"] = df["Close"] / df["Prev_Close"] - 1

    df["Prev20D_High"] = grouped["High"].transform(
        lambda s: s.shift(1).rolling(20, min_periods=20).max()
    )
    df["Close_to_20D_High_Pct"] = (
        df["Close"] / df["Prev20D_High"] * 100
    )

    # 다음 거래일 시가 진입, 신호일 기준 n번째 다음 종가 청산
    df["Entry_Next_Open"] = grouped["Open"].shift(-1)
    for n in HORIZONS:
        df[f"Return_{n}D"] = (
            grouped["Close"].shift(-n) / df["Entry_Next_Open"] - 1
        )

    base = (
        (df["Volume_Multiple"] >= 2.0)
        & (df["Daily_Return"] >= -0.03)
        & (df["Daily_Return"] < 0.03)
        & df["Prev20D_Avg_Volume"].notna()
        & df["Prev20D_High"].notna()
    )

    conditions = {
        "A_기존조건": base,
        "B1_고점95퍼센트": base & (
            df["Close_to_20D_High_Pct"] >= 95
        ),
        "B2_고점97퍼센트": base & (
            df["Close_to_20D_High_Pct"] >= 97
        ),
        "B3_고점99퍼센트": base & (
            df["Close_to_20D_High_Pct"] >= 99
        ),
    }

    all_dates = sorted(df["Date"].dropna().unique())
    if not all_dates:
        raise ValueError("유효한 날짜 데이터가 없습니다.")

    # 각 조건을 전체 이력에서 개별적으로 중복 제거한 뒤 기간별로 비교
    raw_events = {}
    dedup_events = {}

    for name, mask in conditions.items():
        raw_events[name] = df.loc[mask].copy()
        dedup_events[name] = remove_repeated_signals(
            raw_events[name], min_gap=11
        )

    summary_rows = []
    event_frames = []

    for period in PERIODS:
        start_date = pd.Timestamp(
            all_dates[max(0, len(all_dates) - period)]
        )

        for name in conditions:
            for signal_type, source in [
                ("원본", raw_events[name]),
                ("중복제거", dedup_events[name]),
            ]:
                events = source[source["Date"] >= start_date].copy()

                summary_rows.append(
                    summarize(events, period, name, signal_type)
                )

                if not events.empty:
                    events["Period_Days"] = period
                    events["Condition"] = name
                    events["Signal_Type"] = signal_type
                    events["Daily_Return_Pct"] = (
                        events["Daily_Return"] * 100
                    )
                    for n in HORIZONS:
                        events[f"Return_{n}D_Pct"] = (
                            events[f"Return_{n}D"] * 100
                        )
                    event_frames.append(events)

        print(f"{period}거래일 기간 처리 완료")

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(
        BASE_DIR / "volume_high_dedup_comparison_v4.csv",
        index=False, encoding="utf-8-sig"
    )

    if event_frames:
        events_out = pd.concat(event_frames, ignore_index=True)
        columns = [
            "Period_Days", "Condition", "Signal_Type",
            "Date", "Stock_Name", "Ticker", "Close",
            "Volume", "Prev20D_Avg_Volume", "Volume_Multiple",
            "Daily_Return_Pct", "Prev20D_High",
            "Close_to_20D_High_Pct", "Entry_Next_Open",
            "Return_1D_Pct", "Return_3D_Pct",
            "Return_5D_Pct", "Return_10D_Pct",
        ]
        events_out[columns].to_csv(
            BASE_DIR / "volume_high_dedup_events_v4.csv",
            index=False, encoding="utf-8-sig"
        )

    display_cols = [
        "Period_Days", "Condition", "Signal_Type",
        "Signal_Count", "Unique_Stocks",
        "Samples_5D", "Median_5D_Pct", "Positive_5D_Pct",
        "Negative_5D_Pct", "Samples_10D", "Median_10D_Pct",
        "Positive_10D_Pct", "Negative_10D_Pct",
        "Hit_5Pct_10D_Pct",
    ]

    print("\n" + "=" * 90)
    print("V4 고점 조건별 반복 신호 제거 비교")
    print("=" * 90)
    print(f"데이터 기간: {df['Date'].min().date()} ~ {df['Date'].max().date()}")
    print("중복 제거: 같은 종목에서 신호 후 10개 거래행의 신호 제외")
    print("A/B1/B2/B3는 각각 독립적으로 중복 제거했습니다.")
    print("\n" + summary[display_cols].to_string(index=False))
    print("\n생성 파일:")
    print("1. volume_high_dedup_comparison_v4.csv")
    print("2. volume_high_dedup_events_v4.csv")
    print("거래비용, 세금, 슬리피지는 미반영입니다.")


if __name__ == "__main__":
    main()
