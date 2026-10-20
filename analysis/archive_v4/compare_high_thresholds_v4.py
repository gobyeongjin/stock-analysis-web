from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "stock_data_500.csv"
PERIODS = [20, 60, 120, 250]
HORIZONS = [1, 3, 5, 10]


def summarize(events, period, condition):
    result = {
        "Period_Days": period,
        "Condition": condition,
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

    df = pd.read_csv(INPUT_FILE, encoding="utf-8-sig", low_memory=False)

    required = {"Date", "Open", "High", "Close", "Volume", "Ticker", "Stock_Name"}
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

    # 기존 A 조건: 당일을 제외한 직전 20거래일 평균 거래량의 2배 이상
    df["Prev20D_Avg_Volume"] = grouped["Volume"].transform(
        lambda s: s.shift(1).rolling(20, min_periods=20).mean()
    )
    df["Volume_Multiple"] = (
        df["Volume"] / df["Prev20D_Avg_Volume"].where(
            df["Prev20D_Avg_Volume"] > 0
        )
    )

    # 당일 등락률
    df["Prev_Close"] = grouped["Close"].shift(1)
    df["Daily_Return"] = df["Close"] / df["Prev_Close"] - 1

    # 고점 기준은 신호일을 제외한 직전 20거래일의 최고가
    df["Prev20D_High"] = grouped["High"].transform(
        lambda s: s.shift(1).rolling(20, min_periods=20).max()
    )
    df["Close_to_20D_High_Pct"] = (
        df["Close"] / df["Prev20D_High"] * 100
    )

    # 다음 거래일 시가 진입, 신호일로부터 n번째 거래일 종가 청산
    df["Entry_Next_Open"] = grouped["Open"].shift(-1)
    for n in HORIZONS:
        df[f"Return_{n}D"] = (
            grouped["Close"].shift(-n) / df["Entry_Next_Open"] - 1
        )

    base = (
        (df["Volume_Multiple"] >= 2.0)
        & (df["Daily_Return"] >= -0.03)
        & (df["Daily_Return"] < 0.03)
        & df["Prev20D_High"].notna()
        & df["Prev20D_Avg_Volume"].notna()
    )

    # A는 고점 필터 없는 기준, B1~B3는 고점 근접 필터 추가
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

    summary_rows = []
    event_frames = []

    for period in PERIODS:
        start_date = pd.Timestamp(
            all_dates[max(0, len(all_dates) - period)]
        )

        for name, mask in conditions.items():
            events = df.loc[mask & (df["Date"] >= start_date)].copy()
            summary_rows.append(summarize(events, period, name))

            if not events.empty:
                events["Period_Days"] = period
                events["Condition"] = name
                events["Daily_Return_Pct"] = events["Daily_Return"] * 100

                for n in HORIZONS:
                    events[f"Return_{n}D_Pct"] = (
                        events[f"Return_{n}D"] * 100
                    )

                event_frames.append(events)

        print(f"{period}거래일 기간 분석 완료")

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(
        BASE_DIR / "volume_high_threshold_comparison_v4.csv",
        index=False,
        encoding="utf-8-sig",
    )

    if event_frames:
        all_events = pd.concat(event_frames, ignore_index=True)
        columns = [
            "Period_Days", "Condition", "Date", "Stock_Name", "Ticker",
            "Close", "Volume", "Prev20D_Avg_Volume", "Volume_Multiple",
            "Daily_Return_Pct", "Prev20D_High", "Close_to_20D_High_Pct",
            "Entry_Next_Open", "Return_1D_Pct", "Return_3D_Pct",
            "Return_5D_Pct", "Return_10D_Pct",
        ]
        all_events[columns].to_csv(
            BASE_DIR / "volume_high_threshold_events_v4.csv",
            index=False,
            encoding="utf-8-sig",
        )

    display_columns = [
        "Period_Days", "Condition", "Signal_Count", "Unique_Stocks",
        "Samples_5D", "Mean_5D_Pct", "Median_5D_Pct",
        "Positive_5D_Pct", "Negative_5D_Pct",
        "Samples_10D", "Mean_10D_Pct", "Median_10D_Pct",
        "Positive_10D_Pct", "Negative_10D_Pct", "Hit_5Pct_10D_Pct",
    ]

    print("\n" + "=" * 90)
    print("V4 고점 근접 기준 비교")
    print("=" * 90)
    print(f"데이터 기간: {df['Date'].min().date()} ~ {df['Date'].max().date()}")
    print("공통 조건: 거래량 2배 이상, 당일 등락률 -3% 이상 +3% 미만")
    print("고점 기준: 신호일을 제외한 직전 20거래일 최고가 대비 종가 비율")
    print("\n" + summary[display_columns].to_string(index=False))
    print("\n생성 파일:")
    print("1. volume_high_threshold_comparison_v4.csv")
    print("2. volume_high_threshold_events_v4.csv")
    print("주의: 반복 신호 및 거래비용은 아직 반영하지 않았습니다.")


if __name__ == "__main__":
    main()
