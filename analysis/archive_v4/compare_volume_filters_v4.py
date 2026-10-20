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

    df = pd.read_csv(
        INPUT_FILE, encoding="utf-8-sig", low_memory=False
    )

    required = {
        "Date", "Open", "Close", "Volume", "Ticker", "Stock_Name", "High"
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"필수 컬럼이 없습니다: {missing}\n"
            "CSV의 실제 컬럼명을 확인해 주세요."
        )

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

    # 당일을 제외한 직전 20거래일 평균 거래량
    df["Prev20D_Avg_Volume"] = grouped["Volume"].transform(
        lambda s: s.shift(1).rolling(20, min_periods=20).mean()
    )
    df["Volume_Multiple"] = (
        df["Volume"] / df["Prev20D_Avg_Volume"].where(
            df["Prev20D_Avg_Volume"] > 0
        )
    )

    # 신호일 종가 기준 등락률
    df["Prev_Close"] = grouped["Close"].shift(1)
    df["Daily_Return"] = df["Close"] / df["Prev_Close"] - 1

    # 조건 B: 신호일 종가가 직전 20일 최고가의 95% 이상
    df["Prev20D_High"] = grouped["High"].transform(
        lambda s: s.shift(1).rolling(20, min_periods=20).max()
    )
    df["Near_20D_High"] = (
        df["Close"] >= df["Prev20D_High"] * 0.95
    )

    # 조건 C: 거래량 신호 발생 전 변동성이 축소되었는지 확인
    # Daily_Return의 이전 5개/20개 관측치만 사용
    df["Prev_Volatility_5D"] = grouped["Daily_Return"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=5).std()
    )
    df["Prev_Volatility_20D"] = grouped["Daily_Return"].transform(
        lambda s: s.shift(1).rolling(20, min_periods=20).std()
    )
    df["Volatility_Contracted"] = (
        (df["Prev_Volatility_20D"] > 0)
        & (
            df["Prev_Volatility_5D"]
            <= df["Prev_Volatility_20D"] * 0.8
        )
    )

    # 신호 다음 거래일 시가 진입
    df["Entry_Next_Open"] = grouped["Open"].shift(-1)

    # 1/3/5/10 거래일 보유 후 종가 청산
    for n in HORIZONS:
        df[f"Return_{n}D"] = (
            grouped["Close"].shift(-n) / df["Entry_Next_Open"] - 1
        )

    # 공통 기준 A
    base = (
        (df["Volume_Multiple"] >= 2.0)
        & (df["Daily_Return"] >= -0.03)
        & (df["Daily_Return"] < 0.03)
        & df["Prev20D_Avg_Volume"].notna()
    )

    # B/C는 A에 추가 필터를 적용
    conditions = {
        "A_기존조건": base,
        "B_20일고점근처": base & df["Near_20D_High"],
        "C_변동성축소": base & df["Volatility_Contracted"],
    }

    all_dates = sorted(df["Date"].dropna().unique())
    if not all_dates:
        raise ValueError("유효한 날짜 데이터가 없습니다.")

    latest_date = pd.Timestamp(all_dates[-1])
    summary_rows = []
    event_frames = []

    for period in PERIODS:
        start_idx = max(0, len(all_dates) - period)
        start_date = pd.Timestamp(all_dates[start_idx])

        for condition_name, mask in conditions.items():
            events = df.loc[
                mask & (df["Date"] >= start_date)
            ].copy()

            summary_rows.append(
                summarize(events, period, condition_name)
            )

            if not events.empty:
                events["Period_Days"] = period
                events["Condition"] = condition_name
                events["Daily_Return_Pct"] = (
                    events["Daily_Return"] * 100
                )
                for n in HORIZONS:
                    events[f"Return_{n}D_Pct"] = (
                        events[f"Return_{n}D"] * 100
                    )
                event_frames.append(events)

        print(f"{period}거래일 기간 분석 완료")

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(
        BASE_DIR / "volume_filter_comparison_v4.csv",
        index=False, encoding="utf-8-sig"
    )

    if event_frames:
        events_out = pd.concat(event_frames, ignore_index=True)
        output_columns = [
            "Period_Days", "Condition", "Date", "Stock_Name", "Ticker",
            "Close", "Volume", "Prev20D_Avg_Volume", "Volume_Multiple",
            "Daily_Return_Pct", "Prev20D_High",
            "Prev_Volatility_5D", "Prev_Volatility_20D",
            "Entry_Next_Open", "Return_1D_Pct", "Return_3D_Pct",
            "Return_5D_Pct", "Return_10D_Pct",
        ]
        events_out[output_columns].to_csv(
            BASE_DIR / "volume_filter_events_v4.csv",
            index=False, encoding="utf-8-sig"
        )

    display_columns = [
        "Period_Days", "Condition", "Signal_Count", "Unique_Stocks",
        "Samples_5D", "Mean_5D_Pct", "Median_5D_Pct",
        "Positive_5D_Pct", "Negative_5D_Pct", "Hit_5Pct_5D_Pct",
        "Samples_10D", "Mean_10D_Pct", "Median_10D_Pct",
        "Positive_10D_Pct", "Negative_10D_Pct", "Hit_5Pct_10D_Pct",
    ]

    print("\n" + "=" * 80)
    print("V4 거래량 필터 비교")
    print("=" * 80)
    print(f"전체 데이터: {df['Date'].min().date()} ~ {latest_date.date()}")
    print("A: 거래량 2배 이상 + 당일 등락률 -3% 이상, +3% 미만")
    print("B: A + 종가가 직전 20일 최고가의 95% 이상")
    print("C: A + 직전 5일 변동성이 직전 20일 변동성의 80% 이하")
    print("\n" + summary[display_columns].to_string(index=False))
    print("\n생성 파일:")
    print("1. volume_filter_comparison_v4.csv")
    print("2. volume_filter_events_v4.csv")
    print("\n주의: 반복 신호는 아직 제거하지 않았습니다.")
    print("거래비용, 세금, 슬리피지는 미반영입니다.")


if __name__ == "__main__":
    main()
