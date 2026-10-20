from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "stock_data_500.csv"

PERIODS = [20, 60, 120, 250]


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE, encoding="utf-8-sig", low_memory=False)

    required = {"Date", "Open", "Close", "Volume", "Ticker", "Stock_Name"}
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

    # 당일 거래량을 제외한 직전 20거래일 평균
    df["Prev20D_Avg_Volume"] = grouped["Volume"].transform(
        lambda s: s.shift(1).rolling(20, min_periods=20).mean()
    )
    avg_volume = df["Prev20D_Avg_Volume"].where(
        df["Prev20D_Avg_Volume"] > 0
    )
    df["Volume_Multiple"] = df["Volume"] / avg_volume

    # 당일 종가 기준 등락률
    df["Prev_Close"] = grouped["Close"].shift(1)
    df["Daily_Return"] = df["Close"] / df["Prev_Close"] - 1

    # 신호 발생 후 다음 거래일 시가 진입 가정
    df["Entry_Next_Open"] = grouped["Open"].shift(-1)

    for n in [1, 3, 5, 10]:
        df[f"Return_{n}D"] = (
            grouped["Close"].shift(-n) / df["Entry_Next_Open"] - 1
        )

    signal = (
        (df["Volume_Multiple"] >= 2.0)
        & (df["Daily_Return"] >= -0.03)
        & (df["Daily_Return"] < 0.03)
        & df["Prev20D_Avg_Volume"].notna()
    )
    events = df.loc[signal].copy()

    all_dates = sorted(df["Date"].dropna().unique())
    latest_date = pd.Timestamp(all_dates[-1])
    period_rows = []
    candidate_frames = []

    for days in PERIODS:
        start_idx = max(0, len(all_dates) - days)
        start_date = pd.Timestamp(all_dates[start_idx])

        recent = events[events["Date"] >= start_date].copy()
        candidate_frames.append(recent.assign(Period_Days=days))

        row = {
            "Period_Days": days,
            "Start_Date": start_date.date(),
            "End_Date": latest_date.date(),
            "Signal_Count": len(recent),
            "Unique_Stocks": recent["Ticker"].nunique(),
        }

        for n in [1, 3, 5, 10]:
            # 미래 수익률이 계산 가능한 신호만 평가
            returns = recent[f"Return_{n}D"].dropna()
            row[f"Samples_{n}D"] = len(returns)
            row[f"Mean_{n}D_Pct"] = (
                round(returns.mean() * 100, 3)
                if len(returns) else None
            )
            row[f"Median_{n}D_Pct"] = (
                round(returns.median() * 100, 3)
                if len(returns) else None
            )
            row[f"Positive_{n}D_Pct"] = (
                round((returns > 0).mean() * 100, 2)
                if len(returns) else None
            )
            row[f"Hit_5Pct_{n}D_Pct"] = (
                round((returns >= 0.05).mean() * 100, 2)
                if len(returns) else None
            )

        period_rows.append(row)

    summary = pd.DataFrame(period_rows)
    summary.to_csv(
        BASE_DIR / "volume_period_comparison_v4.csv",
        index=False, encoding="utf-8-sig"
    )

    candidates = pd.concat(candidate_frames, ignore_index=True)
    candidates["Daily_Return_Pct"] = candidates["Daily_Return"] * 100
    candidates["Volume_Multiple"] = candidates["Volume_Multiple"].round(2)

    for n in [1, 3, 5, 10]:
        candidates[f"Return_{n}D_Pct"] = candidates[f"Return_{n}D"] * 100

    columns = [
        "Period_Days", "Date", "Stock_Name", "Ticker", "Close",
        "Volume", "Prev20D_Avg_Volume", "Volume_Multiple",
        "Daily_Return_Pct", "Entry_Next_Open",
        "Return_1D_Pct", "Return_3D_Pct",
        "Return_5D_Pct", "Return_10D_Pct",
    ]
    candidates[columns].sort_values(
        ["Period_Days", "Date"],
        ascending=[True, False]
    ).to_csv(
        BASE_DIR / "volume_period_candidates_v4.csv",
        index=False, encoding="utf-8-sig"
    )

    display_cols = [
        "Period_Days", "Start_Date", "End_Date",
        "Signal_Count", "Unique_Stocks",
        "Samples_5D", "Mean_5D_Pct", "Median_5D_Pct",
        "Positive_5D_Pct", "Mean_10D_Pct",
        "Median_10D_Pct", "Positive_10D_Pct",
    ]

    print("=" * 70)
    print("V4 기간별 거래량 신호 비교 완료")
    print("=" * 70)
    print(f"전체 데이터: {df['Date'].min().date()} ~ {latest_date.date()}")
    print("\n[기간별 후보 수 및 수익률]")
    print(summary[display_cols].to_string(index=False))
    print("\n생성 파일:")
    print("1. volume_period_comparison_v4.csv")
    print("2. volume_period_candidates_v4.csv")
    print("\n기준: 거래량 2배 이상, 당일 등락률 -3% 이상 +3% 미만")
    print("수익률은 신호 다음 거래일 시가 진입 가정이며 거래비용은 미반영입니다.")
    print("같은 종목의 반복 신호가 포함될 수 있습니다.")


if __name__ == "__main__":
    main()
