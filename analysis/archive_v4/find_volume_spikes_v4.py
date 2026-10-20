from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "stock_data_500.csv"


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"파일을 찾을 수 없습니다: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE, encoding="utf-8-sig", low_memory=False)

    required = {"Date", "Close", "Volume", "Ticker", "Stock_Name"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"필수 컬럼 누락: {missing}")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    for col in ["Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Date", "Close", "Volume", "Ticker"])
    df = df[(df["Close"] > 0) & (df["Volume"] >= 0)].copy()
    df["Ticker"] = (
        df["Ticker"].astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
    )
    df = df[df["Ticker"] != ""]
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
    safe_avg = df["Prev20D_Avg_Volume"].where(
        df["Prev20D_Avg_Volume"] != 0
    )
    df["Volume_Multiple"] = df["Volume"] / safe_avg

    # 당일 종가 기준 등락률
    df["Prev_Close"] = grouped["Close"].shift(1)
    df["Daily_Return"] = df["Close"] / df["Prev_Close"] - 1

    # 신호 발생 후 종가수익률
    for n in [1, 3, 5, 10]:
        future_close = grouped["Close"].shift(-n)
        df[f"Return_{n}D"] = future_close / df["Close"] - 1

    # 거래량 3배 이상 + 당일 등락률 -3% 이상, +3% 미만
    signal = (
        (df["Volume_Multiple"] >= 3.0)
        & (df["Daily_Return"] >= -0.03)
        & (df["Daily_Return"] < 0.03)
        & df["Prev20D_Avg_Volume"].notna()
    )
    events = df.loc[signal].copy()

    dates = sorted(df["Date"].dropna().unique())
    if not dates:
        raise ValueError("유효한 거래 날짜가 없습니다.")

    start_date = pd.Timestamp(dates[max(0, len(dates) - 20)])
    latest_date = pd.Timestamp(dates[-1])

    # 최근 20거래일 후보
    recent = events[events["Date"] >= start_date].copy()

    for frame in [events, recent]:
        frame["Daily_Return_Pct"] = frame["Daily_Return"] * 100
        frame["Volume_Multiple"] = frame["Volume_Multiple"].round(2)
        for n in [1, 3, 5, 10]:
            frame[f"Return_{n}D_Pct"] = frame[f"Return_{n}D"] * 100

    columns = [
        "Date", "Stock_Name", "Ticker", "Close", "Volume",
        "Prev20D_Avg_Volume", "Volume_Multiple", "Daily_Return_Pct",
        "Return_1D_Pct", "Return_3D_Pct",
        "Return_5D_Pct", "Return_10D_Pct",
    ]

    events[columns].sort_values(
        ["Date", "Volume_Multiple"], ascending=[False, False]
    ).to_csv(
        BASE_DIR / "volume_spike_events_v4.csv",
        index=False, encoding="utf-8-sig"
    )

    recent[columns].sort_values(
        ["Date", "Volume_Multiple"], ascending=[False, False]
    ).to_csv(
        BASE_DIR / "volume_spike_candidates_v4.csv",
        index=False, encoding="utf-8-sig"
    )

    # 전체 과거 신호의 이후 수익률 요약
    summary = []
    for n in [1, 3, 5, 10]:
        returns = events[f"Return_{n}D"].dropna()
        summary.append({
            "Horizon": f"{n}D",
            "Samples": len(returns),
            "Mean_Return_Pct": round(returns.mean() * 100, 3)
                if len(returns) else None,
            "Median_Return_Pct": round(returns.median() * 100, 3)
                if len(returns) else None,
            "Positive_Return_Rate_Pct": round(
                (returns > 0).mean() * 100, 2
            ) if len(returns) else None,
            "Hit_3Pct_Rate_Pct": round(
                (returns >= 0.03).mean() * 100, 2
            ) if len(returns) else None,
            "Hit_5Pct_Rate_Pct": round(
                (returns >= 0.05).mean() * 100, 2
            ) if len(returns) else None,
            "Hit_10Pct_Rate_Pct": round(
                (returns >= 0.10).mean() * 100, 2
            ) if len(returns) else None,
        })

    pd.DataFrame(summary).to_csv(
        BASE_DIR / "volume_spike_summary_v4.csv",
        index=False, encoding="utf-8-sig"
    )

    print("=" * 55)
    print("V4 거래량 급증 탐색 완료")
    print("=" * 55)
    print(f"전체 데이터: {df['Date'].min().date()} ~ {latest_date.date()}")
    print(f"최근 20거래일: {start_date.date()} ~ {latest_date.date()}")
    print(f"전체 기간 신호: {len(events):,}개")
    print(f"최근 기간 후보: {len(recent):,}개")
    print("\n[과거 전체 신호의 이후 수익률]")
    print(pd.DataFrame(summary).to_string(index=False))
    print("\n생성 파일:")
    print("1. volume_spike_candidates_v4.csv")
    print("2. volume_spike_events_v4.csv")
    print("3. volume_spike_summary_v4.csv")
    print("\n참고: 최근 신호는 미래 데이터가 없어 일부 수익률이 비어 있을 수 있습니다.")
    print("거래비용과 세금은 반영하지 않았습니다.")


if __name__ == "__main__":
    main()
