from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "stock_data_500.csv"

RULES = {
    "A_2배_±3퍼센트": {
        "volume": 2.0, "low": -0.03, "high": 0.03
    },
    "B_3배_±3퍼센트": {
        "volume": 3.0, "low": -0.03, "high": 0.03
    },
    "C_4배_±3퍼센트": {
        "volume": 4.0, "low": -0.03, "high": 0.03
    },
    "D_3배_±1퍼센트": {
        "volume": 3.0, "low": -0.01, "high": 0.01
    },
    "E_3배_상승0~3퍼센트": {
        "volume": 3.0, "low": 0.0, "high": 0.03
    },
}


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"파일 없음: {INPUT_FILE}")

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

    # 신호 발생 다음 거래일 시가 진입
    df["Entry_Next_Open"] = grouped["Open"].shift(-1)

    # 다음 거래일 종가부터 1, 3, 5, 10일째 종가까지의 수익률
    for n in [1, 3, 5, 10]:
        exit_close = grouped["Close"].shift(-n)
        df[f"Return_{n}D"] = exit_close / df["Entry_Next_Open"] - 1

    dates = sorted(df["Date"].dropna().unique())
    if not dates:
        raise ValueError("유효한 거래 날짜가 없습니다.")

    recent_start = pd.Timestamp(dates[max(0, len(dates) - 20)])
    latest_date = pd.Timestamp(dates[-1])

    all_events = []
    summary_rows = []
    recent_rows = []

    for rule_name, rule in RULES.items():
        signal = (
            (df["Volume_Multiple"] >= rule["volume"])
            & (df["Daily_Return"] >= rule["low"])
            & (df["Daily_Return"] < rule["high"])
            & df["Prev20D_Avg_Volume"].notna()
        )

        events = df.loc[signal].copy()
        events["Rule"] = rule_name
        events["Daily_Return_Pct"] = events["Daily_Return"] * 100
        events["Volume_Multiple"] = events["Volume_Multiple"].round(2)

        for n in [1, 3, 5, 10]:
            events[f"Return_{n}D_Pct"] = events[f"Return_{n}D"] * 100

        all_events.append(events)

        # 각 기간의 수익률이 계산 가능한 표본만 평가
        row = {
            "Rule": rule_name,
            "Volume_Threshold": rule["volume"],
            "Daily_Return_Min_Pct": rule["low"] * 100,
            "Daily_Return_Max_Exclusive_Pct": rule["high"] * 100,
            "Total_Signals": len(events),
        }

        for n in [1, 3, 5, 10]:
            returns = events[f"Return_{n}D"].dropna()

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
        summary_rows.append(row)

        recent = events[events["Date"] >= recent_start]
        recent_rows.append({
            "Rule": rule_name,
            "Recent_Start": recent_start.date(),
            "Latest_Date": latest_date.date(),
            "Recent_Signals": len(recent),
            "Unique_Stocks": recent["Ticker"].nunique(),
        })

    event_df = pd.concat(all_events, ignore_index=True)

    event_columns = [
        "Rule", "Date", "Stock_Name", "Ticker", "Close", "Volume",
        "Prev20D_Avg_Volume", "Volume_Multiple", "Daily_Return_Pct",
        "Entry_Next_Open",
        "Return_1D_Pct", "Return_3D_Pct",
        "Return_5D_Pct", "Return_10D_Pct",
    ]

    event_df[event_columns].sort_values(
        ["Date", "Rule"], ascending=[False, True]
    ).to_csv(
        BASE_DIR / "volume_rule_events_v4.csv",
        index=False, encoding="utf-8-sig"
    )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(
        BASE_DIR / "volume_rule_comparison_v4.csv",
        index=False, encoding="utf-8-sig"
    )

    recent_df = pd.DataFrame(recent_rows)
    recent_df.to_csv(
        BASE_DIR / "volume_rule_recent_counts_v4.csv",
        index=False, encoding="utf-8-sig"
    )

    display_cols = [
        "Rule", "Total_Signals",
        "Samples_5D", "Mean_5D_Pct", "Median_5D_Pct",
        "Positive_5D_Pct", "Hit_5Pct_5D_Pct",
        "Mean_10D_Pct", "Median_10D_Pct",
        "Positive_10D_Pct",
    ]

    print("=" * 70)
    print("V4 거래량 조건 비교 완료")
    print("=" * 70)
    print(f"데이터 기간: {df['Date'].min().date()} ~ {latest_date.date()}")
    print(f"최근 20거래일: {recent_start.date()} ~ {latest_date.date()}")

    print("\n[전체 기간 성과 비교]")
    print(summary_df[display_cols].to_string(index=False))

    print("\n[최근 20거래일 신호 수]")
    print(recent_df.to_string(index=False))

    print("\n생성 파일:")
    print("1. volume_rule_comparison_v4.csv")
    print("2. volume_rule_recent_counts_v4.csv")
    print("3. volume_rule_events_v4.csv")
    print("\n주의: 거래비용, 세금, 슬리피지는 미반영입니다.")
    print("같은 종목의 반복 신호가 포함될 수 있습니다.")


if __name__ == "__main__":
    main()
