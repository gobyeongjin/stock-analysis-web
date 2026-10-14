
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parents[1]
RESULT_DIR = BASE_DIR / "results"

TRADES_FILE = RESULT_DIR / "matched_stop_loss_trades.csv"
PRICE_FILE = BASE_DIR / "stock_data_500.csv"

# 기존 결과 파일을 덮어쓰지 않도록 실행 시각을 붙임
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
DETAIL_FILE = RESULT_DIR / f"stop_loss_failure_details_{RUN_ID}.csv"
SUMMARY_FILE = RESULT_DIR / f"stop_loss_failure_summary_{RUN_ID}.csv"


def normalize_ticker(value):
    """종목코드를 문자열 6자리 형태로 정리."""
    if pd.isna(value):
        return ""
    value = str(value).strip()
    if value.endswith(".0"):
        value = value[:-2]
    return value.zfill(6)


def require_columns(df, columns, file_name):
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"{file_name}에 필요한 컬럼이 없습니다: {missing}\n"
            f"현재 컬럼: {df.columns.tolist()}"
        )


def main():
    if not TRADES_FILE.exists():
        raise FileNotFoundError(f"거래 결과 파일을 찾을 수 없습니다: {TRADES_FILE}")
    if not PRICE_FILE.exists():
        raise FileNotFoundError(f"주가 데이터 파일을 찾을 수 없습니다: {PRICE_FILE}")

    trades = pd.read_csv(TRADES_FILE)
    prices = pd.read_csv(PRICE_FILE, low_memory=False)

    require_columns(
        trades,
        [
            "Strategy", "Ticker", "Entry_Date", "Entry_Open",
            "Stop_Price", "Actual_Exit_Date", "Exit_Open",
            "Exit_Reason", "Net_Return_Pct"
        ],
        TRADES_FILE.name,
    )
    require_columns(
        prices,
        ["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"],
        PRICE_FILE.name,
    )

    trades["Ticker"] = trades["Ticker"].map(normalize_ticker)
    prices["Ticker"] = prices["Ticker"].map(normalize_ticker)

    for col in ["Entry_Date", "Actual_Exit_Date"]:
        trades[col] = pd.to_datetime(trades[col], errors="coerce")

    prices["Date"] = pd.to_datetime(prices["Date"], errors="coerce")

    for col in ["Entry_Open", "Stop_Price", "Exit_Open", "Net_Return_Pct"]:
        trades[col] = pd.to_numeric(trades[col], errors="coerce")

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        prices[col] = pd.to_numeric(prices[col], errors="coerce")

    prices = prices.dropna(subset=["Date", "Ticker"]).copy()
    prices = prices.sort_values(["Ticker", "Date"])
    prices["Prev_Close"] = prices.groupby("Ticker")["Close"].shift(1)

    # 하한가 추정: 전일 종가 대비 약 30% 하락 + OHLC가 거의 동일.
    # 이는 근사치이며 실제 거래소의 가격제한 규정/종목별 상황과 다를 수 있음.
    ohlc_min = prices[["Open", "High", "Low", "Close"]].min(axis=1)
    ohlc_max = prices[["Open", "High", "Low", "Close"]].max(axis=1)
    prices["Lower_Limit_Like"] = (
        (prices["Prev_Close"] > 0)
        & (prices["Close"] / prices["Prev_Close"] - 1 <= -0.295)
        & ((ohlc_max - ohlc_min) / prices["Prev_Close"] <= 0.003)
    )
    prices["Zero_Volume"] = prices["Volume"].fillna(0).le(0)

    stop_strategies = ["Stop_5pct", "Stop_8pct", "Stop_10pct"]
    stopped = trades[trades["Strategy"].isin(stop_strategies)].copy()

    # 두 종류를 확인:
    # 1) 실제 순손실이 -10%보다 큰 거래
    # 2) 각 손절 기준보다 손실이 더 커진 거래
    stop_level = {
        "Stop_5pct": -5.0,
        "Stop_8pct": -8.0,
        "Stop_10pct": -10.0,
    }
    stopped["Nominal_Stop_Pct"] = stopped["Strategy"].map(stop_level)

    stopped["Worse_Than_Stop"] = (
        stopped["Net_Return_Pct"]
        < stopped["Nominal_Stop_Pct"] - 0.25
    )
    stopped["Loss_Below_Minus10"] = stopped["Net_Return_Pct"] < -10.0

    failures = stopped[
        stopped["Worse_Than_Stop"] | stopped["Loss_Below_Minus10"]
    ].copy()

    if failures.empty:
        print("조건에 해당하는 거래가 없습니다.")
        return

    detail_rows = []

    for _, trade in failures.iterrows():
        start = trade["Entry_Date"]
        end = trade["Actual_Exit_Date"]
        ticker = trade["Ticker"]

        if pd.isna(start) or pd.isna(end):
            continue

        period = prices[
            (prices["Ticker"] == ticker)
            & (prices["Date"] >= start)
            & (prices["Date"] <= end)
        ].copy()

        if period.empty:
            detail_rows.append({
                **trade.to_dict(),
                "Raw_Bars_Found": 0,
                "Diagnosis": "주가 데이터 구간 없음",
            })
            continue

        entry_price = trade["Entry_Open"]
        stop_price = trade["Stop_Price"]

        if pd.notna(entry_price) and entry_price > 0:
            period["Low_Return_Pct"] = (
                period["Low"] / entry_price - 1
            ) * 100
            period["Close_Return_Pct"] = (
                period["Close"] / entry_price - 1
            ) * 100
        else:
            period["Low_Return_Pct"] = np.nan
            period["Close_Return_Pct"] = np.nan

        stop_touch = period[
            (period["Low"] <= stop_price)
            & period["Low"].notna()
        ] if pd.notna(stop_price) else period.iloc[0:0]

        lower_days = period.loc[
            period["Lower_Limit_Like"], "Date"
        ].dt.strftime("%Y-%m-%d").tolist()

        zero_volume_days = period.loc[
            period["Zero_Volume"], "Date"
        ].dt.strftime("%Y-%m-%d").tolist()

        # 청산일 시가가 손절가보다 얼마나 낮았는지 계산
        if (
            pd.notna(trade["Exit_Open"])
            and pd.notna(stop_price)
            and stop_price > 0
        ):
            exit_vs_stop_pct = (
                trade["Exit_Open"] / stop_price - 1
            ) * 100
        else:
            exit_vs_stop_pct = np.nan

        if trade["Exit_Reason"] == "Gap_Stop":
            diagnosis = "갭 하락으로 손절가보다 낮은 시가에 청산"
        elif trade["Exit_Reason"] == "Delayed_Stop_After_Unfillable_Breach":
            diagnosis = "손절 조건 발생 후 즉시 청산하지 못함"
        elif lower_days:
            diagnosis = "하한가 추정 구간 포함: 체결 가능성 확인 필요"
        elif zero_volume_days:
            diagnosis = "거래량 0인 날짜 포함: 체결 가능성 확인 필요"
        elif not stop_touch.empty:
            diagnosis = "장중 손절 조건 발생: 청산 로직과 가격 확인 필요"
        else:
            diagnosis = "손절 조건 및 원시 주가 데이터 추가 확인 필요"

        detail_rows.append({
            **trade.to_dict(),
            "Raw_Bars_Found": len(period),
            "Min_Low_Date": (
                period.loc[period["Low"].idxmin(), "Date"].strftime("%Y-%m-%d")
                if period["Low"].notna().any() else ""
            ),
            "Min_Low": period["Low"].min(),
            "Min_Low_Return_Pct": period["Low_Return_Pct"].min(),
            "Min_Close": period["Close"].min(),
            "Min_Close_Return_Pct": period["Close_Return_Pct"].min(),
            "Stop_Touch_Dates": ";".join(
                stop_touch["Date"].dt.strftime("%Y-%m-%d").tolist()
            ),
            "Lower_Limit_Like_Dates": ";".join(lower_days),
            "Zero_Volume_Dates": ";".join(zero_volume_days),
            "Exit_Open_vs_Stop_Pct": exit_vs_stop_pct,
            "Diagnosis": diagnosis,
        })

    details = pd.DataFrame(detail_rows)

    if details.empty:
        print("분석할 수 있는 거래 상세 내역이 없습니다.")
        return

    details = details.sort_values(
        "Net_Return_Pct", ascending=True
    )
    details.to_csv(DETAIL_FILE, index=False, encoding="utf-8-sig")

    summary = (
        details.groupby(["Strategy", "Exit_Reason"], dropna=False)
        .agg(
            Trade_Count=("Ticker", "size"),
            Avg_Net_Return_Pct=("Net_Return_Pct", "mean"),
            Worst_Net_Return_Pct=("Net_Return_Pct", "min"),
            Lower_Limit_Cases=(
                "Lower_Limit_Like_Dates",
                lambda s: s.ne("").sum()
            ),
            Zero_Volume_Cases=(
                "Zero_Volume_Dates",
                lambda s: s.ne("").sum()
            ),
        )
        .reset_index()
        .sort_values(["Strategy", "Worst_Net_Return_Pct"])
    )
    summary.to_csv(SUMMARY_FILE, index=False, encoding="utf-8-sig")

    print("\n=== 손절 실패 거래 점검 완료 ===")
    print(f"점검 거래 수: {len(details)}")
    print(f"상세 결과: {DETAIL_FILE}")
    print(f"요약 결과: {SUMMARY_FILE}")

    print("\n=== 전략별·청산 사유별 요약 ===")
    print(summary.to_string(index=False))

    print("\n=== 손실이 큰 거래 상위 15건 ===")
    display_cols = [
        c for c in [
            "Strategy", "Stock_Name", "Ticker", "Entry_Date",
            "Entry_Open", "Stop_Price", "Actual_Exit_Date",
            "Exit_Open", "Exit_Reason", "Net_Return_Pct",
            "Min_Low_Return_Pct", "Lower_Limit_Like_Dates",
            "Diagnosis"
        ] if c in details.columns
    ]
    print(details[display_cols].head(15).to_string(index=False))


if __name__ == "__main__":
    main()
