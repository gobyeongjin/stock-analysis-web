
from pathlib import Path
import pandas as pd

from backtest_atr_comparison_v4 import (
    load_prices,
    load_signals,
    run_backtest_v4,
    summarize_results,
    normalize_ticker,
)

BASE_DIR = Path(__file__).resolve().parent
INITIAL_CASH = 1_000_000


def run_comparison(prices, signals, label):
    experiments = {
        "BASELINE": signals.copy(),
        "EXCLUDE_4_TO_6": signals[
            ~((signals["ATR_Pct"] >= 4) & (signals["ATR_Pct"] < 6))
        ].copy(),
    }

    summaries = []

    for name, candidate_signals in experiments.items():
        trades, equity = run_backtest_v4(
            prices,
            candidate_signals,
            "SWING_ATR",
        )

        summary = summarize_results(trades, equity, name)
        summary["Period"] = label
        summary["Candidate_Signals"] = len(candidate_signals)
        summaries.append(summary)

        trades.to_csv(
            BASE_DIR / f"{label.lower()}_{name.lower()}_trades.csv",
            index=False,
            encoding="utf-8-sig",
        )
        equity.to_csv(
            BASE_DIR / f"{label.lower()}_{name.lower()}_equity.csv",
            index=False,
            encoding="utf-8-sig",
        )

    return summaries


def main():
    prices = load_prices()
    prices["Date"] = pd.to_datetime(prices["Date"])
    prices["Ticker"] = prices["Ticker"].map(normalize_ticker)

    signals, boundary, ap = load_signals(prices)
    signals["Date"] = pd.to_datetime(signals["Date"])
    signals["Ticker"] = signals["Ticker"].map(normalize_ticker)

    close_data = prices[["Date", "Ticker", "Close"]]
    signals = signals.merge(
        close_data,
        on=["Date", "Ticker"],
        how="left",
        validate="one_to_one",
    )
    signals["ATR_Pct"] = signals["ATR14"] / signals["Close"] * 100
    signals = signals.dropna(subset=["ATR_Pct"]).copy()

    test_dates = sorted(signals["Date"].unique())
    if len(test_dates) < 10:
        raise ValueError("최종 검증에 사용할 날짜가 부족합니다.")

    split_index = int(len(test_dates) * 0.6)
    selection_end = pd.Timestamp(test_dates[split_index - 1])
    validation_start = pd.Timestamp(test_dates[split_index])

    print("기존 평가 시작일:", boundary.date())
    print("필터 선택 구간 종료일:", selection_end.date())
    print("최종 검증 시작일:", validation_start.date())
    print("Average Precision:", round(ap, 4))

    # 각 구간의 가격과 신호를 함께 제한
    periods = {
        "SELECTION": (
    prices[
        (prices["Date"] >= boundary)
        & (prices["Date"] <= selection_end)
    ].copy(),
    signals[
        (signals["Date"] >= boundary)
        & (signals["Date"] <= selection_end)
    ].copy(),
),
        "VALIDATION": (
            prices[prices["Date"] >= validation_start].copy(),
            signals[signals["Date"] >= validation_start].copy(),
        ),
    }

    all_summaries = []
    for label, (period_prices, period_signals) in periods.items():
        print(f"\n===== {label} =====")
        print("가격 행 수:", len(period_prices))
        print("신호 수:", len(period_signals))

        all_summaries.extend(
            run_comparison(period_prices, period_signals, label)
        )

    result = pd.DataFrame(all_summaries)
    result.to_csv(
        BASE_DIR / "v4_atr_final_validation_results.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print("\n===== 결과 요약 =====")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
