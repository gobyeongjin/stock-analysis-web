
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


def main():
    prices = load_prices()
    prices["Date"] = pd.to_datetime(prices["Date"])
    prices["Ticker"] = prices["Ticker"].map(normalize_ticker)

    signals, boundary, ap = load_signals(prices)
    signals["Date"] = pd.to_datetime(signals["Date"])
    signals["Ticker"] = signals["Ticker"].map(normalize_ticker)

    # 신호 발생 시점에 알 수 있는 종가로 ATR 비율 계산
    close_data = prices[["Date", "Ticker", "Close"]].copy()

    signals = signals.merge(
        close_data,
        on=["Date", "Ticker"],
        how="left",
        validate="one_to_one",
    )

    signals["ATR_Pct"] = signals["ATR14"] / signals["Close"] * 100
    signals = signals.dropna(subset=["ATR_Pct"]).copy()

    experiments = {
        "BASELINE": signals.copy(),
        "EXCLUDE_4_TO_6": signals[
            ~((signals["ATR_Pct"] >= 4) & (signals["ATR_Pct"] < 6))
        ].copy(),
        "EXCLUDE_6_PLUS": signals[
            signals["ATR_Pct"] < 6
        ].copy(),
    }

    print("평가 시작일:", boundary.date())
    print("Average Precision:", round(ap, 4))
    print("ATR 비율은 신호일 ATR14 / 신호일 종가로 계산")

    summaries = []

    for name, candidate_signals in experiments.items():
        print(f"\n===== {name} =====")
        print("후보 신호 수:", len(candidate_signals))

        trades, equity = run_backtest_v4(
            prices,
            candidate_signals,
            "SWING_ATR",
        )

        summary = summarize_results(trades, equity, name)
        summaries.append(summary)

        trades.to_csv(
            BASE_DIR / f"{name.lower()}_trades.csv",
            index=False,
            encoding="utf-8-sig",
        )
        equity.to_csv(
            BASE_DIR / f"{name.lower()}_equity.csv",
            index=False,
            encoding="utf-8-sig",
        )

        print(pd.Series(summary).to_string())

    result = pd.DataFrame(summaries)
    result.to_csv(
        BASE_DIR / "v4_atr_filter_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\n===== 최종 비교 =====")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
