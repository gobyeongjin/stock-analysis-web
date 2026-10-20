
import pandas as pd

import backtest_market_ma as bt
import backtest_stop8_test as base


# ==========================================
# 실험 설정
# ==========================================

INITIAL_CAPITAL = 10_000_000

MARKET_MA_DAYS = 40

THRESHOLDS = [
    0.65,
    0.70,
    0.75,
    0.80,
]


def main():

    # ======================================
    # 1. 데이터 준비 및 모델 학습
    # ======================================

    prices, train, test = base.prepare_data()

    signals, metrics = base.make_signals(
        train, test, prices
    )

    # 기존 0.65 기준으로 생성된 AI 신호
    candidate = signals["A_PRICE_ONLY"].copy()

    candidate["Probability"] = pd.to_numeric(
        candidate["Probability"],
        errors="coerce"
    )

    candidate = candidate.dropna(
        subset=["Probability"]
    )

    print("\n===== AI 확률 기준별 백테스트 =====")

    results = []

    # 기존 설정 보관
    original_cash = bt.INITIAL_CASH
    original_filter = bt.USE_MARKET_FILTER
    original_ma = bt.MARKET_MA_DAYS

    try:

        # 모든 실험에서 동일한 조건 적용
        bt.INITIAL_CASH = INITIAL_CAPITAL
        bt.USE_MARKET_FILTER = True
        bt.MARKET_MA_DAYS = MARKET_MA_DAYS

        for threshold in THRESHOLDS:

            # ==================================
            # 2. AI 확률 기준 적용
            # ==================================

            selected = candidate[
                candidate["Probability"] >= threshold
            ].copy()

            print(
                f"\n확률 기준: {threshold:.2f}"
                f" | 신호 수: {len(selected)}"
            )

            # ==================================
            # 3. 기존 백테스트 실행
            # ==================================

            trades, equity = bt.run_backtest_v4(
                prices,
                selected,
                "SWING_ATR"
            )

            summary = bt.summarize_results(
                trades,
                equity,
                "A_PRICE_ONLY"
            )

            summary["Threshold"] = threshold
            summary["Market_MA_Days"] = MARKET_MA_DAYS
            summary["Signal_Count"] = len(selected)

            results.append(summary)

            # ==================================
            # 4. 결과 파일 저장
            # ==================================

            prefix = (
                f"prob_threshold_{int(threshold * 100)}"
            )

            trades.to_csv(
                bt.BASE_DIR / f"{prefix}_trades.csv",
                index=False,
                encoding="utf-8-sig"
            )

            equity.to_csv(
                bt.BASE_DIR / f"{prefix}_equity.csv",
                index=False,
                encoding="utf-8-sig"
            )

    finally:

        # 기존 설정 복원
        bt.INITIAL_CASH = original_cash
        bt.USE_MARKET_FILTER = original_filter
        bt.MARKET_MA_DAYS = original_ma

    # ======================================
    # 5. 결과 비교
    # ======================================

    result = pd.DataFrame(results)

    columns = [
        "Threshold",
        "Signal_Count",
        "Return_Pct",
        "MDD_Pct",
        "Trades",
        "Win_Rate_Pct",
        "Final_Equity",
    ]

    print("\n===== 확률 기준별 최종 결과 =====")

    print(
        result[columns].to_string(index=False)
    )

    result.to_csv(
        bt.BASE_DIR / "probability_threshold_comparison.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("\n백테스트 완료")


if __name__ == "__main__":
    main()
