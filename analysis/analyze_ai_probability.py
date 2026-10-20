
import pandas as pd
import numpy as np
from pathlib import Path

import backtest_stop8_test as base

BASE_DIR = Path(__file__).resolve().parent

BINS = [0.65, 0.70, 0.75, 0.80, 1.000001]
LABELS = [
    "0.65~0.70",
    "0.70~0.75",
    "0.75~0.80",
    "0.80~1.00",
]


def find_probability_column(df):
    candidates = [
        "AI_Probability",
        "Probability",
        "Pred_Probability",
        "Pred_Prob",
        "Score",
        "AI_Score",
    ]

    for col in candidates:
        if col in df.columns:
            return col

    raise ValueError(
        "확률 컬럼을 찾지 못했습니다.\n"
        f"현재 컬럼: {df.columns.tolist()}"
    )


def main():
    prices, train, test = base.prepare_data()

    signals, metrics = base.make_signals(
        train, test, prices
    )

    df = signals["A_PRICE_ONLY"].copy()
    # 평가 데이터의 실제 결과 연결
    test = test.copy()

    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()
    test["Date"] = pd.to_datetime(test["Date"]).dt.normalize()

    df["Ticker"] = df["Ticker"].astype(str)
    test["Ticker"] = test["Ticker"].astype(str)

    # CSV에서 종목코드 앞자리 0이 사라지는 문제 방지
    df["Ticker"] = df["Ticker"].str.split(".").str[0].str.zfill(6)
    test["Ticker"] = test["Ticker"].str.split(".").str[0].str.zfill(6)

    target_cols = [
        "Label",
        "Future_5D_Close_Return",
        "Future_5D_Return",
    ]

    available_cols = [
        col for col in target_cols
        if col in test.columns
    ]

    print("\n평가 데이터에서 찾은 실제 결과 컬럼:", available_cols)

    if available_cols:
        actual = test[
            ["Date", "Ticker"] + available_cols
        ].drop_duplicates(["Date", "Ticker"])

        df = df.merge(
            actual,
            on=["Date", "Ticker"],
            how="left",
            validate="many_to_one",
        )

        print("실제 결과 연결 완료")
        print("전체 신호:", len(df))

        if "Label" in df.columns:
            print("정답 연결 수:", df["Label"].notna().sum())
    else:
        print("평가 데이터에 실제 결과 컬럼이 없습니다.")

    

    print("\n===== 신호 데이터 컬럼 =====")
    print(df.columns.tolist())

    prob_col = find_probability_column(df)

    print(f"\n사용할 확률 컬럼: {prob_col}")

    df["AI_Prob"] = pd.to_numeric(
        df[prob_col], errors="coerce"
    )

    df = df.dropna(subset=["AI_Prob"])

    # 확률이 0~100 단위라면 0~1로 변환
    if df["AI_Prob"].max() > 1:
        df["AI_Prob"] = df["AI_Prob"] / 100

    df = df[
        (df["AI_Prob"] >= 0.65)
        & (df["AI_Prob"] <= 1)
    ].copy()

    df["Probability_Group"] = pd.cut(
        df["AI_Prob"],
        bins=BINS,
        labels=LABELS,
        right=False,
    )

    print("\n===== 확률 구간별 신호 수 =====")

    counts = (
        df.groupby(
            "Probability_Group",
            observed=False
        )
        .size()
        .reset_index(name="Signal_Count")
    )

    print(counts.to_string(index=False))

    # 실제 정답(Label)이 있는 경우
    if "Label" in df.columns:
        df["Label"] = pd.to_numeric(
            df["Label"], errors="coerce"
        )

        label_summary = (
            df.groupby(
                "Probability_Group",
                observed=False
            )
            .agg(
                Signal_Count=("AI_Prob", "size"),
                Avg_AI_Prob=("AI_Prob", "mean"),
                Actual_Hit_Rate=("Label", "mean"),
            )
            .reset_index()
        )

        label_summary["Avg_AI_Prob"] *= 100
        label_summary["Actual_Hit_Rate"] *= 100

        print("\n===== 확률 구간별 실제 상승 적중률 =====")
        print(label_summary.round(2).to_string(index=False))

        label_summary.to_csv(
            BASE_DIR / "ai_probability_label_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )
    else:
        print(
            "\nLabel 컬럼이 없어 실제 상승 적중률은 "
            "계산하지 않았습니다."
        )

    # 미래 수익률 컬럼이 있는 경우
    return_candidates = [
        "Future_5D_Close_Return",
        "Future_5D_Return",
    ]

    available_returns = [
        col for col in return_candidates
        if col in df.columns
    ]

    for col in available_returns:
        df[col] = pd.to_numeric(
            df[col], errors="coerce"
        )

        summary = (
            df.groupby(
                "Probability_Group",
                observed=False
            )
            .agg(
                Signal_Count=("AI_Prob", "size"),
                Avg_Return=(col, "mean"),
                Median_Return=(col, "median"),
                Positive_Rate=(
                    col,
                    lambda x: (x.dropna() > 0).mean()
                    if x.notna().any()
                    else np.nan
                ),
            )
            .reset_index()
        )

        summary["Positive_Rate"] *= 100

        print(f"\n===== {col} 분석 =====")
        print(summary.round(2).to_string(index=False))

        summary.to_csv(
            BASE_DIR / f"ai_probability_{col}.csv",
            index=False,
            encoding="utf-8-sig",
        )

    df.to_csv(
        BASE_DIR / "ai_probability_detail.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\n분석 완료")


if __name__ == "__main__":
    main()
