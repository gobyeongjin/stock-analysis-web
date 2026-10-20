from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.exceptions import InconsistentVersionWarning

BASE = Path(__file__).resolve().parent
DATA_PATH = BASE / "training_data_v3.csv"
MODEL_PATH = BASE / "model" / "random_forest_v3.pkl"

# 기존 학습 코드와 동일한 날짜 분할
TRAIN_RATIO = 0.8

# 모델의 학습 데이터와 평가 데이터가
# 겹치지 않는지 먼저 검증하는 모드
EMBARGO_DAYS = 5


def precision_at_n(df, score_col, n):
    values = []

    for _, group in df.groupby("Date"):
        top = group.nlargest(n, score_col)

        if len(top) < n:
            continue

        values.append(top["Label"].mean())

    return np.mean(values) if values else np.nan


def main():
    data = pd.read_csv(
        DATA_PATH,
        dtype={"Ticker": str}
    )

    data["Date"] = pd.to_datetime(data["Date"])

    data = data[
        data["Future_5D_Valid"] == True
    ].copy()

    data = data.replace(
        [np.inf, -np.inf], np.nan
    )

    with warnings.catch_warnings():
        warnings.simplefilter(
            "error",
            InconsistentVersionWarning
        )
        model = joblib.load(MODEL_PATH)

    features = list(model.feature_names_in_)

    data = data.dropna(
        subset=features + ["Label"]
    )

    data = data.sort_values(
        ["Date", "Ticker"]
    ).reset_index(drop=True)

    dates = np.sort(data["Date"].unique())
    split_idx = int(len(dates) * TRAIN_RATIO)
    split_date = dates[split_idx]

    test = data[
        data["Date"] >= split_date
    ].copy()

    # 기존 저장 모델은 공백 없이 학습됐으므로
    # 평가 시작일을 뒤로 미루는 것만으로
    # 학습 시 발생한 누출을 제거할 수는 없음.
    print("주의: 기존 모델은 embargo 없이 학습됨")
    print("이 결과는 잠정 평가이며 누출 제거 평가가 아님")

    # 마지막 학습일과 평가 시작일 확인
    train_dates = dates[:split_idx]
    print("학습 마지막 날짜:", pd.Timestamp(train_dates[-1]).date())
    print("평가 시작 날짜:", pd.Timestamp(split_date).date())
    print("권장 경계 공백:", EMBARGO_DAYS, "거래일")

    positive_idx = list(model.classes_).index(1)

    test["AI_Score"] = model.predict_proba(
        test[features]
    )[:, positive_idx]

    test["Label"] = test["Label"].astype(int)

    print("\n" + "=" * 55)
    print("Random Forest V3 평가")
    print("=" * 55)

    print("평가 샘플:", len(test))
    print("평가 날짜 수:", test["Date"].nunique())
    print("Label 1 비율:", f"{test['Label'].mean():.2%}")

    ap = average_precision_score(
        test["Label"],
        test["AI_Score"]
    )

    print("PR-AUC (Average Precision):", f"{ap:.4f}")

    print("\n[일별 상위 N개 적중률]")

    for n in [5, 10]:
        ai = precision_at_n(test, "AI_Score", n)

        # 날짜별 전체 양성 비율:
        # 같은 날 무작위로 N종목을 뽑았을 때의 기대 적중률
        daily_base = (
            test.groupby("Date")
            .filter(lambda g: len(g) >= n)
            .groupby("Date")["Label"]
            .mean()
            .mean()
        )

        print(f"\n상위 {n}개")
        print(f"AI 적중률: {ai:.2%}")
        print(f"무작위 기준: {daily_base:.2%}")
        print(f"차이: {(ai - daily_base) * 100:+.2f}%p")

        if "Volume_Ratio" in test.columns:
            volume = precision_at_n(
                test, "Volume_Ratio", n
            )
            print(f"거래량 상위 기준: {volume:.2%}")

    print("\n[AI 점수 구간별 실제 적중률]")

    bins = [0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.000001]

    test["Score_Bin"] = pd.cut(
        test["AI_Score"],
        bins=bins,
        right=False
    )

    calibration = (
        test.groupby(
            "Score_Bin",
            observed=True
        )
        .agg(
            Samples=("Label", "size"),
            Avg_Score=("AI_Score", "mean"),
            Actual_Hit_Rate=("Label", "mean")
        )
    )

    print(
        calibration.to_string(
            formatters={
                "Avg_Score": "{:.2%}".format,
                "Actual_Hit_Rate": "{:.2%}".format
            }
        )
    )

    print("\n※ 실제 매매 수익률 평가가 아닙니다.")
    print("※ 학습 데이터 경계 누출 가능성이 남아 있습니다.")
    print("※ 모델 파일은 변경하지 않았습니다.")


if __name__ == "__main__":
    main()
