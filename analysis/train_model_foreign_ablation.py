
import os
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

DATA_PATH = os.path.join(BASE_DIR, "training_data_v3.csv")
FOREIGN_PATH = os.path.join(PROJECT_DIR, "foreign_trend_005930.csv")

BASE_FEATURES = [
    "MA20_Gap", "MA60_Gap", "MA20_MA60_Gap",
    "MA20_Change", "MA60_Change", "High_20D_Gap",
    "RSI", "RSI_Change", "Volume_Ratio", "Volume_Ratio_5D",
    "Return_1D", "Return_3D", "Return_5D", "Return_10D",
]

RATIO_FEATURE = "Foreign_Ratio_Change_5Obs"
HOLDING_FEATURE = "Foreign_Holding_Qty_Change_5Obs_Pct"


def make_model():
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )


def score_model(model, features, frame, threshold):
    y_true = frame["Label"].astype(int)
    probability = model.predict_proba(frame[features])[:, 1]
    prediction = (probability >= threshold).astype(int)

    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, prediction),
        "precision": precision_score(
            y_true, prediction, zero_division=0
        ),
        "recall": recall_score(
            y_true, prediction, zero_division=0
        ),
        "f1": f1_score(y_true, prediction, zero_division=0),
        "matrix": confusion_matrix(y_true, prediction),
    }


def main():
    data = pd.read_csv(DATA_PATH)
    foreign = pd.read_csv(FOREIGN_PATH, dtype={"date": str})

    data["Date"] = pd.to_datetime(data["Date"])
    data["DateKey"] = data["Date"].dt.strftime("%Y%m%d")

    data = data[data["Ticker"] == "005930.KS"].copy()
    data = data[
        data["Future_5D_Valid"].astype(str).str.lower() == "true"
    ].copy()

    foreign["date"] = (
        foreign["date"].astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.strip().str.zfill(8)
    )

    for col in ["holding_qty", "holding_ratio"]:
        foreign[col] = pd.to_numeric(
            foreign[col].astype(str).str.replace(",", "", regex=False),
            errors="coerce",
        )

    foreign = (
        foreign.drop_duplicates("date")
        .sort_values("date")
        .reset_index(drop=True)
    )

    # 5개 관측치 전과 비교 (정확히 5거래일 전이라는 뜻은 아님)
    foreign[RATIO_FEATURE] = (
        foreign["holding_ratio"] - foreign["holding_ratio"].shift(5)
    )

    previous_qty = foreign["holding_qty"].shift(5)
    foreign[HOLDING_FEATURE] = (
        (foreign["holding_qty"] / previous_qty - 1) * 100
    )

    foreign_features = foreign.rename(
        columns={"date": "DateKey"}
    )[["DateKey", RATIO_FEATURE, HOLDING_FEATURE]]

    data = data.merge(
        foreign_features,
        on="DateKey",
        how="inner",
        validate="one_to_one",
    )

    feature_sets = {
        "기본 모델": BASE_FEATURES,
        "보유 비율 추가": BASE_FEATURES + [RATIO_FEATURE],
        "보유 수량 변화율 추가": BASE_FEATURES + [HOLDING_FEATURE],
    }

    # 세 모델 모두 같은 행을 사용해 공정하게 비교
    required = BASE_FEATURES + [
        RATIO_FEATURE, HOLDING_FEATURE, "Label"
    ]
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=required)
    data = data.sort_values("Date").reset_index(drop=True)

    if len(data) < 100:
        raise ValueError(f"실험 데이터가 너무 적습니다: {len(data)}행")

    # 시간순 분리: 앞 60% 학습, 다음 20% 검증, 마지막 20% 테스트
    n = len(data)
    train_end = int(n * 0.60)
    valid_end = int(n * 0.80)

    train = data.iloc[:train_end].copy()
    valid = data.iloc[train_end:valid_end].copy()
    test = data.iloc[valid_end:].copy()

    print("===== 데이터 구간 =====")
    print(f"전체 행: {n}")
    print(f"학습: {train['Date'].min().date()} ~ {train['Date'].max().date()} ({len(train)}행)")
    print(f"검증: {valid['Date'].min().date()} ~ {valid['Date'].max().date()} ({len(valid)}행)")
    print(f"테스트: {test['Date'].min().date()} ~ {test['Date'].max().date()} ({len(test)}행)")

    # 검증 데이터에서 F1이 가장 높은 임계값 선택
    thresholds = np.arange(0.30, 0.71, 0.05)

    for name, features in feature_sets.items():
        model = make_model()
        model.fit(train[features], train["Label"].astype(int))

        valid_scores = [
            score_model(model, features, valid, float(t))
            for t in thresholds
        ]
        best = max(valid_scores, key=lambda x: x["f1"])
        threshold = best["threshold"]

        test_score = score_model(model, features, test, threshold)

        print(f"\n===== {name} ({len(features)}개 특성) =====")
        print(f"검증에서 선택한 임계값: {threshold:.2f}")
        print(f"테스트 Accuracy : {test_score['accuracy']:.4f}")
        print(f"테스트 Precision: {test_score['precision']:.4f}")
        print(f"테스트 Recall   : {test_score['recall']:.4f}")
        print(f"테스트 F1       : {test_score['f1']:.4f}")
        print("Confusion matrix:")
        print(test_score["matrix"])

    print("\n실험 완료: 기존 모델 파일은 변경하지 않았습니다.")


if __name__ == "__main__":
    main()
