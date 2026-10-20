
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    average_precision_score,
)

# 현재 analysis 폴더 기준
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "training_data_v3.csv"
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "random_forest_transfer_005930.pkl"
RESULT_PATH = BASE_DIR / "transfer_samsung_results.csv"

SAMSUNG = "005930.KS"
TEST_RATIO = 0.20
THRESHOLD = 0.50

FEATURES = [
    "MA20_Gap", "MA60_Gap", "MA20_MA60_Gap",
    "MA20_Change", "MA60_Change", "High_20D_Gap",
    "RSI", "RSI_Change", "Volume_Ratio", "Volume_Ratio_5D",
    "Return_1D", "Return_3D", "Return_5D", "Return_10D",
]


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"학습 데이터 파일을 찾을 수 없습니다: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)
    required = FEATURES + [
        "Date", "Ticker", "Label", "Future_5D_Valid"
    ]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"데이터에 필요한 컬럼이 없습니다: {missing}")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Ticker"] = df["Ticker"].astype(str).str.strip()

    # 학습 대상과 테스트 대상을 분리
    samsung = df[df["Ticker"] == SAMSUNG].copy()
    other = df[df["Ticker"] != SAMSUNG].copy()

    if samsung.empty:
        raise ValueError(
            f"{SAMSUNG} 데이터가 없습니다. Ticker 값을 확인하세요."
        )

    # 마지막 미래 수익률이 계산되지 않는 행 제외
    valid = df["Future_5D_Valid"].astype(str).str.lower().isin(
        ["true", "1"]
    )
    samsung = samsung.loc[valid.loc[samsung.index]].copy()
    other = other.loc[valid.loc[other.index]].copy()

    # 입력값과 라벨 정리
    for data in (samsung, other):
        data["Label"] = pd.to_numeric(
            data["Label"], errors="coerce"
        )
        data[FEATURES] = data[FEATURES].apply(
            pd.to_numeric, errors="coerce"
        )
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        data.dropna(subset=FEATURES + ["Label", "Date"], inplace=True)
        data["Label"] = data["Label"].astype(int)

    samsung = samsung.sort_values("Date").reset_index(drop=True)

    # 삼성전자 날짜 기준으로 최신 20%를 테스트 기간으로 지정
    unique_dates = np.sort(samsung["Date"].unique())
    split_idx = int(len(unique_dates) * (1 - TEST_RATIO))

    if split_idx <= 0 or split_idx >= len(unique_dates):
        raise ValueError("삼성전자 데이터가 시간순 분할에 부족합니다.")

    test_start = pd.Timestamp(unique_dates[split_idx])

    # 5거래일 미래 라벨의 기간 중첩을 줄이기 위해
    # 테스트 시작일보다 최소 7일 이전 데이터만 학습에 사용
    train_cutoff = test_start - pd.Timedelta(days=7)

    train = other[other["Date"] < train_cutoff].copy()
    test = samsung[samsung["Date"] >= test_start].copy()

    if train.empty or test.empty:
        raise ValueError("학습 또는 테스트 데이터가 비어 있습니다.")

    if train["Label"].nunique() < 2:
        raise ValueError("학습 데이터에 Label 0과 1이 모두 필요합니다.")

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(train[FEATURES], train["Label"])

    probabilities = model.predict_proba(test[FEATURES])[:, 1]
    predictions = (probabilities >= THRESHOLD).astype(int)
    actual = test["Label"].to_numpy()

    print("\n===== 다른 종목 학습 → 삼성전자 테스트 =====")
    print("학습 종목 수:", train["Ticker"].nunique())
    print("학습 행 수:", len(train))
    print("학습 데이터 마지막 날짜:", train["Date"].max().date())
    print("삼성전자 테스트 행 수:", len(test))
    print("테스트 기간:", test["Date"].min().date(), "~",
          test["Date"].max().date())
    print("예측 기준 확률:", THRESHOLD)
    print("Accuracy:", round(accuracy_score(actual, predictions), 4))
    print("Precision:", round(
        precision_score(actual, predictions, zero_division=0), 4
    ))
    print("Recall:", round(
        recall_score(actual, predictions, zero_division=0), 4
    ))
    print("F1:", round(
        f1_score(actual, predictions, zero_division=0), 4
    ))

    if len(np.unique(actual)) == 2:
        print("Average precision:", round(
            average_precision_score(actual, probabilities), 4
        ))

    print("Confusion matrix:")
    print(confusion_matrix(actual, predictions, labels=[0, 1]))

    results = test[
        ["Date", "Ticker", "Label"]
    ].copy()
    results["Predicted_Probability"] = probabilities
    results["Predicted_Label"] = predictions
    results["Threshold"] = THRESHOLD

    results.to_csv(RESULT_PATH, index=False, encoding="utf-8-sig")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "features": FEATURES,
            "threshold": THRESHOLD,
            "training_tickers": sorted(train["Ticker"].unique()),
            "test_start": str(test["Date"].min().date()),
        },
        MODEL_PATH,
    )

    print("\n모델 저장:", MODEL_PATH.name)
    print("테스트 결과 저장:", RESULT_PATH.name)
    print("기존 random_forest.pkl은 변경하지 않았습니다.")


if __name__ == "__main__":
    main()
