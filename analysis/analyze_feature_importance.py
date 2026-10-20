
from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, f1_score

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "training_data_v3.csv"
OUTPUT_PATH = BASE_DIR / "feature_importance.csv"

FEATURES = [
    "MA20_Gap",
    "MA60_Gap",
    "MA20_MA60_Gap",
    "MA20_Change",
    "MA60_Change",
    "High_20D_Gap",
    "Low_20D_Gap",
    "RSI",
    "RSI_Change",
    "Volume_Ratio",
    "Volume_Ratio_5D",
    "Return_1D",
    "Return_3D",
    "Return_5D",
    "Return_10D",
    "Volatility_20D",
    "Volatility_Change",
    "Close_Position",
]

def main():
    df = pd.read_csv(DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    required = FEATURES + ["Date", "Ticker", "Label", "Future_5D_Valid"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"누락된 컬럼: {missing}")

    df["Future_5D_Valid"] = (
        df["Future_5D_Valid"].astype(str).str.lower()
        .isin(["true", "1"])
    )

    df["Label"] = pd.to_numeric(df["Label"], errors="coerce")
    df[FEATURES] = df[FEATURES].apply(pd.to_numeric, errors="coerce")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df[df["Future_5D_Valid"]].dropna(
        subset=FEATURES + ["Label", "Date"]
    )
    df["Label"] = df["Label"].astype(int)

    # 시간순 분할: 마지막 20% 날짜는 평가에만 사용
    dates = np.sort(df["Date"].unique())
    split_date = pd.Timestamp(dates[int(len(dates) * 0.8)])

    train = df[df["Date"] < split_date].copy()
    test = df[df["Date"] >= split_date].copy()

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
    predictions = (probabilities >= 0.5).astype(int)

    print("학습 행 수:", len(train))
    print("평가 행 수:", len(test))
    print("학습 마지막 날짜:", train["Date"].max().date())
    print("평가 시작 날짜:", test["Date"].min().date())
    print("평가 F1:", round(f1_score(
        test["Label"], predictions, zero_division=0
    ), 4))
    print("평가 Average Precision:", round(
        average_precision_score(test["Label"], probabilities), 4
    ))

    # Random Forest 내장 중요도
    impurity = pd.Series(
        model.feature_importances_, index=FEATURES
    )

    # 평가 구간에서 변수 값을 섞었을 때 성능이 얼마나 감소하는지 측정
    perm = permutation_importance(
        model,
        test[FEATURES],
        test["Label"],
        scoring="average_precision",
        n_repeats=5,
        random_state=42,
        n_jobs=-1,
    )

    results = pd.DataFrame({
        "Feature": FEATURES,
        "RF_Importance": impurity.values,
        "Permutation_AP_Mean": perm.importances_mean,
        "Permutation_AP_Std": perm.importances_std,
    }).sort_values("Permutation_AP_Mean", ascending=False)

    results.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print("\n===== 변수 중요도 =====")
    print(results.to_string(index=False))
    print("\n저장 완료:", OUTPUT_PATH.name)

if __name__ == "__main__":
    main()
