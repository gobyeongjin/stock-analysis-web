import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score


print("================================")
print("V2 Ablation Test")
print("================================")


# ==========================================
# 1. 데이터 불러오기
# ==========================================

data = pd.read_csv("training_data_v2.csv")

data["Date"] = pd.to_datetime(data["Date"])

data = data.sort_values("Date").reset_index(drop=True)


# ==========================================
# 2. 전체 Feature
# ==========================================

base_features = [
    "MA20_Gap",
    "MA60_Gap",
    "MA20_MA60_Gap",
    "MA20_Change",
    "MA60_Change",
    "High_20D_Gap",
    "RSI",
    "RSI_Change",
    "Volume_Ratio",
    "Volume_Ratio_5D",
    "Return_1D",
    "Return_3D",
    "Return_5D",
    "Return_10D"
]


new_features = [
    "Golden_Trend",
    "Pullback_MA20",
    "Near_High",
    "Momentum_10D",
    "MA20_Bounce"
]


# ==========================================
# 3. 실험할 Feature 구성
# ==========================================

experiments = {
    "V1_Base": base_features,

    "V2_All": base_features + new_features,

    "V2_No_Golden_Trend": [
        f for f in base_features + new_features
        if f != "Golden_Trend"
    ],

    "V2_No_Pullback_MA20": [
        f for f in base_features + new_features
        if f != "Pullback_MA20"
    ],

    "V2_No_Near_High": [
        f for f in base_features + new_features
        if f != "Near_High"
    ],

    "V2_No_Momentum_10D": [
        f for f in base_features + new_features
        if f != "Momentum_10D"
    ],

    "V2_No_MA20_Bounce": [
        f for f in base_features + new_features
        if f != "MA20_Bounce"
    ]
}


# ==========================================
# 4. Train / Test 분리
# ==========================================

split_date = data["Date"].quantile(0.8)

train = data[
    data["Date"] < split_date
].copy()

test = data[
    data["Date"] >= split_date
].copy()


print()
print(f"Train : {len(train):,}")
print(f"Test  : {len(test):,}")


# ==========================================
# 5. 테스트할 Threshold
# ==========================================

thresholds = [
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80
]


# ==========================================
# 6. 실험
# ==========================================

all_results = []


for experiment_name, features in experiments.items():

    print()
    print("================================")
    print(f"실험 : {experiment_name}")
    print(f"Feature 수 : {len(features)}")
    print("================================")

    X_train = train[features]
    y_train = train["Label"]

    X_test = test[features]
    y_test = test["Label"]

    # Random Forest
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train
    )

    print("학습 완료!")

    # 예측 확률
    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    # Threshold별 평가
    for threshold in thresholds:

        selected = probabilities >= threshold

        selected_count = selected.sum()

        if selected_count == 0:
            continue

        y_pred = selected.astype(int)

        precision = precision_score(
            y_test,
            y_pred,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            y_pred,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            y_pred,
            zero_division=0
        )

        selected_data = test[selected].copy()

        avg_5d_return = (
            selected_data["Future_5D_Return"]
            .mean()
        )

        median_5d_return = (
            selected_data["Future_5D_Return"]
            .median()
        )

        avg_close_return = (
            selected_data["Future_5D_Close_Return"]
            .mean()
        )

        median_close_return = (
            selected_data["Future_5D_Close_Return"]
            .median()
        )

        positive_rate = (
            selected_data["Label"]
            .mean() * 100
        )

        selection_rate = (
            selected_count
            / len(test)
            * 100
        )

        all_results.append({
            "Experiment": experiment_name,
            "Feature_Count": len(features),
            "Threshold": threshold,
            "Selected": selected_count,
            "Selection_Rate": selection_rate,
            "Precision": precision * 100,
            "Recall": recall * 100,
            "F1": f1,
            "Avg_5D_Return": avg_5d_return,
            "Median_5D_Return": median_5d_return,
            "Avg_5D_Close_Return": avg_close_return,
            "Median_5D_Close_Return": median_close_return,
            "Positive_Rate": positive_rate
        })

        print()
        print(f"Threshold : {threshold:.2f}")
        print(
            f"선정 데이터 : "
            f"{selected_count:,}개"
        )
        print(
            f"Precision : "
            f"{precision * 100:.2f}%"
        )
        print(
            f"Recall : "
            f"{recall * 100:.2f}%"
        )
        print(
            f"F1 : "
            f"{f1:.4f}"
        )
        print(
            f"평균 5일 수익률 : "
            f"{avg_5d_return:.2f}%"
        )
        print(
            f"평균 5일 후 종가 수익률 : "
            f"{avg_close_return:.2f}%"
        )


# ==========================================
# 7. 결과 저장
# ==========================================

results = pd.DataFrame(
    all_results
)

results.to_csv(
    "backtest_ablation_results.csv",
    index=False,
    encoding="utf-8-sig"
)


# ==========================================
# 8. 핵심 결과 출력
# ==========================================

print()
print("================================")
print("Ablation Test 완료")
print("================================")

print()
print("전체 결과")
print("--------------------------------")

print(
    results.to_string(
        index=False
    )
)


# ==========================================
# 9. Threshold별 비교
# ==========================================

print()
print("================================")
print("Threshold 0.70 비교")
print("================================")

comparison_70 = results[
    results["Threshold"] == 0.70
].copy()

print(
    comparison_70[
        [
            "Experiment",
            "Feature_Count",
            "Selected",
            "Selection_Rate",
            "Precision",
            "Recall",
            "F1",
            "Avg_5D_Return",
            "Avg_5D_Close_Return"
        ]
    ].to_string(index=False)
)


print()
print("================================")
print("Threshold 0.80 비교")
print("================================")

comparison_80 = results[
    results["Threshold"] == 0.80
].copy()

print(
    comparison_80[
        [
            "Experiment",
            "Feature_Count",
            "Selected",
            "Selection_Rate",
            "Precision",
            "Recall",
            "F1",
            "Avg_5D_Return",
            "Avg_5D_Close_Return"
        ]
    ].to_string(index=False)
)


print()
print(
    "결과 파일 : "
    "backtest_ablation_results.csv"
)