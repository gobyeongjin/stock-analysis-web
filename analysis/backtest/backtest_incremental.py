import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score


# =========================================================
# 1. 데이터 불러오기
# =========================================================

data = pd.read_csv("training_data_v2.csv")

data["Date"] = pd.to_datetime(data["Date"])

data = data.sort_values(
    ["Date", "Stock_Name"]
).reset_index(drop=True)


# =========================================================
# 2. Feature 정의
# =========================================================

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


additional_features = [
    "Pullback_MA20",
    "Near_High",
    "Momentum_10D",
    "Golden_Trend",
    "MA20_Bounce"
]


# =========================================================
# 3. 실험 조합
# =========================================================

experiments = {

    "V1_Base": base_features,

    "V1_Pullback": base_features + [
        "Pullback_MA20"
    ],

    "V1_Pullback_NearHigh": base_features + [
        "Pullback_MA20",
        "Near_High"
    ],

    "V1_Pullback_NearHigh_Momentum": base_features + [
        "Pullback_MA20",
        "Near_High",
        "Momentum_10D"
    ],

    "V1_Pullback_NearHigh_Momentum_Golden": base_features + [
        "Pullback_MA20",
        "Near_High",
        "Momentum_10D",
        "Golden_Trend"
    ],

    "V2_All": base_features + additional_features
}


# =========================================================
# 4. 결측치 제거
# =========================================================

required_columns = list(
    set(
        base_features
        + additional_features
        + ["Label", "Future_5D_Return", "Future_5D_Close_Return"]
    )
)

data = data.dropna(
    subset=required_columns
).reset_index(drop=True)


# =========================================================
# 5. Time Split
# =========================================================

split_date = data["Date"].quantile(0.8)

train = data[data["Date"] <= split_date].copy()
test = data[data["Date"] > split_date].copy()

print("=" * 70)
print("INCREMENTAL FEATURE TEST")
print("=" * 70)

print(f"Train : {len(train):,}")
print(f"Test  : {len(test):,}")
print(f"Split : {split_date}")
print()


# =========================================================
# 6. 결과 저장
# =========================================================

results = []


# =========================================================
# 7. 실험 실행
# =========================================================

for experiment_name, features in experiments.items():

    print()
    print("=" * 70)
    print(f"[{experiment_name}]")
    print("=" * 70)

    X_train = train[features]
    y_train = train["Label"]

    X_test = test[features]
    y_test = test["Label"]

    print(f"Features : {len(features)}")
    print("Features :", ", ".join(features))

    # -----------------------------------------------------
    # Random Forest
    # -----------------------------------------------------

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]

    # -----------------------------------------------------
    # Feature Importance
    # -----------------------------------------------------

    importance = pd.DataFrame({
        "Feature": features,
        "Importance": model.feature_importances_
    }).sort_values(
        "Importance",
        ascending=False
    )

    print()
    print("Top Feature Importance")

    print(
        importance.head(10).to_string(index=False)
    )

    # -----------------------------------------------------
    # Threshold별 평가
    # -----------------------------------------------------

    thresholds = [
        0.70,
        0.75,
        0.80
    ]

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

        selected_returns = test.loc[
            selected,
            "Future_5D_Return"
        ]

        selected_close_returns = test.loc[
            selected,
            "Future_5D_Close_Return"
        ]

        avg_return = selected_returns.mean()
        median_return = selected_returns.median()

        avg_close_return = selected_close_returns.mean()
        median_close_return = selected_close_returns.median()

        selection_rate = selected_count / len(test)

        positive_rate = (
            selected_returns >= 10
        ).mean()

        print()
        print(
            f"Threshold {threshold:.2f}"
        )

        print(
            f"Selected       : {selected_count:,}"
        )

        print(
            f"Selection Rate : {selection_rate:.2%}"
        )

        print(
            f"Precision      : {precision:.4%}"
        )

        print(
            f"Recall         : {recall:.4%}"
        )

        print(
            f"F1             : {f1:.4f}"
        )

        print(
            f"Avg 5D Return  : {avg_return:.2f}%"
        )

        print(
            f"Median 5D      : {median_return:.2f}%"
        )

        print(
            f"Avg Close      : {avg_close_return:.2f}%"
        )

        print(
            f"Median Close   : {median_close_return:.2f}%"
        )

        print(
            f"+10% Rate      : {positive_rate:.2%}"
        )

        results.append({
            "Experiment": experiment_name,
            "Feature_Count": len(features),
            "Threshold": threshold,
            "Selected": selected_count,
            "Selection_Rate": selection_rate,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "Avg_5D_Return": avg_return,
            "Median_5D_Return": median_return,
            "Avg_5D_Close_Return": avg_close_return,
            "Median_5D_Close_Return": median_close_return,
            "Positive_10_Rate": positive_rate
        })


# =========================================================
# 8. 결과 저장
# =========================================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    "backtest_incremental_results.csv",
    index=False,
    encoding="utf-8-sig"
)


# =========================================================
# 9. Threshold 0.70 비교
# =========================================================

print()
print()
print("=" * 100)
print("THRESHOLD 0.70 COMPARISON")
print("=" * 100)

comparison_70 = results_df[
    results_df["Threshold"] == 0.70
][[
    "Experiment",
    "Feature_Count",
    "Selection_Rate",
    "Precision",
    "Recall",
    "F1",
    "Avg_5D_Return",
    "Avg_5D_Close_Return"
]]

print(
    comparison_70.to_string(index=False)
)


# =========================================================
# 10. Threshold 0.80 비교
# =========================================================

print()
print()
print("=" * 100)
print("THRESHOLD 0.80 COMPARISON")
print("=" * 100)

comparison_80 = results_df[
    results_df["Threshold"] == 0.80
][[
    "Experiment",
    "Feature_Count",
    "Selection_Rate",
    "Precision",
    "Recall",
    "F1",
    "Avg_5D_Return",
    "Avg_5D_Close_Return"
]]

print(
    comparison_80.to_string(index=False)
)


print()
print("=" * 70)
print("완료")
print("결과 파일: backtest_incremental_results.csv")
print("=" * 70)