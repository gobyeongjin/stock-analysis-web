import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score
)


# ==========================================
# 1. 데이터 불러오기
# ==========================================

data = pd.read_csv("training_data_v2.csv")

data["Date"] = pd.to_datetime(data["Date"])

# 날짜순 정렬
data = data.sort_values("Date").reset_index(drop=True)


# ==========================================
# 2. Feature 설정
# ==========================================

features = [
    # 기존 Feature
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
    "Return_10D",

    # V2 추가 Feature
    "Golden_Trend",
    "Pullback_MA20",
    "Near_High",
    "Momentum_10D",
    "MA20_Bounce"
]


X = data[features]
y = data["Label"]


# ==========================================
# 3. 시간 기준 Train / Test
# ==========================================

split_date = data["Date"].quantile(0.8)

train_data = data[
    data["Date"] <= split_date
].copy()

test_data = data[
    data["Date"] > split_date
].copy()


X_train = train_data[features]
y_train = train_data["Label"]

X_test = test_data[features]
y_test = test_data["Label"]


print("================================")
print("Backtest V2 데이터")
print("================================")

print(
    f"Train 기간 : "
    f"{train_data['Date'].min().date()} ~ "
    f"{train_data['Date'].max().date()}"
)

print(
    f"Test 기간  : "
    f"{test_data['Date'].min().date()} ~ "
    f"{test_data['Date'].max().date()}"
)

print()

print(
    f"Train : {len(train_data):,}개"
)

print(
    f"Test  : {len(test_data):,}개"
)


# ==========================================
# 4. Random Forest 학습
# ==========================================

print()
print("================================")
print("Random Forest V2 학습")
print("================================")

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


# ==========================================
# 4-1. Feature Importance
# ==========================================

feature_importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    "Importance",
    ascending=False
).reset_index(drop=True)


print()
print("================================")
print("V2 Feature Importance")
print("================================")

print(
    feature_importance.to_string(
        index=False
    )
)


# Feature Importance 저장
feature_importance.to_csv(
    "feature_importance_v2.csv",
    index=False,
    encoding="utf-8-sig"
)

# ==========================================
# 5. Test 데이터 예측
# ==========================================

test_data["AI_Score"] = (
    model.predict_proba(X_test)[:, 1]
)


# ==========================================
# 6. Threshold별 백테스트
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


results = []


print()
print("================================")
print("V2 Threshold별 Backtest")
print("================================")


for threshold in thresholds:

    signal = (
        test_data["AI_Score"] >= threshold
    )

    selected = test_data[signal]

    if len(selected) == 0:
        continue

    # 실제 +10% 도달 여부
    actual_positive = (
        selected["Label"] == 1
    )

    precision = (
        actual_positive.mean()
    )

    # 전체 실제 급등 중 얼마나 찾았는지
    recall = recall_score(
        y_test,
        signal.astype(int),
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        signal.astype(int),
        zero_division=0
    )

    # 실제 미래 수익률
    avg_return = (
        selected["Future_5D_Return"]
        .mean()
    )

    median_return = (
        selected["Future_5D_Return"]
        .median()
    )

    avg_close_return = (
        selected["Future_5D_Close_Return"]
        .mean()
    )

    median_close_return = (
        selected["Future_5D_Close_Return"]
        .median()
    )

    max_return = (
        selected["Future_5D_Return"]
        .max()
    )

    min_return = (
        selected["Future_5D_Return"]
        .min()
    )

    positive_rate = (
        selected["Future_5D_Return"] >= 10
    ).mean()

    results.append({
        "Threshold": threshold,
        "Selected": len(selected),
        "Selection_Rate": len(selected) / len(test_data) * 100,
        "Precision": precision * 100,
        "Recall": recall * 100,
        "F1": f1,
        "Avg_5D_Return": avg_return,
        "Median_5D_Return": median_return,
        "Avg_5D_Close_Return": avg_close_return,
        "Median_5D_Close_Return": median_close_return,
        "Max_5D_Return": max_return,
        "Min_5D_Return": min_return,
        "Positive_Rate": positive_rate * 100
    })


    print()
    print(
        f"Threshold : {threshold:.2f}"
    )

    print(
        f"선정 데이터 : "
        f"{len(selected):,}개"
    )

    print(
        f"선정 비율 : "
        f"{len(selected) / len(test_data) * 100:.2f}%"
    )

    print(
        f"Precision : "
        f"{precision:.4f}"
    )

    print(
        f"Recall : "
        f"{recall:.4f}"
    )

    print(
        f"F1 : "
        f"{f1:.4f}"
    )

    print(
        f"평균 5일 수익률 : "
        f"{avg_return:.2f}%"
    )

    print(
        f"중앙값 5일 수익률 : "
        f"{median_return:.2f}%"
    )

    print(
        f"평균 5일 후 종가 수익률 : "
        f"{avg_close_return:.2f}%"
    )

    print(
        f"중앙값 5일 후 종가 수익률 : "
        f"{median_close_return:.2f}%"
    )

    print(
        f"+10% 도달 비율 : "
        f"{positive_rate * 100:.2f}%"
    )


# ==========================================
# 7. 결과 DataFrame
# ==========================================

results_df = pd.DataFrame(
    results
)


# ==========================================
# 8. 결과 저장
# ==========================================

results_df.to_csv(
    "backtest_results_v2.csv",
    index=False,
    encoding="utf-8-sig"
)


# ==========================================
# 9. Threshold 0.50 후보 저장
# ==========================================

selected_050 = test_data[
    test_data["AI_Score"] >= 0.50
].copy()


# AI Score에 따른 신호 등급
def get_signal_grade(score):

    if score >= 0.80:
        return "강한 급등 관심"

    elif score >= 0.65:
        return "급등 관심"

    elif score >= 0.50:
        return "관찰"

    else:
        return "일반"


selected_050["Signal"] = (
    selected_050["AI_Score"]
    .apply(get_signal_grade)
)


# AI Score 높은 순으로 정렬
selected_050 = selected_050.sort_values(
    "AI_Score",
    ascending=False
)


selected_columns = [
    "Date",
    "Stock_Name",
    "Ticker",
    "AI_Score",
    "Signal",

    "Future_5D_Return",
    "Label",

    "MA20_Gap",
    "MA60_Gap",
    "MA20_MA60_Gap",

    "RSI",
    "Volume_Ratio",

    "Return_1D",
    "Return_5D",
    "Return_10D",

    # V2 Feature
    "Golden_Trend",
    "Pullback_MA20",
    "Near_High",
    "Momentum_10D",
    "MA20_Bounce"
]


selected_050[
    selected_columns
].to_csv(
    "backtest_candidates_v2.csv",
    index=False,
    encoding="utf-8-sig"
)


# ==========================================
# 10. 최종 출력
# ==========================================

print()
print("================================")
print("V2 Backtest 완료")
print("================================")

print()
print("Threshold별 V2 결과")
print("--------------------------------")

print(
    results_df[
        [
            "Threshold",
            "Selected",
            "Selection_Rate",
            "Precision",
            "Recall",
            "F1",
            "Avg_5D_Return",
            "Median_5D_Return",
            "Avg_5D_Close_Return",
            "Median_5D_Close_Return",
            "Positive_Rate"
        ]
    ].to_string(index=False)
)

print()
print(
    "결과 파일 : "
    "backtest_results_v2.csv"
)

print(
    "후보 종목 : "
    "backtest_candidates_v2.csv"
)

print()
print(
    "Threshold 0.50 후보 : "
    f"{len(selected_050):,}개"
)