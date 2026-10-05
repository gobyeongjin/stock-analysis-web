import pandas as pd
import joblib
import os

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)


# ==========================================
# 1. 학습 데이터 불러오기
# ==========================================

data = pd.read_csv("training_data.csv")

data["Date"] = pd.to_datetime(data["Date"])

# 날짜순 정렬
data = data.sort_values("Date").reset_index(drop=True)


# ==========================================
# 2. Feature / Label 지정
# ==========================================

features = [
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
X = data[features]
y = data["Label"]


# ==========================================
# 3. 시간 기준 Train / Test 분리
# ==========================================

split_date = data["Date"].quantile(0.8)

train_data = data[data["Date"] <= split_date]
test_data = data[data["Date"] > split_date]

X_train = train_data[features]
y_train = train_data["Label"]

X_test = test_data[features]
y_test = test_data["Label"]


print("================================")
print("Train / Test 데이터")
print("================================")

print(f"Train : {len(X_train):,}개")
print(f"Test  : {len(X_test):,}개")

print()


# ==========================================
# 4. Random Forest 모델 생성
# ==========================================

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    min_samples_leaf=10,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)


# ==========================================
# 5. 모델 학습
# ==========================================

print("Random Forest 학습 중...")

model.fit(X_train, y_train)

print("학습 완료!")

# ==========================================
# 모델 저장
# ==========================================

os.makedirs("model", exist_ok=True)

model_path = "model/random_forest.pkl"

joblib.dump(model, model_path)

print(f"모델 저장 완료: {model_path}")


# ==========================================
# 6. 예측
# ==========================================

from sklearn.metrics import precision_score, recall_score, f1_score

y_prob = model.predict_proba(X_test)[:, 1]

print("\n================================")
print("Threshold 비교")
print("================================")

for threshold in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:

    y_pred_threshold = (y_prob >= threshold).astype(int)

    precision = precision_score(
        y_test,
        y_pred_threshold,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred_threshold,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred_threshold,
        zero_division=0
    )

    print(
        f"Threshold {threshold:.2f} | "
        f"Precision: {precision:.4f} | "
        f"Recall: {recall:.4f} | "
        f"F1: {f1:.4f}"
    )

y_pred = (y_prob >= 0.5).astype(int)
# ==========================================
# 7. 평가
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

print()
print("================================")
print("모델 평가")
print("================================")

print(f"Accuracy : {accuracy:.4f}")

print()
print("Classification Report")
print(
    classification_report(
        y_test,
        y_pred,
        digits=4
    )
)

print("Confusion Matrix")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


# ==========================================
# 8. Feature 중요도
# ==========================================

importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    "Importance",
    ascending=False
)

print()
print("================================")
print("Feature Importance")
print("================================")

print(importance.to_string(index=False))