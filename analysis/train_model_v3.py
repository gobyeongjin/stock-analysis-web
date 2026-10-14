import os
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "training_data_v3.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "random_forest_v3.pkl")

# ==========================================
# 1. 데이터 불러오기
# ==========================================
data = pd.read_csv(DATA_PATH)
data["Date"] = pd.to_datetime(data["Date"])

# 미래 5일 데이터가 존재하는 행만 학습/평가에 사용
data = data[data["Future_5D_Valid"] == True].copy()

data = data.replace([np.inf, -np.inf], np.nan)

features = [
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

data = data.dropna(subset=features + ["Label"])
data = data.sort_values(["Date", "Ticker"]).reset_index(drop=True)

X = data[features]
y = data["Label"].astype(int)

# ==========================================
# 2. 날짜 기준 분리
# 과거 데이터로 학습하고 이후 데이터로 평가
# ==========================================
unique_dates = np.sort(data["Date"].unique())
split_idx = int(len(unique_dates) * 0.8)
split_date = unique_dates[split_idx]

train_mask = data["Date"] < split_date
test_mask = data["Date"] >= split_date

X_train = X.loc[train_mask]
y_train = y.loc[train_mask]
X_test = X.loc[test_mask]
y_test = y.loc[test_mask]

print("=" * 50)
print("V3 Random Forest 학습")
print("=" * 50)
print(f"학습 기준일 이전: {data.loc[train_mask, 'Date'].min().date()} ~ "
      f"{data.loc[train_mask, 'Date'].max().date()}")
print(f"평가 기간: {data.loc[test_mask, 'Date'].min().date()} ~ "
      f"{data.loc[test_mask, 'Date'].max().date()}")
print(f"학습 데이터: {len(X_train):,}")
print(f"평가 데이터: {len(X_test):,}")
print(f"학습 양성 비율: {y_train.mean():.2%}")
print(f"평가 양성 비율: {y_test.mean():.2%}")

# ==========================================
# 3. 모델 학습
# ==========================================
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    min_samples_leaf=10,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)

model.fit(X_train, y_train)

# ==========================================
# 4. 평가
# ==========================================
proba = model.predict_proba(X_test)[:, 1]
pred = (proba >= 0.5).astype(int)

print("\n[기본 임계값 0.50 평가]")
print(f"Accuracy : {accuracy_score(y_test, pred):.4f}")
print(f"Precision: {precision_score(y_test, pred, zero_division=0):.4f}")
print(f"Recall   : {recall_score(y_test, pred, zero_division=0):.4f}")
print(f"F1       : {f1_score(y_test, pred, zero_division=0):.4f}")

print("\n[분류 보고서]")
print(classification_report(y_test, pred, zero_division=0))
print("[혼동 행렬]")
print(confusion_matrix(y_test, pred))

# ==========================================
# 5. 임계값별 평가
# ==========================================
print("\n[임계값별 평가]")
for threshold in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
    selected = proba >= threshold
    count = int(selected.sum())

    if count == 0:
        print(f"{threshold:.2f}: 선택 종목 없음")
        continue

    selected_y = y_test.to_numpy()[selected]
    selected_close_returns = data.loc[test_mask, "Future_5D_Close_Return"].to_numpy()[selected]

    print(
        f"Threshold={threshold:.2f} | "
        f"선택={count:,} ({count / len(y_test):.2%}) | "
        f"장중 +10% 도달률={selected_y.mean():.2%} | "
        f"5일 후 평균 종가수익률={np.nanmean(selected_close_returns):.2f}% | "
        f"5일 후 중앙값 종가수익률={np.nanmedian(selected_close_returns):.2f}%"
    )

# ==========================================
# 6. 변수 중요도
# ==========================================
importance = pd.Series(
    model.feature_importances_,
    index=features
).sort_values(ascending=False)

print("\n[Feature Importance]")
print(importance.to_string())

# ==========================================
# 7. 모델 및 평가 결과 저장
# ==========================================
os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump(model, MODEL_PATH)

importance.to_csv(
    os.path.join(BASE_DIR, "feature_importance_v3.csv"),
    encoding="utf-8-sig",
    header=["Importance"],
)

print(f"\n모델 저장 완료: {MODEL_PATH}")
print("변수 중요도 저장: feature_importance_v3.csv")
