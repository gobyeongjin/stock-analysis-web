import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
)

import backtest_stop8_test as base

BASE = Path(__file__).resolve().parent

df = pd.read_csv(
    base.TRAIN_FILE,
    dtype={"Ticker": str}
)

df["Date"] = pd.to_datetime(
    df["Date"], errors="coerce"
)

df["Ticker"] = df["Ticker"].map(
    base.bt.normalize_ticker
)

if "Future_5D_Valid" in df.columns:
    valid = (
        df["Future_5D_Valid"]
        .astype(str).str.strip().str.lower()
        .isin(["true", "1", "1.0"])
    )
    df = df[valid].copy()

features = base.FEATURES

for col in features + ["Label"]:
    df[col] = pd.to_numeric(
        df[col], errors="coerce"
    )

df = df.dropna(
    subset=features + ["Label", "Date", "Ticker"]
)

df = df[df["Label"].isin([0, 1])].copy()
df["Label"] = df["Label"].astype(int)

# 날짜 기준 80:20 분할
dates = pd.DatetimeIndex(
    sorted(df["Date"].unique())
)

split = int(len(dates) * 0.8)
test_start = dates[split]

# 평가 직전 5거래일 제거
train_dates = dates[:max(0, split - 5)]

train = df[
    df["Date"].isin(train_dates)
].copy()

test = df[
    df["Date"] >= test_start
].copy()

print("\n===== 498종목 모델 학습 =====")
print("학습 행:", len(train))
print("평가 행:", len(test))
print("학습 종목:", train["Ticker"].nunique())
print("평가 종목:", test["Ticker"].nunique())
print("평가 시작일:", test_start.date())

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=12,
    min_samples_leaf=10,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)

model.fit(
    train[features],
    train["Label"]
)

proba = model.predict_proba(
    test[features]
)[:, 1]

ap = average_precision_score(
    test["Label"], proba
)

print("\n===== 모델 평가 =====")
print("Average Precision:", round(ap, 4))
print(
    "양성 라벨 비율:",
    round(test["Label"].mean(), 4)
)

print("\n===== 임계값별 신호 분석 =====")

for threshold in [0.50, 0.55, 0.60, 0.65, 0.70]:

    selected = proba >= threshold

    precision = precision_score(
        test["Label"], selected,
        zero_division=0
    )

    recall = recall_score(
        test["Label"], selected,
        zero_division=0
    )

    print(
        f"기준 {threshold:.2f} | "
        f"신호 {selected.sum():,}건 | "
        f"종목 {test.loc[selected, 'Ticker'].nunique()}개 | "
        f"Precision {precision:.3f} | "
        f"Recall {recall:.3f}"
    )

# 이후 분석용 전체 예측 결과 저장
output = test[
    ["Date", "Ticker", "Label"]
].copy()

output["Probability"] = proba

output.to_csv(
    BASE / "wide_model_predictions.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n저장 완료: analysis/wide_model_predictions.csv")
