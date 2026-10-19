from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score

import backtest_stop8_test as base

BASE = Path(__file__).resolve().parent
THRESHOLD = 0.65

# 기존 워크포워드 평가 구간 사용
old_metrics = pd.read_csv(
    BASE / "wf_model_metrics.csv"
)

print("기존 지표 컬럼:", old_metrics.columns.tolist())

required = ["Test_Start", "Test_End"]
missing = [c for c in required if c not in old_metrics.columns]

if missing:
    raise ValueError(f"평가 날짜 컬럼 없음: {missing}")

old_metrics["Test_Start"] = pd.to_datetime(
    old_metrics["Test_Start"]
)
old_metrics["Test_End"] = pd.to_datetime(
    old_metrics["Test_End"]
)

# 전체 학습 데이터
df = pd.read_csv(
    base.TRAIN_FILE,
    dtype={"Ticker": str}
)

df["Date"] = pd.to_datetime(
    df["Date"], errors="coerce"
).dt.normalize()

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

df = df.sort_values(
    ["Date", "Ticker"]
).reset_index(drop=True)

dates = pd.DatetimeIndex(
    sorted(df["Date"].unique())
)

print("\n===== 전체 데이터 =====")
print("행 수:", len(df))
print("종목 수:", df["Ticker"].nunique())
print("날짜:", dates.min().date(), "~", dates.max().date())

all_predictions = []
fold_results = []

for fold, row in old_metrics.reset_index(drop=True).iterrows():

    test_start = row["Test_Start"].normalize()
    test_end = row["Test_End"].normalize()

    # 기존 워크포워드와 동일하게 평가 직전
    # 5개 거래일을 학습에서 제외
    start_idx = dates.searchsorted(test_start)

    if start_idx < 5:
        raise ValueError("학습 기간이 너무 짧습니다.")

    train_end_idx = start_idx - 5

    train_dates = dates[:train_end_idx]

    train = df[
        df["Date"].isin(train_dates)
    ].copy()

    test = df[
        df["Date"].between(test_start, test_end)
    ].copy()

    if train.empty or test.empty:
        raise ValueError(f"{fold+1}폴드 데이터 없음")

    print(f"\n===== Fold {fold+1} =====")
    print("학습:", train["Date"].min().date(),
          "~", train["Date"].max().date())
    print("평가:", test_start.date(),
          "~", test_end.date())
    print("학습 행:", len(train))
    print("평가 행:", len(test))

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        train[features],
        train["Label"]
    )

    probability = model.predict_proba(
        test[features]
    )[:, 1]

    ap = average_precision_score(
        test["Label"],
        probability
    )

    predictions = test[
        ["Date", "Ticker", "Label"]
    ].copy()

    predictions["Probability"] = probability
    predictions["Fold"] = fold + 1

    all_predictions.append(predictions)

    selected = predictions[
        predictions["Probability"] >= THRESHOLD
    ]

    precision = (
        selected["Label"].mean()
        if len(selected) else np.nan
    )

    fold_results.append({
        "Fold": fold + 1,
        "Test_Start": test_start,
        "Test_End": test_end,
        "Train_Rows": len(train),
        "Test_Rows": len(test),
        "Test_Stocks": test["Ticker"].nunique(),
        "Positive_Rate": test["Label"].mean(),
        "AP": ap,
        "Signal_Count": len(selected),
        "Signal_Stocks": selected["Ticker"].nunique(),
        "Precision_065": precision,
    })

    print("AP:", round(ap, 4))
    print("양성 비율:", round(test["Label"].mean(), 4))
    print("0.65 신호:", len(selected))
    print("신호 종목:", selected["Ticker"].nunique())
    print("신호 Precision:", round(precision, 4))

# 전체 결과
predictions = pd.concat(
    all_predictions,
    ignore_index=True
)

summary = pd.DataFrame(fold_results)

predictions.to_csv(
    BASE / "wf_498_predictions.csv",
    index=False,
    encoding="utf-8-sig"
)

summary.to_csv(
    BASE / "wf_498_metrics.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n===== 워크포워드 요약 =====")
print(summary.round(4).to_string(index=False))

print("\n===== 전체 신호 분석 =====")

signals = predictions[
    predictions["Probability"] >= THRESHOLD
].copy()

print("전체 신호:", len(signals))
print("신호 발생 종목:", signals["Ticker"].nunique())

daily = signals.groupby("Date").size()

print("신호 발생일:", len(daily))
print("신호 발생일당 평균:", round(daily.mean(), 2))
print("신호 발생일당 중앙값:", round(daily.median(), 2))
print("하루 최대 신호:", daily.max())

print("\n===== 신호가 많은 상위 15종목 =====")
print(
    signals["Ticker"]
    .value_counts()
    .head(15)
    .to_string()
)

print("\n저장 완료:")
print("analysis/wf_498_predictions.csv")
print("analysis/wf_498_metrics.csv")
