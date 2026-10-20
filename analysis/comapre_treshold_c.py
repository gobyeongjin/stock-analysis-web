
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    average_precision_score,
)

# ==========================================
# 1. 설정
# ==========================================
BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "training_data_v4_lending.csv"

BASE_FEATURES = [
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

LAG_FEATURE = "Lending_Balance_Change_Pct_Lag1"

# 비교할 예측 확률 임계값
THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70]


# ==========================================
# 2. 데이터 불러오기
# ==========================================
df = pd.read_csv(DATA_FILE, dtype={"Ticker": str})
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.sort_values(["Ticker", "Date"]).copy()

# 전 거래일 대차잔고 변화율
df[LAG_FEATURE] = (
    df.groupby("Ticker")["Lending_Balance_Change_Pct"].shift(1)
)

features = BASE_FEATURES + [LAG_FEATURE]

for col in features + [
    "Label",
    "Future_5D_Return",
    "Future_5D_Valid",
]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Future_5D_Valid가 True/False로 저장된 CSV도 처리
if "Future_5D_Valid" in df.columns:
    valid_text = (
        df["Future_5D_Valid"]
        .astype(str)
        .str.strip()
        .str.lower()
    )
    valid_mask = valid_text.isin(["true", "1", "1.0"])
    df = df[valid_mask].copy()

required = features + [
    "Label",
    "Future_5D_Return",
    "Date",
    "Ticker",
]

df = df.dropna(subset=required).copy()
df = df[df["Label"].isin([0, 1])].copy()
df["Label"] = df["Label"].astype(int)

df = df.sort_values(["Date", "Ticker"]).reset_index(drop=True)

if df.empty:
    raise ValueError(
        "유효한 데이터가 없습니다. 컬럼명과 결측값을 확인해 주세요."
    )


# ==========================================
# 3. 날짜 기준 학습 / 평가 분리
# ==========================================
dates = sorted(df["Date"].unique())
split_index = int(len(dates) * 0.8)

if split_index < 6 or split_index >= len(dates):
    raise ValueError("학습/평가 분리에 필요한 날짜가 부족합니다.")

split_date = pd.Timestamp(dates[split_index])

# 미래 5거래일 라벨이 경계를 넘는 것을 줄이기 위해
# 경계 직전 5개 관측 날짜를 학습에서 제외
purge_dates = set(dates[max(0, split_index - 5):split_index])

train = df[
    (df["Date"] < split_date)
    & (~df["Date"].isin(purge_dates))
].copy()

test = df[df["Date"] >= split_date].copy()

print("===== 데이터 분리 =====")
print("종목 수:", df["Ticker"].nunique())
print("전체 행:", len(df))
print("학습 행:", len(train))
print("평가 행:", len(test))
print("학습 마지막 날짜:", train["Date"].max())
print("평가 시작 날짜:", test["Date"].min())
print("학습 상승 비율:", round(train["Label"].mean(), 4))
print("평가 상승 비율:", round(test["Label"].mean(), 4))


# ==========================================
# 4. C 모델 학습
# ==========================================
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_leaf=10,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)

model.fit(train[features], train["Label"])

probabilities = model.predict_proba(test[features])[:, 1]
actual_labels = test["Label"].to_numpy()

# Future_5D_Return의 단위가 퍼센트인지 비율인지 확인 필요.
# 기존 데이터가 10을 의미하는 퍼센트 단위라면 10.0,
# 0.10을 의미하는 비율 단위라면 0.10을 사용해야 함.
future_return = test["Future_5D_Return"].to_numpy()

print("\n===== 모델 기본 성능 =====")
print(
    "Average Precision:",
    round(average_precision_score(actual_labels, probabilities), 4),
)


# ==========================================
# 5. 임계값별 성능 평가
# ==========================================
results = []

for threshold in THRESHOLDS:
    predictions = (probabilities >= threshold).astype(int)
    selected = predictions == 1

    selected_count = int(selected.sum())

    if selected_count > 0:
        selected_labels = actual_labels[selected]
        selected_returns = future_return[selected]

        precision = precision_score(
            actual_labels,
            predictions,
            zero_division=0,
        )

        # 예측된 종목들의 실제 평균 5거래일 수익률
        mean_return = float(selected_returns.mean())

        # 예측된 종목 중 +10% 이상 수익률을 기록한 비율
        # Future_5D_Return이 퍼센트 단위라고 가정
        hit_rate_10 = float((selected_returns >= 10.0).mean())

        # 예측된 종목 중 실제 Label=1 비율
        label_hit_rate = float(selected_labels.mean())
    else:
        precision = 0.0
        mean_return = float("nan")
        hit_rate_10 = float("nan")
        label_hit_rate = float("nan")

    recall = recall_score(
        actual_labels,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        actual_labels,
        predictions,
        zero_division=0,
    )

    row = {
        "Threshold": threshold,
        "Predicted_Count": selected_count,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Mean_Future_5D_Return": mean_return,
        "Actual_10pct_Return_Rate": hit_rate_10,
        "Label_Hit_Rate": label_hit_rate,
    }
    results.append(row)

result_df = pd.DataFrame(results)

print("\n===== 임계값별 비교 =====")
print(
    result_df.to_string(
        index=False,
        formatters={
            "Threshold": "{:.2f}".format,
            "Precision": "{:.4f}".format,
            "Recall": "{:.4f}".format,
            "F1": "{:.4f}".format,
            "Mean_Future_5D_Return": "{:.2f}".format,
            "Actual_10pct_Return_Rate": "{:.2%}".format,
            "Label_Hit_Rate": "{:.2%}".format,
        },
    )
)

# 결과 파일 저장
output_file = BASE_DIR / "threshold_results_c.csv"
result_df.to_csv(output_file, index=False, encoding="utf-8-sig")

print("\n결과 저장:", output_file)
print("기존 모델 파일은 수정하지 않았습니다.")
