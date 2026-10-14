
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
# 1. 파일 경로 및 설정
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

LENDING_FEATURE = "Lending_Balance_Change_Pct"
LAG_FEATURE = "Lending_Balance_Change_Pct_Lag1"


# ==========================================
# 2. 데이터 불러오기 및 전처리
# ==========================================
df = pd.read_csv(DATA_FILE, dtype={"Ticker": str})
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

# 종목별 날짜순 정렬 후 전 거래일 대차잔고 변화율 생성
df = df.sort_values(["Ticker", "Date"]).copy()

df[LAG_FEATURE] = (
    df.groupby("Ticker")[LENDING_FEATURE].shift(1)
)

# 숫자형 변환
features_to_convert = BASE_FEATURES + [
    LENDING_FEATURE,
    LAG_FEATURE,
    "Label",
]

for col in features_to_convert:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# 미래 5거래일 정답이 유효한 행만 사용
if "Future_5D_Valid" in df.columns:
    valid = (
        df["Future_5D_Valid"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "1.0"])
    )
    df = df[valid].copy()

# A와 C가 동일한 데이터 행에서 평가되도록 구성
required_columns = (
    BASE_FEATURES
    + [LENDING_FEATURE, LAG_FEATURE, "Label", "Date", "Ticker"]
)

df = df.dropna(subset=required_columns).copy()
df["Label"] = df["Label"].astype(int)

# 이진 분류 라벨만 남아 있는지 확인
df = df[df["Label"].isin([0, 1])].copy()

df = df.sort_values(["Date", "Ticker"]).reset_index(drop=True)

if df.empty:
    raise ValueError(
        "사용 가능한 데이터가 없습니다. "
        "대차잔고 변화율과 날짜가 정상적으로 연결되었는지 확인하세요."
    )


# ==========================================
# 3. 날짜 기준 학습 / 평가 데이터 분리
# ==========================================
dates = sorted(df["Date"].unique())

if len(dates) < 10:
    raise ValueError("학습 및 평가에 필요한 날짜 데이터가 부족합니다.")

split_index = int(len(dates) * 0.8)
split_date = pd.Timestamp(dates[split_index])

# 미래 5거래일 정답이 학습/평가 경계를 넘지 않도록
# 경계 직전 5개 관측 날짜를 학습 데이터에서 제외
purge_dates = set(dates[max(0, split_index - 5):split_index])

train = df[
    (df["Date"] < split_date)
    & (~df["Date"].isin(purge_dates))
].copy()

test = df[df["Date"] >= split_date].copy()

if train.empty or test.empty:
    raise ValueError("학습 또는 평가 데이터가 비었습니다.")

print("===== 데이터 분리 =====")
print("종목 수:", df["Ticker"].nunique())
print("전체 행:", len(df))
print("학습 행:", len(train))
print("평가 행:", len(test))
print("학습 마지막 날짜:", train["Date"].max())
print("평가 시작 날짜:", test["Date"].min())
print("학습 상승 비율:", round(train["Label"].mean(), 4))
print("평가 상승 비율:", round(test["Label"].mean(), 4))
print("경계에서 제외한 학습 날짜 수:", len(purge_dates))


# ==========================================
# 4. 모델 평가 함수
# ==========================================
def evaluate(name, columns):
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(train[columns], train["Label"])

    pred = model.predict(test[columns])
    prob = model.predict_proba(test[columns])[:, 1]

    print(f"\n===== {name} =====")
    print(
        "F1:",
        round(f1_score(test["Label"], pred, zero_division=0), 4),
    )
    print(
        "Precision:",
        round(precision_score(test["Label"], pred, zero_division=0), 4),
    )
    print(
        "Recall:",
        round(recall_score(test["Label"], pred, zero_division=0), 4),
    )
    print(
        "Average Precision:",
        round(average_precision_score(test["Label"], prob), 4),
    )


# ==========================================
# 5. A 모델과 C 모델 비교
# ==========================================
evaluate(
    "A 모델: 기존 지표 18개",
    BASE_FEATURES,
)

evaluate(
    "C 모델: 기존 지표 18개 + 전 거래일 대차잔고 변화율",
    BASE_FEATURES + [LAG_FEATURE],
)

print("\n===== 비교 완료 =====")
print("기존 모델 파일은 수정하지 않았습니다.")
