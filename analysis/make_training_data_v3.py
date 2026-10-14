import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(BASE_DIR, "stock_data_500.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "training_data_v3.csv")

# ==========================================
# 1. 데이터 불러오기
# ==========================================
data = pd.read_csv(INPUT_PATH)
data["Date"] = pd.to_datetime(data["Date"])

data = data.sort_values(
    ["Stock_Name", "Date"]
).reset_index(drop=True)

grouped = data.groupby("Stock_Name", sort=False)

# ==========================================
# 2. 이동평균
# ==========================================
data["MA20"] = grouped["Close"].transform(
    lambda x: x.rolling(20).mean()
)
data["MA60"] = grouped["Close"].transform(
    lambda x: x.rolling(60).mean()
)

# ==========================================
# 3. RSI
# ==========================================
delta = grouped["Close"].diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)

avg_gain = gain.groupby(data["Stock_Name"]).transform(
    lambda x: x.rolling(14).mean()
)
avg_loss = loss.groupby(data["Stock_Name"]).transform(
    lambda x: x.rolling(14).mean()
)

rs = avg_gain / avg_loss
data["RSI"] = 100 - (100 / (1 + rs))
data["RSI_Change"] = data.groupby("Stock_Name")["RSI"].diff(5)

# ==========================================
# 4. 거래량
# ==========================================
data["Volume_MA20"] = grouped["Volume"].transform(
    lambda x: x.rolling(20).mean()
)
data["Volume_Ratio"] = data["Volume"] / data["Volume_MA20"]

data["Volume_MA5"] = grouped["Volume"].transform(
    lambda x: x.rolling(5).mean()
)
data["Volume_Ratio_5D"] = data["Volume"] / data["Volume_MA5"]

# ==========================================
# 5. 최근 수익률
# ==========================================
for days in [1, 3, 5, 10]:
    data[f"Return_{days}D"] = (
        grouped["Close"].pct_change(days) * 100
    )

# ==========================================
# 6. 변동성
# ==========================================
data["Volatility_20D"] = (
    data.groupby("Stock_Name")["Return_5D"]
    .transform(lambda x: x.rolling(20).std())
)

# 최근 변동성이 5거래일 전보다 얼마나 변했는지
# 음수: 변동성 감소, 양수: 변동성 증가
data["Volatility_Change"] = (
    data.groupby("Stock_Name")["Volatility_20D"]
    .pct_change(5)
)

# ==========================================
# 7. 이동평균 대비 가격 위치
# ==========================================
data["MA20_Gap"] = data["Close"] / data["MA20"] - 1
data["MA60_Gap"] = data["Close"] / data["MA60"] - 1
data["MA20_MA60_Gap"] = data["MA20"] / data["MA60"] - 1

data["MA20_Change"] = (
    data.groupby("Stock_Name")["MA20"].pct_change(5) * 100
)
data["MA60_Change"] = (
    data.groupby("Stock_Name")["MA60"].pct_change(5) * 100
)

# ==========================================
# 8. 20일 최고가 / 최저가
# ==========================================
data["High_20D"] = grouped["High"].transform(
    lambda x: x.rolling(20).max()
)
data["Low_20D"] = grouped["Low"].transform(
    lambda x: x.rolling(20).min()
)

data["High_20D_Gap"] = data["Close"] / data["High_20D"] - 1
data["Low_20D_Gap"] = data["Close"] / data["Low_20D"] - 1

# ==========================================
# 9. 당일 종가 위치
# ==========================================
day_range = data["High"] - data["Low"]

data["Close_Position"] = np.where(
    day_range > 0,
    (data["Close"] - data["Low"]) / day_range,
    0.5
)

# ==========================================
# 10. 미래 성과 및 Label
# ==========================================
future_highs = pd.concat(
    [
        data.groupby("Stock_Name")["High"].shift(-i)
        for i in range(1, 6)
    ],
    axis=1
)

future_max = future_highs.max(axis=1)

# 향후 5거래일 중 장중 +10% 도달 여부
data["Future_5D_Return"] = (
    future_max / data["Close"] - 1
) * 100

data["Label"] = (
    data["Future_5D_Return"] >= 10
).astype(int)

# 5거래일 후 종가 수익률: 별도 평가용
data["Future_5D_Close"] = (
    data.groupby("Stock_Name")["Close"].shift(-5)
)

data["Future_5D_Close_Return"] = (
    data["Future_5D_Close"] / data["Close"] - 1
) * 100

# 마지막 5거래일은 미래 데이터가 없으므로 학습에서 제외
data["Future_5D_Valid"] = (
    data.groupby("Stock_Name")["Close"].shift(-5).notna()
)

# OHLC 가격 범위가 맞지 않는 행 제외
valid_ohlc = (
    (data["High"] >= data["Low"])
    & (data["Close"] >= data["Low"])
    & (data["Close"] <= data["High"])
    & (data["Low"] > 0)
)

print(f"OHLC 이상 행 제외: {(~valid_ohlc).sum():,}개")

data = data.loc[valid_ohlc].copy()

data = data.sort_values(
    ["Stock_Name", "Date"]
).reset_index(drop=True)

# ==========================================
# 11. 학습 데이터 컬럼
# ==========================================
features = [
    "Date",
    "Stock_Name",
    "Ticker",
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
    "Future_5D_Return",
    "Future_5D_Close_Return",
    "Label",
    "Future_5D_Valid",
]

training_data = data[features].copy()

# 지표 결측값 및 무한대 제거
training_data = training_data.replace(
    [np.inf, -np.inf], np.nan
)

training_data = training_data.dropna(
    subset=[
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
)

# 미래 5일 데이터가 없는 행은 CSV에는 남겨두되,
# Label을 학습할 때 반드시 제외해야 함
training_data.to_csv(
    OUTPUT_PATH,
    index=False,
    encoding="utf-8-sig"
)

valid = training_data[
    training_data["Future_5D_Valid"]
].copy()

print("=" * 40)
print("V3 학습 데이터 생성 완료")
print("=" * 40)
print(f"전체 행 수: {len(training_data):,}")
print(f"미래 5일 데이터 유효 행 수: {len(valid):,}")
print(f"종목 수: {training_data['Stock_Name'].nunique()}")

print("\nLabel 분포 (미래 5일 데이터 유효 행 기준)")
print(valid["Label"].value_counts().sort_index())

print("\nLabel 비율 (%)")
print(
    valid["Label"]
    .value_counts(normalize=True)
    .sort_index()
    .mul(100)
    .round(2)
)

print(f"\n저장 파일: {OUTPUT_PATH}")
