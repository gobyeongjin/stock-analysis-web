import pandas as pd
import numpy as np


# ==========================================
# 1. 데이터 불러오기
# ==========================================

data = pd.read_csv("stock_data_500.csv")

data["Date"] = pd.to_datetime(data["Date"])

# 종목별 날짜순 정렬
data = data.sort_values(
    ["Stock_Name", "Date"]
).reset_index(drop=True)


# ==========================================
# 2. 기술적 지표 계산
# ==========================================

# 종목별 20일 이동평균
data["MA20"] = (
    data.groupby("Stock_Name")["Close"]
    .transform(lambda x: x.rolling(20).mean())
)

# 종목별 60일 이동평균
data["MA60"] = (
    data.groupby("Stock_Name")["Close"]
    .transform(lambda x: x.rolling(60).mean())
)


# ==========================================
# 3. RSI 계산
# ==========================================

delta = (
    data.groupby("Stock_Name")["Close"]
    .diff()
)

gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)

avg_gain = (
    gain.groupby(data["Stock_Name"])
    .transform(lambda x: x.rolling(14).mean())
)

avg_loss = (
    loss.groupby(data["Stock_Name"])
    .transform(lambda x: x.rolling(14).mean())
)

rs = avg_gain / avg_loss

data["RSI"] = 100 - (100 / (1 + rs))

# 5일 전 RSI와 비교
data["RSI_Change"] = (
    data.groupby("Stock_Name")["RSI"]
    .diff(5)
)


# ==========================================
# 4. 거래량 비율
# ==========================================

# 20일 평균 거래량
data["Volume_MA20"] = (
    data.groupby("Stock_Name")["Volume"]
    .transform(lambda x: x.rolling(20).mean())
)

# 현재 거래량 / 20일 평균 거래량
data["Volume_Ratio"] = (
    data["Volume"] / data["Volume_MA20"]
)


# ==========================================
# 5. 수익률 Feature
# ==========================================

# 1일 수익률
data["Return_1D"] = (
    data.groupby("Stock_Name")["Close"]
    .pct_change(1) * 100
)

# 3일 수익률
data["Return_3D"] = (
    data.groupby("Stock_Name")["Close"]
    .pct_change(3) * 100
)

# 5일 수익률
data["Return_5D"] = (
    data.groupby("Stock_Name")["Close"]
    .pct_change(5) * 100
)

# 10일 수익률
data["Return_10D"] = (
    data.groupby("Stock_Name")["Close"]
    .pct_change(10) * 100
)


# ==========================================
# 6. 거래량 추가 Feature
# ==========================================

# 5일 평균 거래량
data["Volume_MA5"] = (
    data.groupby("Stock_Name")["Volume"]
    .transform(lambda x: x.rolling(5).mean())
)

# 현재 거래량 / 5일 평균 거래량
data["Volume_Ratio_5D"] = (
    data["Volume"] / data["Volume_MA5"]
)


# ==========================================
# 7. 가격 변동성
# ==========================================

data["Volatility_20D"] = (
    data.groupby("Stock_Name")["Return_5D"]
    .transform(lambda x: x.rolling(20).std())
)


# ==========================================
# 8. 이동평균 대비 가격 위치
# ==========================================

# 현재가가 MA20보다 얼마나 위/아래에 있는지
data["MA20_Gap"] = (
    data["Close"] / data["MA20"] - 1
)

# 현재가가 MA60보다 얼마나 위/아래에 있는지
data["MA60_Gap"] = (
    data["Close"] / data["MA60"] - 1
)

# MA20이 MA60보다 얼마나 위/아래에 있는지
data["MA20_MA60_Gap"] = (
    data["MA20"] / data["MA60"] - 1
)


# ==========================================
# 9. 추가 Feature
# ==========================================

# 최근 5일간 MA20의 변화율
data["MA20_Change"] = (
    data.groupby("Stock_Name")["MA20"]
    .pct_change(5) * 100
)

# 최근 5일간 MA60의 변화율
data["MA60_Change"] = (
    data.groupby("Stock_Name")["MA60"]
    .pct_change(5) * 100
)

# 최근 20일 최고가
data["High_20D"] = (
    data.groupby("Stock_Name")["High"]
    .transform(lambda x: x.rolling(20).max())
)

# 현재가가 최근 20일 최고가에서 얼마나 떨어져 있는지
data["High_20D_Gap"] = (
    data["Close"] / data["High_20D"] - 1
)


# ==========================================
# 10. Label A
# 향후 5거래일 중 한 번이라도 +10% 이상 상승했는지 확인

future_highs = pd.concat(
    [
        data.groupby("Stock_Name")["High"].shift(-1),
        data.groupby("Stock_Name")["High"].shift(-2),
        data.groupby("Stock_Name")["High"].shift(-3),
        data.groupby("Stock_Name")["High"].shift(-4),
        data.groupby("Stock_Name")["High"].shift(-5)
    ],
    axis=1
)

future_max = future_highs.max(axis=1)

data["Future_5D_Return"] = (
    future_max / data["Close"] - 1
) * 100

data["Label"] = (
    data["Future_5D_Return"] >= 10
).astype(int)

# 5거래일 후 종가
data["Future_5D_Close"] = (
    data.groupby("Stock_Name")["Close"].shift(-5)
)

# 현재 종가 대비 5거래일 후 종가 수익률
data["Future_5D_Close_Return"] = (
    data["Future_5D_Close"] / data["Close"] - 1
) * 100


# ==========================================
# 11. ML 학습에 사용할 컬럼
# ==========================================

features = [
    "Date",
    "Stock_Name",
    "Ticker",

    # 이동평균 관련
    "MA20_Gap",
    "MA60_Gap",
    "MA20_MA60_Gap",

    # 새로 추가된 Feature
    "MA20_Change",
    "MA60_Change",
    "High_20D_Gap",

    # RSI
    "RSI",
    "RSI_Change",

    # 거래량
    "Volume_Ratio",
    "Volume_Ratio_5D",

    # 수익률
    "Return_1D",
    "Return_3D",
    "Return_5D",
    "Return_10D",

    # 변동성
    "Volatility_20D",

    # 미래 수익률 및 Label
    "Future_5D_Return",
    "Future_5D_Close_Return",
    "Label"
]

training_data = data[features].copy()


# ==========================================
# 12. 결측값 제거
# ==========================================

training_data = training_data.dropna()


# ==========================================
# 13. 저장
# ==========================================

training_data.to_csv(
    "training_data.csv",
    index=False,
    encoding="utf-8-sig"
)


# ==========================================
# 14. 결과 확인
# ==========================================

print("================================")
print("학습 데이터 생성 완료")
print("================================")

print(
    f"전체 데이터 : {len(training_data):,}개"
)

print(
    f"종목 수 : {training_data['Stock_Name'].nunique()}개"
)

print()

print("Label 분포")
print(
    training_data["Label"].value_counts()
)

print()

print("Label 비율")
print(
    training_data["Label"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

print()

print("생성된 Feature")
print(
    training_data.columns.tolist()
)

print()

print("저장 파일 : training_data.csv")

