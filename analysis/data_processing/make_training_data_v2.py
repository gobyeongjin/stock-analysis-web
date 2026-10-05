import pandas as pd
import numpy as np


# ==========================================
# 1. 데이터 불러오기
# ==========================================

print("================================")
print("V2 Training Data 생성")
print("================================")

print()
print("원본 데이터 불러오는 중...")

data = pd.read_csv("stock_data_500.csv")

data["Date"] = pd.to_datetime(data["Date"])

# 종목별 날짜순 정렬
data = data.sort_values(
    ["Stock_Name", "Date"]
).reset_index(drop=True)

print(f"원본 데이터 : {len(data):,}개")


# ==========================================
# 2. 이동평균
# ==========================================

print()
print("이동평균 계산 중...")

data["MA20"] = (
    data.groupby("Stock_Name")["Close"]
    .transform(
        lambda x: x.rolling(20).mean()
    )
)

data["MA60"] = (
    data.groupby("Stock_Name")["Close"]
    .transform(
        lambda x: x.rolling(60).mean()
    )
)


# ==========================================
# 3. MA Feature
# ==========================================

data["MA20_Gap"] = (
    data["Close"] / data["MA20"] - 1
) * 100


data["MA60_Gap"] = (
    data["Close"] / data["MA60"] - 1
) * 100


data["MA20_MA60_Gap"] = (
    data["MA20"] / data["MA60"] - 1
) * 100


data["MA20_Change"] = (
    data.groupby("Stock_Name")["MA20"]
    .pct_change(5)
) * 100


data["MA60_Change"] = (
    data.groupby("Stock_Name")["MA60"]
    .pct_change(5)
) * 100


# ==========================================
# 4. 20일 고점
# ==========================================

data["High_20D"] = (
    data.groupby("Stock_Name")["High"]
    .transform(
        lambda x: x.rolling(20).max()
    )
)


data["High_20D_Gap"] = (
    data["Close"] /
    data["High_20D"] - 1
) * 100


# ==========================================
# 5. RSI
# ==========================================

print("RSI 계산 중...")


def calculate_rsi(series, period=14):

    delta = series.diff()

    gain = delta.clip(lower=0)

    loss = -delta.clip(upper=0)

    avg_gain = (
        gain.rolling(period).mean()
    )

    avg_loss = (
        loss.rolling(period).mean()
    )

    rs = avg_gain / avg_loss

    rsi = 100 - (
        100 / (1 + rs)
    )

    return rsi


data["RSI"] = (
    data.groupby("Stock_Name")["Close"]
    .transform(calculate_rsi)
)


data["RSI_Change"] = (
    data.groupby("Stock_Name")["RSI"]
    .diff()
)


# ==========================================
# 6. 거래량 Feature
# ==========================================

print("거래량 Feature 계산 중...")


data["Volume_MA20"] = (
    data.groupby("Stock_Name")["Volume"]
    .transform(
        lambda x: x.rolling(20).mean()
    )
)


data["Volume_Ratio"] = (
    data["Volume"] /
    data["Volume_MA20"]
)


data["Volume_Ratio_5D"] = (
    data.groupby("Stock_Name")["Volume_Ratio"]
    .transform(
        lambda x: x.rolling(5).mean()
    )
)


# ==========================================
# 7. 수익률 Feature
# ==========================================

print("수익률 Feature 계산 중...")


data["Return_1D"] = (
    data.groupby("Stock_Name")["Close"]
    .pct_change(1)
) * 100


data["Return_3D"] = (
    data.groupby("Stock_Name")["Close"]
    .pct_change(3)
) * 100


data["Return_5D"] = (
    data.groupby("Stock_Name")["Close"]
    .pct_change(5)
) * 100


data["Return_10D"] = (
    data.groupby("Stock_Name")["Close"]
    .pct_change(10)
) * 100


# ==========================================
# 8. V2 신규 Feature
# ==========================================

print()
print("================================")
print("V2 신규 Feature 계산")
print("================================")


# ------------------------------------------
# 8-1. Golden Trend
# ------------------------------------------

data["Golden_Trend"] = (
    (data["Close"] > data["MA20"]) &
    (data["MA20"] > data["MA60"])
).astype(int)


# ------------------------------------------
# 8-2. Pullback MA20
# ------------------------------------------

data["Pullback_MA20"] = (
    abs(
        data["Close"] -
        data["MA20"]
    )
    / data["MA20"]
)


# ------------------------------------------
# 8-3. 60일 최고가
# ------------------------------------------

data["High_60D"] = (
    data.groupby("Stock_Name")["High"]
    .transform(
        lambda x: x.rolling(60).max()
    )
)


# ------------------------------------------
# 8-4. Near High
# ------------------------------------------

data["Near_High"] = (
    data["Close"] /
    data["High_60D"]
)


# ------------------------------------------
# 8-5. Momentum 10D
# ------------------------------------------

data["Momentum_10D"] = (
    data.groupby("Stock_Name")["Close"]
    .transform(
        lambda x:
        x / x.shift(10) - 1
    )
)


# ------------------------------------------
# 8-6. MA20 Bounce
# ------------------------------------------

data["MA20_Bounce"] = (
    (data["Low"] <= data["MA20"]) &
    (data["Close"] > data["MA20"])
).astype(int)


# ==========================================
# 9. 미래 5일 종가 수익률
# ==========================================

print()
print("미래 수익률 계산 중...")


data["Future_5D_Close_Return"] = (
    data.groupby("Stock_Name")["Close"]
    .shift(-5)
    / data["Close"] - 1
) * 100


# ==========================================
# 10. 미래 5일 최고 수익률
# ==========================================

data["Future_5D_High"] = (
    data.groupby("Stock_Name")["High"]
    .transform(
        lambda x: pd.concat(
            [
                x.shift(-1),
                x.shift(-2),
                x.shift(-3),
                x.shift(-4),
                x.shift(-5)
            ],
            axis=1
        ).max(axis=1)
    )
)


data["Future_5D_Return"] = (
    data["Future_5D_High"]
    / data["Close"] - 1
) * 100


# ==========================================
# 11. Label
# ==========================================

data["Label"] = (
    data["Future_5D_Return"] >= 10
).astype(int)


# ==========================================
# 12. Feature 목록
# ==========================================

features = [

    # -----------------------------
    # 기존 Feature 14개
    # -----------------------------

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


    # -----------------------------
    # V2 신규 Feature 5개
    # -----------------------------

    "Golden_Trend",
    "Pullback_MA20",
    "Near_High",
    "Momentum_10D",
    "MA20_Bounce"
]


# ==========================================
# 13. 최종 데이터 구성
# ==========================================

selected_columns = [

    "Date",
    "Stock_Name",
    "Ticker",

    "Open",
    "High",
    "Low",
    "Close",
    "Volume",

    "Future_5D_Return",
    "Future_5D_Close_Return",

    "Label"

] + features


data = data[selected_columns]


# ==========================================
# 14. 결측치 제거
# ==========================================

before_drop = len(data)

data = data.dropna().reset_index(
    drop=True
)

after_drop = len(data)

print()
print(
    f"결측치 제거 : "
    f"{before_drop - after_drop:,}개"
)


# ==========================================
# 15. 날짜순 정렬
# ==========================================

data = data.sort_values(
    ["Date", "Stock_Name"]
).reset_index(drop=True)


# ==========================================
# 16. 데이터 저장
# ==========================================

output_file = "training_data_v2.csv"

data.to_csv(
    output_file,
    index=False,
    encoding="utf-8-sig"
)


# ==========================================
# 17. 결과 확인
# ==========================================

print()
print("================================")
print("V2 Training Data 생성 완료")
print("================================")

print()

print(
    f"최종 데이터 : {len(data):,}개"
)

print(
    f"Feature 개수 : {len(features)}개"
)

print()

print("Feature 목록")
print("--------------------------------")

for i, feature in enumerate(
    features,
    start=1
):
    print(
        f"{i:2d}. {feature}"
    )


# ==========================================
# 18. Label 분포
# ==========================================

print()
print("Label 분포")
print("--------------------------------")

print(
    data["Label"]
    .value_counts()
)


print()

print("Label 비율")
print("--------------------------------")

print(
    data["Label"]
    .value_counts(
        normalize=True
    ) * 100
)


# ==========================================
# 19. 데이터 기간
# ==========================================

print()
print("데이터 기간")
print("--------------------------------")

print(
    f"{data['Date'].min().date()}"
    f" ~ "
    f"{data['Date'].max().date()}"
)


# ==========================================
# 20. V2 Feature 확인
# ==========================================

print()
print("V2 Feature 샘플")
print("--------------------------------")

print(
    data[
        [
            "Date",
            "Stock_Name",
            "Golden_Trend",
            "Pullback_MA20",
            "Near_High",
            "Momentum_10D",
            "MA20_Bounce"
        ]
    ].head(10).to_string(index=False)
)


print()
print(
    f"저장 완료 : {output_file}"
)