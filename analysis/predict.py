import pandas as pd
import yfinance as yf
import joblib


# ==========================================
# 1. 모델 및 Feature
# ==========================================

MODEL_PATH = "model/random_forest.pkl"

FEATURES = [
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


# ==========================================
# 2. 주가 데이터 가져오기
# ==========================================

def get_stock_data(ticker):

    data = yf.download(
        ticker,
        period="1y",
        auto_adjust=False
    )

    if data.empty:
        raise ValueError("주가 데이터를 가져오지 못했습니다.")

    # yfinance MultiIndex 처리
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data.reset_index()

    return data


# ==========================================
# 3. AI Feature 계산
#    make_training_data.py와 동일
# ==========================================

def make_features(data):

    data = data.copy()

    # --------------------------------------
    # 이동평균
    # --------------------------------------

    data["MA20"] = (
        data["Close"]
        .rolling(20)
        .mean()
    )

    data["MA60"] = (
        data["Close"]
        .rolling(60)
        .mean()
    )

    # --------------------------------------
    # RSI
    # --------------------------------------

    delta = data["Close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss

    data["RSI"] = (
        100 - (100 / (1 + rs))
    )

    # make_training_data.py와 동일
    data["RSI_Change"] = (
        data["RSI"].diff(5)
    )

    # --------------------------------------
    # 거래량
    # --------------------------------------

    data["Volume_MA20"] = (
        data["Volume"]
        .rolling(20)
        .mean()
    )

    data["Volume_Ratio"] = (
        data["Volume"] /
        data["Volume_MA20"]
    )

    data["Volume_MA5"] = (
        data["Volume"]
        .rolling(5)
        .mean()
    )

    data["Volume_Ratio_5D"] = (
        data["Volume"] /
        data["Volume_MA5"]
    )

    # --------------------------------------
    # 수익률
    # --------------------------------------

    data["Return_1D"] = (
        data["Close"]
        .pct_change(1) * 100
    )

    data["Return_3D"] = (
        data["Close"]
        .pct_change(3) * 100
    )

    data["Return_5D"] = (
        data["Close"]
        .pct_change(5) * 100
    )

    data["Return_10D"] = (
        data["Close"]
        .pct_change(10) * 100
    )

    # --------------------------------------
    # MA Gap
    # --------------------------------------

    data["MA20_Gap"] = (
        data["Close"] /
        data["MA20"] - 1
    )

    data["MA60_Gap"] = (
        data["Close"] /
        data["MA60"] - 1
    )

    data["MA20_MA60_Gap"] = (
        data["MA20"] /
        data["MA60"] - 1
    )

    # --------------------------------------
    # MA 변화
    # --------------------------------------

    data["MA20_Change"] = (
        data["MA20"]
        .pct_change(5) * 100
    )

    data["MA60_Change"] = (
        data["MA60"]
        .pct_change(5) * 100
    )

    # --------------------------------------
    # 20일 고점
    # --------------------------------------

    data["High_20D"] = (
        data["High"]
        .rolling(20)
        .max()
    )

    data["High_20D_Gap"] = (
        data["Close"] /
        data["High_20D"] - 1
    )

    return data


# ==========================================
# 4. AI 예측
# ==========================================

def predict_stock(ticker):

    # 모델 불러오기
    model = joblib.load(MODEL_PATH)

    # 데이터 가져오기
    data = get_stock_data(ticker)

    # Feature 계산
    data = make_features(data)

    # 마지막 데이터
    latest = data.iloc[-1]

    # 모델 입력 데이터
    X = data[FEATURES].iloc[[-1]]

    # 결측값 확인
    if X.isnull().any().any():
        raise ValueError(
            "AI 예측에 필요한 Feature에 결측값이 있습니다."
        )

    # 상승 확률
    probability = model.predict_proba(X)[0][1]

    # 0.5 기준
    prediction = int(probability >= 0.5)

    result = {
        "current_price": latest["Close"],
        "ma20": latest["MA20"],
        "ma60": latest["MA60"],
        "rsi": latest["RSI"],
        "volume_ratio": latest["Volume_Ratio"],
        "return_5d": latest["Return_5D"],
        "ai_probability": probability,
        "prediction": prediction
    }

    return result


# ==========================================
# 5. 테스트
# ==========================================

if __name__ == "__main__":

    stock_codes = {
        "삼성전자": "005930.KS",
        "SK하이닉스": "000660.KS",
        "현대차": "005380.KS",
        "에코프로": "086520.KQ"
    }

    stock_name = input(
        "종목명을 입력하세요: "
    )

    if stock_name not in stock_codes:

        print("등록되지 않은 종목입니다.")
        exit()

    ticker = stock_codes[stock_name]

    result = predict_stock(ticker)

    print()
    print("================================")
    print(f"       {stock_name} AI 분석")
    print("================================")

    print(
        f"현재가              : "
        f"{result['current_price']:,.0f}원"
    )

    print(
        f"MA20                : "
        f"{result['ma20']:,.0f}원"
    )

    print(
        f"MA60                : "
        f"{result['ma60']:,.0f}원"
    )

    print(
        f"RSI                 : "
        f"{result['rsi']:.2f}"
    )

    print(
        f"거래량 비율          : "
        f"{result['volume_ratio']:.2f}배"
    )

    print(
        f"5일 수익률           : "
        f"{result['return_5d']:.2f}%"
    )

    print("--------------------------------")

    print(
        f"AI 상승 신호 확률    : "
        f"{result['ai_probability'] * 100:.2f}%"
    )

    if result["prediction"] == 1:
        print("AI 판단              : 상승 신호")
    else:
        print("AI 판단              : 일반 신호")

    print("================================")