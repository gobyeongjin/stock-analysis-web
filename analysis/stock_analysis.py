import yfinance as yf



def get_stock_data(ticker):
    data = yf.download(ticker, period="1y")

    data.columns = data.columns.get_level_values(0)

    return data


def calculate_indicators(data):
    """주요 기술적 지표를 계산한다."""

    

    # 수익률
    data["Return"] = data["Close"].pct_change()

    # 이동평균
    data["MA20"] = data["Close"].rolling(20).mean()
    data["MA60"] = data["Close"].rolling(60).mean()

    # RSI
    delta = data["Close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss

    data["RSI"] = 100 - (100 / (1 + rs))

    # 거래량 이동평균
    data["Volume_MA20"] = data["Volume"].rolling(20).mean()

    # 거래량 비율
    data["Volume_Ratio"] = (
        data["Volume"] / data["Volume_MA20"]
    )

    # 거래량 급증
    data["Volume_Surge"] = data["Volume_Ratio"] >= 2


    #5거래일 동안 주가 변화율
    data["Return_5D"] = data["Close"].pct_change(5) * 100

    return data

   


def analyze_stock(data):
    current_price = data["Close"].iloc[-1]
    ma20 = data["MA20"].iloc[-1]
    ma60 = data["MA60"].iloc[-1]
    current_rsi = data["RSI"].iloc[-1]
    volume_ratio = data["Volume_Ratio"].iloc[-1]

    # -----------------------
    # RSI 판단
    # -----------------------
    if current_rsi <= 30:
        rsi_status = "과매도"
    elif current_rsi >= 70:
        rsi_status = "과매수"
    else:
        rsi_status = "중립"


    # -----------------------
    # 추세 판단
    # -----------------------
    if current_price > ma20 and ma20 > ma60:
        trend_status = "상승 추세"
    elif current_price < ma20 and ma20 < ma60:
        trend_status = "하락 추세"
    else:
        trend_status = "혼조"


    # -----------------------
    # 거래량 판단
    # -----------------------
    if volume_ratio >= 2:
        volume_status = "거래량 급증"
    elif volume_ratio >= 1.5:
        volume_status = "거래량 증가"
    else:
        volume_status = "평균 수준"


    


    # -----------------------
    # 종합 점수
    # -----------------------
    score = 50

    # 추세
    if current_price > ma20:
        score += 15

    if ma20 > ma60:
        score += 15

    # RSI
    if current_rsi <= 30:
        score += 15
    elif current_rsi >= 70:
        score -= 15

    # 거래량
    if volume_ratio >= 1.5:
        score += 10



    # 급등 신호 점수
    surge_score = 0
    surge_signals = []

    return_5d = data["Return_5D"].iloc[-1]

    if volume_ratio >= 2:
        surge_score += 25
        surge_signals.append("거래량 급증")

    if return_5d >= 5:
        surge_score += 20
        surge_signals.append("5일 상승")

    if current_price > ma20:
        surge_score += 15
        surge_signals.append("MA20 상회")

    if ma20 > ma60:
        surge_score += 15
        surge_signals.append("상승 추세")

    if 50 <= current_rsi <= 70:
        surge_score += 10
        surge_signals.append("상승 모멘텀")

    if surge_score >= 60:
        surge_status = "급등 관심"
    elif surge_score >= 40:
        surge_status = "관심"
    else:
        surge_status = "낮음"


    # -----------------------
    # 최종 판단
    # -----------------------
    if score >= 70:
        recommendation = "매수 관심"
    elif score < 40:
        recommendation = "매도 관심"
    else:
        recommendation = "중립"


    result = {
        "current_price": current_price,
        "ma20": ma20,
        "ma60": ma60,
        "rsi": current_rsi,
        "rsi_status": rsi_status,
        "trend_status": trend_status,
        "volume_ratio": volume_ratio,
        "volume_status": volume_status,
        "score": score,
        "recommendation": recommendation,
        "return_5d": return_5d,
        "surge_score": surge_score,
        "surge_status": surge_status,
        "surge_signals": surge_signals
    }

    return result


from stock_analysis import get_stock_data, calculate_indicators, analyze_stock
from visualization import plot_all


#테스트
if __name__ == "__main__":

    stock_codes = {
        "삼성전자": "005930.KS",
        "에코프로": "086520.KQ",
        "SK하이닉스": "000660.KS",
        "SK이노베이션": "096770.KS",
        "현대차": "005380.KS",
        "디앤디파마텍": "347850.KQ"
    }

    stock_name = input("종목명을 입력하세요: ")

    if stock_name not in stock_codes:
        print("등록되지 않은 종목입니다.")
        exit()

    ticker = stock_codes[stock_name]

    data = get_stock_data(ticker)
    data = calculate_indicators(data)
    result = analyze_stock(data)

    print("================================")
    print(f"       {stock_name} 분석 결과")
    print("================================")

    print(f"현재가      : {result['current_price']:,.0f}원")
    print(f"MA20        : {result['ma20']:,.0f}원")
    print(f"MA60        : {result['ma60']:,.0f}원")

    print(f"RSI         : {result['rsi']:.2f}")
    print(f"RSI 상태    : {result['rsi_status']}")

    print(f"추세        : {result['trend_status']}")

    print(f"거래량 비율 : {result['volume_ratio']:.2f}배")
    print(f"거래량 상태 : {result['volume_status']}")

    print("================================")
    print(f"종합 점수   : {result['score']}점")
    print(f"종합 판단   : {result['recommendation']}")

    print("================================")
    print(f"5일 수익률      : {result['return_5d']:.2f}%")
    print(f"급등 신호 점수  : {result['surge_score']}점")
    print(f"급등 신호       : {result['surge_status']}")

    if result["surge_signals"]:
        print("신호            : " + ", ".join(result["surge_signals"]))
    else:
        print("신호            : 없음")

    print("================================")

    from visualization import plot_all
    plot_all(data, stock_name)