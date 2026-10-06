import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from predict import predict_stock

import pandas as pd

# ==========================================
# FastAPI 앱 생성
# ==========================================

app = FastAPI(
    title="국내 주식 데이터 분석 API",
    description="주가 데이터 및 AI 분석 결과를 제공하는 API",
    version="1.0.0"
)
@app.get("/")
def home():
    return FileResponse(
        os.path.join(os.path.dirname(__file__), "frontend", "index.html")
    )


# ==========================================
# CORS 설정
# ==========================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================
# 기본 API
# ==========================================

@app.get("/api/stocks")
def get_stocks():

    try:

        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        CSV_PATH = os.path.join(BASE_DIR, "stock_data_500.csv")

        data = pd.read_csv(CSV_PATH)

        stocks = (
            data[["Stock_Name", "Ticker"]]
            .drop_duplicates()
            .sort_values("Stock_Name")
        )

        result = []

        for _, row in stocks.iterrows():
            result.append({
                "name": row["Stock_Name"],
                "ticker": row["Ticker"]
            })

        return result

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
@app.get("/api/stocks/ranking")
def get_stock_ranking():
    try:
        stocks = pd.read_csv("stock_data_500.csv")

        stocks = (
            stocks[["Stock_Name", "Ticker"]]
            .drop_duplicates()
        )

        results = []

        for _, row in stocks.iterrows():

            try:
                result = predict_stock(row["Ticker"])

                score = float(result["ai_probability"])

                results.append({
                    "name": row["Stock_Name"],
                    "ticker": row["Ticker"],
                    "current_price": float(result["current_price"]),
                    "ai_score": score,
                    "rsi": float(result["rsi"]),
                    "return_5d": float(result["return_5d"])
                })

            except Exception as e:
                print(
                    f"{row['Stock_Name']} 분석 실패: {e}"
                )

        results.sort(
            key=lambda x: x["ai_score"],
            reverse=True
        )

        return {
            "count": len(results),
            "stocks": results
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ==========================================
# 주식 AI 분석 API
# ==========================================

@app.get("/api/stock/{ticker}")
def get_stock_prediction(ticker: str):

    try:
        result = predict_stock(ticker)

        return {
            "ticker": ticker,
            "current_price": float(result["current_price"]),
            "ma20": float(result["ma20"]),
            "ma60": float(result["ma60"]),
            "rsi": float(result["rsi"]),
            "volume_ratio": float(result["volume_ratio"]),
            "return_5d": float(result["return_5d"]),
            "ai_probability": float(result["ai_probability"]),
            "prediction": int(result["prediction"])
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

# ==========================================
# 주가 차트 데이터 API
# ==========================================


@app.get("/api/stock/{ticker}/chart")
def get_stock_chart(ticker: str):

    try:
        import yfinance as yf

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

        chart_data = []

        for _, row in data.iterrows():

            chart_data.append({
                "date": row["Date"].strftime("%Y-%m-%d"),
                "close": float(row["Close"]),
                "volume": int(row["Volume"])
            })

        return {
            "ticker": ticker,
            "data": chart_data
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )