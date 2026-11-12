import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from predict import predict_stock

import pandas as pd


# ==========================================
# 로컬 / Render 데이터 경로
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

STOCK_CSV_PATH = os.path.join(
    BASE_DIR,
    "stock_data_deploy.csv"
    if os.getenv("RENDER")
    else "stock_data_500.csv"
)

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
        CSV_PATH = STOCK_CSV_PATH

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
        stocks = pd.read_csv(STOCK_CSV_PATH)

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

from screener_api import router as screener_router
app.include_router(screener_router)

from screener_api import pages_router
app.include_router(pages_router)

# ==========================================
# AI 추천 TOP 10 API
# ==========================================

import json
from pathlib import Path


AI_DATA_URL = (
    "https://raw.githubusercontent.com/"
    "gobyeongjin/stock-analysis-web/"
    "ai-data/ai_top10.json"
)

def load_latest_ai_data():
    import urllib.request

    local_path = Path(__file__).resolve().parent / "ai_top10.json"

    try:
        request = urllib.request.Request(
            AI_DATA_URL,
            headers={"User-Agent": "stock-analysis-web"}
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            payload = json.load(response)

        if not isinstance(payload.get("top10"), list):
            raise ValueError("잘못된 AI 데이터 형식")

        if not isinstance(payload.get("all_stocks"), list):
            raise ValueError("전체 종목 데이터가 없습니다.")

        return payload

    except Exception as exc:
        print(f"[AI TOP10] GitHub 데이터 요청 실패: {exc}")

        if local_path.exists():
            with local_path.open("r", encoding="utf-8") as f:
                return json.load(f)

        raise HTTPException(
            status_code=503,
            detail="AI 추천 데이터를 불러올 수 없습니다."
        )


@app.get("/api/ai/top10")
def get_ai_top10():
    data = load_latest_ai_data()

    return {
        "as_of": data["as_of"],
        "model": data["model"],
        "target": data["target"],
        "count": len(data["top10"]),
        "stocks": data["top10"],
        "note": data["note"]
    }

@app.get("/ai-top10", include_in_schema=False)
def ai_top10_page():
    return FileResponse(
        os.path.join(
            os.path.dirname(__file__),
            "frontend",
            "ai_top10.html"
        )
    )



# ==========================================
# 18개 특징 AI 모델 - 개별 종목 검색
# ==========================================

@app.get("/api/ai/search")
def search_ai_stock(q: str = ""):
    payload = load_latest_ai_data()

    query = q.strip().lower()

    if not query:
        raise HTTPException(
            status_code=400,
            detail="종목명 또는 종목코드를 입력해 주세요."
        )

    stocks = payload.get("all_stocks", [])

    # 정확히 일치하는 종목 우선
    exact = [
        s for s in stocks
        if s["name"].lower() == query
        or s["ticker"].lower() == query
        or s["ticker"].split(".")[0].lower() == query
    ]

    # 일부만 입력해도 검색
    partial = [
        s for s in stocks
        if query in s["name"].lower()
        or query in s["ticker"].lower()
    ]

    matches = exact if exact else partial

    if not matches:
        raise HTTPException(
            status_code=404,
            detail="검색 결과가 없습니다. 분석 대상 또는 거래대금 조건을 확인해 주세요."
        )

    results = []

    for stock in matches[:20]:
        results.append({
            "name": stock["name"],
            "ticker": stock["ticker"],
            "ai_score": round(
                stock["ai_probability"] * 100, 2
            ),
            "current_price": stock["close"],
            "rsi": stock["rsi"],
            "return_5d_pct": stock["return_5d_pct"],
            "volume_ratio": stock["volume_ratio"],
        })

    return {
        "as_of": payload.get("as_of"),
        "count": len(results),
        "stocks": results,
        "note": payload.get("note"),
    }


# ==========================================
# Multi-user Portfolio & Leaderboard
# ==========================================
ENABLE_MULTIUSER = os.getenv(
    "ENABLE_MULTIUSER", "false"
).lower() == "true"

if ENABLE_MULTIUSER:
    from multiuser_db import ensure_unique_nickname_index
    from multiuser_auth_api import router as auth_router
    from multiuser_leaderboard_api import router as leaderboard_router
    from multiuser_profile_api import router as profile_router
    from multiuser_portfolio_api import router as my_portfolio_router
    from multiuser_portfolio_live import router as my_portfolio_live_router

    ensure_unique_nickname_index()

    app.include_router(auth_router)
    app.include_router(leaderboard_router)
    app.include_router(profile_router)
    app.include_router(my_portfolio_router)
    app.include_router(my_portfolio_live_router)

    @app.get("/login", include_in_schema=False)
    def login_page():
        return FileResponse(
            str(Path(__file__).resolve().parent / "frontend" / "login.html")
        )

    @app.get("/portfolio", include_in_schema=False)
    def portfolio_page():
        return FileResponse(
            str(Path(__file__).resolve().parent / "frontend" / "portfolio.html")
        )


# 종목 상세 차트 (FastAPI)
from stock_detail_api import router as stock_detail_router
app.include_router(stock_detail_router)
