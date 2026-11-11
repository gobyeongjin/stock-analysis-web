from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException

from predict import predict_stock

router = APIRouter()
BASE_DIR = Path(__file__).resolve().parent


@router.get("/api/stocks")
def get_stocks():
    csv_path = BASE_DIR / "stock_data_500.csv"

    if not csv_path.exists():
        raise HTTPException(
            status_code=503,
            detail="종목 데이터 파일이 없습니다."
        )

    try:
        data = pd.read_csv(
            csv_path,
            usecols=["Stock_Name", "Ticker"]
        )

        stocks = (
            data.dropna()
            .drop_duplicates()
            .sort_values("Stock_Name")
        )

        return [
            {
                "name": row.Stock_Name,
                "ticker": row.Ticker
            }
            for row in stocks.itertuples(index=False)
        ]

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="종목 목록 조회에 실패했습니다."
        )


@router.get("/api/stock/{ticker}")
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

    except Exception:
        raise HTTPException(
            status_code=503,
            detail="종목 분석 데이터를 조회하지 못했습니다."
        )
