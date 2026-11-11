from pathlib import Path

from ai_data_loader import load_ai_data
from stock_data_loader import load_stock_data

import pandas as pd
from fastapi import APIRouter, HTTPException, Request
from fastapi.templating import Jinja2Templates

router = APIRouter()

BASE_DIR = Path(__file__).resolve().parent
TEMPLATES = Jinja2Templates(directory=str(BASE_DIR / "frontend"))

@router.get("/stock/{ticker}/", include_in_schema=False)
def stock_detail(request: Request, ticker: str):
    try:
        df = load_stock_data()
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"주가 데이터 조회 실패: {exc}"
        )

    stock = df.loc[df["Ticker"] == ticker].copy()

    if stock.empty:
        raise HTTPException(404, "종목을 찾을 수 없습니다.")

    stock = stock.sort_values("Date").drop_duplicates(
        "Date", keep="last"
    )
    stock = stock.dropna(
        subset=["Open", "High", "Low", "Close"]
    )

    if stock.empty:
        raise HTTPException(404, "유효한 주가 데이터가 없습니다.")

    close = stock["Close"]

    stock["MA20"] = close.rolling(20).mean()
    stock["MA60"] = close.rolling(60).mean()

    delta = close.diff()
    avg_gain = delta.clip(lower=0).rolling(14).mean()
    avg_loss = (-delta.clip(upper=0)).rolling(14).mean()

    rs = avg_gain / avg_loss.replace(0, float("nan"))
    stock["RSI"] = 100 - 100 / (1 + rs)

    stock.loc[
        (avg_loss == 0) & (avg_gain > 0), "RSI"
    ] = 100
    stock.loc[
        (avg_loss == 0) & (avg_gain == 0), "RSI"
    ] = 50

    recent = stock.tail(120)

    def number(value):
        return round(float(value), 4) if pd.notna(value) else None

    candles = [
        {
            "time": row.Date.strftime("%Y-%m-%d"),
            "open": number(row.Open),
            "high": number(row.High),
            "low": number(row.Low),
            "close": number(row.Close),
        }
        for row in recent.itertuples()
    ]

    volume = [
        {
            "time": row.Date.strftime("%Y-%m-%d"),
            "value": number(row.Volume),
            "color": "#f16d7a"
            if row.Close >= row.Open else "#57a4ef",
        }
        for row in recent.itertuples()
    ]

    ma20 = [
        {"time": row.Date.strftime("%Y-%m-%d"), "value": number(row.MA20)}
        for row in recent.itertuples()
        if pd.notna(row.MA20)
    ]

    ma60 = [
        {"time": row.Date.strftime("%Y-%m-%d"), "value": number(row.MA60)}
        for row in recent.itertuples()
        if pd.notna(row.MA60)
    ]

    rsi = [
        {"time": row.Date.strftime("%Y-%m-%d"), "value": number(row.RSI)}
        for row in recent.itertuples()
        if pd.notna(row.RSI)
    ]

    latest = stock.iloc[-1]
    prev = stock.iloc[-2] if len(stock) >= 2 else None

    change_pct = (
        (latest["Close"] / prev["Close"] - 1) * 100
        if prev is not None and prev["Close"]
        else None
    )

    ai_info = None

    try:
        predictions = load_ai_data()
        matched = next(
            (
                x for x in predictions.get("all_stocks", [])
                if x.get("ticker") == ticker
            ),
            None
        )

        if matched:
            ai_info = dict(matched)
            ai_info["as_of"] = predictions.get("as_of")
            score = ai_info.get("ai_probability")

            if score is None:
                score = ai_info.get("ai_score")
                ai_info["score_pct"] = round(float(score), 2) if score is not None else None
            else:
                ai_info["score_pct"] = round(float(score) * 100, 2)

    except Exception as exc:
        print(f"종목 상세 AI 데이터 조회 실패: {exc}")

    return TEMPLATES.TemplateResponse(
        request=request,
        name="stock_detail.html",
        context={
            "ticker": ticker,
            "name": latest["Stock_Name"],
            "close": number(latest["Close"]),
            "as_of": latest["Date"].strftime("%Y-%m-%d"),
            "change_pct": round(change_pct, 2)
            if change_pct is not None else None,
            "latest_rsi": number(latest["RSI"]),
            "ai_info": ai_info,
            "chart_data": {
                "candles": candles,
                "volume": volume,
                "ma20": ma20,
                "ma60": ma60,
                "rsi": rsi,
            },
        },
    )
