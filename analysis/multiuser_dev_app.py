from multiuser_stock_api import router as stock_router
from stock_detail_api import router as stock_detail_router
from pathlib import Path
import json

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from multiuser_auth_api import router as auth_router
from multiuser_leaderboard_api import router as leaderboard_router
from multiuser_profile_api import router as profile_router
from multiuser_portfolio_api import router as portfolio_router
from multiuser_portfolio_live import router as live_router

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(title="Multi-user Portfolio Development")

app.include_router(auth_router)
app.include_router(leaderboard_router)
app.include_router(profile_router)
app.include_router(portfolio_router)
app.include_router(live_router)


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(
        BASE_DIR / "frontend" / "index.html"
    )




@app.get("/api/ai/top10")
def get_ai_top10():
    path = BASE_DIR / "ai_top10.json"

    if not path.exists():
        raise HTTPException(
            status_code=503,
            detail="AI 추천 결과가 없습니다."
        )

    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        return {
            "as_of": data["as_of"],
            "model": data["model"],
            "target": data["target"],
            "count": len(data["top10"]),
            "stocks": data["top10"],
            "note": data["note"]
        }

    except (json.JSONDecodeError, KeyError, TypeError) as e:
        raise HTTPException(
            status_code=500,
            detail="AI 추천 데이터 형식이 올바르지 않습니다."
        )

@app.get("/portfolio", include_in_schema=False)
def portfolio_page():
    return FileResponse(
        BASE_DIR / "frontend" / "portfolio.html"
    )


@app.get("/login", include_in_schema=False)
def login_page():
    return FileResponse(
        BASE_DIR / "frontend" / "login.html"
    )

app.include_router(stock_router)


app.include_router(stock_detail_router)
