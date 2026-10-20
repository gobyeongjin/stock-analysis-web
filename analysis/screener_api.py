"""FastAPI router for the rule-based screener. Run screener_build.py first."""
import json
import os

from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/api/screen", tags=["screener"])
BASE = os.path.dirname(os.path.abspath(__file__))
_cache = {}


def _load(name):
    path = os.path.join(BASE, name)
    if not os.path.exists(path):
        raise HTTPException(status_code=503, detail=f"{name} 없음: screener_build.py를 먼저 실행하세요")
    mtime = os.path.getmtime(path)
    if name not in _cache or _cache[name][0] != mtime:
        with open(path, encoding="utf-8") as f:
            _cache[name] = (mtime, json.load(f))
    return _cache[name][1]


NOTE = ("과거 통계는 해당 조건에 걸린 종목의 이후 수익률이며, 미래 수익을 보장하지 않습니다. "
        "수익은 다음 거래일 시가 매수 기준이고 거래비용 전 수치입니다.")


@router.get("/rules")
def list_rules():
    today, st = _load("screener_today.json"), _load("screener_stats.json")
    out = []
    for name, v in today["rules"].items():
        h5 = next((s for s in st["stats"] if s["rule"] == name and s["horizon_days"] == 5), None)
        out.append({"rule": name, "label": v["label"], "count": v["count"], "stats_5d": h5})
    return {"as_of": today["as_of"], "rules": out, "note": NOTE}


@router.get("/{rule}")
def screen(rule: str):
    today, st = _load("screener_today.json"), _load("screener_stats.json")
    if rule not in today["rules"]:
        raise HTTPException(status_code=404, detail=f"알 수 없는 조건: {rule}")
    v = today["rules"][rule]
    return {
        "as_of": today["as_of"], "rule": rule, "label": v["label"], "count": v["count"],
        "stocks": v["stocks"],
        "stats": [s for s in st["stats"] if s["rule"] in (rule, "ALL_LIQUID")],
        "cost_pct": st["cost_pct"], "data_range": st["data_range"], "note": NOTE,
    }


# ---- page route: serves frontend/screener.html at /screener
from fastapi.responses import FileResponse  # noqa: E402

pages_router = APIRouter()


@pages_router.get("/screener", include_in_schema=False)
def screener_page():
    return FileResponse(os.path.join(BASE, "frontend", "screener.html"))
