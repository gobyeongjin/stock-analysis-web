import json
import os
import threading
import time
from pathlib import Path
from urllib.request import Request, urlopen

AI_DATA_URL = os.getenv(
    "AI_DATA_URL",
    "https://raw.githubusercontent.com/"
    "gobyeongjin/stock-analysis-web/"
    "ai-data/ai_top10.json"
)

LOCAL_JSON = Path(__file__).resolve().parent / "ai_top10.json"
CACHE_SECONDS = 300

_cache = None
_cache_time = 0.0
_lock = threading.Lock()


def validate_data(data):
    if not isinstance(data, dict):
        raise ValueError("AI 데이터 형식 오류")

    if not isinstance(data.get("top10"), list):
        raise ValueError("TOP10 데이터 누락")

    if len(data["top10"]) != 10:
        raise ValueError("TOP10 종목 수 오류")

    if not isinstance(data.get("all_stocks"), list):
        raise ValueError("전체 종목 데이터 누락")

    if not data["all_stocks"]:
        raise ValueError("전체 종목 데이터가 비어 있습니다")

    for key in ("as_of", "model", "target", "note"):
        if key not in data:
            raise ValueError(f"필수 항목 누락: {key}")

    return data


def load_ai_data():
    global _cache, _cache_time

    with _lock:
        now = time.monotonic()

        if _cache is not None and now - _cache_time < CACHE_SECONDS:
            return _cache

        try:
            request = Request(
                AI_DATA_URL,
                headers={"User-Agent": "stock-analysis-web"}
            )

            with urlopen(request, timeout=5) as response:
                data = json.load(response)

            data = validate_data(data)

            # 이전 정상 데이터보다 오래된 데이터는 사용하지 않음
            if _cache is not None:
                if data["as_of"] < _cache["as_of"]:
                    raise ValueError("이전 기준일의 데이터")

            _cache = data
            _cache_time = time.monotonic()
            return data

        except Exception as exc:
            print(f"AI 데이터 원격 조회 실패: {exc}")

            if _cache is not None:
                _cache_time = time.monotonic()
                return _cache

            if LOCAL_JSON.exists():
                with LOCAL_JSON.open("r", encoding="utf-8") as f:
                    data = validate_data(json.load(f))

                _cache = data
                _cache_time = time.monotonic()
                return data

            raise RuntimeError("사용 가능한 AI 추천 데이터가 없습니다")
