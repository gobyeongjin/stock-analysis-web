import os
import threading
import time
from io import BytesIO
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

STOCK_DATA_URL = os.getenv(
    "STOCK_DATA_URL",
    "https://raw.githubusercontent.com/"
    "gobyeongjin/stock-analysis-web/"
    "ai-data/stock_data_500_recent.csv"
)

LOCAL_CSV = BASE_DIR / "stock_data_deploy.csv"

CACHE_SECONDS = 300

_cache = None
_cache_time = 0.0
_lock = threading.Lock()

REQUIRED_COLUMNS = [
    "Date", "Open", "High", "Low",
    "Close", "Volume", "Stock_Name", "Ticker"
]


def validate_stock_data(df):

    missing = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing:
        raise ValueError(f"주가 데이터 컬럼 누락: {missing}")

    if df.empty:
        raise ValueError("주가 데이터가 비어 있습니다.")

    df = df.copy()

    df["Date"] = pd.to_datetime(
        df["Date"], errors="coerce"
    )

    df["Ticker"] = df["Ticker"].astype(str)

    df = df.dropna(
        subset=["Date", "Open", "High", "Low", "Close"]
    )

    if df.empty:
        raise ValueError("유효한 주가 데이터가 없습니다.")

    return (
        df.sort_values(["Ticker", "Date"])
        .drop_duplicates(
            ["Ticker", "Date"],
            keep="last"
        )
        .reset_index(drop=True)
    )


def load_stock_data():

    global _cache, _cache_time

    with _lock:

        now = time.monotonic()

        if (
            _cache is not None
            and now - _cache_time < CACHE_SECONDS
        ):
            return _cache

        try:
            request = Request(
                STOCK_DATA_URL,
                headers={
                    "User-Agent": "stock-analysis-web"
                }
            )

            with urlopen(request, timeout=20) as response:
                content = response.read()

            df = pd.read_csv(BytesIO(content))
            df = validate_stock_data(df)

            _cache = df
            _cache_time = time.monotonic()

            print(
                f"최신 주가 데이터 원격 조회 성공: "
                f"{len(df)}행"
            )

            return df

        except Exception as exc:

            print(f"최신 주가 데이터 원격 조회 실패: {exc}")

            if _cache is not None:
                _cache_time = time.monotonic()
                return _cache

            if LOCAL_CSV.exists():

                df = pd.read_csv(LOCAL_CSV)
                df = validate_stock_data(df)

                _cache = df
                _cache_time = time.monotonic()

                print("기존 배포용 CSV 사용")

                return df

            raise RuntimeError(
                "사용 가능한 주가 데이터가 없습니다."
            )
