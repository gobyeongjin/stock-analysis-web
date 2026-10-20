
import os
import pandas as pd
from pykrx_openapi import KRXOpenAPI

# KRX API 클라이언트
client = KRXOpenAPI("7A28036F5273400681902AF6EB8A7E45A520368D")

# 조회 날짜: YYYYMMDD
BASE_DATE = "20261008"

# 출력 파일
OUTPUT_FILE = "krx_stock_sample.csv"


def collect_stocks():
    # 해당 날짜의 주식 일별매매정보 조회
    result = client.get_stock_daily_trade(BASE_DATE)

    # API 응답 구조 확인
    if isinstance(result, dict):
        rows = result.get("OutBlock_1", [])
    elif isinstance(result, list):
        rows = result
    else:
        raise ValueError(f"예상하지 못한 응답 형식: {type(result)}")

    if not rows:
        raise ValueError(
            "조회 결과가 비어 있습니다. 날짜와 API 응답 구조를 확인하세요."
        )

    df = pd.DataFrame(rows)

    # KRX 필드명을 기존 프로젝트의 컬럼명으로 변환
    column_map = {
        "BAS_DD": "Date",
        "ISU_CD": "Ticker",
        "ISU_NM": "Stock_Name",
        "TDD_CLSPRC": "Close",
        "TDD_OPNPRC": "Open",
        "TDD_HGPRC": "High",
        "TDD_LWPRC": "Low",
        "ACC_TRDVOL": "Volume",
        "ACC_TRDVAL": "Trading_Value",
        "MKTCAP": "Market_Cap",
        "LIST_SHRS": "Listed_Shares",
    }

    df = df.rename(columns=column_map)

    # 날짜 및 종목코드 정리
    df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
    df["Ticker"] = df["Ticker"].astype(str).str.zfill(6)

    # 우선 분석에 필요한 주요 컬럼만 저장
    wanted = [
        "Date", "Ticker", "Stock_Name",
        "Open", "High", "Low", "Close", "Volume",
        "Trading_Value", "Market_Cap",
    ]
    available = [col for col in wanted if col in df.columns]
    df = df[available]

    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print(f"저장 완료: {OUTPUT_FILE}")
    print(f"데이터 행 수: {len(df):,}")
    print(df.head(5).to_string(index=False))


if __name__ == "__main__":
    collect_stocks()
