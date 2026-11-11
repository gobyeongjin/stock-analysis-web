import json
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

BASE = Path(__file__).resolve().parent
FILE = BASE / "ai_top10_recent_test.json"

data = json.loads(FILE.read_text(encoding="utf-8"))

as_of = datetime.strptime(data["as_of"], "%Y-%m-%d").date()
today = datetime.now(ZoneInfo("Asia/Seoul")).date()

# 평일 기준으로 최근 거래일이 지나치게 오래되지 않았는지 확인
business_days = 0
day = as_of

while day < today:
    day += timedelta(days=1)
    if day.weekday() < 5:
        business_days += 1

if as_of > today:
    raise ValueError("예측 기준일이 미래입니다.")

if business_days > 3:
    raise ValueError(
        f"주가 데이터가 오래됐습니다: {as_of}, "
        f"경과 평일 {business_days}일"
    )

top10 = data.get("top10", [])

if len(top10) != 10:
    raise ValueError(f"TOP10 종목 수 오류: {len(top10)}")

tickers = [stock["ticker"] for stock in top10]

if len(set(tickers)) != 10:
    raise ValueError("TOP10에 중복 종목이 있습니다.")

print("AI TOP10 검증 성공")
print("예측 기준일:", as_of)
print("TOP10 종목 수:", len(top10))
print("경과 평일:", business_days)
