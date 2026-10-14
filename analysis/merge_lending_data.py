
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent

PRICE_FILE = BASE_DIR / "analysis" / "training_data_v3.csv"
LENDING_FILE = BASE_DIR / "lending_multi.csv"
OUTPUT_FILE = BASE_DIR / "analysis" / "training_data_v4_lending.csv"

# 원본 데이터 불러오기
price = pd.read_csv(PRICE_FILE, dtype={"Ticker": str})
lending = pd.read_csv(LENDING_FILE, dtype={"Ticker": str})

# 날짜 정규화
price["Date"] = pd.to_datetime(price["Date"], errors="coerce")
lending["dt"] = pd.to_datetime(lending["dt"], errors="coerce")

# 종목 코드 정규화: 005930.KS -> 005930
price["Ticker"] = (
    price["Ticker"].str.replace(r"\.KS$", "", regex=True).str.zfill(6)
)
lending["Ticker"] = lending["Ticker"].str.zfill(6)

# 대차잔고 숫자 변환
lending["rmnd"] = pd.to_numeric(lending["rmnd"], errors="coerce")
lending["Lending_Balance_Change_Pct"] = pd.to_numeric(
    lending["Lending_Balance_Change_Pct"], errors="coerce"
)

# 날짜와 종목 코드 기준으로 중복 제거
lending = lending.drop_duplicates(
    subset=["Ticker", "dt"], keep="last"
)

# 모든 종목의 2026-10-12 잔고가 0이므로
# 해당 날짜의 대차잔고 정보를 결합에서 제외
invalid_latest = (
    (lending["dt"] == pd.Timestamp("2026-10-12"))
    & (lending["rmnd"] == 0)
)
print("제외한 최신 0 잔고 행:", int(invalid_latest.sum()))
lending = lending.loc[~invalid_latest].copy()

# 필요한 열만 남김
lending = lending[
    [
        "Ticker",
        "dt",
        "rmnd",
        "Lending_Balance_Change_Pct",
    ]
]

# 당일 공개 시점이 확인되지 않았으므로,
# 우선 동일 날짜를 기준으로 결합하고 누수 방지 검토를 위한
# 원본 변화율 열을 별도로 유지
merged = price.merge(
    lending,
    left_on=["Ticker", "Date"],
    right_on=["Ticker", "dt"],
    how="left",
    validate="many_to_one",
)

merged = merged.drop(columns=["dt"])

merged.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

print("\n===== 결합 결과 =====")
print("기존 주가 데이터 행:", len(price))
print("결합 후 데이터 행:", len(merged))
print("대차잔고 변화율 보유 행:",
      int(merged["Lending_Balance_Change_Pct"].notna().sum()))
print("대차잔고 변화율 결측 행:",
      int(merged["Lending_Balance_Change_Pct"].isna().sum()))
print("포함된 종목 수:", merged.loc[
    merged["Lending_Balance_Change_Pct"].notna(), "Ticker"
].nunique())
print("저장 위치:", OUTPUT_FILE)
