
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

STOCK_FILE = BASE_DIR / "analysis" / "stock_data_500.csv"
FOREIGN_FILE = BASE_DIR / "foreign_trend_005930.csv"
OUTPUT_FILE = BASE_DIR / "analysis" / "stock_data_with_foreign.csv"


def main():
    # 파일 읽기
    stocks = pd.read_csv(STOCK_FILE)
    foreign = pd.read_csv(FOREIGN_FILE, dtype={"date": str})

    print("기존 주가 데이터:", stocks.shape)
    print("외국인 데이터:", foreign.shape)

    # 날짜 형식 통일
    stocks["Date"] = (
        pd.to_datetime(stocks["Date"], errors="coerce")
        .dt.strftime("%Y%m%d")
    )

    foreign["date"] = (
        foreign["date"]
        .astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.strip()
        .str.zfill(8)
    )

    # 컬럼명 변경
    foreign = foreign.rename(columns={
        "date": "Date",
        "change_qty": "Foreign_Change_Qty",
        "holding_qty": "Foreign_Holding_Qty",
        "holding_ratio": "Foreign_Holding_Ratio",
    })

    new_columns = [
        "Foreign_Change_Qty",
        "Foreign_Holding_Qty",
        "Foreign_Holding_Ratio",
    ]

    # 숫자 변환
    for col in new_columns:
        foreign[col] = pd.to_numeric(
            foreign[col].astype(str).str.replace(",", "", regex=False),
            errors="coerce",
        )

    # 날짜별 중복 제거
    foreign = foreign[["Date"] + new_columns].drop_duplicates(
        subset=["Date"]
    )

    # 기존 파일에 같은 이름의 컬럼이 있다면 제거
    stocks = stocks.drop(
        columns=[col for col in new_columns if col in stocks.columns]
    )

    # 결과 컬럼을 빈 값으로 초기화
    for col in new_columns:
        stocks[col] = float("nan")

    # 삼성전자 행만 선택
    mask = stocks["Ticker"].eq("005930.KS")

    # 삼성전자 데이터와 외국인 데이터를 날짜 기준으로 결합
    samsung = stocks.loc[mask, ["Date"]].merge(
        foreign,
        on="Date",
        how="left",
        validate="many_to_one",
        sort=False,
    )

    # 결합 결과 반영
    stocks.loc[mask, new_columns] = samsung[new_columns].to_numpy()

    # 새 파일 저장
    stocks.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    # 결과 확인
    matched = stocks.loc[mask, "Foreign_Holding_Qty"].notna().sum()

    print("\n===== 결합 결과 =====")
    print("저장 파일:", OUTPUT_FILE)
    print("전체 행 수:", len(stocks))
    print("삼성전자 행 수:", int(mask.sum()))
    print("매칭된 행 수:", int(matched))
    print("매칭되지 않은 행 수:", int(mask.sum() - matched))
    print("기존 전체 행 수 유지:", len(stocks) == 574586)

    print("\n===== 삼성전자 최근 데이터 =====")
    print(
        stocks.loc[
            mask,
            ["Date", "Close"] + new_columns,
        ].tail(10).to_string(index=False)
    )


if __name__ == "__main__":
    main()
