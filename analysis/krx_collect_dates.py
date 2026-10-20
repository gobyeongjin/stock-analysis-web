
import pandas as pd
from pykrx_openapi import KRXOpenAPI

client = KRXOpenAPI()

DATES = ["20261006", "20261007", "20261008"]

COLUMN_MAP = {
    "BAS_DD": "Date",
    "ISU_CD": "Ticker",
    "ISU_NM": "Stock_Name",
    "MKT_NM": "Market",
    "TDD_CLSPRC": "Close",
    "TDD_OPNPRC": "Open",
    "TDD_HGPRC": "High",
    "TDD_LWPRC": "Low",
    "ACC_TRDVOL": "Volume",
    "ACC_TRDVAL": "Trading_Value",
    "MKTCAP": "Market_Cap",
    "LIST_SHRS": "Listed_Shares",
}


def get_rows(result):
    if isinstance(result, dict):
        return result.get("OutBlock_1", [])
    if isinstance(result, list):
        return result
    raise TypeError(f"예상하지 못한 응답 형식: {type(result)}")


def main():
    all_rows = []

    for date in DATES:
        for market, method in [
            ("KOSPI", client.get_stock_daily_trade),
            ("KOSDAQ", client.get_kosdaq_stock_daily_trade),
        ]:
            try:
                result = method(date)
                rows = get_rows(result)

                for row in rows:
                    row["Requested_Market"] = market
                    all_rows.append(row)

                print(f"{date} {market}: {len(rows)}행")

            except Exception as e:
                print(f"{date} {market} 오류: {e}")

    if not all_rows:
        print("수집된 데이터가 없습니다.")
        return

    df = pd.DataFrame(all_rows).rename(columns=COLUMN_MAP)

    df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
    df["Ticker"] = df["Ticker"].astype(str).str.zfill(6)

    output_columns = [
        "Date", "Ticker", "Stock_Name", "Market",
        "Open", "High", "Low", "Close", "Volume",
        "Trading_Value", "Market_Cap", "Listed_Shares",
    ]
    available = [c for c in output_columns if c in df.columns]

    df = df[available].drop_duplicates(["Date", "Ticker"])
    df = df.sort_values(["Date", "Ticker"])

    output_file = "krx_stock_dates_test.csv"
    df.to_csv(output_file, index=False, encoding="utf-8-sig")

    print(f"\n저장 완료: {output_file}")
    print(f"전체 행 수: {len(df):,}")
    print("\n날짜별 행 수:")
    print(df.groupby("Date").size().to_string())


if __name__ == "__main__":
    main()
