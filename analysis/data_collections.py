import yfinance as yf
import pandas as pd
import time
import requests
from io import StringIO


# ========================================
# 설정
# ========================================

STOCK_LIST_URL = (
    "https://aikstockdata.com/data/public/quotes.csv"
)

OUTPUT_FILE = "stock_data_500.csv"

# KOSPI 250 + KOSDAQ 250
KOSPI_COUNT = 250
KOSDAQ_COUNT = 250


# ========================================
# 1. 국내 전체 종목 목록 가져오기
# ========================================

print("================================")
print("국내 주식 종목 목록 불러오는 중")
print("================================")


def get_stock_list():

    try:

        headers = {
            "User-Agent": "stock-analysis-web"
        }

        response = requests.get(
            STOCK_LIST_URL,
            headers=headers,
            timeout=30
        )

        response.raise_for_status()

        data = pd.read_csv(
            StringIO(response.text),
            dtype={
                "종목코드": str
            }
        )

        return data

    except Exception as e:

        print(
            f"종목 목록 가져오기 실패: {e}"
        )

        return pd.DataFrame()


stock_list = get_stock_list()


# ========================================
# 2. 종목 목록 확인
# ========================================

if stock_list.empty:

    print()
    print("종목 목록을 가져오지 못했습니다.")
    print("프로그램을 종료합니다.")

    exit()


print()
print(
    f"전체 종목 : "
    f"{len(stock_list):,}개"
)

print()

print("컬럼 확인")
print(
    stock_list.columns.tolist()
)


# ========================================
# 3. KOSPI / KOSDAQ만 선택
# ========================================

stock_list = stock_list[
    stock_list["mrktCtg"].isin(
        ["KOSPI", "KOSDAQ"]
    )
].copy()


print()
print("시장별 종목 수")
print("--------------------------------")

print(
    "KOSPI :",
    len(
        stock_list[
            stock_list["mrktCtg"] == "KOSPI"
        ]
    )
)

print(
    "KOSDAQ :",
    len(
        stock_list[
            stock_list["mrktCtg"] == "KOSDAQ"
        ]
    )
)


# ========================================
# 4. 종목코드 정리
# ========================================

stock_list["종목코드"] = (
    stock_list["종목코드"]
    .astype(str)
    .str.zfill(6)
)


# ========================================
# 5. 시가총액 숫자로 변환
# ========================================

stock_list["mrktTotAmt"] = (
    stock_list["mrktTotAmt"]
    .astype(str)
    .str.replace(",", "", regex=False)
)

stock_list["mrktTotAmt"] = pd.to_numeric(
    stock_list["mrktTotAmt"],
    errors="coerce"
)


# 시가총액이 없는 종목 제거
stock_list = stock_list.dropna(
    subset=["mrktTotAmt"]
)


# ========================================
# 6. KOSPI / KOSDAQ 분리
# ========================================

kospi = stock_list[
    stock_list["mrktCtg"] == "KOSPI"
].copy()

kosdaq = stock_list[
    stock_list["mrktCtg"] == "KOSDAQ"
].copy()


# ========================================
# 7. 시가총액 기준 내림차순 정렬
# ========================================

kospi = kospi.sort_values(
    "mrktTotAmt",
    ascending=False
)

kosdaq = kosdaq.sort_values(
    "mrktTotAmt",
    ascending=False
)


# ========================================
# 8. KOSPI 250 + KOSDAQ 250
# ========================================

kospi = kospi.head(
    KOSPI_COUNT
)

kosdaq = kosdaq.head(
    KOSDAQ_COUNT
)


# ========================================
# 9. 하나로 합치기
# ========================================

selected_stocks = pd.concat(
    [
        kospi,
        kosdaq
    ],
    ignore_index=True
)


# ========================================
# 10. Yahoo Finance 티커 생성
# ========================================

selected_stocks["Ticker"] = ""


kospi_mask = (
    selected_stocks["mrktCtg"] == "KOSPI"
)

kosdaq_mask = (
    selected_stocks["mrktCtg"] == "KOSDAQ"
)


selected_stocks.loc[
    kospi_mask,
    "Ticker"
] = (
    selected_stocks.loc[
        kospi_mask,
        "종목코드"
    ] + ".KS"
)


selected_stocks.loc[
    kosdaq_mask,
    "Ticker"
] = (
    selected_stocks.loc[
        kosdaq_mask,
        "종목코드"
    ] + ".KQ"
)


# ========================================
# 11. 종목명 + 티커 딕셔너리
# ========================================

stock_codes = {}

for _, row in selected_stocks.iterrows():

    stock_name = row["종목명"]

    ticker = row["Ticker"]

    stock_codes[
        stock_name
    ] = ticker


# ========================================
# 12. 종목 목록 확인
# ========================================

print()
print("================================")
print("종목 목록 생성 완료")
print("================================")

print(
    f"KOSPI 사용 : "
    f"{len(kospi)}개"
)

print(
    f"KOSDAQ 사용 : "
    f"{len(kosdaq)}개"
)

print(
    f"총 사용 종목 : "
    f"{len(stock_codes)}개"
)


print()
print("사용 종목 예시")
print("--------------------------------")


for name, ticker in list(
    stock_codes.items()
)[:20]:

    print(
        f"{name} : {ticker}"
    )


# ========================================
# 13. 개별 종목 데이터 수집
# ========================================

def collect_stock_data(
    stock_name,
    ticker
):

    print(
        f"{stock_name} "
        f"({ticker}) 데이터 수집 중..."
    )

    try:

        data = yf.download(
            ticker,
            period="5y",
            interval="1d",
            auto_adjust=False,
            progress=False
        )


        # 데이터가 없는 경우
        if data.empty:

            print(
                f"{stock_name} : "
                f"데이터 없음"
            )

            return None


        # ========================================
        # yfinance MultiIndex 처리
        # ========================================

        if isinstance(
            data.columns,
            pd.MultiIndex
        ):

            data.columns = (
                data.columns
                .get_level_values(0)
            )


        # ========================================
        # Index → Date
        # ========================================

        data = data.reset_index()


        # ========================================
        # 종목 정보 추가
        # ========================================

        data["Stock_Name"] = (
            stock_name
        )

        data["Ticker"] = (
            ticker
        )


        return data


    except Exception as e:

        print(
            f"{stock_name} 오류 : "
            f"{e}"
        )

        return None


# ========================================
# 14. 전체 종목 데이터 수집
# ========================================

if __name__ == "__main__":

    all_data = []

    total = len(
        stock_codes
    )

    success_count = 0

    fail_count = 0


    for i, (
        stock_name,
        ticker
    ) in enumerate(
        stock_codes.items(),
        start=1
    ):

        print()
        print(
            f"[{i}/{total}]"
        )


        data = collect_stock_data(
            stock_name,
            ticker
        )


        if data is not None:

            all_data.append(
                data
            )

            success_count += 1

        else:

            fail_count += 1


        # ========================================
        # Yahoo Finance 요청 간격
        # ========================================

        time.sleep(1)


    # ========================================
    # 15. 데이터 저장
    # ========================================

    if all_data:

        result = pd.concat(
            all_data,
            ignore_index=True
        )


        result.to_csv(
            OUTPUT_FILE,
            index=False,
            encoding="utf-8-sig"
        )


        # ========================================
        # 16. 결과 출력
        # ========================================

        print()

        print("================================")
        print("데이터 수집 완료")
        print("================================")

        print(
            f"수집 성공 종목 : "
            f"{result['Stock_Name'].nunique()}개"
        )

        print(
            f"수집 실패 종목 : "
            f"{fail_count}개"
        )

        print(
            f"전체 데이터 : "
            f"{len(result):,}개"
        )

        print(
            f"파일 : "
            f"{OUTPUT_FILE}"
        )


    else:

        print()

        print(
            "수집된 데이터가 없습니다."
        )