
from kiwoom_api import KiwoomAPI


def to_int(value):
    """키움 API 금액 문자열을 정수로 변환"""
    if value is None or str(value).strip() == "":
        return None

    return int(str(value).replace(",", "").strip())


def main():

    api = KiwoomAPI()

    api.get_access_token()

    print("키움 모의투자 인증 성공")
    print("서버:", api.base_url)

    # kt00001: 예수금상세현황요청
    data = api.request(
        api_id="kt00001",
        endpoint="/api/dostk/acnt",
        params={
            "qry_tp": "2"  # 일반조회
        }
    )

    print("\n===== 모의투자 계좌 예수금 =====")

    fields = {
        "entr": "예수금",
        "ord_alow_amt": "주문가능금액",
        "pymn_alow_amt": "출금가능금액",
        "d1_entra": "D+1 추정예수금",
        "d2_entra": "D+2 추정예수금",
    }

    for key, label in fields.items():

        amount = to_int(data.get(key))

        if amount is None:
            print(f"{label}: 응답값 없음")
        else:
            print(f"{label}: {amount:,}원")


if __name__ == "__main__":
    main()
