
from kiwoom_api import KiwoomAPI


def to_int(value):
    """키움 API 숫자 문자열을 정수로 변환"""
    if value is None or str(value).strip() == "":
        return None

    return int(str(value).replace(",", "").strip())


def to_float(value):
    """키움 API 수익률 문자열을 실수로 변환"""
    if value is None or str(value).strip() == "":
        return None

    return float(str(value).replace(",", "").strip())


def format_money(value):
    amount = to_int(value)
    return f"{amount:,}원" if amount is not None else "응답값 없음"


def main():

    api = KiwoomAPI()
    api.get_access_token()

    print("키움 모의투자 인증 성공")
    print("서버:", api.base_url)

    # kt00018: 계좌평가잔고내역요청
    data = api.request(
        api_id="kt00018",
        endpoint="/api/dostk/acnt",
        params={
            "qry_tp": "2",          # 개별조회
            "dmst_stex_tp": "KRX"  # 모의투자 지원 거래소
        }
    )

    print("\n===== 계좌 전체 현황 =====")

    print("총매입금액:",
          format_money(data.get("tot_pur_amt")))

    print("총평가금액:",
          format_money(data.get("tot_evlt_amt")))

    print("총평가손익:",
          format_money(data.get("tot_evlt_pl")))

    profit_rate = to_float(data.get("tot_prft_rt"))

    print(
        "총수익률:",
        f"{profit_rate:.2f}%"
        if profit_rate is not None
        else "응답값 없음"
    )

    print(
        "추정예탁자산:",
        format_money(data.get("prsm_dpst_aset_amt"))
    )

    # 보유 종목 목록
    holdings = data.get("acnt_evlt_remn_indv_tot") or []

    print("\n===== 보유 종목 현황 =====")

    if not holdings:
        print("현재 보유 종목이 없습니다.")
        return

    for item in holdings:

        raw_code = str(item.get("stk_cd", ""))

        # A005930 -> 005930
        ticker = (
            raw_code[1:]
            if raw_code.startswith("A")
            else raw_code
        )

        name = item.get("stk_nm", "")

        quantity = to_int(item.get("rmnd_qty"))
        available_qty = to_int(
            item.get("trde_able_qty")
        )

        current_price = to_int(item.get("cur_prc"))
        profit = to_int(item.get("evltv_prft"))
        profit_rate = to_float(item.get("prft_rt"))

        print(f"\n종목명: {name}")
        print(f"종목코드: {ticker}")
        print(f"보유수량: {quantity}주")
        print(f"매매가능수량: {available_qty}주")

        print(
            "현재가:",
            f"{current_price:,}원"
            if current_price is not None
            else "응답값 없음"
        )

        print(
            "평가손익:",
            f"{profit:,}원"
            if profit is not None
            else "응답값 없음"
        )

        print(
            "수익률:",
            f"{profit_rate:.2f}%"
            if profit_rate is not None
            else "응답값 없음"
        )


if __name__ == "__main__":
    main()
