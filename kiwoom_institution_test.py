
import os
import json
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

APP_KEY = os.getenv("KIWOOM_APP_KEY")
SECRET_KEY = os.getenv("KIWOOM_SECRET_KEY")
BASE_URL = "https://api.kiwoom.com"


def get_access_token():
    url = f"{BASE_URL}/oauth2/token"
    headers = {
        "Content-Type": "application/json;charset=UTF-8"
    }
    body = {
        "grant_type": "client_credentials",
        "appkey": APP_KEY,
        "secretkey": SECRET_KEY,
    }

    response = requests.post(
        url, headers=headers, json=body, timeout=30
    )
    response.raise_for_status()
    result = response.json()

    token = result.get("token") or result.get("access_token")
    if not token:
        raise RuntimeError(
            f"토큰 발급 실패: {json.dumps(result, ensure_ascii=False)}"
        )

    return token


def main():
    if not APP_KEY or not SECRET_KEY:
        raise RuntimeError(".env 인증 정보를 확인하세요.")

    token = get_access_token()

    url = f"{BASE_URL}/api/dostk/frgnistt"
    headers = {
        "Content-Type": "application/json;charset=UTF-8",
        "authorization": f"Bearer {token}",
        "cont-yn": "N",
        "next-key": "",
        "api-id": "ka10131",
    }

    # 최근 1일, 코스피, 순매수, 종목 기준, 금액 기준, KRX
    body = {
    "dt": "0",
    "strt_dt": "20261001",
    "end_dt": "20261002",
    "mrkt_tp": "001",
    "netslmt_tp": "2",
    "stk_inds_tp": "0",
    "amt_qty_tp": "1",
    "stex_tp": "1",
}

    response = requests.post(
        url, headers=headers, json=body, timeout=30
    )
    response.raise_for_status()
    result = response.json()

    if result.get("return_code") != 0:
        raise RuntimeError(
            f"API 오류: {result.get('return_msg')}"
        )

    rows = result.get("orgn_frgnr_cont_trde_prst", [])
    print("응답 코드:", result.get("return_code"))
    print("응답 메시지:", result.get("return_msg"))
    print("응답 데이터 건수:", len(rows))

    for row in rows[:10]:
        print(
            "종목코드:", repr(row.get("stk_cd")),
            "| 종목명:", row.get("stk_nm"),
            "| 기관 순매매금액:", row.get("orgn_nettrde_amt"),
            "| 외국인 순매매금액:", row.get("frgnr_nettrde_amt")
        )

    # 원본 응답 저장
    with open(
        "institution_api_test.json", "w", encoding="utf-8"
    ) as file:
        json.dump(result, file, ensure_ascii=False, indent=2)

    if not rows:
        print("응답은 성공했지만 데이터 목록이 비어 있습니다.")
        return

    df = pd.DataFrame(rows)

    # 삼성전자만 추출
    df = df[df["stk_cd"].astype(str).str.strip() == "005930"].copy()

    if df.empty:
        print("응답 목록에 삼성전자 데이터가 없습니다.")
        return

    # 숫자 변환: +, - 부호를 유지
    numeric_cols = [
        "prid_stkpc_flu_rt",
        "orgn_nettrde_amt",
        "orgn_nettrde_qty",
        "orgn_cont_netprps_dys",
        "orgn_cont_netprps_qty",
        "orgn_cont_netprps_amt",
        "frgnr_nettrde_qty",
        "frgnr_nettrde_amt",
        "frgnr_cont_netprps_dys",
        "frgnr_cont_netprps_qty",
        "frgnr_cont_netprps_amt",
        "nettrde_qty",
        "nettrde_amt",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df.to_csv(
        "institution_trend_005930.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("응답 건수:", len(df))
    print("CSV 저장: institution_trend_005930.csv")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
