
import os
from pathlib import Path

import requests
from dotenv import load_dotenv


# 프로젝트 최상위 폴더
BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")


class KiwoomAPI:

    def __init__(self):

        self.app_key = os.getenv("KIWOOM_MOCK_APP_KEY")
        self.secret_key = os.getenv("KIWOOM_MOCK_SECRET_KEY")
        if not self.app_key or not self.secret_key:
            raise ValueError(
                ".env에서 키움 API 인증키를 읽지 못했습니다."
            )

        # 모의투자 서버로 고정
        self.base_url = "https://mockapi.kiwoom.com"

        self.token = None

    def get_access_token(self):

        response = requests.post(
            f"{self.base_url}/oauth2/token",
            headers={
                "Content-Type": "application/json;charset=UTF-8"
            },
            json={
                "grant_type": "client_credentials",
                "appkey": self.app_key,
                "secretkey": self.secret_key,
            },
            timeout=30,
        )

        response.raise_for_status()

        data = response.json()

        if str(data.get("return_code")) != "0":
            raise RuntimeError(
                f"토큰 발급 실패: {data.get('return_msg')}"
            )

        self.token = data.get("token")

        if not self.token:
            raise RuntimeError(
                "접근 토큰이 반환되지 않았습니다."
            )

        return True

    def request(self, api_id, endpoint, params):

        if not self.token:
            self.get_access_token()

        headers = {
            "Content-Type": "application/json;charset=UTF-8",
            "authorization": f"Bearer {self.token}",
            "api-id": api_id,
        }

        response = requests.post(
            f"{self.base_url}{endpoint}",
            headers=headers,
            json=params,
            timeout=30,
        )

        response.raise_for_status()

        data = response.json()

        if str(data.get("return_code", 0)) != "0":
            raise RuntimeError(
                f"API 요청 실패: {data.get('return_msg')}"
            )

        return data
