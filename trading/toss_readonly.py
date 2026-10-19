import os
import requests
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


class TossReadOnlyAPI:
    BASE_URL = "https://openapi.tossinvest.com"

    def __init__(self):
        self.client_id = os.getenv("TOSS_CLIENT_ID")
        self.client_secret = os.getenv("TOSS_CLIENT_SECRET")
        self.token = None

        if not self.client_id or not self.client_secret:
            raise ValueError(
                ".env에 TOSS_CLIENT_ID와 TOSS_CLIENT_SECRET을 설정하세요."
            )

    def authenticate(self):
        response = requests.post(
            f"{self.BASE_URL}/oauth2/token",
            data={
                "grant_type": "client_credentials",
                "client_id": self.client_id,
                "client_secret": self.client_secret,
            },
            timeout=15,
        )
        response.raise_for_status()

        data = response.json()
        self.token = data["access_token"]
        return True

    def _get(self, endpoint, account_seq=None):
        if not self.token:
            self.authenticate()

        headers = {
            "Authorization": f"Bearer {self.token}"
        }

        if account_seq is not None:
            headers["X-Tossinvest-Account"] = str(account_seq)

        response = requests.get(
            f"{self.BASE_URL}{endpoint}",
            headers=headers,
            timeout=15,
        )
        response.raise_for_status()
        return response.json()

    def get_accounts(self):
        return self._get("/api/v1/accounts")

    def get_holdings(self, account_seq):
        return self._get(
            "/api/v1/holdings",
            account_seq=account_seq,
        )
