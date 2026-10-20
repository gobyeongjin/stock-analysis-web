from kiwoom_api import KiwoomAPI


api = KiwoomAPI()

api.get_access_token()

print("키움 REST API 모의투자 서버 연결 성공!")
print("서버:", api.base_url)
print("토큰 발급:", "성공")