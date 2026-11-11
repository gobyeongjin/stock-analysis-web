import os
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel, Field
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired

from multiuser_auth import register_user, authenticate_user
from multiuser_db import get_connection

router = APIRouter(prefix="/api/auth", tags=["Authentication"])

COOKIE_NAME = "stock_session"
SESSION_MAX_AGE = 60 * 60 * 24 * 7


def get_secret():
    secret = os.getenv("SESSION_SECRET")

    if not secret and not os.getenv("RENDER"):
        path = Path(__file__).resolve().parent / ".env.multiuser.local"

        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.startswith("SESSION_SECRET="):
                    secret = line.split("=", 1)[1]
                    break

    if not secret or len(secret) < 32:
        raise RuntimeError("SESSION_SECRET 설정이 필요합니다.")

    return secret


serializer = URLSafeTimedSerializer(
    get_secret(),
    salt="stock-analysis-session-v1"
)


class RegisterRequest(BaseModel):
    email: str = Field(min_length=3, max_length=254)
    password: str = Field(min_length=12, max_length=128)


class LoginRequest(BaseModel):
    email: str
    password: str


def get_current_user(request: Request):
    token = request.cookies.get(COOKIE_NAME)

    if not token:
        raise HTTPException(status_code=401, detail="로그인이 필요합니다.")

    try:
        data = serializer.loads(token, max_age=SESSION_MAX_AGE)
        user_id = int(data["user_id"])
    except (BadSignature, SignatureExpired, KeyError, ValueError, TypeError):
        raise HTTPException(status_code=401, detail="로그인이 만료되었거나 유효하지 않습니다.")

    with get_connection() as conn:
        user = conn.execute(
            "SELECT id, email FROM users WHERE id = ?",
            (user_id,)
        ).fetchone()

    if user is None:
        raise HTTPException(status_code=401, detail="사용자를 찾을 수 없습니다.")

    return dict(user)


@router.post("/register", status_code=201)
def register(payload: RegisterRequest):
    try:
        user_id = register_user(payload.email, payload.password)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {"message": "회원가입 완료", "user_id": user_id}


@router.post("/login")
def login(payload: LoginRequest, response: Response):
    user = authenticate_user(payload.email, payload.password)

    if user is None:
        raise HTTPException(status_code=401, detail="이메일 또는 비밀번호가 올바르지 않습니다.")

    token = serializer.dumps({"user_id": user["id"]})

    response.set_cookie(
        key=COOKIE_NAME,
        value=token,
        httponly=True,
        secure=bool(os.getenv("RENDER")),
        samesite="lax",
        max_age=SESSION_MAX_AGE,
        path="/",
    )

    return {"message": "로그인 성공", "user": user}


@router.get("/me")
def me(request: Request):
    return {"user": get_current_user(request)}


@router.post("/logout")
def logout(response: Response):
    response.delete_cookie(
        key=COOKIE_NAME,
        path="/",
        samesite="lax",
        secure=bool(os.getenv("RENDER")),
        httponly=True,
    )
    return {"message": "로그아웃 완료"}
