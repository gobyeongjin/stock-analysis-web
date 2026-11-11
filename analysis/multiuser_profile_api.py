import sqlite3
from psycopg import IntegrityError as PostgresIntegrityError

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field, ConfigDict

from multiuser_auth_api import get_current_user
from multiuser_db import get_connection

router = APIRouter(prefix="/api/my-profile", tags=["My Profile"])


class ProfileUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    nickname: str = Field(min_length=1, max_length=20)
    ranking_opt_in: bool = False


@router.get("")
def get_profile(request: Request):
    user = get_current_user(request)

    with get_connection() as conn:
        row = conn.execute(
            """
            SELECT nickname, ranking_opt_in
            FROM users
            WHERE id = ?
            """,
            (user["id"],)
        ).fetchone()

    if row is None:
        raise HTTPException(status_code=404, detail="회원 정보가 없습니다.")

    return {
        "nickname": row["nickname"],
        "ranking_opt_in": bool(row["ranking_opt_in"]),
    }


@router.patch("")
def update_profile(payload: ProfileUpdate, request: Request):
    user = get_current_user(request)

    nickname = payload.nickname.strip()

    if not nickname:
        raise HTTPException(
            status_code=400,
            detail="닉네임을 입력하세요."
        )

    with get_connection() as conn:
        # 닉네임 중복 확인
        duplicate = conn.execute(
            """
            SELECT id FROM users
            WHERE nickname = ? AND id != ?
            """,
            (nickname, user["id"])
        ).fetchone()

        if duplicate is not None:
            raise HTTPException(
                status_code=409,
                detail="이미 사용 중인 닉네임입니다."
            )

        try:
            conn.execute(
                """
                UPDATE users
                SET nickname = ?, ranking_opt_in = ?
                WHERE id = ?
                """,
                (
                    nickname,
                    int(payload.ranking_opt_in),
                    user["id"]
                )
            )
        except (sqlite3.IntegrityError, PostgresIntegrityError):
            raise HTTPException(
                status_code=409,
                detail="이미 사용 중인 닉네임입니다."
            )

    return {
        "nickname": nickname,
        "ranking_opt_in": payload.ranking_opt_in,
        "message": "프로필 설정이 저장되었습니다."
    }
