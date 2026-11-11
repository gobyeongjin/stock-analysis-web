import sqlite3

from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError, VerificationError

from multiuser_db import get_connection, init_db


password_hasher = PasswordHasher()


def register_user(email: str, password: str):
    email = email.strip().lower()

    if not email or "@" not in email:
        raise ValueError("올바른 이메일을 입력하세요.")

    if len(password) < 12:
        raise ValueError("비밀번호는 12자 이상이어야 합니다.")

    password_hash = password_hasher.hash(password)

    init_db()

    try:
        with get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO users (email, password_hash)
                VALUES (?, ?)
                """,
                (email, password_hash),
            )

            return cursor.lastrowid

    except sqlite3.IntegrityError:
        raise ValueError("이미 가입된 이메일입니다.")


def authenticate_user(email: str, password: str):
    email = email.strip().lower()

    init_db()

    with get_connection() as conn:
        row = conn.execute(
            """
            SELECT id, email, password_hash
            FROM users
            WHERE email = ?
            """,
            (email,),
        ).fetchone()

    if row is None:
        return None

    try:
        password_hasher.verify(row["password_hash"], password)
    except (VerifyMismatchError, VerificationError):
        return None

    return {
        "id": row["id"],
        "email": row["email"],
    }
