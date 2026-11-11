import sqlite3
from contextlib import contextmanager
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "multiuser_dev.db"


@contextmanager
def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")

    try:
        with conn:
            yield conn
    finally:
        conn.close()


def init_db():
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
                    DEFAULT (datetime('now'))
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                ticker TEXT NOT NULL,
                stock_name TEXT NOT NULL,
                trade_type TEXT NOT NULL
                    CHECK (trade_type IN ('BUY', 'SELL')),
                trade_date TEXT NOT NULL,
                price REAL NOT NULL CHECK (price > 0),
                quantity INTEGER NOT NULL CHECK (quantity > 0),
                fee REAL NOT NULL DEFAULT 0 CHECK (fee >= 0),
                tax REAL NOT NULL DEFAULT 0 CHECK (tax >= 0),
                memo TEXT DEFAULT '',
                created_at TEXT NOT NULL
                    DEFAULT (datetime('now')),
                FOREIGN KEY (user_id)
                    REFERENCES users(id)
                    ON DELETE CASCADE
            )
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_user_trades
            ON trades(user_id, ticker, trade_date, id)
        """)


def migrate_ranking_columns():
    """
    기존 회원 데이터를 유지하면서
    닉네임과 랭킹 공개 동의 컬럼을 추가한다.
    """
    init_db()

    with get_connection() as conn:
        columns = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(users)")
        }

        if "nickname" not in columns:
            conn.execute("""
                ALTER TABLE users
                ADD COLUMN nickname TEXT
            """)

        if "ranking_opt_in" not in columns:
            conn.execute("""
                ALTER TABLE users
                ADD COLUMN ranking_opt_in INTEGER NOT NULL DEFAULT 0
                CHECK (ranking_opt_in IN (0, 1))
            """)


def ensure_unique_nickname_index():
    """
    닉네임이 설정된 회원에게만
    고유 제약조건을 적용한다.
    """
    migrate_ranking_columns()

    with get_connection() as conn:
        duplicates = conn.execute("""
            SELECT nickname, COUNT(*) AS cnt
            FROM users
            WHERE nickname IS NOT NULL
            GROUP BY nickname
            HAVING COUNT(*) > 1
        """).fetchall()

        if duplicates:
            raise ValueError(
                "중복 닉네임이 존재하여 인덱스를 생성할 수 없습니다."
            )

        conn.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS
            idx_users_nickname_unique
            ON users(nickname)
            WHERE nickname IS NOT NULL
        """)


if __name__ == "__main__":
    init_db()
    print(f"다중 사용자 개발 DB 생성 완료: {DB_PATH}")
