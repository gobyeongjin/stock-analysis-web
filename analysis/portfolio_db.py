from contextlib import contextmanager
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "portfolio.db"


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
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
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
                    DEFAULT (datetime('now'))
            )
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_trades_ticker_date
            ON trades(ticker, trade_date, id)
        """)

    print(f"SQLite DB 준비 완료: {DB_PATH}")


if __name__ == "__main__":
    init_db()
