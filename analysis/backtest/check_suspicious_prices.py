from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
FILE = BASE / "stock_data_500.csv"

df = pd.read_csv(FILE, dtype={"Ticker": str})
df["Date"] = pd.to_datetime(df["Date"])
df["Ticker"] = df["Ticker"].str.strip()
df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)

tickers = ["950160.KQ", "000250.KQ", "038500.KQ"]
dates = {
    "950160.KQ": ["2026-07-13", "2026-07-16", "2026-07-20"],
    "000250.KQ": ["2026-03-26", "2026-03-30", "2026-03-31"],
    "038500.KQ": ["2026-01-22", "2026-01-29", "2026-02-05"],
}

g = df.groupby("Ticker", sort=False)
df["Prev_Close"] = g["Close"].shift(1)
df["Daily_Close_Change_Pct"] = (
    df["Close"] / df["Prev_Close"] - 1
) * 100

columns = [
    "Date", "Stock_Name", "Ticker", "Open", "High",
    "Low", "Close", "Adj Close", "Volume",
    "Prev_Close", "Daily_Close_Change_Pct"
]
columns = [c for c in columns if c in df.columns]

for ticker in tickers:
    print(f"\n{'=' * 25} {ticker} {'=' * 25}")
    selected_dates = pd.to_datetime(dates[ticker])
    mask = (
        (df["Ticker"] == ticker) &
        df["Date"].between(
            selected_dates.min() - pd.Timedelta(days=12),
            selected_dates.max() + pd.Timedelta(days=12)
        )
    )
    print(df.loc[mask, columns].to_string(index=False))

out = BASE / "results" / "suspicious_price_check.csv"
df[df["Ticker"].isin(tickers)][columns].to_csv(
    out, index=False, encoding="utf-8-sig"
)
print(f"\n저장 위치: {out}")
