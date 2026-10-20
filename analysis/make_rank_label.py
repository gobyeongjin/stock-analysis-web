import argparse
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--train", default="training_data.csv")
ap.add_argument("--prices", default="stock_data_500.csv")
ap.add_argument("--out", default="training_data_rank.csv")
ap.add_argument("--top", type=float, default=0.20, help="top fraction per day labeled 1")
ap.add_argument("--hold", type=int, default=5)
ap.add_argument("--min-names", type=int, default=20)
a = ap.parse_args()

tr = pd.read_csv(a.train, parse_dates=["Date"])
px = pd.read_csv(a.prices, parse_dates=["Date"], usecols=["Date", "Ticker", "Open", "Close"])
px = px.dropna().drop_duplicates(["Date", "Ticker"]).sort_values(["Ticker", "Date"])

g = px.groupby("Ticker")
entry = g["Open"].shift(-1)                 # buy next-day open
exit_ = g["Close"].shift(-a.hold)           # sell close of day `hold`
px["Fwd"] = exit_ / entry - 1

d = tr.merge(px[["Date", "Ticker", "Fwd"]], on=["Date", "Ticker"], how="left")
d = d.dropna(subset=["Fwd"]).copy()

cnt = d.groupby("Date")["Fwd"].transform("count")
d = d[cnt >= a.min_names].copy()
d["Rank"] = d.groupby("Date")["Fwd"].rank(pct=True)
d["Label_touch"] = d["Label"]
d["Label"] = (d["Rank"] >= 1 - a.top).astype(int)

d = d.drop(columns=["Rank"]).sort_values(["Date", "Ticker"])
d.to_csv(a.out, index=False)

print(f"rows {len(d):,} | tickers {d.Ticker.nunique()} | {d.Date.min().date()} ~ {d.Date.max().date()}")
print(f"new label rate {d.Label.mean():.1%} (old touch label {d.Label_touch.mean():.1%})")
print(f"mean fwd of label=1: {d.loc[d.Label==1,'Fwd'].mean()*100:.2f}% | label=0: {d.loc[d.Label==0,'Fwd'].mean()*100:.2f}%")
print("label rate by year:")
print(d.groupby(d.Date.dt.year)["Label"].mean().round(3).to_string())
