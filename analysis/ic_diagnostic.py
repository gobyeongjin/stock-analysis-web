import argparse
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--scores", nargs="+",
                default=["touch=results/wf_scores.csv", "rank=results_rank/wf_scores.csv"])
ap.add_argument("--prices", default="stock_data_500.csv")
ap.add_argument("--train", default="training_data.csv")
ap.add_argument("--hold", type=int, default=5)
ap.add_argument("--step", type=int, default=5, help="use every Nth day to avoid overlapping windows")
ap.add_argument("--min-names", type=int, default=25)
ap.add_argument("--cost", type=float, default=0.6, help="round-trip cost in %% (for reference)")
a = ap.parse_args()

RAW = ["Return_1D", "Return_5D", "Return_10D", "RSI", "Volume_Ratio", "High_20D_Gap", "MA20_Gap"]

px = pd.read_csv(a.prices, parse_dates=["Date"], usecols=["Date", "Ticker", "Open", "Close"])
px = px.dropna().drop_duplicates(["Date", "Ticker"]).sort_values(["Ticker", "Date"])
g = px.groupby("Ticker")
px["Fwd"] = g["Close"].shift(-a.hold) / g["Open"].shift(-1) - 1
px["Vol10"] = g["Close"].pct_change(fill_method=None).groupby(px["Ticker"]).transform(
    lambda x: x.rolling(10, min_periods=8).std()) * 100

tr = pd.read_csv(a.train, parse_dates=["Date"], usecols=["Date", "Ticker"] + RAW)
D = tr.merge(px[["Date", "Ticker", "Fwd", "Vol10"]], on=["Date", "Ticker"], how="left")

cols = {}
for item in a.scores:
    name, path = item.split("=", 1)
    s = pd.read_csv(path, parse_dates=["Date"], usecols=["Date", "Ticker", "Score"])
    s = s.rename(columns={"Score": f"Model_{name}"})
    D = D.merge(s, on=["Date", "Ticker"], how="inner" if not cols else "left")
    cols[f"Model_{name}"] = 1  # sign +1

D = D.dropna(subset=["Fwd"])
dates = np.sort(D["Date"].unique())[:: a.step]
D = D[D["Date"].isin(dates)]
print(f"OOS days used: {len(dates)} (every {a.step}th) | rows {len(D):,} | "
      f"{pd.Timestamp(dates[0]).date()} ~ {pd.Timestamp(dates[-1]).date()}")
print(f"unconditional mean fwd {D.Fwd.mean()*100:.2f}% | median {D.Fwd.median()*100:.2f}%\n")

factors = list(cols) + ["Vol10"] + RAW


def daily_stats(col):
    ics, rows = [], []
    for dt, g in D.groupby("Date"):
        g = g.dropna(subset=[col])
        if len(g) < a.min_names:
            continue
        r = g[col].rank(method="first")
        ics.append((dt, r.corr(g["Fwd"].rank())))
        q = pd.qcut(r, 5, labels=False)
        m = g.groupby(q)["Fwd"].mean() * 100
        row = {f"Q{int(k) + 1}": v for k, v in m.items()}
        row["ALL"] = g["Fwd"].mean() * 100
        row["Date"] = dt
        rows.append(row)
    return pd.Series(dict(ics)), pd.DataFrame(rows).set_index("Date")


print(f"{'factor':22s} {'meanIC':>7s} {'t-stat':>7s} {'IC>0':>6s} | "
      f"{'Q1':>6s} {'Q2':>6s} {'Q3':>6s} {'Q4':>6s} {'Q5':>6s} | {'Q5-ALL':>7s} {'Q5-Q1':>7s}")
detail = {}
for col in factors:
    ic, q = daily_stats(col)
    t = ic.mean() / ic.std() * np.sqrt(len(ic)) if ic.std() > 0 else np.nan
    mq = q.mean()
    detail[col] = q
    print(f"{col:22s} {ic.mean():7.3f} {t:7.2f} {(ic > 0).mean():6.1%} | "
          + " ".join(f"{mq.get(f'Q{k}', np.nan):6.2f}" for k in range(1, 6))
          + f" | {mq['Q5'] - mq['ALL']:7.2f} {mq['Q5'] - mq['Q1']:7.2f}")

print("\nQ columns = avg 5-day fwd return % (gross, before costs; round-trip cost ~"
      f"{a.cost}%). Q5 = highest factor value.")
print("\n== Q5 minus ALL (excess %), by year ==")
for col in list(cols):
    q = detail[col]
    ex = (q["Q5"] - q["ALL"]).groupby(q.index.year).agg(["mean", "count"])
    print(col, " | ".join(f"{y}: {r['mean']:+.2f}% (n={int(r['count'])})" for y, r in ex.iterrows()))
