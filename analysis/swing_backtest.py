"""
swing_backtest.py  -  max-N-position swing strategy with walk-forward AI scores
"""
import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore", category=RuntimeWarning)

FEATURES = [
    "MA20_Gap", "MA60_Gap", "MA20_MA60_Gap", "MA20_Change", "MA60_Change",
    "High_20D_Gap", "RSI", "RSI_Change", "Volume_Ratio", "Volume_Ratio_5D",
    "Return_1D", "Return_3D", "Return_5D", "Return_10D",
]

# decided in advance, do NOT add more after looking at results
EXIT_PRESETS = {
    "A_hold5": dict(take=0.0, stop=0.0, hold=5),
    "B_TP10_SL7_h5": dict(take=0.10, stop=0.07, hold=5),
    "C_TP10_SL5_h5": dict(take=0.10, stop=0.05, hold=5),
    "D_TP15_SL7_h10": dict(take=0.15, stop=0.07, hold=10),
}


def load_data(train_path, price_path):
    tr = pd.read_csv(train_path, parse_dates=["Date"])
    px = pd.read_csv(price_path, parse_dates=["Date"])
    need = ["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"]
    miss = [c for c in need if c not in px.columns]
    if miss:
        sys.exit(f"{price_path} is missing columns: {miss}\nfound: {list(px.columns)}")
    miss = [c for c in FEATURES + ["Label"] if c not in tr.columns]
    if miss:
        sys.exit(f"{train_path} is missing columns: {miss}")
    px = px[need].dropna(subset=["Open", "High", "Low", "Close"])
    px = px[(px[["Open", "High", "Low", "Close"]] > 0).all(axis=1)]
    px = px.drop_duplicates(["Date", "Ticker"]).sort_values(["Ticker", "Date"])
    tr = tr.dropna(subset=FEATURES + ["Label"]).sort_values("Date")
    return tr, px


def build_panels(px):
    P = {k: px.pivot(index="Date", columns="Ticker", values=k).sort_index()
         for k in ["Open", "High", "Low", "Close", "Volume"]}
    ret = P["Close"].pct_change(fill_method=None)
    P["Vol10"] = ret.rolling(10, min_periods=8).std() * 100
    P["TO"] = (P["Close"] * P["Volume"]).rolling(20, min_periods=10).mean()
    return P


def walk_forward_scores(tr, start, retrain_days=63, purge=10, n_estimators=200):
    dates = np.sort(tr["Date"].unique())
    start_i = int(np.searchsorted(dates, np.datetime64(start)))
    if start_i - purge - 1 < 0:
        sys.exit("start date is too early: not enough training history")
    out = []
    for b in range(start_i, len(dates), retrain_days):
        block = dates[b:b + retrain_days]
        train_end = dates[b - purge - 1]
        trn = tr[tr["Date"] <= train_end]
        tst = tr[(tr["Date"] >= block[0]) & (tr["Date"] <= block[-1])].copy()
        if trn["Label"].nunique() < 2 or tst.empty:
            continue
        m = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=8, min_samples_leaf=10,
            class_weight="balanced", random_state=42, n_jobs=-1,
        ).fit(trn[FEATURES], trn["Label"])
        tst["Score"] = m.predict_proba(tst[FEATURES])[:, 1]
        out.append(tst[["Date", "Ticker", "Label", "Return_1D", "Score"]])
        print(f"  retrain {pd.Timestamp(block[0]).date()} | train rows {len(trn):,} "
              f"| test rows {len(tst):,}", flush=True)
    return pd.concat(out, ignore_index=True)


def simulate(scores, P, cfg, mode="ai", seed=0):
    dates = P["Open"].index
    tick = list(P["Open"].columns)
    t_idx = {t: i for i, t in enumerate(tick)}
    d_idx = {d: i for i, d in enumerate(dates)}
    O, H, L, C = (P[k].to_numpy() for k in ("Open", "High", "Low", "Close"))
    VOL10, TO = P["Vol10"].to_numpy(), P["TO"].to_numpy()

    sc = scores.copy()
    sc["di"] = sc["Date"].map(d_idx)
    sc["ti"] = sc["Ticker"].map(t_idx)
    sc = sc.dropna(subset=["di", "ti"])
    sc["di"] = sc["di"].astype(int)
    sc["ti"] = sc["ti"].astype(int)
    by_day = {k: g for k, g in sc.groupby("di")}

    rng = np.random.default_rng(seed)
    half = cfg["cost"] / 2
    start_day = int(sc["di"].min()) + 1
    cash, positions, cooldown = 1.0, [], {}
    eq_dates, eq_vals, exposure, trades = [dates[start_day - 1]], [1.0], [0.0], []

    for i in range(start_day, len(dates)):
        equity_prev = cash + sum(p["shares"] * p["last_close"] for p in positions)

        # entries at today's open from yesterday's close signals
        slots = cfg["max_pos"] - len(positions)
        g = by_day.get(i - 1)
        if slots > 0 and g is not None:
            ti = g["ti"].to_numpy()
            o, pc = O[i, ti], C[i - 1, ti]
            gap = o / pc - 1
            ok = (gap <= cfg["gap_max"]) & (g["Return_1D"].to_numpy() < cfg["ret1d_max"])
            ok &= TO[i - 1, ti] >= cfg["min_turnover"]
            held = [p["ti"] for p in positions]
            ok &= ~np.isin(ti, held)
            ok &= np.array([cooldown.get(int(x), -1) < i for x in ti])
            pool = g[ok]
            ai = pool[pool["Score"] >= cfg["threshold"]]
            n = len(ai)
            if n == 0:
                cand = ai
            elif mode == "ai":
                cand = ai.sort_values("Score", ascending=False)
            elif mode == "vol":
                cand = pool.assign(V=VOL10[i - 1, pool["ti"].to_numpy()]) \
                           .dropna(subset=["V"]).nlargest(n, "V")
            else:
                cand = pool.sample(n=min(n, len(pool)),
                                   random_state=int(rng.integers(1_000_000_000)))
            alloc_each = equity_prev / cfg["max_pos"]
            for r in cand.head(slots).itertuples():
                a = min(alloc_each, cash)
                if a < 1e-6:
                    break
                ep = O[i, r.ti]
                cash -= a
                positions.append(dict(
                    ti=r.ti, entry_i=i, shares=a * (1 - half) / ep, basis=a,
                    stop=ep * (1 - cfg["stop"]) if cfg["stop"] else 0.0,
                    target=ep * (1 + cfg["take"]) if cfg["take"] else np.inf,
                    last_close=ep,
                ))

        # exits (new positions can also hit stop/target on entry day)
        still = []
        for p in positions:
            j = p["ti"]
            o, h, l, c = O[i, j], H[i, j], L[i, j], C[i, j]
            if np.isnan(c) or np.isnan(o):
                still.append(p)
                continue
            days = i - p["entry_i"] + 1
            px_exit = reason = None
            if o <= p["stop"]:
                px_exit, reason = o, "stop_gap"
            elif o >= p["target"]:
                px_exit, reason = o, "target_gap"
            elif l <= p["stop"]:
                px_exit, reason = p["stop"], "stop"
            elif h >= p["target"]:
                px_exit, reason = p["target"], "target"
            elif days >= cfg["hold"]:
                px_exit, reason = c, "time"
            if px_exit is None:
                p["last_close"] = c
                still.append(p)
                continue
            proceeds = p["shares"] * px_exit * (1 - half)
            cash += proceeds
            trades.append(dict(Ticker=tick[j], Entry=dates[p["entry_i"]], Exit=dates[i],
                               Ret=proceeds / p["basis"] - 1, Days=days, Reason=reason))
            cooldown[j] = i + cfg["cooldown"]
        positions = still

        pos_val = sum(p["shares"] * p["last_close"] for p in positions)
        eq = cash + pos_val
        eq_dates.append(dates[i])
        eq_vals.append(eq)
        exposure.append(pos_val / eq if eq > 0 else 0.0)

    eq = pd.Series(eq_vals, index=pd.DatetimeIndex(eq_dates))
    return eq, pd.DataFrame(trades), float(np.mean(exposure))


def metrics(eq, trades=None, exposure=np.nan):
    r = eq.pct_change().dropna()
    years = len(r) / 252
    out = {
        "Total%": (eq.iloc[-1] / eq.iloc[0] - 1) * 100,
        "CAGR%": ((eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1) * 100 if years > 0 else np.nan,
        "Sharpe": r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0.0,
        "MDD%": (eq / eq.cummax() - 1).min() * 100,
        "Expo%": exposure * 100,
    }
    if trades is not None and len(trades):
        w, l = trades.loc[trades.Ret > 0, "Ret"], trades.loc[trades.Ret <= 0, "Ret"]
        out.update({
            "Trades": len(trades),
            "Win%": (trades.Ret > 0).mean() * 100,
            "Avg%": trades.Ret.mean() * 100,
            "Med%": trades.Ret.median() * 100,
            "PF": w.sum() / abs(l.sum()) if l.sum() != 0 else np.nan,
            "Days": trades.Days.mean(),
        })
    last = eq.groupby(eq.index.year).last()
    prev = last.shift(1).fillna(eq.iloc[0])
    for y, v in (last / prev - 1).items():
        out[f"Y{y}%"] = v * 100
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="training_data.csv")
    ap.add_argument("--prices", default="stock_data_500.csv")
    ap.add_argument("--start", default="2024-01-01", help="first out-of-sample date")
    ap.add_argument("--retrain-days", type=int, default=63)
    ap.add_argument("--rescore", action="store_true", help="ignore cached walk-forward scores")
    ap.add_argument("--thresholds", type=float, nargs="+", default=[0.70, 0.80])
    ap.add_argument("--exits", default="all", help="all or one of: " + ", ".join(EXIT_PRESETS))
    ap.add_argument("--max-pos", type=int, default=4)
    ap.add_argument("--cost", type=float, default=0.006, help="round-trip cost (0.006 = 0.6%%)")
    ap.add_argument("--min-turnover", type=float, default=1e9, help="20d avg traded value (KRW)")
    ap.add_argument("--gap-max", type=float, default=0.07)
    ap.add_argument("--ret1d-max", type=float, default=20.0, help="skip if signal-day return >= this (%%)")
    ap.add_argument("--cooldown", type=int, default=3)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--outdir", default="results")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    tr, px = load_data(a.train, a.prices)
    P = build_panels(px)

    cache = os.path.join(a.outdir, "wf_scores.csv")
    if os.path.exists(cache) and not a.rescore:
        scores = pd.read_csv(cache, parse_dates=["Date"])
        print(f"loaded cached walk-forward scores: {cache}")
    else:
        print("walk-forward scoring ...")
        scores = walk_forward_scores(tr, a.start, a.retrain_days)
        scores.to_csv(cache, index=False)

    print("\n== signal quality by year (walk-forward) ==")
    scores["Year"] = scores["Date"].dt.year
    for th in a.thresholds:
        for y, g in scores.groupby("Year"):
            s = g[g.Score >= th]
            if len(s):
                print(f"th {th:.2f} | {y} | signals {len(s):6d} | hit {s.Label.mean():6.1%} "
                      f"| base {g.Label.mean():6.1%} | lift {s.Label.mean() / g.Label.mean():.2f}x")

    exits = EXIT_PRESETS if a.exits == "all" else {a.exits: EXIT_PRESETS[a.exits]}
    base_cfg = dict(max_pos=a.max_pos, cost=a.cost, min_turnover=a.min_turnover,
                    gap_max=a.gap_max, ret1d_max=a.ret1d_max, cooldown=a.cooldown)

    rows = []
    cols = [c for c in scores["Ticker"].unique() if c in P["Close"].columns]
    first = scores["Date"].min()
    ew = P["Close"][cols].pct_change(fill_method=None).mean(axis=1)
    ew = ew[ew.index >= first]
    ew = ew.fillna(0)
    ew.iloc[0] = 0.0
    ew_eq = (1 + ew).cumprod()
    rows.append({"Config": "EW universe B&H", **metrics(ew_eq)})

    for ename, ex in exits.items():
        for th in a.thresholds:
            cfg = {**base_cfg, **ex, "threshold": th}
            tag = f"{ename}|th{th:.2f}"
            eq, trades, expo = simulate(scores, P, cfg, "ai")
            trades.to_csv(os.path.join(a.outdir, f"swing_trades_{ename}_{th:.2f}.csv"), index=False)
            eq.to_csv(os.path.join(a.outdir, f"swing_equity_{ename}_{th:.2f}.csv"))
            rows.append({"Config": f"{tag}|AI", **metrics(eq, trades, expo)})
            eq, trades, expo = simulate(scores, P, cfg, "vol")
            rows.append({"Config": f"{tag}|VolTop", **metrics(eq, trades, expo)})
            rnd = []
            for s in range(a.seeds):
                eq, trades, expo = simulate(scores, P, cfg, "random", seed=s)
                rnd.append(metrics(eq, trades, expo))
            rows.append({"Config": f"{tag}|Random", **pd.DataFrame(rnd).mean().to_dict()})
            print(f"done {tag}", flush=True)

    res = pd.DataFrame(rows).set_index("Config")
    res.to_csv(os.path.join(a.outdir, "swing_summary.csv"), encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print(f"\n== summary (cost {a.cost:.1%} round trip, max {a.max_pos} positions) ==")
    print(res.round(2).to_string())
    print("\nAI must beat BOTH VolTop and Random on Sharpe/Total, in more than one year.")


if __name__ == "__main__":
    main()
