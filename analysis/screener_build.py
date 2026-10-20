"""
screener_build.py - rule-based screener: today's matches + honest historical stats

Reads  stock_data_500.csv (Date, Ticker, Open, High, Low, Close, Volume, [Stock_Name])
Writes screener_today.json, screener_stats.json, screener_stats_by_year.csv
Re-run after updating stock_data_500.csv.
"""
import argparse
import json

import numpy as np
import pandas as pd

HORIZONS = [1, 3, 5, 10]

RULES = {
    "volume_surge": "거래량 3배 이상 + 당일 상승",
    "breakout_20d": "20일 고점 돌파",
    "top_5d_return": "5일 수익률 상위 5%",
    "ma_aligned": "이평선 정배열 (종가>MA20>MA60>MA120, MA20 상승)",
    "ma_top5_combo": "이평선 정배열 + 5일 수익률 상위 5%",
}

# how to sort today's matches inside each rule
SORT_KEY = {
    "volume_surge": "VolRatio",
    "breakout_20d": "BreakoutPct",
    "top_5d_return": "Ret5",
    "ma_aligned": "Ret5",
    "ma_top5_combo": "Ret5",
}

ap = argparse.ArgumentParser()
ap.add_argument("--prices", default="stock_data_500.csv")
ap.add_argument("--min-turnover", type=float, default=1e9, help="20d avg traded value (KRW)")
ap.add_argument("--cost", type=float, default=0.6, help="round-trip cost %% for reference")
ap.add_argument("--top", type=int, default=50, help="max stocks listed per rule")
a = ap.parse_args()

px = pd.read_csv(a.prices, parse_dates=["Date"])
has_name = "Stock_Name" in px.columns
need = ["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"]
px = px[need + (["Stock_Name"] if has_name else [])]
px = px.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
px = px.drop_duplicates(["Date", "Ticker"]).sort_values(["Ticker", "Date"]).reset_index(drop=True)


def grp():
    return px.groupby("Ticker")


def hac_mean_t(series, max_lag):
    """Newey-West HAC 방식으로 평균의 t통계량을 계산한다."""
    x = np.asarray(series.dropna(), dtype=float)
    n = len(x)

    if n < 3:
        return None

    centered = x - x.mean()
    lag = min(max_lag, n - 1)

    long_var = np.dot(centered, centered) / n

    for k in range(1, lag + 1):
        weight = 1 - k / (lag + 1)
        gamma = np.dot(centered[k:], centered[:-k]) / n
        long_var += 2 * weight * gamma

    if long_var <= 1e-14:
        return None

    se = np.sqrt(long_var / n)
    return float(x.mean() / se) if se > 0 else None



# ---------------- features (all use data up to and including the signal day)
px["Ret1"] = grp()["Close"].pct_change(fill_method=None)
px["Ret5"] = grp()["Close"].pct_change(5, fill_method=None)
px["VolMA20"] = grp()["Volume"].transform(lambda s: s.rolling(20, min_periods=15).mean())
px["VolRatio"] = px["Volume"] / px["VolMA20"]
px["High20Prev"] = grp()["High"].transform(lambda s: s.rolling(20, min_periods=15).max().shift(1))
px["BreakoutPct"] = px["Close"] / px["High20Prev"] - 1
for n in (20, 60, 120):
    px[f"MA{n}"] = grp()["Close"].transform(lambda s, n=n: s.rolling(n, min_periods=n).mean())
px["MA20_5"] = grp()["MA20"].shift(5)
px["Turnover20"] = (px["Close"] * px["Volume"]).groupby(px["Ticker"]).transform(
    lambda s: s.rolling(20, min_periods=15).mean())

liquid = px["Turnover20"] >= a.min_turnover

rule_mask = {
    "volume_surge": (px["VolRatio"] >= 3) & (px["Ret1"] > 0),
    "breakout_20d": px["Close"] > px["High20Prev"],
    "top_5d_return": px["Ret5"].where(liquid).groupby(px["Date"]).rank(pct=True) >= 0.95,
    "ma_aligned": (px["Close"] > px["MA20"]) & (px["MA20"] > px["MA60"])
                  & (px["MA60"] > px["MA120"]) & (px["MA20"] > px["MA20_5"]),
    "ma_top5_combo": (
    (px["Close"] > px["MA20"])
    & (px["MA20"] > px["MA60"])
    & (px["MA60"] > px["MA120"])
    & (px["MA20"] > px["MA20_5"])
    & (px["Ret5"].where(liquid).groupby(px["Date"]).rank(pct=True) >= 0.95)
),
}
for k in rule_mask:
    rule_mask[k] = rule_mask[k].fillna(False) & liquid

# ---------------- forward returns: buy next-day open, sell close of day h
entry = grp()["Open"].shift(-1)
for h in HORIZONS:
    px[f"Fwd{h}"] = grp()["Close"].shift(-h) / entry - 1
    base = px.loc[liquid].groupby("Date")[f"Fwd{h}"].transform("mean")
    px[f"Base{h}"] = base                       # same-day average of all liquid stocks
    px[f"Exc{h}"] = px[f"Fwd{h}"] - px[f"Base{h}"]



def stat_row(sub, h, name, label):
    fwd_col = f"Fwd{h}"
    exc_col = f"Exc{h}"

    # 미래 수익률과 초과수익률이 모두 존재하는 관측치만 사용
    valid = sub.dropna(subset=[fwd_col, exc_col])
    f = valid[fwd_col]
    e = valid[exc_col]

    # 날짜별로 해당 조건 종목들의 평균 초과수익률 계산
    by_day = valid.groupby("Date")[exc_col].mean().dropna()

    # 날짜별 평균 초과수익률의 자기상관을 고려한 HAC t값
    t = hac_mean_t(by_day, max_lag=h - 1)

    return {
        "rule": name,
        "label": label,
        "horizon_days": h,
        "samples": int(len(f)),
        "signal_days": int(len(by_day)),
        "mean_pct": round(f.mean() * 100, 3) if len(f) else None,
        "median_pct": round(f.median() * 100, 3) if len(f) else None,
        "win_pct": round((f > 0).mean() * 100, 1) if len(f) else None,
        # 개별 종목 관측치 기준 평균 초과수익률
        "excess_mean_pct": round(e.mean() * 100, 3) if len(e) else None,
        "excess_median_pct": round(e.median() * 100, 3) if len(e) else None,
        # 날짜별 평균 초과수익률과 그 평균에 대한 HAC 검정
        "excess_daily_mean_pct": round(by_day.mean() * 100, 3) if len(by_day) else None,
        "excess_daily_t_stat": None if t is None else round(t, 2),
        # 기존 페이지와의 호환성을 위해 기존 키도 유지
        "excess_t_stat": None if t is None else round(t, 2),
        "mean_after_cost_pct": round(f.mean() * 100 - a.cost, 3) if len(f) else None,
    }


stats, by_year = [], []
allrows = px[liquid]
for h in HORIZONS:
    stats.append(stat_row(allrows, h, "ALL_LIQUID", "기준선: 유동성 필터 통과 전 종목"))
for name, label in RULES.items():
    sub = px[rule_mask[name]]
    for h in HORIZONS:
        stats.append(stat_row(sub, h, name, label))
    for y, gy in sub.groupby(sub["Date"].dt.year):
        r = stat_row(gy, 5, name, label)
        r["year"] = int(y)
        by_year.append(r)

for h in HORIZONS:
    px[f"Fwd{h}"] = grp()["Close"].shift(-h) / entry - 1
    base = px.loc[liquid].groupby("Date")[f"Fwd{h}"].transform("mean")
    px[f"Base{h}"] = base
    px[f"Exc{h}"] = px[f"Fwd{h}"] - px[f"Base{h}"]


# ---------------- portfolio backtest: ma_top5_combo
# 신호 다음 거래일 시가 매수, 5번째 거래일 종가 매도
# 주당 최대 1회 매수, 최대 3종목 보유, 왕복 비용 차감

px["BTEntryDate"] = grp()["Date"].shift(-1)
px["BTEntryOpen"] = grp()["Open"].shift(-1)
px["BTExitDate"] = grp()["Date"].shift(-5)
px["BTExitClose"] = grp()["Close"].shift(-5)

candidates = px.loc[
    rule_mask["ma_top5_combo"],
    [
        "Date", "Ticker", "Ret5",
        "BTEntryDate", "BTEntryOpen",
        "BTExitDate", "BTExitClose",
    ],
].dropna(
    subset=[
        "BTEntryDate", "BTEntryOpen",
        "BTExitDate", "BTExitClose",
    ]
).copy()

# 하루에 신호가 여러 개면 Ret5가 가장 높은 종목 하나만 선택
candidates = (
    candidates.sort_values(
        ["Date", "Ret5"],
        ascending=[True, False],
    )
    .drop_duplicates("Date", keep="first")
    .sort_values(["BTEntryDate", "Ret5"], ascending=[True, False])
)

entries_by_date = {
    date: group.sort_values("Ret5", ascending=False)
    for date, group in candidates.groupby("BTEntryDate")
}

daily_rows = px[["Date", "Ticker", "Close"]].copy()
daily_closes = {
    date: dict(zip(group["Ticker"], group["Close"]))
    for date, group in daily_rows.groupby("Date")
}

trading_dates = sorted(px["Date"].dropna().unique())

cash = 1.0
holdings = []
bought_weeks = set()
trade_returns = []
trade_log = []
equity_rows = []

for raw_date in trading_dates:
    date = pd.Timestamp(raw_date)
    week = (date.isocalendar().year, date.isocalendar().week)

    # 1) 오늘 시가에 진입할 후보 확인
    day_candidates = entries_by_date.get(date, [])

    for _, candidate in day_candidates.iterrows() if not isinstance(day_candidates, list) else []:
        if week in bought_weeks:
            break

        if len(holdings) >= 3:
            break

        ticker = candidate["Ticker"]

        # 같은 종목을 중복 보유하지 않음
        if any(p["Ticker"] == ticker for p in holdings):
            continue

        entry_price = float(candidate["BTEntryOpen"])
        if entry_price <= 0 or cash <= 0:
            continue

        # 직전 평가자산의 약 1/3을 종목당 배정
        current_equity = cash + sum(
            p["amount"] * p["last_close"] / p["entry_price"]
            for p in holdings
        )
        allocation = min(cash, current_equity / 3)

        cash -= allocation

        holdings.append({
            "Ticker": ticker,
            "entry_date": date,
            "entry_price": entry_price,
            "amount": allocation,
            "exit_date": pd.Timestamp(candidate["BTExitDate"]),
            "exit_price": float(candidate["BTExitClose"]),
            "last_close": entry_price,
        })

        bought_weeks.add(week)

    # 2) 당일 종가로 보유 종목 평가 및 매도
    close_map = daily_closes.get(date, {})

    for p in holdings:
        if p["Ticker"] in close_map:
            p["last_close"] = float(close_map[p["Ticker"]])

    remaining = []

    for p in holdings:
        if p["exit_date"] <= date:
            gross_return = p["exit_price"] / p["entry_price"] - 1
            net_return = gross_return - a.cost / 100

            cash += p["amount"] * (1 + net_return)
            trade_returns.append(net_return * 100)

            trade_log.append({
           "Ticker": p["Ticker"],
           "EntryDate": p["entry_date"],
           "ExitDate": date,
           "EntryPrice": p["entry_price"],
           "ExitPrice": p["exit_price"],
           "AllocatedCapital": p["amount"],
           "NetReturnPct": net_return * 100, })
        else:
            remaining.append(p)

    holdings = remaining

    # 3) 당일 종가 기준 전체 자산 계산
    market_value = sum(
        p["amount"] * p["last_close"] / p["entry_price"]
        for p in holdings
    )
    equity = cash + market_value

    equity_rows.append({
        "Date": date,
        "Equity": equity,
        "Holdings": len(holdings),
    })

# ---------------- portfolio results
equity_df = pd.DataFrame(equity_rows)

if not equity_df.empty:
    equity_df["Drawdown"] = (
        equity_df["Equity"] / equity_df["Equity"].cummax() - 1
    )

    total_return = (equity_df["Equity"].iloc[-1] - 1) * 100
    max_drawdown = equity_df["Drawdown"].min() * 100

    print("\n== portfolio backtest: ma_top5_combo ==")
    print(f"Trades: {len(trade_returns)}")
    print(f"Total return: {total_return:.2f}%")
    print(f"Max drawdown: {max_drawdown:.2f}%")
    print(
        "Win rate: "
        f"{(np.mean(np.array(trade_returns) > 0) * 100):.1f}%"
        if trade_returns else "Win rate: N/A"
    )
    print(
        "Average net trade return: "
        f"{np.mean(trade_returns):.3f}%"
        if trade_returns else "Average net trade return: N/A"
    )

    equity_df.to_csv(
        "portfolio_backtest.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print("Saved: portfolio_backtest.csv")

pd.DataFrame(trade_log).to_csv(
    "portfolio_trades.csv",
    index=False,
    encoding="utf-8-sig",
)

print("Saved: portfolio_trades.csv")

if trade_log:
    trades_df = pd.DataFrame(trade_log)
    print("\n== worst 10 trades ==")
    print(
        trades_df.sort_values("NetReturnPct").head(10).to_string(index=False)
    )


# ---------------- period-separated validation
periods = [
    ("exploration_2022_2024", 2022, 2024),
    ("validation_2025_2026", 2025, 2026),
]

period_rows = []

period_rules = [
    ("ALL_LIQUID", "기준선: 전체 유동성 종목", liquid)
] + [
    (name, label, rule_mask[name])
    for name, label in RULES.items()
]

for period_name, start_year, end_year in periods:
    year_mask = px["Date"].dt.year.between(start_year, end_year)

    for name, label, mask in period_rules:
        sub = px[mask & year_mask]

        if sub.empty:
            continue

        r = stat_row(sub, 5, name, label)
        r["period"] = period_name
        r["start_year"] = start_year
        r["end_year"] = end_year
        period_rows.append(r)

pd.DataFrame(period_rows).to_csv(
    "screener_stats_by_period.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n== period-separated validation ==")
period_df = pd.DataFrame(period_rows)

if not period_df.empty:
    print(period_df[
        [
            "period",
            "rule",
            "samples",
            "mean_after_cost_pct",
            "median_pct",
            "win_pct",
            "excess_mean_pct",
            "excess_t_stat",
        ]
    ].to_string(index=False))


# ---------------- today's matches
asof = px["Date"].max()
today = px[px["Date"] == asof]
out_rules = {}
for name, label in RULES.items():
    m = today[rule_mask[name].loc[today.index]].sort_values(SORT_KEY[name], ascending=False)
    stocks = []
    for r in m.head(a.top).itertuples():
        stocks.append({
            "ticker": r.Ticker,
            "name": getattr(r, "Stock_Name", r.Ticker) if has_name else r.Ticker,
            "close": float(r.Close),
            "return_1d_pct": round(float(r.Ret1) * 100, 2) if pd.notna(r.Ret1) else None,
            "return_5d_pct": round(float(r.Ret5) * 100, 2) if pd.notna(r.Ret5) else None,
            "volume_ratio": round(float(r.VolRatio), 2) if pd.notna(r.VolRatio) else None,
            "turnover_20d": float(r.Turnover20),
        })
    out_rules[name] = {"label": label, "count": int(len(m)), "stocks": stocks}

with open("screener_today.json", "w", encoding="utf-8") as f:
    json.dump({"as_of": str(asof.date()), "min_turnover": a.min_turnover, "rules": out_rules},
              f, ensure_ascii=False)
with open("screener_stats.json", "w", encoding="utf-8") as f:
    json.dump({"cost_pct": a.cost,
               "data_range": [str(px["Date"].min().date()), str(asof.date())],
               "stats": stats}, f, ensure_ascii=False)
pd.DataFrame(by_year).to_csv("screener_stats_by_year.csv", index=False, encoding="utf-8-sig")

print(f"as of {asof.date()} | tickers {px.Ticker.nunique()} | liquid today {int(liquid[px.Date == asof].sum())}")
print("\n== today's matches ==")
for k, v in out_rules.items():
    print(f"{k:15s} {v['count']:4d}")
print("\n== history: 5-day forward (buy next open -> close of day 5, gross) ==")
df = pd.DataFrame([s for s in stats if s["horizon_days"] == 5])
print(df[["rule", "samples", "signal_days", "mean_pct", "median_pct", "win_pct",
          "excess_mean_pct", "excess_t_stat"]].to_string(index=False))
print(f"\n(round-trip cost ~{a.cost}%. 'excess' = vs same-day average of all liquid stocks.)")
