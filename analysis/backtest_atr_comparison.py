
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

BASE_DIR = Path(__file__).resolve().parent
TRAIN_FILE = BASE_DIR / "training_data_v4_lending.csv"
PRICE_FILE = BASE_DIR / "stock_data_500.csv"

INITIAL_CASH = 1_000_000
MAX_POSITIONS = 5
THRESHOLD = 0.65

COMMISSION = 0.00015
SELL_TAX = 0.0015
SLIPPAGE = 0.0005

FEATURES = [
    "MA20_Gap", "MA60_Gap", "MA20_MA60_Gap",
    "MA20_Change", "MA60_Change", "High_20D_Gap",
    "Low_20D_Gap", "RSI", "RSI_Change",
    "Volume_Ratio", "Volume_Ratio_5D", "Return_1D",
    "Return_3D", "Return_5D", "Return_10D",
    "Volatility_20D", "Volatility_Change", "Close_Position",
]
LAG_FEATURE = "Lending_Balance_Change_Pct_Lag1"


def normalize_ticker(value):
    value = str(value).strip()
    if value.endswith(".KS") or value.endswith(".KQ"):
        value = value.rsplit(".", 1)[0]
    return value.zfill(6)


def load_prices():
    prices = pd.read_csv(PRICE_FILE, dtype={"Ticker": str})
    prices["Date"] = pd.to_datetime(prices["Date"], errors="coerce")
    prices["Ticker"] = prices["Ticker"].map(normalize_ticker)

    for col in ["Open", "High", "Low", "Close"]:
        prices[col] = pd.to_numeric(prices[col], errors="coerce")

    prices = prices.dropna(
        subset=["Date", "Ticker", "Open", "High", "Low", "Close"]
    )
    prices = prices.sort_values(["Ticker", "Date"])
    prices = prices.drop_duplicates(["Ticker", "Date"], keep="last")

    # ATR(14): 오늘 매수 신호를 계산할 때는 전일까지의 ATR 사용
    prices["Prev_Close"] = prices.groupby("Ticker")["Close"].shift(1)

    tr1 = prices["High"] - prices["Low"]
    tr2 = (prices["High"] - prices["Prev_Close"]).abs()
    tr3 = (prices["Low"] - prices["Prev_Close"]).abs()

    prices["TR"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    prices["ATR14"] = (
        prices.groupby("Ticker")["TR"]
        .transform(lambda s: s.rolling(14, min_periods=14).mean())
    )

    # 신호 날짜에 사용할 ATR은 해당 날짜까지의 값.
    # 신호는 종가 이후 발생하고 다음 거래일 시가에 진입한다.
    return prices


def load_signals(prices):
    df = pd.read_csv(TRAIN_FILE, dtype={"Ticker": str})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Ticker"] = df["Ticker"].map(normalize_ticker)
    df = df.sort_values(["Ticker", "Date"]).copy()

    df[LAG_FEATURE] = (
        df.groupby("Ticker")["Lending_Balance_Change_Pct"].shift(1)
    )

    features = FEATURES + [LAG_FEATURE]

    for col in features + ["Label"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "Future_5D_Valid" in df.columns:
        valid = (
            df["Future_5D_Valid"].astype(str).str.strip().str.lower()
            .isin(["true", "1", "1.0"])
        )
        df = df[valid].copy()

    df = df.dropna(subset=features + ["Label", "Date", "Ticker"])
    df = df[df["Label"].isin([0, 1])].copy()
    df["Label"] = df["Label"].astype(int)

    dates = sorted(df["Date"].unique())
    split = int(len(dates) * 0.8)
    boundary = pd.Timestamp(dates[split])
    purge = set(dates[max(0, split - 5):split])

    train = df[
        (df["Date"] < boundary) & (~df["Date"].isin(purge))
    ].copy()
    test = df[df["Date"] >= boundary].copy()

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(train[features], train["Label"])

    test["Probability"] = model.predict_proba(test[features])[:, 1]
    test = test[test["Probability"] >= THRESHOLD].copy()

    # ATR와 신호를 연결. ATR은 신호 날짜 기준으로 저장.
    atr = prices[["Date", "Ticker", "ATR14"]]
    test = test.merge(atr, on=["Date", "Ticker"], how="inner")
    test = test.dropna(subset=["ATR14"])

    return test[["Date", "Ticker", "Probability", "ATR14"]], boundary


def run_backtest(name, signals, prices, mode):
    if mode == "day_fixed":
        tp_pct, sl_pct, max_hold = 0.02, 0.015, 1
        tp_atr, sl_atr = None, None
    elif mode == "day_atr":
        tp_pct, sl_pct, max_hold = None, None, 1
        tp_atr, sl_atr = 1.5, 1.0
    elif mode == "swing_fixed":
        tp_pct, sl_pct, max_hold = 0.05, 0.03, 5
        tp_atr, sl_atr = None, None
    else:
        tp_pct, sl_pct, max_hold = None, None, 5
        tp_atr, sl_atr = 2.5, 1.5

    lookup = {
        (r.Ticker, r.Date): r
        for r in prices.itertuples(index=False)
    }
    all_dates = sorted(prices["Date"].unique())
    date_idx = {d: i for i, d in enumerate(all_dates)}

    signal_map = {}
    for r in signals.itertuples(index=False):
        signal_map.setdefault(r.Date, []).append(r)

    cash = float(INITIAL_CASH)
    positions = {}
    closed_today = set()
    trades = []
    equity_rows = []

    for i, date in enumerate(all_dates):
        closed_today = set()

        # 1. 보유 포지션 청산 확인
        exits = []

        for ticker, pos in positions.items():
            bar = lookup.get((ticker, date))
            if bar is None:
                continue

            held = i - pos["entry_idx"] + 1
            target, stop = pos["target"], pos["stop"]

            exit_price = None
            reason = None

            # 시가가 손절/익절 가격을 넘어 갭 발생한 경우
            if bar.Open <= stop:
                exit_price, reason = bar.Open, "손절 갭"
            elif bar.Open >= target:
                exit_price, reason = bar.Open, "익절 갭"
            elif bar.Low <= stop and bar.High >= target:
                exit_price, reason = stop, "양쪽 도달: 손절 우선"
            elif bar.Low <= stop:
                exit_price, reason = stop, "손절"
            elif bar.High >= target:
                exit_price, reason = target, "익절"
            elif held >= max_hold:
                exit_price, reason = bar.Close, "기간 종료"

            if exit_price is not None:
                exits.append((ticker, exit_price, reason))

        for ticker, raw_exit, reason in exits:
            pos = positions.pop(ticker)
            closed_today.add(ticker)

            exit_price = raw_exit * (1 - SLIPPAGE)
            gross = exit_price * pos["shares"]
            net = gross * (1 - COMMISSION - SELL_TAX)
            cash += net

            pnl = net - pos["cost"]
            trades.append({
                "Strategy": name,
                "Ticker": ticker,
                "Entry_Date": pos["entry_date"],
                "Exit_Date": date,
                "Probability": pos["probability"],
                "Exit_Reason": reason,
                "Net_PnL": pnl,
                "Return_Pct": pnl / pos["cost"] * 100,
                "Holding_Days": i - pos["entry_idx"] + 1,
            })

        # 2. 이전 거래일 신호로 오늘 시가에 매수
        if i > 0:
            prev_date = all_dates[i - 1]
            candidates = signal_map.get(prev_date, [])
            candidates = sorted(
                candidates, key=lambda r: r.Probability, reverse=True
            )

            slots = MAX_POSITIONS - len(positions)
            candidates = [
                r for r in candidates
                if r.Ticker not in positions
                and r.Ticker not in closed_today
                and (r.Ticker, date) in lookup
            ][:max(slots, 0)]

            if candidates and cash > 0:
                allocation = cash / len(candidates)

                for sig in candidates:
                    bar = lookup[(sig.Ticker, date)]
                    entry = bar.Open * (1 + SLIPPAGE)

                    if mode.endswith("_atr"):
                        # 신호일 종가까지 알려진 ATR 사용
                        atr = float(sig.ATR14)
                        target = entry + tp_atr * atr
                        stop = entry - sl_atr * atr
                    else:
                        target = entry * (1 + tp_pct)
                        stop = entry * (1 - sl_pct)

                    unit_cost = entry * (1 + COMMISSION)
                    shares = int(allocation // unit_cost)
                    if shares <= 0:
                        continue

                    cost = shares * entry * (1 + COMMISSION)
                    if cost > cash:
                        continue

                    cash -= cost
                    positions[sig.Ticker] = {
                        "shares": shares,
                        "entry_price": entry,
                        "entry_date": date,
                        "entry_idx": i,
                        "probability": float(sig.Probability),
                        "cost": cost,
                        "target": target,
                        "stop": stop,
                    }

        # 3. 일별 자산 평가: 현금 + 보유 종목의 종가 평가액
        equity = cash
        for ticker, pos in positions.items():
            bar = lookup.get((ticker, date))
            if bar is not None:
                equity += pos["shares"] * bar.Close

        equity_rows.append({"Date": date, "Equity": equity})

    # 평가 기간 종료 시 잔여 포지션 청산
    if all_dates:
        last_date = all_dates[-1]
        for ticker in list(positions.keys()):
            pos = positions.pop(ticker)
            bar = lookup.get((ticker, last_date))
            if bar is None:
                continue

            exit_price = bar.Close * (1 - SLIPPAGE)
            gross = exit_price * pos["shares"]
            net = gross * (1 - COMMISSION - SELL_TAX)
            cash += net
            pnl = net - pos["cost"]

            trades.append({
                "Strategy": name,
                "Ticker": ticker,
                "Entry_Date": pos["entry_date"],
                "Exit_Date": last_date,
                "Probability": pos["probability"],
                "Exit_Reason": "평가 기간 종료",
                "Net_PnL": pnl,
                "Return_Pct": pnl / pos["cost"] * 100,
                "Holding_Days": date_idx[last_date] - pos["entry_idx"] + 1,
            })

        equity_rows.append({"Date": last_date, "Equity": cash})

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_rows).drop_duplicates("Date", keep="last")

    if not trades_df.empty:
        wins = (trades_df["Net_PnL"] > 0).mean() * 100
        net_pnl = trades_df["Net_PnL"].sum()
    else:
        wins, net_pnl = 0, 0

    if not equity_df.empty:
        equity_df["Peak"] = equity_df["Equity"].cummax()
        equity_df["Drawdown"] = (
            equity_df["Equity"] / equity_df["Peak"] - 1
        )
        mdd = equity_df["Drawdown"].min() * 100
        final_equity = equity_df.iloc[-1]["Equity"]
    else:
        mdd = 0
        final_equity = INITIAL_CASH

    print(f"\n===== {name} =====")
    print("거래 횟수:", len(trades_df))
    print("승률:", round(wins, 2), "%")
    print("순손익:", round(net_pnl), "원")
    print("최종 평가금액:", round(final_equity), "원")
    print("수익률:", round((final_equity / INITIAL_CASH - 1) * 100, 2), "%")
    print("MDD:", round(mdd, 2), "%")

    return trades_df, equity_df


def main():
    prices = load_prices()
    signals, boundary = load_signals(prices)

    print("평가 시작일:", boundary.date())
    print("AI 신호 수:", len(signals))
    print("임계값:", THRESHOLD)
    print("초기 투자금:", INITIAL_CASH)

    strategies = [
        ("DAY_FIXED", "day_fixed"),
        ("DAY_ATR", "day_atr"),
        ("SWING_FIXED", "swing_fixed"),
        ("SWING_ATR", "swing_atr"),
    ]

    all_trades = []
    all_equity = []

    for name, mode in strategies:
        trades, equity = run_backtest(name, signals, prices, mode)

        if not trades.empty:
            all_trades.append(trades)

        if not equity.empty:
            equity["Strategy"] = name
            all_equity.append(equity)

    if all_trades:
        pd.concat(all_trades, ignore_index=True).to_csv(
            BASE_DIR / "backtest_atr_trades.csv",
            index=False,
            encoding="utf-8-sig",
        )

    if all_equity:
        pd.concat(all_equity, ignore_index=True).to_csv(
            BASE_DIR / "backtest_atr_equity.csv",
            index=False,
            encoding="utf-8-sig",
        )

    print("\n결과 파일:")
    print("backtest_atr_trades.csv")
    print("backtest_atr_equity.csv")
    print("기존 모델 파일은 수정하지 않았습니다.")


if __name__ == "__main__":
    main()
