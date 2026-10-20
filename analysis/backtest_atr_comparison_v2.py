
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score

BASE_DIR = Path(__file__).resolve().parent
TRAIN_FILE = BASE_DIR / "training_data_v4_lending.csv"
PRICE_FILE = BASE_DIR / "stock_data_500.csv"

INITIAL_CASH = 1_000_000
MAX_POSITIONS = 5
THRESHOLD = 0.65

# 잠정 비용 가정. 실제 증권사 수수료/세금은 별도 확인 필요.
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
LENDING = "Lending_Balance_Change_Pct"
LAG_FEATURE = "Lending_Balance_Change_Pct_Lag1"


def normalize_ticker(value):
    value = str(value).strip()
    if value.endswith(".KS") or value.endswith(".KQ"):
        value = value.rsplit(".", 1)[0]
    return value.zfill(6)


def load_prices():
    df = pd.read_csv(PRICE_FILE, dtype={"Ticker": str})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Ticker"] = df["Ticker"].map(normalize_ticker)

    for col in ["Open", "High", "Low", "Close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(
        subset=["Date", "Ticker", "Open", "High", "Low", "Close"]
    )
    df = df.sort_values(["Ticker", "Date"])
    df = df.drop_duplicates(["Ticker", "Date"], keep="last")

    df["Prev_Close"] = df.groupby("Ticker")["Close"].shift(1)
    tr = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - df["Prev_Close"]).abs(),
            (df["Low"] - df["Prev_Close"]).abs(),
        ],
        axis=1,
    ).max(axis=1)

    # 단순 이동평균 ATR(14)
    df["ATR14"] = tr.groupby(df["Ticker"]).transform(
        lambda s: s.rolling(14, min_periods=14).mean()
    )

    return df.reset_index(drop=True)


def load_signals(prices):
    df = pd.read_csv(TRAIN_FILE, dtype={"Ticker": str})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Ticker"] = df["Ticker"].map(normalize_ticker)
    df = df.sort_values(["Ticker", "Date"]).copy()

    # 이전 행의 대여잔고 변화율만 사용
    df[LAG_FEATURE] = df.groupby("Ticker")[LENDING].shift(1)
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
    if split <= 5 or split >= len(dates):
        raise ValueError("학습/평가 날짜가 충분하지 않습니다.")

    boundary = pd.Timestamp(dates[split])
    purge_dates = set(dates[max(0, split - 5):split])

    train = df[
        (df["Date"] < boundary)
        & (~df["Date"].isin(purge_dates))
    ].copy()
    test = df[df["Date"] >= boundary].copy()

    if train.empty or test.empty:
        raise ValueError("학습 데이터 또는 평가 데이터가 비어 있습니다.")

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

    if test["Label"].nunique() > 1:
        ap = average_precision_score(
            test["Label"], test["Probability"]
        )
    else:
        ap = float("nan")

    test = test[test["Probability"] >= THRESHOLD].copy()

    # 신호가 생성된 날짜의 ATR을 연결
    atr = prices[["Date", "Ticker", "ATR14"]]
    test = test.merge(atr, on=["Date", "Ticker"], how="left")
    test = test.dropna(subset=["ATR14"])
    test = test[test["ATR14"] > 0].copy()

    signals = test[["Date", "Ticker", "Probability", "ATR14"]]
    return signals, boundary, ap


def get_exit(bar, pos, held_days, strategy):
    """
    반환: (원시 체결 가격, 청산 사유)
    모든 가격은 해당 날짜의 OHLC만 사용.
    """
    stop = pos["stop"]
    target = pos["target"]

    # 시가 갭은 목표가/손절가가 아니라 실제 시가를 기준으로 체결
    if bar.Open <= stop:
        return bar.Open, "손절 갭"
    if bar.Open >= target:
        return bar.Open, "익절 갭"

    # 일봉만으로 도달 순서를 알 수 없으므로 손절을 우선 적용
    if bar.Low <= stop and bar.High >= target:
        return stop, "양쪽 도달: 손절 우선"
    if bar.Low <= stop:
        return stop, "손절"
    if bar.High >= target:
        return target, "익절"

    if strategy.startswith("DAY"):
        return bar.Close, "당일 종가 청산"

    if held_days >= 5:
        return bar.Close, "최대 보유기간 종료"

    return None, None


def run_backtest(name, mode, signals, prices):
    if mode == "day_fixed":
        fixed_tp, fixed_sl = 0.02, 0.015
        atr_tp, atr_sl, max_hold = None, None, 1
    elif mode == "day_atr":
        fixed_tp, fixed_sl = None, None
        atr_tp, atr_sl, max_hold = 1.5, 1.0, 1
    elif mode == "swing_fixed":
        fixed_tp, fixed_sl = 0.05, 0.03
        atr_tp, atr_sl, max_hold = None, None, 5
    else:
        fixed_tp, fixed_sl = None, None
        atr_tp, atr_sl, max_hold = 2.5, 1.5, 5

    price_lookup = {
        (r.Ticker, r.Date): r
        for r in prices.itertuples(index=False)
    }
    dates = sorted(prices["Date"].unique())
    date_idx = {d: i for i, d in enumerate(dates)}

    signal_map = {}
    for r in signals.itertuples(index=False):
        signal_map.setdefault(r.Date, []).append(r)

    cash = float(INITIAL_CASH)
    positions = {}
    trades = []
    equity_rows = []

    for i, date in enumerate(dates):
        # 보유 종목과 현금은 해당 거래일 시작 시점의 상태
        start_positions = set(positions.keys())
        start_cash = cash
        closed_today = set()

        # 1) 시가에 신규 매수
        # 당일 청산 대금은 매수 자금에 포함하지 않음.
        # 당일 청산될 기존 포지션의 슬롯도 미리 재사용하지 않음.
        if i > 0:
            signal_date = dates[i - 1]
            candidates = signal_map.get(signal_date, [])
            candidates = sorted(
                candidates,
                key=lambda r: r.Probability,
                reverse=True,
            )

            slots = max(0, MAX_POSITIONS - len(positions))
            chosen = []

            for sig in candidates:
                ticker = sig.Ticker
                if ticker in start_positions:
                    continue
                if (ticker, date) not in price_lookup:
                    continue
                if any(x.Ticker == ticker for x in chosen):
                    continue
                chosen.append(sig)
                if len(chosen) >= slots:
                    break

            # 시작 현금을 기준으로 매수 예산을 계산
            allocation = start_cash / len(chosen) if chosen else 0

            for sig in chosen:
                ticker = sig.Ticker
                bar = price_lookup[(ticker, date)]

                entry = bar.Open * (1 + SLIPPAGE)

                if mode.endswith("_atr"):
                    atr_value = float(sig.ATR14)
                    target = entry + atr_tp * atr_value
                    stop = entry - atr_sl * atr_value
                else:
                    target = entry * (1 + fixed_tp)
                    stop = entry * (1 - fixed_sl)

                unit_cost = entry * (1 + COMMISSION)
                shares = int(allocation // unit_cost)
                if shares <= 0:
                    continue

                cost = shares * unit_cost
                if cost > cash:
                    continue

                cash -= cost
                positions[ticker] = {
                    "shares": shares,
                    "entry_price": entry,
                    "entry_date": date,
                    "entry_idx": i,
                    "probability": float(sig.Probability),
                    "cost": cost,
                    "target": target,
                    "stop": stop,
                }

        # 2) 신규 매수 포함 모든 포지션의 당일 OHLC 확인
        exits = []
        for ticker, pos in list(positions.items()):
            bar = price_lookup.get((ticker, date))
            if bar is None:
                continue

            held_days = i - pos["entry_idx"] + 1
            raw_exit, reason = get_exit(bar, pos, held_days, name)

            if raw_exit is not None:
                exits.append((ticker, raw_exit, reason))

        for ticker, raw_exit, reason in exits:
            pos = positions.pop(ticker)
            closed_today.add(ticker)

            exit_price = raw_exit * (1 - SLIPPAGE)
            gross_proceeds = exit_price * pos["shares"]
            net_proceeds = gross_proceeds * (
                1 - COMMISSION - SELL_TAX
            )
            cash += net_proceeds

            pnl = net_proceeds - pos["cost"]
            trades.append({
                "Strategy": name,
                "Ticker": ticker,
                "Entry_Date": pos["entry_date"],
                "Exit_Date": date,
                "Entry_Price": pos["entry_price"],
                "Exit_Price": exit_price,
                "Shares": pos["shares"],
                "Probability": pos["probability"],
                "Exit_Reason": reason,
                "Net_PnL": pnl,
                "Return_Pct": pnl / pos["cost"] * 100,
                "Holding_Days": i - pos["entry_idx"] + 1,
            })

        # 3) 종가 기준 일별 자산 평가
        equity = cash
        for ticker, pos in positions.items():
            bar = price_lookup.get((ticker, date))
            if bar is not None:
                equity += pos["shares"] * bar.Close

        equity_rows.append({
            "Date": date,
            "Cash": cash,
            "Open_Positions": len(positions),
            "Equity": equity,
        })

    # 4) 마지막 날 종가로 남은 포지션 청산
    if dates:
        last_date = dates[-1]

        for ticker in list(positions.keys()):
            pos = positions.pop(ticker)
            bar = price_lookup.get((ticker, last_date))
            if bar is None:
                continue

            exit_price = bar.Close * (1 - SLIPPAGE)
            proceeds = (
                exit_price * pos["shares"]
                * (1 - COMMISSION - SELL_TAX)
            )
            cash += proceeds
            pnl = proceeds - pos["cost"]

            trades.append({
                "Strategy": name,
                "Ticker": ticker,
                "Entry_Date": pos["entry_date"],
                "Exit_Date": last_date,
                "Entry_Price": pos["entry_price"],
                "Exit_Price": exit_price,
                "Shares": pos["shares"],
                "Probability": pos["probability"],
                "Exit_Reason": "평가 기간 종료",
                "Net_PnL": pnl,
                "Return_Pct": pnl / pos["cost"] * 100,
                "Holding_Days": date_idx[last_date] - pos["entry_idx"] + 1,
            })

        # 최종 자산은 청산 후 현금으로 기록
        if equity_rows and equity_rows[-1]["Date"] == last_date:
            equity_rows[-1]["Cash"] = cash
            equity_rows[-1]["Open_Positions"] = 0
            equity_rows[-1]["Equity"] = cash

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_rows)

    if not equity_df.empty:
        equity_df["Peak"] = equity_df["Equity"].cummax()
        equity_df["Drawdown_Pct"] = (
            equity_df["Equity"] / equity_df["Peak"] - 1
        ) * 100
        mdd = equity_df["Drawdown_Pct"].min()
        final_equity = equity_df.iloc[-1]["Equity"]
    else:
        mdd = 0.0
        final_equity = INITIAL_CASH

    if not trades_df.empty:
        wins = (trades_df["Net_PnL"] > 0).mean() * 100
        net_pnl = trades_df["Net_PnL"].sum()
        avg_return = trades_df["Return_Pct"].mean()
    else:
        wins, net_pnl, avg_return = 0.0, 0.0, 0.0

    print(f"\n===== {name} =====")
    print("거래 횟수:", len(trades_df))
    print("승률:", round(wins, 2), "%")
    print("평균 거래 수익률:", round(avg_return, 2), "%")
    print("순손익:", round(net_pnl), "원")
    print("최종 평가금액:", round(final_equity), "원")
    print(
        "수익률:",
        round((final_equity / INITIAL_CASH - 1) * 100, 2),
        "%",
    )
    print("MDD:", round(mdd, 2), "%")

    return trades_df, equity_df


def main():
    prices = load_prices()
    signals, boundary, ap = load_signals(prices)

    print("===== 백테스트 설정 =====")
    print("평가 시작일:", boundary.date())
    print("AI 임계값:", THRESHOLD)
    print("초기 투자금:", INITIAL_CASH, "원")
    print("AI 신호 수:", len(signals))
    print("Average Precision:", round(ap, 4))

    strategies = [
        ("DAY_FIXED", "day_fixed"),
        ("DAY_ATR", "day_atr"),
        ("SWING_FIXED", "swing_fixed"),
        ("SWING_ATR", "swing_atr"),
    ]

    all_trades = []
    all_equity = []

    for name, mode in strategies:
        trades, equity = run_backtest(name, mode, signals, prices)

        if not trades.empty:
            all_trades.append(trades)
        if not equity.empty:
            equity["Strategy"] = name
            all_equity.append(equity)

    if all_trades:
        pd.concat(all_trades, ignore_index=True).to_csv(
            BASE_DIR / "backtest_atr_v2_trades.csv",
            index=False,
            encoding="utf-8-sig",
        )

    if all_equity:
        pd.concat(all_equity, ignore_index=True).to_csv(
            BASE_DIR / "backtest_atr_v2_equity.csv",
            index=False,
            encoding="utf-8-sig",
        )

    print("\n결과 저장 완료")
    print("backtest_atr_v2_trades.csv")
    print("backtest_atr_v2_equity.csv")
    print("기존 random_forest.pkl은 수정하지 않았습니다.")


if __name__ == "__main__":
    main()
