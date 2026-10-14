
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score

BASE_DIR = Path(__file__).resolve().parent
TRAIN_FILE = BASE_DIR / "training_data_v4_lending.csv"
PRICE_FILE = BASE_DIR / "stock_data_500.csv"

USE_MARKET_FILTER = False
MARKET_MA_DAYS = 20
MARKET_FILTER_FILE = BASE_DIR / "kospi_market_filter.csv"


INITIAL_CASH = 1_000_000.0
MAX_POSITIONS = 5
THRESHOLD = 0.65

# 잠정 비용 가정: 실제 증권사 비용은 별도로 확인
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

    df = df[
        (df["Open"] > 0)
        & (df["High"] > 0)
        & (df["Low"] > 0)
        & (df["Close"] > 0)
        & (df["High"] >= df["Low"])
    ].copy()

    df["Prev_Close"] = df.groupby("Ticker")["Close"].shift(1)

    true_range = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - df["Prev_Close"]).abs(),
            (df["Low"] - df["Prev_Close"]).abs(),
        ],
        axis=1,
    ).max(axis=1)

    # V3와 동일한 단순 이동평균 ATR(14)
    df["ATR14"] = true_range.groupby(df["Ticker"]).transform(
        lambda s: s.rolling(14, min_periods=14).mean()
    )

    print("가격 데이터 로딩 완료:", len(df), "행")
    print("종목 수:", df["Ticker"].nunique())

    return df.reset_index(drop=True)



def load_signals(prices):
    df = pd.read_csv(TRAIN_FILE, dtype={"Ticker": str})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Ticker"] = df["Ticker"].map(normalize_ticker)
    df = df.sort_values(["Ticker", "Date"]).copy()

    # 이전 행의 대여잔고 변화율 사용
    df[LAG_FEATURE] = df.groupby("Ticker")[LENDING].shift(1)
    features = FEATURES + [LAG_FEATURE]

    for col in features + ["Label"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "Future_5D_Valid" in df.columns:
        valid = (
            df["Future_5D_Valid"]
            .astype(str)
            .str.strip()
            .str.lower()
            .isin(["true", "1", "1.0"])
        )
        df = df[valid].copy()

    df = df.dropna(
        subset=features + ["Label", "Date", "Ticker"]
    )
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

    test["Probability"] = model.predict_proba(
        test[features]
    )[:, 1]

    if test["Label"].nunique() > 1:
        ap = average_precision_score(
            test["Label"],
            test["Probability"],
        )
    else:
        ap = float("nan")

    test = test[test["Probability"] >= THRESHOLD].copy()

    # 신호일의 ATR 연결
    atr = prices[["Date", "Ticker", "ATR14"]].copy()

    test = test.merge(
        atr,
        on=["Date", "Ticker"],
        how="left",
    )

    test = test.dropna(subset=["ATR14"])
    test = test[test["ATR14"] > 0].copy()

    signals = test[
        ["Date", "Ticker", "Probability", "ATR14"]
    ].copy()

    return signals, boundary, ap


def load_market_filter():
    market = pd.read_csv(
        BASE_DIR / "kospi_market_filter.csv"
    )

    market["Date"] = pd.to_datetime(
        market["Date"]
    ).dt.normalize()

    market["Close"] = pd.to_numeric(
        market["Close"],
        errors="coerce"
    )

    market = (
        market[["Date", "Close"]]
        .dropna()
        .sort_values("Date")
        .drop_duplicates("Date", keep="last")
        .copy()
    )

    # 코스피 이동평균선 기간 변경
    ma_col = f"MA{MARKET_MA_DAYS}"

    market[ma_col] = (
        market["Close"]
        .rolling(
            MARKET_MA_DAYS,
            min_periods=MARKET_MA_DAYS
        )
        .mean()
    )

    market["Market_OK"] = (
        market["Close"] > market[ma_col]
    )

    return (
        market.set_index("Date")["Market_OK"]
        .to_dict()
    )



def run_backtest_v4(prices, signals, strategy_name):
    mode_map = {
        "DAY_FIXED": ("day", "fixed"),
        "DAY_ATR": ("day", "atr"),
        "SWING_FIXED": ("swing", "fixed"),
        "SWING_ATR": ("swing", "atr"),
    }

    if strategy_name not in mode_map:
        raise ValueError(f"알 수 없는 전략: {strategy_name}")

    mode, stop_type = mode_map[strategy_name]

    prices = prices.copy()
    signals = signals.copy()

    prices["Date"] = pd.to_datetime(prices["Date"])
    signals["Date"] = pd.to_datetime(signals["Date"])

    dates = sorted(prices["Date"].unique())

    price_groups = {
        date: group.set_index("Ticker", drop=False)
        for date, group in prices.groupby("Date")
    }

        # 종목별 마지막 유효 종가
    last_close = {}

    # 가격 누락 기록
    missing_price_log = []

    signal_groups = {
        date: group.sort_values(
            "Probability", ascending=False
        )
        for date, group in signals.groupby("Date")
    }

    cash = float(INITIAL_CASH)
    positions = []
    pending_entries = []
    trades = []
    equity_rows = []

    market_filter = (
    load_market_filter()
    if USE_MARKET_FILTER
    else {}
)

    for i, date in enumerate(dates):

        day_prices = price_groups[date]
        date = pd.Timestamp(date)

        # 오늘 확인 가능한 종가 갱신
        for _, bar in day_prices.iterrows():
            ticker = bar["Ticker"]
            close = float(bar["Close"])

            if np.isfinite(close) and close > 0:
                last_close[ticker] = close

        # 1. 오늘 진입 예정인 신호를 오늘 시가에 매수
        due = [
            x for x in pending_entries
            if x["Entry_Date"] == date
        ]

        # 기존 매수·매도 로직 계속
        pending_entries = [
            x for x in pending_entries
            if x["Entry_Date"] != date
        ]

        for item in due:
            ticker = item["Ticker"]

            if ticker not in day_prices.index:
                continue

            if len(positions) >= MAX_POSITIONS:
                break

            if any(p["Ticker"] == ticker for p in positions):
                continue

            bar = day_prices.loc[ticker]
            if isinstance(bar, pd.DataFrame):
                bar = bar.iloc[-1]

            atr = float(item["ATR14"])
            raw_entry = float(bar["Open"])

            if not (
                np.isfinite(atr) and atr > 0
                and np.isfinite(raw_entry) and raw_entry > 0
            ):
                continue

            entry_price = raw_entry * (1 + SLIPPAGE)

            if stop_type == "fixed":
                target = entry_price * (
                    1.02 if mode == "day" else 1.05
                )
                stop = entry_price * (
                    0.985 if mode == "day" else 0.97
                )
            else:
                target = entry_price + atr * (
                    1.5 if mode == "day" else 2.5
                )

                atr_stop = entry_price - atr * (
                    1.0 if mode == "day" else 1.5
                )

                # 기존 ATR 손절가와 -8% 손절가 중 높은 가격 선택
                if mode == "swing":
                    stop = max(
                        atr_stop,
                        entry_price * 0.92
                    )
                else:
                    stop = atr_stop

            slots_left = MAX_POSITIONS - len(positions)
            allocation = cash / slots_left
            shares = int(
                allocation / (entry_price * (1 + COMMISSION))
            )

            if shares <= 0:
                continue

            entry_cost = shares * entry_price * (1 + COMMISSION)
            if entry_cost > cash:
                continue

            cash -= entry_cost

            positions.append({
                "Ticker": ticker,
                "Signal_Date": item["Signal_Date"],
                "Entry_Date": date,
                "Entry_Price": entry_price,
                "Entry_ATR": atr,
                "Target_Price": target,
                "Stop_Price": stop,
                "Shares": shares,
                "Entry_Cost": entry_cost,
                "Probability": item["Probability"],
            })

        # 2. 기존 보유 포지션의 청산 여부 확인
        remaining = []

        for pos in positions:
            ticker = pos["Ticker"]

            if ticker not in day_prices.index:
                remaining.append(pos)
                continue

            bar = day_prices.loc[ticker]
            if isinstance(bar, pd.DataFrame):
                bar = bar.iloc[-1]

            # 거래일 기준 보유 일수
            entry_idx = dates.index(pos["Entry_Date"])
            held_days = dates.index(date) - entry_idx + 1

            raw_exit = None
            reason = None

            if bar["Open"] <= pos["Stop_Price"]:
                raw_exit, reason = bar["Open"], "손절 갭"
            elif bar["Open"] >= pos["Target_Price"]:
                raw_exit, reason = bar["Open"], "익절 갭"
            else:
                hit_stop = bar["Low"] <= pos["Stop_Price"]
                hit_target = bar["High"] >= pos["Target_Price"]

                if hit_stop:
                    raw_exit, reason = pos["Stop_Price"], (
                        "양쪽 도달: 손절 우선"
                        if hit_target else "손절"
                    )
                elif hit_target:
                    raw_exit, reason = pos["Target_Price"], "익절"
                elif mode == "day":
                    raw_exit, reason = bar["Close"], "당일 종가 청산"
                elif held_days >= 5:
                    raw_exit, reason = bar["Close"], "최대 보유기간 종료"

            if raw_exit is None:
                remaining.append(pos)
                continue

            exit_price, net_proceeds, pnl = calc_exit_pnl(
                pos, raw_exit, pos["Shares"]
            )
            cash += net_proceeds

            trades.append({
                "Strategy": strategy_name,
                "Ticker": ticker,
                "Signal_Date": pos["Signal_Date"],
                "Entry_Date": pos["Entry_Date"],
                "Exit_Date": date,
                "Entry_Price": pos["Entry_Price"],
                "Entry_ATR": pos["Entry_ATR"],
                "Target_Price": pos["Target_Price"],
                "Stop_Price": pos["Stop_Price"],
                "Raw_Exit_Price": raw_exit,
                "Exit_Price_After_Slippage": exit_price,
                "Exit_Open": bar["Open"],
                "Exit_High": bar["High"],
                "Exit_Low": bar["Low"],
                "Exit_Close": bar["Close"],
                "Shares": pos["Shares"],
                "Entry_Cost": pos["Entry_Cost"],
                "Net_Proceeds": net_proceeds,
                "PnL": pnl,
                "Return_Pct": pnl / pos["Entry_Cost"] * 100,
                "Holding_Days": held_days,
                "Exit_Reason": reason,
                "Probability": pos["Probability"],
            })

        positions = remaining

        # 3. 오늘 신호는 다음 거래일 진입 후보로만 예약
        if i + 1 < len(dates):
            next_date = pd.Timestamp(dates[i + 1])

            # 신호 발생일 기준 코스피 시장 필터
            market_ok = (
                market_filter.get(date, False)
                if USE_MARKET_FILTER
                else True
            )

            for _, signal in signal_groups.get(
                date, pd.DataFrame()
            ).iterrows():
                ticker = signal["Ticker"]

                if any(p["Ticker"] == ticker for p in positions):
                    continue

                if any(
                    x["Ticker"] == ticker
                    for x in pending_entries
                ):
                    continue

                # 코스피가 20일 이동평균선 아래면 신규 매수 예약 금지
                if not market_ok:
                    continue

                pending_entries.append({
                    "Ticker": ticker,
                    "Signal_Date": date,
                    "Entry_Date": next_date,
                    "ATR14": float(signal["ATR14"]),
                    "Probability": float(signal["Probability"]),
                })

    # 4. 오늘 종가 기준 자산 평가
        market_value = 0.0

        for pos in positions:
            ticker = pos["Ticker"]

            if ticker in day_prices.index:
                bar = day_prices.loc[ticker]

                if isinstance(bar, pd.DataFrame):
                    bar = bar.iloc[-1]

                close = float(bar["Close"])

                if np.isfinite(close) and close > 0:
                    last_close[ticker] = close
                else:
                    close = last_close.get(ticker)

            else:
                close = last_close.get(ticker)

                missing_price_log.append({
                    "Date": date,
                    "Ticker": ticker,
                    "Last_Close": close
                })

            if close is None or not np.isfinite(close) or close <= 0:
                raise ValueError(
                    f"{date.date()} {ticker}: "
                    "평가에 사용할 유효한 가격이 없습니다."
                )

            market_value += close * pos["Shares"]

        equity_rows.append({
            "Date": date,
            "Cash": cash,
            "Market_Value": market_value,
            "Equity": cash + market_value,
            "Open_Positions": len(positions),
        })

    # 5. 마지막 날짜까지 남은 포지션 강제 청산
    if positions:
        final_date = pd.Timestamp(dates[-1])
        final_prices = price_groups[dates[-1]]

        for pos in positions:
            ticker = pos["Ticker"]

            if ticker in final_prices.index:
                bar = final_prices.loc[ticker]
                if isinstance(bar, pd.DataFrame):
                    bar = bar.iloc[-1]
                raw_exit = float(bar["Close"])
            else:
                raw_exit = pos["Entry_Price"]

            exit_price, net_proceeds, pnl = calc_exit_pnl(
                pos, raw_exit, pos["Shares"]
            )
            cash += net_proceeds

            trades.append({
                "Strategy": strategy_name,
                "Ticker": ticker,
                "Signal_Date": pos["Signal_Date"],
                "Entry_Date": pos["Entry_Date"],
                "Exit_Date": final_date,
                "Entry_Price": pos["Entry_Price"],
                "Entry_ATR": pos["Entry_ATR"],
                "Target_Price": pos["Target_Price"],
                "Stop_Price": pos["Stop_Price"],
                "Raw_Exit_Price": raw_exit,
                "Exit_Price_After_Slippage": exit_price,
                "Exit_Open": np.nan,
                "Exit_High": np.nan,
                "Exit_Low": np.nan,
                "Exit_Close": raw_exit,
                "Shares": pos["Shares"],
                "Entry_Cost": pos["Entry_Cost"],
                "Net_Proceeds": net_proceeds,
                "PnL": pnl,
                "Return_Pct": pnl / pos["Entry_Cost"] * 100,
                "Holding_Days": sum(
                    d >= pos["Entry_Date"] for d in dates
                ),
                "Exit_Reason": "최종일 강제 청산",
                "Probability": pos["Probability"],
            })

        if equity_rows:
            equity_rows[-1]["Cash"] = cash
            equity_rows[-1]["Market_Value"] = 0.0
            equity_rows[-1]["Equity"] = cash
            equity_rows[-1]["Open_Positions"] = 0

    return pd.DataFrame(trades), pd.DataFrame(equity_rows)



from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score

BASE_DIR = Path(__file__).resolve().parent
TRAIN_FILE = BASE_DIR / "training_data_v4_lending.csv"
PRICE_FILE = BASE_DIR / "stock_data_500.csv"

INITIAL_CASH = 1_000_000.0
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

LENDING = "Lending_Balance_Change_Pct"
LAG_FEATURE = "Lending_Balance_Change_Pct_Lag1"


def normalize_ticker(value):
    value = str(value).strip()

    if value.endswith(".KS") or value.endswith(".KQ"):
        value = value.rsplit(".", 1)[0]

    return value.zfill(6)


def summarize_results(trades, equity, strategy_name):
    if equity.empty:
        return {
            "Strategy": strategy_name,
            "Final_Equity": INITIAL_CASH,
            "Return_Pct": 0.0,
            "MDD_Pct": 0.0,
            "Trades": 0,
            "Win_Rate_Pct": 0.0,
            "Total_PnL": 0.0,
        }

    # 일별 평가자산을 기준으로 최대낙폭 계산
    values = equity["Equity"].astype(float)
    running_peak = values.cummax()
    drawdown = (values / running_peak - 1) * 100
    mdd = float(drawdown.min())

    final_equity = float(values.iloc[-1])

    if trades.empty:
        total_pnl = 0.0
        win_rate = 0.0
        trade_count = 0
    else:
        total_pnl = float(trades["PnL"].sum())
        win_rate = float((trades["PnL"] > 0).mean() * 100)
        trade_count = len(trades)

    return {
        "Strategy": strategy_name,
        "Final_Equity": round(final_equity, 2),
        "Return_Pct": round(
            (final_equity / INITIAL_CASH - 1) * 100, 2
        ),
        "MDD_Pct": round(mdd, 2),
        "Trades": trade_count,
        "Win_Rate_Pct": round(win_rate, 2),
        "Total_PnL": round(total_pnl, 2),
    }


def calc_exit_pnl(pos, raw_exit, shares):
    # 매도 시 슬리피지 반영
    exit_price = float(raw_exit) * (1 - SLIPPAGE)

    # 매도 수수료와 매도 관련 세금 반영
    net_proceeds = (
        exit_price
        * shares
        * (1 - COMMISSION - SELL_TAX)
    )

    # 매수 시 발생한 비용을 차감한 순손익
    pnl = net_proceeds - pos["Entry_Cost"]

    return exit_price, net_proceeds, pnl


def main():
    prices = load_prices()

    # 가격 데이터와 신호 데이터의 종목 코드 형식을 통일
    prices["Ticker"] = prices["Ticker"].map(normalize_ticker)

    signals, boundary, ap = load_signals(prices)

    signals.to_csv(
        BASE_DIR / "v4_all_signals.csv",
        index=False,
        encoding="utf-8-sig",
     )

    print("\n===== 데이터 확인 =====")
    print("평가 시작일:", boundary.date())
    print("매수 후보 신호 수:", len(signals))
    print("Average Precision:", round(ap, 4))

    strategies = [
        "DAY_FIXED",
        "DAY_ATR",
        "SWING_FIXED",
        "SWING_ATR",
    ]

    all_trades = []
    all_equity = []
    summaries = []

    for strategy in strategies:
        print(f"\n===== {strategy} 실행 =====")

        trades, equity = run_backtest_v4(
            prices, signals, strategy
        )

        all_trades.append(trades)
        all_equity.append(equity)

        summary = summarize_results(
            trades, equity, strategy
        )
        summaries.append(summary)

        print(pd.Series(summary).to_string())

        trades.to_csv(
            BASE_DIR / f"v4_{strategy.lower()}_trades.csv",
            index=False,
            encoding="utf-8-sig",
        )

        equity.to_csv(
            BASE_DIR / f"v4_{strategy.lower()}_equity.csv",
            index=False,
            encoding="utf-8-sig",
        )

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(
        BASE_DIR / "v4_strategy_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    if all_trades:
        pd.concat(all_trades, ignore_index=True).to_csv(
            BASE_DIR / "v4_all_trades.csv",
            index=False,
            encoding="utf-8-sig",
        )

    print("\n===== 전략별 최종 비교 =====")
    print(summary_df.to_string(index=False))
    print("\nCSV 저장 완료")


if __name__ == "__main__":
    main()


