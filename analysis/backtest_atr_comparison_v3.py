
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score

BASE_DIR = Path(__file__).resolve().parent
TRAIN_FILE = BASE_DIR / "training_data_v4_lending.csv"
PRICE_FILE = BASE_DIR / "stock_data_500.csv"

INITIAL_CASH = 1_000_000.0
MAX_POSITIONS = 5
THRESHOLD = 0.65

# 잠정 비용 가정: 실제 적용 전 증권사 수수료와 세금 확인
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

    # 비정상적인 가격 데이터 제외
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

    # V2와 동일한 단순 이동평균 ATR(14)
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

    # 신호가 생성된 날짜의 ATR 연결
    atr = prices[["Date", "Ticker", "ATR14"]]

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




def get_exit(bar, pos, held_days, mode):
    """
    반환값: (원시 청산 가격, 청산 사유)

    일봉 OHLC만으로 장중 목표가와 손절가의 도달 순서를
    알 수 없으므로 둘 다 도달하면 손절을 우선한다.
    """
    stop = pos["Stop_Price"]
    target = pos["Target_Price"]

    # 시가 갭 발생 시 시가로 청산
    if bar.Open <= stop:
        return bar.Open, "손절 갭"

    if bar.Open >= target:
        return bar.Open, "익절 갭"

    hit_stop = bar.Low <= stop
    hit_target = bar.High >= target

    if hit_stop and hit_target:
        return stop, "양쪽 도달: 손절 우선"

    if hit_stop:
        return stop, "손절"

    if hit_target:
        return target, "익절"

    # 당일 매매는 당일 종가에 청산
    if mode.startswith("day"):
        return bar.Close, "당일 종가 청산"

    # 스윙 매매는 최대 5거래일 보유
    if held_days >= 5:
        return bar.Close, "최대 보유기간 종료"

    return None, None


def calc_exit_pnl(pos, raw_exit, shares):
    """
    원시 청산 가격에 슬리피지, 매도 수수료, 매도세를 반영한다.
    """
    exit_price = raw_exit * (1 - SLIPPAGE)
    gross_proceeds = exit_price * shares

    net_proceeds = gross_proceeds * (
        1 - COMMISSION - SELL_TAX
    )

    pnl = net_proceeds - pos["Entry_Cost"]

    return exit_price, net_proceeds, pnl



def run_backtest(prices, signals, strategy_name):
    cash = INITIAL_CASH
    positions = []
    trades = []
    equity_rows = []

    mode_map = {
        "DAY_FIXED": ("day", "fixed"),
        "DAY_ATR": ("day", "atr"),
        "SWING_FIXED": ("swing", "fixed"),
        "SWING_ATR": ("swing", "atr"),
    }

    if strategy_name not in mode_map:
        raise ValueError(f"알 수 없는 전략: {strategy_name}")

    mode, stop_type = mode_map[strategy_name]

    price_dates = sorted(prices["Date"].unique())
    price_groups = {
        date: group.set_index("Ticker", drop=False)
        for date, group in prices.groupby("Date")
    }

    signal_groups = {
        date: group.sort_values(
            "Probability", ascending=False
        )
        for date, group in signals.groupby("Date")
    }

    for date_index, date in enumerate(price_dates):
        day_prices = price_groups[date]
        next_date = (
            price_dates[date_index + 1]
            if date_index + 1 < len(price_dates)
            else None
        )

        # 1. 기존 보유 종목의 청산 여부 확인
        remaining_positions = []

        for pos in positions:
            ticker = pos["Ticker"]

            if ticker not in day_prices.index:
                remaining_positions.append(pos)
                continue

            bar = day_prices.loc[ticker]

            if isinstance(bar, pd.DataFrame):
                bar = bar.iloc[-1]

            held_days = (
                pd.Timestamp(date) - pos["Entry_Date"]
            ).days

            # 거래일 기준 보유일 계산
            held_days = sum(
                d >= pos["Entry_Date"] and d <= date
                for d in price_dates
            )

            raw_exit, reason = get_exit(
                bar, pos, held_days, mode
            )

            if raw_exit is None:
                remaining_positions.append(pos)
                continue

            exit_price, net_proceeds, pnl = calc_exit_pnl(
                pos, raw_exit, pos["Shares"]
            )

            cash += net_proceeds

            trades.append({
                "Strategy": strategy_name,
                "Ticker": ticker,
                "Entry_Date": pos["Entry_Date"],
                "Exit_Date": pd.Timestamp(date),
                "Entry_Price": pos["Entry_Price"],
                "Entry_ATR": pos["Entry_ATR"],
                "Target_Price": pos["Target_Price"],
                "Stop_Price": pos["Stop_Price"],
                "Raw_Exit_Price": raw_exit,
                "Exit_Price_After_Slippage": exit_price,
                "Exit_Open": bar.Open,
                "Exit_High": bar.High,
                "Exit_Low": bar.Low,
                "Exit_Close": bar.Close,
                "Shares": pos["Shares"],
                "Entry_Cost": pos["Entry_Cost"],
                "Net_Proceeds": net_proceeds,
                "PnL": pnl,
                "Return_Pct": (
                    pnl / pos["Entry_Cost"] * 100
                    if pos["Entry_Cost"] else 0
                ),
                "Holding_Days": held_days,
                "Exit_Reason": reason,
                "Probability": pos["Probability"],
            })

        positions = remaining_positions

        # 2. 다음 거래일에 진입할 후보 선정
        if next_date is not None and len(positions) < MAX_POSITIONS:
            candidates = signal_groups.get(date, pd.DataFrame())

            for _, signal in candidates.iterrows():
                if len(positions) >= MAX_POSITIONS:
                    break

                ticker = signal["Ticker"]

                if any(p["Ticker"] == ticker for p in positions):
                    continue

                next_prices = price_groups[next_date]

                if ticker not in next_prices.index:
                    continue

                next_bar = next_prices.loc[ticker]

                if isinstance(next_bar, pd.DataFrame):
                    next_bar = next_bar.iloc[-1]

                entry_atr = float(signal["ATR14"])

                if not np.isfinite(entry_atr) or entry_atr <= 0:
                    continue

                raw_entry = float(next_bar.Open)

                if not np.isfinite(raw_entry) or raw_entry <= 0:
                    continue

                entry_price = raw_entry * (1 + SLIPPAGE)

                # 목표가와 손절가 배수 설정
                if stop_type == "fixed":
                    if mode == "day":
                        target_price = entry_price * 1.02
                        stop_price = entry_price * 0.985
                    else:
                        target_price = entry_price * 1.05
                        stop_price = entry_price * 0.97
                else:
                    if mode == "day":
                        target_price = entry_price + 1.5 * entry_atr
                        stop_price = entry_price - 1.0 * entry_atr
                    else:
                        target_price = entry_price + 2.5 * entry_atr
                        stop_price = entry_price - 1.5 * entry_atr

                # 매수 수수료까지 포함한 수량 계산
                available_cash = cash
                if available_cash <= 0:
                    break

                allocation = available_cash / (
                    MAX_POSITIONS - len(positions)
                )

                shares = int(
                    allocation / (entry_price * (1 + COMMISSION))
                )

                if shares <= 0:
                    continue

                entry_cost = (
                    shares * entry_price * (1 + COMMISSION)
                )

                if entry_cost > cash:
                    continue

                cash -= entry_cost

                positions.append({
                    "Ticker": ticker,
                    "Entry_Date": pd.Timestamp(next_date),
                    "Entry_Price": entry_price,
                    "Entry_ATR": entry_atr,
                    "Target_Price": target_price,
                    "Stop_Price": stop_price,
                    "Shares": shares,
                    "Entry_Cost": entry_cost,
                    "Probability": float(signal["Probability"]),
                })

        # 3. 일별 자산 평가: 현금 + 보유 종목 평가액
        market_value = 0.0

        for pos in positions:
            ticker = pos["Ticker"]

            if ticker in day_prices.index:
                bar = day_prices.loc[ticker]

                if isinstance(bar, pd.DataFrame):
                    bar = bar.iloc[-1]

                market_value += (
                    float(bar.Close) * pos["Shares"]
                )
            else:
                market_value += (
                    pos["Entry_Price"] * pos["Shares"]
                )

        equity_rows.append({
            "Date": pd.Timestamp(date),
            "Cash": cash,
            "Market_Value": market_value,
            "Equity": cash + market_value,
            "Open_Positions": len(positions),
        })

    # 4. 마지막 날짜에 남은 종목은 마지막 종가로 강제 청산
    if positions:
        final_date = price_dates[-1]
        final_prices = price_groups[final_date]

        for pos in positions:
            ticker = pos["Ticker"]

            if ticker in final_prices.index:
                bar = final_prices.loc[ticker]

                if isinstance(bar, pd.DataFrame):
                    bar = bar.iloc[-1]

                raw_exit = float(bar.Close)
            else:
                raw_exit = pos["Entry_Price"]

            exit_price, net_proceeds, pnl = calc_exit_pnl(
                pos, raw_exit, pos["Shares"]
            )

            cash += net_proceeds

            trades.append({
                "Strategy": strategy_name,
                "Ticker": ticker,
                "Entry_Date": pos["Entry_Date"],
                "Exit_Date": pd.Timestamp(final_date),
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
                "Return_Pct": (
                    pnl / pos["Entry_Cost"] * 100
                    if pos["Entry_Cost"] else 0
                ),
                "Holding_Days": sum(
                    d >= pos["Entry_Date"] and d <= final_date
                    for d in price_dates
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


def analyze_results(trades, equity, strategy_name):
    print("\n" + "=" * 65)
    print(f"전략: {strategy_name}")
    print("=" * 65)

    if equity.empty:
        print("자산 기록이 없습니다.")
        return

    equity = equity.copy()
    equity["Date"] = pd.to_datetime(equity["Date"])
    equity = equity.sort_values("Date")

    start_equity = float(equity.iloc[0]["Equity"])
    final_equity = float(equity.iloc[-1]["Equity"])
    peak = equity["Equity"].cummax()
    drawdown = equity["Equity"] / peak - 1
    mdd = float(drawdown.min() * 100)

    print(f"초기 자산: {INITIAL_CASH:,.0f}원")
    print(f"시작 기록 자산: {start_equity:,.0f}원")
    print(f"최종 자산: {final_equity:,.0f}원")
    print(f"총 수익: {final_equity - INITIAL_CASH:,.0f}원")
    print(
        f"총 수익률: "
        f"{(final_equity / INITIAL_CASH - 1) * 100:.2f}%"
    )
    print(f"최대 낙폭(MDD): {mdd:.2f}%")

    if trades.empty:
        print("완료된 거래가 없습니다.")
        return

    trades = trades.copy()
    trades["Entry_Date"] = pd.to_datetime(trades["Entry_Date"])
    trades["Exit_Date"] = pd.to_datetime(trades["Exit_Date"])
    trades = trades.sort_values("PnL", ascending=False)

    total_pnl = float(trades["PnL"].sum())
    wins = trades[trades["PnL"] > 0]
    losses = trades[trades["PnL"] < 0]

    print(f"\n총 거래 수: {len(trades)}")
    print(f"수익 거래: {len(wins)}")
    print(f"손실 거래: {len(losses)}")
    print(f"승률: {len(wins) / len(trades) * 100:.2f}%")
    print(f"거래당 평균 수익률: {trades['Return_Pct'].mean():.2f}%")
    print(f"거래 손익 합계: {total_pnl:,.0f}원")

    if not wins.empty:
        print(f"평균 수익 거래 손익: {wins['PnL'].mean():,.0f}원")

    if not losses.empty:
        print(f"평균 손실 거래 손익: {losses['PnL'].mean():,.0f}원")
        print(f"최대 손실 거래: {losses['PnL'].min():,.0f}원")
        print(f"최저 거래 수익률: {losses['Return_Pct'].min():.2f}%")

    # 상위 수익 거래에 대한 의존도 확인
    positive_pnl = wins["PnL"].sum()

    if positive_pnl > 0:
        top_n = min(5, len(wins))
        top_winners = wins.head(top_n)
        top_pnl = float(top_winners["PnL"].sum())

        print(f"\n상위 {top_n}개 수익 거래 손익: {top_pnl:,.0f}원")
        print(
            "전체 수익 거래 손익 중 상위 거래 비중: "
            f"{top_pnl / positive_pnl * 100:.2f}%"
        )

        remaining = trades.drop(top_winners.index)
        print(
            f"상위 {top_n}개 거래 제외 후 거래 손익 합계: "
            f"{remaining['PnL'].sum():,.0f}원"
        )

    # 심각한 손실 거래 확인
    severe = trades[trades["Return_Pct"] <= -5]

    print(f"\n수익률 -5% 이하 거래: {len(severe)}건")

    if not severe.empty:
        cols = [
            "Ticker", "Entry_Date", "Exit_Date",
            "Return_Pct", "PnL", "Exit_Reason",
        ]
        print(severe[cols].to_string(index=False))

    # 월별 청산 손익: 거래가 종료된 월 기준
    trades["Exit_Month"] = trades["Exit_Date"].dt.to_period("M").astype(str)
    monthly = trades.groupby("Exit_Month", as_index=False).agg(
        Trade_Count=("PnL", "size"),
        Monthly_Trade_PnL=("PnL", "sum"),
        Average_Trade_Return=("Return_Pct", "mean"),
    )

    print("\n월별 청산 거래 손익")
    print(monthly.to_string(index=False))

    # 청산 손익과 최종 자산 변화가 일치하는지 확인
    expected_equity = INITIAL_CASH + total_pnl
    difference = final_equity - expected_equity

    print("\n[손익 검산]")
    print(f"초기 자산 + 거래 손익: {expected_equity:,.0f}원")
    print(f"실제 최종 자산: {final_equity:,.0f}원")
    print(f"차이: {difference:,.0f}원")

    if abs(difference) > 1:
        print("주의: 자산 변화와 거래 손익이 일치하지 않습니다.")
    else:
        print("검산 통과: 자산 변화와 거래 손익이 일치합니다.")


def main():
    prices = load_prices()
    signals, boundary, ap = load_signals(prices)

    print("\n평가 시작일:", boundary.date())
    print("Average Precision:", round(ap, 4))
    print("확률 기준:", THRESHOLD)
    print("AI 매매 신호 수:", len(signals))

    all_trades = []
    all_equity = []

    strategies = [
        "DAY_FIXED",
        "DAY_ATR",
        "SWING_FIXED",
        "SWING_ATR",
    ]

    for strategy in strategies:
        trades, equity = run_backtest(
            prices, signals, strategy
        )

        if not trades.empty:
            all_trades.append(trades)

        if not equity.empty:
            equity["Strategy"] = strategy
            all_equity.append(equity)

        analyze_results(trades, equity, strategy)

    trades_file = BASE_DIR / "backtest_atr_v3_trades.csv"
    equity_file = BASE_DIR / "backtest_atr_v3_equity.csv"

    if all_trades:
        combined_trades = pd.concat(
            all_trades, ignore_index=True
        )
        combined_trades.to_csv(
            trades_file, index=False, encoding="utf-8-sig"
        )
        print("\n거래 내역 저장:", trades_file)
    else:
        print("\n저장할 거래 내역이 없습니다.")

    if all_equity:
        combined_equity = pd.concat(
            all_equity, ignore_index=True
        )
        combined_equity.to_csv(
            equity_file, index=False, encoding="utf-8-sig"
        )
        print("자산 기록 저장:", equity_file)

    print("\nV3 백테스트 완료")
    print("기존 V2 파일은 수정하지 않았습니다.")


if __name__ == "__main__":
    main()




