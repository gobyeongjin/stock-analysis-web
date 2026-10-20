
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score

# =====================================================
# 1. 설정
# =====================================================
BASE_DIR = Path(__file__).resolve().parent

TRAIN_FILE = BASE_DIR / "training_data_v4_lending.csv"
PRICE_FILE = BASE_DIR / "stock_data_500.csv"

INITIAL_CASH = 1_000_000
MAX_POSITIONS = 5

# 초기 실험용 가정: 실제 증권사 비용에 맞춰 수정할 것
COMMISSION_RATE = 0.00015
SELL_TAX_RATE = 0.0015
SLIPPAGE_RATE = 0.0005

# 최초 비교할 AI 확률 기준
THRESHOLD = 0.65

# 단타
DAY_TAKE_PROFIT = 0.02
DAY_STOP_LOSS = 0.015

# 스윙
SWING_TAKE_PROFIT = 0.05
SWING_STOP_LOSS = 0.03
SWING_MAX_DAYS = 5

FEATURES = [
    "MA20_Gap",
    "MA60_Gap",
    "MA20_MA60_Gap",
    "MA20_Change",
    "MA60_Change",
    "High_20D_Gap",
    "Low_20D_Gap",
    "RSI",
    "RSI_Change",
    "Volume_Ratio",
    "Volume_Ratio_5D",
    "Return_1D",
    "Return_3D",
    "Return_5D",
    "Return_10D",
    "Volatility_20D",
    "Volatility_Change",
    "Close_Position",
]

LENDING_FEATURE = "Lending_Balance_Change_Pct"


def normalize_ticker(value):
    """005930.KS, 005930, 5930 등의 코드를 6자리 문자열로 통일."""
    value = str(value).strip()
    if value.endswith(".KS") or value.endswith(".KQ"):
        value = value.rsplit(".", 1)[0]
    return value.zfill(6)


def load_data():
    if not TRAIN_FILE.exists():
        raise FileNotFoundError(f"학습 데이터가 없습니다: {TRAIN_FILE}")

    if not PRICE_FILE.exists():
        raise FileNotFoundError(f"주가 데이터가 없습니다: {PRICE_FILE}")

    df = pd.read_csv(TRAIN_FILE, dtype={"Ticker": str})
    prices = pd.read_csv(PRICE_FILE, dtype={"Ticker": str})

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    prices["Date"] = pd.to_datetime(prices["Date"], errors="coerce")

    df["Ticker"] = df["Ticker"].map(normalize_ticker)
    prices["Ticker"] = prices["Ticker"].map(normalize_ticker)

    # 원본 학습 데이터는 수정하지 않고 메모리에서만 처리
    df = df.sort_values(["Ticker", "Date"]).copy()

    # 대차잔고 변화율의 전 관측값 사용
    lag_feature = "Lending_Balance_Change_Pct_Lag1"
    df[lag_feature] = (
        df.groupby("Ticker")[LENDING_FEATURE].shift(1)
    )

    model_features = FEATURES + [lag_feature]

    for col in model_features + ["Label"]:
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
        subset=model_features + ["Label", "Date", "Ticker"]
    )
    df = df[df["Label"].isin([0, 1])].copy()
    df["Label"] = df["Label"].astype(int)

    price_columns = ["Date", "Ticker", "Open", "High", "Low", "Close"]
    missing = [c for c in price_columns if c not in prices.columns]
    if missing:
        raise ValueError(f"주가 데이터에 필요한 컬럼이 없습니다: {missing}")

    for col in ["Open", "High", "Low", "Close"]:
        prices[col] = pd.to_numeric(prices[col], errors="coerce")

    prices = prices.dropna(subset=price_columns)
    prices = prices.sort_values(["Ticker", "Date"])
    prices = prices.drop_duplicates(["Ticker", "Date"], keep="last")

    # 학습 데이터와 주가 데이터의 날짜/종목을 연결
    df = df.sort_values(["Date", "Ticker"]).reset_index(drop=True)

    # 모든 종목이 같은 날에 거래되지 않을 수 있으므로
    # 실제 해당 종목의 OHLC 행은 별도로 조회한다.
    return df, prices, model_features


def apply_cost(price, side):
    """슬리피지를 불리한 방향으로 적용."""
    if side == "buy":
        return price * (1 + SLIPPAGE_RATE)
    return price * (1 - SLIPPAGE_RATE)


def backtest(strategy_name, signal_table, prices):
    if strategy_name == "day":
        tp_rate = DAY_TAKE_PROFIT
        sl_rate = DAY_STOP_LOSS
        max_days = 1
    else:
        tp_rate = SWING_TAKE_PROFIT
        sl_rate = SWING_STOP_LOSS
        max_days = SWING_MAX_DAYS

    price_lookup = {
        (row.Ticker, row.Date): row
        for row in prices.itertuples(index=False)
    }

    trading_dates = sorted(prices["Date"].unique())

    # 신호는 종가 기준으로 생성하고 다음 거래일 시가에 매수
    signal_by_date = {}
    for row in signal_table.itertuples(index=False):
        signal_by_date.setdefault(row.Date, []).append(
            (row.Ticker, float(row.Probability))
        )

    cash = float(INITIAL_CASH)
    positions = {}
    trades = []

    for date_index, date in enumerate(trading_dates):
        # ---------------------------------------------
        # 1) 기존 보유 종목의 익절/손절/기간 종료 확인
        # ---------------------------------------------
        to_close = []

        for ticker, pos in positions.items():
            bar = price_lookup.get((ticker, date))
            if bar is None:
                continue

            entry = pos["entry_price"]
            target = entry * (1 + tp_rate)
            stop = entry * (1 - sl_rate)

            held_days = date_index - pos["entry_date_index"] + 1

            exit_price = None
            exit_reason = None

            # 진입일 시가에 매수한 뒤 같은 날의 고가/저가 확인
            if bar.Open <= stop:
                exit_price = bar.Open
                exit_reason = "손절(시가 갭)"
            elif bar.Open >= target:
                exit_price = bar.Open
                exit_reason = "익절(시가 갭)"
            elif bar.Low <= stop and bar.High >= target:
                # 일봉으로는 장중 순서를 알 수 없으므로 보수적으로 손절
                exit_price = stop
                exit_reason = "손절(익절·손절 모두 도달)"
            elif bar.Low <= stop:
                exit_price = stop
                exit_reason = "손절"
            elif bar.High >= target:
                exit_price = target
                exit_reason = "익절"
            elif held_days >= max_days:
                exit_price = bar.Close
                exit_reason = "기간 종료"

            if exit_price is not None:
                to_close.append(
                    (ticker, exit_price, exit_reason, date)
                )

        for ticker, raw_exit, reason, exit_date in to_close:
            pos = positions.pop(ticker)

            # 매도 슬리피지와 비용
            exit_price = apply_cost(raw_exit, "sell")
            gross = exit_price * pos["shares"]
            commission = gross * COMMISSION_RATE
            tax = gross * SELL_TAX_RATE
            net_proceeds = gross - commission - tax

            cash += net_proceeds

            total_cost = pos["buy_cost"]
            pnl = net_proceeds - total_cost
            pnl_pct = pnl / total_cost if total_cost else 0

            trades.append({
                "Strategy": strategy_name,
                "Ticker": ticker,
                "Signal_Date": pos["signal_date"],
                "Entry_Date": pos["entry_date"],
                "Exit_Date": exit_date,
                "Probability": pos["probability"],
                "Shares": pos["shares"],
                "Entry_Price": pos["entry_price"],
                "Exit_Price": exit_price,
                "Exit_Reason": reason,
                "Net_PnL": pnl,
                "Return_Pct": pnl_pct * 100,
                "Holding_Days": (
                    trading_dates.index(exit_date)
                    - pos["entry_date_index"] + 1
                ),
            })

        # ---------------------------------------------
        # 2) 전 거래일 신호를 오늘 시가에 매수
        # ---------------------------------------------
        if date_index == 0:
            continue

        previous_date = trading_dates[date_index - 1]
        candidates = signal_by_date.get(previous_date, [])

        # 확률이 높은 종목부터 검토
        candidates = sorted(
            candidates, key=lambda x: x[1], reverse=True
        )

        available_slots = MAX_POSITIONS - len(positions)
        if available_slots <= 0:
            continue

        candidates = [
            (ticker, prob)
            for ticker, prob in candidates
            if ticker not in positions
            and (ticker, date) in price_lookup
        ][:available_slots]

        if not candidates or cash <= 0:
            continue

        # 남은 현금을 후보 종목에 균등 배분
        allocation = cash / len(candidates)

        for ticker, probability in candidates:
            bar = price_lookup[(ticker, date)]

            # 매수는 다음 거래일 시가 기준
            raw_entry = bar.Open
            entry_price = apply_cost(raw_entry, "buy")

            # 수수료를 포함해 배분 금액 내에서 매수
            unit_cost = entry_price * (1 + COMMISSION_RATE)
            shares = int(allocation // unit_cost)

            if shares <= 0:
                continue

            buy_value = entry_price * shares
            buy_fee = buy_value * COMMISSION_RATE
            total_buy_cost = buy_value + buy_fee

            if total_buy_cost > cash:
                continue

            cash -= total_buy_cost

            positions[ticker] = {
                "shares": shares,
                "entry_price": entry_price,
                "entry_date": date,
                "entry_date_index": date_index,
                "signal_date": previous_date,
                "probability": probability,
                "buy_cost": total_buy_cost,
            }

    # 평가 기간 마지막 날까지 남은 포지션은 마지막 종가에 청산
    if trading_dates:
        last_date = trading_dates[-1]

        for ticker in list(positions.keys()):
            pos = positions.pop(ticker)
            bar = price_lookup.get((ticker, last_date))

            if bar is None:
                continue

            exit_price = apply_cost(bar.Close, "sell")
            gross = exit_price * pos["shares"]
            net_proceeds = (
                gross
                - gross * COMMISSION_RATE
                - gross * SELL_TAX_RATE
            )
            cash += net_proceeds

            pnl = net_proceeds - pos["buy_cost"]

            trades.append({
                "Strategy": strategy_name,
                "Ticker": ticker,
                "Signal_Date": pos["signal_date"],
                "Entry_Date": pos["entry_date"],
                "Exit_Date": last_date,
                "Probability": pos["probability"],
                "Shares": pos["shares"],
                "Entry_Price": pos["entry_price"],
                "Exit_Price": exit_price,
                "Exit_Reason": "평가 기간 종료",
                "Net_PnL": pnl,
                "Return_Pct": pnl / pos["buy_cost"] * 100,
                "Holding_Days": (
                    trading_dates.index(last_date)
                    - pos["entry_date_index"] + 1
                ),
            })

    trades_df = pd.DataFrame(trades)

    if trades_df.empty:
        print(f"\n[{strategy_name}] 거래가 발생하지 않았습니다.")
        return trades_df

    wins = (trades_df["Net_PnL"] > 0).sum()
    total_trades = len(trades_df)
    final_equity = INITIAL_CASH + trades_df["Net_PnL"].sum()

    print(f"\n===== {strategy_name.upper()} 전략 결과 =====")
    print("거래 횟수:", total_trades)
    print("승률:", round(wins / total_trades * 100, 2), "%")
    print("순손익:", round(trades_df["Net_PnL"].sum()), "원")
    print("최종 평가금액:", round(final_equity), "원")
    print(
        "수익률:",
        round((final_equity / INITIAL_CASH - 1) * 100, 2),
        "%",
    )
    print("평균 거래 수익률:", round(trades_df["Return_Pct"].mean(), 2), "%")

    return trades_df


# =====================================================
# 2. 학습 / 평가 분리
# =====================================================
df, prices, model_features = load_data()

dates = sorted(df["Date"].unique())
split_index = int(len(dates) * 0.8)

if split_index < 6 or split_index >= len(dates):
    raise ValueError("학습/평가 데이터가 부족합니다.")

split_date = pd.Timestamp(dates[split_index])

# 미래 5거래일 라벨이 경계를 넘지 않도록 일부 학습 날짜 제외
purge_dates = set(dates[max(0, split_index - 5):split_index])

train = df[
    (df["Date"] < split_date)
    & (~df["Date"].isin(purge_dates))
].copy()

test = df[df["Date"] >= split_date].copy()

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_leaf=10,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)

model.fit(train[model_features], train["Label"])

test["Probability"] = model.predict_proba(
    test[model_features]
)[:, 1]

print("===== 백테스트 설정 =====")
print("초기 투자금:", INITIAL_CASH, "원")
print("최대 보유 종목 수:", MAX_POSITIONS)
print("AI 임계값:", THRESHOLD)
print("평가 시작일:", split_date.date())
print("학습 종목 수:", train["Ticker"].nunique())
print("평가 신호 행 수:", len(test))
print(
    "Average Precision:",
    round(
        average_precision_score(test["Label"], test["Probability"]),
        4,
    ),
)

# 지정 임계값 이상인 종목만 매수 신호로 사용
signals = test.loc[
    test["Probability"] >= THRESHOLD,
    ["Date", "Ticker", "Probability"],
].copy()

# 주가 데이터에 실제 존재하는 날짜만 사용
available = prices[["Date", "Ticker"]].drop_duplicates()
signals = signals.merge(
    available,
    on=["Date", "Ticker"],
    how="inner",
)

print("AI 매수 신호 수:", len(signals))

# 두 전략 모두 같은 신호·같은 평가 기간 사용
day_trades = backtest("day", signals, prices)
swing_trades = backtest("swing", signals, prices)

# 결과를 별도 파일로 저장
if not day_trades.empty:
    day_trades.to_csv(
        BASE_DIR / "backtest_day_trades.csv",
        index=False,
        encoding="utf-8-sig",
    )

if not swing_trades.empty:
    swing_trades.to_csv(
        BASE_DIR / "backtest_swing_trades.csv",
        index=False,
        encoding="utf-8-sig",
    )

print("\n백테스트 완료. 기존 모델 파일은 수정하지 않았습니다.")
