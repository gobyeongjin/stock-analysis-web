from pathlib import Path
import inspect
import pandas as pd
import backtest_market_ma as bt

BASE = Path(__file__).resolve().parent

signals = pd.read_csv(
    BASE / "wf_498_signals.csv",
    dtype={"Ticker": str},
    parse_dates=["Date"]
)

metrics = pd.read_csv(
    BASE / "wf_model_metrics.csv",
    parse_dates=["Test_Start", "Test_End"]
)

signals["Ticker"] = signals["Ticker"].map(bt.normalize_ticker)
signals["Date"] = signals["Date"].dt.normalize()

start = metrics["Test_Start"].min()
end = metrics["Test_End"].max()

bt.PRICE_FILE = BASE / "stock_data_500_updated.csv"

prices_all = bt.load_prices()
prices_all["Date"] = pd.to_datetime(
    prices_all["Date"]
).dt.normalize()

prices_all = prices_all.sort_values(
    ["Ticker", "Date"]
).copy()

prices_all["Stock_MA20"] = (
    prices_all.groupby("Ticker")["Close"]
    .transform(
        lambda s: s.rolling(20, min_periods=20).mean()
    )
)

prices = prices_all[
    prices_all["Date"].between(start, end)
].copy()

signals = signals[
    signals["Date"].between(start, end)
].copy()

stock_filter = (
    prices_all.set_index(["Date", "Ticker"])[
        ["Close", "Stock_MA20"]
    ]
    .to_dict("index")
)

source = inspect.getsource(bt.run_backtest_v4)

def replace_once(old, new):
    global source
    count = source.count(old)
    if count != 1:
        raise RuntimeError(
            f"코드 위치 확인 실패: {count}개\n{old}"
        )
    source = source.replace(old, new, 1)

# 1. HYBRID 시장 필터
replace_once(
    """                if not market_ok:
                    continue""",
    """                if not market_ok:
                    stock_row = stock_filter.get((date, ticker))

                    if stock_row is None:
                        continue

                    if not (
                        pd.notna(stock_row["Stock_MA20"])
                        and stock_row["Close"] > stock_row["Stock_MA20"]
                    ):
                        continue"""
)

# 2. 쿨다운 상태 초기화
replace_once(
    """    trades = []
    equity_rows = []""",
    """    trades = []
    equity_rows = []
    cooldown_until = {}"""
)

# 3. 손절 발생 시 쿨다운 기록
old_exit = """            cash += net_proceeds

            trades.append({"""

if source.count(old_exit) != 2:
    raise RuntimeError("청산 코드 구조가 예상과 다릅니다.")

source = source.replace(
    old_exit,
    """            cash += net_proceeds

            if "손절" in str(reason):
                cooldown_until[ticker] = i + cooldown_days

            trades.append({""",
    1
)

# 4. 매수 시 쿨다운 검사
replace_once(
    """        for item in due:
            ticker = item["Ticker"]

            if ticker not in day_prices.index:""",
    """        for item in due:
            ticker = item["Ticker"]

            if cooldown_days > 0 and i <= cooldown_until.get(ticker, -1):
                continue

            if ticker not in day_prices.index:"""
)

# 5. 신호 예약 시 쿨다운 검사
replace_once(
    """                ticker = signal["Ticker"]

                if any(p["Ticker"] == ticker for p in positions):""",
    """                ticker = signal["Ticker"]

                if cooldown_days > 0 and i <= cooldown_until.get(ticker, -1):
                    continue

                if any(p["Ticker"] == ticker for p in positions):"""
)

# 6. 종목당 투자 비중 제한
replace_once(
    """            allocation = cash / slots_left
            shares = int(""",
    """            allocation = cash / slots_left

            if position_cap is not None:
                portfolio_value = cash

                for held in positions:
                    held_ticker = held["Ticker"]

                    if held_ticker in day_prices.index:
                        held_bar = day_prices.loc[held_ticker]

                        if isinstance(held_bar, pd.DataFrame):
                            held_bar = held_bar.iloc[-1]

                        held_price = float(held_bar["Open"])
                    else:
                        held_price = float(
                            last_close.get(held_ticker, 0)
                        )

                    portfolio_value += (
                        held["Shares"] * held_price
                    )

                allocation = min(
                    allocation,
                    portfolio_value * position_cap
                )

            if risk_pct is not None:
                stop_distance = entry_price - stop

                if (
                    not pd.notna(stop_distance)
                    or stop_distance <= 0
                ):
                    continue

                risk_budget = portfolio_value * risk_pct

                risk_allocation = (
                    risk_budget
                    * entry_price
                    / stop_distance
                )

                allocation = min(
                    allocation,
                    risk_allocation
                )

            shares = int("""
)

source = source.replace(
    "def run_backtest_v4(",
    "def run_backtest_position_cap(",
    1
)

bt.INITIAL_CASH = 1_000_000
bt.MAX_POSITIONS = 2
bt.MARKET_MA_DAYS = 40
bt.USE_MARKET_FILTER = True

results = []

for risk_pct in [None, 0.01, 0.015, 0.02]:
    cap = 0.25
    namespace = dict(vars(bt))
    namespace["stock_filter"] = stock_filter
    namespace["cooldown_days"] = 5
    namespace["position_cap"] = cap
    namespace["risk_pct"] = risk_pct

    exec(source, namespace)

    runner = namespace["run_backtest_position_cap"]

    trades, equity = runner(
        prices,
        signals,
        "SWING_ATR"
    )

    name = (
        "BASE_CAP25"
        if risk_pct is None
        else f"RISK_{risk_pct * 100:.1f}"
    )

    result = bt.summarize_results(
        trades, equity, name
    )

    results.append(result)

    trades.to_csv(
        BASE / f"atr_risk_{name}_trades.csv",
        index=False,
        encoding="utf-8-sig"
    )

    equity.to_csv(
        BASE / f"atr_risk_{name}_equity.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print(f"{name} 완료: {len(trades)}건")

summary = pd.DataFrame(results)

columns = [
    "Strategy",
    "Final_Equity",
    "Return_Pct",
    "MDD_Pct",
    "Trades",
    "Win_Rate_Pct",
]

print("\n===== CD5 투자 비중 제한 비교 =====")
print(
    summary[columns]
    .round(2)
    .to_string(index=False)
)

summary.to_csv(
    BASE / "atr_risk_comparison.csv",
    index=False,
    encoding="utf-8-sig"
)
