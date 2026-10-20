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

# HYBRID: 시장 필터 미통과 시 개별 종목 MA20 확인
old_market = """                if not market_ok:
                    continue"""

new_market = """                if not market_ok:
                    stock_row = stock_filter.get((date, ticker))

                    if stock_row is None:
                        continue

                    if not (
                        pd.notna(stock_row["Stock_MA20"])
                        and stock_row["Close"] > stock_row["Stock_MA20"]
                    ):
                        continue"""

# 청산 직후 쿨다운 시작
old_exit = """            cash += net_proceeds

            trades.append({"""

new_exit = """            cash += net_proceeds

            if "손절" in str(reason):
                cooldown_until[ticker] = i + cooldown_days

            trades.append({"""

# 다음 거래일 예약 시 쿨다운 검사
old_signal = """                ticker = signal["Ticker"]

                if any(p["Ticker"] == ticker for p in positions):"""

new_signal = """                ticker = signal["Ticker"]

                if cooldown_days > 0 and i <= cooldown_until.get(ticker, -1):
                    continue

                if any(p["Ticker"] == ticker for p in positions):"""

# 실제 매수 시점에도 검사
old_entry = """        for item in due:
            ticker = item["Ticker"]

            if ticker not in day_prices.index:"""

new_entry = """        for item in due:
            ticker = item["Ticker"]

            if cooldown_days > 0 and i <= cooldown_until.get(ticker, -1):
                continue

            if ticker not in day_prices.index:"""

# 함수 내부 상태 초기화
old_init = """    trades = []
    equity_rows = []"""

new_init = """    trades = []
    equity_rows = []
    cooldown_until = {}"""

replacements = [
    (old_market, new_market),
    (old_exit, new_exit),
    (old_signal, new_signal),
    (old_entry, new_entry),
    (old_init, new_init),
]

for old, new in replacements:
    count = source.count(old)

    expected = 2 if old == old_exit else 1

    if count != expected:
        raise RuntimeError(
            f"코드 교체 위치 확인 실패: {count}개 "
            f"(예상: {expected}개)\n{old}"
        )

    source = source.replace(old, new, 1)

source = source.replace(
    "def run_backtest_v4(",
    "def run_backtest_cooldown(",
    1
)

bt.INITIAL_CASH = 1_000_000
bt.MAX_POSITIONS = 2
bt.MARKET_MA_DAYS = 40
bt.USE_MARKET_FILTER = True

results = []

for cooldown_days in [0, 3, 5, 10]:
    namespace = dict(vars(bt))
    namespace["stock_filter"] = stock_filter
    namespace["cooldown_days"] = cooldown_days

    exec(source, namespace)

    runner = namespace["run_backtest_cooldown"]

    trades, equity = runner(
        prices,
        signals,
        "SWING_ATR"
    )

    name = f"HYBRID_CD{cooldown_days}"

    result = bt.summarize_results(
        trades, equity, name
    )

    results.append(result)

    trades.to_csv(
        BASE / f"cooldown_{name}_trades.csv",
        index=False,
        encoding="utf-8-sig"
    )

    equity.to_csv(
        BASE / f"cooldown_{name}_equity.csv",
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

print("\n===== HYBRID 쿨다운 비교 =====")
print(
    summary[columns]
    .round(2)
    .to_string(index=False)
)

summary.to_csv(
    BASE / "cooldown_comparison.csv",
    index=False,
    encoding="utf-8-sig"
)
