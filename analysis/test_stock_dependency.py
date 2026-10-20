from pathlib import Path
import inspect
import pandas as pd
import backtest_market_ma as bt

BASE = Path(__file__).resolve().parent

# 기존 AI 신호 불러오기
signals = pd.read_csv(
    BASE / "wf_all_signals.csv",
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

# 전체 주가 데이터에서 종목 MA20 계산
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

# 원본 엔진을 수정하지 않고 HYBRID 함수 생성
source = inspect.getsource(bt.run_backtest_v4)

old = """                if not market_ok:
                    continue"""

new = """                if not market_ok:
                    stock_row = stock_filter.get((date, ticker))

                    if stock_row is None:
                        continue

                    stock_close = stock_row["Close"]
                    stock_ma20 = stock_row["Stock_MA20"]

                    if not (
                        pd.notna(stock_ma20)
                        and stock_close > stock_ma20
                    ):
                        continue"""

if source.count(old) != 1:
    raise RuntimeError("시장 필터 코드 위치가 예상과 다릅니다.")

hybrid_source = source.replace(
    "def run_backtest_v4(",
    "def run_backtest_hybrid(",
    1
).replace(old, new, 1)

# 공통 설정
bt.INITIAL_CASH = 1_000_000
bt.MAX_POSITIONS = 2
bt.MARKET_MA_DAYS = 40
bt.USE_MARKET_FILTER = True

namespace = dict(vars(bt))
namespace["stock_filter"] = stock_filter

exec(hybrid_source, namespace)
run_hybrid = namespace["run_backtest_hybrid"]

# 종목 제외 실험
experiments = {
    "ALL": [],
    "NO_009150": ["009150"],
    "NO_402340": ["402340"],
    "NO_BOTH": ["009150", "402340"],
}

results = []

months = pd.period_range(
    start.to_period("M"),
    end.to_period("M"),
    freq="M"
)

for exp_name, excluded in experiments.items():

    test_signals = signals[
        ~signals["Ticker"].isin(excluded)
    ].copy()

    for strategy_name, runner in [
        ("MA40", bt.run_backtest_v4),
        ("HYBRID", run_hybrid),
    ]:

        trades, equity = runner(
            prices,
            test_signals,
            "SWING_ATR"
        )

        result = bt.summarize_results(
            trades,
            equity,
            f"{exp_name}_{strategy_name}"
        )

        if not trades.empty:
            monthly_counts = (
                trades.groupby(
                    pd.to_datetime(
                        trades["Entry_Date"]
                    ).dt.to_period("M")
                )
                .size()
                .reindex(months, fill_value=0)
            )
        else:
            monthly_counts = pd.Series(
                0, index=months
            )

        result["Experiment"] = exp_name
        result["Filter"] = strategy_name
        result["Zero_Months"] = int(
            (monthly_counts == 0).sum()
        )

        results.append(result)

        print(
            f"{exp_name} / {strategy_name}: "
            f"{len(trades)}건, "
            f"{result['Return_Pct']:.2f}%"
        )

summary = pd.DataFrame(results)

columns = [
    "Experiment",
    "Filter",
    "Return_Pct",
    "MDD_Pct",
    "Trades",
    "Zero_Months",
    "Win_Rate_Pct",
    "Final_Equity",
]

print("\n===== 종목 의존도 비교 =====")
print(
    summary[columns]
    .round(2)
    .to_string(index=False)
)

summary.to_csv(
    BASE / "stock_dependency_comparison.csv",
    index=False,
    encoding="utf-8-sig"
)
