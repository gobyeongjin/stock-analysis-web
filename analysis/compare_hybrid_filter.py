from pathlib import Path
import inspect
import pandas as pd
import backtest_market_ma as bt

BASE = Path(__file__).resolve().parent

# 1. 기존 AI 신호 및 평가 기간
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

bt.PRICE_FILE = BASE / "stock_data_500_updated.csv"

# 이동평균 계산을 위해 전체 가격 데이터를 먼저 불러오기
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
        lambda s: s.rolling(
            20, min_periods=20
        ).mean()
    )
)

prices = prices_all[
    prices_all["Date"].between(start, end)
].copy()

signals = signals[
    signals["Date"].between(start, end)
].copy()

# 종목별 신호일 종가 및 MA20 조회
stock_filter = (
    prices_all.set_index(["Date", "Ticker"])[
        ["Close", "Stock_MA20"]
    ]
    .to_dict("index")
)

# 2. 기존 함수에서 시장 필터 부분만 교체
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
    raise RuntimeError(
        "시장 필터 코드 위치가 예상과 다릅니다. "
        "원본은 수정하지 않았습니다."
    )

hybrid_source = source.replace(
    "def run_backtest_v4(",
    "def run_backtest_hybrid(",
    1
).replace(old, new, 1)

# 3. 공통 설정을 먼저 적용한 뒤 함수 생성
bt.INITIAL_CASH = 1_000_000
bt.MAX_POSITIONS = 2
bt.MARKET_MA_DAYS = 40
bt.USE_MARKET_FILTER = True

namespace = dict(vars(bt))
namespace["stock_filter"] = stock_filter

exec(hybrid_source, namespace)

run_hybrid = namespace["run_backtest_hybrid"]

# 4. 세 전략 비교
strategies = [
    ("MA40", bt.run_backtest_v4, True),
    ("NO_FILTER", bt.run_backtest_v4, False),
    ("HYBRID", run_hybrid, True),
]

results = []

months = pd.period_range(
    start.to_period("M"),
    end.to_period("M"),
    freq="M"
)

for name, runner, use_filter in strategies:
    bt.USE_MARKET_FILTER = use_filter

    # exec로 만든 함수는 별도 전역 변수 공간을 사용함
    namespace["USE_MARKET_FILTER"] = use_filter

    trades, equity = runner(
        prices,
        signals,
        "SWING_ATR"
    )

    result = bt.summarize_results(
        trades, equity, name
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

    result["Trades_Per_Month"] = round(
        len(trades) / len(months), 2
    )
    result["Zero_Trade_Months"] = int(
        (monthly_counts == 0).sum()
    )
    result["Jul_Sep_2026_Trades"] = int(
        monthly_counts.loc[
            "2026-07":"2026-09"
        ].sum()
    )

    results.append(result)

    trades.to_csv(
        BASE / f"hybrid_{name}_trades.csv",
        index=False,
        encoding="utf-8-sig"
    )
    equity.to_csv(
        BASE / f"hybrid_{name}_equity.csv",
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
    "Trades_Per_Month",
    "Zero_Trade_Months",
    "Jul_Sep_2026_Trades",
    "Win_Rate_Pct",
]

print("\n===== 하이브리드 전략 비교 =====")
print(
    summary[columns]
    .round(2)
    .to_string(index=False)
)

summary.to_csv(
    BASE / "hybrid_comparison.csv",
    index=False,
    encoding="utf-8-sig"
)
