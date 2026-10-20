from pathlib import Path
import inspect
import pandas as pd
import backtest_market_ma as bt

BASE = Path(__file__).resolve().parent

# 기존 정상 동작한 compare_positions.py와 동일한 데이터 처리
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
signals["Date"] = pd.to_datetime(signals["Date"]).dt.normalize()

start = metrics["Test_Start"].min()
end = metrics["Test_End"].max()

bt.PRICE_FILE = BASE / "stock_data_500_updated.csv"
prices = bt.load_prices()
prices["Date"] = pd.to_datetime(prices["Date"]).dt.normalize()

prices = prices[prices["Date"].between(start, end)].copy()
signals = signals[signals["Date"].between(start, end)].copy()

print("\n평가 기간:", start.date(), "~", end.date())
print("가격 행:", len(prices))
print("AI 신호:", len(signals))

market = pd.read_csv(BASE / "kospi_market_filter.csv")
market["Date"] = pd.to_datetime(market["Date"])

if market["Date"].max() < end:
    raise ValueError("코스피 데이터가 평가 종료일까지 없습니다.")

# 기존 함수는 변경하지 않고 고정 배분 버전만 생성
source = inspect.getsource(bt.run_backtest_v4)

old = "allocation = cash / slots_left"
new = "allocation = min(cash, INITIAL_CASH / MAX_POSITIONS)"

if source.count(old) != 1:
    raise RuntimeError("매수 예산 계산 코드를 찾지 못했습니다.")

fixed_source = source.replace(
    "def run_backtest_v4(",
    "def run_backtest_fixed(",
    1
).replace(old, new, 1)

namespace = dict(vars(bt))
exec(fixed_source, namespace)
run_fixed = namespace["run_backtest_fixed"]

bt.INITIAL_CASH = 1_000_000
bt.MAX_POSITIONS = 2
bt.MARKET_MA_DAYS = 40
bt.USE_MARKET_FILTER = True

results = []

for name, runner in [
    ("ORIGINAL_2", bt.run_backtest_v4),
    ("FIXED_500K", run_fixed),
]:
    trades, equity = runner(prices, signals, "SWING_ATR")

    result = bt.summarize_results(trades, equity, name)
    results.append(result)

    trades.to_csv(
        BASE / f"allocation_{name}_trades.csv",
        index=False,
        encoding="utf-8-sig"
    )

    equity.to_csv(
        BASE / f"allocation_{name}_equity.csv",
        index=False,
        encoding="utf-8-sig"
    )

print("\n===== 자금 배분 방식 비교 =====")
print(pd.DataFrame(results).to_string(index=False))
