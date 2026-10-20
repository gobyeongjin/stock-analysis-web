from pathlib import Path
import pandas as pd
import backtest_market_ma as bt

BASE = Path(__file__).resolve().parent

# 1. 기존 워크포워드 AI 신호
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

# 2. 주가 데이터
bt.PRICE_FILE = BASE / "stock_data_500_updated.csv"

prices = bt.load_prices()
prices["Date"] = pd.to_datetime(prices["Date"]).dt.normalize()

prices = prices[
    prices["Date"].between(start, end)
].copy()

signals = signals[
    signals["Date"].between(start, end)
].copy()

# 3. 코스피 데이터 범위 확인
market = pd.read_csv(
    BASE / "kospi_market_filter.csv",
    parse_dates=["Date"]
)

market["Date"] = market["Date"].dt.normalize()

if market["Date"].max() < end:
    raise ValueError("코스피 데이터가 백테스트 종료일까지 없습니다.")

# 4. 공통 매매 설정
bt.INITIAL_CASH = 1_000_000
bt.MAX_POSITIONS = 2

strategies = [
    ("NO_FILTER", False, 40),
    ("MA10", True, 10),
    ("MA20", True, 20),
    ("MA40", True, 40),
]

results = []

for name, use_filter, ma_days in strategies:
    bt.USE_MARKET_FILTER = use_filter
    bt.MARKET_MA_DAYS = ma_days

    trades, equity = bt.run_backtest_v4(
        prices,
        signals,
        "SWING_ATR"
    )

    result = bt.summarize_results(
        trades,
        equity,
        name
    )

    # 월별 평균 거래 횟수: 평가 기간 전체 기준
    months = (
        (end.year - start.year) * 12
        + end.month - start.month + 1
    )

    result["Trades_Per_Month"] = round(
        len(trades) / months, 2
    )

    results.append(result)

    trades.to_csv(
        BASE / f"market_{name}_trades.csv",
        index=False,
        encoding="utf-8-sig"
    )

    equity.to_csv(
        BASE / f"market_{name}_equity.csv",
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
    "Win_Rate_Pct",
    "Total_PnL"
]

print("\n===== 시장 필터 비교 결과 =====")
print(
    summary[columns].round(2).to_string(index=False)
)

summary.to_csv(
    BASE / "market_filter_comparison.csv",
    index=False,
    encoding="utf-8-sig"
)
