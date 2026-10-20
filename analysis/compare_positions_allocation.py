import pandas as pd
from pathlib import Path
import backtest_market_ma_allocation as bt

BASE = Path(__file__).resolve().parent

# 기존 평가 결과 불러오기
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

# 기존 가격 로더 재사용
# 원본 파일을 변경하지 않고 업데이트 파일을 읽음
bt.PRICE_FILE = BASE / "stock_data_500_updated.csv"
prices = bt.load_prices()
prices["Date"] = pd.to_datetime(prices["Date"]).dt.normalize()

prices = prices[
    prices["Date"].between(start, end)
].copy()

signals = signals[
    signals["Date"].between(start, end)
].copy()

print("\n평가 기간:", start.date(), "~", end.date())
print("가격 행:", len(prices))
print("AI 신호:", len(signals))

# 코스피 지수는 기존 파일을 사용
market = pd.read_csv(BASE / "kospi_market_filter.csv")
market["Date"] = pd.to_datetime(market["Date"])

if market["Date"].max() < end:
    raise ValueError("코스피 데이터가 평가 종료일까지 없습니다.")

original_cash = bt.INITIAL_CASH
original_filter = bt.USE_MARKET_FILTER
original_ma = bt.MARKET_MA_DAYS

results = []

try:
    bt.INITIAL_CASH = 1_000_000
    bt.MARKET_MA_DAYS = 40

    for max_positions in [3, 4, 5]:
        bt.MAX_POSITIONS = max_positions
        name = f"B_MA40_{max_positions}STOCKS"
        use_filter = True
        bt.USE_MARKET_FILTER = use_filter

        trades, equity = bt.run_backtest_v4(
            prices,
            signals,
            "SWING_ATR"
        )

        if "Position_Count" in equity.columns:
            counts = equity["Position_Count"]
            print(
                f"[보유 종목 진단] 최대설정={max_positions} | "
                f"실제최대={counts.max():.0f}개 | "
                f"평균={counts.mean():.2f}개 | "
                f"3종목 이상={int((counts >= 3).sum())}일"
            )
        else:
            print("[오류] Position_Count 컬럼이 없어.")

        result = bt.summarize_results(
            trades,
            equity,
            name
        )
        results.append(result)


finally:
    bt.INITIAL_CASH = original_cash
    bt.USE_MARKET_FILTER = original_filter
    bt.MARKET_MA_DAYS = original_ma

print("\n===== 스윙 전략 비교 =====")
print(pd.DataFrame(results).to_string(index=False))
