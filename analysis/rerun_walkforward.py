
import pandas as pd
from pathlib import Path

import backtest_market_ma as bt

BASE_DIR = Path(__file__).resolve().parent

prices = bt.load_prices()

# 날짜 자료형 통일
prices["Date"] = pd.to_datetime(
    prices["Date"],
    errors="coerce"
).dt.normalize()

prices["Ticker"] = prices["Ticker"].map(
    bt.normalize_ticker
)

print("\n===== 전체 가격 데이터 확인 =====")
print("전체 행:", len(prices))
print("최초 날짜:", prices["Date"].min())
print("마지막 날짜:", prices["Date"].max())

signals = pd.read_csv(
    BASE_DIR / "wf_all_signals.csv",
    dtype={"Ticker": str},
    parse_dates=["Date"]
)

signals["Ticker"] = signals["Ticker"].map(
    bt.normalize_ticker
)

signals["Date"] = pd.to_datetime(
    signals["Date"],
    errors="coerce"
).dt.normalize()

metrics = pd.read_csv(
    BASE_DIR / "wf_model_metrics.csv",
    parse_dates=["Test_Start", "Test_End"]
)

start = pd.Timestamp(
    metrics["Test_Start"].min()
).normalize()

end = pd.Timestamp(
    metrics["Test_End"].max()
).normalize()

prices = prices[
    prices["Date"].between(start, end)
].copy()

print("\n===== 평가 기간 확인 =====")
print("시작:", start)
print("종료:", end)
print("평가 가격 행:", len(prices))
print("평가 거래일:", prices["Date"].nunique())
print("평가 최초 날짜:", prices["Date"].min())
print("평가 마지막 날짜:", prices["Date"].max())

if prices["Date"].nunique() < 100:
    raise ValueError(
        "평가 가격 데이터가 비정상적으로 적습니다."
    )

print("\n===== AI 신호 확인 =====")
print("전체 신호 수:", len(signals))
print("신호 최초 날짜:", signals["Date"].min())
print("신호 마지막 날짜:", signals["Date"].max())

print(
    "평가 기간 내 신호:",
    signals["Date"].between(start, end).sum()
)

# 평가 기간 내 신호만 사용
signals = signals[
    signals["Date"].between(start, end)
].copy()

# 전략별 신호 생성
strategies = {
    "A_NO_FILTER": signals.copy(),

    "B_VOLUME_1_0": signals[
        signals["Volume_Ratio"] <= 1.0
    ].copy(),
}


print("\n===== 전략별 신호 확인 =====")
# 신호 발생 날짜 + 종목이 동시에 일치하는지 확인
matched = signals.merge(
    prices[["Date", "Ticker"]].drop_duplicates(),
    on=["Date", "Ticker"],
    how="left",
    indicator=True
)

print("\n===== 신호와 가격 데이터 매칭 =====")
print(matched["_merge"].value_counts())

print("\n매칭되지 않은 신호:")
print(
    matched[matched["_merge"] == "left_only"][
        ["Date", "Ticker", "Probability"]
    ].head(10).to_string(index=False)
)

for name, candidate in strategies.items():

    print(f"\n전략: {name}")
    print("신호 수:", len(candidate))
    print("컬럼:", candidate.columns.tolist())

    print(
        candidate[
            ["Date", "Ticker", "Probability"]
        ].head(5).to_string(index=False)
    )

    print(
        "가격 데이터 날짜와 일치하는 신호:",
        candidate["Date"].isin(prices["Date"]).sum()
    )

    print(
        "가격 데이터 종목과 일치하는 신호:",
        candidate["Ticker"].isin(prices["Ticker"]).sum()
    )

original_cash = bt.INITIAL_CASH
original_filter = bt.USE_MARKET_FILTER
original_ma = bt.MARKET_MA_DAYS

results = []

try:
    bt.INITIAL_CASH = 10_000_000
    bt.USE_MARKET_FILTER = True
    bt.MARKET_MA_DAYS = 40

    for name, candidate in strategies.items():

        trades, equity = bt.run_backtest_v4(
            prices,
            candidate,
            "SWING_ATR"
        )

        result = bt.summarize_results(
            trades,
            equity,
            name
        )

        results.append(result)

        trades.to_csv(
            BASE_DIR / f"wf_{name}_trades.csv",
            index=False,
            encoding="utf-8-sig"
        )

        equity.to_csv(
            BASE_DIR / f"wf_{name}_equity.csv",
            index=False,
            encoding="utf-8-sig"
        )

finally:
    bt.INITIAL_CASH = original_cash
    bt.USE_MARKET_FILTER = original_filter
    bt.MARKET_MA_DAYS = original_ma

print("\n===== 수정 후 워크포워드 결과 =====")

print(
    pd.DataFrame(results).to_string(index=False)
)
