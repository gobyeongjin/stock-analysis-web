from pathlib import Path
import numpy as np
import pandas as pd

import backtest_stop8_test as base
import backtest_market_ma as bt

ROOT = Path(__file__).resolve().parent
THRESHOLD = 0.65

# ==========================================
# 1. AI 예측 신호
# ==========================================
pred = pd.read_csv(
    ROOT / "wf_14_predictions.csv",
    dtype={"Ticker": str}
)

pred["Date"] = pd.to_datetime(
    pred["Date"]
).dt.normalize()

pred["Ticker"] = pred["Ticker"].map(
    bt.normalize_ticker
)

pred = pred[
    pred["Probability"] >= THRESHOLD
].copy()

# 기존 백테스트와 동일한 제외 종목
pred = pred[
    ~pred["Ticker"].isin(base.EXCLUDED_TICKERS)
].copy()

print("===== AI 신호 =====")
print("신호 수:", len(pred))
print("종목 수:", pred["Ticker"].nunique())

# ==========================================
# 2. 거래량 비율
# ==========================================
source = pd.read_csv(
    base.TRAIN_FILE,
    dtype={"Ticker": str},
    usecols=["Date", "Ticker", "Volume_Ratio"]
)

source["Date"] = pd.to_datetime(
    source["Date"]
).dt.normalize()

source["Ticker"] = source["Ticker"].map(
    bt.normalize_ticker
)

source = source.drop_duplicates(
    ["Date", "Ticker"]
)

# ==========================================
# 3. 가격 데이터로 ATR14 계산
# ==========================================
prices = bt.load_prices()

prices["Date"] = pd.to_datetime(
    prices["Date"]
).dt.normalize()

prices["Ticker"] = prices["Ticker"].map(
    bt.normalize_ticker
)

prices = prices.sort_values(
    ["Ticker", "Date"]
).copy()

print("\n===== 가격 데이터 =====")
print("행 수:", len(prices))
print("종목 수:", prices["Ticker"].nunique())

previous_close = (
    prices.groupby("Ticker")["Close"].shift(1)
)

tr1 = prices["High"] - prices["Low"]
tr2 = (prices["High"] - previous_close).abs()
tr3 = (prices["Low"] - previous_close).abs()

prices["TR"] = np.maximum.reduce([
    tr1.to_numpy(),
    tr2.to_numpy(),
    tr3.to_numpy()
])

prices["ATR14"] = (
    prices.groupby("Ticker")["TR"]
    .transform(
        lambda x: x.rolling(
            window=14,
            min_periods=14
        ).mean()
    )
)

atr = prices[
    ["Date", "Ticker", "ATR14"]
].copy()

# ==========================================
# 4. AI 신호 + 거래량 + ATR 결합
# ==========================================
signals = pred.merge(
    source,
    on=["Date", "Ticker"],
    how="left",
    validate="many_to_one"
)

signals = signals.merge(
    atr,
    on=["Date", "Ticker"],
    how="left",
    validate="many_to_one"
)

print("\n===== 병합 확인 =====")
print("병합 전 신호:", len(pred))
print("병합 후 신호:", len(signals))
print("ATR 결측:", signals["ATR14"].isna().sum())
print(
    "거래량 비율 결측:",
    signals["Volume_Ratio"].isna().sum()
)

signals = signals.dropna(
    subset=["ATR14", "Volume_Ratio"]
)

signals = signals[signals["ATR14"] > 0]

signals = signals[
    [
        "Date", "Ticker", "Volume_Ratio",
        "Probability", "ATR14", "Fold"
    ]
].sort_values(
    ["Date", "Probability"],
    ascending=[True, False]
)

# ==========================================
# 5. 저장
# ==========================================
output = ROOT / "wf_14_signals.csv"

signals.to_csv(
    output,
    index=False,
    encoding="utf-8-sig"
)

print("\n===== 최종 신호 =====")
print("신호 수:", len(signals))
print("종목 수:", signals["Ticker"].nunique())
print(
    "기간:",
    signals["Date"].min(),
    "~",
    signals["Date"].max()
)

print("\n===== 샘플 =====")
print(signals.head(5).to_string(index=False))

print("\n저장 완료:", output)
