from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path.cwd()
ANALYSIS = ROOT / "analysis"
sys.path.insert(0, str(ROOT))

import backtest_walkforward_a_ma_dd as wf

print("\n========== 1. 원본 데이터 품질 ==========")

raw = pd.read_csv(
    ANALYSIS / "stock_data_500.csv",
    dtype={"Ticker": str},
    low_memory=False
)
raw["Date"] = pd.to_datetime(raw["Date"])
raw = raw.sort_values(["Ticker", "Date"])

duplicates = raw.duplicated(["Ticker", "Date"]).sum()
print("원본 데이터 행:", len(raw))
print("종목·날짜 중복:", duplicates)
print("종목 수:", raw["Ticker"].nunique())

print("\n========== 2. Label 정확성 ==========")

training = pd.read_csv(
    ANALYSIS / "training_data_v3.csv",
    dtype={"Ticker": str},
    low_memory=False
)
training["Date"] = pd.to_datetime(training["Date"])

highs = pd.concat(
    [
        raw.groupby("Ticker")["High"].shift(-i)
        for i in range(1, 6)
    ],
    axis=1
)

future_max = highs.max(axis=1)
recalc_label = (
    (future_max / raw["Close"] - 1) * 100 >= 10
).astype(int)

valid = raw.groupby("Ticker")["Close"].shift(-5).notna()

check = raw.loc[valid, ["Ticker", "Date"]].copy()
check["Recalc_Label"] = recalc_label.loc[valid].values

merged = training.merge(
    check,
    on=["Ticker", "Date"],
    how="inner",
    validate="many_to_one"
)

mismatch = (
    merged["Label"].astype(int) != merged["Recalc_Label"]
).sum()

print("비교 가능한 행:", len(merged))
print("Label 불일치:", mismatch)
print("Label 불일치율:", round(mismatch / max(len(merged), 1) * 100, 4), "%")

print("\n========== 3. Walk-forward 경계 ==========")

universe = wf.load_technical_universe()
dates = np.sort(universe["Date"].unique())

for fold, cutoff, start, end in wf.fold_ranges(dates):
    train = universe[universe["Date"] < cutoff]
    test = universe[
        (universe["Date"] >= start)
        & (universe["Date"] <= end)
    ]

    last_train = train["Date"].max()
    first_test = test["Date"].min()

    # 평가 시작 이전의 전체 거래일 기준 간격
    calendar = np.sort(raw["Date"].unique())
    gap = (
        np.searchsorted(calendar, np.datetime64(first_test))
        - np.searchsorted(calendar, np.datetime64(last_train))
        - 1
    )

    print(
        f"Fold {fold}: "
        f"학습 마지막={last_train.date()}, "
        f"평가 시작={first_test.date()}, "
        f"사이 거래일={gap}, "
        f"5일 Label 경계={'확인 필요' if gap < 5 else '통과'}"
    )

print("\n========== 4. 평가 데이터 제외 비율 ==========")

all_training = training.copy()
all_training["Future_5D_Valid"] = (
    all_training["Future_5D_Valid"]
    .astype(str).str.lower().isin(["true", "1", "1.0"])
)

for fold, cutoff, start, end in wf.fold_ranges(dates):
    subset = all_training[
        (all_training["Date"] >= start)
        & (all_training["Date"] <= end)
    ]
    if len(subset) == 0:
        continue

    excluded = (~subset["Future_5D_Valid"]).sum()
    print(
        f"Fold {fold}: 전체={len(subset):,}, "
        f"미래 데이터 없음={excluded:,} "
        f"({excluded / len(subset) * 100:.2f}%)"
    )

print("\n========== 5. 연도별 거래 손익 ==========")

for slip in ["0.05", "0.10"]:
    path = ANALYSIS / f"ma40_fixed_levels_volume_dd_{slip}_trades.csv"
    trades = pd.read_csv(path)
    trades["Exit_Date"] = pd.to_datetime(trades["Exit_Date"])
    trades["Year"] = trades["Exit_Date"].dt.year

    result = trades.groupby("Year").agg(
        Trades=("PnL", "size"),
        PnL=("PnL", "sum"),
        Avg_Return=("Return_Pct", "mean"),
        Win_Rate=("PnL", lambda x: (x > 0).mean() * 100)
    )

    print(f"\n슬리피지 {slip}%")
    print(result.round(2).to_string())

print("\n========== 검증 종료 ==========")
print("주의: 연도별 PnL은 청산연도별 실현손익이며, 연도별 자산 수익률이나 MDD는 아닙니다.")
