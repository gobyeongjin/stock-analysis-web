from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

BASE = Path(__file__).resolve().parent
DATA = BASE / "analysis" / "training_data_live.csv"
OUTPUT = BASE / "analysis" / "live_signals_202610.csv"

CUTOFF = pd.Timestamp("2026-10-02")
THRESHOLD = 0.65

TECH = [
    "MA20_Gap", "MA60_Gap", "MA20_MA60_Gap",
    "MA20_Change", "MA60_Change",
    "High_20D_Gap", "Low_20D_Gap",
    "RSI", "RSI_Change",
    "Volume_Ratio", "Volume_Ratio_5D",
    "Return_1D", "Return_3D", "Return_5D",
    "Return_10D", "Volatility_20D",
    "Volatility_Change", "Close_Position",
]

df = pd.read_csv(DATA, dtype={"Ticker": str}, low_memory=False)
df["Date"] = pd.to_datetime(df["Date"])

df["Code"] = (
    df["Ticker"].str.split(".", regex=False)
    .str[0].str.zfill(6)
)

for col in TECH:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=TECH + ["Date", "Code"])

# 학습 시점에 미래 정답이 완전히 확정된 날짜만 사용
dates = np.sort(df.loc[df["Date"] <= CUTOFF, "Date"].unique())

if len(dates) < 6:
    raise ValueError("학습 날짜가 부족합니다.")

last_train_date = pd.Timestamp(dates[-6])

valid = (
    df["Future_5D_Valid"].astype(str)
    .str.strip().str.lower()
    .isin(["true", "1", "1.0"])
)

train = df.loc[
    (df["Date"] <= last_train_date) & valid
].copy()

train["Label"] = pd.to_numeric(train["Label"], errors="coerce")
train = train[train["Label"].isin([0, 1])]

test = df.loc[df["Date"] > CUTOFF].copy()

print("학습 종료일:", last_train_date.date())
print("학습 행:", len(train))
print("신규 예측 행:", len(test))
print("예측 기간:", test["Date"].min().date(), "~", test["Date"].max().date())

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    min_samples_leaf=10,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)

model.fit(train[TECH], train["Label"].astype(int))

test["Probability"] = model.predict_proba(test[TECH])[:, 1]

signals = test.loc[
    test["Probability"] >= THRESHOLD,
    ["Date", "Code", "Ticker", "Stock_Name", "Probability"]
].copy()

signals = signals.sort_values(
    ["Date", "Probability"],
    ascending=[True, False]
)

signals.to_csv(OUTPUT, index=False, encoding="utf-8-sig")

print("\n===== 신규 AI 신호 =====")
print("신호 수:", len(signals))
print("신호 발생 종목:", signals["Code"].nunique())
print("날짜별 신호 수:")
print(signals.groupby("Date").size().to_string())
print("\n저장:", OUTPUT)
