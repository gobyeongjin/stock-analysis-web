from pathlib import Path
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parent
ANALYSIS = ROOT / "analysis"

SOURCE = ANALYSIS / "training_data_v3.csv"
MODEL_PATH = ANALYSIS / "rf_frozen_20261002.joblib"
META_PATH = ANALYSIS / "rf_frozen_20261002.json"

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

CUTOFF = pd.Timestamp("2026-10-02")

if MODEL_PATH.exists() or META_PATH.exists():
    raise FileExistsError("고정 모델 또는 메타데이터가 이미 존재합니다.")

df = pd.read_csv(SOURCE, dtype={"Ticker": str}, low_memory=False)
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

valid = (
    df["Future_5D_Valid"].astype(str)
    .str.strip().str.lower()
    .isin(["true", "1", "1.0"])
)

df = df.loc[valid & (df["Date"] <= CUTOFF)].copy()

for col in TECH + ["Label"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=TECH + ["Label", "Date"])
df = df[df["Label"].isin([0, 1])]

print("학습 데이터:", len(df))
print("학습 종료일:", df["Date"].max().date())

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    min_samples_leaf=10,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)

model.fit(df[TECH], df["Label"].astype(int))

joblib.dump(model, MODEL_PATH)

metadata = {
    "cutoff": str(CUTOFF.date()),
    "last_training_date": str(df["Date"].max().date()),
    "training_rows": len(df),
    "features": TECH,
    "threshold": 0.65,
    "model": "RandomForestClassifier",
    "parameters": model.get_params(),
}

META_PATH.write_text(
    json.dumps(metadata, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print("\n===== 모델 고정 완료 =====")
print("모델:", MODEL_PATH)
print("설정:", META_PATH)
