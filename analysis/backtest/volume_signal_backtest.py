from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = BASE_DIR / "stock_data_500.csv"
RESULT_DIR = BASE_DIR / "results"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_FILE)
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)

required = ["Date", "Open", "High", "Low", "Close", "Volume", "Ticker"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"필수 열이 없습니다: {missing}")

g = df.groupby("Ticker", group_keys=False)

# 과거 데이터만 사용해 지표 계산
df["Volume_MA20"] = g["Volume"].transform(
    lambda s: s.shift(1).rolling(20, min_periods=20).mean()
)
df["Volume_Ratio"] = df["Volume"] / df["Volume_MA20"]

df["Prev_Close"] = g["Close"].shift(1)
df["Daily_Return_Pct"] = (df["Close"] / df["Prev_Close"] - 1) * 100

# 종가가 당일 고가-저가 범위에서 차지하는 위치
price_range = df["High"] - df["Low"]
df["Close_Position"] = np.where(
    price_range > 0,
    (df["Close"] - df["Low"]) / price_range,
    np.nan
)

# A: 거래량 급증 + 종가가 고가 부근
df["Signal_A"] = (
    (df["Volume_Ratio"] >= 2.0) &
    (df["Close_Position"] >= 0.8)
)

# B: 거래량 급증 + 주가 변동이 작음
df["Signal_B"] = (
    (df["Volume_Ratio"] >= 2.0) &
    (df["Daily_Return_Pct"].between(-1.0, 1.0))
)

# C: A와 B를 동시에 만족
df["Signal_C"] = df["Signal_A"] & df["Signal_B"]

# 신호 발생 다음 거래일 시가 매수,
# 신호일 기준 6번째 다음 행의 시가 매도
# 즉, 진입 후 약 5거래일 보유
df["Entry_Open"] = g["Open"].shift(-1)
df["Exit_Open_5D"] = g["Open"].shift(-6)

# 왕복 거래비용 0.1%씩 가정
fee = 0.001
df["Return_5D_Pct"] = (
    (df["Exit_Open_5D"] * (1 - fee)) /
    (df["Entry_Open"] * (1 + fee)) - 1
) * 100

signals = {
    "A_buy_pressure_estimate": "Signal_A",
    "B_accumulation_pattern": "Signal_B",
    "C_combined": "Signal_C",
}

summary = []
detail_frames = []

for name, col in signals.items():
    sample = df.loc[
        df[col] &
        df["Entry_Open"].gt(0) &
        df["Exit_Open_5D"].gt(0) &
        df["Return_5D_Pct"].notna()
    ].copy()

    returns = sample["Return_5D_Pct"]

    summary.append({
        "Strategy": name,
        "Trades": len(sample),
        "Win_Rate_Pct": (returns > 0).mean() * 100 if len(sample) else np.nan,
        "Average_Return_Pct": returns.mean() if len(sample) else np.nan,
        "Median_Return_Pct": returns.median() if len(sample) else np.nan,
        "Positive_5D_Pct": (returns > 0).mean() * 100 if len(sample) else np.nan,
        "Negative_5D_Pct": (returns < 0).mean() * 100 if len(sample) else np.nan,
    })

    sample["Strategy"] = name
    detail_frames.append(sample[
        ["Date", "Ticker", "Stock_Name", "Volume_Ratio",
         "Daily_Return_Pct", "Close_Position", "Entry_Open",
         "Exit_Open_5D", "Return_5D_Pct", "Strategy"]
    ])

summary_df = pd.DataFrame(summary)
summary_path = RESULT_DIR / "volume_signal_summary.csv"
detail_path = RESULT_DIR / "volume_signal_details.csv"

summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
if detail_frames:
    pd.concat(detail_frames, ignore_index=True).to_csv(
        detail_path, index=False, encoding="utf-8-sig"
    )

print("\n=== 거래량 신호 백테스트 결과 ===")
print(summary_df.round(3).to_string(index=False))
print(f"\n요약 저장: {summary_path}")
print(f"상세 저장: {detail_path}")
print("\n주의: 종목별 신호 수익률은 보유 기간이 겹칠 수 있으며,")
print("이 결과만으로 실제 매집 또는 미래 수익을 확정할 수 없습니다.")
