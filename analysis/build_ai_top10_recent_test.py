
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent

DATA = BASE / "stock_data_500_recent.csv"
MODEL = BASE / "model" / "random_forest_strategy_b_500.pkl"
OUTPUT = BASE / "ai_top10_recent_test.json"

FEATURES = [
    "MA20_Gap", "MA60_Gap", "MA20_MA60_Gap",
    "MA20_Change", "MA60_Change",
    "High_20D_Gap", "Low_20D_Gap",
    "RSI", "RSI_Change",
    "Volume_Ratio", "Volume_Ratio_5D",
    "Return_1D", "Return_3D", "Return_5D", "Return_10D",
    "Volatility_20D", "Volatility_Change",
    "Close_Position",
]

print("데이터 로딩...")

df = pd.read_csv(
    DATA,
    dtype={"Ticker": str},
    parse_dates=["Date"],
)

df = df.drop_duplicates(
    ["Ticker", "Date"],
    keep="last",
)

df = df.sort_values(
    ["Ticker", "Date"]
).reset_index(drop=True)

# 기존 코드와 동일하게 당일 데이터 제외
cutoff = pd.Timestamp.today().normalize() - pd.Timedelta(days=1)
df = df.loc[df["Date"] <= cutoff].copy()

if df.empty:
    raise RuntimeError("예측에 사용할 확정 거래일 데이터가 없습니다.")

latest_date = df["Date"].max()
print("예측 기준일:", latest_date.date())

parts = []

for ticker, g in df.groupby("Ticker", sort=False):

    g = g.copy().sort_values("Date")

    c = g["Close"]
    v = g["Volume"]

    # 이동평균
    g["MA20"] = c.rolling(20).mean()
    g["MA60"] = c.rolling(60).mean()

    # RSI (학습 데이터와 동일한 단순 이동평균 방식)
    delta = c.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss
    g["RSI"] = 100 - 100 / (1 + rs)
    g["RSI_Change"] = g["RSI"].diff(5)

    # 거래량
    g["Volume_Ratio"] = v / v.rolling(20).mean()
    g["Volume_Ratio_5D"] = v / v.rolling(5).mean()
    g["Avg_Turnover_20D"] = ((g["Close"] * g["Volume"]).rolling(20).mean())

    # 최근 수익률
    for n in (1, 3, 5, 10):
        g[f"Return_{n}D"] = c.pct_change(n) * 100

    # 변동성
    g["Volatility_20D"] = (
        g["Return_5D"].rolling(20).std()
    )

    g["Volatility_Change"] = (
        g["Volatility_20D"].pct_change(5)
    )

    # 이동평균 대비 가격 위치
    g["MA20_Gap"] = c / g["MA20"] - 1
    g["MA60_Gap"] = c / g["MA60"] - 1
    g["MA20_MA60_Gap"] = g["MA20"] / g["MA60"] - 1

    g["MA20_Change"] = g["MA20"].pct_change(5) * 100
    g["MA60_Change"] = g["MA60"].pct_change(5) * 100

    # 최근 20일 최고가 / 최저가
    g["High_20D_Gap"] = (
        c / g["High"].rolling(20).max() - 1
    )

    g["Low_20D_Gap"] = (
        c / g["Low"].rolling(20).min() - 1
    )

    # 당일 종가 위치
    day_range = g["High"] - g["Low"]

    g["Close_Position"] = np.where(
        day_range > 0,
        (c - g["Low"]) / day_range,
        0.5,
    )

    parts.append(g.tail(1))

latest = pd.concat(parts, ignore_index=True)

# 모든 종목을 같은 날짜 기준으로 비교
latest = latest.loc[
    latest["Date"] == latest_date
].copy()

latest[FEATURES] = latest[FEATURES].replace(
    [np.inf, -np.inf],
    np.nan,
)

valid = latest.dropna(
    subset=FEATURES + ["Avg_Turnover_20D"]
).copy()

# 최근 20거래일 평균 거래대금 10억 원 이상
valid = valid[
    valid["Avg_Turnover_20D"] >= 1_000_000_000
].copy()

print("거래대금 필터 통과 종목:", len(valid))

model = joblib.load(MODEL)

if list(getattr(model, "feature_names_in_", [])) != FEATURES:
    raise ValueError("모델 특징 목록이 현재 코드와 일치하지 않습니다.")

positive_index = list(model.classes_).index(1)

valid["ai_probability"] = model.predict_proba(
    valid[FEATURES]
)[:, positive_index]

valid = valid.sort_values(
    ["ai_probability", "Ticker"],
    ascending=[False, True],
)

results = []

for _, r in valid.iterrows():
    results.append({
        "ticker": str(r["Ticker"]),
        "name": str(r.get("Stock_Name", r["Ticker"])),
        "close": round(float(r["Close"]), 2),
        "ai_probability": round(float(r["ai_probability"]), 6),
        "rsi": round(float(r["RSI"]), 2),
        "return_5d_pct": round(float(r["Return_5D"]), 2),
        "volume_ratio": round(float(r["Volume_Ratio"]), 2),
        "avg_turnover_20d": round( float(r["Avg_Turnover_20D"])),
    })

payload = {
    "as_of": latest_date.strftime("%Y-%m-%d"),
    "model": MODEL.name,
    "target": "5거래일 이내 +10% 도달 여부",
    "eligible_count": len(results),
    "top10": results[:10],
    "all_stocks": results,
    "note": (
        "AI 모델 예측 점수입니다. 실제 상승 확률이나 "
        "투자 수익을 보장하지 않습니다."
    ),
}

OUTPUT.write_text(
    json.dumps(payload, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print(f"유효 종목: {len(results)} / {df['Ticker'].nunique()}")
print(f"저장 완료: {OUTPUT}")

print("\n===== AI TOP 10 =====")

for i, row in enumerate(results[:10], 1):
    print(
        f"{i:2}. {row['name']} ({row['ticker']})"
        f" | 점수 {row['ai_probability']:.2%}"
        f" | 종가 {row['close']:,.0f}원"
    )
