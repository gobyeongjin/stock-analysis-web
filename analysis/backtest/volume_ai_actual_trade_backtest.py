from pathlib import Path
import pandas as pd
import numpy as np

BASE = Path(__file__).resolve().parents[1]
RESULTS = BASE / "results"
RAW_FILE = BASE / "stock_data_500.csv"
RESULTS.mkdir(parents=True, exist_ok=True)

# 원본 가격 데이터
raw = pd.read_csv(RAW_FILE, dtype={"Ticker": str})
raw["Date"] = pd.to_datetime(raw["Date"])
raw["Ticker"] = raw["Ticker"].str.strip()
raw = raw.sort_values(["Ticker", "Date"]).reset_index(drop=True)

g = raw.groupby("Ticker", sort=False)

# 신호 발생 다음 날 시가 매수,
# 신호 발생일로부터 6번째 거래일 시가 매도
raw["Entry_Open"] = g["Open"].shift(-1)
raw["Exit_Open_5D"] = g["Open"].shift(-6)

# 거래량 필터: 직전 20거래일 평균 사용
raw["Volume_MA20"] = g["Volume"].transform(
    lambda s: s.shift(1).rolling(20, min_periods=20).mean()
)
raw["Volume_Ratio_20"] = raw["Volume"] / raw["Volume_MA20"]

price_range = raw["High"] - raw["Low"]
raw["Close_Position"] = np.where(
    price_range > 0,
    (raw["Close"] - raw["Low"]) / price_range,
    np.nan
)

raw["Prev_Close"] = g["Close"].shift(1)
raw["Daily_Return_Pct"] = (
    raw["Close"] / raw["Prev_Close"] - 1
) * 100

raw["Filter_A"] = (
    (raw["Volume_Ratio_20"] >= 2) &
    (raw["Close_Position"] >= 0.8)
)
raw["Filter_B"] = (
    (raw["Volume_Ratio_20"] >= 2) &
    raw["Daily_Return_Pct"].between(-1, 1)
)
raw["Filter_C"] = raw["Filter_A"] & raw["Filter_B"]

# 매수·매도 각각 0.1% 비용 가정
fee = 0.001
raw["Actual_Trade_Return_Pct"] = (
    raw["Exit_Open_5D"] * (1 - fee) /
    (raw["Entry_Open"] * (1 + fee)) - 1
) * 100

price_cols = [
    "Date", "Ticker", "Entry_Open", "Exit_Open_5D",
    "Actual_Trade_Return_Pct", "Filter_A", "Filter_B", "Filter_C"
]
prices = raw[price_cols].drop_duplicates(["Date", "Ticker"])

files = [
    "backtest_candidates.csv",
    "backtest_candidates_v2.csv"
]
filters = {
    "Baseline": None,
    "A_BuyPressure": "Filter_A",
    "B_SidewaysVolume": "Filter_B",
    "C_Combined": "Filter_C",
}

summaries = []

for filename in files:
    path = RESULTS / filename
    if not path.exists():
        continue

    cand = pd.read_csv(path, dtype={"Ticker": str})
    cand["Date"] = pd.to_datetime(cand["Date"])
    cand["Ticker"] = cand["Ticker"].str.strip()
    cand["AI_Score"] = pd.to_numeric(cand["AI_Score"], errors="coerce")

    data = cand.merge(
        prices, on=["Date", "Ticker"], how="left",
        validate="many_to_one"
    )

    for threshold in [0.70, 0.80]:
        for filter_name, filter_col in filters.items():
            s = data[data["AI_Score"] >= threshold].copy()

            if filter_col:
                s = s[s[filter_col].fillna(False)]

            s = s[
                s["Actual_Trade_Return_Pct"].notna() &
                s["Entry_Open"].gt(0) &
                s["Exit_Open_5D"].gt(0)
            ]

            r = s["Actual_Trade_Return_Pct"]
            labels = pd.to_numeric(s["Label"], errors="coerce")

            summaries.append({
                "Candidate_File": filename,
                "AI_Threshold": threshold,
                "Filter": filter_name,
                "Trades": len(s),
                "Actual_Win_Rate_Pct":
                    (r > 0).mean() * 100 if len(s) else np.nan,
                "Average_Net_Return_Pct":
                    r.mean() if len(s) else np.nan,
                "Median_Net_Return_Pct":
                    r.median() if len(s) else np.nan,
                "Target_10Pct_Hit_Rate_Pct":
                    labels.mean() * 100 if len(labels.dropna()) else np.nan,
            })

result = pd.DataFrame(summaries)
out = RESULTS / "volume_ai_actual_trade_summary.csv"
result.to_csv(out, index=False, encoding="utf-8-sig")

print("\n=== 실제 매매 가정 기준 비교 ===")
print(result.round(3).to_string(index=False))
print(f"\n결과 저장: {out}")
print("\n주의: 신호별 수익률이며, 보유 기간 중 신호가 겹칠 수 있습니다.")
print("포트폴리오 수익률이나 미래 수익을 보장하는 결과는 아닙니다.")
