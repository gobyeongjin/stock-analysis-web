from pathlib import Path
import pandas as pd
import numpy as np

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "stock_data_500.csv"
RESULTS = BASE / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

# 원본 일봉 데이터
raw = pd.read_csv(DATA, dtype={"Ticker": str})
raw["Date"] = pd.to_datetime(raw["Date"])
raw["Ticker"] = raw["Ticker"].str.strip()
raw = raw.sort_values(["Ticker", "Date"]).reset_index(drop=True)

g = raw.groupby("Ticker", sort=False)

# 당일을 제외한 직전 20거래일 평균 거래량
raw["Volume_MA20"] = g["Volume"].transform(
    lambda s: s.shift(1).rolling(20, min_periods=20).mean()
)
raw["Volume_Ratio_20"] = raw["Volume"] / raw["Volume_MA20"]

raw["Prev_Close"] = g["Close"].shift(1)
raw["Daily_Return_Pct"] = (
    raw["Close"] / raw["Prev_Close"] - 1
) * 100

day_range = raw["High"] - raw["Low"]
raw["Close_Position"] = np.where(
    day_range > 0,
    (raw["Close"] - raw["Low"]) / day_range,
    np.nan
)

# A: 거래량 급증 + 종가가 당일 고가 부근
raw["Filter_A"] = (
    (raw["Volume_Ratio_20"] >= 2.0) &
    (raw["Close_Position"] >= 0.8)
)

# B: 거래량 급증 + 전일 종가 대비 변동률이 -1~+1%
raw["Filter_B"] = (
    (raw["Volume_Ratio_20"] >= 2.0) &
    raw["Daily_Return_Pct"].between(-1, 1)
)

# C: A와 B 모두 만족
raw["Filter_C"] = raw["Filter_A"] & raw["Filter_B"]

signal_cols = [
    "Date", "Ticker", "Volume_Ratio_20",
    "Daily_Return_Pct", "Close_Position",
    "Filter_A", "Filter_B", "Filter_C"
]
signals = raw[signal_cols].drop_duplicates(["Date", "Ticker"])

all_results = []
detail_results = []

candidate_files = [
    "backtest_candidates.csv",
    "backtest_candidates_v2.csv"
]

for filename in candidate_files:
    path = RESULTS / filename
    if not path.exists():
        continue

    candidates = pd.read_csv(path, dtype={"Ticker": str})
    candidates["Date"] = pd.to_datetime(candidates["Date"])
    candidates["Ticker"] = candidates["Ticker"].str.strip()
    candidates["AI_Score"] = pd.to_numeric(
        candidates["AI_Score"], errors="coerce"
    )
    candidates["Future_5D_Return"] = pd.to_numeric(
        candidates["Future_5D_Return"], errors="coerce"
    )

    data = candidates.merge(
        signals, on=["Date", "Ticker"], how="left",
        validate="many_to_one"
    )

    filters = {
        "Baseline": None,
        "A_BuyPressure": "Filter_A",
        "B_SidewaysVolume": "Filter_B",
        "C_Combined": "Filter_C",
    }

    for threshold in [0.70, 0.80]:
        for filter_name, filter_col in filters.items():
            selected = data[data["AI_Score"] >= threshold].copy()

            if filter_col is not None:
                selected = selected[
                    selected[filter_col].fillna(False)
                ]

            returns = selected["Future_5D_Return"].dropna()

            all_results.append({
                "Candidate_File": filename,
                "AI_Threshold": threshold,
                "Filter": filter_name,
                "Selected": len(selected),
                "Return_Samples": len(returns),
                "Positive_Return_Rate_Pct":
                    (returns > 0).mean() * 100 if len(returns) else np.nan,
                "Average_5D_Return":
                    returns.mean() if len(returns) else np.nan,
                "Median_5D_Return":
                    returns.median() if len(returns) else np.nan,
                "Label_Positive_Rate_Pct":
                    pd.to_numeric(
                        selected["Label"], errors="coerce"
                    ).mean() * 100
                    if "Label" in selected and len(selected) else np.nan,
            })

            selected["Candidate_File"] = filename
            selected["AI_Threshold"] = threshold
            selected["Applied_Filter"] = filter_name
            detail_results.append(selected)

summary = pd.DataFrame(all_results)
summary_path = RESULTS / "volume_ai_filter_summary.csv"
summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

if detail_results:
    details = pd.concat(detail_results, ignore_index=True)
    details_path = RESULTS / "volume_ai_filter_details.csv"
    details.to_csv(details_path, index=False, encoding="utf-8-sig")
    print(f"상세 결과 저장: {details_path}")

print("\n=== AI 점수 + 거래량 필터 비교 ===")
if not summary.empty:
    print(summary.round(3).to_string(index=False))
else:
    print("후보 데이터 파일을 찾지 못했습니다.")

print(f"\n요약 결과 저장: {summary_path}")
print("\n주의: 과거 데이터 분석 결과이며 미래 수익을 보장하지 않습니다.")
