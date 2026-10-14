from pathlib import Path
import pandas as pd
import numpy as np

BASE = Path(__file__).resolve().parents[1]
RESULTS = BASE / "results"

PRICE_FILE = BASE / "stock_data_500.csv"
CANDIDATE_FILE = RESULTS / "backtest_candidates.csv"

AI_THRESHOLD = 0.80
HOLDING_DAYS = 5
FEE = 0.001  # 매수·매도 각각 0.1%

# -----------------------------
# 1. 가격 데이터 불러오기
# -----------------------------
raw = pd.read_csv(PRICE_FILE, dtype={"Ticker": str})
raw["Date"] = pd.to_datetime(raw["Date"])
raw["Ticker"] = raw["Ticker"].str.strip()

required = [
    "Date", "Ticker", "Open", "High", "Low",
    "Close", "Volume"
]
missing = [c for c in required if c not in raw.columns]
if missing:
    raise ValueError(f"가격 데이터에 필요한 열이 없습니다: {missing}")

for col in ["Open", "High", "Low", "Close", "Volume"]:
    raw[col] = pd.to_numeric(raw[col], errors="coerce")

raw = raw.dropna(subset=["Date", "Ticker", "Open", "Close"])
raw = raw.sort_values(["Ticker", "Date"])
raw = raw.drop_duplicates(["Ticker", "Date"], keep="last")
raw = raw.reset_index(drop=True)

raw["Prev_Close"] = raw.groupby("Ticker")["Close"].shift(1)
price_change = raw["Open"] / raw["Prev_Close"] - 1

# 시가·고가·저가·종가가 같고 전일 종가 대비
# 약 30% 변동한 날을 가격제한폭 고정 추정으로 분류
same_ohlc = (
    raw["Open"].eq(raw["High"]) &
    raw["High"].eq(raw["Low"]) &
    raw["Low"].eq(raw["Close"])
)

raw["Locked_Upper"] = same_ohlc & (price_change >= 0.295)
raw["Locked_Lower"] = same_ohlc & (price_change <= -0.295)

# 종목별 날짜 -> 행 번호
groups = {}
date_index = {}

for ticker, group in raw.groupby("Ticker", sort=False):
    group = group.sort_values("Date").reset_index(drop=True)
    groups[ticker] = group
    date_index[ticker] = {
        date: idx for idx, date in enumerate(group["Date"])
    }

# -----------------------------
# 2. AI 매수 후보 불러오기
# -----------------------------
cand = pd.read_csv(CANDIDATE_FILE, dtype={"Ticker": str})
cand["Date"] = pd.to_datetime(cand["Date"])
cand["Ticker"] = cand["Ticker"].str.strip()
cand["AI_Score"] = pd.to_numeric(
    cand["AI_Score"], errors="coerce"
)

cand = cand[cand["AI_Score"] >= AI_THRESHOLD].copy()
cand = cand.drop_duplicates(["Date", "Ticker"])
cand = cand.sort_values(["Date", "Ticker"]).reset_index(drop=True)

# -----------------------------
# 3. 순차 매매 시뮬레이션
# -----------------------------
trades = []
skips = []

# 종목별 기존 포지션의 실제 매도 행 번호
busy_until = {}

def skip_signal(row, reason):
    skips.append({
        "Signal_Date": row["Date"],
        "Ticker": row["Ticker"],
        "Stock_Name": row.get("Stock_Name", ""),
        "AI_Score": row["AI_Score"],
        "Reason": reason
    })

for _, signal in cand.iterrows():
    ticker = signal["Ticker"]
    signal_date = signal["Date"]

    if ticker not in groups:
        skip_signal(signal, "가격데이터 없음")
        continue

    group = groups[ticker]
    idx_map = date_index[ticker]
    signal_idx = idx_map.get(signal_date)

    if signal_idx is None:
        skip_signal(signal, "신호 날짜 없음")
        continue

    entry_idx = signal_idx + 1

    if entry_idx >= len(group):
        skip_signal(signal, "매수일 데이터 없음")
        continue

    # 기존 포지션이 아직 끝나지 않았으면 중복 매수 금지
    if entry_idx <= busy_until.get(ticker, -1):
        skip_signal(signal, "기존 포지션 보유 중")
        continue

    entry = group.iloc[entry_idx]

    if pd.isna(entry["Volume"]) or entry["Volume"] <= 0:
        skip_signal(signal, "매수일 거래량 0")
        continue

    if pd.isna(entry["Open"]) or entry["Open"] <= 0:
        skip_signal(signal, "매수 가격 오류")
        continue

    if entry["Locked_Upper"] or entry["Locked_Lower"]:
        skip_signal(signal, "매수일 가격제한폭 고정 추정")
        continue

    # 매수일로부터 5거래일 뒤 시가에 매도 시도
    scheduled_exit_idx = entry_idx + HOLDING_DAYS

    if scheduled_exit_idx >= len(group):
        skip_signal(signal, "보유기간 이후 데이터 부족")
        continue

    # 거래량 0 또는 하한가 고정 추정이면 매도를 미루고
    # 이후 처음 매도 가능한 시가를 사용
    exit_idx = None

    for i in range(scheduled_exit_idx, len(group)):
        candidate_exit = group.iloc[i]

        if (
            pd.isna(candidate_exit["Volume"]) or
            candidate_exit["Volume"] <= 0
        ):
            continue

        if (
            pd.isna(candidate_exit["Open"]) or
            candidate_exit["Open"] <= 0
        ):
            continue

        if candidate_exit["Locked_Lower"]:
            continue

        exit_idx = i
        break

    if exit_idx is None:
        skip_signal(signal, "매도 가능한 가격 데이터 없음")
        continue

    exit_row = group.iloc[exit_idx]
    entry_price = float(entry["Open"])
    exit_price = float(exit_row["Open"])

    net_return = (
        exit_price * (1 - FEE) /
        (entry_price * (1 + FEE)) - 1
    ) * 100

    trades.append({
        "Signal_Date": signal_date,
        "Stock_Name": signal.get("Stock_Name", ""),
        "Ticker": ticker,
        "AI_Score": signal["AI_Score"],
        "Entry_Date": entry["Date"],
        "Entry_Open": entry_price,
        "Scheduled_Exit_Date": group.iloc[
            scheduled_exit_idx
        ]["Date"],
        "Actual_Exit_Date": exit_row["Date"],
        "Exit_Open": exit_price,
        "Exit_Delay_Days": exit_idx - scheduled_exit_idx,
        "Net_Return_Pct": net_return
    })

    # 매도일까지 동일 종목 재매수 금지
    busy_until[ticker] = exit_idx

# -----------------------------
# 4. 결과 저장
# -----------------------------
trade_df = pd.DataFrame(trades)
skip_df = pd.DataFrame(skips)

trade_path = RESULTS / "portfolio_execution_trades.csv"
skip_path = RESULTS / "portfolio_execution_skips.csv"
summary_path = RESULTS / "portfolio_execution_summary.csv"

trade_df.to_csv(
    trade_path, index=False, encoding="utf-8-sig"
)
skip_df.to_csv(
    skip_path, index=False, encoding="utf-8-sig"
)

if len(trade_df):
    r = trade_df["Net_Return_Pct"]
    summary = {
        "AI_Threshold": AI_THRESHOLD,
        "Holding_Days": HOLDING_DAYS,
        "Candidate_Signals": len(cand),
        "Executed_Trades": len(trade_df),
        "Skipped_Signals": len(skip_df),
        "Win_Rate_Pct": (r > 0).mean() * 100,
        "Average_Net_Return_Pct": r.mean(),
        "Median_Net_Return_Pct": r.median(),
        "Best_Return_Pct": r.max(),
        "Worst_Return_Pct": r.min(),
        "Average_Exit_Delay_Days":
            trade_df["Exit_Delay_Days"].mean()
    }
else:
    summary = {
        "AI_Threshold": AI_THRESHOLD,
        "Holding_Days": HOLDING_DAYS,
        "Candidate_Signals": len(cand),
        "Executed_Trades": 0,
        "Skipped_Signals": len(skip_df)
    }

pd.DataFrame([summary]).to_csv(
    summary_path, index=False, encoding="utf-8-sig"
)

print("\n=== 중복 매수 제한 및 체결 조건 반영 결과 ===")
print(pd.DataFrame([summary]).round(3).to_string(index=False))

if len(skip_df):
    print("\n=== 신호 제외 사유 ===")
    print(skip_df["Reason"].value_counts().to_string())

print("\n저장 완료:")
print(trade_path)
print(skip_path)
print(summary_path)

print(
    "\n주의: 일봉 데이터 기반 근사 시뮬레이션이며 "
    "실제 호가·주문 체결·세금·슬리피지를 완벽히 반영하지 않습니다."
)
print(
    "또한 종목별 중복 매수만 제한하며, "
    "총 투자금·동시 보유 종목 수·현금 잔고는 제한하지 않습니다."
)
