from pathlib import Path
import pandas as pd
import numpy as np

# ==========================================
# 설정
# ==========================================
BASE = Path(__file__).resolve().parents[1]
RESULTS = BASE / "results"

PRICE_FILE = BASE / "stock_data_500.csv"
CANDIDATE_FILE = RESULTS / "backtest_candidates.csv"

AI_THRESHOLD = 0.80
HOLDING_DAYS = 5
FEE = 0.001  # 매수·매도 각각 0.1%

# None은 손절 없음
STOP_SETTINGS = {
    "No_Stop": None,
    "Stop_5pct": -0.05,
    "Stop_8pct": -0.08,
    "Stop_10pct": -0.10,
}

# ==========================================
# 1. 데이터 로드 및 검증
# ==========================================
RESULTS.mkdir(parents=True, exist_ok=True)

raw = pd.read_csv(PRICE_FILE, dtype={"Ticker": str})
cand = pd.read_csv(CANDIDATE_FILE, dtype={"Ticker": str})

raw["Date"] = pd.to_datetime(raw["Date"], errors="coerce")
cand["Date"] = pd.to_datetime(cand["Date"], errors="coerce")

raw["Ticker"] = raw["Ticker"].str.strip()
cand["Ticker"] = cand["Ticker"].str.strip()

price_columns = [
    "Date", "Ticker", "Open", "High", "Low", "Close", "Volume"
]

missing = [c for c in price_columns if c not in raw.columns]
if missing:
    raise ValueError(f"가격 데이터에 필요한 열이 없습니다: {missing}")

candidate_columns = ["Date", "Ticker", "AI_Score"]
missing = [c for c in candidate_columns if c not in cand.columns]
if missing:
    raise ValueError(f"매수 후보 파일에 필요한 열이 없습니다: {missing}")

for col in ["Open", "High", "Low", "Close", "Volume"]:
    raw[col] = pd.to_numeric(raw[col], errors="coerce")

cand["AI_Score"] = pd.to_numeric(cand["AI_Score"], errors="coerce")

raw = raw.dropna(subset=["Date", "Ticker", "Open", "High", "Low", "Close"])
raw = raw.sort_values(["Ticker", "Date"])
raw = raw.drop_duplicates(["Ticker", "Date"], keep="last")
raw = raw.reset_index(drop=True)

# 전일 종가 대비 등락률
raw["Prev_Close"] = raw.groupby("Ticker")["Close"].shift(1)
raw["Daily_Change"] = raw["Close"] / raw["Prev_Close"] - 1

# 가격제한폭 고정 추정:
# OHLC가 모두 같고 종가가 전일 종가 대비 약 30% 움직인 경우
same_ohlc = (
    raw["Open"].eq(raw["High"])
    & raw["High"].eq(raw["Low"])
    & raw["Low"].eq(raw["Close"])
)

raw["Locked_Upper"] = same_ohlc & (raw["Daily_Change"] >= 0.295)
raw["Locked_Lower"] = same_ohlc & (raw["Daily_Change"] <= -0.295)

# 종목별 가격 데이터와 날짜 인덱스
groups = {}
date_index = {}

for ticker, group in raw.groupby("Ticker", sort=False):
    group = group.sort_values("Date").reset_index(drop=True)
    groups[ticker] = group
    date_index[ticker] = {
        date: idx for idx, date in enumerate(group["Date"])
    }

# AI 점수 기준 적용
cand = cand.dropna(subset=["Date", "Ticker", "AI_Score"])
cand = cand[cand["AI_Score"] >= AI_THRESHOLD].copy()

# 같은 날짜·종목 신호 중복 제거
cand = (
    cand.sort_values(["Date", "Ticker", "AI_Score"],
                     ascending=[True, True, False])
        .drop_duplicates(["Date", "Ticker"], keep="first")
        .sort_values(["Date", "Ticker"])
        .reset_index(drop=True)
)

print(f"AI 점수 {AI_THRESHOLD:.2f} 이상 후보: {len(cand):,}건")

# ==========================================
# 2. 전략별 매매 시뮬레이션
# ==========================================
all_trades = []
all_skips = []
all_summaries = []

for strategy_name, stop_pct in STOP_SETTINGS.items():
    trades = []
    skips = []

    # 종목별 마지막 포지션 청산 인덱스
    busy_until = {}

    for _, signal in cand.iterrows():
        ticker = signal["Ticker"]
        signal_date = signal["Date"]

        def skip(reason):
            skips.append({
                "Strategy": strategy_name,
                "Signal_Date": signal_date,
                "Ticker": ticker,
                "Stock_Name": signal.get("Stock_Name", ""),
                "AI_Score": signal["AI_Score"],
                "Reason": reason,
            })

        if ticker not in groups:
            skip("가격데이터 없음")
            continue

        group = groups[ticker]
        signal_idx = date_index[ticker].get(signal_date)

        if signal_idx is None:
            skip("신호 날짜 없음")
            continue

        entry_idx = signal_idx + 1

        if entry_idx >= len(group):
            skip("매수일 데이터 없음")
            continue

        # 이전 포지션이 끝나기 전 같은 종목 추가 매수 금지
        if entry_idx <= busy_until.get(ticker, -1):
            skip("기존 포지션 보유 중")
            continue

        entry = group.iloc[entry_idx]

        if pd.isna(entry["Volume"]) or entry["Volume"] <= 0:
            skip("매수일 거래량 0")
            continue

        if pd.isna(entry["Open"]) or entry["Open"] <= 0:
            skip("매수 가격 오류")
            continue

        if entry["Locked_Upper"] or entry["Locked_Lower"]:
            skip("매수일 가격제한폭 고정 추정")
            continue

        entry_price = float(entry["Open"])
        stop_price = (
            entry_price * (1 + stop_pct)
            if stop_pct is not None else None
        )

        scheduled_exit_idx = entry_idx + HOLDING_DAYS

        if scheduled_exit_idx >= len(group):
            skip("보유기간 이후 데이터 부족")
            continue

        exit_idx = None
        exit_price = None
        exit_reason = None

        # 매수일 이후부터 청산될 때까지 매일 손절 여부 확인
        for i in range(entry_idx, len(group)):
            day = group.iloc[i]

            volume = day["Volume"]
            open_price = day["Open"]
            low_price = day["Low"]

            tradable_volume = (
                pd.notna(volume) and volume > 0
            )
            valid_open = (
                pd.notna(open_price) and open_price > 0
            )

            # 거래량 0이면 그날 체결된 가격으로 간주하지 않음
            if tradable_volume and valid_open:
                if stop_price is not None and pd.notna(low_price):
                    if low_price <= stop_price:
                        # 하한가 고정 추정이면 손절 주문 체결을
                        # 보장할 수 없으므로 다음 거래일로 넘김
                        if day["Locked_Lower"]:
                            pass
                        elif open_price <= stop_price:
                            # 시가부터 손절가 이하: 갭 하락
                            # 손절가가 아니라 시가로 청산 가정
                            exit_idx = i
                            exit_price = float(open_price)
                            exit_reason = "Gap_Stop"
                            break
                        else:
                            # 장중 저가가 손절가에 도달:
                            # 손절가 체결로 근사
                            exit_idx = i
                            exit_price = float(stop_price)
                            exit_reason = "Stop_Loss"
                            break

                # 보유 기간이 지난 후 첫 청산 가능 시가에 매도
                if i >= scheduled_exit_idx:
                    if not day["Locked_Lower"]:
                        exit_idx = i
                        exit_price = float(open_price)
                        if i == scheduled_exit_idx:
                            exit_reason = "Time_Exit"
                        else:
                            exit_reason = "Delayed_Time_Exit"
                        break

        # 마지막 데이터까지 매도하지 못한 포지션은 미청산 처리
        if exit_idx is None:
            trades.append({
                "Strategy": strategy_name,
                "Signal_Date": signal_date,
                "Stock_Name": signal.get("Stock_Name", ""),
                "Ticker": ticker,
                "AI_Score": signal["AI_Score"],
                "Entry_Date": entry["Date"],
                "Entry_Open": entry_price,
                "Stop_Price": stop_price,
                "Scheduled_Exit_Date": group.iloc[
                    scheduled_exit_idx
                ]["Date"],
                "Actual_Exit_Date": pd.NaT,
                "Exit_Open": np.nan,
                "Exit_Reason": "Unresolved",
                "Holding_Trading_Days": len(group) - 1 - entry_idx,
                "Exit_Delay_Days": np.nan,
                "Net_Return_Pct": np.nan,
            })

            # 미청산 포지션은 데이터 종료까지 계속 보유 중으로 처리
            busy_until[ticker] = len(group) - 1
            continue

        exit_row = group.iloc[exit_idx]

        # 매수·매도 수수료 반영 순수익률
        net_return = (
            exit_price * (1 - FEE)
            / (entry_price * (1 + FEE)) - 1
        ) * 100

        trades.append({
            "Strategy": strategy_name,
            "Signal_Date": signal_date,
            "Stock_Name": signal.get("Stock_Name", ""),
            "Ticker": ticker,
            "AI_Score": signal["AI_Score"],
            "Entry_Date": entry["Date"],
            "Entry_Open": entry_price,
            "Stop_Price": stop_price,
            "Scheduled_Exit_Date": group.iloc[
                scheduled_exit_idx
            ]["Date"],
            "Actual_Exit_Date": exit_row["Date"],
            "Exit_Open": exit_price,
            "Exit_Reason": exit_reason,
            "Holding_Trading_Days": exit_idx - entry_idx,
            "Exit_Delay_Days": max(0, exit_idx - scheduled_exit_idx),
            "Net_Return_Pct": net_return,
        })

        # 청산일 이후부터 동일 종목의 새 포지션 허용
        busy_until[ticker] = exit_idx

    trade_df = pd.DataFrame(trades)
    skip_df = pd.DataFrame(skips)

    all_trades.extend(trades)
    all_skips.extend(skips)

    if len(trade_df):
        closed = trade_df[
            trade_df["Net_Return_Pct"].notna()
        ].copy()

        unresolved_count = int(
            trade_df["Exit_Reason"].eq("Unresolved").sum()
        )

        if len(closed):
            returns = closed["Net_Return_Pct"]
            summary = {
                "Strategy": strategy_name,
                "Stop_Loss_Pct": (
                    "None" if stop_pct is None
                    else f"{stop_pct * 100:.0f}%"
                ),
                "Candidate_Signals": len(cand),
                "Closed_Trades": len(closed),
                "Unresolved_Positions": unresolved_count,
                "Skipped_Signals": len(skip_df),
                "Win_Rate_Pct": (returns > 0).mean() * 100,
                "Average_Net_Return_Pct": returns.mean(),
                "Median_Net_Return_Pct": returns.median(),
                "Worst_Return_Pct": returns.min(),
                "Best_Return_Pct": returns.max(),
                "Stop_Exit_Count": int(
                    closed["Exit_Reason"].isin(
                        ["Stop_Loss", "Gap_Stop"]
                    ).sum()
                ),
                "Gap_Stop_Count": int(
                    closed["Exit_Reason"].eq("Gap_Stop").sum()
                ),
                "Delayed_Exit_Count": int(
                    closed["Exit_Reason"].eq(
                        "Delayed_Time_Exit"
                    ).sum()
                ),
            }
        else:
            summary = {
                "Strategy": strategy_name,
                "Stop_Loss_Pct": (
                    "None" if stop_pct is None
                    else f"{stop_pct * 100:.0f}%"
                ),
                "Candidate_Signals": len(cand),
                "Closed_Trades": 0,
                "Unresolved_Positions": unresolved_count,
                "Skipped_Signals": len(skip_df),
                "Win_Rate_Pct": np.nan,
                "Average_Net_Return_Pct": np.nan,
                "Median_Net_Return_Pct": np.nan,
                "Worst_Return_Pct": np.nan,
                "Best_Return_Pct": np.nan,
                "Stop_Exit_Count": 0,
                "Gap_Stop_Count": 0,
                "Delayed_Exit_Count": 0,
            }
    else:
        summary = {
            "Strategy": strategy_name,
            "Stop_Loss_Pct": (
                "None" if stop_pct is None
                else f"{stop_pct * 100:.0f}%"
            ),
            "Candidate_Signals": len(cand),
            "Closed_Trades": 0,
            "Unresolved_Positions": 0,
            "Skipped_Signals": len(skip_df),
            "Win_Rate_Pct": np.nan,
            "Average_Net_Return_Pct": np.nan,
            "Median_Net_Return_Pct": np.nan,
            "Worst_Return_Pct": np.nan,
            "Best_Return_Pct": np.nan,
            "Stop_Exit_Count": 0,
            "Gap_Stop_Count": 0,
            "Delayed_Exit_Count": 0,
        }

    all_summaries.append(summary)

# ==========================================
# 3. 결과 저장
# ==========================================
summary_df = pd.DataFrame(all_summaries)
trade_df = pd.DataFrame(all_trades)
skip_df = pd.DataFrame(all_skips)

summary_path = RESULTS / "stop_loss_comparison_summary.csv"
trades_path = RESULTS / "stop_loss_comparison_trades.csv"
skips_path = RESULTS / "stop_loss_comparison_skips.csv"

summary_df.to_csv(
    summary_path, index=False, encoding="utf-8-sig"
)
trade_df.to_csv(
    trades_path, index=False, encoding="utf-8-sig"
)
skip_df.to_csv(
    skips_path, index=False, encoding="utf-8-sig"
)

print("\n=== 손절 조건별 비교 ===")
print(summary_df.round(3).to_string(index=False))

print("\n=== 매도 사유별 거래 수 ===")
if len(trade_df):
    print(
        trade_df.groupby(
            ["Strategy", "Exit_Reason"]
        ).size().to_string()
    )

print("\n=== 신호 제외 사유 ===")
if len(skip_df):
    print(
        skip_df.groupby(
            ["Strategy", "Reason"]
        ).size().to_string()
    )

print("\n저장 완료:")
print(summary_path)
print(trades_path)
print(skips_path)

print("\n주의:")
print("- 수익률은 청산이 완료된 거래만으로 계산합니다.")
print("- 미청산 포지션은 승률·평균 수익률 계산에서 제외합니다.")
print("- 일봉 저가만으로 장중 손절가 체결을 확정할 수 없습니다.")
print("- 갭 하락은 손절가가 아닌 시가로 청산하는 것으로 가정합니다.")
print("- 하한가 고정 추정일은 매도가 불가능할 수 있어 청산을 미룹니다.")
print("- 가격제한폭 판정은 OHLC 기반 근사치입니다.")
print("- 동시 보유 종목 수와 총 투자금은 제한하지 않습니다.")
