from pathlib import Path
import pandas as pd
import numpy as np

BASE = Path(__file__).resolve().parents[1]
RESULTS = BASE / "results"

PRICE_FILE = BASE / "stock_data_500.csv"
CANDIDATE_FILE = RESULTS / "backtest_candidates.csv"

AI_THRESHOLD = 0.80
HOLDING_DAYS = 5
FEE = 0.001

STRATEGIES = {
    "No_Stop": None,
    "Stop_5pct": -0.05,
    "Stop_8pct": -0.08,
    "Stop_10pct": -0.10,
}


def load_data():
    raw = pd.read_csv(PRICE_FILE, dtype={"Ticker": str})
    cand = pd.read_csv(CANDIDATE_FILE, dtype={"Ticker": str})

    raw["Date"] = pd.to_datetime(raw["Date"], errors="coerce")
    cand["Date"] = pd.to_datetime(cand["Date"], errors="coerce")

    raw["Ticker"] = raw["Ticker"].astype("string").str.strip()
    cand["Ticker"] = cand["Ticker"].astype("string").str.strip()

    needed = [
        "Date", "Ticker", "Open", "High", "Low",
        "Close", "Volume"
    ]
    missing = [c for c in needed if c not in raw.columns]
    if missing:
        raise ValueError(f"가격 데이터 열 누락: {missing}")

    for c in ["Date", "Ticker", "AI_Score"]:
        if c not in cand.columns:
            raise ValueError(f"후보 파일에 필요한 열이 없습니다: {c}")

    for c in ["Open", "High", "Low", "Close", "Volume"]:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")

    cand["AI_Score"] = pd.to_numeric(
        cand["AI_Score"], errors="coerce"
    )

    raw = raw.dropna(
        subset=["Date", "Ticker", "Open", "High", "Low", "Close"]
    )
    raw = raw.sort_values(["Ticker", "Date"])
    raw = raw.drop_duplicates(["Ticker", "Date"], keep="last")
    raw = raw.reset_index(drop=True)

    raw["Prev_Close"] = raw.groupby("Ticker")["Close"].shift(1)
    raw["Daily_Change"] = raw["Close"] / raw["Prev_Close"] - 1

    # OHLC가 모두 같고 전일 종가 대비 약 30% 변동한 날:
    # 가격제한폭 고정 추정치이며 실제 체결 가능 여부를 확정하지 않음
    same_ohlc = (
        raw["Open"].eq(raw["High"])
        & raw["High"].eq(raw["Low"])
        & raw["Low"].eq(raw["Close"])
    )
    raw["Locked_Upper"] = same_ohlc & (raw["Daily_Change"] >= 0.295)
    raw["Locked_Lower"] = same_ohlc & (raw["Daily_Change"] <= -0.295)

    groups = {}
    date_maps = {}

    for ticker, group in raw.groupby("Ticker", sort=False):
        group = group.sort_values("Date").reset_index(drop=True)
        groups[ticker] = group
        date_maps[ticker] = {
            date: idx for idx, date in enumerate(group["Date"])
        }

    cand = cand.dropna(subset=["Date", "Ticker", "AI_Score"])
    cand = cand[cand["AI_Score"] >= AI_THRESHOLD].copy()
    cand = cand.sort_values(
        ["Date", "Ticker", "AI_Score"],
        ascending=[True, True, False]
    )
    cand = cand.drop_duplicates(["Date", "Ticker"], keep="first")
    cand = cand.sort_values(["Date", "Ticker"]).reset_index(drop=True)

    return groups, date_maps, cand


def can_trade_open(day):
    return (
        pd.notna(day["Volume"])
        and day["Volume"] > 0
        and pd.notna(day["Open"])
        and day["Open"] > 0
    )


def simulate_one(group, entry_idx, stop_pct):
    """
    같은 진입 거래에 대해 손절 조건만 바꿔 독립적으로 시뮬레이션.
    entry_idx는 다음 거래일 시가 매수 행.
    5거래일 보유 후 매도 시가는 entry_idx + 5.
    """
    entry = group.iloc[entry_idx]
    entry_price = float(entry["Open"])
    stop_price = (
        entry_price * (1 + stop_pct)
        if stop_pct is not None else None
    )
    scheduled_exit_idx = entry_idx + HOLDING_DAYS

    pending_stop = False
    lower_lock_breach = False
    zero_volume_breach = False

    for i in range(entry_idx, len(group)):
        day = group.iloc[i]
        tradable = can_trade_open(day)
        locked_lower = bool(day["Locked_Lower"])

        # 손절 주문이 가격제한폭 등으로 체결되지 못한 경우,
        # 첫 매도 가능한 시가에 청산하는 것으로 가정
        if pending_stop and tradable and not locked_lower:
            exit_price = float(day["Open"])
            if exit_price <= stop_price:
                reason = "Gap_Stop"
            else:
                reason = "Delayed_Stop_After_Unfillable_Breach"

            return make_trade(
                group, entry_idx, i, entry_price, exit_price,
                stop_price, reason, lower_lock_breach,
                zero_volume_breach
            )

        # 예정 매도일 이후 첫 매도 가능한 시가에 청산.
        # 시가 청산을 가정하므로 해당 날짜의 저가를 보고
        # 사후적으로 손절됐다고 판단하지 않음.
        if i >= scheduled_exit_idx and tradable and not locked_lower:
            reason = (
                "Time_Exit" if i == scheduled_exit_idx
                else "Delayed_Time_Exit"
            )
            return make_trade(
                group, entry_idx, i, entry_price, float(day["Open"]),
                stop_price, reason, lower_lock_breach,
                zero_volume_breach
            )

        # 손절 없음 전략은 손절 조건을 검사하지 않음
        if stop_price is None:
            continue

        low = day["Low"]
        if pd.isna(low) or low > stop_price:
            continue

        # 저가가 손절선에 닿았지만 거래량이 없거나
        # 하한가 고정 추정이면 당일 체결을 가정하지 않음
        if not tradable or locked_lower:
            pending_stop = True
            if locked_lower:
                lower_lock_breach = True
            if not tradable:
                zero_volume_breach = True
            continue

        open_price = float(day["Open"])

        # 시가가 이미 손절가 아래면 손절가 체결이 아니라
        # 갭 하락한 시가로 청산
        if open_price <= stop_price:
            return make_trade(
                group, entry_idx, i, entry_price, open_price,
                stop_price, "Gap_Stop", lower_lock_breach,
                zero_volume_breach
            )

        # 장중 손절가 도달: 손절가 체결로 근사
        return make_trade(
            group, entry_idx, i, entry_price, float(stop_price),
            stop_price, "Stop_Loss", lower_lock_breach,
            zero_volume_breach
        )

    return {
        "Exit_Index": len(group) - 1,
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
        "Lower_Lock_Breach": lower_lock_breach,
        "Zero_Volume_Breach": zero_volume_breach,
    }


def make_trade(
    group, entry_idx, exit_idx, entry_price, exit_price,
    stop_price, reason, lower_lock_breach, zero_volume_breach
):
    entry = group.iloc[entry_idx]
    scheduled_idx = entry_idx + HOLDING_DAYS

    net_return = (
        exit_price * (1 - FEE)
        / (entry_price * (1 + FEE)) - 1
    ) * 100

    return {
        "Exit_Index": exit_idx,
        "Entry_Date": entry["Date"],
        "Entry_Open": entry_price,
        "Stop_Price": stop_price,
        "Scheduled_Exit_Date": group.iloc[scheduled_idx]["Date"],
        "Actual_Exit_Date": group.iloc[exit_idx]["Date"],
        "Exit_Open": exit_price,
        "Exit_Reason": reason,
        "Holding_Trading_Days": exit_idx - entry_idx,
        "Exit_Delay_Days": max(0, exit_idx - scheduled_idx),
        "Net_Return_Pct": net_return,
        "Lower_Lock_Breach": lower_lock_breach,
        "Zero_Volume_Breach": zero_volume_breach,
    }


def unique_output_paths():
    # 기존 결과 파일도 덮어쓰지 않도록 번호를 붙임
    suffix = 1
    while True:
        tag = "" if suffix == 1 else f"_{suffix}"
        paths = {
            "summary": RESULTS / f"matched_stop_loss_summary{tag}.csv",
            "trades": RESULTS / f"matched_stop_loss_trades{tag}.csv",
            "skips": RESULTS / f"matched_stop_loss_skips{tag}.csv",
        }
        if not any(p.exists() for p in paths.values()):
            return paths
        suffix += 1


def main():
    groups, date_maps, candidates = load_data()
    print(f"AI 점수 {AI_THRESHOLD:.2f} 이상 후보: {len(candidates):,}건")

    # --------------------------------------------------
    # 공통 진입 집합 구성
    # 한 종목의 이전 공통 거래가 네 전략 모두에서 종료될 때까지
    # 다음 공통 진입을 허용하지 않음.
    # 따라서 네 전략의 진입 종목·날짜가 동일함.
    # --------------------------------------------------
    accepted = []
    skips = []
    busy_until = {}

    for _, signal in candidates.iterrows():
        ticker = signal["Ticker"]
        signal_date = signal["Date"]

        def skip(reason):
            skips.append({
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
        signal_idx = date_maps[ticker].get(signal_date)

        if signal_idx is None:
            skip("신호 날짜 없음")
            continue

        entry_idx = signal_idx + 1
        if entry_idx >= len(group):
            skip("매수일 데이터 없음")
            continue

        if entry_idx <= busy_until.get(ticker, -1):
            skip("공통 진입 규칙상 기존 포지션 보유 중")
            continue

        entry = group.iloc[entry_idx]
        if not can_trade_open(entry):
            skip("매수일 거래량 또는 시가 조건 불충족")
            continue

        if entry["Locked_Upper"] or entry["Locked_Lower"]:
            skip("매수일 가격제한폭 고정 추정")
            continue

        scheduled_idx = entry_idx + HOLDING_DAYS
        if scheduled_idx >= len(group):
            skip("보유기간 이후 데이터 부족")
            continue

        accepted.append({
            "Signal_Date": signal_date,
            "Ticker": ticker,
            "Stock_Name": signal.get("Stock_Name", ""),
            "AI_Score": signal["AI_Score"],
            "Entry_Index": entry_idx,
        })

        # --------------------------------------------------
        # 각 전략을 같은 진입점에서 시뮬레이션
        # 다음 진입은 네 전략 중 가장 늦은 청산 이후로 제한
        # --------------------------------------------------
        outcomes = {}
        exit_indices = []

        for strategy, stop_pct in STRATEGIES.items():
            result = simulate_one(group, entry_idx, stop_pct)
            outcomes[strategy] = result
            exit_indices.append(result["Exit_Index"])

        common_exit_idx = max(exit_indices)
        busy_until[ticker] = common_exit_idx

        for strategy, result in outcomes.items():
            trades_row = {
                "Strategy": strategy,
                "Signal_Date": signal_date,
                "Stock_Name": signal.get("Stock_Name", ""),
                "Ticker": ticker,
                "AI_Score": signal["AI_Score"],
            }
            trades_row.update({
                k: v for k, v in result.items() if k != "Exit_Index"
            })
            # 비교 검증용 공통 진입 식별값
            trades_row["Entry_ID"] = (
                f"{ticker}_{signal_date.strftime('%Y%m%d')}"
            )
            accepted[-1].setdefault("Entry_ID", trades_row["Entry_ID"])
            all_trade_rows.append(trades_row)

    # --------------------------------------------------
    # 요약 및 파일 저장
    # --------------------------------------------------
    trade_df = pd.DataFrame(all_trade_rows)
    skip_df = pd.DataFrame(skips)
    summary_rows = []

    for strategy, stop_pct in STRATEGIES.items():
        df = trade_df[trade_df["Strategy"] == strategy].copy()
        closed = df[df["Net_Return_Pct"].notna()].copy()
        returns = closed["Net_Return_Pct"]

        summary_rows.append({
            "Strategy": strategy,
            "Stop_Loss_Pct": (
                "None" if stop_pct is None
                else f"{stop_pct * 100:.0f}%"
            ),
            "Common_Entry_Count": len(accepted),
            "Closed_Trades": len(closed),
            "Unresolved_Positions": int(
                df["Exit_Reason"].eq("Unresolved").sum()
            ),
            "Common_Skipped_Signals": len(skip_df),
            "Win_Rate_Pct": (
                (returns > 0).mean() * 100 if len(returns) else np.nan
            ),
            "Average_Net_Return_Pct": (
                returns.mean() if len(returns) else np.nan
            ),
            "Median_Net_Return_Pct": (
                returns.median() if len(returns) else np.nan
            ),
            "Worst_Return_Pct": (
                returns.min() if len(returns) else np.nan
            ),
            "Best_Return_Pct": (
                returns.max() if len(returns) else np.nan
            ),
            "Stop_Exit_Count": int(
                closed["Exit_Reason"].isin([
                    "Stop_Loss", "Gap_Stop",
                    "Delayed_Stop_After_Unfillable_Breach"
                ]).sum()
            ),
            "Gap_Stop_Count": int(
                closed["Exit_Reason"].eq("Gap_Stop").sum()
            ),
            "Lower_Lock_Breach_Count": int(
                df["Lower_Lock_Breach"].fillna(False).sum()
            ),
            "Zero_Volume_Breach_Count": int(
                df["Zero_Volume_Breach"].fillna(False).sum()
            ),
            "Delayed_Exit_Count": int(
                closed["Exit_Reason"].isin([
                    "Delayed_Time_Exit",
                    "Delayed_Stop_After_Unfillable_Breach"
                ]).sum()
            ),
        })

    summary_df = pd.DataFrame(summary_rows)

    # 같은 진입 신호 집합인지 검증
    counts = trade_df.groupby("Strategy")["Entry_ID"].nunique()
    print("\n=== 공통 진입 집합 검증 ===")
    print(counts.to_string())

    if len(counts) and counts.nunique() != 1:
        raise RuntimeError("전략별 진입 건수가 달라졌습니다. 결과를 사용하지 마세요.")

    paths = unique_output_paths()
    summary_df.to_csv(paths["summary"], index=False, encoding="utf-8-sig")
    trade_df.to_csv(paths["trades"], index=False, encoding="utf-8-sig")
    skip_df.to_csv(paths["skips"], index=False, encoding="utf-8-sig")

    print("\n=== 동일 진입 신호 기준 손절 비교 ===")
    print(summary_df.round(3).to_string(index=False))

    print("\n=== 청산 사유별 거래 수 ===")
    print(
        trade_df.groupby(["Strategy", "Exit_Reason"])
        .size().to_string()
    )

    print("\n저장 파일:")
    for path in paths.values():
        print(path)

    print("\n주의:")
    print("- 모든 전략은 동일한 진입 종목·날짜 집합을 사용합니다.")
    print("- 다음 공통 진입은 네 전략 중 가장 늦은 청산 이후에 허용합니다.")
    print("- 손절가는 일봉 저가 기준 근사치이며 실제 체결을 보장하지 않습니다.")
    print("- 갭 하락은 손절가가 아닌 시가로 청산한다고 가정합니다.")
    print("- 하한가 고정 추정 시 청산을 미루며, 실제 주문 대기열은 알 수 없습니다.")
    print("- 미청산 거래는 수익률 요약에서 제외하므로 별도로 확인해야 합니다.")
    print("- 동시 보유 종목 수와 총 투자금은 제한하지 않습니다.")


# 전략별 결과를 모으는 목록
all_trade_rows = []

if __name__ == "__main__":
    main()
