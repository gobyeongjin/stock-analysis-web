from pathlib import Path
import pandas as pd
import numpy as np

BASE = Path(__file__).resolve().parents[1]
RESULTS = BASE / "results"
PRICE_FILE = BASE / "analysis" / "stock_data_500.csv"
CANDIDATE_FILE = BASE / "analysis" / "results" / "backtest_candidates.csv"

AI_THRESHOLD = 0.80
MAX_HOLDINGS = 3
MAX_BUYS_PER_WEEK = 1

FEE = 0.001       # 매수·매도 각각 0.1%
SLIPPAGE = 0.0005 # 매수·매도 각각 0.05%

STRATEGIES = [
    {"name": "Baseline_5D", "days": 5,
     "stop": None, "target": None},
    {"name": "Stable_5D", "days": 5,
     "stop": -0.05, "target": 0.08},
    {"name": "Balanced_5D", "days": 5,
     "stop": -0.08, "target": 0.10},
    {"name": "Balanced_10D", "days": 10,
     "stop": -0.08, "target": 0.10},
]


def unique_path(path):
    """기존 결과 파일을 덮어쓰지 않도록 새 이름 생성"""
    if not path.exists():
        return path

    stamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    new_path = path.with_name(
        f"{path.stem}_{stamp}{path.suffix}"
    )

    count = 2
    while new_path.exists():
        new_path = path.with_name(
            f"{path.stem}_{stamp}_{count}{path.suffix}"
        )
        count += 1

    return new_path


def load_data():
    prices = pd.read_csv(PRICE_FILE)
    candidates = pd.read_csv(CANDIDATE_FILE)

    prices["Date"] = pd.to_datetime(prices["Date"])
    candidates["Date"] = pd.to_datetime(candidates["Date"])

    prices["Ticker"] = prices["Ticker"].astype(str)
    candidates["Ticker"] = candidates["Ticker"].astype(str)

    prices = prices.sort_values(
        ["Ticker", "Date"]
    ).drop_duplicates(["Ticker", "Date"])

    candidates["AI_Score"] = pd.to_numeric(
        candidates["AI_Score"], errors="coerce"
    )

    # AI 점수 0.80 이상인 후보만 사용
    candidates = candidates[
        candidates["AI_Score"] >= AI_THRESHOLD
    ].copy()

    candidates = candidates.sort_values(
        ["Date", "AI_Score"],
        ascending=[True, False]
    ).drop_duplicates(["Date", "Ticker"])

    return prices, candidates
def simulate(prices, candidates, strategy):
    all_trades = []

    price_map = {
        ticker: group.reset_index(drop=True)
        for ticker, group in prices.groupby("Ticker")
    }

    signals = {
        date: group.sort_values(
            "AI_Score", ascending=False
        )
        for date, group in candidates.groupby("Date")
    }

    dates = sorted(prices["Date"].unique())

    holdings = []
    cash = 1.0
    equity_history = []
    weekly_buys = {}

    for date in dates:
        date = pd.Timestamp(date)
        remaining = []

        # 1. 기존 보유 종목 매도 여부 확인
        for pos in holdings:
            frame = price_map[pos["Ticker"]]
            idx = pos["entry_idx"] + pos["held_days"]

            if idx >= len(frame) or frame.loc[idx, "Date"] != date:
                remaining.append(pos)
                continue

            bar = frame.loc[idx]
            pos["held_days"] += 1

            op = float(bar["Open"])
            high = float(bar["High"])
            low = float(bar["Low"])
            close = float(bar["Close"])

            entry = pos["entry_price"]
            stop_price = (
                entry * (1 + strategy["stop"])
                if strategy["stop"] is not None else None
            )
            target_price = (
                entry * (1 + strategy["target"])
                if strategy["target"] is not None else None
            )

            exit_price = None
            reason = None

            # 시가 갭 발생 시 시가 기준 청산
            if stop_price is not None and op <= stop_price:
                exit_price, reason = op, "Gap_Stop"

            elif target_price is not None and op >= target_price:
                exit_price, reason = op, "Gap_Target"

            # 하루 안에 손절·목표가 모두 닿으면 보수적으로 손절 우선
            elif stop_price is not None and low <= stop_price:
                exit_price, reason = stop_price, "Stop_Loss"

            elif target_price is not None and high >= target_price:
                exit_price, reason = target_price, "Take_Profit"

            elif pos["held_days"] >= strategy["days"]:
                exit_price, reason = close, "Time_Exit"

            if exit_price is None:
                remaining.append(pos)
                continue

            net_return = (
                exit_price * (1 - SLIPPAGE)
                / (entry * (1 + SLIPPAGE))
                * (1 - FEE) ** 2
                - 1
            )

            proceeds = pos["capital"] * (1 + net_return)
            cash += proceeds

            all_trades.append({
                "Strategy": strategy["name"],
                "Ticker": pos["Ticker"],
                "Stock_Name": pos["Stock_Name"],
                "Signal_Date": pos["signal_date"],
                "Entry_Date": pos["entry_date"],
                "Exit_Date": date,
                "AI_Score": pos["AI_Score"],
                "Entry_Price": entry,
                "Exit_Price": exit_price,
                "Holding_Days": pos["held_days"],
                "Exit_Reason": reason,
                "Net_Return_Pct": net_return * 100,
                "Net_PnL": pos["capital"] * net_return,
            })

        holdings = remaining

        # 2. 현재 자산 기록
        market_value = 0.0
        for pos in holdings:
            frame = price_map[pos["Ticker"]]
            idx = pos["entry_idx"] + pos["held_days"] - 1
            market_value += (
                pos["capital"]
                * float(frame.loc[idx, "Close"])
                / pos["entry_price"]
            )

        equity = cash + market_value
        equity_history.append({
            "Date": date,
            "Strategy": strategy["name"],
            "Equity": equity,
            "Open_Positions": len(holdings),
        })

        # 3. 주 1회 이하 신규 매수
        iso = date.isocalendar()
        week_key = (iso.year, iso.week)

        if weekly_buys.get(week_key, 0) >= MAX_BUYS_PER_WEEK:
            continue

        if len(holdings) >= MAX_HOLDINGS or date not in signals:
            continue

        held_tickers = {p["Ticker"] for p in holdings}

        for candidate in signals[date].itertuples(index=False):
            ticker = candidate.Ticker

            if ticker in held_tickers or ticker not in price_map:
                continue

            frame = price_map[ticker]
            matches = frame.index[frame["Date"] == date].tolist()

            if not matches:
                continue

            entry_idx = matches[0] + 1
            if entry_idx >= len(frame):
                continue

            bar = frame.loc[entry_idx]

            if bar["Open"] <= 0 or bar["Volume"] <= 0:
                continue

            entry_price = float(bar["Open"])
            capital = min(equity / MAX_HOLDINGS, cash)

            if capital <= 0:
                break

            cash -= capital

            holdings.append({
                "Ticker": ticker,
                "Stock_Name": getattr(
                    candidate, "Stock_Name", ticker
                ),
                "AI_Score": float(candidate.AI_Score),
                "signal_date": date,
                "entry_date": bar["Date"],
                "entry_idx": entry_idx,
                "entry_price": entry_price,
                "held_days": 0,
                "capital": capital,
            })

            weekly_buys[week_key] = (
                weekly_buys.get(week_key, 0) + 1
            )
            break

    return pd.DataFrame(all_trades), pd.DataFrame(equity_history)


def main():
    RESULTS.mkdir(exist_ok=True)

    prices, candidates = load_data()

    if candidates.empty:
        print("AI 점수 0.80 이상인 후보가 없습니다.")
        return

    summaries = []
    trade_results = []
    equity_results = []

    for strategy in STRATEGIES:
        trades, equity = simulate(
            prices, candidates, strategy
        )

        if trades.empty:
            summaries.append({
                "Strategy": strategy["name"],
                "Trades": 0,
            })
            continue

        peak = equity["Equity"].cummax()
        drawdown = (
            equity["Equity"] / peak - 1
        ) * 100

        summaries.append({
            "Strategy": strategy["name"],
            "Trades": len(trades),
            "Win_Rate_Pct": round(
                (trades["Net_Return_Pct"] > 0).mean() * 100, 2
            ),
            "Avg_Return_Pct": round(
                trades["Net_Return_Pct"].mean(), 2
            ),
            "Median_Return_Pct": round(
                trades["Net_Return_Pct"].median(), 2
            ),
            "Worst_Trade_Pct": round(
                trades["Net_Return_Pct"].min(), 2
            ),
            "Total_Return_Pct": round(
                (equity.iloc[-1]["Equity"] - 1) * 100, 2
            ),
            "Max_Drawdown_Pct": round(
                abs(min(0, drawdown.min())), 2
            ),
            "Avg_Holding_Days": round(
                trades["Holding_Days"].mean(), 2
            ),
            "Stop_Count": int(
                trades["Exit_Reason"].isin(
                    ["Stop_Loss", "Gap_Stop"]
                ).sum()
            ),
        })

        trade_results.append(trades)
        equity_results.append(equity)

    outputs = [
        (
            pd.DataFrame(summaries),
            "swing_weekly_summary.csv"
        ),
        (
            pd.concat(trade_results, ignore_index=True)
            if trade_results else pd.DataFrame(),
            "swing_weekly_trades.csv"
        ),
        (
            pd.concat(equity_results, ignore_index=True)
            if equity_results else pd.DataFrame(),
            "swing_weekly_equity.csv"
        ),
    ]

    for df, filename in outputs:
        path = unique_path(RESULTS / filename)
        df.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"저장 완료: {path}")

    print("\n=== 스윙 전략 비교 ===")
    print(pd.DataFrame(summaries).to_string(index=False))


if __name__ == "__main__":
    main()