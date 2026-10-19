"""Crash regime experiment: A technical model, SWING_ATR, KOSPI drawdown entry filters.
Run at repository root: python backtest_walkforward_a_crash.py
Requires backtest_engine_sizing.py and CSV inputs:
lending_100.csv, analysis/training_data_v3.csv, analysis/stock_data_500.csv,
analysis/kospi_history.csv.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parent
ANALYSIS = ROOT / "analysis"
sys.path.insert(0, str(ROOT))
import backtest_engine_cost_debug as engine
# Standalone feature definitions; no backtest_lending_b2.py dependency.
TECH = ["MA20_Gap", "MA60_Gap", "MA20_MA60_Gap", "MA20_Change", "MA60_Change",
        "High_20D_Gap", "Low_20D_Gap", "RSI", "RSI_Change", "Volume_Ratio",
        "Volume_Ratio_5D", "Return_1D", "Return_3D", "Return_5D", "Return_10D",
        "Volatility_20D", "Volatility_Change", "Close_Position"]
LEND = ["Lending_Change_1D", "Lending_Change_5D", "Lending_Change_20D",
        "Lending_Up_Days_5D", "Lending_MA20_Gap"]
THRESHOLD = 0.65
# Use the full index history. Never silently fall back to the short benchmark CSV.
def load_kospi():
    path = ANALYSIS / "kospi_history.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} 파일이 없습니다. 2021년 이후 KOSPI 데이터를 수집하세요.")
    idx = pd.read_csv(path)
    if not {"Date", "Close"}.issubset(idx.columns):
        raise ValueError("kospi_history.csv 에 Date, Close 열이 필요합니다.")
    idx["Date"] = pd.to_datetime(idx["Date"], errors="coerce")
    idx["Close"] = pd.to_numeric(idx["Close"], errors="coerce")
    idx = idx.dropna(subset=["Date", "Close"]).sort_values("Date").drop_duplicates("Date")
    if idx["Date"].min() > pd.Timestamp("2024-06-01"):
        raise ValueError("KOSPI 과거 데이터가 부족해 MA60 검증이 불가능합니다.")
    print("KOSPI 전체 이력 사용:", path)
    return idx

# Fixed, predeclared folds: 60%-70%, 70%-80%, 80%-90%, 90%-100%.
# Five trading-date embargo before each test fold.
START_FRACTION = 0.60
FOLDS = 4
PURGE_DATES = 5
STRATEGY = "SWING_ATR"



def load_technical_universe():
    """Load technical features without lending data."""
    path = ANALYSIS / "training_data_v3.csv"

    df = pd.read_csv(
        path,
        dtype={"Ticker": str},
        low_memory=False
    )

    required = TECH + [
        "Date", "Ticker", "Label", "Future_5D_Valid"
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"필수 컬럼 누락: {missing}")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Code"] = (
        df["Ticker"].str.split(".", regex=False)
        .str[0].str.zfill(6)
    )

    valid = (
        df["Future_5D_Valid"].astype(str)
        .str.strip().str.lower()
        .isin(["true", "1", "1.0"])
    )

    df = df.loc[valid].copy()

    for col in TECH + ["Label"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(
        subset=TECH + ["Label", "Date", "Code"]
    )

    df = df[df["Label"].isin([0, 1])]
    df = df.sort_values(["Date", "Ticker"])
    df = df.drop_duplicates(
        ["Date", "Code"], keep="last"
    ).reset_index(drop=True)

    if df.empty:
        raise ValueError("유효한 학습 데이터가 없습니다.")

    print(
        f"기술적 지표 데이터: {len(df):,}행, "
        f"{df['Code'].nunique()}종목",
        flush=True
    )

    return df


def fold_ranges(dates):
    n = len(dates)
    start = int(n * START_FRACTION)
    if start < PURGE_DATES + 30:
        raise ValueError("워크포워드 학습에 필요한 날짜가 부족합니다.")
    edges = np.linspace(start, n, FOLDS + 1, dtype=int)
    for i in range(FOLDS):
        a, b = int(edges[i]), int(edges[i + 1])
        if b <= a or a - PURGE_DATES <= 0:
            raise ValueError("워크포워드 분할 오류")
        yield i + 1, dates[a - PURGE_DATES], dates[a], dates[b - 1]


def main():
    df = load_technical_universe()
    dates = np.sort(df["Date"].unique())
    print(f"공통 표본 {len(df):,}행 / 종목 {df['Code'].nunique()}개 / 거래일 {len(dates)}개")
    print(f"임계값 {THRESHOLD}, 초기 자본 {engine.INITIAL_CASH:,.0f}원 (폴드마다 재설정)")

    prices_all = engine.load_prices()
    calendar = np.sort(prices_all["Date"].unique())
    summaries, trades_out, equity_out = [], [], []
    for fold, train_cutoff, test_start, test_end in fold_ranges(dates):
        if fold != 1:
            continue
        train = df[df["Date"] < train_cutoff]
        test = df[(df["Date"] >= test_start) & (df["Date"] <= test_end)]
        if train.empty or test.empty:
            continue
        print(f"\n[폴드 {fold}] 학습 {len(train):,}행, 평가 {len(test):,}행"
              f" | {pd.Timestamp(test_start).date()} ~ {pd.Timestamp(test_end).date()}", flush=True)
        model = RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=10,
            class_weight="balanced", random_state=42, n_jobs=-1
        )
        model.fit(train[TECH], train["Label"].astype(int))
        signals = test[["Date", "Code"]].rename(columns={"Code": "Ticker"}).copy()
        signals["Probability"] = model.predict_proba(test[TECH])[:, 1]
        signals = signals[signals["Probability"] >= THRESHOLD].copy()

        # Allow five following trading dates to execute the last signal and exit.
        end_pos = np.searchsorted(calendar, np.datetime64(test_end), side="right")
        end_pos = min(end_pos + 4, len(calendar) - 1)
        end_price_date = calendar[end_pos]
        prices = prices_all[
            (prices_all["Date"] >= test_start) & (prices_all["Date"] <= end_price_date)
        ].copy()
        signals = signals.merge(
            prices[["Date", "Ticker", "ATR14"]],
            on=["Date", "Ticker"], how="left", validate="one_to_one"
        )
        signals = signals[signals["ATR14"].notna() & (signals["ATR14"] > 0)].copy()

        versions = ["DD_STOP_2"]

        cost_scenarios = {
            "BASE_COST": (0.00015, 0.0005, 0.0015),
            "COST_1": (0.0005, 0.0010, 0.0015),
        }

        for name in versions:
            for cost_name, (commission, slippage, sell_tax) in cost_scenarios.items():
                s = signals[
                    ["Date", "Ticker", "Probability", "ATR14"]
                ].copy()
                s["Size_Factor"] = 1.0

                print(
                    f"  {name} / {cost_name}: 신호 {len(s):,}건, 백테스트 중...",
                    flush=True
                )

                trades, equity = engine.run_backtest_v4(
                    prices, s, STRATEGY,
                    risk_mode=name,
                    commission=commission,
                    slippage=slippage,
                    sell_tax=sell_tax,
                )

                result = engine.summarize_results(
                    trades, equity, STRATEGY
                )

                result.update({
                    "Fold": fold,
                    "Filter": name,
                    "Cost_Scenario": cost_name,
                    "Commission": commission,
                    "Slippage": slippage,
                    "Sell_Tax": sell_tax,
                    "Train_End": (
                        pd.Timestamp(train_cutoff) -
                        pd.Timedelta(days=1)
                    ).date(),
                    "Test_Start": pd.Timestamp(test_start).date(),
                    "Test_End": pd.Timestamp(test_end).date(),
                    "Train_Rows": len(train),
                    "Test_Rows": len(test),
                    "Signal_Count": len(s),
                    "Threshold": THRESHOLD,
                })

                summaries.append(result)

                if not trades.empty:
                    trades = trades.copy()
                    trades.insert(0, "Fold", fold)
                    trades.insert(1, "Filter", name)
                    trades.insert(2, "Cost_Scenario", cost_name)
                    trades_out.append(trades)

                if not equity.empty:
                    equity = equity.copy()
                    equity.insert(0, "Fold", fold)
                    equity.insert(1, "Filter", name)
                    equity.insert(2, "Cost_Scenario", cost_name)
                    equity_out.append(equity)

    if not summaries:
        raise RuntimeError("평가 결과가 없습니다. 입력 데이터를 확인하세요.")

    result = pd.DataFrame(summaries)
    summary_path = ANALYSIS / "walkforward_a_cost_debug_summary.csv"
    result.to_csv(summary_path, index=False, encoding="utf-8-sig")
    if trades_out:
        pd.concat(trades_out, ignore_index=True).to_csv(
            ANALYSIS / "walkforward_a_cost_debug_trades.csv", index=False, encoding="utf-8-sig"
        )
    if equity_out:
        pd.concat(equity_out, ignore_index=True).to_csv(
            ANALYSIS / "walkforward_a_cost_debug_equity.csv", index=False, encoding="utf-8-sig"
        )
    columns = ["Fold", "Filter", "Cost_Scenario", "Test_Start", "Test_End",
               "Signal_Count", "Return_Pct", "MDD_Pct", "Trades", "Win_Rate_Pct"]
    print("\n=== 거래비용 민감도 실험 ===")
    print(result[columns].to_string(index=False))

    # Equal-weight arithmetic fold mean, NOT an investable continuous equity curve.
    overview = (result.groupby(["Filter", "Cost_Scenario"], as_index=False)
                .agg(Avg_Return_Pct=("Return_Pct", "mean"),
                     Avg_MDD_Pct=("MDD_Pct", "mean"),
                     Positive_Folds=("Return_Pct", lambda x: int((x > 0).sum())),
                     Total_Trades=("Trades", "sum")))
    overview[["Avg_Return_Pct", "Avg_MDD_Pct"]] = overview[
        ["Avg_Return_Pct", "Avg_MDD_Pct"]
    ].round(2)
    overview.to_csv(ANALYSIS / "walkforward_a_cost_debug_overview.csv", index=False, encoding="utf-8-sig")
    print("\n=== 폴드별 단순 평균 (연속 복리 수익률 아님) ===")
    print(overview.to_string(index=False))
    print("\n저장:", summary_path)
    print("평가가격 누락일은 마지막 유효 종가로 표시하며, 마지막날 호가 없는 종목은 강제매도하지 않습니다.")
    print("주의: 위험관리 임계값은 탐색용 가설이며 독립적인 최종 OOS 검증은 아닙니다.")
    print("주의: 폴드마다 초기 자본을 재설정합니다. 수익률은 복리 누적 성과가 아닙니다.")
    print("주의: 위험관리 조건은 신규 매수에만 적용됩니다. 기존 보유분은 매도하지 않습니다.")
    print("주의: 이 파일은 기존 가격 누락 평가 로직을 유지하며, 신규 매수 시점의 비중만 조절합니다.")


if __name__ == "__main__":
    main()
