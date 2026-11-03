from pathlib import Path
import inspect
import textwrap
import types

import numpy as np
import pandas as pd

import backtest_engine_ma60_fixed_levels_volume as ma60

ROOT = Path(__file__).resolve().parent
ANALYSIS = ROOT / "analysis"

START = pd.Timestamp("2026-10-06")
END = pd.Timestamp.max.normalize()

INITIAL_CASH = 1_000_000.0
MAX_POSITIONS = 2
SLIPPAGE = 0.0007

VARIANTS = {
    "BASE": [
        (-0.20, 0.25),
        (-0.15, 0.50),
        (-0.10, 0.75),
    ],
    "EARLY_DEFENSE": [
        (-0.20, 0.125),
        (-0.15, 0.25),
        (-0.10, 0.50),
        (-0.05, 0.75),
    ],
}


def load_live_prices():
    path = ANALYSIS / "stock_data_live.csv"

    df = pd.read_csv(path, dtype={"Ticker": str})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Ticker"] = df["Ticker"].map(ma60.normalize_ticker)

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(
        subset=["Date", "Ticker", "Open", "High", "Low", "Close"]
    )

    df = df.sort_values(["Ticker", "Date"])
    df = df.drop_duplicates(["Ticker", "Date"], keep="last")

    df = df[
        (df["Open"] > 0)
        & (df["High"] > 0)
        & (df["Low"] > 0)
        & (df["Close"] > 0)
        & (df["High"] >= df["Low"])
    ].copy()

    df["Prev_Close"] = df.groupby("Ticker")["Close"].shift(1)

    true_range = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - df["Prev_Close"]).abs(),
            (df["Low"] - df["Prev_Close"]).abs(),
        ],
        axis=1,
    ).max(axis=1)

    df["ATR14"] = true_range.groupby(df["Ticker"]).transform(
        lambda s: s.rolling(14, min_periods=14).mean()
    )

    # ATR 계산은 전체 과거 데이터로 수행한 후 신규 기간만 선택
    return df[
        (df["Date"] >= START)
        & (df["Date"] <= END)
    ].copy()


def load_live_signals():
    path = ANALYSIS / "live_signals_forward.csv"
    df = pd.read_csv(path, dtype={"Ticker": str})

    df["Date"] = pd.to_datetime(df["Date"])
    df["Ticker"] = df["Ticker"].map(ma60.normalize_ticker)
    df["Probability"] = pd.to_numeric(
        df["Probability"], errors="coerce"
    )

    return df[
        (df["Date"] >= START)
        & (df["Date"] <= END)
    ].copy()


def load_live_ma40():
    path = ANALYSIS / "kospi_history_live.csv"
    df = pd.read_csv(path)

    df["Date"] = pd.to_datetime(df["Date"])
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")

    df = (
        df[["Date", "Close"]]
        .dropna()
        .sort_values("Date")
        .drop_duplicates("Date", keep="last")
    )

    df["MA40"] = df["Close"].rolling(
        40, min_periods=40
    ).mean()

    df["Market_OK"] = df["Close"] > df["MA40"]

    return df.set_index("Date")["Market_OK"].to_dict()


def make_engine(levels):
    # 원본 엔진의 함수 코드와 전역 변수들을 복사
    namespace = dict(ma60.__dict__)

    namespace["INITIAL_CASH"] = INITIAL_CASH
    namespace["MAX_POSITIONS"] = MAX_POSITIONS

    # MA60 필터를 신규 MA40 필터로 교체
    namespace["load_ma60_filter"] = load_live_ma40

    source = textwrap.dedent(
        inspect.getsource(ma60.run_backtest_v4)
    )

    start_marker = '        elif risk_mode == "DD_STOP_STEP":'
    end_marker = '        elif risk_mode in ('

    if source.count(start_marker) != 1:
        raise ValueError("위험관리 시작 분기 확인 필요")

    if source.count(end_marker) != 1:
        raise ValueError("위험관리 종료 분기 확인 필요")

    before, remainder = source.split(start_marker, 1)
    _, after = remainder.split(end_marker, 1)

    lines = [start_marker]

    for i, (threshold, factor) in enumerate(levels):
        keyword = "if" if i == 0 else "elif"
        lines.append(
            f"            {keyword} drawdown <= {threshold}:"
        )
        lines.append(f"                dd_factor = {factor}")

    lines.extend([
        "            else:",
        "                dd_factor = 1.0",
        "",
        "            stop_factor = (",
        "                0.5 if recent_stops >= 2 else 1.0",
        "            )",
        "",
        "            risk_factor = dd_factor * stop_factor",
        "",
    ])

    modified = (
        before
        + "\n".join(lines)
        + "\n"
        + end_marker
        + after
    )

    compile(modified, "<live_engine>", "exec")
    exec(modified, namespace)

    return namespace["run_backtest_v4"]


def main():
    prices = load_live_prices()
    signals = load_live_signals()

    if prices.empty or signals.empty:
        raise ValueError("가격 또는 신호 데이터가 비어 있습니다.")

    # 신호 발생일의 ATR14 연결 (미래 데이터 사용 방지)
    atr_data = prices[["Date", "Ticker", "ATR14"]].copy()

    signals = signals.merge(
        atr_data,
        on=["Date", "Ticker"],
        how="left",
        validate="many_to_one",
    )

    signals["ATR14"] = pd.to_numeric(
        signals["ATR14"], errors="coerce"
    )

    missing_atr = signals["ATR14"].isna().sum()

    if missing_atr:
        raise ValueError(f"신호의 ATR14 누락: {missing_atr}건")

    print("ATR14 연결 완료:", len(signals), "건")

    print("\n===== 신규 기간 백테스트 =====")
    print("가격 기간:", prices["Date"].min().date(),
          "~", prices["Date"].max().date())
    print("가격 종목:", prices["Ticker"].nunique())
    print("AI 신호:", len(signals))
    print("KOSPI MA40:", "kospi_history_live.csv")

    rows = []

    for name, levels in VARIANTS.items():
        print(f"\n[{name}] 실행 중...")

        engine = make_engine(levels)

        trades, equity = engine(
            prices.copy(),
            signals.copy(),
            "SWING_ATR",
            risk_mode="DD_STOP_STEP",
            slippage=SLIPPAGE,
        )

        summary = ma60.summarize_results(
            trades, equity, "SWING_ATR"
        )

        latest_date = prices["Date"].max()
        date_tag = pd.Timestamp(latest_date).strftime("%Y%m%d")
        prefix = f"{date_tag}_live_ma40_{name.lower()}"

        trades_path = ANALYSIS / f"forward_{prefix}_trades.csv"
        equity_path = ANALYSIS / f"forward_{prefix}_equity.csv"

        for path in [trades_path, equity_path]:
            if path.exists():
                raise FileExistsError(
                    f"기존 결과 보호: {path}"
                )

        trades.to_csv(
            trades_path, index=False, encoding="utf-8-sig"
        )
        equity.to_csv(
            equity_path, index=False, encoding="utf-8-sig"
        )

        summary["Variant"] = name
        rows.append(summary)

        print("거래:", len(trades))
        print("결과:", summary)

    result = pd.DataFrame(rows)

    summary_path = ANALYSIS / f"forward_ma40_risk_summary_{date_tag}.csv"

    if summary_path.exists():
        raise FileExistsError(
            f"기존 결과 보호: {summary_path}"
        )

    result.to_csv(
        summary_path, index=False, encoding="utf-8-sig"
    )

    print("\n===== BASE vs EARLY_DEFENSE =====")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
