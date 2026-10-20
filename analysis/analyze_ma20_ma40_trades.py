
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

FILE_20 = BASE_DIR / "market_ma20_trades.csv"
FILE_40 = BASE_DIR / "market_ma40_trades.csv"

START_DATE = pd.Timestamp("2026-06-01")
END_DATE = pd.Timestamp("2026-07-31")


def load_trades(path, strategy):
    df = pd.read_csv(path, dtype={"Ticker": str})

    for col in ["Signal_Date", "Entry_Date", "Exit_Date"]:
        df[col] = pd.to_datetime(df[col])

    df["Ticker"] = df["Ticker"].str.zfill(6)
    df["Strategy"] = strategy

    df["PnL"] = pd.to_numeric(df["PnL"])
    df["Return_Pct"] = pd.to_numeric(df["Return_Pct"])

    # 동일 종목의 동일 진입일을 거래 식별자로 사용
    df["Trade_Key"] = (
        df["Ticker"]
        + "_"
        + df["Entry_Date"].dt.strftime("%Y-%m-%d")
    )

    return df


ma20 = load_trades(FILE_20, "MA20")
ma40 = load_trades(FILE_40, "MA40")

keys20 = set(ma20["Trade_Key"])
keys40 = set(ma40["Trade_Key"])

common = keys20 & keys40
only20 = keys20 - keys40
only40 = keys40 - keys20

ma20["Trade_Group"] = ma20["Trade_Key"].apply(
    lambda x: "COMMON" if x in common else "ONLY_MA20"
)

ma40["Trade_Group"] = ma40["Trade_Key"].apply(
    lambda x: "COMMON" if x in common else "ONLY_MA40"
)


def summarize(df):
    if df.empty:
        return {
            "Trades": 0,
            "Win_Rate": 0,
            "Avg_Return": 0,
            "Total_PnL": 0,
        }

    return {
        "Trades": len(df),
        "Win_Rate": round(
            (df["PnL"] > 0).mean() * 100, 2
        ),
        "Avg_Return": round(
            df["Return_Pct"].mean(), 2
        ),
        "Total_PnL": round(
            df["PnL"].sum(), 0
        ),
    }


print("\n===== 전체 거래 비교 =====")

groups = {
    "MA20_ONLY": ma20[
        ma20["Trade_Group"] == "ONLY_MA20"
    ],
    "MA40_ONLY": ma40[
        ma40["Trade_Group"] == "ONLY_MA40"
    ],
    "COMMON_MA20": ma20[
        ma20["Trade_Group"] == "COMMON"
    ],
    "COMMON_MA40": ma40[
        ma40["Trade_Group"] == "COMMON"
    ],
}

summary_rows = []

for name, df in groups.items():
    row = {"Group": name}
    row.update(summarize(df))
    summary_rows.append(row)

summary = pd.DataFrame(summary_rows)
print(summary.to_string(index=False))


# 2026년 6~7월 진입 거래 비교
print("\n===== 2026년 6~7월 진입 거래 =====")

period_rows = []

for name, df in groups.items():
    period = df[
        (df["Entry_Date"] >= START_DATE)
        & (df["Entry_Date"] <= END_DATE)
    ].copy()

    row = {"Group": name}
    row.update(summarize(period))
    period_rows.append(row)

period_summary = pd.DataFrame(period_rows)
print(period_summary.to_string(index=False))


# MA20에서만 발생한 손실 거래
ma20_only = groups["MA20_ONLY"]

losses20 = ma20_only[
    (ma20_only["Entry_Date"] >= START_DATE)
    & (ma20_only["Entry_Date"] <= END_DATE)
    & (ma20_only["PnL"] < 0)
].sort_values("PnL")

print("\n===== MA40에서 발생하지 않은 MA20 손실 거래 =====")

columns = [
    "Ticker",
    "Signal_Date",
    "Entry_Date",
    "Exit_Date",
    "PnL",
    "Return_Pct",
    "Exit_Reason",
]

print(losses20[columns].head(20).to_string(index=False))


# MA40에서만 발생한 수익 거래
ma40_only = groups["MA40_ONLY"]

wins40 = ma40_only[
    (ma40_only["Entry_Date"] >= START_DATE)
    & (ma40_only["Entry_Date"] <= END_DATE)
    & (ma40_only["PnL"] > 0)
].sort_values("PnL", ascending=False)

print("\n===== MA40에서만 발생한 수익 거래 =====")

print(wins40[columns].head(20).to_string(index=False))


# 공통 거래라도 포지션 규모에 따라 손익이 달라질 수 있음
common20 = groups["COMMON_MA20"][
    ["Trade_Key", "PnL", "Return_Pct"]
].rename(columns={
    "PnL": "MA20_PnL",
    "Return_Pct": "MA20_Return",
})

common40 = groups["COMMON_MA40"][
    ["Trade_Key", "PnL", "Return_Pct"]
].rename(columns={
    "PnL": "MA40_PnL",
    "Return_Pct": "MA40_Return",
})

common_compare = common20.merge(
    common40,
    on="Trade_Key",
    how="inner",
)

common_compare["PnL_Diff"] = (
    common_compare["MA40_PnL"]
    - common_compare["MA20_PnL"]
)

print("\n===== 공통 거래 손익 차이 상위 10개 =====")

print(
    common_compare.sort_values(
        "PnL_Diff",
        ascending=False
    ).head(10).to_string(index=False)
)


# 결과 저장
summary.to_csv(
    BASE_DIR / "ma_trade_group_summary.csv",
    index=False,
    encoding="utf-8-sig",
)

period_summary.to_csv(
    BASE_DIR / "ma_trade_period_summary.csv",
    index=False,
    encoding="utf-8-sig",
)

losses20.to_csv(
    BASE_DIR / "ma20_only_losses_jun_jul.csv",
    index=False,
    encoding="utf-8-sig",
)

wins40.to_csv(
    BASE_DIR / "ma40_only_wins_jun_jul.csv",
    index=False,
    encoding="utf-8-sig",
)

common_compare.to_csv(
    BASE_DIR / "ma_common_trade_comparison.csv",
    index=False,
    encoding="utf-8-sig",
)

print("\n분석 CSV 저장 완료")
