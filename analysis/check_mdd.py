import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

for strategy in ["A_NO_FILTER", "B_VOLUME_1_0"]:

    equity = pd.read_csv(
        BASE_DIR / f"wf_{strategy}_equity.csv",
        parse_dates=["Date"]
    )

    equity = equity.sort_values("Date").copy()

    equity["Peak"] = equity["Equity"].cummax()

    equity["Drawdown_Pct"] = (
        equity["Equity"] / equity["Peak"] - 1
    ) * 100

    worst = equity.nsmallest(
        10, "Drawdown_Pct"
    )

    print(f"\n===== {strategy} 최대 낙폭 날짜 =====")

    print(
        worst[
            [
                "Date",
                "Cash",
                "Market_Value",
                "Equity",
                "Open_Positions",
                "Drawdown_Pct"
            ]
        ].to_string(index=False)
    )