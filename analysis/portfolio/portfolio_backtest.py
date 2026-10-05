import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier


# ==========================================
# 1. 설정
# ==========================================

THRESHOLDS = [0.65, 0.70, 0.75, 0.80]

INITIAL_CAPITAL = 1.0
DAILY_ALLOCATION = 0.20

# 매수/매도 각각 0.1%
TRANSACTION_COST = 0.001

# 5거래일 보유
HOLDING_DAYS = 5


# ==========================================
# 2. 데이터 불러오기
# ==========================================

training_data = pd.read_csv("training_data.csv")
stock_data = pd.read_csv("stock_data_500.csv")

training_data["Date"] = pd.to_datetime(
    training_data["Date"]
)

stock_data["Date"] = pd.to_datetime(
    stock_data["Date"]
)


# ==========================================
# 3. Feature
# ==========================================

features = [
    "MA20_Gap",
    "MA60_Gap",
    "MA20_MA60_Gap",
    "MA20_Change",
    "MA60_Change",
    "High_20D_Gap",
    "RSI",
    "RSI_Change",
    "Volume_Ratio",
    "Volume_Ratio_5D",
    "Return_1D",
    "Return_3D",
    "Return_5D",
    "Return_10D"
]


# ==========================================
# 4. Train / Test 분리
# ==========================================

data = training_data.sort_values(
    "Date"
).reset_index(drop=True)

split_date = data["Date"].quantile(0.8)

train_data = data[
    data["Date"] <= split_date
].copy()

test_data = data[
    data["Date"] > split_date
].copy()


X_train = train_data[features]
y_train = train_data["Label"]

X_test = test_data[features]


print("================================")
print("Portfolio Backtest")
print("================================")

print(
    f"Train 기간 : "
    f"{train_data['Date'].min().date()} ~ "
    f"{train_data['Date'].max().date()}"
)

print(
    f"Test 기간  : "
    f"{test_data['Date'].min().date()} ~ "
    f"{test_data['Date'].max().date()}"
)


# ==========================================
# 5. Random Forest
# ==========================================

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    min_samples_leaf=10,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train,
    y_train
)

print("Random Forest 학습 완료!")


# ==========================================
# 6. AI Score
# ==========================================

test_data["AI_Score"] = (
    model.predict_proba(X_test)[:, 1]
)


# ==========================================
# 7. 가격 데이터
# ==========================================

price_data = stock_data[
    [
        "Date",
        "Ticker",
        "Open",
        "Close"
    ]
].copy()

price_data = price_data.dropna(
    subset=["Open", "Close"]
)


test_data = test_data.merge(
    price_data,
    on=["Date", "Ticker"],
    how="left"
)

test_data = test_data.dropna(
    subset=["Open", "Close"]
)

test_data = test_data.sort_values(
    ["Date", "Ticker"]
).reset_index(drop=True)


# ==========================================
# 8. 거래일
# ==========================================

dates = sorted(
    test_data["Date"].unique()
)

date_to_index = {
    date: i
    for i, date in enumerate(dates)
}


# ==========================================
# 9. 가격 조회
# ==========================================

price_lookup = price_data.set_index(
    ["Date", "Ticker"]
)


# ==========================================
# 10. Portfolio Backtest
# ==========================================

def run_backtest(threshold):

    cash = INITIAL_CAPITAL

    positions = []

    daily_results = []

    trade_count = 0
    winning_trades = 0
    losing_trades = 0

    total_trade_return = 0.0


    for date_index, current_date in enumerate(dates):

        # --------------------------------------
        # 1. 기존 포지션 청산
        # --------------------------------------

        positions_to_keep = []

        for position in positions:

            if position["Exit_Index"] != date_index:

                positions_to_keep.append(position)
                continue


            ticker = position["Ticker"]

            try:

                exit_price = price_lookup.loc[
                    (current_date, ticker),
                    "Open"
                ]

            except KeyError:

                position["Exit_Index"] += 1

                positions_to_keep.append(position)

                continue


            if pd.isna(exit_price) or exit_price <= 0:

                position["Exit_Index"] += 1

                positions_to_keep.append(position)

                continue


            gross_value = (
                position["Shares"]
                * exit_price
            )

            net_value = (
                gross_value
                * (1 - TRANSACTION_COST)
            )

            cash += net_value


            entry_cost = position[
                "Invested_Capital"
            ]

            trade_return = (
                net_value / entry_cost - 1
            )

            total_trade_return += trade_return

            trade_count += 1


            if trade_return > 0:
                winning_trades += 1
            else:
                losing_trades += 1


        positions = positions_to_keep


        # --------------------------------------
        # 2. 현재 포지션 평가
        # --------------------------------------

        position_value = 0.0

        for position in positions:

            ticker = position["Ticker"]

            try:

                current_price = price_lookup.loc[
                    (current_date, ticker),
                    "Close"
                ]

            except KeyError:

                current_price = position[
                    "Entry_Price"
                ]

            if pd.isna(current_price):

                current_price = position[
                    "Entry_Price"
                ]

            position_value += (
                position["Shares"]
                * current_price
            )


        portfolio_value = (
            cash + position_value
        )


        # --------------------------------------
        # 3. 오늘 후보
        # --------------------------------------

        current_data = test_data[
            test_data["Date"] == current_date
        ].copy()

        candidates = current_data[
            current_data["AI_Score"] >= threshold
        ].copy()


        # --------------------------------------
        # 4. 기존 보유 종목 제외
        # --------------------------------------

        held_tickers = {
            position["Ticker"]
            for position in positions
        }

        candidates = candidates[
            ~candidates["Ticker"].isin(
                held_tickers
            )
        ]


        # --------------------------------------
        # 5. 다음 거래일
        # --------------------------------------

        next_index = date_index + 1

        if next_index >= len(dates):

            next_date = None

        else:

            next_date = dates[next_index]


        # --------------------------------------
        # 6. 신규 투자
        # --------------------------------------

        if (
            next_date is not None
            and len(candidates) > 0
            and cash > 0
        ):

            allocation = min(
                portfolio_value
                * DAILY_ALLOCATION,
                cash
            )


            valid_candidates = []

            for _, row in candidates.iterrows():

                ticker = row["Ticker"]

                try:

                    next_open = price_lookup.loc[
                        (next_date, ticker),
                        "Open"
                    ]

                except KeyError:

                    continue


                if (
                    pd.isna(next_open)
                    or next_open <= 0
                ):

                    continue


                valid_candidates.append(
                    (
                        ticker,
                        next_open
                    )
                )


            if len(valid_candidates) > 0:

                per_stock = (
                    allocation
                    / len(valid_candidates)
                )


                for ticker, entry_price in valid_candidates:

                    required_cash = (
                        per_stock
                        * (1 + TRANSACTION_COST)
                    )


                    if required_cash > cash:

                        continue


                    exit_index = (
                        next_index
                        + HOLDING_DAYS
                    )


                    if exit_index >= len(dates):

                        continue


                    shares = (
                        per_stock
                        / entry_price
                    )


                    cash -= required_cash


                    positions.append({

                        "Ticker": ticker,

                        "Entry_Date": next_date,

                        "Entry_Price": entry_price,

                        "Shares": shares,

                        "Invested_Capital": required_cash,

                        "Exit_Index": exit_index
                    })


        # --------------------------------------
        # 7. 신규 투자 후 평가
        # --------------------------------------

        position_value = 0.0

        for position in positions:

            ticker = position["Ticker"]

            try:

                current_price = price_lookup.loc[
                    (current_date, ticker),
                    "Close"
                ]

            except KeyError:

                current_price = position[
                    "Entry_Price"
                ]

            if pd.isna(current_price):

                current_price = position[
                    "Entry_Price"
                ]

            position_value += (
                position["Shares"]
                * current_price
            )


        portfolio_value = (
            cash + position_value
        )


        daily_results.append({

            "Date": current_date,

            "Portfolio_Value":
                portfolio_value,

            "Cash":
                cash,

            "Position_Value":
                position_value,

            "Active_Positions":
                len(positions)
        })


    # ======================================
    # 결과 DataFrame
    # ======================================

    results = pd.DataFrame(
        daily_results
    )

    results["Daily_Return"] = (
        results["Portfolio_Value"]
        .pct_change()
    )


    # ======================================
    # 성과 계산
    # ======================================

    initial_value = (
        results["Portfolio_Value"].iloc[0]
    )

    final_value = (
        results["Portfolio_Value"].iloc[-1]
    )


    total_return = (
        final_value
        / initial_value
        - 1
    ) * 100


    number_of_days = len(results)


    annual_return = (
        (
            final_value
            / initial_value
        )
        ** (252 / number_of_days)
        - 1
    ) * 100


    daily_returns = (
        results["Daily_Return"]
        .dropna()
    )


    annual_volatility = (
        daily_returns.std()
        * np.sqrt(252)
        * 100
    )


    if daily_returns.std() > 0:

        sharpe = (
            daily_returns.mean()
            / daily_returns.std()
            * np.sqrt(252)
        )

    else:

        sharpe = 0


    # --------------------------------------
    # Drawdown
    # --------------------------------------

    results["High_Watermark"] = (
        results["Portfolio_Value"]
        .cummax()
    )

    results["Drawdown"] = (
        results["Portfolio_Value"]
        / results["High_Watermark"]
        - 1
    )


    max_drawdown = (
        results["Drawdown"].min()
        * 100
    )


    if trade_count > 0:

        win_rate = (
            winning_trades
            / trade_count
            * 100
        )

        avg_trade_return = (
            total_trade_return
            / trade_count
            * 100
        )

    else:

        win_rate = 0
        avg_trade_return = 0


    summary = {

        "Strategy":
            f"AI {threshold:.2f}",

        "Threshold":
            threshold,

        "Final_Value":
            final_value,

        "Total_Return":
            total_return,

        "Annual_Return":
            annual_return,

        "Annual_Volatility":
            annual_volatility,

        "Sharpe":
            sharpe,

        "Max_Drawdown":
            max_drawdown,

        "Trades":
            trade_count,

        "Win_Rate":
            win_rate,

        "Average_Trade_Return":
            avg_trade_return
    }


    return summary, results


# ==========================================
# 11. AI 전략 실행
# ==========================================

all_summaries = []
all_daily_results = {}


for threshold in THRESHOLDS:

    print()
    print(
        f"AI {threshold:.2f} 백테스트 실행..."
    )

    summary, results = run_backtest(
        threshold
    )

    all_summaries.append(summary)

    all_daily_results[
        threshold
    ] = results


# ==========================================
# 12. KOSPI 벤치마크
# ==========================================

print()
print("================================")
print("KOSPI Benchmark 다운로드")
print("================================")


test_start = dates[0]
test_end = dates[-1]


kospi = yf.download(
    "^KS11",
    start=test_start,
    end=test_end + pd.Timedelta(days=1),
    auto_adjust=False,
    progress=False
)


if isinstance(kospi.columns, pd.MultiIndex):

    kospi.columns = (
        kospi.columns
        .get_level_values(0)
    )


kospi = kospi[
    ["Close"]
].dropna()


kospi.index = pd.to_datetime(
    kospi.index
)


# Test 기간에 해당하는 KOSPI만 사용
kospi = kospi[
    (kospi.index >= test_start)
    & (kospi.index <= test_end)
].copy()


# 시작값 = 1
kospi["Benchmark_Value"] = (
    kospi["Close"]
    / kospi["Close"].iloc[0]
)


kospi["Daily_Return"] = (
    kospi["Benchmark_Value"]
    .pct_change()
)


kospi["High_Watermark"] = (
    kospi["Benchmark_Value"]
    .cummax()
)


kospi["Drawdown"] = (
    kospi["Benchmark_Value"]
    / kospi["High_Watermark"]
    - 1
)


kospi_total_return = (
    kospi["Benchmark_Value"].iloc[-1]
    - 1
) * 100


kospi_days = len(kospi)


kospi_annual_return = (
    kospi["Benchmark_Value"].iloc[-1]
    ** (252 / kospi_days)
    - 1
) * 100


kospi_daily_returns = (
    kospi["Daily_Return"]
    .dropna()
)


kospi_volatility = (
    kospi_daily_returns.std()
    * np.sqrt(252)
    * 100
)


if kospi_daily_returns.std() > 0:

    kospi_sharpe = (
        kospi_daily_returns.mean()
        / kospi_daily_returns.std()
        * np.sqrt(252)
    )

else:

    kospi_sharpe = 0


kospi_mdd = (
    kospi["Drawdown"].min()
    * 100
)


kospi_summary = {

    "Strategy":
        "KOSPI Buy & Hold",

    "Threshold":
        np.nan,

    "Final_Value":
        kospi["Benchmark_Value"].iloc[-1],

    "Total_Return":
        kospi_total_return,

    "Annual_Return":
        kospi_annual_return,

    "Annual_Volatility":
        kospi_volatility,

    "Sharpe":
        kospi_sharpe,

    "Max_Drawdown":
        kospi_mdd,

    "Trades":
        1,

    "Win_Rate":
        np.nan,

    "Average_Trade_Return":
        np.nan
}


# ==========================================
# 13. 전체 비교표
# ==========================================

comparison = pd.DataFrame(
    [kospi_summary]
    + all_summaries
)


print()
print("================================")
print("최종 성과 비교")
print("================================")


print(
    comparison[
        [
            "Strategy",
            "Total_Return",
            "Annual_Return",
            "Annual_Volatility",
            "Sharpe",
            "Max_Drawdown",
            "Trades",
            "Win_Rate",
            "Average_Trade_Return"
        ]
    ].to_string(index=False)
)


# ==========================================
# 14. 누적자산 데이터 생성
# ==========================================

equity_curves = pd.DataFrame()


# KOSPI
equity_curves["KOSPI"] = (
    kospi["Benchmark_Value"]
    .reindex(dates)
    .ffill()
)


# AI 전략
for threshold in THRESHOLDS:

    results = all_daily_results[
        threshold
    ]

    strategy_name = (
        f"AI_{threshold:.2f}"
    )

    equity_curves[
        strategy_name
    ] = (
        results
        .set_index("Date")[
            "Portfolio_Value"
        ]
        .reindex(dates)
        .ffill()
    )


equity_curves.index = pd.to_datetime(
    equity_curves.index
)


# ==========================================
# 15. 누적수익률 계산
# ==========================================

cumulative_returns = (
    equity_curves - 1
) * 100


# ==========================================
# 16. CSV 저장
# ==========================================

comparison.to_csv(
    "portfolio_backtest_comparison.csv",
    index=False,
    encoding="utf-8-sig"
)


equity_curves.to_csv(
    "portfolio_equity_curves.csv",
    encoding="utf-8-sig"
)


cumulative_returns.to_csv(
    "portfolio_cumulative_returns.csv",
    encoding="utf-8-sig"
)


kospi.to_csv(
    "kospi_benchmark.csv",
    encoding="utf-8-sig"
)


# ==========================================
# 17. 누적수익률 그래프
# ==========================================

plt.figure(
    figsize=(12, 6)
)


for column in equity_curves.columns:

    plt.plot(
        equity_curves.index,
        equity_curves[column],
        label=column
    )


plt.axhline(
    1.0,
    linestyle="--",
    linewidth=1
)


plt.title(
    "AI Portfolio vs KOSPI"
)

plt.xlabel(
    "Date"
)

plt.ylabel(
    "Portfolio Value"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()


plt.savefig(
    "portfolio_cumulative_return.png",
    dpi=150
)


plt.show()


# ==========================================
# 18. 0.80 Drawdown 그래프
# ==========================================

ai_080 = all_daily_results[
    0.80
].copy()


ai_080["High_Watermark"] = (
    ai_080["Portfolio_Value"]
    .cummax()
)


ai_080["Drawdown"] = (
    ai_080["Portfolio_Value"]
    / ai_080["High_Watermark"]
    - 1
)


plt.figure(
    figsize=(12, 5)
)


plt.plot(
    ai_080["Date"],
    ai_080["Drawdown"] * 100
)


plt.axhline(
    0,
    linestyle="--",
    linewidth=1
)


plt.title(
    "AI 0.80 Portfolio Drawdown"
)

plt.xlabel(
    "Date"
)

plt.ylabel(
    "Drawdown (%)"
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()


plt.savefig(
    "portfolio_drawdown_080.png",
    dpi=150
)


plt.show()


# ==========================================
# 19. KOSPI Drawdown 그래프
# ==========================================

plt.figure(
    figsize=(12, 5)
)


plt.plot(
    kospi.index,
    kospi["Drawdown"] * 100
)


plt.axhline(
    0,
    linestyle="--",
    linewidth=1
)


plt.title(
    "KOSPI Buy & Hold Drawdown"
)

plt.xlabel(
    "Date"
)

plt.ylabel(
    "Drawdown (%)"
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()


plt.savefig(
    "kospi_drawdown.png",
    dpi=150
)


plt.show()


# ==========================================
# 20. 완료
# ==========================================

print()
print("================================")
print("백테스트 완료")
print("================================")

print()
print("생성된 파일:")

print(
    "- portfolio_backtest_comparison.csv"
)

print(
    "- portfolio_equity_curves.csv"
)

print(
    "- portfolio_cumulative_returns.csv"
)

print(
    "- kospi_benchmark.csv"
)

print(
    "- portfolio_cumulative_return.png"
)

print(
    "- portfolio_drawdown_080.png"
)

print(
    "- kospi_drawdown.png"
)