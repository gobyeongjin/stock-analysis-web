import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score


# =========================================================
# 1. 데이터 불러오기
# =========================================================

data = pd.read_csv("training_data_v2.csv")

data["Date"] = pd.to_datetime(data["Date"])

data = data.sort_values(
    ["Date", "Stock_Name"]
).reset_index(drop=True)


# =========================================================
# 2. V1 Feature
# =========================================================

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


# =========================================================
# 3. 필요한 데이터만 사용
# =========================================================

required_columns = features + [
    "Label",
    "Future_5D_Return",
    "Future_5D_Close_Return"
]

data = data.dropna(
    subset=required_columns
).reset_index(drop=True)


# =========================================================
# 4. 날짜 확인
# =========================================================

dates = sorted(
    data["Date"].unique()
)

print("=" * 80)
print("WALK-FORWARD BACKTEST")
print("=" * 80)

print(
    f"전체 데이터 : {len(data):,}"
)

print(
    f"시작 날짜   : {dates[0]}"
)

print(
    f"종료 날짜   : {dates[-1]}"
)

print()


# =========================================================
# 5. 날짜 구간 설정
# =========================================================

n_dates = len(dates)

# 전체 기간을 5개 구간으로 나눔
# 앞쪽은 학습, 뒤쪽은 테스트

test_periods = 5

test_size = n_dates // test_periods


results = []


# =========================================================
# 6. Walk Forward
# =========================================================

for i in range(test_periods):

    test_start_idx = i * test_size

    test_end_idx = (
        (i + 1) * test_size
        if i < test_periods - 1
        else n_dates
    )

    # 최소 학습기간 확보
    if test_start_idx == 0:
        continue

    train_end_idx = test_start_idx

    train_dates = dates[
        :train_end_idx
    ]

    test_dates = dates[
        test_start_idx:test_end_idx
    ]

    train_start_date = train_dates[0]
    train_end_date = train_dates[-1]

    test_start_date = test_dates[0]
    test_end_date = test_dates[-1]


    # -----------------------------------------------------
    # 데이터 분리
    # -----------------------------------------------------

    train = data[
        data["Date"].isin(train_dates)
    ].copy()

    test = data[
        data["Date"].isin(test_dates)
    ].copy()


    print()
    print("=" * 80)
    print(f"PERIOD {i}")
    print("=" * 80)

    print(
        f"Train : {train_start_date} ~ {train_end_date}"
    )

    print(
        f"Test  : {test_start_date} ~ {test_end_date}"
    )

    print(
        f"Train Rows : {len(train):,}"
    )

    print(
        f"Test Rows  : {len(test):,}"
    )


    # -----------------------------------------------------
    # Feature / Target
    # -----------------------------------------------------

    X_train = train[features]

    y_train = train["Label"]

    X_test = test[features]

    y_test = test["Label"]


    # -----------------------------------------------------
    # Random Forest
    # -----------------------------------------------------

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


    # -----------------------------------------------------
    # 예측 확률
    # -----------------------------------------------------

    probabilities = model.predict_proba(
        X_test
    )[:, 1]


    # -----------------------------------------------------
    # Threshold
    # -----------------------------------------------------

    thresholds = [
        0.70,
        0.75,
        0.80
    ]


    for threshold in thresholds:

        selected = (
            probabilities >= threshold
        )

        selected_count = selected.sum()

        if selected_count == 0:
            continue


        y_pred = selected.astype(int)


        # -------------------------------------------------
        # Classification
        # -------------------------------------------------

        precision = precision_score(
            y_test,
            y_pred,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            y_pred,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            y_pred,
            zero_division=0
        )


        # -------------------------------------------------
        # 실제 수익률
        # -------------------------------------------------

        selected_returns = test.loc[
            selected,
            "Future_5D_Return"
        ]

        selected_close_returns = test.loc[
            selected,
            "Future_5D_Close_Return"
        ]


        avg_return = (
            selected_returns.mean()
        )

        median_return = (
            selected_returns.median()
        )

        avg_close_return = (
            selected_close_returns.mean()
        )

        median_close_return = (
            selected_close_returns.median()
        )


        selection_rate = (
            selected_count /
            len(test)
        )


        positive_rate = (
            selected_returns >= 10
        ).mean()


        # -------------------------------------------------
        # 출력
        # -------------------------------------------------

        print()

        print(
            f"Threshold {threshold:.2f}"
        )

        print(
            f"Selected      : {selected_count:,}"
        )

        print(
            f"Selection     : {selection_rate:.2%}"
        )

        print(
            f"Precision     : {precision:.2%}"
        )

        print(
            f"Recall        : {recall:.2%}"
        )

        print(
            f"F1            : {f1:.4f}"
        )

        print(
            f"Avg 5D High   : {avg_return:.2f}%"
        )

        print(
            f"Median 5D High: {median_return:.2f}%"
        )

        print(
            f"Avg Close     : {avg_close_return:.2f}%"
        )

        print(
            f"Median Close  : {median_close_return:.2f}%"
        )

        print(
            f"+10% Rate     : {positive_rate:.2%}"
        )


        # -------------------------------------------------
        # 저장
        # -------------------------------------------------

        results.append({

            "Period": i,

            "Train_Start": train_start_date,

            "Train_End": train_end_date,

            "Test_Start": test_start_date,

            "Test_End": test_end_date,

            "Train_Rows": len(train),

            "Test_Rows": len(test),

            "Threshold": threshold,

            "Selected": selected_count,

            "Selection_Rate": selection_rate,

            "Precision": precision,

            "Recall": recall,

            "F1": f1,

            "Avg_5D_Return": avg_return,

            "Median_5D_Return": median_return,

            "Avg_5D_Close_Return": avg_close_return,

            "Median_5D_Close_Return": median_close_return,

            "Positive_10_Rate": positive_rate
        })


# =========================================================
# 7. 결과 DataFrame
# =========================================================

results_df = pd.DataFrame(
    results
)


# =========================================================
# 8. CSV 저장
# =========================================================

results_df.to_csv(
    "backtest_walk_forward_results.csv",
    index=False,
    encoding="utf-8-sig"
)


# =========================================================
# 9. 전체 요약
# =========================================================

print()
print()
print("=" * 100)
print("WALK-FORWARD SUMMARY")
print("=" * 100)


summary = (
    results_df
    .groupby("Threshold")
    [[
        "Selection_Rate",
        "Precision",
        "Recall",
        "F1",
        "Avg_5D_Return",
        "Median_5D_Return",
        "Avg_5D_Close_Return",
        "Median_5D_Close_Return",
        "Positive_10_Rate"
    ]]
    .mean()
    .reset_index()
)


print(
    summary.to_string(
        index=False
    )
)


# =========================================================
# 10. 기간별 상세 결과
# =========================================================

print()
print()
print("=" * 100)
print("PERIOD-BY-PERIOD RESULT")
print("=" * 100)


detail = results_df[
    results_df["Threshold"].isin(
        [0.70, 0.80]
    )
][[
    "Period",
    "Test_Start",
    "Test_End",
    "Threshold",
    "Selection_Rate",
    "Precision",
    "Recall",
    "F1",
    "Avg_5D_Return",
    "Avg_5D_Close_Return"
]]


print(
    detail.to_string(
        index=False
    )
)


print()
print("=" * 80)
print("완료")
print(
    "결과 파일: backtest_walk_forward_results.csv"
)
print("=" * 80)