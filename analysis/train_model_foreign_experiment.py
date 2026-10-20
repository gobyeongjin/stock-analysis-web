

import os
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

BASE_FEATURES = [
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
    "Return_10D",
]

FOREIGN_FEATURES = [
    "Foreign_Ratio_Change_5Obs",
    "Foreign_Holding_Qty_Change_5Obs_Pct",
]


def main():
    # 1. 기존 V3 학습 데이터와 외국인 데이터 불러오기
    data = pd.read_csv("training_data_v3.csv")
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    FOREIGN_PATH = os.path.join(
    os.path.dirname(BASE_DIR),
    "foreign_trend_005930.csv"
)

    foreign = pd.read_csv(
        FOREIGN_PATH,
        dtype={"date": str},
)

    data["Date"] = pd.to_datetime(data["Date"])
    data["DateKey"] = data["Date"].dt.strftime("%Y%m%d")

    # 삼성전자만 사용
    data = data[data["Ticker"] == "005930.KS"].copy()

    # 미래 5거래일 데이터가 실제로 존재하는 행만 사용
    data = data[data["Future_5D_Valid"].astype(str).str.lower() == "true"].copy()

    # 2. 외국인 데이터 날짜 정리
    foreign["date"] = (
        foreign["date"]
        .astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.strip()
        .str.zfill(8)
    )

    for col in ["holding_qty", "holding_ratio"]:
        foreign[col] = pd.to_numeric(
            foreign[col].astype(str).str.replace(",", "", regex=False),
            errors="coerce",
        )

    foreign = foreign.drop_duplicates("date").copy()
    foreign = foreign.sort_values("date").reset_index(drop=True)

    # 3. 외국인 보유량 및 비율 변화 특성 생성
    # 5개 관측치 전과 비교하며, 정확히 5거래일을 뜻하지는 않음.
    foreign["Foreign_Ratio_Change_5Obs"] = (
        foreign["holding_ratio"] - foreign["holding_ratio"].shift(5)
    )

    previous_holding = foreign["holding_qty"].shift(5)

    foreign["Foreign_Holding_Qty_Change_5Obs_Pct"] = (
        (foreign["holding_qty"] / previous_holding - 1) * 100
    )

    foreign_features = foreign.rename(columns={"date": "DateKey"})[
        ["DateKey"] + FOREIGN_FEATURES
    ]

    # 4. 날짜 기준으로 결합
    data = data.merge(
        foreign_features,
        on="DateKey",
        how="inner",
        validate="one_to_one",
    )

    # 무한대 및 결측값 제거
    all_features = BASE_FEATURES + FOREIGN_FEATURES

    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=all_features + ["Label"])
    data = data.sort_values("Date").reset_index(drop=True)

    if len(data) < 100:
        raise ValueError(
            f"실험 데이터가 너무 적습니다: {len(data)}행"
        )

    # 5. 시간순으로 학습/테스트 분리
    split_index = int(len(data) * 0.8)

    train_data = data.iloc[:split_index].copy()
    test_data = data.iloc[split_index:].copy()

    if train_data["Label"].nunique() < 2:
        raise ValueError("학습 데이터에 Label 클래스가 2개 모두 있어야 합니다.")

    if test_data["Label"].nunique() < 2:
        raise ValueError("테스트 데이터에 Label 클래스가 2개 모두 있어야 합니다.")

    # 6. 동일한 행에서 기본 모델과 외국인 특성 추가 모델 비교
    def evaluate(features, name):
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )

        model.fit(train_data[features], train_data["Label"])

        probabilities = model.predict_proba(test_data[features])[:, 1]
        predictions = (probabilities >= 0.5).astype(int)

        print(f"\n===== {name} =====")
        print(f"학습 행 수: {len(train_data)}")
        print(f"테스트 행 수: {len(test_data)}")
        print(f"테스트 양성 비율: {test_data['Label'].mean():.2%}")
        print(f"Accuracy : {accuracy_score(test_data['Label'], predictions):.4f}")
        print(f"Precision: {precision_score(test_data['Label'], predictions, zero_division=0):.4f}")
        print(f"Recall   : {recall_score(test_data['Label'], predictions, zero_division=0):.4f}")
        print(f"F1       : {f1_score(test_data['Label'], predictions, zero_division=0):.4f}")
        print("Confusion matrix:")
        print(confusion_matrix(test_data["Label"], predictions))

        return model

    print("===== 실험 데이터 =====")
    print(f"삼성전자 공통 데이터: {len(data)}행")
    print(f"학습 기간: {train_data['Date'].min().date()} ~ {train_data['Date'].max().date()}")
    print(f"테스트 기간: {test_data['Date'].min().date()} ~ {test_data['Date'].max().date()}")
    print("기본 특성 수:", len(BASE_FEATURES))
    print("추가 외국인 특성 수:", len(FOREIGN_FEATURES))

    baseline_model = evaluate(BASE_FEATURES, "기본 모델: 주가 특성 14개")
    foreign_model = evaluate(
        BASE_FEATURES + FOREIGN_FEATURES,
        "비교 모델: 주가 특성 14개 + 외국인 특성 2개",
    )

    # 결과 모델은 별도 파일로 저장하지 않음.
    # 기존 random_forest.pkl은 변경하지 않음.
    print("\n실험 완료. 기존 모델 파일은 변경하지 않았습니다.")


if __name__ == "__main__":
    main()
