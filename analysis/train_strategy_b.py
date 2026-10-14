
from pathlib import Path
import sys

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

BASE_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(BASE_DIR))

import backtest_stop8_test as base


MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "random_forest_strategy_b.pkl"


def main():

    # 기존 백테스트와 동일한 전처리
    prices, train, test = base.prepare_data()

    # 전체 데이터 결합
    df = pd.concat(
        [train, test],
        ignore_index=True
    )

    df = df.sort_values(
        ["Date", "Ticker"]
    ).reset_index(drop=True)

    # 최근 5거래일 제외:
    # 미래 5일 라벨이 완성되지 않은 데이터의 사용 방지
    dates = sorted(df["Date"].unique())

    if len(dates) <= 5:
        raise ValueError("학습 데이터가 부족합니다.")

    cutoff = pd.Timestamp(dates[-5])

    train_final = df[
        df["Date"] < cutoff
    ].copy()

    print("===== 전략 B 최종 모델 학습 =====")
    print("학습 시작일:", train_final["Date"].min())
    print("학습 종료일:", train_final["Date"].max())
    print("학습 데이터:", len(train_final))
    print("입력 변수:", len(base.FEATURES))
    print("학습 종목 수:", train_final["Ticker"].nunique())

    print("\n연도별 학습 데이터:")
    print(
        train_final.groupby(
            train_final["Date"].dt.year
        ).agg(
            데이터수=("Ticker", "size"),
            종목수=("Ticker", "nunique")
        )
    )

    print("\n===== 학습에 사용된 종목 =====")
    print(
        train_final.groupby("Ticker")
        .agg(
            시작일=("Date", "min"),
            종료일=("Date", "max"),
            데이터수=("Date", "size")
        )
        .to_string()
    )

    print("\n매수 제외 종목:")
    print(base.EXCLUDED_TICKERS)

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        train_final[base.FEATURES],
        train_final["Label"]
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(model, MODEL_PATH)

    print("\n모델 저장 완료!")
    print("경로:", MODEL_PATH)

    print("\n모델 입력 변수:")
    for feature in model.feature_names_in_:
        print("-", feature)


if __name__ == "__main__":
    main()
