from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

import backtest_atr_8pct as base


BASE_DIR = Path(__file__).resolve().parent

TRAIN_FILE = BASE_DIR / "training_data_v4_lending.csv"
MODEL_PATH = BASE_DIR / "model" / "random_forest_strategy_b_500.pkl"

FEATURES = base.FEATURES


def main():

    # 1. 학습 데이터 로딩
    df = pd.read_csv(
        TRAIN_FILE,
        dtype={"Ticker": str}
    )

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    # 2. 모델 입력 변수와 라벨만 검사
    for col in FEATURES + ["Label"]:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    if "Future_5D_Valid" in df.columns:
        valid = (
            df["Future_5D_Valid"]
            .astype(str)
            .str.strip()
            .str.lower()
            .isin(["true", "1", "1.0"])
        )
        df = df[valid].copy()

    df = df.dropna(
        subset=FEATURES + ["Label", "Date", "Ticker"]
    )

    df = df[df["Label"].isin([0, 1])].copy()

    df["Label"] = df["Label"].astype(int)

    df = df.sort_values(
        ["Date", "Ticker"]
    ).reset_index(drop=True)

    print("===== 전략 B 500종목 학습 =====")
    print("전체 데이터:", len(df))
    print("종목 수:", df["Ticker"].nunique())
    print("시작일:", df["Date"].min())
    print("종료일:", df["Date"].max())
    print("입력 변수:", len(FEATURES))

    # 3. 최근 5거래일 제외
    dates = sorted(df["Date"].unique())

    if len(dates) <= 5:
        raise ValueError("학습 데이터가 부족합니다.")

    cutoff = pd.Timestamp(dates[-5])

    train = df[df["Date"] < cutoff].copy()

    print("\n===== 최종 학습 데이터 =====")
    print("학습 행:", len(train))
    print("학습 종목:", train["Ticker"].nunique())
    print("학습 종료일:", train["Date"].max())

    # 4. RandomForest 학습
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        train[FEATURES],
        train["Label"]
    )

    # 5. 저장
    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(model, MODEL_PATH)

    print("\n모델 저장 완료!")
    print(MODEL_PATH)

    print("\n입력 변수:")
    for feature in model.feature_names_in_:
        print("-", feature)


if __name__ == "__main__":
    main()