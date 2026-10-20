from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import InconsistentVersionWarning

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "analysis/model/random_forest_v3.pkl"
DATA_PATH = BASE_DIR / "analysis/training_data_v3.csv"

THRESHOLD = 0.65


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"모델 없음: {MODEL_PATH}")

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"데이터 없음: {DATA_PATH}")

    # 모델과 sklearn 버전이 다르면 예측하지 않고 중단
    with warnings.catch_warnings():
        warnings.simplefilter("error", InconsistentVersionWarning)
        model = joblib.load(MODEL_PATH)

    features = list(model.feature_names_in_)

    data = pd.read_csv(
        DATA_PATH,
        dtype={"Ticker": str}
    )

    required = ["Date", "Ticker", *features]
    missing = [c for c in required if c not in data.columns]

    if missing:
        raise ValueError(f"필수 컬럼 누락: {missing}")

    data["Date"] = pd.to_datetime(data["Date"])

    latest = (
        data.sort_values(["Ticker", "Date"])
        .groupby("Ticker", sort=False)
        .tail(1)
        .copy()
    )

    latest_date = latest["Date"].max()
    latest = latest[latest["Date"] == latest_date].copy()

    latest[features] = latest[features].replace(
        [np.inf, -np.inf], np.nan
    )

    latest = latest.dropna(subset=features)

    if latest.empty:
        raise ValueError("예측 가능한 종목이 없습니다.")

    positive_index = list(model.classes_).index(1)

    latest["AI_Probability"] = model.predict_proba(
        latest[features]
    )[:, positive_index]

    selected = latest[
        latest["AI_Probability"] >= THRESHOLD
    ].sort_values(
        "AI_Probability",
        ascending=False
    )

    print("=" * 55)
    print("AI 매수 신호 분석 - 주문 미생성")
    print("=" * 55)

    print("데이터 기준일:", latest_date.date())
    print("분석 종목 수:", len(latest))
    print("AI 기준:", THRESHOLD)
    print("기준 통과 종목 수:", len(selected))

    print("\n[상위 10개 종목]")

    for _, row in selected.head(10).iterrows():
        print(
            f"{row['Ticker']} | "
            f"AI={row['AI_Probability']:.2%}"
        )

    print("\n※ 과거 데이터 기반 분석입니다.")
    print("※ 실제 주문 및 가상 주문 모두 생성하지 않았습니다.")


if __name__ == "__main__":
    main()
