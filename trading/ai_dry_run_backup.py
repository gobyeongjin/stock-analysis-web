from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

from trading.strategy_signals import StrategySignals


BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    BASE_DIR / "analysis" / "model" / "random_forest_v3.pkl"
)

DATA_PATH = BASE_DIR / "analysis" / "training_data_v3.csv"

THRESHOLD = 0.65


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"모델 없음: {MODEL_PATH}")

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"데이터 없음: {DATA_PATH}")

    model = joblib.load(MODEL_PATH)

    if not hasattr(model, "feature_names_in_"):
        raise ValueError("모델 입력 변수 이름이 없습니다.")

    features = list(model.feature_names_in_)

    data = pd.read_csv(DATA_PATH)
    required = ["Date", "Ticker", "Close", *features]

    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError(f"필수 컬럼 누락: {missing}")

    data["Date"] = pd.to_datetime(data["Date"])

    # 각 종목의 마지막 데이터만 선택
    data = data.sort_values(["Ticker", "Date"])
    latest = data.groupby("Ticker", sort=False).tail(1).copy()

    # 날짜가 오래된 데이터는 실시간 신호로 취급하지 않음
    latest_date = latest["Date"].max()
    latest = latest[latest["Date"] == latest_date].copy()

    latest[features] = latest[features].replace(
        [np.inf, -np.inf], np.nan
    )
    latest = latest.dropna(subset=features + ["Close"])

    if latest.empty:
        raise ValueError("예측 가능한 종목이 없습니다.")

    probabilities = model.predict_proba(latest[features])

    positive_index = list(model.classes_).index(1)
    latest["AI_Probability"] = probabilities[:, positive_index]

    selected = latest[
        latest["AI_Probability"] >= THRESHOLD
    ].sort_values("AI_Probability", ascending=False)

    print("=" * 55)
    print("AI 가상 매매 테스트")
    print("=" * 55)
    print("데이터 기준일:", latest_date.date())
    print("분석 종목 수:", len(latest))
    print("매수 기준:", THRESHOLD)
    print("기준 통과 종목 수:", len(selected))
    print("※ 과거 저장 데이터 기반 테스트, 실시간 추천 아님")
    print()

    strategy = StrategySignals(threshold=THRESHOLD)

    # 테스트용 최대 5종목, 종목당 1주
    for _, row in selected.head(5).iterrows():
        result = strategy.evaluate_buy(
            symbol=str(row["Ticker"]).split(".")[0].zfill(6),
            probability=float(row["AI_Probability"]),
            price=float(row["Close"]),
            quantity=1,
            already_held=False,
        )

        print(
            f"{result['symbol']} | "
            f"AI={result['ai_probability']:.2%} | "
            f"가격={result['price']} | "
            f"상태={result['status']} | "
            f"실제 전송={result['submitted']}"
        )


if __name__ == "__main__":
    main()
