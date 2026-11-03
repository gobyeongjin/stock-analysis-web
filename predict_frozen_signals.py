from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
ANALYSIS = ROOT / "analysis"

MODEL_PATH = ANALYSIS / "rf_frozen_20261002.joblib"
META_PATH = ANALYSIS / "rf_frozen_20261002.json"
FEATURE_PATH = ANALYSIS / "training_data_live.csv"
OUTPUT_PATH = ANALYSIS / "live_signals_forward.csv"


def main():
    # 1. 고정 모델과 설정 불러오기
    model = joblib.load(MODEL_PATH)

    metadata = json.loads(
        META_PATH.read_text(encoding="utf-8")
    )

    features = metadata["features"]
    cutoff = pd.Timestamp(metadata["cutoff"])
    threshold = float(metadata["threshold"])

    print("===== 고정 AI 모델 =====")
    print("학습 기준일:", cutoff.date())
    print("예측 기준 확률:", threshold)
    print("기술적 지표:", len(features), "개")

    # 2. 최신 Feature 데이터 불러오기
    df = pd.read_csv(
        FEATURE_PATH,
        dtype={"Ticker": str, "Code": str},
        low_memory=False,
    )

    df["Date"] = pd.to_datetime(
        df["Date"], errors="coerce"
    )

    # 학습 기준일 이후만 예측
    df = df.loc[df["Date"] > cutoff].copy()

    if df.empty:
        raise ValueError("학습 기준일 이후 신규 데이터가 없습니다.")

    # 3. 모델 입력 데이터 검증
    missing_cols = [
        col for col in features if col not in df.columns
    ]

    if missing_cols:
        raise ValueError(
            f"기술적 지표 컬럼 누락: {missing_cols}"
        )

    for col in features:
        df[col] = pd.to_numeric(
            df[col], errors="coerce"
        )

    df = df.replace(
        [np.inf, -np.inf], np.nan
    )

    before = len(df)
    df = df.dropna(subset=features + ["Date"])

    print("예측 제외 행:", before - len(df))

    if df.empty:
        raise ValueError("예측 가능한 데이터가 없습니다.")

    # 4. 재학습 없이 예측만 수행
    if list(model.classes_) != [0, 1]:
        raise ValueError(
            f"예상과 다른 모델 클래스: {model.classes_}"
        )

    df["Probability"] = model.predict_proba(
        df[features]
    )[:, 1]

    # 5. 기존 전략의 확률 기준 적용
    signals = df.loc[
        df["Probability"] >= threshold
    ].copy()

    signals = signals.sort_values(
        ["Date", "Probability"],
        ascending=[True, False],
    )

    # Ticker에서 6자리 종목코드 생성
    if "Code" not in signals.columns:
        signals["Code"] = (
            signals["Ticker"]
            .astype(str)
            .str.split(".").str[0]
            .str.zfill(6)
        )

    output_cols = [
        "Date", "Code", "Ticker",
        "Stock_Name", "Probability",
    ]

    signals = signals[output_cols]

    if signals.duplicated(["Date", "Ticker"]).any():
        raise ValueError("날짜·종목 기준 중복 신호가 있습니다.")

    # 기존 10월 결과 파일은 건드리지 않음
    signals.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n===== 전진 검증 AI 신호 =====")
    print("예측 시작일:", df["Date"].min().date())
    print("예측 종료일:", df["Date"].max().date())
    print("전체 예측 행:", len(df))
    print("매수 후보 신호:", len(signals))

    print("\n===== 날짜별 신호 =====")
    print(
        signals.groupby("Date")
        .size()
        .to_string()
    )

    print("\n저장:", OUTPUT_PATH)
    print("모델 재학습: 없음")


if __name__ == "__main__":
    main()
