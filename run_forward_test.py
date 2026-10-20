from pathlib import Path
import subprocess
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parent
ANALYSIS = ROOT / "analysis"

STEPS = [
    ("주가 갱신", "analysis/data_processing/update_live_data.py"),
    ("기술적 지표 생성", "analysis/make_training_data_live.py"),
    ("KOSPI 갱신", "analysis/update_kospi_live.py"),
    ("고정 AI 예측", "predict_frozen_signals.py"),
]

BACKTEST = "backtest_forward_ma40_risk.py"


def run_step(name, script):
    path = ROOT / script

    if not path.is_file():
        raise FileNotFoundError(f"실행 파일 없음: {path}")

    print(f"\n[실행] {name}", flush=True)

    subprocess.run(
        [sys.executable, str(path)],
        cwd=ROOT,
        check=True,
    )

    print(f"[완료] {name}", flush=True)


def get_latest_date():
    path = ANALYSIS / "stock_data_live.csv"

    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path, usecols=["Date"])
    dates = pd.to_datetime(df["Date"], errors="coerce")

    if dates.isna().any():
        raise ValueError("주가 데이터에 잘못된 날짜가 있습니다.")

    return dates.max().strftime("%Y%m%d")


def expected_outputs(date_tag):
    outputs = []

    for variant in ["base", "early_defense"]:
        prefix = f"forward_{date_tag}_live_ma40_{variant}"

        outputs.append(ANALYSIS / f"{prefix}_trades.csv")
        outputs.append(ANALYSIS / f"{prefix}_equity.csv")

    outputs.append(
        ANALYSIS / f"forward_ma40_risk_summary_{date_tag}.csv"
    )

    return outputs


def main():
    print("\n===== 자동 전진 검증 시작 =====", flush=True)

    for name, script in STEPS:
        run_step(name, script)

    date_tag = get_latest_date()
    outputs = expected_outputs(date_tag)

    existing = [path for path in outputs if path.exists()]

    print("\n===== 백테스트 결과 확인 =====")
    print("최신 주가 날짜:", date_tag)
    print("예상 결과 파일:", len(outputs))
    print("기존 결과 파일:", len(existing))

    if len(existing) == len(outputs):
        print("\n이미 해당 날짜의 검증 결과가 있습니다.")
        print("기존 결과를 유지하고 백테스트를 건너뜁니다.")
        return

    if existing:
        print("\n일부 결과 파일만 존재합니다.")

        for path in existing:
            print(" -", path.name)

        raise RuntimeError(
            "부분 결과가 존재하므로 자동 실행을 중단합니다."
        )

    run_step("전진 검증 백테스트", BACKTEST)

    missing = [path for path in outputs if not path.exists()]

    if missing:
        raise RuntimeError(
            f"백테스트 결과 파일 누락: {[p.name for p in missing]}"
        )

    print("\n===== 전체 전진 검증 완료 =====")


if __name__ == "__main__":
    main()
