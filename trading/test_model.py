
from pathlib import Path
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    BASE_DIR / "analysis" / "model" / "random_forest.pkl"
)

print("모델 경로:", MODEL_PATH)

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"모델 파일이 없습니다: {MODEL_PATH}"
    )

model = joblib.load(MODEL_PATH)

print("\n기존 AI 모델 로딩 성공!")
print("모델 종류:", type(model).__name__)

if hasattr(model, "n_features_in_"):
    print("입력 변수 개수:", model.n_features_in_)

if hasattr(model, "feature_names_in_"):
    print("\n모델 입력 변수:")
    for feature in model.feature_names_in_:
        print("-", feature)
else:
    print("\n모델에 입력 변수 이름이 저장되어 있지 않습니다.")
    print("학습 코드에서 FEATURES 목록을 확인해야 합니다.")
