from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

ROOT = Path(__file__).resolve().parent
TRAIN_PATH = ROOT / 'analysis' / 'training_data_v3.csv'
LENDING_PATH = ROOT / 'lending_100.csv'
OUT_PATH = ROOT / 'analysis' / 'lending_100_ab_metrics.csv'
FEATURES = [
    'MA20_Gap', 'MA60_Gap', 'MA20_MA60_Gap', 'MA20_Change',
    'MA60_Change', 'High_20D_Gap', 'Low_20D_Gap', 'RSI',
    'RSI_Change', 'Volume_Ratio', 'Volume_Ratio_5D', 'Return_1D',
    'Return_3D', 'Return_5D', 'Return_10D', 'Volatility_20D',
    'Volatility_Change', 'Close_Position',
]


def main():
    for p in [TRAIN_PATH, LENDING_PATH]:
        if not p.exists():
            raise FileNotFoundError(p)

    print('학습 데이터 로딩...', flush=True)
    d = pd.read_csv(TRAIN_PATH, dtype={'Ticker': str})
    d['Date'] = pd.to_datetime(d['Date'], errors='coerce')
    d['Code'] = d['Ticker'].str.split('.').str[0].str.zfill(6)
    d = d[d['Future_5D_Valid'].astype(str).str.lower().isin(['true', '1'])].copy()

    print('대차잔고 데이터 로딩...', flush=True)
    l = pd.read_csv(LENDING_PATH, dtype={'Ticker': str})
    l['dt'] = pd.to_datetime(l['dt'], errors='coerce')
    l['Code'] = l['Ticker'].str.split('.').str[0].str.zfill(6)
    l['Lending_Balance_Change_Pct'] = pd.to_numeric(l['Lending_Balance_Change_Pct'], errors='coerce')
    l = l[['Code', 'dt', 'Lending_Balance_Change_Pct']].dropna(subset=['Code', 'dt'])
    l = l.sort_values(['Code', 'dt']).drop_duplicates(['Code', 'dt'], keep='last')

    # Shift by one observed lending date per stock, then align the feature to the
    # next market date via backward as-of join. No same-day lending data is used.
    l['Lending_Lag1'] = l.groupby('Code')['Lending_Balance_Change_Pct'].shift(1)
    l = l[['Code', 'dt', 'Lending_Lag1']]
    # Exact-date matching prevents stale features being carried indefinitely.
    merged = d.merge(l, left_on=['Code', 'Date'], right_on=['Code', 'dt'], how='inner', validate='many_to_one')
    merged = merged.replace([np.inf, -np.inf], np.nan)
    merged = merged.dropna(subset=FEATURES + ['Label', 'Lending_Lag1', 'Date'])
    merged = merged.sort_values(['Date', 'Code']).reset_index(drop=True)
    if merged.empty:
        raise RuntimeError('결합된 유효 데이터가 없습니다. 날짜·종목 및 결측값을 확인하세요.')

    dates = np.sort(merged['Date'].unique())
    if len(dates) < 30:
        raise RuntimeError('평가에 사용할 날짜가 너무 적습니다.')
    split_idx = int(len(dates) * 0.8)
    split_date = pd.Timestamp(dates[split_idx])
    # Remove 5 trading dates before test start to avoid overlapping future-5-day labels.
    train_end_idx = split_idx - 5
    if train_end_idx <= 0:
        raise RuntimeError('학습 기간이 너무 짧습니다.')
    train_end_exclusive = pd.Timestamp(dates[train_end_idx])
    train = merged[merged['Date'] < train_end_exclusive]
    test = merged[merged['Date'] >= split_date]
    print(f'결합 후: {len(merged):,}행, {merged.Code.nunique()}종목')
    print(f'학습: {len(train):,}행 ({train.Date.min().date()} ~ {train.Date.max().date()})')
    print(f'평가: {len(test):,}행 ({test.Date.min().date()} ~ {test.Date.max().date()})')
    print(f'학습·평가 사이 제외된 거래일: 5일')

    results = []
    y_train, y_test = train['Label'].astype(int), test['Label'].astype(int)
    for name, cols in [('A_technical', FEATURES), ('B_with_lending', FEATURES + ['Lending_Lag1'])]:
        print(f'\n{name} 학습 중...', flush=True)
        model = RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=10,
                                       class_weight='balanced', random_state=42, n_jobs=-1)
        model.fit(train[cols], y_train)
        probabilities = model.predict_proba(test[cols])[:, 1]
        for threshold in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
            predictions = (probabilities >= threshold).astype(int)
            results.append({
                'Model': name, 'Threshold': threshold, 'Train_Rows': len(train),
                'Test_Rows': len(test), 'Predicted_Positive': int(predictions.sum()),
                'Accuracy': accuracy_score(y_test, predictions),
                'Precision': precision_score(y_test, predictions, zero_division=0),
                'Recall': recall_score(y_test, predictions, zero_division=0),
                'F1': f1_score(y_test, predictions, zero_division=0),
            })

    output = pd.DataFrame(results)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUT_PATH, index=False, encoding='utf-8-sig')
    print('\n=== A/B 비교 ===')
    print(output.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    print('\n결과 저장:', OUT_PATH)
    print('주의: 임계값을 테스트 성능으로 선택하면 성능 추정에 편향이 생깁니다.')
    print('주의: 공표 시각 확인 전까지 1거래일 지연은 보수적인 가정일 뿐입니다.')


if __name__ == '__main__':
    main()
