from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

ROOT = Path(__file__).resolve().parent
TECH_PATH = ROOT / 'analysis' / 'training_data_v3.csv'
LEND_PATH = ROOT / 'lending_100.csv'
OUT_PATH = ROOT / 'analysis' / 'lending_b2_comparison.csv'
FEATURES = ['MA20_Gap','MA60_Gap','MA20_MA60_Gap','MA20_Change','MA60_Change',
            'High_20D_Gap','Low_20D_Gap','RSI','RSI_Change','Volume_Ratio',
            'Volume_Ratio_5D','Return_1D','Return_3D','Return_5D','Return_10D',
            'Volatility_20D','Volatility_Change','Close_Position']
LENDING_FEATURES = ['Lending_Change_1D','Lending_Change_5D','Lending_Change_20D',
                    'Lending_Up_Days_5D','Lending_MA20_Gap']

def main():
    for p in (TECH_PATH, LEND_PATH):
        if not p.exists():
            raise FileNotFoundError(f'필요한 파일이 없습니다: {p}')
    print('대차잔고 데이터 읽는 중...', flush=True)
    lend = pd.read_csv(LEND_PATH, dtype={'Ticker': str}, low_memory=False)
    lend['Code'] = lend['Ticker'].astype(str).str.split('.').str[0].str.zfill(6)
    lend['dt'] = pd.to_datetime(lend['dt'], errors='coerce')
    lend['rmnd'] = pd.to_numeric(lend['rmnd'].astype(str).str.replace(',','',regex=False), errors='coerce')
    lend = lend.dropna(subset=['dt','rmnd']).sort_values(['Code','dt']).drop_duplicates(['Code','dt'],keep='last')
    g = lend.groupby('Code',sort=False)['rmnd']
    # 날짜 t의 대차잔고는 다음 거래일부터만 사용한다고 가정한다.
    lend['Lending_Change_1D'] = g.pct_change(fill_method=None).replace([np.inf,-np.inf],np.nan)*100
    lend['Lending_Change_5D'] = (lend['rmnd']/g.shift(5)-1).replace([np.inf,-np.inf],np.nan)*100
    lend['Lending_Change_20D'] = (lend['rmnd']/g.shift(20)-1).replace([np.inf,-np.inf],np.nan)*100
    lend['Lending_Up_Days_5D'] = lend['Lending_Change_1D'].gt(0).astype(float).groupby(lend['Code']).transform(lambda s:s.rolling(5,min_periods=5).sum())
    ma20 = g.transform(lambda s:s.rolling(20,min_periods=20).mean())
    lend['Lending_MA20_Gap'] = ((lend['rmnd']/ma20)-1).replace([np.inf,-np.inf],np.nan)*100
    # 거래일 기준으로 하루 뒤로 옮기고, 다음 날짜에 이미 알려진 이전 거래일 값만 붙인다.
    for f in LENDING_FEATURES:
        lend[f] = lend.groupby('Code',sort=False)[f].shift(1)
    lend = lend[['Code','dt']+LENDING_FEATURES]
    print('기술적 지표 데이터 읽는 중...',flush=True)
    data = pd.read_csv(TECH_PATH, dtype={'Ticker':str},low_memory=False)
    data['Date'] = pd.to_datetime(data['Date'],errors='coerce')
    data['Code'] = data['Ticker'].str.split('.').str[0].str.zfill(6)
    data = data[data['Future_5D_Valid'].astype(str).str.lower().isin(['true','1'])].copy()
    data = data.merge(lend,left_on=['Code','Date'],right_on=['Code','dt'],how='inner',validate='many_to_one')
    data = data.replace([np.inf,-np.inf],np.nan).dropna(subset=FEATURES+LENDING_FEATURES+['Label','Date'])
    data = data.sort_values(['Date','Ticker']).reset_index(drop=True)
    dates = np.sort(data['Date'].unique())
    if len(dates)<50: raise RuntimeError('학습 가능한 날짜가 부족합니다.')
    split_idx = int(len(dates)*0.8)
    split_date = dates[split_idx]
    # 5거래일 미래 라벨이 평가 기간을 침범하지 않도록 경계 직전 5거래일 제거
    train_end_idx = split_idx-5
    if train_end_idx<=0: raise RuntimeError('분할 가능한 날짜가 부족합니다.')
    train_mask = data['Date'] < dates[train_end_idx]
    test_mask = data['Date'] >= split_date
    train = data.loc[train_mask]
    test = data.loc[test_mask]
    print(f'결합 후: {len(data):,}행, {data.Code.nunique()}종목')
    print(f'학습: {len(train):,}행 ({train.Date.min().date()} ~ {train.Date.max().date()})')
    print(f'평가: {len(test):,}행 ({test.Date.min().date()} ~ {test.Date.max().date()})')
    print('학습·평가 경계 5거래일 제외')
    records=[]
    for name,cols in [('A_technical',FEATURES),('B1_lending_1d',FEATURES+LENDING_FEATURES[:1]),('B2_lending_trend',FEATURES+LENDING_FEATURES)]:
        print(f'\n{name} 학습 중...',flush=True)
        model=RandomForestClassifier(n_estimators=200,max_depth=8,min_samples_leaf=10,
                                     class_weight='balanced',random_state=42,n_jobs=-1)
        model.fit(train[cols],train['Label'].astype(int))
        proba=model.predict_proba(test[cols])[:,1]
        y=test['Label'].astype(int)
        for threshold in [0.50,0.55,0.60,0.65,0.70,0.75,0.80]:
            pred=(proba>=threshold).astype(int)
            records.append({'Model':name,'Threshold':threshold,'Train_Rows':len(train),
                            'Test_Rows':len(test),'Predicted_Positive':int(pred.sum()),
                            'Accuracy':accuracy_score(y,pred),
                            'Precision':precision_score(y,pred,zero_division=0),
                            'Recall':recall_score(y,pred,zero_division=0),
                            'F1':f1_score(y,pred,zero_division=0)})
        if name=='B2_lending_trend':
            imp=pd.Series(model.feature_importances_,index=cols)
            print('\nB2 대차잔고 변수 중요도:')
            print(imp.reindex(LENDING_FEATURES).sort_values(ascending=False).to_string())
    result=pd.DataFrame(records)
    OUT_PATH.parent.mkdir(parents=True,exist_ok=True)
    result.to_csv(OUT_PATH,index=False,encoding='utf-8-sig')
    print('\n=== A / B1 / B2 비교 ===')
    print(result.to_string(index=False,float_format=lambda x:f'{x:.4f}'))
    print('\n저장:',OUT_PATH)
    print('주의: 데이터 공표 지연이 1거래일을 초과하면 더 긴 시차가 필요합니다.')

if __name__=='__main__':
    main()
