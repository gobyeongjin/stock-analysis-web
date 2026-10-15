"""Compare A vs B2 using the existing backtest_atr_8pct.py execution engine.
Place this file at stock-analysis-web root; do not modify the existing backtester.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parent
ANALYSIS = ROOT / 'analysis'
sys.path.insert(0, str(ANALYSIS))
try:
    import backtest_atr_8pct as engine
except ModuleNotFoundError as exc:
    raise SystemExit('analysis/backtest_atr_8pct.py 파일이 필요합니다.') from exc

TECH = ['MA20_Gap','MA60_Gap','MA20_MA60_Gap','MA20_Change','MA60_Change',
        'High_20D_Gap','Low_20D_Gap','RSI','RSI_Change','Volume_Ratio',
        'Volume_Ratio_5D','Return_1D','Return_3D','Return_5D','Return_10D',
        'Volatility_20D','Volatility_Change','Close_Position']
LEND = ['Lending_Change_1D','Lending_Change_5D','Lending_Change_20D',
        'Lending_Up_Days_5D','Lending_MA20_Gap']
THRESHOLD = 0.65  # 기존 백테스트 설정 유지. 평가 데이터로 재선택하지 않음.
STRATEGIES = ['DAY_FIXED','DAY_ATR','SWING_FIXED','SWING_ATR']


def load_common_data():
    lending = pd.read_csv(ROOT / 'lending_100.csv', dtype={'Ticker':str}, low_memory=False)
    lending['Code'] = lending['Ticker'].str.split('.').str[0].str.zfill(6)
    lending['dt'] = pd.to_datetime(lending['dt'], errors='coerce')
    lending['rmnd'] = pd.to_numeric(lending['rmnd'].astype(str).str.replace(',','',regex=False), errors='coerce')
    lending = (lending.dropna(subset=['dt','rmnd'])
               .sort_values(['Code','dt']).drop_duplicates(['Code','dt'],keep='last').copy())
    grouped = lending.groupby('Code',sort=False)['rmnd']
    lending['Lending_Change_1D'] = grouped.pct_change(fill_method=None).replace([np.inf,-np.inf],np.nan)*100
    lending['Lending_Change_5D'] = (lending['rmnd']/grouped.shift(5)-1).replace([np.inf,-np.inf],np.nan)*100
    lending['Lending_Change_20D'] = (lending['rmnd']/grouped.shift(20)-1).replace([np.inf,-np.inf],np.nan)*100
    lending['Lending_Up_Days_5D'] = (lending['Lending_Change_1D'].gt(0).astype(float)
        .groupby(lending['Code']).transform(lambda s:s.rolling(5,min_periods=5).sum()))
    ma20 = grouped.transform(lambda s:s.rolling(20,min_periods=20).mean())
    lending['Lending_MA20_Gap'] = (lending['rmnd']/ma20-1).replace([np.inf,-np.inf],np.nan)*100
    # Same one-observation lag as compare_lending_b2.py.
    lending[LEND] = lending.groupby('Code',sort=False)[LEND].shift(1)
    lending = lending[['Code','dt']+LEND]

    data = pd.read_csv(ANALYSIS / 'training_data_v3.csv',dtype={'Ticker':str},low_memory=False)
    data['Date'] = pd.to_datetime(data['Date'],errors='coerce')
    data['Code'] = data['Ticker'].str.split('.').str[0].str.zfill(6)
    data = data[data['Future_5D_Valid'].astype(str).str.lower().isin(['true','1'])].copy()
    data = data.merge(lending,left_on=['Code','Date'],right_on=['Code','dt'],how='inner',validate='many_to_one')
    data = data.replace([np.inf,-np.inf],np.nan).dropna(subset=TECH+LEND+['Label','Date'])
    data = data.sort_values(['Date','Ticker']).reset_index(drop=True)
    dates = np.sort(data['Date'].unique())
    split = int(len(dates)*0.8)
    if split <= 5 or split >= len(dates):
        raise ValueError('날짜가 부족합니다.')
    train = data[data['Date'] < dates[split-5]].copy()
    test = data[data['Date'] >= dates[split]].copy()
    if train.empty or test.empty:
        raise ValueError('학습/평가 데이터가 비어 있습니다.')
    return train,test


def main():
    train,test = load_common_data()
    print(f'공통 표본: 학습 {len(train):,}행 / 평가 {len(test):,}행')
    print(f'평가: {test.Date.min().date()} ~ {test.Date.max().date()}')
    print('초기 자본:', f'{engine.INITIAL_CASH:,.0f}', '원 (기존 코드 설정)')
    prices = engine.load_prices()
    # Include the next trading day after the last signal for execution.
    calendar = np.sort(prices['Date'].unique())
    end_idx = np.searchsorted(calendar, np.datetime64(test['Date'].max()), side='right')
    end_date = calendar[min(end_idx+4, len(calendar)-1)]
    prices = prices[(prices['Date'] >= test['Date'].min()) & (prices['Date'] <= end_date)].copy()
    # Use all stocks for calendar and valuation, as in the original engine.
    if prices.empty:
        raise ValueError('평가 기간의 가격 데이터가 없습니다.')
    atr = prices[['Date','Ticker','ATR14']].copy()
    summaries=[]
    for name,features in [('A_technical',TECH),('B2_lending_trend',TECH+LEND)]:
        print(f'\n{name} 학습 중...',flush=True)
        model = RandomForestClassifier(n_estimators=200,max_depth=8,min_samples_leaf=10,
                    class_weight='balanced',random_state=42,n_jobs=-1)
        model.fit(train[features],train['Label'].astype(int))
        signal = test[['Date','Code']].rename(columns={'Code':'Ticker'}).copy()
        signal['Probability'] = model.predict_proba(test[features])[:,1]
        signal = signal[signal['Probability'] >= THRESHOLD].copy()
        signal = signal.merge(atr,on=['Date','Ticker'],how='left',validate='one_to_one')
        signal = signal.dropna(subset=['ATR14'])
        signal = signal[signal['ATR14'] > 0]
        signal.to_csv(ANALYSIS/f'lending_b2_{name}_signals.csv',index=False,encoding='utf-8-sig')
        print(f'후보 신호: {len(signal):,}건')
        for strategy in STRATEGIES:
            print(f'  {strategy} 실행 중...',flush=True)
            trades,equity = engine.run_backtest_v4(prices,signal,strategy)
            summary = engine.summarize_results(trades,equity,strategy)
            summary['Model'] = name
            summary['Threshold'] = THRESHOLD
            summary['Signal_Count'] = len(signal)
            summaries.append(summary)
            trades.to_csv(ANALYSIS/f'lending_b2_{name}_{strategy}_trades.csv',index=False,encoding='utf-8-sig')
            equity.to_csv(ANALYSIS/f'lending_b2_{name}_{strategy}_equity.csv',index=False,encoding='utf-8-sig')
    out = pd.DataFrame(summaries)
    cols=['Model','Strategy','Threshold','Signal_Count','Final_Equity','Return_Pct',
          'MDD_Pct','Trades','Win_Rate_Pct','Total_PnL']
    out = out[cols]
    path=ANALYSIS/'lending_b2_backtest_summary.csv'
    out.to_csv(path,index=False,encoding='utf-8-sig')
    print('\n=== A vs B2 백테스트 결과 ===')
    print(out.to_string(index=False))
    print('\n저장:',path)
    print('주의: 공표 시차, 체결 유동성, 생존편향 등은 추가 검증이 필요합니다.')

if __name__=='__main__':
    main()
