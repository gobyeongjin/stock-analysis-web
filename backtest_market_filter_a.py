"""Compare baseline A SWING_ATR vs KOSPI MA20/MA60 market filters.
Run from stock-analysis-web root: python backtest_market_filter_a.py
Requires analysis/backtest_atr_8pct.py, backtest_lending_b2.py, and their input CSVs.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parent
ANALYSIS = ROOT / 'analysis'
sys.path.insert(0, str(ANALYSIS))
import backtest_atr_8pct as engine
from backtest_lending_b2 import load_common_data, TECH, THRESHOLD

INDEX_CSV = ANALYSIS / 'kospi_benchmark.csv'
OUT = ANALYSIS / 'market_filter_a_summary.csv'


def load_kospi():
    if INDEX_CSV.exists():
        idx = pd.read_csv(INDEX_CSV)
        idx.columns = [str(c).strip() for c in idx.columns]
        date_col = next((c for c in ['Date','date','날짜','Datetime'] if c in idx), None)
        close_col = next((c for c in ['Close','close','Adj Close','종가','KOSPI','kospi'] if c in idx), None)
        if date_col and close_col:
            x = idx[[date_col,close_col]].copy()
            x.columns = ['Date','Close']
            x['Date'] = pd.to_datetime(x['Date'],errors='coerce')
            x['Close'] = pd.to_numeric(x['Close'].astype(str).str.replace(',','',regex=False),errors='coerce')
            x = x.dropna().sort_values('Date').drop_duplicates('Date',keep='last')
            if len(x) >= 80:
                print('KOSPI 로컬 CSV 사용:',INDEX_CSV)
                return x
        print('기존 KOSPI CSV에 날짜/종가 열이 없거나 데이터가 부족하여 yfinance 조회를 시도합니다.')
    try:
        import yfinance as yf
    except ImportError as e:
        raise SystemExit('KOSPI 데이터가 필요합니다. analysis/kospi_benchmark.csv를 준비하거나 python -m pip install yfinance 실행') from e
    print('KOSPI (^KS11) 다운로드 중...')
    x = yf.download('^KS11',start='2021-01-01',end='2026-10-15',auto_adjust=False,progress=False)
    if x.empty:
        raise RuntimeError('KOSPI 다운로드 실패. 네트워크 연결 또는 CSV 파일을 확인하세요.')
    if isinstance(x.columns,pd.MultiIndex):
        x.columns = x.columns.get_level_values(0)
    x = x.reset_index()[['Date','Close']]
    x['Date'] = pd.to_datetime(x['Date']).dt.tz_localize(None)
    x['Close'] = pd.to_numeric(x['Close'],errors='coerce')
    return x.dropna().sort_values('Date').drop_duplicates('Date')


def main():
    train,test = load_common_data()
    print(f'학습 {len(train):,}행, 평가 {len(test):,}행')
    model = RandomForestClassifier(n_estimators=200,max_depth=8,min_samples_leaf=10,
                                   class_weight='balanced',random_state=42,n_jobs=-1)
    model.fit(train[TECH],train['Label'].astype(int))
    signal = test[['Date','Code']].rename(columns={'Code':'Ticker'}).copy()
    signal['Probability'] = model.predict_proba(test[TECH])[:,1]
    signal = signal[signal['Probability'] >= THRESHOLD].copy()

    prices = engine.load_prices()
    calendar = np.sort(prices['Date'].unique())
    end_idx = np.searchsorted(calendar,np.datetime64(test['Date'].max()),side='right')
    end_date = calendar[min(end_idx+4,len(calendar)-1)]
    prices = prices[(prices['Date']>=test['Date'].min())&(prices['Date']<=end_date)].copy()
    atr = prices[['Date','Ticker','ATR14']]
    signal = signal.merge(atr,on=['Date','Ticker'],how='left',validate='one_to_one')
    signal = signal.dropna(subset=['ATR14'])
    signal = signal[signal['ATR14']>0].copy()

    kospi = load_kospi()
    for n in [20,60]:
        kospi[f'MA{n}'] = kospi['Close'].rolling(n,min_periods=n).mean()
    kospi = kospi.rename(columns={'Close':'KOSPI_Close'})
    # As-of backward ensures only index data at or before signal day is used.
    signal = pd.merge_asof(signal.sort_values('Date'),
                           kospi[['Date','KOSPI_Close','MA20','MA60']].sort_values('Date'),
                           on='Date',direction='backward')
    # No forward-fill across stale index observations (>7 calendar days).
    index_dates = kospi[['Date']].rename(columns={'Date':'Index_Date'}).sort_values('Index_Date')
    signal = pd.merge_asof(signal.sort_values('Date'),index_dates,left_on='Date',right_on='Index_Date',direction='backward')
    stale = (signal['Date']-signal['Index_Date']).dt.days > 7
    for c in ['KOSPI_Close','MA20','MA60']:
        signal.loc[stale,c] = np.nan

    variants = {
        'BASE_A':signal,
        'KOSPI_MA20':signal[signal['KOSPI_Close']>signal['MA20']],
        'KOSPI_MA60':signal[signal['KOSPI_Close']>signal['MA60']],
        'KOSPI_MA20_AND_MA60':signal[(signal['KOSPI_Close']>signal['MA20']) & (signal['KOSPI_Close']>signal['MA60'])],
    }
    rows=[]
    for name,signals in variants.items():
        print(f'{name}: 신호 {len(signals):,}건 / SWING_ATR 실행 중...',flush=True)
        s = signals[['Date','Ticker','Probability','ATR14']].copy()
        trades,equity = engine.run_backtest_v4(prices,s,'SWING_ATR')
        summary = engine.summarize_results(trades,equity,'SWING_ATR')
        rows.append({'Filter':name,'Signal_Count':len(s),**summary})
        trades.to_csv(ANALYSIS/f'market_filter_{name}_trades.csv',index=False,encoding='utf-8-sig')
        equity.to_csv(ANALYSIS/f'market_filter_{name}_equity.csv',index=False,encoding='utf-8-sig')
    result = pd.DataFrame(rows)
    result.to_csv(OUT,index=False,encoding='utf-8-sig')
    print('\n=== A 모델 시장 추세 필터 비교 ===')
    print(result[['Filter','Signal_Count','Return_Pct','MDD_Pct','Trades','Win_Rate_Pct','Final_Equity']].to_string(index=False))
    print('\n저장:',OUT)
    print('주의: 이 기간은 이미 여러 전략을 비교한 탐색 구간입니다. 개선 결과는 별도 기간에서 재검증하세요.')

if __name__=='__main__':
    main()
