"""Compare A-model SWING_ATR stock trend filters, keeping the existing trading engine.
Place in stock-analysis-web root and run: python backtest_stock_filter_a.py
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
from backtest_market_filter_a import load_kospi


def main():
    train, test = load_common_data()
    print(f'학습 {len(train):,}행 / 평가 {len(test):,}행')
    model = RandomForestClassifier(
        n_estimators=200, max_depth=8, min_samples_leaf=10,
        class_weight='balanced', random_state=42, n_jobs=-1
    )
    model.fit(train[TECH], train['Label'].astype(int))
    signals = test[['Date', 'Code']].rename(columns={'Code': 'Ticker'}).copy()
    signals['Probability'] = model.predict_proba(test[TECH])[:, 1]
    signals = signals[signals['Probability'] >= THRESHOLD].copy()

    # Calculate MAs on full price history, before restricting to test dates.
    prices = engine.load_prices()
    prices = prices.sort_values(['Ticker', 'Date']).copy()
    grouped = prices.groupby('Ticker', sort=False)['Close']
    for n in (40, 60):
        prices[f'Stock_MA{n}'] = grouped.transform(
            lambda s: s.rolling(n, min_periods=n).mean()
        )

    calendar = np.sort(prices['Date'].unique())
    end_idx = np.searchsorted(calendar, np.datetime64(test['Date'].max()), side='right')
    end_date = calendar[min(end_idx + 4, len(calendar) - 1)]
    prices = prices[(prices['Date'] >= test['Date'].min()) &
                    (prices['Date'] <= end_date)].copy()
    signals = signals.merge(
        prices[['Date', 'Ticker', 'ATR14', 'Close', 'Stock_MA40', 'Stock_MA60']],
        on=['Date', 'Ticker'], how='left', validate='one_to_one'
    )
    signals = signals[signals['ATR14'].notna() & (signals['ATR14'] > 0)].copy()

    kospi = load_kospi().sort_values('Date').copy()
    kospi['KOSPI_MA60'] = kospi['Close'].rolling(60, min_periods=60).mean()
    kospi = kospi.rename(columns={'Date': 'Index_Date', 'Close': 'KOSPI_Close'})
    signals = pd.merge_asof(
        signals.sort_values('Date'),
        kospi[['Index_Date', 'KOSPI_Close', 'KOSPI_MA60']].sort_values('Index_Date'),
        left_on='Date', right_on='Index_Date', direction='backward'
    )
    stale = (signals['Date'] - signals['Index_Date']).dt.days > 7
    signals.loc[stale, ['KOSPI_Close', 'KOSPI_MA60']] = np.nan

    variants = {
        'BASE_A': signals,
        'STOCK_MA40': signals[signals['Close'] > signals['Stock_MA40']],
        'STOCK_MA60': signals[signals['Close'] > signals['Stock_MA60']],
        'STOCK_MA40_KOSPI_MA60': signals[
            (signals['Close'] > signals['Stock_MA40']) &
            (signals['KOSPI_Close'] > signals['KOSPI_MA60'])
        ],
    }
    rows = []
    for name, selected in variants.items():
        print(f'{name}: 신호 {len(selected):,}건 / SWING_ATR 실행 중...', flush=True)
        s = selected[['Date', 'Ticker', 'Probability', 'ATR14']].copy()
        trades, equity = engine.run_backtest_v4(prices, s, 'SWING_ATR')
        summary = engine.summarize_results(trades, equity, 'SWING_ATR')
        rows.append({'Filter': name, 'Signal_Count': len(s), **summary})
        trades.to_csv(ANALYSIS / f'stock_filter_{name}_trades.csv', index=False, encoding='utf-8-sig')
        equity.to_csv(ANALYSIS / f'stock_filter_{name}_equity.csv', index=False, encoding='utf-8-sig')

    result = pd.DataFrame(rows)
    output = ANALYSIS / 'stock_filter_a_summary.csv'
    result.to_csv(output, index=False, encoding='utf-8-sig')
    print('\n=== A 모델 종목 추세 필터 비교 ===')
    cols = ['Filter', 'Signal_Count', 'Return_Pct', 'MDD_Pct', 'Trades', 'Win_Rate_Pct', 'Final_Equity']
    print(result[cols].to_string(index=False))
    print('\n저장:', output)
    print('주의: 기존 평가 구간은 탐색용입니다. 전략 선택 후 별도 미사용 기간 검증이 필요합니다.')


if __name__ == '__main__':
    main()
