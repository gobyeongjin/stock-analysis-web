import os
import time
import calendar
import json
import random
from datetime import date
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / '.env')
APP_KEY = os.getenv('KIWOOM_APP_KEY')
SECRET_KEY = os.getenv('KIWOOM_SECRET_KEY')
BASE_URL = os.getenv('KIWOOM_BASE_URL', 'https://mockapi.kiwoom.com').rstrip('/')

START_YEAR, START_MONTH = 2021, 10
END_YEAR, END_MONTH = 2026, 10
INPUT_CSV = BASE_DIR / 'analysis' / 'stock_data_500.csv'
OUTPUT_CSV = BASE_DIR / 'lending_100.csv'
PROGRESS_FILE = BASE_DIR / 'lending_100_progress.json'
DELAY_SECONDS = 1.5
MAX_RETRIES = 6

session = requests.Session()


def get_access_token():
    response = session.post(
        f'{BASE_URL}/oauth2/token',
        headers={'Content-Type': 'application/json;charset=UTF-8'},
        json={'grant_type': 'client_credentials', 'appkey': APP_KEY, 'secretkey': SECRET_KEY},
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()
    if data.get('return_code') != 0 or not data.get('token'):
        raise RuntimeError(f'토큰 발급 실패: {data}')
    return data['token']


def fetch_month(token, code, year, month):
    last_day = calendar.monthrange(year, month)[1]
    body = {
        'stk_cd': code,
        'strt_dt': f'{year}{month:02d}01',
        'end_dt': f'{year}{month:02d}{last_day:02d}',
        'all_tp': '0',
    }
    headers = {
        'Content-Type': 'application/json;charset=UTF-8',
        'authorization': f'Bearer {token}',
        'api-id': 'ka20068',
        'cont-yn': 'N',
        'next-key': '',
    }
    for attempt in range(MAX_RETRIES + 1):
        try:
            response = session.post(
                f'{BASE_URL}/api/dostk/slb', headers=headers, json=body, timeout=30,
            )
            if response.status_code == 429 or response.status_code in (500, 502, 503, 504):
                if attempt == MAX_RETRIES:
                    response.raise_for_status()
                retry_after = response.headers.get('Retry-After', '')
                try:
                    wait = min(max(float(retry_after), 0), 120)
                except ValueError:
                    wait = min(3 * (2 ** attempt), 90)
                wait = max(wait, 1.5) + random.uniform(0, 0.5)
                print(f'    HTTP {response.status_code}: {wait:.1f}초 대기 후 재시도 ({attempt+1}/{MAX_RETRIES})')
                time.sleep(wait)
                continue
            response.raise_for_status()
            data = response.json()
            if data.get('return_code', 0) != 0:
                raise RuntimeError(f'API 오류 {code} {year}-{month:02d}: {data}')
            rows = data.get('dbrt_trde_trnsn', [])
            if not isinstance(rows, list):
                raise RuntimeError(f'예상과 다른 데이터 형식: {type(rows).__name__}')
            # This TR may paginate; don't silently mark a partial month complete.
            if response.headers.get('cont-yn', 'N').upper() == 'Y':
                raise RuntimeError('연속조회(cont-yn=Y)가 필요한 응답입니다. 페이지 처리 추가 필요')
            return rows
        except (requests.Timeout, requests.ConnectionError) as exc:
            if attempt == MAX_RETRIES:
                raise
            wait = min(3 * (2 ** attempt), 90)
            print(f'    통신 오류 {type(exc).__name__}: {wait}초 후 재시도 ({attempt+1}/{MAX_RETRIES})')
            time.sleep(wait)
    raise RuntimeError('재시도 횟수 초과')


def load_progress():
    if not PROGRESS_FILE.exists():
        return set()
    with PROGRESS_FILE.open('r', encoding='utf-8') as f:
        obj = json.load(f)
    return set(obj.get('completed', []))


def save_progress(completed):
    temp = PROGRESS_FILE.with_suffix('.json.tmp')
    with temp.open('w', encoding='utf-8') as f:
        json.dump({'completed': sorted(completed)}, f, ensure_ascii=False, indent=2)
    temp.replace(PROGRESS_FILE)


def save_rows(rows):
    if not rows:
        return
    incoming = pd.DataFrame(rows)
    if OUTPUT_CSV.exists():
        previous = pd.read_csv(OUTPUT_CSV, dtype={'Ticker': str, 'dt': str})
        combined = pd.concat([previous, incoming], ignore_index=True)
    else:
        combined = incoming
    if not {'Ticker', 'dt'}.issubset(combined.columns):
        raise RuntimeError('저장 데이터에 Ticker 또는 dt 열이 없습니다.')
    combined['Ticker'] = combined['Ticker'].astype(str).str.zfill(6)
    combined['dt'] = combined['dt'].astype(str).str.replace(r'\.0$', '', regex=True)
    combined = combined.drop_duplicates(subset=['Ticker', 'dt'], keep='last')
    temp = OUTPUT_CSV.with_suffix('.csv.tmp')
    combined.to_csv(temp, index=False, encoding='utf-8-sig')
    temp.replace(OUTPUT_CSV)


def finalize():
    if not OUTPUT_CSV.exists():
        print('저장된 데이터가 없습니다.')
        return
    df = pd.read_csv(OUTPUT_CSV, dtype={'Ticker': str, 'dt': str})
    df['dt'] = pd.to_datetime(df['dt'], format='%Y%m%d', errors='coerce')
    df['rmnd'] = pd.to_numeric(df['rmnd'].astype(str).str.replace(',', '', regex=False), errors='coerce')
    df = df.dropna(subset=['dt', 'rmnd']).sort_values(['Ticker', 'dt'])
    prev = df.groupby('Ticker')['rmnd'].shift(1)
    df['Lending_Balance_Change_Pct'] = (df['rmnd'] - prev) / prev.where(prev != 0) * 100
    temp = OUTPUT_CSV.with_suffix('.csv.tmp')
    df.to_csv(temp, index=False, encoding='utf-8-sig')
    temp.replace(OUTPUT_CSV)
    print(f'현재 저장: {df["Ticker"].nunique()}종목, {len(df)}건')


def main():
    if not APP_KEY or not SECRET_KEY:
        raise ValueError('.env에서 키움 API 키를 읽지 못했습니다.')
    if not INPUT_CSV.exists():
        raise FileNotFoundError(INPUT_CSV)
    stocks = (pd.read_csv(INPUT_CSV, dtype={'Ticker': str})[['Stock_Name', 'Ticker']]
              .drop_duplicates(subset=['Ticker']).head(100))
    completed = load_progress()
    if OUTPUT_CSV.exists() and not PROGRESS_FILE.exists():
        print('주의: 기존 CSV가 있지만 진행 기록이 없습니다. 기존 데이터는 보존하며 전 구간을 재조회합니다.')
    token = get_access_token()
    failed = []
    try:
        for _, stock in stocks.iterrows():
            name = stock['Stock_Name']
            code = str(stock['Ticker']).split('.')[0].zfill(6)
            print(f'\n[{name} / {code}]')
            year, month = START_YEAR, START_MONTH
            while (year, month) <= (END_YEAR, END_MONTH):
                if (year, month) > (date.today().year, date.today().month):
                    break
                key = f'{code}:{year:04d}-{month:02d}'
                if key not in completed:
                    try:
                        rows = fetch_month(token, code, year, month)
                        for row in rows:
                            row['Ticker'] = code
                            row['Stock_Name'] = name
                        save_rows(rows)
                        completed.add(key)
                        save_progress(completed)
                        print(f'  {year}-{month:02d}: {len(rows)}건 저장')
                    except Exception as exc:
                        failed.append(key)
                        print(f'  {year}-{month:02d} 실패 (재실행 시 재시도): {exc}')
                        if isinstance(exc, RuntimeError) and ('API 오류' in str(exc) or '연속조회' in str(exc)):
                            print('API 응답 형식/지원 문제일 수 있어 안전하게 중단합니다.')
                            return
                    time.sleep(DELAY_SECONDS)
                year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    except KeyboardInterrupt:
        print('\n중단 요청을 받았습니다. 저장된 데이터와 진행 기록은 유지됩니다.')
    finally:
        finalize()
        print(f'완료 월: {len(completed)}, 이번 실행 실패 월: {len(failed)}')
        if failed:
            print('실패 월 예시:', ', '.join(failed[:10]))
        print('저장 파일:', OUTPUT_CSV)
        print('진행 기록:', PROGRESS_FILE)


if __name__ == '__main__':
    main()
