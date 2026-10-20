
from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
TRADE_FILE = BASE_DIR / "backtest_atr_v2_trades.csv"
EQUITY_FILE = BASE_DIR / "backtest_atr_v2_equity.csv"

trades = pd.read_csv(TRADE_FILE)
equity = pd.read_csv(EQUITY_FILE)

swing = trades[trades["Strategy"] == "SWING_ATR"].copy()
swing["Entry_Date"] = pd.to_datetime(swing["Entry_Date"])
swing["Exit_Date"] = pd.to_datetime(swing["Exit_Date"])

print("\n===== 1. 스윙 ATR 전체 손익 검산 =====")
print("거래 수:", len(swing))
print("거래별 순손익 합계:", round(swing["Net_PnL"].sum(), 2))
print("평균 수익률:", round(swing["Return_Pct"].mean(), 2), "%")
print("승률:", round((swing["Net_PnL"] > 0).mean() * 100, 2), "%")

print("\n===== 2. 손절가 대비 실제 청산 가격 =====")
# ATR 기준 목표가/손절가 자체가 거래 파일에 저장되지 않았으므로
# 매수가 대비 손실률이 큰 거래부터 확인
losses = swing[swing["Return_Pct"] < -5].copy()
print("수익률 -5% 미만 거래 수:", len(losses))

cols = [
    "Ticker", "Entry_Date", "Exit_Date", "Entry_Price",
    "Exit_Price", "Exit_Reason", "Return_Pct", "Net_PnL",
    "Holding_Days",
]
if not losses.empty:
    print(
        losses.sort_values("Return_Pct")[cols]
        .head(20).to_string(index=False)
    )

print("\n===== 3. 수익 상위 거래 집중도 =====")
winners = swing.sort_values("Net_PnL", ascending=False)
total_profit = swing["Net_PnL"].sum()

for n in [1, 3, 5, 10]:
    top_sum = winners.head(n)["Net_PnL"].sum()
    share = top_sum / total_profit * 100 if total_profit else np.nan
    print(f"상위 {n}건 손익 합계: {top_sum:,.0f}원 "
          f"(전체 순손익 대비 {share:.1f}%)")

without_top5 = swing.iloc[5:]["Net_PnL"].sum()
print("상위 5건을 제외한 단순 손익 합계:", round(without_top5), "원")

print("\n===== 4. 보유기간 및 청산 사유 =====")
print(
    swing.groupby("Exit_Reason").agg(
        거래수=("Net_PnL", "size"),
        누적손익=("Net_PnL", "sum"),
        평균수익률=("Return_Pct", "mean"),
        최대손실률=("Return_Pct", "min"),
        평균보유일=("Holding_Days", "mean"),
    ).round(2).to_string()
)

print("\n===== 5. 일별 자산 데이터 확인 =====")
eq = equity[equity["Strategy"] == "SWING_ATR"].copy()
eq["Date"] = pd.to_datetime(eq["Date"])
eq = eq.sort_values("Date")

if not eq.empty:
    print("일별 자산 행 수:", len(eq))
    print("첫 자산:", round(eq.iloc[0]["Equity"]))
    print("마지막 자산:", round(eq.iloc[-1]["Equity"]))
    print("최저 자산:", round(eq["Equity"].min()))
    print("최고 자산:", round(eq["Equity"].max()))

    eq["Peak"] = eq["Equity"].cummax()
    eq["Drawdown_Pct"] = (eq["Equity"] / eq["Peak"] - 1) * 100
    print("일별 자산 기준 MDD:",
          round(eq["Drawdown_Pct"].min(), 2), "%")

print("\n검증 출력 완료")
