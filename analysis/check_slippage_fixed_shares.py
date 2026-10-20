from pathlib import Path
import pandas as pd
import numpy as np

BASE = Path(__file__).resolve().parent
df = pd.read_csv(BASE / "ma40_fixed_levels_volume_dd_0.05_trades.csv")

COMMISSION = 0.00015
SELL_TAX = 0.0015
OLD_SLIP = 0.0005
NEW_SLIP = 0.001

raw_entry = df["Entry_Price"] / (1 + OLD_SLIP)
raw_exit = df["Raw_Exit_Price"]
shares = df["Shares"]

def calculate_pnl(slippage):
    entry_price = raw_entry * (1 + slippage)
    exit_price = raw_exit * (1 - slippage)

    entry_cost = shares * entry_price * (1 + COMMISSION)
    net_proceeds = shares * exit_price * (1 - COMMISSION - SELL_TAX)

    return net_proceeds - entry_cost

old_pnl = calculate_pnl(OLD_SLIP)
new_pnl = calculate_pnl(NEW_SLIP)

error = np.abs(old_pnl - df["PnL"])

print("\n===== 동일 수량 슬리피지 통제 실험 =====")
print(f"거래 수: {len(df)}")
print(f"0.05% 총손익: {old_pnl.sum():,.0f}원")
print(f"0.10% 총손익: {new_pnl.sum():,.0f}원")
print(f"총손익 차이: {(new_pnl.sum() - old_pnl.sum()):,.0f}원")

print("\n===== 원본 PnL 일치 여부 =====")
print(f"최대 오차: {error.max():.6f}원")
print(f"평균 오차: {error.mean():.6f}원")

print("\n===== 거래별 비교 =====")
print(f"손익 감소: {(new_pnl < old_pnl).sum()}건")
print(f"손익 증가: {(new_pnl > old_pnl).sum()}건")
print(f"모든 거래에서 손익 감소: {bool((new_pnl < old_pnl).all())}")
