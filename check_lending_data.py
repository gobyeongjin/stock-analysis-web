import pandas as pd

df = pd.read_csv("lending_multi.csv", dtype={"Ticker": str})
df["dt"] = pd.to_datetime(df["dt"], errors="coerce")
df["rmnd"] = pd.to_numeric(df["rmnd"], errors="coerce")

# 종목별 최신 데이터와 잔고 0인 행 확인
print("===== 잔고가 0인 행 =====")
print(
    df.loc[df["rmnd"].eq(0),
           ["Stock_Name", "Ticker", "dt", "rmnd"]]
    .sort_values(["Ticker", "dt"])
    .to_string(index=False)
)

print("\n===== 종목별 최신 날짜 =====")
print(
    df.sort_values("dt")
      .groupby(["Stock_Name", "Ticker"])
      .tail(1)[["Stock_Name", "Ticker", "dt", "rmnd"]]
      .sort_values("Ticker")
      .to_string(index=False)
)

print("\n===== 종목별 기간 =====")
print(
    df.groupby(["Stock_Name", "Ticker"])["dt"]
      .agg(["min", "max", "count"])
      .to_string()
)