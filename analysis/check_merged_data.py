
import pandas as pd

df = pd.read_csv(
    "training_data_v4_lending.csv",
    dtype={"Ticker": str}
)

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

print("===== 전체 데이터 =====")
print("전체 행:", len(df))
print("대차잔고 변화율 보유:", df["Lending_Balance_Change_Pct"].notna().sum())
print("대차잔고 변화율 결측:", df["Lending_Balance_Change_Pct"].isna().sum())

print("\n===== 종목별 연결 결과 =====")
linked = df[df["Lending_Balance_Change_Pct"].notna()]

print(
    linked.groupby(["Stock_Name", "Ticker"])
    .agg(
        연결_행수=("Date", "size"),
        시작일=("Date", "min"),
        종료일=("Date", "max"),
    )
    .to_string()
)

print("\n===== 대차잔고 데이터가 없는 종목 =====")
linked_tickers = set(linked["Ticker"].dropna())
all_tickers = set(df["Ticker"].dropna())

print("전체 종목 수:", len(all_tickers))
print("대차잔고 연결 종목 수:", len(linked_tickers))
print("대차잔고 연결이 없는 종목 수:", len(all_tickers - linked_tickers))
