import K_Means_Clustering as km
import os
import pandas as pd
import trade_pairs as tp
import numpy as np
import statsmodels.api as sm

# save all cointegrated stocks per year to csv's
year = 2015
for i in range(11):
    km.get_cointegrated_stocks_by_year(year+i)

# execute all the pairs trades

df_combined = pd.DataFrame()
year = 2015
df_returns = km.get_stock_returns_upto_year(2026)
for i in range(11):
    cointegrated_pairs_path = f"data/cointegrated-pairs-{year+i}.csv"
    if not os.path.exists(cointegrated_pairs_path):
        df_cointegrated_pairs = km.get_cointegrated_stocks_by_year(year)
    else:
        df_cointegrated_pairs = pd.read_csv(cointegrated_pairs_path)
    
    for j in range(len(df_cointegrated_pairs)):
        stock1_label = df_cointegrated_pairs.iloc[j, 0]
        stock2_label = df_cointegrated_pairs.iloc[j, 1]
        series1 = df_returns[df_returns["id"] == stock1_label].copy()
        series2 = df_returns[df_returns["id"] == stock2_label].copy()
        series1["price_index"] = (1 + series1["stock_ret"]).cumprod() * 1
        series2["price_index"] = (1 + series2["stock_ret"]).cumprod() * 1
    
        len(series1["stock_ret"].dropna())
        # Find the first index where date >= 20150000
        first_idx = series1[series1["date"] >= (year+i) * 10000].index[0]
        
        # Get position of that index
        pos = series1.index.get_loc(first_idx)
        
        # Get 6 rows before that position, plus alldone rows from that position onward
        series1 = series1.iloc[max(0, pos-6):]
        
        series1 = series1[series1["date"] < (year+i+1)*10000]

        
        first_idx = series2[series2["date"] >= (year+i)*10000].index[0]
        
        # Get position of that index
        pos = series2.index.get_loc(first_idx)
        
        # Get 6 rows before that position, plus all rows from that position onward
        series2 = series2.iloc[max(0, pos-6):]
        series2 = series2[series2["date"] < (year+i+1)*10000]
        try:
            temp, temp2 = tp.pairs_trade_monthly_with_risk(series1.set_index('date')["price_index"], series2.set_index('date')["price_index"],
                                                stock1_label, stock2_label)
            df_combined = pd.concat([df_combined, temp2], ignore_index=True)
        except:
            print("skip pair")
    print(year+i, "done")

# remove the months that we did not end up executing
df_combined = df_combined[df_combined["holding_months"] > 0]

# Extract year from entry_date (integer -> string -> slice)
df_combined["year"] = df_combined["entry_date"].astype(str).str[:4].astype(int)

# Count how many rows per year
count_per_year = df_combined.groupby("year").size().reset_index(name="count")

# Convert to datetime
df_combined["exit_date"] = pd.to_datetime(df_combined["exit_date"], format="%Y%m%d")

# Extract year-month period
df_combined["year_month"] = df_combined["exit_date"].dt.to_period("M")

# Count trades per month
count_per_month = df_combined.groupby("year_month").size().reset_index(name="count")

# Create full monthly range
full_range = pd.period_range(start=pd.to_datetime(20150130, format="%Y%m%d"), end=df_combined["exit_date"].max(), freq="M")
full_df = pd.DataFrame({"year_month": full_range})

# Merge and fill missing with 0
count_per_month_full = full_df.merge(count_per_month, on="year_month", how="left").fillna(0)
count_per_month_full["count"] = count_per_month_full["count"].astype(int)
count_per_month_full.to_csv("pairs-trade-counts-per-month.csv")

# create new df for mean of every month
new_df = (
    df_combined.groupby("exit_date")["pnl_dollars"]
      .mean()  # average PnL across rows with same exit_date
      .reset_index(name="avg_pnl_dollars")
)

# returns and cumulative log returns every year
new_df["avg_pnl_dollars"] = new_df["avg_pnl_dollars"] / 100 + 1
new_df["log"] = np.log(new_df["avg_pnl_dollars"])
new_df["sum"] = new_df["log"].cumsum()
new_df["bench"] = np.exp(new_df["sum"]) - 1
pd.set_option('display.max_rows', None)

# save it to a csv
new_df.to_csv("pair-trade-per-month.csv")