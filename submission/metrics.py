"""
metrics.py
--------------------
This module takes the 2 csv's: one containing the monthly returns from the main strategy and
one containing the monthly returns from the pairs trading strategy, combines them into one overall strategy
by doing a weighted average of returns based on how many stocks were traded each month in each strategy
and finally calculates various Porfolio Performance Statistics for the overall strategy.
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as sm
from pandas.tseries.offsets import *

PRED_PATH = "final_results.csv"
pred = pd.read_csv(PRED_PATH, parse_dates=["date"])
pred = pred[pred["returns"] != 0]


PAIRS_COUNT_PATH = "pairs-trade-counts-per-month.csv"
pairs_trades_count = pd.read_csv(PAIRS_COUNT_PATH)
pairs_trades_count["date"] = pd.to_datetime(
    pairs_trades_count["year_month"], format="%Y-%m"
)

# Extract year and month
pairs_trades_count["year"] = pairs_trades_count["date"].dt.year
pairs_trades_count["month"] = pairs_trades_count["date"].dt.month

PAIRS_PATH = "pair-trade-per-month.csv"
pairs_trades = pd.read_csv(PAIRS_PATH)

pairs_trades["date"] = pd.to_datetime(pairs_trades["exit_date"], format="%Y%m%d")

pairs_trades["year"] = pairs_trades["date"].dt.year
pairs_trades["month"] = pairs_trades["date"].dt.month

pairs_trades["returns_pairs"] = pairs_trades["avg_pnl_dollars"] - 1

pairs_trades.drop(
    ["Unnamed: 0", "exit_date", "log", "sum", "bench", "avg_pnl_dollars"],
    axis=1,
    inplace=True,
)

monthly_pairs = pairs_trades.groupby(["year", "month"], as_index=False).agg(
    {"returns_pairs": "sum"}
)

full_months = pd.date_range(start="2015-01-01", end="2025-07", freq="M")
full_df = pd.DataFrame({"date": full_months})
full_df["year"] = full_df["date"].dt.year
full_df["month"] = full_df["date"].dt.month

monthly_pairs = full_df.merge(monthly_pairs, on=["year", "month"], how="left")

monthly_pairs["returns_pairs"] = monthly_pairs["returns_pairs"].fillna(0)

monthly_pairs = monthly_pairs[["date", "year", "month", "returns_pairs"]]


SNP_RETURNS_PATH = "mkt_ind.csv"
snp_returns = pd.read_csv(SNP_RETURNS_PATH)
snp_returns = snp_returns[snp_returns["year"] >= 2015]

monthly_portfolio = pred.copy()
monthly_portfolio["date"] = pd.to_datetime(pred["date"])

# Extract year and month
monthly_portfolio["year"] = monthly_portfolio["date"].dt.year.copy()
monthly_portfolio["month"] = monthly_portfolio["date"].dt.month.copy()

monthly_portfolio.drop("Unnamed: 0", axis=1, inplace=True)
monthly_portfolio = monthly_portfolio.merge(
    pairs_trades_count, how="inner", on=["year", "month"]
)
monthly_portfolio["returns"] = (
    monthly_portfolio["returns"] * 200
    + monthly_pairs["returns_pairs"] * pairs_trades_count["count"] * 2
) / (200 + pairs_trades_count["count"] * 2)


# SHARPE RATIO
sharpe = (
    monthly_portfolio["returns"].mean()
    / monthly_portfolio["returns"].std()
    * np.sqrt(12)
)  # Sharpe ratio is annualized
print("Sharpe Ratio:", sharpe)
ann_std = monthly_portfolio["returns"].std() * np.sqrt(12)
print("Annualized Standard Deviation:", ann_std)


monthly_portfolio_merged = monthly_portfolio.merge(
    snp_returns, how="inner", on=["year", "month"]
)

# ANNUALIZED ALPHA
monthly_portfolio_merged = monthly_portfolio.merge(
    snp_returns, how="inner", on=["year", "month"]
)
# Newy-West regression for heteroskedasticity and autocorrelation robust standard errors
nw_ols = sm.ols(formula="returns ~ rf", data=monthly_portfolio_merged).fit(
    cov_type="HAC", cov_kwds={"maxlags": 3}, use_t=True
)
print(nw_ols.summary())

# Specifically, the alpha, t-statistic, and Information ratio are:
print("CAPM Alpha:", nw_ols.params["Intercept"] * 12)
print("t-statistic:", nw_ols.tvalues["Intercept"])

# INFORMATION RATIO
print(
    "Information Ratio:",
    nw_ols.params["Intercept"] / np.sqrt(nw_ols.mse_resid) * np.sqrt(12),
)  # Information ratio is annualized

# MAX ONE-MONTH LOSS
max_1m_loss = monthly_portfolio["returns"].min()
print("Max 1-Month Loss:", max_1m_loss)


# MAX DRAWDOWN
# Calculate Drawdown of the long-short Portfolio
# you can use the same formula to calculate the Sharpe ratio for the long and short portfolios separately
monthly_portfolio["log_returns"] = np.log(
    monthly_portfolio["returns"] + 1
)  # calculate log returns
monthly_portfolio["cumsum_log_returns"] = monthly_portfolio["log_returns"].cumsum(
    axis=0
)  # calculate cumulative log returns
rolling_peak = monthly_portfolio["cumsum_log_returns"].cummax()
drawdowns = rolling_peak - monthly_portfolio["cumsum_log_returns"]
max_drawdown = drawdowns.max()
print("Maximum Drawdown:", max_drawdown)

from collections import Counter, defaultdict

# Load all results from predictions (for holdings info)
ALL_RESULTS_PATH = "actual_pred_per_stock.csv"  # <-- make sure this file exists
all_results = pd.read_csv(ALL_RESULTS_PATH, parse_dates=["date"])

top_holdings_counter = Counter()
profit_contributions = defaultdict(float)
turnover_list = []
turnover_by_date = {}
prev_holdings = set()

# Iterate through each month in the dataset
for year in sorted(all_results["date"].dt.year.unique()):
    for month in range(1, 13):
        filtered = all_results[
            (all_results["date"].dt.year == year)
            & (all_results["date"].dt.month == month)
        ]
        if filtered.empty:
            continue

        # Sort by predicted return to identify top (long) positions
        filtered = filtered.sort_values(by="predicted_values")

        # Top 100 = long positions
        most_positive = filtered.tail(100)
        current_holdings = set(most_positive["gvkey"])

        # Update appearance counter
        top_holdings_counter.update(most_positive["gvkey"])

        # Record realized profit contribution for each gvkey
        for _, row in most_positive.iterrows():
            profit_contributions[row["gvkey"]] += row["actual_values"]

        # Calculate monthly portfolio turnover (fraction of holdings replaced)
        if prev_holdings:
            overlap = len(current_holdings & prev_holdings)
            turnover = 1 - (overlap / len(current_holdings))
            turnover_list.append(turnover)
            date = pd.Timestamp(year=year, month=month, day=1)
            turnover_by_date[date] = turnover

        prev_holdings = current_holdings

# --- Results Summary ---
print("\n--- Portfolio Holdings and Turnover Analysis ---")

# Top 10 most frequently held stocks
top_10_holdings = top_holdings_counter.most_common(10)
print("\nTop 10 Holdings (by frequency of appearance):")
for i, (gvkey, count) in enumerate(top_10_holdings, start=1):
    print(f"{i}. gvkey: {gvkey}, months held: {count}")

# Top 10 most profitable stocks
sorted_profit = sorted(profit_contributions.items(), key=lambda x: x[1], reverse=True)
top_10_profitable = sorted_profit[:10]
print("\nTop 10 Most Profitable Stocks (by cumulative actual return):")
for i, (gvkey, total_profit) in enumerate(top_10_profitable, start=1):
    print(f"{i}. gvkey: {gvkey}, cumulative contribution: {total_profit:.4f}")

# Average monthly turnover
avg_turnover = sum(turnover_list) / len(turnover_list)
print(f"\nAverage Monthly Portfolio Turnover: {avg_turnover:.2%}")

"""
This section analyzes which fundamental features contributed most to the
portfolio's performance by examining the gain-based feature importances
from all yearly XGBoost models.
"""

import joblib
import os
import xgboost as xgb

feature_importances = []

# Loop through all saved yearly models
for year in range(2014, 2025):
    # Put correct path for model
    model_path = f"data/mod-final-strat-model-{year}.joblib"
    if not os.path.exists(model_path):
        continue

    # Load XGBoost model
    model = joblib.load(model_path)

    # Extract feature importance (gain = contribution to model accuracy)
    importance_dict = model.get_booster().get_score(importance_type="gain")
    importance_df = pd.DataFrame(
        list(importance_dict.items()), columns=["feature", "importance"]
    )
    importance_df["year"] = year
    feature_importances.append(importance_df)

# Combine all years into a single DataFrame
importance_all = pd.concat(feature_importances, ignore_index=True)

# Calculate average importance across all years and sort descending
avg_importance = (
    importance_all.groupby("feature")["importance"].mean().sort_values(ascending=False)
)

# Display top 10 features
top_features = avg_importance.head(10)
print("\n--- Main Fundamental Signals Contributing to Portfolio Performance ---")
for i, (feature, score) in enumerate(top_features.items(), start=1):
    print(f"{i}. {feature}: {score:.4f}")
