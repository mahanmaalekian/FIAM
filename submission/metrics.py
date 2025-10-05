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
