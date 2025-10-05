
"""
pairs_trading.py
--------------------

This module provides tools to perform monthly pairs trading backtests
with risk management on cointegrated stock pairs.

Functions:
- pairs_trade_monthly_with_risk(prices1, prices2, ...):
    Executes a monthly pairs trading strategy on two stock price series,
    managing positions with entry/exit z-scores, stop-loss, and maximum holding periods.
    Returns both the detailed signals DataFrame and the trades DataFrame.

- execute_all_pairs_trades():
    Runs the pairs trading backtest for all cointegrated pairs from 2015 onward.
    Aggregates results, counts trades per month, calculates average PnL per exit date,
    and saves summary CSV files for further analysis.

Usage:
- Import the module and call `execute_all_pairs_trades()` to run the full backtest.
- Use `pairs_trade_monthly_with_risk()` for testing or analyzing a specific stock pair.
"""
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm

from k_means_clusters import (get_cointegrated_stocks_by_year,
                              get_stock_returns_upto_year)
from pairs_trading import execute_all_pairs_trades


def pairs_trade_monthly_with_risk(
    prices1,
    prices2,
    entry=2.0,
    exit=0.5,
    lookback=6,
    capital=100,
    max_holding=6,
    stop_loss_pct=0.04,
):
    """
    Monthly pairs trading backtest with risk management.

    Parameters:
    - prices1, prices2: pd.Series of stock prices
    - entry: z-score threshold to enter a trade
    - exit: z-score threshold to exit a trade
    - lookback: months to compute spread statistics
    - capital: capital allocated per trade
    - max_holding: max holding period (months)
    - stop_loss_pct: max loss before exiting trade

    Returns:
    - df: DataFrame with z-scores, positions, actions, and unrealized PnL
    - trades_df: DataFrame of completed trades with entry/exit info and close reasons
    """


    df = pd.DataFrame({"p1": prices1, "p2": prices2}).dropna().copy()
    n = len(df)
    if n < lookback + 1:
        raise ValueError("Not enough data for lookback")

    trades = []
    position = 0
    entry_spread = None
    entry_date = None
    entry_index = None
    entry_shares1 = None
    entry_shares2 = None
    holding = 0

    # Pre-allocate columns
    df["z"] = np.nan
    df["action"] = ""
    df["position"] = 0
    df["pnl_unrealized"] = 0.0

    for t in range(lookback, n):
        hist = df.iloc[:t]
        y = hist["p1"]
        x = hist["p2"]
        X = sm.add_constant(x)
        model = sm.OLS(y, X).fit()
        beta = model.params[1]

        spread_hist = y - beta * x
        mu = spread_hist.mean()
        sigma = spread_hist.std(ddof=0) if spread_hist.std(ddof=0) > 0 else 1e-8

        spread_t = df["p1"].iat[t] - beta * df["p2"].iat[t]
        z_t = (spread_t - mu) / sigma
        date_t = df.index[t]

        reason = None
        pnl_dollars = 0.0

        if position == 0:
            if z_t > entry:
                position = -1
                entry_spread = spread_t
                entry_sigma = sigma
                entry_date = date_t
                entry_index = t
                holding = 0
                entry_shares1 = -capital / df["p1"].iat[t]  # STORE
                entry_shares2 = +beta * capital / df["p2"].iat[t]  # STORE
                df.at[date_t, "action"] = "ENTER_SHORT"
            elif z_t < -entry:
                position = +1
                entry_spread = spread_t
                entry_sigma = sigma
                entry_date = date_t
                entry_index = t
                holding = 0
                entry_shares1 = +capital / df["p1"].iat[t]  # STORE
                entry_shares2 = -beta * capital / df["p2"].iat[t]  # STORE
                df.at[date_t, "action"] = "ENTER_LONG"
            else:
                df.at[date_t, "action"] = "HOLD"
        else:
            # Use stored shares
            pnl_dollars = entry_shares1 * (
                df["p1"].iat[t] - df["p1"].iat[entry_index]
            ) + entry_shares2 * (df["p2"].iat[t] - df["p2"].iat[entry_index])

            df.at[date_t, "pnl_unrealized"] = pnl_dollars

            holding += 1

            if abs(z_t) < exit:
                reason = "z_cross"
            elif pnl_dollars < -stop_loss_pct * capital:
                reason = "stop_loss"
            elif holding >= max_holding:
                reason = "max_holding"
            else:
                reason = None

            if reason is not None:
                exit_spread = spread_t
                exit_date = date_t
                trades.append(
                    {
                        "entry_date": entry_date,
                        "exit_date": exit_date,
                        "direction": "LONG" if position == 1 else "SHORT",
                        "entry_spread": entry_spread,
                        "exit_spread": exit_spread,
                        "pnl_dollars": pnl_dollars,
                        "holding_months": holding,
                        "close_reason": reason,
                    }
                )
                position = 0
                entry_spread = None
                entry_date = None
                entry_sigma = None
                entry_index = None
                entry_shares1 = None  # RESET
                entry_shares2 = None  # RESET
                holding = 0
                df.at[date_t, "action"] = f"EXIT_{reason.upper()}"
            else:
                df.at[date_t, "action"] = "HOLD"

        df.at[date_t, "z"] = z_t
        df.at[date_t, "position"] = position

    # Forced liquidation: USE SHARES METHOD
    if position != 0 and entry_index is not None:
        last_t = n - 1

        # Calculate PnL using shares (consistent with main loop)
        pnl_dollars = entry_shares1 * (
            df["p1"].iat[last_t] - df["p1"].iat[entry_index]
        ) + entry_shares2 * (df["p2"].iat[last_t] - df["p2"].iat[entry_index])

        # Recalculate spread for recording purposes
        y = df["p1"].iloc[: last_t + 1]
        x = df["p2"].iloc[: last_t + 1]
        X = sm.add_constant(x)
        model = sm.OLS(y, X).fit()
        beta = model.params[1]
        spread_last = df["p1"].iat[last_t] - beta * df["p2"].iat[last_t]

        trades.append(
            {
                "entry_date": entry_date,
                "exit_date": df.index[last_t],
                "direction": "LONG" if position == 1 else "SHORT",
                "entry_spread": entry_spread,
                "exit_spread": spread_last,
                "pnl_dollars": pnl_dollars,
                "holding_months": holding,
                "close_reason": "forced_liquidation_end_of_sample",
            }
        )

        df.at[df.index[last_t], "pnl_unrealized"] = pnl_dollars
        df.at[df.index[last_t], "action"] = "FORCED_LIQUIDATION"
        df.at[df.index[last_t], "position"] = 0

    trades_df = pd.DataFrame(trades)
    return df, trades_df


def execute_all_pairs_trades():
    """
    Run monthly pairs trading backtests for all cointegrated stock pairs from 2015 onward.

    Steps:
    1. Load or compute cointegrated stock pairs for each year.
    2. Compute price indices from stock returns.
    3. Execute pairs trading using `pairs_trade_monthly_with_risk`.
    4. Aggregate trade results across all pairs and years.
    5. Count trades per month and average PnL per exit date.
    6. Save results to CSV files.

    Outputs:
    - 'pairs-trade-counts-per-month.csv': number of trades closed per month
    - 'pair-trade-per-month.csv': average PnL per exit date
    """
    year = 2015
    for i in range(11):
        get_cointegrated_stocks_by_year(year + i)
    df_combined = pd.DataFrame()
    year = 2015
    df_returns = get_stock_returns_upto_year(2026)
    for i in range(11):
        cointegrated_pairs_path = f"data/cointegrated-pairs-{year + i}.csv"
        if not os.path.exists(cointegrated_pairs_path):
            df_cointegrated_pairs = get_cointegrated_stocks_by_year(year)
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
            first_idx = series1[series1["date"] >= (year + i) * 10000].index[0]

            # Get position of that index
            pos = series1.index.get_loc(first_idx)

            # Get 6 rows before that position, plus alldone rows from that position onward
            series1 = series1.iloc[max(0, pos - 6) :]

            series1 = series1[series1["date"] < (year + i + 1) * 10000]

            first_idx = series2[series2["date"] >= (year + i) * 10000].index[0]

            # Get position of that index
            pos = series2.index.get_loc(first_idx)

            # Get 6 rows before that position, plus all rows from that position onward
            series2 = series2.iloc[max(0, pos - 6) :]
            series2 = series2[series2["date"] < (year + i + 1) * 10000]
            try:
                temp, temp2 = tp.pairs_trade_monthly_with_risk(
                    series1.set_index("date")["price_index"],
                    series2.set_index("date")["price_index"],
                    stock1_label,
                    stock2_label,
                )
                df_combined = pd.concat([df_combined, temp2], ignore_index=True)
            except:
                print("skip pair")
        print(year + i, "done")
        df_combined = df_combined[df_combined["holding_months"] > 0]

        # Convert to datetime
        df_combined["exit_date"] = pd.to_datetime(
            df_combined["exit_date"], format="%Y%m%d"
        )

        # Extract year-month period
        df_combined["year_month"] = df_combined["exit_date"].dt.to_period("M")

        # Count trades per month
        count_per_month = (
            df_combined.groupby("year_month").size().reset_index(name="count")
        )

        # Create full monthly range
        full_range = pd.period_range(
            start=pd.to_datetime(20150130, format="%Y%m%d"),
            end=df_combined["exit_date"].max(),
            freq="M",
        )
        full_df = pd.DataFrame({"year_month": full_range})

        # Merge and fill missing with 0
        count_per_month_full = full_df.merge(
            count_per_month, on="year_month", how="left"
        ).fillna(0)
        count_per_month_full["count"] = count_per_month_full["count"].astype(int)
        count_per_month_full.to_csv("pairs-trade-counts-per-month.csv")

    new_df = (
        df_combined.groupby("exit_date")["pnl_dollars"]
        .mean()  # average PnL across rows with same exit_date
        .reset_index(name="avg_pnl_dollars")
    )
    new_df.to_csv("pair-trade-per-month.csv")
    execute_all_pairs_trades()
