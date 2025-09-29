import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint

def pairs_trade_monthly_with_risk(prices1, prices2, label1, label2,
                                  entry=2.0, exit=0.5, lookback=6,
                                  capital=100,
                                  max_holding=6,            # months
                                  stop_loss_pct=0.04,       # exit if loss > 10% of capital
                                  coint_pval_threshold=0.10 # if pval > this, abandon pair
                                 ):
    """
    Monthly pairs backtest with rules for non-converging trades.
    Returns signals DataFrame and trades DataFrame (with close_reason).
    """

    df = pd.DataFrame({'p1': prices1, 'p2': prices2}).dropna().copy()
    n = len(df)
    if n < lookback + 1:
        raise ValueError("Not enough data for lookback")

    trades = []
    position = 0              # -1 short spread, +1 long spread, 0 flat
    entry_spread = None
    entry_date = None
    entry_sigma = None
    entry_index = None
    holding = 0

    # Pre-allocate columns
    df['z'] = np.nan
    df['action'] = ''
    df['position'] = 0
    df['pnl_unrealized'] = 0.0

    for t in range(lookback, n):
        # Use history up to t-1 (no lookahead)
        hist = df.iloc[:t]
        y = hist['p1']
        x = hist['p2']
        X = sm.add_constant(x)
        model = sm.OLS(y, X).fit()
        beta = model.params[1]

        spread_hist = y - beta * x
        mu = spread_hist.mean()
        sigma = spread_hist.std(ddof=0) if spread_hist.std(ddof=0) > 0 else 1e-8

        spread_t = df['p1'].iat[t] - beta * df['p2'].iat[t]
        z_t = (spread_t - mu) / sigma
        date_t = df.index[t]

        reason = None
        pnl_dollars = 0.0

        # If we currently have no position -> maybe enter
        if position == 0:
            if z_t > entry:
                position = -1
                entry_spread = spread_t
                entry_sigma = sigma
                entry_date = date_t
                entry_index = t
                holding = 0
                df.at[date_t, 'action'] = 'ENTER_SHORT'
            elif z_t < -entry:
                position = +1
                entry_spread = spread_t
                entry_sigma = sigma
                entry_date = date_t
                entry_index = t
                holding = 0
                df.at[date_t, 'action'] = 'ENTER_LONG'
            else:
                df.at[date_t, 'action'] = 'HOLD'
        else:
            # We are in a trade: compute mark-to-market PnL for monitoring
            # pnl in spread units:
            pnl_spread_units = (spread_t - entry_spread) * position
            # convert to dollars by scaling with capital and entry_sigma
            pnl_dollars = (pnl_spread_units / entry_sigma) * capital
            df.at[date_t, 'pnl_unrealized'] = pnl_dollars

            holding += 1

            # exit conditions (in priority order)
            # 1) normal exit (z moved back inside exit band)
            if abs(z_t) < exit:
                reason = 'z_cross'
            # 2) stop-loss: loss exceeds stop_loss_pct * capital
            elif pnl_dollars < -stop_loss_pct * capital:
                reason = 'stop_loss'
            # 3) max holding exceeded
            elif holding >= max_holding:
                reason = 'max_holding'
            else:
                reason = None

            if reason is not None:
                exit_spread = spread_t
                exit_date = date_t
                # compute final pnl_dollars (already computed above)
                trades.append({
                    'entry_date': entry_date,
                    'exit_date': exit_date,
                    'direction': 'LONG' if position == 1 else 'SHORT',
                    'entry_spread': entry_spread,
                    'exit_spread': exit_spread,
                    'pnl_dollars': pnl_dollars,
                    'holding_months': holding,
                    'close_reason': reason
                })
                # reset position
                position = 0
                entry_spread = None
                entry_date = None
                entry_sigma = None
                entry_index = None
                holding = 0
                df.at[date_t, 'action'] = f'EXIT_{reason.upper()}'
            else:
                df.at[date_t, 'action'] = 'HOLD'

        df.at[date_t, 'z'] = z_t
        df.at[date_t, 'position'] = position

    # End of sample: force-close any open trade at last price
    if position != 0 and entry_spread is not None:
        # use last available t (n-1)
        last_t = n - 1
        y = df['p1'].iloc[:last_t]
        x = df['p2'].iloc[:last_t]
        X = sm.add_constant(x)
        model = sm.OLS(y, X).fit()
        beta = model.params[1]
        spread_last = df['p1'].iat[last_t] - beta * df['p2'].iat[last_t]
        sigma_last = (y - beta * x).std(ddof=0) if (y - beta * x).std(ddof=0) > 0 else 1e-8
        pnl_spread_units = (spread_last - entry_spread) * position
        pnl_dollars = (pnl_spread_units / entry_sigma) * capital if entry_sigma is not None else (pnl_spread_units / sigma_last) * capital

        trades.append({
            'entry_date': entry_date,
            'exit_date': df.index[last_t],
            'direction': 'LONG' if position == 1 else 'SHORT',
            'entry_spread': entry_spread,
            'exit_spread': spread_last,
            'pnl_dollars': pnl_dollars,
            'holding_months': holding,
            'close_reason': 'forced_liquidation_end_of_sample'
        })

        # update final row
        df.at[df.index[last_t], 'pnl_unrealized'] = pnl_dollars
        df.at[df.index[last_t], 'action'] = 'FORCED_LIQUIDATION'
        df.at[df.index[last_t], 'position'] = 0

    trades_df = pd.DataFrame(trades)
    return df, trades_df
    