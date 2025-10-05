"""
xgboost_returns_predictor.py
--------------------

This module trains and applies an XGBoost regression model to predict future
monthly stock returns based on historical financial features. It uses a
rolling-window approach, retraining each year to simulate a realistic
out-of-sample prediction process.

The main entry point is:
    xgboost_predict_returns()
"""

import pandas as pd
import xgboost as xgb


def xgboost_predict_returns():
    """
    Train and apply an XGBoost model to predict stock returns using a rolling window.

    This function loads company-level financial data from 'stocks.parquet', selects
    relevant numerical features, and trains an XGBoost regression model to predict
    future monthly stock returns. It uses a **rolling time-window approach**:
    - The model is trained on all data up to a given year,
    - validated on the following two years,
    - and tested on the next one-year period.
    The window then advances by one year and repeats until the end date is reached.

    Each test period's predictions are stored along with the true returns, then
    concatenated and written to 'data/xgboost_results.csv'.

    Model details
    --------------
    - Objective: squared error regression (`reg:squarederror`)
    - Evaluation metric: RMSE
    - Depth: 12
    - Learning rate: 0.05
    - Subsample and feature sample: 0.8
    - GPU acceleration: enabled (`tree_method='gpu_hist'`)

    Returns
    -------
    None
        Writes the full prediction DataFrame to 'data/xgboost_results.csv'.
    """
    df = pd.read_parquet("stocks.parquet")

    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")

    features = df.columns.difference(
        [
            "id",
            "date",
            "ret_eom",
            "gvkey",
            "iid",
            "excntry",
            "year",
            "month",
            "char_date",
            "char_eom",
            "stock_ret",
        ]
    ).tolist()

    # Initial training and validation periods
    initial_train_end = pd.to_datetime("2012-12-31")
    validation_end = pd.to_datetime("2014-12-31")
    end_date = pd.to_datetime("2025-05-31")

    # Collect predictions
    predictions = []

    while validation_end < end_date:
        # Masks
        train_mask = df["date"] <= initial_train_end
        valid_mask = (df["date"] > initial_train_end) & (df["date"] <= validation_end)
        test_mask = (df["date"] > validation_end) & (
            df["date"] <= validation_end + pd.DateOffset(years=1)
        )

        train_X, train_y = df.loc[train_mask, features], df.loc[train_mask, "stock_ret"]
        valid_X, valid_y = df.loc[valid_mask, features], df.loc[valid_mask, "stock_ret"]
        test_X = df.loc[test_mask, features]

        dtrain = xgb.DMatrix(train_X, label=train_y)
        dvalid = xgb.DMatrix(valid_X, label=valid_y)
        dtest = xgb.DMatrix(test_X)

        params = {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "max_depth": 12,
            "eta": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "seed": 42,
            "tree_method": "gpu_hist",  # enables GPU acceleration
            "predictor": "gpu_predictor",  # ensures GPU is used for prediction too
        }

        model = xgb.train(
            params,
            dtrain,
            num_boost_round=1000,
            evals=[(dtrain, "train"), (dvalid, "valid")],
            early_stopping_rounds=50,
            verbose_eval=False,
        )

        # Predictions for test window
        preds = model.predict(dtest)

        # Save with stock_id + date
        tmp = df.loc[test_mask, ["gvkey", "date", "stock_ret"]].copy()
        tmp["predicted_return"] = preds
        predictions.append(tmp)

        # Expand window by 1 year
        initial_train_end += pd.DateOffset(years=1)
        validation_end += pd.DateOffset(years=1)

    pred_df = pd.concat(predictions, ignore_index=True)

    # Save to csv
    pred_df.to_csv("data/xgboost_results.csv")
