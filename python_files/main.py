"""
main.py
--------------------
This module implements a workflow for predicting stock returns using
XGBoost models enhanced with HMM state features and text sentiment data.
It also runs the pairs trading algorithm to execute the pairs trades
during the OOS period.

Functions:
- main(): Executes the full workflow of:
    - Loading and preprocessing stock data
    - Runs the 3 other models (FinBert, HMM and XGBOOST) who's predictions are used as inputs to the final XGBOOST
    - Training XGBoost models for the final portfolio per year with hyperparameter optimization
    - Predicting next-year returns and computing monthly long-short portfolio returns
    - Saving results and trained models to disk

Data Sources:
- Stock returns and fundamental data
- Sentiment data CSV

Outputs:
- Trained XGBoost models per year: 'data/final-strat-model-{year}.joblib'
- Monthly long-short returns: 'final_results.csv'
"""

import glob

import joblib
import pandas as pd
import xgboost as xgb
from scipy.stats import randint, uniform
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, train_test_split

from data_loader import load_data_to_parquet
from hmm_state_trainer import create_hmm_states_per_year
from pairs_trading import execute_all_pairs_trades
from run_finbert_sentiment import run_finbert_sentiment_analysis
from xgboost_returns_predictor import xgboost_predict_returns

XGBOOST_RESULTS_PATH = "./data/xgboost_result.csv"
TEXT_SENTIMENT_PATH = "./data/text_data_sentiment.csv"


def main():
    # load the stock data and save it to a parquet file
    load_data_to_parquet()

    # create the csv's with the hmm states for each year
    create_hmm_states_per_year()

    # create the csv with all the xgboost intial predicted returns
    xgboost_predict_returns()

    # create the csv with the sentiment data for every stock per year
    run_finbert_sentiment_analysis()

    df_xgboost_results = pd.read_csv(XGBOOST_RESULTS_PATH)

    df_text_sentiment = pd.read_csv(TEXT_SENTIMENT_PATH)

    # this dataframe will contain the monthly returns of our portfolio during the OOS period
    df_returns = pd.DataFrame(columns=["date", "returns"])

    # Train the final model for each year in the OOS and use it to select the stocks for every month of that year
    for idx, year in enumerate(range(2014, 2025), start=1):
        # HMM
        # load the the predicted states data for the year we want to select the stocks
        df_hmm = pd.read_csv(f"./data/hmm-{year + 1}.csv")

        # training data
        df_hmm_curr = df_hmm[df_hmm["date"] >= year * 10000]
        # prediction data
        df_hmm_next = df_hmm[df_hmm["date"] >= (year + 1) * 10000]

        # XGBOOST
        # training data
        df_xgboost_results_curr = df_xgboost_results[
            (df_xgboost_results["date"] >= f"{year}-01-01")
            & (df_xgboost_results["date"] <= f"{year}-12-31")
        ]

        df_xgboost_results_new = df_xgboost_results_curr.copy()

        # turn the dates into datetime
        df_xgboost_results_new["date"] = pd.to_datetime(df_xgboost_results_new["date"])
        df_xgboost_results_new["date"] = (
            df_xgboost_results_new["date"].dt.strftime("%Y%m%d").astype(int)
        )

        # prediction data
        df_xgboost_results_next = df_xgboost_results[
            (df_xgboost_results["date"] >= f"{year + 1}-01-01")
            & (df_xgboost_results["date"] <= f"{year + 1}-12-31")
        ]

        # turn the dates into datetime
        df_xgboost_results_next["date"]
        df_xgboost_next_new = df_xgboost_results_next.copy()
        df_xgboost_next_new["date"] = pd.to_datetime(df_xgboost_next_new["date"])
        df_xgboost_next_new["date"] = (
            df_xgboost_next_new["date"].dt.strftime("%Y%m%d").astype(int)
        )

        # SENTIMENT
        # training data
        df_text_sentiment_curr = df_text_sentiment[df_text_sentiment["year"] == year]
        # prediction data
        df_text_sentiment_next = df_text_sentiment[
            df_text_sentiment["year"] == year + 1
        ]

        # merge the xgboost stock predictions, sentiments (from text data) and HMM states
        # in order to build the dataframe to train the final xgboost model
        df = pd.merge(
            df_xgboost_results_new,
            df_text_sentiment_curr,
            on="gvkey",
            how="left",
        )
        df = pd.merge(df, df_hmm_curr, on=["gvkey", "date"], how="left")

        # TRAIN THE FINAL XGBOOST MODEL
        X = df.drop(
            [
                "Unnamed: 0_x",
                "actual_return",
                "model_used",
                "year",
                "Unnamed: 0_y",
                "gvkey",
                "date",
            ],
            axis=1,
        )

        # the target is the returns
        y = df["actual_return"]

        # Define the XGBoost regressor model
        xgbr = xgb.XGBRegressor(
            objective="reg:squarederror",
            n_estimators=1500,
            eval_metric="rmse",
            early_stopping_rounds=50,
            n_jobs=-1,
            tree_method="gpu_hist",  # This enables GPU acceleration
        )

        # Define the parameter space for RandomizedSearchCV
        param_distributions = {
            "learning_rate": uniform(0.005, 0.3),
            "max_depth": randint(10, 15),
            "subsample": uniform(0.5, 0.4),
            "colsample_bytree": uniform(0.5, 0.4),
            "gamma": uniform(0, 0.5),
            "min_child_weight": randint(1, 10),
        }

        # Set up RandomizedSearchCV
        random_search = RandomizedSearchCV(
            xgbr,
            param_distributions=param_distributions,
            n_iter=25,
            cv=5,
            scoring="neg_mean_squared_error",
            random_state=42,
            n_jobs=-1,
            verbose=1,
        )

        # Split the data into training and testing sets for this year
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Perform the randomized search
        random_search.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        # Get the best model from the search
        best_model = random_search.best_estimator_

        # save the model
        model_filename = f"data/final-strat-model-{year}.joblib"
        joblib.dump(best_model, model_filename)

        # create the dataframe that contains the data for the year we will make our predictions on
        # this is the dataframe that we will input into the model (best_model) that has just been trained
        df = pd.merge(
            df_xgboost_next_new, df_text_sentiment_next, on="gvkey", how="left"
        )
        df = pd.merge(df, df_hmm_next, on=["gvkey", "date"], how="left")

        gvkey_date = df[["gvkey", "date"]]
        X = df.drop(
            [
                "Unnamed: 0_x",
                "actual_return",
                "model_used",
                "year",
                "Unnamed: 0_y",
                "gvkey",
                "date",
            ],
            axis=1,
        )

        y = df["actual_return"]

        predictions = best_model.predict(X)

        test_mse = mean_squared_error(y, predictions)
        print(test_mse)

        # predicted values
        df_results = pd.DataFrame(
            {
                "gvkey": gvkey_date["gvkey"],
                "date": gvkey_date["date"],
                "actual_values": y,
                "predicted_values": predictions,
            }
        )

        # get the right date format
        df_results["date"] = pd.to_datetime(
            df_results["date"].astype(str), format="%Y%m%d"
        )

        # sort the values by date
        df_results.sort_values(by="date", inplace=True)

        # save the the predictions to a file
        df_results.to_csv(f"data/results_{year + 1}.csv", index=False)

        # iterate through each month to select the stocks that will be
        # short/long in our portfolio
        for month in range(1, 13):
            # only get the data for the current month
            filtered = df_results[
                (df_results["date"].dt.year == (year + 1))
                & (df_results["date"].dt.month == month)
            ]

            # sort the predicted stocks
            filtered.sort_values(by="predicted_values", inplace=True)
            # the most negative predicted returns
            most_negative = filtered.head(100).copy()

            # the most positive predicted returns
            most_positive = filtered.tail(100).copy()

            # monthly return by shorting the most negative stocks and longing the most positive
            monthly_return = (
                most_positive["actual_values"].mean()
                - most_negative["actual_values"].mean()
            )

            date = pd.to_datetime({"year": [year + 1], "month": [month], "day": [1]})[0]
            new_row = {"date": date, "returns": monthly_return}

            # add this monthly return to our final returns dataframe
            df_returns = pd.concat(
                [df_returns, pd.DataFrame([new_row])], ignore_index=True
            )
    # at this point we have the returns for everymonth across the entire
    # OOS period, so we can seve it to a dataframe
    df_returns.to_csv("final_results.csv")

    # save all the prediction results to one single file
    all_results = pd.concat([pd.read_csv(f) for f in glob.glob("data/results_*.csv")])
    all_results["date"] = pd.to_datetime(all_results["date"])

    all_results.to_csv("actual_pred_per_stock.csv")

    # we also execute the pairs trades (the implementation will save the trades to a csv)
    execute_all_pairs_trades()


if __name__ == "__main__":
    main()
