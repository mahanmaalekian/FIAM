import joblib
import pandas as pd
import xgboost as xgb
from scipy.stats import randint, uniform
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, train_test_split

from data_loader import load_data_to_parquet
from hmm_state_trainer import create_hmm_states_per_year
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

    # create the csv with the sentiment

    df_xgboost_results = pd.read_csv(XGBOOST_RESULTS_PATH)

    df_text_sentiment = pd.read_csv(TEXT_SENTIMENT_PATH)

    returns_df = pd.DataFrame(columns=["date", "returns"])
    for idx, year in enumerate(range(2014, 2025), start=1):
        print(f"\n[{idx}/11] Processing year {year}...")

        # HMM
        df_hmm = pd.read_csv(f"./data/{year}-hmm.csv")

        # training data
        df_hmm_year = df_hmm[df_hmm["date"] >= year * 10000]
        # prediction data
        df_hmm_next = df_hmm[df_hmm["date"] >= (year + 1) * 10000]

        # XGBOOST
        # training data
        df_xgboost_results_per_year = df_xgboost_results[
            (df_xgboost_results["date"] >= f"{year}-01-01")
            & (df_xgboost_results["date"] <= f"{year}-12-31")
        ]

        df_xgboost_results_per_year_new = df_xgboost_results_per_year.copy()

        df_xgboost_results_per_year_new["date"] = pd.to_datetime(
            df_xgboost_results_per_year_new["date"]
        )
        df_xgboost_results_per_year_new["date"] = (
            df_xgboost_results_per_year_new["date"].dt.strftime("%Y%m%d").astype(int)
        )

        df_xgboost_results_next = df_xgboost_results[
            (df_xgboost_results["date"] >= f"{year + 1}-01-01")
            & (df_xgboost_results["date"] <= f"{year + 1}-12-31")
        ]

        df_xgboost_results_next["date"]
        df_xgboost_next_new = df_xgboost_results_next.copy()

        df_xgboost_next_new["date"] = pd.to_datetime(df_xgboost_next_new["date"])
        df_xgboost_next_new["date"] = (
            df_xgboost_next_new["date"].dt.strftime("%Y%m%d").astype(int)
        )

        # SENTIMENT
        # training
        df_text_sentiment_year = df_text_sentiment[df_text_sentiment["year"] == year]
        # prediction data
        df_text_sentiment_next = df_text_sentiment[
            df_text_sentiment["year"] == year + 1
        ]

        df = pd.merge(
            df_xgboost_results_per_year_new,
            df_text_sentiment_year,
            on="gvkey",
            how="left",
        )

        df = pd.merge(df, df_hmm_year, on=["gvkey", "date"], how="left")

        # TRAIN THE MODEL

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
        # data we will predict on

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

        results_df = pd.DataFrame(
            {
                "gvkey": gvkey_date["gvkey"],
                "date": gvkey_date["date"],
                "actual_values": y,
                "predicted_values": predictions,
            }
        )

        results_df["date"] = pd.to_datetime(
            results_df["date"].astype(str), format="%Y%m%d"
        )
        results_df.sort_values(by="date", inplace=True)
        for month in range(1, 13):
            filtered = results_df[
                (results_df["date"].dt.year == (year + 1))
                & (results_df["date"].dt.month == month)
            ]
            filtered.sort_values(by="predicted_values", inplace=True)
            most_negative = filtered.head(100).copy()
            most_positive = filtered.tail(100).copy()
            monthly_return = (
                most_positive["actual_values"].mean()
                - most_negative["actual_values"].mean()
            )

            date = pd.to_datetime({"year": [year + 1], "month": [month], "day": [1]})[0]
            new_row = {"date": date, "returns": monthly_return}
            returns_df = pd.concat(
                [returns_df, pd.DataFrame([new_row])], ignore_index=True
            )
            returns_df.to_csv("final_results.csv")


if __name__ == "__main__":
    main()

