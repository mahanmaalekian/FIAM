"""
hmm_state_trainer.py
--------------------

This module trains a Hidden Markov Model (HMM) on North American company data
to infer underlying market states for each company and month. It processes
the input data, standardizes features, trains one HMM per year (using up to
the previous 10 years of data), and saves yearly results to CSV files.

The main entry point is:
    create_hmm_states_per_year()
"""

import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler


def create_hmm_states_per_year():
    """
    Train a yearly Hidden Markov Model (HMM) on company data to infer market states.

    This function loads financial data from 'data/stocks.parquet', cleans and scales it,
    and fits an HMM using up to the previous 10 years of data for each year after 2014.
    The predicted hidden market states are saved to CSV files in the 'data/' directory,
    one file per year.

    Process overview:
    -----------------
    1. Load and clean data, removing unnecessary columns and NaN-heavy features.
    2. Standardize numerical columns using sklearn's StandardScaler.
    3. For each year (starting from 2015):
        - Train an HMM on up to 10 years of past data.
        - Predict hidden states for the current year.
        - Save the results as 'data/{year}.csv'.

    Notes
    -----
    - Uses 3 hidden states and diagonal covariance matrices.
    - Prints the log-likelihood of each yearly model on its test data.

    Returns
    -------
    None
        The function writes output CSV files to the 'data/' folder.
    """
    # Load data
    data = pd.read_parquet("data/stocks.parquet")

    # Drop irrelevant columns
    drop_columns = ["id", "iid", "ret_eom", "excntry", "month", "char_eom", "char_date"]
    data = data.drop(columns=[c for c in drop_columns if c in data.columns])
    threshold = 0.5 * len(data)
    data = data.dropna(axis=1, thresh=threshold)
    data = data.dropna()

    # Extract year column after cleaning
    year_col = data["year"].copy()
    data = data.drop(columns=["year"])
    date_col = data["date"].copy()
    data = data.drop(columns=["date"])
    gvkey_col = data["gvkey"].copy()
    data = data.drop(columns=["gvkey"])

    # Standardize features across the whole dataset (important for consistency)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(data)
    # Reattach year info
    data_scaled = pd.DataFrame(scaled_features, columns=data.columns)
    data_scaled["year"] = year_col.values
    data_scaled["date"] = date_col.values
    data_scaled["gvkey"] = gvkey_col.values

    # HMM parameters
    n_hidden_states = 3

    # Dictionary to store results
    results = {}

    years = sorted(data_scaled["year"].unique())

    for i, yr in enumerate(years):
        # Skip first year (no past data to train on)
        if i == 0 or yr <= 2014:
            continue

        # Train on all data up to previous year
        train_data = (
            data_scaled[(data_scaled["year"] >= yr - 10) & (data_scaled["year"] < yr)]
            .drop(columns=["year", "gvkey", "date"])
            .values
        )
        test_data = (
            data_scaled[(data_scaled["year"] >= yr - 10) & (data_scaled["year"] <= yr)]
            .drop(columns=["year", "gvkey", "date"])
            .values
        )

        if len(train_data) < n_hidden_states or len(test_data) == 0:
            continue

        # Fit HMM on past data
        model = GaussianHMM(
            n_components=n_hidden_states,
            covariance_type="diag",
            n_iter=1000,
            random_state=42,
        )
        model.fit(train_data)

        # Predict states for current year only (out-of-sample)
        hidden_states = model.predict(test_data)

        # Log-likelihood on current year
        log_likelihood = model.score(test_data)

        # Store results
        results[yr] = {
            "model": model,
            "hidden_states": hidden_states,
            "log_likelihood": log_likelihood,
        }

        print(f"Year {yr}: log-likelihood (on test year) = {log_likelihood:.2f}")
        ret = data_scaled[
            (data_scaled["year"] >= yr - 10) & (data_scaled["year"] <= yr)
        ].copy()  # Make an explicit copy
        ret.loc[:, "hmm_state"] = (
            hidden_states  # Assign all hidden states, not just hidden_states[0]
        )
        ret = ret[["gvkey", "hmm_state", "date"]]
        ret.to_csv(f"data/hmm-{yr}.csv")
