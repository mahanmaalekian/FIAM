#!/usr/bin/env python
# coding: utf-8

# Remove unwanted warning
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Data Management
import pandas as pd
import numpy as np
import polars as pl
import pyarrow as pa

# Feature Engineering
from sklearn.preprocessing import StandardScaler

# Cointegration and Statistics
from statsmodels.tsa.stattools import coint
import statsmodels.api as sm

# Machine Learning
from sklearn.cluster import KMeans
from kneed import KneeLocator


# Operating System
import os
import gc



# Global Variables
CSV_FILENAME = "stocks.csv"
PARQET_FILENAME = "stocks.parquet"
WORKING_DIR = "data/"
FEATURES = ["gvkey"]
load_coint_pairs = False


# Helper Functions
def calculate_cointegration(series_1, series_2, sig_level=0.05):
    # Engle–Granger test
    coint_res = coint(series_1, series_2)
    coint_t, p_value, crit_values = coint_res

    # regression to estimate hedge ratio
    X = sm.add_constant(series_2)
    model = sm.OLS(series_1, X).fit()
    hedge_ratio = model.params[1]

    # cointegration flag (use both critical value + p-value check)
    coint_flag = int((coint_t < crit_values[1]) and (p_value < sig_level))
    

    # define a "strength score" (lower p-value = stronger)
    # you can also do something like (crit_values[1] - coint_t) for distance from threshold
    score = -p_value  

    return {
        "flag": coint_flag,
        "hedge_ratio": hedge_ratio,
        "p_value": p_value,
        "score": score
    }


def get_stock_returns_upto_year(year:int):
    if not os.path.exists(os.path.join(
        WORKING_DIR, CSV_FILENAME
    )):
    # read sample data
        file_path = os.path.join(
            WORKING_DIR, "ret_sample.csv"
        )
        data = pl.read_csv(file_path)
        data = data.filter(pl.col("excntry").is_in(["CAN","USA"]))
        # write the North American csv
        data.write_csv(os.path.join(WORKING_DIR, CSV_FILENAME))

    if not os.path.exists(os.path.join(
        WORKING_DIR, PARQET_FILENAME
        )):
        # write the parquet file (more memory efficient)
        data = pd.read_csv(os.path.join(WORKING_DIR, CSV_FILENAME), dtype={4: str})
        data.to_parquet(PARQET_FILENAME, index=False, compression="snappy")

    # read the parquet file
    df_used_comps = pd.read_parquet(os.path.join(WORKING_DIR, PARQET_FILENAME))
    year *= 10000
    # only keep until the year we want
    df_used_comps = df_used_comps[df_used_comps["date"] < year]
    return df_used_comps[["date", "id", "stock_ret"]]

def optimize_cointegration_testing_combined(clusters_clean, df_ret, 
                                           corr_threshold=0.6, 
                                           max_pairs_per_cluster=100):
    """
    Combine correlation filtering with parallel processing
    """
    all_pairs_to_test = []
    
    for label in clusters_clean.unique():
        cluster_assets = clusters_clean[clusters_clean == label].index.tolist()
        if len(cluster_assets) <= 1:
            continue
            
        # Pre-filter by correlation
        cluster_data = df_ret[cluster_assets]
        corr_matrix = cluster_data.corr()
        
        high_corr_pairs = []
        for i, asset1 in enumerate(cluster_assets):
            for j, asset2 in enumerate(cluster_assets[i+1:], i+1):
                corr_val = abs(corr_matrix.loc[asset1, asset2])
                if corr_val > corr_threshold:
                    high_corr_pairs.append((asset1, asset2, corr_val, label))
        
        # Sort by correlation and limit
        high_corr_pairs.sort(key=lambda x: x[2], reverse=True)
        high_corr_pairs = high_corr_pairs[:max_pairs_per_cluster]
        
        all_pairs_to_test.extend([(pair[0], pair[1], pair[3]) for pair in high_corr_pairs])
    
    print(f"After correlation filtering: {len(all_pairs_to_test)} pairs to test")
    return all_pairs_to_test

def get_cointegrated_stocks_by_year(year: int):
    """ returns a DataFrame of the cointegrated stocks up to the given year"""
    # ### Data Extraction

    if not os.path.exists(os.path.join(
            WORKING_DIR, CSV_FILENAME
        )):
        # read sample data
        file_path = os.path.join(
            WORKING_DIR, "ret_sample.csv"
        )
        data = pl.read_csv(file_path)
        data = data.filter(pl.col("excntry").is_in(["CAN","USA"]))
        # write the North American csv
        data.write_csv(os.path.join(WORKING_DIR, CSV_FILENAME))

    if not os.path.exists(os.path.join(
        WORKING_DIR, PARQET_FILENAME
        )):
        # write the parquet file (more memory efficient)
        data = pd.read_csv(os.path.join(WORKING_DIR, CSV_FILENAME), dtype={4: str})
        data.to_parquet(PARQET_FILENAME, index=False, compression="snappy")

    # read the parquet file
    data = pd.read_parquet(os.path.join(WORKING_DIR, PARQET_FILENAME))




    # this will give us all the companies we want by removing ones from the same company
    df_no_duplicate = data.drop_duplicates(subset=["id"], keep='first')
    df_no_duplicate = df_no_duplicate.drop_duplicates(subset=["gvkey"], keep="first")
    df_ids = df_no_duplicate["id"]
    del df_no_duplicate
    gc.collect()
    df_used_comps = data[data["id"].isin(df_ids.values)]
    del data
    gc.collect()
    # Get companies that exist in the target year
    companies_in_target_year = df_used_comps[df_used_comps['date'] // 10000 == year]['id'].unique()

    # Keep only rows for companies that exist in target year
    df_used_comps = df_used_comps[df_used_comps['id'].isin(companies_in_target_year)]
    # only keep until the year we want
    year *= 10000
    df_used_comps = df_used_comps[df_used_comps["date"] < year]



    # get a df for just the returns for all companies
    # pivot so that gvkey are columns, date are rows, stock_ret are values
    df_ret = df_used_comps.pivot(index="date", columns="id", values="stock_ret")
    df_ret = df_ret.sort_index()
    df_ret.fillna(value=0, inplace=True)
    initial_price = 100
    df_ret = (1 + df_ret).cumprod() * initial_price


    # the features that we need for K-Means Clustering
    k_means_features = ["id", "excntry", "beta_60m", "betadown_252d", "ivol_capm_252d", "corr_1260d",
                        "dolvol_126d", "bidaskhl_21d", "market_equity", "be_me", "ebitda_mev", "ni_be"]
    df_kmeans_raw = df_used_comps[k_means_features]


    # find the mean of each feature for each company
    numeric_profiles = df_kmeans_raw.groupby("id").mean(numeric_only=True)

    # the country is also relevant for finding out if 2 stocks are related, so we one-hot encode the country
    categorical_profiles = df_kmeans_raw.groupby("id").agg({
        "excntry": "first",
    }).reset_index()
    categorical_dummies = pd.get_dummies(categorical_profiles, columns=["excntry"], prefix="country")
    df_kmeans = numeric_profiles.merge(categorical_dummies, on='id')


    # clean up data we don't need to save memory
    del df_kmeans_raw
    gc.collect()


    # Remove any companies with NaN values
    df_kmeans.dropna(inplace=True)
    print("has NaN values:", df_kmeans.isnull().values.any())


    # ### Feature Scaling


    non_scalable = df_kmeans[["id", "country_USA", "country_CAN"]].copy()
    scalable = df_kmeans.drop(columns=["id", "country_USA", "country_CAN"])

    scaler = StandardScaler()
    scaler = scaler.fit_transform(scalable)
    df_kmeans_scaled = pd.DataFrame(scaler, columns=scalable.columns, index=scalable.index)
    df_kmeans_scaled = pd.concat([df_kmeans_scaled, non_scalable], axis=1)
    df_kmeans_scaled = df_kmeans_scaled.set_index('id')

    del df_kmeans
    gc.collect()



    # ### K-Means Clustering

    # Find the optpimum number of clusters
    X = df_kmeans_scaled.copy()
    K = range(1, 20)
    distortions = []
    for k in K:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(X)
        distortions.append(kmeans.inertia_)

    kl = KneeLocator(K,distortions, curve="convex", direction="decreasing")
    c = kl.elbow
    print("Optimum Clusters:", c)



    # Fit K-Means Model
    k_means = KMeans(n_clusters=c)
    k_means.fit(X)


    # Return the series
    clustered_series = pd.Series(index=X.index, data=k_means.labels_.flatten())
    clusters_clean = clustered_series[clustered_series != -1]



    if not load_coint_pairs:
        df_coint = optimize_cointegration_testing_combined(
            clusters_clean, df_ret, 
            corr_threshold=0.6, 
            max_pairs_per_cluster=100,
        )


    # Loop through and calculate cointegrate pairs
    cointegrated_pairs = []
    for tup in df_coint:
        series_1 = df_ret[tup[0]].values.astype(float)
        series_2 = df_ret[tup[1]].values.astype(float)
        temp = calculate_cointegration(series_1, series_2)
        temp["base asset"] = tup[0]
        temp["compare asset"] = tup[1]
        cointegrated_pairs.append(temp)
    df_coint_new = pd.DataFrame(cointegrated_pairs)

    # rank the top 50 trades and return them
    df_coint_new = df_coint_new[df_coint_new["flag"] == 1]

    df_coint_new.sort_values(by="score")
    df_coint_csv = df_coint_new.iloc[:, 4:]
    df_coint_csv = df_coint_csv.iloc[:50]
    df_coint_csv.to_csv(f"data/cointegrated-pairs-{int(year/10000)}.csv", index=False)
    print(year/10000, "done")
    return df_coint_csv




