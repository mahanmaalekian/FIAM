"""
data_loader.py
--------------

Converts raw global stock data into filtered and memory-efficient formats.
Filters for North American companies (CAN, USA) and saves both CSV and Parquet
versions in the `data/` directory.
"""

# Operating System
import os

# Data Management
import pandas as pd
import polars as pl
import pyarrow as pa

# Global Variables
GLOBAL_CSV_FILENAME = "ret_sample.csv"
NA_CSV_FILENAME = "stocks.csv"
PARQET_FILENAME = "stocks.parquet"
DATA_DIR = "data/"


def load_data_to_parquet():
    """
    Load, filter, and convert stock data into Parquet format.

    Steps:
    1. If `stocks.csv` doesn't exist, read `ret_sample.csv` and keep only
       North American companies.
    2. If `stocks.parquet` doesn't exist, convert the filtered CSV into
       a compressed Parquet file.

    Returns
    -------
    None
        Writes `stocks.csv` and/or `stocks.parquet` to the `data/` directory.
    """
    if not os.path.exists(os.path.join(DATA_DIR, NA_CSV_FILENAME)):
        # read sample data
        file_path = os.path.join(DATA_DIR, GLOBAL_CSV_FILENAME)
        data = pl.read_csv(file_path)
        data = data.filter(pl.col("excntry").is_in(["CAN", "USA"]))
        # write the North American csv
        data.write_csv(os.path.join(DATA_DIR, NA_CSV_FILENAME))

    if not os.path.exists(os.path.join(DATA_DIR, PARQET_FILENAME)):
        # write the parquet file (more memory efficient)
        data = pd.read_csv(os.path.join(DATA_DIR, NA_CSV_FILENAME), dtype={4: str})
        data.to_parquet(
            os.path.join(DATA_DIR, PARQET_FILENAME), index=False, compression="snappy"
        )
