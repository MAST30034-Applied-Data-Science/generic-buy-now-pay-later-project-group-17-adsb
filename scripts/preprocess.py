import numpy as np
import pandas as pd
import argparse
from pathlib import Path


def generate_data(merchants_file, transactions_directory, transactions_output):
    merchants = pd.read_parquet(merchants_file)
    transactions = pd.read_parquet(transactions_directory)

    transactions = transactions.merge(merchants, how="left", on="merchant_abn")
    transactions["order_datetime"] = pd.to_datetime(transactions["order_datetime"])

    transactions.to_parquet(transactions_output)


def etl(data_dir, data_config):
    print("Begin ETL")

    merchants = Path(data_dir, data_config["merchants"]).resolve()
    transactions = Path(data_dir, data_config["transactions"]).resolve()
    transactions_output = Path(data_dir, data_config["transactions_output"]).resolve()
    generate_data(merchants, transactions, transactions_output)

    print("Completed ETL")
