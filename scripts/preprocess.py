import numpy as np
import pandas as pd
import argparse
from pathlib import Path


def read_transactions(transactions_directories):
    return pd.concat([pd.read_parquet(trans_dir) for trans_dir in transactions_directories])


def get_merchants(merchants_file):
    return pd.read_parquet(merchants_file)


def get_consumers(consumer_mapping, consumer):
    return pd.read_parquet(consumer_mapping).merge(pd.read_csv(consumer, sep="|"), how="inner", on="consumer_id")


def merge_data(transactions, merchants, consumers):
    transactions = transactions.merge(merchants, how="left", on="merchant_abn")
    transactions = transactions.merge(consumers, how="left", on="user_id")
    transactions["order_datetime"] = pd.to_datetime(transactions["order_datetime"])

    return transactions


def clean(out):
    pass


def etl(data_dir, data_config):
    print("Begin ETL")

    # resolve full paths from config and data directory
    merchants = get_merchants(Path(data_dir, data_config["merchants"]).resolve())
    transactions = read_transactions([Path(data_dir, path).resolve() for path in data_config["transactions"]])
    consumers = get_consumers(Path(data_dir, data_config["consumer_mapping"]).resolve(), Path(data_dir, data_config["consumer"]).resolve())
    transactions_output = Path(data_dir, data_config["transactions_output"]).resolve()

    # merge all relevant tables
    out = merge_data(transactions, merchants, consumers)
    out.to_parquet(transactions_output)

    print("Completed ETL")
