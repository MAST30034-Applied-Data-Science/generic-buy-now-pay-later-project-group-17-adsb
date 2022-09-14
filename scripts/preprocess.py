import numpy as np
import pandas as pd
import argparse
from pathlib import Path


def read_transactions(transactions_directories):
    return pd.concat([pd.read_parquet(trans_dir) for trans_dir in transactions_directories])


def get_merchants(merchants_file):
    # todo: normalise tags
    return pd.read_parquet(merchants_file)


def get_consumers(consumer_mapping, consumer):
    return pd.read_parquet(consumer_mapping).merge(pd.read_csv(consumer, sep="|"), how="inner", on="consumer_id")


def merge_data(transactions, merchants, consumers):
    # drop transactions with no valid linked merchant
    transactions = transactions.merge(merchants, how="inner", on="merchant_abn")
    transactions = transactions.merge(consumers, how="left", on="user_id")
    transactions["order_datetime"] = pd.to_datetime(transactions["order_datetime"])

    # drop and rename columns
    transactions = transactions.rename(columns={"name_x": "merchant_name", "name_y": "consumer_name"})
    transactions = transactions.drop(columns=["consumer_id", "address"])

    return transactions


# remove all transactions with values outside of IQR of the merchant
def remove_outliers(data):
    pass


def clean(out):
    # out = remove_outliers(out)
    return out


def etl(data_dir, data_config):
    print("Begin ETL")

    # resolve full paths from config and data directory
    merchants = get_merchants(Path(data_dir, data_config["merchants"]).resolve())
    transactions = read_transactions([Path(data_dir, path).resolve() for path in data_config["transactions"]])
    consumers = get_consumers(Path(data_dir, data_config["consumer_mapping"]).resolve(), Path(data_dir, data_config["consumer"]).resolve())
    transactions_output = Path(data_dir, data_config["transactions_output"]).resolve()

    # merge all relevant tables
    out = merge_data(transactions, merchants, consumers)
    out = clean(out)

    # output final data to parquet file
    out.to_parquet(transactions_output)

    print("Completed ETL")
