import numpy as np
import pandas as pd
import argparse
from pathlib import Path
import re

def read_transactions(transactions_directories):
    return pd.concat([pd.read_parquet(trans_dir) for trans_dir in transactions_directories])


def get_merchants(merchants_file):
    return pd.read_parquet(merchants_file)

def process_tags(tag):
    result = re.search(r'^[\[\(]{2}(.+?(?:, ?.+)*)[\]\)], [\[\(]([a-z])[\]\)], [\(\[].+: (\d+\.?\d+)[\)\]]{2}$', tag)
    return result.group(1), result.group(2), result.group(3)

# todo: implement tag normalisation here, return df with one-hot? columns
def normalise_tags(merchants):
    merchants[["sector_tags", "revenue_band", "take_rate"]] = merchants.apply(lambda row: process_tags(row.tags),axis='columns', result_type='expand')
    merchants["sector_tags"] = merchants["sector_tags"].str.lower().str.replace(' +', ' ', regex=True).str.strip()
    return merchants

def get_consumers(consumer_mapping, consumer):
    return pd.read_parquet(consumer_mapping).merge(pd.read_csv(consumer, sep="|"), how="inner", on="consumer_id")

def get_census(census_file):
    return pd.read_csv(census_file)

def merge_data(transactions, merchants, consumers, census):
    # drop transactions with no valid linked merchant
    transactions = transactions.merge(merchants, how="inner", on="merchant_abn")
    transactions = transactions.merge(consumers, how="left", on="user_id")
    transactions = transactions.merge(census, how="left", on="postcode")

    transactions["order_datetime"] = pd.to_datetime(transactions["order_datetime"])

    # drop and rename columns
    transactions = transactions.rename(columns={"name_x": "merchant_name", "name_y": "consumer_name"})
    transactions = transactions.drop(columns=["consumer_id", "address"])

    return transactions


# remove all transactions with values outside of IQR of the merchant
def remove_outliers(data):
    data = data.reset_index()
    data_noOutlier = data[~data.groupby('merchant_abn')['dollar_value'].apply(find_outlier)]

    return data_noOutlier.set_index("index")

def remove_nomerchant(transactions, merchants):
    return transactions[transactions["merchant_abn"].isin(merchants.index.to_numpy())]

# Get IQR range and remove outliers
def find_outlier(merchant):
    Q3 = np.quantile(merchant, 0.75)
    Q1 = np.quantile(merchant, 0.25)
    IQR = Q3 - Q1
    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    return ~merchant.between(lower_limit, upper_limit)


def clean(out):
    # out = remove_outliers(out)
    return out


def etl(data_dir, data_config):
    print("Begin ETL")

    # resolve full paths from config and data directory
    output_dir = Path(data_dir, data_config["output_dir"]).resolve()

    merchants = normalise_tags(get_merchants(Path(data_dir, data_config["merchants"]).resolve()))
    merchants.to_parquet(Path(output_dir, "merchants.parquet"))

    transactions = read_transactions([Path(data_dir, path).resolve() for path in data_config["transactions"]])
    transactions = remove_nomerchant(transactions, merchants)
    transactions = remove_outliers(transactions)
    transactions.to_parquet(Path(output_dir, "transactions.parquet"))

    consumers = get_consumers(Path(data_dir, data_config["consumer_mapping"]).resolve(), Path(data_dir, data_config["consumer"]).resolve())
    consumers.to_parquet(Path(output_dir, "consumers.parquet"))

    census = get_census(Path(data_dir, data_config["census"]).resolve())

    # merge all relevant tables
    out = merge_data(transactions, merchants, consumers, census)
    out = clean(out)

    # output final data to parquet file
    out.to_parquet(Path(output_dir, "merged_transactions.parquet"))

    print("Completed ETL")
