import numpy as np
import pandas as pd
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("transactions_directory", help="path to transactions directory", type=Path)
parser.add_argument("merchants_file", help="path to merchants file", type=Path)
parser.add_argument("transactions_output", help="filepath to output transactions to", type=Path)
args = parser.parse_args()

def generate_data():
    merchants = pd.read_parquet(args.merchants_file)
    transactions = pd.read_parquet(args.transactions_directory)

    transactions = transactions.merge(merchants, how="left", on="merchant_abn")
    transactions["order_datetime"] = pd.to_datetime(transactions["order_datetime"])

    transactions.to_parquet(args.transactions_output)


if __name__ == "__main__":
    generate_data()