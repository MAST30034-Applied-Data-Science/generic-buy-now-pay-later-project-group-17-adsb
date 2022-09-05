import numpy as np
import pandas as pd

data_directory = "../data/"
tables_directory = data_directory + "tables/"
transactions_directory = tables_directory + "transactions_20210228_20210827_snapshot/"
merchants_file_loc = tables_directory + "tbl_merchants.parquet"

transactions_output = data_directory + "curated/transactions.parquet"

def generate_data():
    merchants = pd.read_parquet(merchants_file_loc)
    transactions = pd.read_parquet(transactions_directory)

    transactions = transactions.merge(merchants, how="left", on="merchant_abn")
    transactions["order_datetime"] = pd.to_datetime(transactions["order_datetime"])

    transactions.to_parquet(transactions_output)


if __name__ == "__main__":
    generate_data()