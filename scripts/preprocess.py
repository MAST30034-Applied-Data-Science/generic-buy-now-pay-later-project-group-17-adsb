import numpy as np
import pandas as pd
import random
from pathlib import Path
import re
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import metrics

seed = 42
random.seed(seed)
np.random.seed(seed)

def read_transactions(transactions_directories):
    return pd.concat([pd.read_parquet(trans_dir) for trans_dir in transactions_directories])


def get_merchants(merchants_file):
    return pd.read_parquet(merchants_file)

def process_tags(tag):
    result = re.search(r'^[\[\(]{2}(.+?(?:, ?.+)*)[\]\)], [\[\(]([a-z])[\]\)], [\(\[].+: (\d+\.?\d+)[\)\]]{2}$', tag)
    return result.group(1), result.group(2), result.group(3)

def normalise_tags(merchants):
    merchants[["sector_tags", "revenue_band", "take_rate"]] = merchants.apply(lambda row: process_tags(row.tags),axis='columns', result_type='expand')
    merchants["sector_tags"] = merchants["sector_tags"].str.lower().str.replace(' +', ' ', regex=True).str.strip()
    return merchants

def get_consumers(consumer_mapping, consumer):
    return pd.read_parquet(consumer_mapping).merge(pd.read_csv(consumer, sep="|"), how="inner", on="consumer_id")


def get_census(census_dir):
    G01 = pd.read_csv(Path(census_dir, "2021Census_G01_AUST_POA.csv"))
    G02 = pd.read_csv(Path(census_dir, "2021Census_G02_AUST_POA.csv"))
    G017 = pd.read_csv(Path(census_dir, "2021Census_G17A_AUST_POA.csv"))
    G017c = pd.read_csv(Path(census_dir, "2021Census_G17C_AUST_POA.csv"))

    dataset = pd.merge(G01, G02, how="outer", on="POA_CODE_2021")
    dataset = pd.merge(dataset, G017, how="outer", on="POA_CODE_2021")
    dataset = pd.merge(dataset, G017c, how="outer", on="POA_CODE_2021")

    high_income_brackets = ["P_3000_3499_Tot", "P_3500_more_Tot", "P_2000_2999_Tot"]

    return dataset[[*high_income_brackets, "POA_CODE_2021", "Tot_P_M", "Tot_P_F","Tot_P_P", "M_Neg_Nil_income_Tot", "F_Neg_Nil_income_Tot", "Median_age_persons", "Median_mortgage_repay_monthly", "Median_tot_prsnl_inc_weekly", "Median_tot_hhd_inc_weekly"]]


def preprocess_census(dataset):
    high_income_brackets = ["P_3000_3499_Tot", "P_3500_more_Tot", "P_2000_2999_Tot"]

    dataset['postcode'] = dataset['POA_CODE_2021'].apply(lambda x: x[3:])

    dataset['nill_income_percent_M'] = dataset['M_Neg_Nil_income_Tot'] / dataset['Tot_P_M']
    dataset['nill_income_percent_F'] = dataset['F_Neg_Nil_income_Tot'] / dataset['Tot_P_F']

    dataset["high_income_proportion"] = dataset[high_income_brackets].sum(axis=1) / dataset["Tot_P_P"]

    dataset = dataset[["postcode", "Median_tot_hhd_inc_weekly", 'nill_income_percent_F', 'nill_income_percent_M', "Median_age_persons", "high_income_proportion", "Median_mortgage_repay_monthly"]]

    # 0 is an invalid value for all of the above columns, replace with na
    dataset = dataset.replace(0, np.nan)

    #impute all NaNs with column medians
    dataset = dataset.fillna(dataset.median())

    dataset['house_repay_to_income'] = dataset["Median_mortgage_repay_monthly"] / (dataset["Median_tot_hhd_inc_weekly"] * 4.333333)

    return dataset

def merge_data(transactions, merchants, consumers, census):
    # drop transactions with no valid linked merchant
    transactions = transactions.merge(merchants, how="inner", on="merchant_abn")
    transactions = transactions.merge(consumers, how="inner", on="user_id")
    transactions = transactions.merge(census, how="inner", on="postcode")

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

def remove_fraud(data, fraud_model):
    consumer_transactions_day = data.groupby(['user_id', 'order_datetime']).agg(total_dollar=pd.NamedAgg(column='dollar_value', aggfunc="sum")).reset_index()
    consumer_transactions_day["normal_total_dollar"] = (consumer_transactions_day.groupby('user_id')['total_dollar'].apply(lambda x: (x - x.median()) / (x.quantile(0.75) - x.quantile(0.25))))
    consumer_transactions_day["fraud_prob"] = consumer_transactions_day['normal_total_dollar'].apply(lambda x: (fraud_model.predict([[x]]))[0] if x>2 else 0.00001) / 100
    consumer_transactions_day["generated_prob"] = np.random.random(size=len(consumer_transactions_day))
    consumer_transactions_day["remove"] = consumer_transactions_day["generated_prob"] < consumer_transactions_day['fraud_prob']

    consumer_transactions_day = consumer_transactions_day[["user_id", "order_datetime", "remove"]]

    data = data.merge(consumer_transactions_day, how="left", on=["user_id", "order_datetime"])
    data = data[data["remove"] == False].drop(columns="remove")

    return data

def get_fraud_model(data, fraud_prob_file):
    fraud_prob = pd.read_csv(fraud_prob_file)

    consumer_transactions_day = data.groupby(['user_id', 'order_datetime']).agg(total_dollar=pd.NamedAgg(column='dollar_value', aggfunc="sum")).reset_index()
    consumer_transactions_day["normal_total_dollar"] = (consumer_transactions_day.groupby('user_id')['total_dollar'].apply(lambda x: (x-x.median())/(x.quantile(0.75)-x.quantile(0.25))))
    fraud_consumer_norm_prob = consumer_transactions_day.merge(fraud_prob, how='inner', on=['user_id', 'order_datetime'])

    lm = linear_model.LinearRegression()
    X = fraud_consumer_norm_prob["normal_total_dollar"]
    y = fraud_consumer_norm_prob["fraud_probability"]

    X = X.values.reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=seed)

    lm.fit(X_train, y_train)
    y_pred = lm.predict(X_test)
    print("Fraud model test R2:", metrics.r2_score(y_test, y_pred))

    # lm.fit(X, y)

    return lm

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

    fraud_model = get_fraud_model(transactions, Path(data_dir, data_config["consumer_fraud"]))
    transactions = remove_fraud(transactions, fraud_model)

    transactions = remove_outliers(transactions)

    transactions.to_parquet(Path(output_dir, "transactions.parquet"))

    consumers = get_consumers(Path(data_dir, data_config["consumer_mapping"]).resolve(), Path(data_dir, data_config["consumer"]).resolve())
    consumers.to_parquet(Path(output_dir, "consumers.parquet"))

    census = preprocess_census(get_census(Path(data_dir, data_config["census"]).resolve()))
    census.to_csv(Path(output_dir, "census.csv"), header=True, index=False)
    # reread to fix column types
    census = pd.read_csv(Path(output_dir, "census.csv"))

    # merge all relevant tables
    out = merge_data(transactions, merchants, consumers, census)
    out = clean(out)

    # output final data to parquet file
    out.to_parquet(Path(output_dir, "merged_transactions.parquet"))

    print("Completed ETL")
