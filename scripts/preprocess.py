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
    """Read in transactions into a single dataframe from a list of directories

    Args:
        transactions_directories (list): list of paths to transaction parquet files

    Returns:
        dataframe: dataframe of all transactions
    """        
    return pd.concat([pd.read_parquet(trans_dir) for trans_dir in transactions_directories])


def get_merchants(merchants_file):
    """Read merchants parquet file into a dataframe

    Args:
        merchants_file (PosixPath): path to merchant parquet file

    Returns:
        dataframe: dataframe of merchants
    """    
    return pd.read_parquet(merchants_file)

def process_tags(tag):
    """Split uncleaned tag string into cleaned tags, take rate and revenue band

    Args:
        tag (string): uncleaned tag string

    Returns:
        string, string, string: sector_tags, revenue_band, take_rate of merchant
    """    
    result = re.search(r'^[\[\(]{2}(.+?(?:, ?.+)*)[\]\)], [\[\(]([a-z])[\]\)], [\(\[].+: (\d+\.?\d+)[\)\]]{2}$', tag)
    return result.group(1), result.group(2), result.group(3)

def normalise_tags(merchants):
    """Split merchant tags into sector tags, revenue band and take rate, and clean sector tags

    Args:
        merchants (dataframe): raw merchants dataframe

    Returns:
        dataframe: cleaned merchants dataframe
    """    
    merchants[["sector_tags", "revenue_band", "take_rate"]] = merchants.apply(lambda row: process_tags(row.tags),axis='columns', result_type='expand')
    merchants["sector_tags"] = merchants["sector_tags"].str.lower().str.replace(' +', ' ', regex=True).str.strip()
    return merchants

def get_consumers(consumer_mapping, consumer):
    """Read consumer parquet csv and mapping into a dataframe

    Args:
        consumer_mapping (PosixPath): path to consumer mapping parquet file
        consumer (PosixPath): path to consumer csv file

    Returns:
        dataframe: cleaned consumer dataframe
    """    
    return pd.read_parquet(consumer_mapping).merge(pd.read_csv(consumer, sep="|"), how="inner", on="consumer_id")


def get_census(census_dir):
    """Select relevant census features

    Args:
        census_dir (PosixPath): path to census csv

    Returns:
        dataframe: census data dataframe with relevant features
    """    
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
    """Engineer relevant census features

    Args:
        dataset (dataframe): initial census dataframe

    Returns:
        dataframe: census dataframe with engineered features
    """    
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
    """Merge data together

    Args:
        transactions (dataframe): processed transactions dataframe
        merchants (dataframe): processed merchants dataframe
        consumers (dataframe): processed consumer dataframe
        census (dataframe): processed census dataframe

    Returns:
        dataframe: merged transactions dataframe
    """    
    # drop transactions with no valid linked merchant
    transactions = transactions.merge(merchants, how="inner", on="merchant_abn")
    transactions = transactions.merge(consumers, how="inner", on="user_id")
    transactions = transactions.merge(census, how="inner", on="postcode")

    transactions["order_datetime"] = pd.to_datetime(transactions["order_datetime"])

    # drop and rename columns
    transactions = transactions.rename(columns={"name_x": "merchant_name", "name_y": "consumer_name"})
    transactions = transactions.drop(columns=["consumer_id", "address"])

    return transactions


def remove_outliers(data):
    """Remove all transactions with values outside of IQR of the merchant

    Args:
        data (dataframe): raw transactions dataframe

    Returns:
        dataframe: transactions dataframe with outliers removed
    """    
    data = data.reset_index()
    data_noOutlier = data[~data.groupby('merchant_abn')['dollar_value'].apply(find_outlier)]

    return data_noOutlier.set_index("index")

def remove_fraud(data, fraud_model):
    """Remove likely fradulent transactions based on customer-day spending

    Args:
        data (dataframe): transactions dataframe
        fraud_model (sklearn model): model that predicts the fraud probability of transactions

    Returns:
        dataframe: transactions dataframe with likely fraudulent transactions removed
    """

    # sum transactions values per customer-day
    consumer_transactions_day = data.groupby(['user_id', 'order_datetime']).agg(total_dollar=pd.NamedAgg(column='dollar_value', aggfunc="sum")).reset_index()
    # standardise these customer-day transactions
    consumer_transactions_day["normal_total_dollar"] = (consumer_transactions_day.groupby('user_id')['total_dollar'].apply(lambda x: (x - x.median()) / (x.quantile(0.75) - x.quantile(0.25))))
    # generate fraud probability for scaled values of 2 or more
    consumer_transactions_day["fraud_prob"] = consumer_transactions_day['normal_total_dollar'].apply(lambda x: (fraud_model.predict([[x]]))[0] if x>2 else 0.00001) / 100
    # remove entries with probability equal to their fraud probability
    consumer_transactions_day["generated_prob"] = np.random.random(size=len(consumer_transactions_day))
    consumer_transactions_day["remove"] = consumer_transactions_day["generated_prob"] < consumer_transactions_day['fraud_prob']

    consumer_transactions_day = consumer_transactions_day[["user_id", "order_datetime", "remove"]]

    # remove transactions corresponding to likely fraudlent customer-days
    data = data.merge(consumer_transactions_day, how="left", on=["user_id", "order_datetime"])
    data = data[data["remove"] == False].drop(columns="remove")

    return data

def get_fraud_model(data, fraud_prob_file):
    """Train a fraud model using a linear model

    Args:
        data (dataframe): tansactions dataframe
        fraud_prob_file (PosixPath): path to fraud dataset csv

    Returns:
        sklearn model: fraud model predicting fraud probability from transactions
    """    
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
    """Remove transactions with no valid linked merchant

    Args:
        transactions (dataframe): raw transactions dataframe
        merchants (dataframe): merchants dataframe

    Returns:
        dataframe: transactions dataframe with transactions with no valid linked merchants removed
    """    
    return transactions[transactions["merchant_abn"].isin(merchants.index.to_numpy())]

# Get IQR range and remove outliers
def find_outlier(merchant):
    """Remove outlying transactions for a given merchant

    Args:
        merchant (array): transactions of a given merchant

    Returns:
        array: boolean array of outlying transactions
    """    
    Q3 = np.quantile(merchant, 0.75)
    Q1 = np.quantile(merchant, 0.25)
    IQR = Q3 - Q1
    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    return ~merchant.between(lower_limit, upper_limit)


def etl(data_dir, data_config):
    """Run the ETL pipeline, outputting curated data

    Args:
        data_dir (PosixPath): path to data directory
        data_config (dict): data configuration
    """    
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

    # output final data to parquet file
    out.to_parquet(Path(output_dir, "merged_transactions.parquet"))

    print("Completed ETL")
