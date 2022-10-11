import json
from pathlib import Path
import pandas as pd
import numpy as np

pd.options.mode.chained_assignment = None


# retention = (Customers End - Customers New) / Customers Start
def customer_retention(merchant_abn, transactions, month_period):
    def month_diff(d1, d2):
        return (d1.year - d2.year) * 12 + d1.month - d2.month

    def new_customers(aggregated, period):
        new = aggregated[aggregated["date_segment"] == period]["user_id"]
        old = aggregated[aggregated["date_segment"] == period + 1]["user_id"]

        return len(set(new) - set(old))

    # filter for specific merchant
    transactions = transactions.loc[transactions["merchant_abn"] == merchant_abn]

    if len(transactions) == 0:
        return []

    latest_date = max(transactions["order_datetime"])
    transactions["date_segment"] = transactions["order_datetime"].apply(
        lambda x: month_diff(latest_date, x) // month_period)
    max_segment = max(transactions["date_segment"])

    # aggregate by customer-timescale for specific merchant to get customers active within the timeframes
    aggregate_by = ["user_id", "date_segment"]
    aggregated_transactions = transactions.groupby(aggregate_by).count().add_suffix('_Count').reset_index()[
        aggregate_by]

    n_cust = lambda x: len(aggregated_transactions[aggregated_transactions["date_segment"] == x])

    retentions = [0 if n_cust(i + 1) == 0 else (n_cust(i) - new_customers(aggregated_transactions, i)) / n_cust(i + 1)
                  for i in range(0, max_segment - 1)]

    return retentions

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def get_rankings(merchants, transactions, consumers, census):
    transactions["order_datetime"] = pd.to_datetime(transactions["order_datetime"])

    # add retention score to rankings
    rankings = merchants.reset_index()[["merchant_abn", "name", "sector_tags"]]
    rankings["retention"] = rankings["merchant_abn"].apply(
        lambda x: np.mean(customer_retention(x, transactions, 6) or [0]))

    # add momentum and revenue rankings
    transactions["month"] = pd.DatetimeIndex(transactions['order_datetime']).year * 12 + pd.DatetimeIndex(
        transactions['order_datetime']).month
    transaction_momentum = transactions.groupby(["merchant_abn", "month"]).agg(
        monthly_revenue=("dollar_value", sum)).reset_index()
    transactions_groupby = transaction_momentum.groupby("merchant_abn")

    sma_periods = [6, 12]
    col_names = [f"{sma_period}-month-sma" for sma_period in sma_periods]
    diff_col_names = [f"{sma_period}-month-sma-diff" for sma_period in sma_periods]

    for i, sma_period in enumerate(sma_periods):
        transaction_momentum[col_names[i]] = \
        transactions_groupby.rolling(window=sma_period, on="month").mean().reset_index(drop=True).fillna(0)[
            "monthly_revenue"]
        transaction_momentum[col_names[i] + "-shifted"] = transaction_momentum[col_names[i]].shift(1)

    transaction_momentum[diff_col_names] = \
    transaction_momentum.sort_values(["merchant_abn", "month"]).groupby("merchant_abn").diff().fillna(0)[col_names]

    revenue_and_momentum = transaction_momentum.groupby("merchant_abn").last().reset_index()
    # *6 to get back roc from SMA
    revenue_and_momentum["momentum_roc"] = (
                (revenue_and_momentum["6-month-sma-diff"] / revenue_and_momentum["6-month-sma-shifted"]) * 6 * 4).replace(
        [np.inf, -np.inf, np.nan], 0)
    revenue_and_momentum["momentum_score"] = sigmoid(revenue_and_momentum["momentum_roc"])

    revenue_and_momentum = revenue_and_momentum.merge(merchants.reset_index()[["merchant_abn", "take_rate"]],
                                                      how="inner", on="merchant_abn")
    revenue_and_momentum["adj_revenue"] = revenue_and_momentum["12-month-sma"] * pd.to_numeric(
        revenue_and_momentum["take_rate"])
    revenue_and_momentum["revenue_score"] = revenue_and_momentum["adj_revenue"] / max(
        revenue_and_momentum["adj_revenue"])

    rankings = rankings.merge(revenue_and_momentum[["merchant_abn", "momentum_score", "revenue_score"]], how="left", on="merchant_abn")

    # add customer quality score
    census["Median_age_persons"] = (census["Median_age_persons"] - census["Median_age_persons"].mean()) / census[
        "Median_age_persons"].std()
    mean = census["Median_age_persons"].mean()

    census_consumers = consumers.merge(census)

    def get_gender_nill_income(x):
        return {"Undisclosed": 0.5 * (x["nill_income_percent_M"] + x["nill_income_percent_F"]),
                "Female": x["nill_income_percent_F"], "Male": x["nill_income_percent_M"]}[x["gender"]]

    census_consumers["nill_income_percent"] = census_consumers[
        ["gender", "nill_income_percent_M", "nill_income_percent_F"]].apply(get_gender_nill_income, axis=1)

    census_consumers = census_consumers[
        ["house_repay_to_income", "Median_age_persons", "postcode", "user_id", "high_income_proportion",
         "nill_income_percent"]]

    census_consumers["age_factor"] = census_consumers["Median_age_persons"].apply(lambda x: sigmoid(x - mean))
    census_consumers = census_consumers.drop("Median_age_persons", axis=1)

    census_consumers["nill_income_percent"] = census_consumers["nill_income_percent"].clip(upper=0.15) / 0.15

    customer_qualities = transactions.merge(merchants, on="merchant_abn")[["merchant_abn", "user_id"]].merge(
        census_consumers).drop(["user_id", "postcode"], axis=1)
    merchant_customer_qualities = customer_qualities.groupby("merchant_abn").mean().reset_index()

    merchant_customer_qualities["customer_quality"] = 0.1 * (
                1 - merchant_customer_qualities["nill_income_percent"]) + 0.25 * (1 - merchant_customer_qualities[
        "house_repay_to_income"]) + 0.4 * merchant_customer_qualities["age_factor"] + 0.35 * (
                                                                  1 - merchant_customer_qualities[
                                                              "high_income_proportion"])
    rankings["customer_quality"] = merchant_customer_qualities["customer_quality"]

    rankings["score"] = 0.6 * rankings["revenue_score"] + 0.15 * rankings["momentum_score"] + 0.15 * rankings["retention"] + 0.1 * rankings["customer_quality"]
    rankings = rankings.sort_values("score", ascending=False).reset_index(drop=True)

    return rankings


def output_overall(rankings, path, n=100):
    rankings = rankings.sort_values("score", ascending=False).head(n)
    rankings.to_csv(path, index=False)


def output_groupings(rankings, dir, groupings, n=10):
    for group in set(groupings.values()):
        tags_list = [i for i in groupings.keys() if groupings[i] == group]
        group_rankings = rankings[rankings["sector_tags"].isin(tags_list)].sort_values("score", ascending=False).head(n)
        group_rankings.to_csv(Path(dir, group + "_rankings.csv"), index=False)


def output(root_dir, model_config, data_dir, data_config):
    print("Start Model")

    merchant_groupings = json.load(open(Path(root_dir, model_config["merchant_groupings"])))
    data_output_dir = Path(data_dir, data_config["output_dir"]).resolve()
    output_dir = Path(root_dir, model_config["output_dir"]).resolve()
    # merged_transactions = pd.read_parquet(Path(data_output_dir, "merged_transactions.parquet"))
    merchants = pd.read_parquet(Path(data_output_dir, "merchants.parquet"))
    transactions = pd.read_parquet(Path(data_output_dir, "transactions.parquet"))
    consumers = pd.read_parquet(Path(data_output_dir, "consumers.parquet"))
    census = pd.read_csv(Path(data_output_dir, "census.csv"))

    rankings = get_rankings(merchants, transactions, consumers, census)
    output_overall(rankings, Path(output_dir, "rankings.csv"))
    output_groupings(rankings, output_dir, merchant_groupings)

    print("Done")
