from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats

# sample customers using a truncated geometric distribution
# todo: make time_scale param work
def sample_customers(n, transactions, merchant_abn, p=0.5, time_scale="month"):
    def month_diff(d1, d2):
        return (d1.year - d2.year) * 12 + d1.month - d2.month

    # truncated geometric distribution pdf
    def truncgeom_pdf(max_val):
        return np.array([stats.geom.pmf(i, p) for i in range(1, max_val + 2)])/(stats.geom.cdf(max_val + 1, p))

    # filter for specific merchant
    transactions = transactions.loc[transactions["merchant_abn"] == merchant_abn]

    latest_date = max(transactions["order_datetime"])
    transactions["date_segment"] = transactions["order_datetime"].apply(lambda x: month_diff(latest_date, x))
    max_segment = max(transactions["date_segment"])

    # aggregate by customer-timescale for specific merchant
    aggregate_by = ["user_id", "date_segment"]
    aggregated_transactions = transactions.groupby(aggregate_by).count().add_suffix('_Count').reset_index()[aggregate_by]

    # segment_samples = np.floor(truncgeom_pdf(max_segment) * n)
    # segment_samples[0] = n - np.sum(segment_samples[1:])
    weights = truncgeom_pdf(max_segment)

    aggregated_transactions["weight"] = aggregated_transactions["date_segment"].apply(lambda x: weights[x])

    # # todo: iterate through each segment and sample each separetely then join? - is this better than weighted sampling? consider 1 customer in a month
    # samples_df = [aggregated_transactions.sample(n=segment_samples[i], replace=True)]

    return aggregated_transactions.sample(n=n, replace=True, weights="weight", random_state=1)["user_id"].reset_index(drop=True)

def print_sample(data_dir, data_config):
    output_dir = Path(data_dir, data_config["output_dir"]).resolve()
    transactions = pd.read_parquet(Path(output_dir, "transactions.parquet"))
    transactions["order_datetime"] = pd.to_datetime(transactions["order_datetime"])

    sampled_customers = sample_customers(10000, transactions, 28000487688, 0.4)
    print(sampled_customers)