import json
from pathlib import Path
import pandas as pd


def get_rankings(merged_transactions, merchants):
    merchants = merchants.reset_index()
    merchants["score"] = merchants.index
    return merchants


def output_overall(rankings, path, n=100):
    rankings = rankings.sort_values("score", ascending=False).head(n)
    with open(path, "w") as f:
        f.write("\n".join(str(i) for i in rankings["merchant_abn"]))


def output_groupings(rankings, dir, groupings, n=10):
    for group in set(groupings.values()):
        tags_list = [i for i in groupings.keys() if groupings[i] == group]
        group_rankings = rankings[rankings["sector_tags"].isin(tags_list)].sort_values("score", ascending=False).head(n)
        with open(Path(dir, group + "_rankings.txt"), "w") as f:
            f.write("\n".join(str(i) for i in group_rankings["merchant_abn"]))


def output(root_dir, model_config, data_dir, data_config):
    merchant_groupings = json.load(open(Path(root_dir, model_config["merchant_groupings"])))
    data_output_dir = Path(data_dir, data_config["output_dir"]).resolve()
    output_dir = Path(root_dir, model_config["output_dir"]).resolve()
    merged_transactions = pd.read_parquet(Path(data_output_dir, "merged_transactions.parquet"))
    merchants = pd.read_parquet(Path(data_output_dir, "merchants.parquet"))

    rankings = get_rankings(merged_transactions, merchants)
    output_overall(rankings, Path(output_dir, "rankings.txt"))
    output_groupings(rankings, output_dir, merchant_groupings)
