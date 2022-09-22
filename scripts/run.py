import argparse
from pathlib import Path
import os
import toml
import download
import preprocess
import model

parser = argparse.ArgumentParser()
parser.add_argument("working_directory", help="absolute path to working directory", type=Path)
parser.add_argument("config_path", help="path to config", type=Path)
args = parser.parse_args()

os.chdir(args.working_directory)
config = toml.load(args.config_path)

data_directory = config["directories"]["data"]

# download external datasets
download.download(Path(data_directory).resolve(), config["data"]["census"])

# ETL
preprocess.etl(Path(data_directory).resolve(), config["data"])

# modelling - sample testing todo: remove
model.print_sample(Path(data_directory).resolve(), config["data"])