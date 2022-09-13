from urllib.request import urlretrieve
import os
import argparse
from pathlib import Path


def download_csv(name, urls, rel_dir):
    # check if it exists as it makedir will raise an error if it does exist
    if not os.path.exists(rel_dir):
        os.makedirs(rel_dir)

    print(f"Begin {name}")
    output_dir = Path(rel_dir, name)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # download
    for name, url in urls.items():
        urlretrieve(url, Path(output_dir, name))

    print(f"Completed {name}")


def download(output_directory):
    # Download Population Data
    download_csv("pop", {"pop.csv": 'https://www.abs.gov.au/statistics/people/population/regional-population-age-and-sex/2021/32350DS0005_2001-21.xlsx'}, output_directory)