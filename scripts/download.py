from urllib.request import urlretrieve
import os
from pathlib import Path
import zipfile
import shutil

# todo: add download census data

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


# todo: refactor to be more robust and extract in temp folder eg if census_dir = test/census
# code pulled from external_data.ipynb
def download_census(data_dir, census_dir):
    print("Begin census")

    url = 'https://www.abs.gov.au/census/find-census-data/datapacks/download/2021_GCP_POA_for_AUS_short-header.zip'
    urlretrieve(url, Path(data_dir, 'pop_zip'))

    with zipfile.ZipFile(Path(data_dir, "pop_zip"), 'r') as zip_ref:
        zip_ref.extractall(data_dir)

    shutil.rmtree(Path(data_dir, "Metadata"))
    shutil.rmtree(Path(data_dir, "Readme"))

    census_path = Path(data_dir, census_dir)
    if census_path.is_dir():
        shutil.rmtree(census_path)

    os.rename(Path(data_dir, "2021 Census GCP Postal Areas for AUS"), Path(data_dir, census_path))

    print("Completed census")


def download(output_directory, census_dir):
    # Download Population Data
    # download_csv("pop", {"pop.csv": 'https://www.abs.gov.au/statistics/people/population/regional-population-age-and-sex/2021/32350DS0005_2001-21.xlsx'}, output_directory)
    download_census(output_directory, census_dir)