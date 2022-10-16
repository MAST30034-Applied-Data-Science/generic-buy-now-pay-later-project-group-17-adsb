from urllib.request import urlretrieve
import os
from pathlib import Path
import zipfile
import shutil


def download_census(data_dir, census_dir):
    """Download census data and unzip

    Args:
        data_dir (PosixPath): path to data directory
        census_dir (PosixPath): path to output census data to
    """    
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
    """Download external datasets

    Args:
        output_directory (PosixPath): directory to output ext datasets to
        census_dir (PosixPath): path to output census data to
    """    
    # Download Census Data
    download_census(output_directory, census_dir)