from urllib.request import urlretrieve
import os

# from the current notebooks directory, go back one level to the data directory
output_relative_dir = '../data/'

# check if it exists as it makedir will raise an error if it does exist
if not os.path.exists(output_relative_dir):
    os.makedirs(output_relative_dir)

# Download Population Data 
print(f"Begin population")
url = 'https://www.abs.gov.au/statistics/people/population/regional-population-age-and-sex/2021/32350DS0005_2001-21.xlsx'
pop_output_dir = output_relative_dir + 'pop' 

if not os.path.exists(pop_output_dir):
    os.makedirs(pop_output_dir)

output_dir = f"{pop_output_dir}/pop.csv"

# download
urlretrieve(url, output_dir)
print(f"Completed population")