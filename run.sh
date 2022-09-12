#!/usr/bin/bash

source path.config
working_directory="${PWD}/"

python $download_script "${working_directory}${data_directory}"
python $preprocess_script "${working_directory}${transactions_directory}" "${working_directory}${merchants_file_loc}" "${working_directory}${transactions_output}"
# python scripts/run.py