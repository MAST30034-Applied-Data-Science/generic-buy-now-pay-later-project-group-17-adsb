#!/usr/bin/bash

working_directory="${PWD}/"
config_path="config.toml"
run_script="scripts/run.py"

python $run_script "$working_directory" $config_path