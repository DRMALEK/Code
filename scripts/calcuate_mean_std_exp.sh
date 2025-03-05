#!/bin/bash

# Define some variables
export WORKSPACE_PATH=$(ws_find first_exp)
export DATA_PATH=$(ws_find first_exp)/MecanoDataset

## run the code with specific parameters
python $WORKSPACE_PATH/code/scripts/calculate_mean_std_of_dataset.py --data_path $DATA_PATH --output_path $WORKSPACE_PATH/mean_std.txt
