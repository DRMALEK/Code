#!/bin/bash

# Define some variables
export DATA_PATH=$HOME/MecanoDatasetSample
export WORKSPACE_PATH=$(ws_find first_exp)
export CODE_PATH=$HOME/code

# first copy the data from the home directory to the scratch directory or temporary directory
#cp -r $DATA_PATH $TMPDIR 

# second copy code to the workspace
#cp -r $CODE_PATH $WORKSPACE_PATH

echo "Data is copied to the temporary directory"

## create a virtual environment
#python3 -m venv $WORKSPACE_PATH/venv

## install dependencies
source $WORKSPACE_PATH/venv/bin/activate
#pip install -r $WORKSPACE_PATH/code/requirements-3.9.txt

#echo "Dependencies are installed"

## run the code with specific parameters
python $WORKSPACE_PATH/code/main.py --training_type train --config $WORKSPACE_PATH/code/config/train/mobilenetbaseline.yaml --data_path $TMPDIR/RGB_frames
