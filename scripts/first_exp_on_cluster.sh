#!bin/sh

# define some variables
export DATA_PATH=/home/st/st_us-053030/st_st181130/MeccanoDataset/RGB_frames
export WORKSPACE_PATH=$(ws_find first_exp)
export CODE_PATH=$HOME/code

# first copy the data from the home directory to the scratch directory or temporary directory
cp -r $DATA_PATH $TMPDIR 

# second copy the code from the home directory to the scratch directory or temporary directory
cp -r $CODE_PATH $WORKSPACE_PATH

# third run the code

## create a virtual environment
python3 -m venv $WORKSPACE_PATH/venv

## install dependencies
source $WORKSPACE_PATH/venv/bin/activate
pip install -r $WORKSPACE_PATH/code/requirements.txt

## run the code with specific parameters
python $WORKSPACE_PATH/code/main.py --training_type train --config $WORKSPACE_PATH/code/config/train/config.yaml
