#!bin/sh

# Define some variables
export DATA_PATH=$HOME/MeccanoDataset/RGB_frames
export WORKSPACE_PATH=$(ws_find first_exp)
export CODE_PATH=$HOME/code

# first copy the data from the home directory to the scratch directory or temporary directory
cp -r $DATA_PATH $TMPDIR 

echo "Data is copied to the temporary directory"

# second copy the code from the home directory to the scratch directory or temporary directory
cp -r $CODE_PATH $WORKSPACE_PATH

echo "Code is copied to the workspace directory"

# third run the code

## create a virtual environment
python3 -m venv $WORKSPACE_PATH/venv

## install dependencies
source $WORKSPACE_PATH/venv/bin/activate
pip install -r $WORKSPACE_PATH/code/requirements.txt

echo "Dependencies are installed"

## run the code with specific parameters
python $WORKSPACE_PATH/code/main.py --training_type train --config $WORKSPACE_PATH/code/config/train/mobilenetbaseline.yaml --path_to_data_dir $TMPDIR/RGB_frames