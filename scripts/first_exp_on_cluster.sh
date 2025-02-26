#!bin/bash

# Define some variables
export DATA_PATH=$HOME/MeccanoDataset/RGB_frames
export WORKSPACE_PATH=$(ws_find first_exp)
export CODE_PATH=$HOME/code

# first copy the data from the home directory to the scratch directory or temporary directory
cp -r $DATA_PATH $TMPDIR 

echo "Data is copied to the temporary directory"

## run the code with specific parameters
python $WORKSPACE_PATH/code/main.py --training_type train --config $WORKSPACE_PATH/code/config/train/mobilenetbaseline.yaml --path_to_data_dir $TMPDIR/RGB_frames