import argparse
from framework_activity_recognition.io import load_config_file
from framework_activity_recognition.driver import train, test_benchmark
from framework_activity_recognition.datautils import prepare_drivenact
from framework_activity_recognition.dataset import MeccanoDataset
import torch

def main():
    #parser = argparse.ArgumentParser("Training framework")
    #parser.add_argument("training_type", type=str, choices={"train", "test"}, help="choose training type")
    #parser.add_argument("config_path", type=str, help="path to the yaml configuration file")
    #parser.add_argument("--checkpoint", type=str, help="path to the .pt file to continue training.\
    #    This will be ignored if pretrained model is used")
    #parser.add_argument("--output", type=str, help="path to save checkpoints")
    #args = parser.parse_args()

    #config_path = "config/train/mobilenetbaseline.yaml"
    #config = load_config_file(config_path)
    
    #MeccanoDatasetTrain = MeccanoDataset(config, mode="test")

    #frames, label, index, _ = MeccanoDatasetTrain.__getitem__(4)
    
    
    state_dict = torch.load('/home/malek/Master Thesis/I3D_8x8_R50_MECCANO.pyth', map_location=torch.device('cpu'))
    
    for k, v in state_dict["model_state"].items():
        print(k)
    
    #train(config)
    
    #if args.training_type == "test":
    #    test_benchmark(config)

    # load the model with config file
    # load the trained weights
    # test the model 




if __name__ == "__main__":
    main()