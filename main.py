import argparse
from framework_activity_recognition.io import load_config_file
from framework_activity_recognition.driver import train, test_benchmark

def main():
    parser = argparse.ArgumentParser("Training framework")
    parser.add_argument("--training_type", type=str, choices={"train", "test"}, help="choose training type")
    parser.add_argument("--config_path", type=str, help="path to the yaml configuration file")
    parser.add_argument("--checkpoint", type=str, help="path to the .pt file to continue training.\
        This will be ignored if pretrained model is used")
    parser.add_argument("--data_path", type=str, help="path to the data directory")
    parser.add_argument("--pretrained_model", type=str, help="path to the pretrained model")


    args = parser.parse_args()

    config = load_config_file(args.config_path)

    config["data"]["path_to_data_dir"] = args.data_path


    if args.training_type == "train":
        config["pretraining"]["path"] = args.pretrained_model
        config["config"]["path"]= args.config_path
        train(config)

    if args.training_type == "test":
        config["architecture"]["model"]= args.pretrained_model
        test_benchmark(config)

if __name__ == "__main__":
    main()