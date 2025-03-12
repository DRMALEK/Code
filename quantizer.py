import argparse
import torch
from torch.quantization import quantize, prepare
from torch.quantization import convert


def main():
    parser = argparse.ArgumentParser("Quantization module")
    
    parser.add_argument("--model_path", type=str, help="path to the model")
    parser.add_argument("--data_path", type=str, help="path to the data directory for calibration")
    parser.add_argument("--quantization_type", type=str, help="type of quantization")

    args = parser.parse_args()

    if args.quantization_type == "static":
        # Static Quantization
        model = torch.load(args.model_path)
        model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        model = prepare(model)
        model = convert(model)
        torch.save(model, args.model_path)

    elif args.quantization_type == "dynamic":
        pass
    

if __name__=='__main__':
    pass