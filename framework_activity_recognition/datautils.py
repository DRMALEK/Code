import torch
import torchvision
import torchvision.transforms as transforms
import os
from framework_activity_recognition.dataset import MeccanoDataset

def prepare_drivenact(config_file):
    """
    Construct Drive&Act train and validation Dataset instance based on the given configuration file
    Arguments:
        config_file: configuration file to construct Dataset instance
    """
    dataset_train = MeccanoDataset(config_file, "train")

    dataset_val = MeccanoDataset(config_file, "val")

    dataset_test = MeccanoDataset(config_file, "test")

    return dataset_train, dataset_test

