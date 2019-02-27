#author Simon Kleine
import os

import torch
import torch.nn as nn
import torch.optim as optim
import accelerometerfeatures.utils.pytorch.dataset as dataset
import numpy as np
from argparse import ArgumentParser

import simplecnn.target_label_to_number

class ConvolutionalNeuralNetwork (nn.Module):

    # inputlayer in the paper has 130 sampling points, remember for
    # windowsize in DataLoader
    # paper doesn't specify which kind of pooling is used
    def __init__(self):
        super(ConvoloutinalNeuralNetwork.self).__init__()
        self.firstlayer = nn.Sequential(nn.Conv1d(18, 1, 7),
                                        nn.nn.MaxPool1d(kernel_size=2))
        self.secondlayer = nn.Sequential(nn.Conv1d(18, 1, 7),
                                        nn.nn.MaxPool1d(kernel_size=2))
        self.thirdlayer = nn.Linear(114, 7)

    def forward(self, x):
        x = self.firstlayer(x)
        x = self.secondlayer(x)
        x = self.thirdlayer(x)
        return x


if __name__ == '__main__':
    argparser = ArgumentParser()
    argparser.add_argument('training_data_file_path')
    args = argparser.parse_args()

    print("Loading Dataset...")
    data = dataset.AccelerometerDatasetLoader(
        args.training_data_file_path, perform_interpolation=True)
