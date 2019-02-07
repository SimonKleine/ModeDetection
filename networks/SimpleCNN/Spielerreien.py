#author Simon Kleine

import torch
import torch.nn as nn
import torch.optim as optim
import numpy
from torch.autograd import Variable

import accelerometerfeatures.utils.pytorch.dataset as dataset

import targetlabel_to_number

EPOCH = 1

class Network (nn.Module):

    def __init__(self):
        super(Network, self).__init__()
        self.firstLayer = nn.Sequential(nn.Conv1d(3, 1, 20),
                                        nn.ReLU(),
                                        nn.MaxPool1d(kernel_size=2))
        self.secondLayer = nn.Linear(230, 7)

    def forward(self, x):
        x = self.firstLayer(x)
        x = self.secondLayer(x)

        return x


data = dataset.AccelerometerDatasetLoader(
    "/home/simonk/Documents/Bachelorarbeit"
    "/data/accelerometer_data/1st_study.csv",
    perform_interpolation=True)
train_windows = data.get_dataset_for_users(
    ["a526f3566e9c9024dfa7378eb4291d787a09fd37"])
train_windows_no_label = torch.Tensor(
    [window[0] for window in train_windows])
targetlabel_to_number = targetlabel_to_number.targetlabel_to_number()
target_matrix = targetlabel_to_number.getTargetMatrix(train_windows)

cnn = Network()
optimizer = optim.Adam(cnn.parameters(), lr=0.1)
loss_func = nn.CrossEntropyLoss()

list = [1, train_windows_no_label[0]]
input = train_windows_no_label[0].unsqueeze(0)



output = cnn(input)
print(output)
print(output[0])
print()