#author Simon Kleine

import torch
import torch.nn as nn
import torch.optim as optim

import accelerometerfeatures.utils.pytorch.dataset as dataset

import targetlabel_to_number


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
    "/data/accelerometer_data/1st_study.csv", perform_interpolation=True)
data_windows_user1 = data.get_dataset_for_users(["a526f3566e9c9024dfa7378eb4291d787a09fd37"])
tensor_data = torch.Tensor([window[0] for window in data_windows_user1])
network = Network()
network(tensor_data)
test = targetlabel_to_number.targetlabel_to_number()
test1 = test.getTargetMatrix(data_windows_user1)
print(test1)