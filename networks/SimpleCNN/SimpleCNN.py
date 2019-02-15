#author Simon Kleine
import os

import torch
import torch.nn as nn
import torch.optim as optim
import accelerometerfeatures.utils.pytorch.dataset as dataset
import numpy as np
from argparse import ArgumentParser

import targetlabel_to_number

EPOCH = 3

class Network (nn.Module):

    def __init__(self):
        super(Network, self).__init__()
        self.firstLayer = nn.Sequential(nn.Conv1d(3, 1, 20),
                                        nn.ReLU(),
                                        nn.MaxPool1d(kernel_size=2))
        self.secondLayer = nn.Linear(230, 7)

    def forward(self, x):
        x = self.firstLayer(x)
        #print(x.shape)
        x = self.secondLayer(x)
        #print(x.shape)

        return x
        print(np.argmax(output))

def accuracy(cnn, users_valid):
    print("Calculating accuracy..")
    train_windows = data.get_dataset_for_users(users_valid)
    train_windows_no_label = torch.Tensor(
        [window[0] for window in train_windows])
    target_matrix1d = targetlabel_to_number.getTargetMatrix1d(
        train_windows)
    output_list = []
    for step, input in enumerate(train_windows_no_label):
        output = cnn(input.unsqueeze(0))
        output = output.detach()
        output_list.append(np.argmax(output))
    output_list = np.array(output_list)
    target_matrix1d = np.array(target_matrix1d)
    same = sum(output_list == target_matrix1d)
    not_same = sum(output_list != target_matrix1d)
    acc = same / (same + not_same) * 100

    return acc


argparser = ArgumentParser()
argparser.add_argument('training_data_file_path')
args = argparser.parse_args()
print(args)
print("Loading Dataset..")
data = dataset.AccelerometerDatasetLoader(args.training_data_file_path,
                                          perform_interpolation=True)
users = data.users
users_train = users[0:len(users)-2]
users_valid = users[len(users)-2: len(users)]

print("Creating training windows..")
train_windows = data.get_dataset_for_users(users_train)
train_windows_no_label = torch.Tensor(
    [window[0] for window in train_windows])
targetlabel_to_number = targetlabel_to_number.targetlabel_to_number()
target_matrix1d = targetlabel_to_number.getTargetMatrix1d(
                      train_windows)

cnn = Network()
#if os.path.isfile("cnn.pt"):
#    cnn = torch.load("cnn.pt")
optimizer = optim.Adam(cnn.parameters(), lr=0.01)
loss_func = nn.CrossEntropyLoss()
for epoch in range(EPOCH):
    print("Training in progress(Epoch:", epoch, ")..")
    for step, input in enumerate(train_windows_no_label):
       output = cnn(input.unsqueeze(0))
       loss = loss_func(output[0], target_matrix1d[step].unsqueeze(0))
       optimizer.zero_grad()
       loss.backward()
       optimizer.step()

torch.save(cnn, "cnn.pt")
print("accuracy= ", accuracy(cnn, users_valid))
