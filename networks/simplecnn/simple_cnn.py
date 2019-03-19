# author Simon Kleine

from argparse import ArgumentParser

import accelerometerfeatures.utils.pytorch.dataset as dataset
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import target_label_to_number

EPOCH = 3


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.first_layer = nn.Sequential(nn.Conv1d(3, 1, 20),
                                         nn.ReLU(),
                                         nn.MaxPool1d(kernel_size=2))
        self.second_layer = nn.Linear(230, 7)

    def forward(self, x):
        x = self.first_layer(x)
        # print(x.shape)
        x = self.second_layer(x)
        # print(x.shape)

        return x


def accuracy(cnn, users_valid):
    print("Calculating accuracy..")
    train_windows = data.get_dataset_for_users(users_valid)
    train_windows_no_label = torch.Tensor(
        [window[0] for window in train_windows])
    target_matrix_1d = target_label_to_number.get_target_matrix_1d(
        train_windows)
    output_list = []
    for step, input in enumerate(train_windows_no_label):
        output = cnn(input.unsqueeze(0))
        output = output.detach()
        output_list.append(np.argmax(output))
    output_list = np.array(output_list)
    target_matrix_1d = np.array(target_matrix_1d)
    print(output_list)
    print(target_matrix_1d)
    same = sum(output_list == target_matrix_1d)
    not_same = sum(output_list != target_matrix_1d)
    acc = same / (same + not_same) * 100

    return acc


if __name__ == '__main__':
    argparser = ArgumentParser()
    argparser.add_argument('training_data_file_path')
    args = argparser.parse_args()

    print("Loading Dataset...")
    data = dataset.AccelerometerDatasetLoader(
        args.training_data_file_path, perform_interpolation=True)

    users = data.users
    users_train = users[0: len(users)-2]
    users_valid = users[len(users)-2: len(users)]

    print("Creating training windows..")
    train_windows = data.get_dataset_for_users(users_train)
    train_windows_no_label = torch.Tensor(
        [window[0] for window in train_windows])
    target_matrix_1d = target_label_to_number.get_target_matrix_1d(
                          train_windows)

    cnn = Network()
    # if os.path.isfile("cnn.pt"):
    #    cnn = torch.load("cnn.pt")
    optimizer = optim.Adam(cnn.parameters(), lr=0.01)
    loss_func = nn.CrossEntropyLoss()
    for epoch in range(EPOCH):
        print("Training in progress(Epoch:", epoch, ")..")
        for step, input in enumerate(train_windows_no_label):
            output = cnn(input.unsqueeze(0))
            loss = loss_func(output[0],
                             target_matrix_1d[step].unsqueeze(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    torch.save(cnn, "cnn.pt")
    print("accuracy= ", accuracy(cnn, users_valid))
