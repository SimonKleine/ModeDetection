#author Simon Kleine
import os

import torch
import torch.nn as nn
import torch.optim as optim
import accelerometerfeatures.utils.pytorch.dataset as dataset
import numpy as np
from argparse import ArgumentParser
#import networks.simplecnn as simplecnn
import target_label_to_number
import random


class ConvolutionalNeuralNetwork (nn.Module):

    # inputlayer in the paper has 130 sampling points, remember for
    # windowsize in DataLoader
    # paper doesn't specify which kind of pooling is used
    def __init__(self):
        super(ConvolutionalNeuralNetwork, self).__init__()
        '''
        self.firstlayer = nn.Sequential(nn.Conv1d(3, 18, 7),
                                        nn.MaxPool1d(kernel_size=2))
        self.secondlayer = nn.Sequential(nn.Conv1d(18, 324, 7),
                                        nn.MaxPool1d(kernel_size=2))
        self.thirdlayer = nn.Linear(37260, 7)
        '''
        self.firtconvolutionlayer = nn.Conv1d(3, 18, 3)
        self.firstactivationlayer = nn.ReLU()
        self.firstpoolinglayer = nn.LPPool1d(7, kernel_size=2)
        self.secondconvolutionlayer = nn.Conv1d(18, 324, 3)
        self.secondactivationlayer = nn.ReLU()
        self.secondpoolinglayer = nn.LPPool1d(7, kernel_size=2)
        self.firstlinearlayer = nn.Linear(38232, 5)

    def forward(self, x):
        '''
        x = self.firstlayer(x)
        x = self.secondlayer(x)
        print(x.shape)
        x = self.thirdlayer(x)
        return x
        '''
        x = self.firtconvolutionlayer(x)
        x = self.firstactivationlayer(x)
        x = self.firstpoolinglayer(x)
        x = self.secondconvolutionlayer(x)
        x = self.secondactivationlayer(x)
        x = self.secondpoolinglayer(x)
        x = x.view(1, 38232)
        x = self.firstlinearlayer(x)
        return x

def get_accuracy(cnn, target_matrix_1d, train_windows_no_label):
    print("Calculating accuracy..")
    output_list = []
    for step, input in enumerate(train_windows_no_label):
        input = input.cuda()
        output = cnn(input.unsqueeze(0))
        output = output.detach()
        output = output.cpu()
        output_list.append(np.argmax(output))
    output_list = np.array(output_list)
    target_matrix_1d = np.array(target_matrix_1d)
    same = sum(output_list == target_matrix_1d)
    not_same = sum(output_list != target_matrix_1d)
    acc = same / (same + not_same) * 100

    return acc


if __name__ == '__main__':
    random.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    EPOCH = 50
    overall_accuracy_list = []
    argparser = ArgumentParser()
    argparser.add_argument('training_data_file_path')
    args = argparser.parse_args()

    print("Loading Dataset...")
    data = dataset.AccelerometerDatasetLoader(
        args.training_data_file_path, perform_interpolation=True)

    users = data.users
    logfile = open("logfilecnn_epoch=50.txt", "w")
    for current_user in users:
        users_train = users.copy()
        users_train.remove(current_user)
        users_valid = [current_user]
        print("User ", (len(overall_accuracy_list) + 1),
              " von ", len(users_train))

        print("Creating training windows..")
        train_windows = data.get_dataset_for_users(users_train)
        valid_windows = data.get_dataset_for_users(users_valid)
        train_windows_no_label = torch.Tensor(
            [window[0] for window in train_windows])
        valid_windows_no_label = torch.Tensor(
            [window[0] for window in valid_windows])
        target_matrix_1d = \
            target_label_to_number.get_target_matrix_1d(
            train_windows)
        valid_target_matrix_1d = \
            target_label_to_number.get_target_matrix_1d(
                valid_windows)
        cnn = ConvolutionalNeuralNetwork()
        # if os.path.isfile("cnn.pt"):
        #    cnn = torch.load("cnn.pt")
        cnn.cuda()
        optimizer = optim.Adam(cnn.parameters(), lr=0.0005)
        loss_func = nn.CrossEntropyLoss()
        for epoch in range(EPOCH):
            print("Training in progress(Epoch:", epoch + 1, "/", EPOCH, ")..")
            for step, input in enumerate(train_windows_no_label):
                input = input.cuda()
                target_matrix_1d = target_matrix_1d.cuda()
                output = cnn(input.unsqueeze(0))
                loss = loss_func(output[0].unsqueeze(0),
                                 target_matrix_1d[step].unsqueeze(0))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        #file_name_network = "cnn."
        #file_name_network = file_name_network.__add__(current_user)
        #file_name_network = file_name_network.__add__(".pt")
        #torch.save(cnn, file_name_network)
        if len(valid_target_matrix_1d) == 0:
            continue
        if len(valid_windows_no_label) == 0:
            continue
        accuracy = get_accuracy(cnn, valid_target_matrix_1d,
                            valid_windows_no_label)
        string_for_logfile = "User: "
        string_for_logfile = string_for_logfile.__add__(current_user)
        string_for_logfile = string_for_logfile.__add__(", Accuracy: ")
        string_for_logfile = string_for_logfile.__add__(str(accuracy))
        string_for_logfile = string_for_logfile.__add__("\n")


        logfile.write(string_for_logfile)
        overall_accuracy_list.append(accuracy)
    overall_accuracy = sum(overall_accuracy_list) / len(overall_accuracy_list)
    final_string = "Average Accuracy"
    final_string = final_string.__add__(str(overall_accuracy))
    logfile.write(final_string)



