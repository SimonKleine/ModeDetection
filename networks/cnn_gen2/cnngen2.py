#author Simon Kleine
import os

import torch
import torch.nn as nn
import torch.optim as optim
import accelerometerfeatures.utils.pytorch.dataset as dataset
import numpy as np
import random
from argparse import ArgumentParser
#import networks.simplecnn as simplecnn
import target_label_to_number


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
        self.firtconvolutionlayer = nn.Conv1d(3, 18, 4)
        self.firstpoolinglayer = nn.AvgPool1d(kernel_size=3)
        self.secondconvolutionlayer = nn.Conv1d(18, 324, 4)
        self.secondpoolinglayer = nn.AvgPool1d(kernel_size=3)
        self.firstlinearlayer = nn.Linear(16848, 7)
     


    def forward(self, x):
        '''
        x = self.firstlayer(x)
        x = self.secondlayer(x)
        print(x.shape)
        x = self.thirdlayer(x)
        return x
        '''
        x = self.firtconvolutionlayer(x)
        x = self.firstpoolinglayer(x)
        x = self.secondconvolutionlayer(x)
        x = self.secondpoolinglayer(x)
        x = x.view(1, 16848)
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

def shuffle(train_windows, target_matrix):
    print("shuffling")
    head_train = []
    tail_train = []
    head_target = []
    tail_target = []
    input_train = train_windows
    input_target = target_matrix
    for x in range (1, 100):
        splitpoint1 = random.randint(1, len(train_windows))
        splitpoint2 = random.randint(splitpoint1, len(train_windows))
        head_train = input_train[:splitpoint1]
        middle_train = input_train[splitpoint1:splitpoint2]
        tail_train = input_train[splitpoint2:]
        input_train = torch.cat((middle_train, head_train))
        input_train = torch.cat((input_train, tail_train))
        head_target = input_target[:splitpoint1]
        middle_target = input_target[splitpoint1:splitpoint2]
        tail_target = input_target[splitpoint2:]
        input_target = torch.cat(((middle_target, head_target)))
        input_target =  torch.cat((input_target, tail_target))


        return input_train, input_target




if __name__ == '__main__':
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
        optimizer = optim.Adam(cnn.parameters(), lr=0.1)
        loss_func = nn.CrossEntropyLoss()
        for epoch in range(EPOCH):
            print("Training in progress(Epoch:", epoch + 1, "/", EPOCH, ")..")
            #shufflearray = shuffle(train_windows_no_label, target_matrix_1d)
            #train_windows_no_label = shufflearray[0]
            #target_matrix_1d = shufflearray[1]
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

