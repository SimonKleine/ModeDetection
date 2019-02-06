# author Simon Kleine

import torch
from accelerometerfeatures.utils.pytorch.dataset import AccelerometerDataset



class targetlabel_to_number(object):


    # This method takes in an AccelerometerDataset and returns the
    # label of each window as a number between 0 and 6 in form of a
    # list.
    # These numbers represent then the transportation mode used in
    # each window of the AccelerometerDataset.
    def targetlabel_to_number (self, AccelerometerDataset):
        labels_as_numbers = []

        for window in AccelerometerDataset:
            labels_as_numbers.append(
                targetlabel_to_number.getLabelNumber(self, window))

        return torch.Tensor(labels_as_numbers)

    # This method assigns to each label of a window a number.
    def getLabelNumber(self, window):
        mode_list = ["bike", "car", "walk", "bus", "train",
                     "metro", "tram"]
        current_label_number = 0

        for mode in enumerate(mode_list):
            if window[1] == mode[1]:
                current_label_number = mode[0]

        return current_label_number

    # This method takes in an AccelerometerDataset and returns a
    # targetmatrix corresponding to the windows in that set.
    def getTargetMatrix(self, AccelerometerDataset):
        target_List = targetlabel_to_number.targetlabel_to_number(self,
            AccelerometerDataset)
        targetMatrix = []
        possible_Targetnumbers = [0, 1, 2, 3, 4, 5, 6]

        for element in target_List:
            list = []
            for targetnumber in enumerate(possible_Targetnumbers):
                if targetnumber[1] == element:
                    list.append(1)
                else:
                    list.append(0)
            targetMatrix.append(list)

        return targetMatrix