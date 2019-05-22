# author Simon Kleine

import torch


def target_label_to_number(accelerometer_dataset):
    """
    This method takes in an AccelerometerDataset and returns the label of each
    window as a number between 0 and 6 in form of a list.
    These numbers represent then the transportation mode used in each window of
    the AccelerometerDataset.
    """
    labels_as_numbers = []

    for window in accelerometer_dataset:
        labels_as_numbers.append(get_label_number(window))

    return torch.Tensor(labels_as_numbers)


def get_label_number(window):
    """This method assigns to each label of a window a number."""
    mode_list = ["bike", "car", "walk", "bus", "train"]
    current_label_number = 0

    for mode in enumerate(mode_list):
        if window[1] == mode[1]:
            current_label_number = mode[0]

    return current_label_number


def get_target_matrix(accelerometer_dataset):
    """
    This method takes in an AccelerometerDataset and returns a target matrix
    (for example [0, 0, 1, 0, 0, 0]) corresponding to the windows in that
    set.
    """
    target_list = target_label_to_number(accelerometer_dataset)
    target_matrix = []
    possible_target_numbers = [0, 1, 2, 3, 4]

    for element in target_list:
        target_vector = []
        for target_number in enumerate(possible_target_numbers):
            if target_number[1] == element:
                target_vector.append(1)
            else:
                target_vector.append(0)
            assert len(target_vector) == len(possible_target_numbers)
        target_matrix.append(target_vector)

    return target_matrix


def get_target_matrix_1d(accelerometer_dataset):
    target_matrix_1d = target_label_to_number(accelerometer_dataset).long()
    return target_matrix_1d
