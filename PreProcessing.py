import math

import kagglehub
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from NeuralNetwork import NeuralNetwork
from Layer import Layer
import Activations

def download_data():
    path = kagglehub.dataset_download("sachinpatel21/az-handwritten-alphabets-in-csv-format")
    data_csv = pd.read_csv(f"{path}/handWritten.csv")
    data = data_csv.to_numpy()

    np.random.shuffle(data)
    Y_train = data[:, 0].reshape(1, -1)
    X_train = data[:, 1:] / 255.0

    save_data(Y_train, X_train)

def generate_minority_data(num_classes):
    Y_train, X_train = load_data()
    data = np.hstack((Y_train, X_train))
    m = data.shape[0]
    class_data = [[] for _ in range(num_classes)]

    for i in range(m):
        class_type = data[i, 0]
        class_data[class_type].append(data[i, :])

    max_count, max_ind = 0
    for i in range(num_classes):
        if len(class_data[i]) > max_count:
            max_count = len(class_data[i])
            max_ind = i

    copy_count = [[] for _ in range(num_classes)]
    new_data = np.vstack(class_data[max_ind])
    for i in range(num_classes):
        if i == max_ind:
            continue

        count = max_count / len(class_data)
        if count % 1 > 0.5:
            count = math.ceil(count)
        else:
            count = int(count)

        curr_class = np.vstack(class_data[i])
        new_data = np.vstack((new_data, curr_class))
        for j in range(count - 1):
            curr_class_copy = np.copy(curr_class)
            new_data = np.vstack((new_data, curr_class_copy))

    Y_train = new_data[:, 0].reshape(1, -1)
    X_train = new_data[:, 1:]

    save_data(Y_train, X_train)

def shuffle_data(Y_train, X_train):
    data = np.hstack((Y_train, X_train))
    np.random.shuffle(data)
    Y_train = data[:, 0].reshape(1, -1)
    X_train = data[:, 1:] / 255.0
    return Y_train, X_train

def save_data(Y_train, X_train):
    np.save("Y_train.npy", Y_train)
    np.save("X_train.npy", X_train)

def load_data():
    Y = np.load("Y_train.npy")
    X = np.load("X_train.npy")
    return Y, X

