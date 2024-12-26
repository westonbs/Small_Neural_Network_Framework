import numpy as np

def relu(z):
    return np.maximum(0, z)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def softmax(z):
    return np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)

def d_relu(z):
    return np.where(z > 0, 1, 0)

def d_softmax(a):
    m = a.shape[0]
    max_indices = np.argmax(a, axis=1)
    y_true_indices = np.zeros(a.shape)
    y_true_indices[np.arange(m), max_indices] = 1

    return (a - y_true_indices) / m

def dJ_softmax(a):
    m = a.shape[0]
    max_indices = np.argmax(a, axis=1)

    #a[np.arange(m), max_indices] extracts the max values in each row of a
    return -1 / (a[np.arange(m), max_indices] * m)