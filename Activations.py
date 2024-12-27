import numpy as np

def relu(z):
    return np.maximum(0, z)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def softmax(z):
    z_stable = z - np.max(z, axis=1, keepdims=True)

    exp_z = np.exp(z_stable)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def d_relu(z, dJ_da):
    return np.where(z > 0, 1, 0) * dJ_da

def d_softmax(a, y):
    m = a.shape[0]

    y_true_indices = np.zeros(a.shape)
    y_true_indices[np.arange(m), y.flatten()] = 1

    return (a - y_true_indices) / m

#todo: learn advanced indexing to finish this
def dJ_softmax(a, y):
    m = a.shape[0]
    y_temp = y.flatten()

    y_true = np.zeros(a.shape)
    y_true[np.arange(m), y_temp] = 1

    return -y_true / (a * m)

def softmax_output(a, y):
    m = a.shape[0]
    max_indices = np.argmax(a, axis=1)
    y_true_indices = np.zeros(a.shape)
    y_true_indices[np.arange(m), max_indices] = 1

    return y_true_indices

def dJ_relu(z):
    return 0