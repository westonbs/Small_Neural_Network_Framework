import numpy as np
from tqdm import tqdm

class NeuralNetwork:
    """"
    Class representing a neural network with multiple dense layers

    Attributes:
        layers (list): list of layers in the network
        batch_size (int): size of the batch used for training
        layers_weights (list): list of weights of each layer
        layers_bias (list): list of bias of each layer

    Methods:
        predict(inputs): predicts output of the network for given input
        train(inputs, targets, epochs, learning_rate): trains the network for given number of epochs
    """
    def __init__(self, layers, batch_size):
        self.layers = layers
        self.batch_size = batch_size
        self.layers_weights = []
        self.layers_bias = []
        for i in range(len(layers) - 1):
            self.layers_weights.append(layers[i].weights)
            self.layers_bias.append(layers[i].bias)
            layers[i].next = layers[i + 1]
            layers[i].next.prev = layers[i]

    def predict(self, inputs):
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def train(self, inputs, targets, epochs, learning_rate):
        if not isinstance(inputs, np.ndarray):
            inputs = np.array(inputs)

        m, n = inputs.shape
        iterations = (m + self.batch_size - 1) // self.batch_size

        for epoch in range(epochs):
            start_index, end_index = 0, self.batch_size

            with tqdm(total=iterations, desc=f"Epoch {epoch + 1}/{epochs}") as pbar:
                for _ in range(iterations):
                    cache = {}
                    forward_inputs = inputs[start_index:end_index]
                    for layer in self.layers:
                        forward_inputs = layer.forward(forward_inputs, cache)

                    prev_layer, dJ_dz = None, None
                    dJ_dw_k, dJ_db_k = [], []
                    for layer in reversed(self.layers):
                        dJ_dz, dJ_dw, dJ_db = layer.backward(inputs[start_index:end_index], cache,
                                                             dJ_dz, prev_layer)
                        dJ_dw_k.append(dJ_dw)
                        dJ_db_k.append(dJ_db)
                        prev_layer = layer

                    for layer, dJ_dw, dJ_db in zip(self.layers, reversed(dJ_dw_k),
                                                   reversed(dJ_db_k)):
                        layer.weights -= learning_rate * dJ_dw
                        layer.bias -= learning_rate * dJ_db

                    start_index += self.batch_size
                    end_index = min(end_index + self.batch_size, m)
