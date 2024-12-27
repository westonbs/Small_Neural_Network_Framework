import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

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
        for i in range(len(layers)):
            self.layers_weights.append(layers[i].weights)
            self.layers_bias.append(layers[i].bias)

    def save_weights_bias(self):
        weights = {}
        biases = {}
        for i in range(len(self.layers)):
            weights[f'W{i}'] = self.layers_weights[i]
            biases[f'b{i}'] = self.layers_bias[i]

        np.savez("model_parameters.npz", **weights, **biases)

    def load_weights_bias(self):
        self.layers_weights = []
        self.layers_bias = []
        loaded_params = np.load("model_parameters.npz")
        for i in range(len(self.layers)):
            self.layers_weights.append(loaded_params[f"W{i}"])
            self.layers_bias.append(loaded_params[f"b{i}"])
            self.layers[i].weights = self.layers_weights[i]
            self.layers[i].bias = self.layers_bias[i]

    def predict(self, inputs):
        for layer in self.layers:
            inputs = layer.forward(inputs, {})
        return inputs

    def test_accuracy(self, inputs, targets):
        # Predictions and true labels
        predictions = self.predict(inputs)
        m = targets.shape[0]

        # Predicted classes
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = targets.flatten()

        # Debugging shapes and values
        print(f"Predicted classes shape: {predicted_classes.shape}")
        print(f"True classes shape: {true_classes.shape}")
        print(f"First 5 Predicted: {predicted_classes[:100]}")
        print(f"First 5 True: {true_classes[:100]}")
        print(f"Middle Prediction: {predictions[200000]}")

        # Count correct predictions
        correct = np.sum(predicted_classes == true_classes)
        print(f"Correct predictions: {correct}")

        # Calculate accuracy
        accuracy = correct / m
        return accuracy

    def cost(self, inputs, targets):
        predictions = self.predict(inputs)
        m = predictions.shape[0]

        log_probs = -np.log(predictions[np.arange(m), targets.flatten()])
        cost = np.mean(log_probs)

        return cost

    def train(self, inputs, targets, epochs, learning_rate):
        if not isinstance(inputs, np.ndarray):
            inputs = np.array(inputs)

        cost_values = []
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

                    Y_train_curr_range = targets[0, start_index:end_index]
                    prev_layer, dJ_dz = None, None
                    dJ_dw_k, dJ_db_k = [], []
                    for layer in reversed(self.layers):
                        if layer.index == 0:
                            layer_inputs = inputs[start_index:end_index]
                        else:
                            layer_inputs = cache[f'a^{layer.index - 1}']

                        dJ_dz, dJ_dw, dJ_db = layer.backward(layer_inputs, Y_train_curr_range,
                                                             cache, dJ_dz, prev_layer)
                        dJ_dw_k.append(dJ_dw)
                        dJ_db_k.append(dJ_db)
                        prev_layer = layer

                    for layer, dJ_dw, dJ_db in zip(self.layers, reversed(dJ_dw_k),
                                                   reversed(dJ_db_k)):
                        layer.weights -= learning_rate * dJ_dw
                        layer.bias -= learning_rate * dJ_db

                    start_index += self.batch_size
                    end_index = min(end_index + self.batch_size, m)

                    pbar.update(1)

                cost_values.append(self.cost(inputs, targets))
        plt.figure(figsize=(8, 5))
        plt.plot(range(len(cost_values)), cost_values, label='Sparse Cross-Entropy Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Cost')
        plt.title('Cost vs. Epoch')
        plt.legend()
        plt.grid(True)
        plt.show()