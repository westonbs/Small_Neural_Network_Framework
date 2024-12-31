import numpy as np
import cupy as cp
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
        prev_output_shape = None
        for i in range(len(layers)):
            if i != 0:
                layers[i].set_input_shape(prev_output_shape)

            layers[i].set_index(i)

            self.layers_weights.append(layers[i].weights)
            self.layers_bias.append(layers[i].bias)

            prev_output_shape = layers[i].output_shape

    def save_weights_bias(self):
        weights = {}
        biases = {}
        for i in range(len(self.layers)):
            weights[f'W{i}'] = self.layers_weights[i]
            biases[f'b{i}'] = self.layers_bias[i]

        if isinstance(self.layers[0], np.ndarray):
            np.savez("model_parameters.npz", **weights, **biases)
        else:
            self.convert_weight_bias_type()
            np.savez("model_parameters.npz", **weights, **biases)

    def load_weights_bias(self, framework):
        self.layers_weights = []
        self.layers_bias = []
        loaded_params = np.load("model_parameters.npz")

        for i in range(len(self.layers)):
            self.layers_weights.append(loaded_params[f"W{i}"])
            self.layers_bias.append(loaded_params[f"b{i}"])
            self.layers[i].weights = self.layers_weights[i]
            self.layers[i].bias = self.layers_bias[i]

        if framework == "numpy":
            return
        elif framework == "cupy":
            self.convert_weight_bias_type()
        else:
            raise ValueError("Invalid framework. Please choose either numpy or cupy.")

    def convert_weight_bias_type(self):
        print(len(self.layers_weights), len(self.layers))
        for i in range(len(self.layers)):
            new_weights, new_bias = self.layers[i].convert_weight_bias_type()
            self.layers_weights[i] = new_weights
            self.layers_bias[i] = new_bias

    def predict(self, inputs):
        for layer in self.layers:
            inputs = layer.forward(inputs, {})
        return inputs

    def test_accuracy_numpy(self, inputs, targets):
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

    def test_accuracy_cupy(self, inputs, targets):
        # Predictions and true labels
        predictions = self.predict(inputs)
        m = targets.shape[0]

        # Predicted classes
        predicted_classes = cp.argmax(predictions, axis=1)
        true_classes = targets.flatten()

        # Debugging shapes and values
        print(f"Predicted classes shape: {predicted_classes.shape}")
        print(f"True classes shape: {true_classes.shape}")
        print(f"First 5 Predicted: {predicted_classes[:100]}")
        print(f"First 5 True: {true_classes[:100]}")
        print(f"Middle Prediction: {predictions[200000]}")

        # Count correct predictions
        correct = cp.sum(predicted_classes == true_classes)
        print(f"Correct predictions: {correct}")

        # Calculate accuracy
        accuracy = correct / m
        return accuracy

    def test_accuracy(self, inputs, targets):
        if isinstance(inputs, np.ndarray):
            return self.test_accuracy_numpy(inputs, targets)
        else:
            return self.test_accuracy_cupy(inputs, targets)

    def cost_numpy(self, inputs, targets):
        predictions = self.predict(inputs)
        m = predictions.shape[0]

        # Add numerical stability with clipping
        predictions = np.clip(predictions, 1e-8, 1.0)

        log_probs = -np.log(predictions[np.arange(m), targets.flatten().astype(int)])
        cost = np.mean(log_probs)

        return cost

    def cost_cupy(self, inputs, targets):
        predictions = self.predict(inputs)
        m = predictions.shape[0]

        # Add numerical stability with clipping
        predictions = cp.clip(predictions, 1e-8, 1.0)

        log_probs = -cp.log(predictions[cp.arange(m), targets.flatten().astype(int)])
        cost = cp.mean(log_probs)

        return cost

    def cost(self, inputs, targets):
        if isinstance(inputs, np.ndarray):
            return self.cost_numpy(inputs, targets)
        else:
            return self.cost_cupy(inputs, targets)

    def plot_cost(self, cost_values) -> None:
        processed_cost_values = [
            cv.get() if isinstance(cv, cp.ndarray) else cv for cv in cost_values
        ]
        plt.figure(figsize=(8, 5))
        plt.plot(range(len(processed_cost_values)), processed_cost_values,
                 label='Sparse Cross-Entropy Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Cost')
        plt.title('Cost vs. Epoch')
        plt.legend()
        plt.grid(True)
        plt.show()

    def compute_gradients(self, inputs, targets):
        if isinstance(inputs, cp.ndarray) and not isinstance(self.layers_weights[0], cp.ndarray):
            self.convert_weight_bias_type()
        elif isinstance(inputs, np.ndarray) and not isinstance(self.layers_weights[0], np.ndarray):
            self.convert_weight_bias_type()

        cache = {}
        forward_inputs = inputs
        for layer in self.layers:
            forward_inputs = layer.forward(forward_inputs, cache)

        prev_layer, dJ_dz = None, None
        dJ_dw_k, dJ_db_k = [], []
        for layer in reversed(self.layers):
            if layer.index == 0:
                layer_inputs = inputs
            else:
                layer_inputs = cache[f'a^{layer.index - 1}']
            dJ_dz, dJ_dw, dJ_db = layer.backward(layer_inputs, targets,
                                                 cache, dJ_dz, prev_layer)
            dJ_dw_k.append(dJ_dw)
            dJ_db_k.append(dJ_db)
            prev_layer = layer

        return dJ_dw_k, dJ_db_k

    def train(self, inputs, targets, epochs, learning_rate):
        if isinstance(inputs, cp.ndarray) and not isinstance(self.layers_weights[0], cp.ndarray):
            self.convert_weight_bias_type()
        elif isinstance(inputs, np.ndarray) and not isinstance(self.layers_weights[0], np.ndarray):
            self.convert_weight_bias_type()

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

        return cost_values