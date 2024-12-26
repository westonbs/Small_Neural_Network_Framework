import numpy as np
import Activations
from Neuron import Neuron

class Layer:
    """
    Class representing a single layer in a neural network, handles forward and backward propagation

    Attributes:
        input_shape (tuple or int): Shape of input data
        output_shape (tuple or int): Shape of output data
        activation_function (callable): Activation function applied to layer outputs.
        derivation_function (callable): Derivative of the activation function
        weights (np.ndarray): Weight matrix for neurons of layer, size: input x output
        bias (np.ndarray): Bias vector for neurons of layer, size: 1 x output

    Methods:
        forward(X, cache): forward propagation
        backward(X, cache, delta, next_layer): backward propagation
    """
    def __init__(self, input_shape, output_shape, activation, index):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.activation_function = getattr(Activations, activation)
        self.derivation_function = getattr(Activations, f"d_{activation}")
        self.cost_derivation_function = getattr(Activations, f"d_{activation}_d_a")
        self.index = index

        self.neurons = []
        self.weights = np.random.randn(input_shape, output_shape)
        self.bias = np.random.randn(1, output_shape)
        for i in range(output_shape):
            self.neurons.append(Neuron(self.weights[:, i], self.bias, self.activation_function))

    def set_index(self, index):
        self.index = index

    def forward(self, X, cache):
        if X.shape != self.input_shape:
            raise ValueError(f"Input shape {X.shape} does not match the expected input "
                             f"shape {self.input_shape}.")

        z = np.dot(X, self.weights) + self.bias
        a = self.activation_function(z)
        if cache is not None:
            cache['z^{self.index}'] = z
            cache['a^{self.index}'] = a

        return a

    def backward(self, X, cache, delta, next_layer):
        if X.shape != self.input_shape:
            raise ValueError(f"Input shape {X.shape} does not match the expected input "
                             f"shape {self.input_shape}.")
        if delta is not None and delta.shape != self.output_shape:
            raise ValueError(f"Output shape {delta.shape} does not match the expected output "
                             f"shape {self.output_shape}.")

        #should cache values, for now computing them everytime
        z = cache['z^{self.index}']
        if next_layer is None:
            dJ_da = self.cost_derivation_function(cache['a^{self.index}'])
        else:
            next_weights = next_layer.weights
            dJ_da = np.matmul(delta, next_weights.T)

        dJ_dz = np.matmul(dJ_da, self.derivation_function(z).T)
        dJ_dw = np.matmul(X.T, dJ_dz)
        dJ_db = dJ_dz

        return dJ_dz, dJ_dw, dJ_db

    def __str__(self):
        return f"Layer with {self.input_shape} inputs and {self.output_shape} outputs."
