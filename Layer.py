import numpy as np
import cupy as cp
import ActivationsNp
import ActivationsCp

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
    def __init__(self, output_shape, activation, input_shape=None):
        self.activation_function_np = getattr(ActivationsNp, activation)
        self.derivation_function_np = getattr(ActivationsNp, f"d_{activation}")
        self.cost_derivation_function_np = getattr(ActivationsNp, f"dJ_{activation}")
        self.activation_function_cp = getattr(ActivationsCp, activation)
        self.derivation_function_cp = getattr(ActivationsCp, f"d_{activation}")
        self.cost_derivation_function_cp = getattr(ActivationsCp, f"dJ_{activation}")
        self.input_shape, self.output_shape = input_shape, output_shape
        self.weights, self.bias = None, None
        self.index = None

        if input_shape is not None:
            self.set_input_shape(input_shape)

    def set_input_shape(self, input_shape):
        self.input_shape = input_shape
        self.weights = (np.random.randn(input_shape, self.output_shape)
                        * np.sqrt(2 / input_shape))
        self.bias = np.zeros((1, self.output_shape))

    def set_index(self, index):
        self.index = index

    def convert_weight_bias_type(self):
        if isinstance(self.weights, np.ndarray):
            self.weights = cp.asarray(self.weights, dtype=cp.float32)
            self.bias = cp.asarray(self.bias, dtype=cp.float32)
        else:
            self.weights = cp.asnumpy(self.weights)
            self.bias = cp.asnumpy(self.bias)

        return self.weights, self.bias

    def forward_numpy(self, X, cache):
        if X.shape[1] != self.input_shape:
            raise ValueError(f"Input shape {X.shape} does not match the expected input "
                             f"shape {self.input_shape}.")

        z = np.dot(X, self.weights) + self.bias
        a = self.activation_function_np(z)
        if cache is not None:
            cache[f'z^{self.index}'] = z
            cache[f'a^{self.index}'] = a

        return a

    def forward_cupy(self, X, cache):
        if X.shape[1] != self.input_shape:
            raise ValueError(f"Input shape {X.shape} does not match the expected input shape {self.input_shape}.")

        # Ensure dtype consistency (float32 for CuPy)
        X = X.astype(cp.float32)
        self.weights = self.weights.astype(cp.float32)
        self.bias = self.bias.astype(cp.float32)

        z = cp.dot(X, self.weights) + self.bias
        a = self.activation_function_cp(z)

        if cache is not None:
            cache[f'z^{self.index}'] = z
            cache[f'a^{self.index}'] = a

        return a

    def forward(self, X, cache):
        if isinstance(X, np.ndarray):
            return self.forward_numpy(X, cache)
        else:
            return self.forward_cupy(X, cache)

    def backward_numpy(self, X, Y, cache, delta, next_layer):
        # should cache values, for now computing them everytime
        a = cache[f'a^{self.index}']
        z = cache[f'z^{self.index}']
        if next_layer is None:
            dJ_dz = self.derivation_function_np(a, Y)
        else:
            next_weights = next_layer.weights
            dJ_da = np.matmul(delta, next_weights.T)
            dJ_dz = self.derivation_function_np(z, dJ_da)

        dJ_dw = np.matmul(X.T, dJ_dz)
        dJ_db = np.sum(dJ_dz, axis=0, keepdims=True)

        return dJ_dz, dJ_dw, dJ_db

    def backward_cupy(self, X, Y, cache, delta, next_layer):
        a = cache[f'a^{self.index}']
        z = cache[f'z^{self.index}']
        if next_layer is None:
            dJ_dz = self.derivation_function_cp(a, Y)
        else:
            next_weights = next_layer.weights
            dJ_da = cp.matmul(delta, next_weights.T)
            dJ_dz = self.derivation_function_cp(z, dJ_da)

        dJ_dw = cp.matmul(X.T, dJ_dz)
        dJ_db = cp.sum(dJ_dz, axis=0, keepdims=True)

        return dJ_dz, dJ_dw, dJ_db

    def backward(self, X, Y, cache, delta, next_layer):
        if isinstance(X, np.ndarray):
            return self.backward_numpy(X, Y, cache, delta, next_layer)
        else:
            return self.backward_cupy(X, Y, cache, delta, next_layer)

    def __str__(self):
        return f"Layer with {self.input_shape} inputs and {self.output_shape} outputs."
