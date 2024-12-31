import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from NeuralNetwork import NeuralNetwork
import Layer
from PreProcessing import *
import ActivationsNp
import ActivationsCp

def numerical_gradient_check(model, X, Y, epsilon=1e-3):
    diffs = []
    max_diff = 0
    grads_weights, grads_bias = model.compute_gradients(X, Y)
    grads_weights = grads_weights[::-1]
    num_grads = []
    for i in range(len(model.layers_weights) - 1, -1, -1):  # Iterate through layers
        m, n = model.layers_weights[i].shape
        for j in range(m):
            for k in range(n):
                original_weight = model.layers_weights[i][j, k]

                # Add epsilon to test numerical gradient
                model.layers_weights[i][j, k] = original_weight + epsilon
                pos_cost = model.cost(X, Y).astype(float)

                # Subtract epsilon to test numerical gradient
                model.layers_weights[i][j, k] = original_weight - epsilon
                neg_cost = model.cost(X, Y).astype(float)

                # Restore original weight
                model.layers_weights[i][j, k] = original_weight

                # Compute numerical gradient
                numerical_grad = (pos_cost - neg_cost) / (2 * epsilon)
                num_grads.append(numerical_grad)
                # Extract analytical gradient
                analytical_grad = grads_weights[i][j, k]

                # Ensure correct dtype casting to float32
                numerical_grad = float(numerical_grad)
                analytical_grad = float(analytical_grad)

                # Calculate the relative difference
                diff = abs(numerical_grad - analytical_grad) / (abs(analytical_grad) + 1e-9)
                if diff > max_diff:
                    max_diff = diff

                # Log errors for debugging
                if diff > 1e-3:  # Use a higher threshold if needed
                    print(f"Gradient Check Failed at Layer {i} - Weight[{j},{k}]:")
                    print(
                        f"Numerical Grad: {numerical_grad}, Analytical Grad: {analytical_grad}, Diff: {diff}")

                diffs.append(diff)

    return diffs, max_diff, num_grads

def numerical_gradient(model, X, Y, epsilon=1e-4):
    num_grads = []
    for i in range(len(model.layers_weights) - 1, -1, -1):  # Iterate through layers
        m, n = model.layers_weights[i].shape
        for j in range(m):
            for k in range(n):
                original_weight = model.layers_weights[i][j, k]

                # Add epsilon to test numerical gradient
                model.layers_weights[i][j, k] = original_weight + epsilon
                pos_cost = model.cost(X, Y)

                # Subtract epsilon to test numerical gradient
                model.layers_weights[i][j, k] = original_weight - epsilon
                neg_cost = model.cost(X, Y)

                # Restore original weight
                model.layers_weights[i][j, k] = original_weight

                # Compute numerical gradient
                numerical_grad = (pos_cost - neg_cost) / (2 * epsilon)

                num_grads.append(numerical_grad)
                # Extract analytical gradient

    return num_grads

model = NeuralNetwork(
    [Layer(128, 'relu', 784),
     Layer(64, 'relu'),
     Layer(26, 'softmax')],
    4096
)

#download_data()
X_train, Y_train = load_data()
print(X_train.shape, Y_train.shape)
X_train = cp.asarray(X_train, dtype=cp.float32)
Y_train = cp.asarray(Y_train, dtype=cp.int32)

model.load_weights_bias('cupy')
cost = model.train(X_train, Y_train, 10000, 0.001)
model.plot_cost(cost)
model.convert_weight_bias_type()
X_train = cp.asnumpy(X_train)
Y_train = cp.asnumpy(Y_train)
model.test_accuracy_numpy(X_train, Y_train)
model.save_weights_bias()
