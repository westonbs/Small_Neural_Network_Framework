import numpy as np
import matplotlib.pyplot as plt
from NeuralNetwork import NeuralNetwork
from Layer import Layer
from PreProcessing import *
import Activations

model = NeuralNetwork(
    [Layer(784, 25, 'relu', 0),
     Layer(25, 20, 'relu', 1),
     Layer(20, 15, 'relu', 2),
     Layer(15, 26, 'softmax', 3)],
    64
)

Y_train, X_train = load_data()
model.load_weights_bias()
model.train(X_train, Y_train, 20, .001)
model.save_weights_bias()

print(model.test_accuracy(X_train, Y_train))