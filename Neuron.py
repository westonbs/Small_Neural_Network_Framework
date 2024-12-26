import numpy as np

class Neuron:
    """
    Class representing a simple neuron

    Attributes:
        activation_function (callable): activation function of neuron, ietf. sigmoid, relu
        weight (numpy.ndarray or None): weights of the neuron, initialized as random nparray
        bias (float or None): bias of neuron, initialized as random float

    Methods:
        set_weight(weight): sets weight of neuron
        set_bias(bias): sets bias of neuron
        compute_activation(example): computes activation of neuron for input example.
    """
    def __init__(self, weight, bias, activation_function):
        self.activation_function = activation_function
        self.weight = weight
        self.bias = bias
    
    def set_weight(self, weight):
        self.weight = weight

    def set_weight_ind(self, weight, ind):
        self.weight[ind] = weight
        
    def set_bias(self, bias):
        self.bias = bias
        
    def compute_activation(self, example):
        if self.weight is None or self.bias is None:
            raise ValueError("Weight and bias must be set before computing the activation.")
        return self.activation_function(np.dot(example, self.weight) + self.bias)
    
    def __str__(self):
        return 'Neuron with weight: ' + str(self.weight) + " and bias: " + str(self.bias)
    