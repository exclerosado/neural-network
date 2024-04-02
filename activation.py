import numpy as np


# Sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def der_sigmoid(z):
    return z * (1 - z)


# Tanh
def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


def der_tanh(z):
    return 1 - (z ** 2)


# ReLU
def relu(x):
    return np.maximum(0, x)


def der_relu(z):
    return np.where(z <= 0, 0, 1)
