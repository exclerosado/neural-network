from activation import sigmoid, der_sigmoid
import numpy as np
from tqdm import tqdm

np.random.seed(3413)


class Neural_Network():
    def __init__(self, input_size, hidden1_size, hidden2_size, output_size):
        self.input_size = input_size
        self.hidden1_size = hidden1_size
        self.hidden2_size = hidden2_size
        self.output_size = output_size

        self.hidden1_weights = np.random.randn(input_size, hidden1_size)
        self.hidden1_bias = np.zeros((1, hidden1_size))

        self.hidden2_weights = np.random.randn(hidden1_size, hidden2_size)
        self.hidden2_bias = np.zeros((1, hidden2_size))

        self.output_weights = np.random.randn(hidden2_size, output_size)
        self.output_bias = np.zeros((1, output_size))


    def train(self, x, y, epochs, alpha):
        for epoch in tqdm(range(epochs), desc='Trainning model ---> '):
            z1 = sigmoid(np.dot(x, self.hidden1_weights) + self.hidden1_bias)
            z2 = sigmoid(np.dot(z1, self.hidden2_weights) + self.hidden2_bias)
            z3 = sigmoid(np.dot(z2, self.output_weights) + self.output_bias)

            error = z3 - y

            delta_output = error * der_sigmoid(z3)
            delta_output_weights = z2.T.dot(delta_output)
            delta_output_bias = np.mean(delta_output)

            error_hidden2 = delta_output.dot(self.output_weights.T) * der_sigmoid(z2)
            delta_hidden2_weights = z1.T.dot(error_hidden2)
            delta_hidden2_bias = np.mean(error_hidden2)

            error_hidden1 = error_hidden2.dot(self.hidden2_weights.T) * der_sigmoid(z1)
            delta_hidden1_weights = x.T.dot(error_hidden1)
            delta_hidden1_bias = np.mean(error_hidden1)

            self.output_weights -= alpha * delta_output_weights
            self.output_bias -= alpha * delta_output_bias
            self.hidden2_weights -= alpha * delta_hidden2_weights
            self.hidden2_bias -= alpha * delta_hidden2_bias
            self.hidden1_weights -= alpha * delta_hidden1_weights
            self.hidden1_bias -= alpha * delta_hidden1_bias


        return self.hidden1_weights, self.hidden2_weights, self.output_weights, self.hidden1_bias, self.hidden2_bias, self.output_bias
    

    def predict(self, x):
        z1 = sigmoid(np.dot(x, self.hidden1_weights) + self.hidden1_bias)
        z2 = sigmoid(np.dot(z1, self.hidden2_weights) + self.hidden2_bias)
        z3 = sigmoid(np.dot(z2, self.output_weights) + self.output_bias)
        return z3
