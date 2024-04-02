'''
Rede Neural Artificial usando apenas Numpy a matemática.
Prova de conceito a partir de estudos da pós graduação em inteligência artificial.
Autor: Alef Matias @exclerosado
'''

from activation import *
import numpy as np
from tqdm import tqdm

# Valor fixado para ajudar na etapa de avaliação dos resultados obtidos
np.random.seed(653)

class Neural_Network():
    def __init__(self, input_size, hidden_size, output_size, activation):
        # Neste bloco é definida a arquitetura da rede neural
        # É importante ressaltar que o parâmetro input_size receba o mesmo tamanho dos dados de entrada da rede
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.hidden_weights = np.random.randn(input_size, hidden_size)
        self.output_weights = np.random.randn(hidden_size, output_size)

        self.hidden_bias = np.zeros((1, hidden_size))
        self.output_bias = np.zeros((1, output_size))

        self.activation = activation

        # Condicional para definir qual tipo de função de ativação será utilizada
        if activation == 'sigmoid':
            self.activation_function = sigmoid
            self.der_activation = der_sigmoid
        elif activation == 'tanh':
            self.activation_function = tanh
            self.der_activation = der_tanh
        elif activation == 'relu':
            self.activation_function = relu
            self.der_activation = der_relu


    def train(self, x, y, epochs, alpha):
        for epoch in tqdm(range(epochs)):
            # Propagação para frente (feedforward)
            z1 = self.activation_function(np.dot(x, self.hidden_weights) + self.hidden_bias)
            z2 = sigmoid(np.dot(z1, self.output_weights) + self.output_bias)

            error = z2 - y

            # Cálculo dos deltas para atualização dos pesos na etapa de retropropagação (backpropagation)
            delta_output_weights = z1.T.dot(error * der_sigmoid(z2))
            delta_output_bias = np.mean(error)

            error_hidden = error.dot(self.output_weights.T) * self.der_activation(z1)

            delta_hidden_weights = x.T.dot(error_hidden)
            delta_hidden_bias = np.mean(error_hidden)

            # Atualização dos pesos (backpropagation)
            self.output_weights -= alpha * delta_output_weights
            self.output_bias -= alpha * delta_output_bias
            self.hidden_weights -= alpha * delta_hidden_weights
            self.hidden_bias -= alpha * delta_hidden_bias

        return self.hidden_weights, self.output_weights, self.hidden_bias, self.output_bias
    

    def predict(self, x):
        z1 = self.activation_function(np.dot(x, self.hidden_weights) + self.hidden_bias)
        z2 = sigmoid(np.dot(z1, self.output_weights) + self.output_bias)
        return z2
    

    def accuracy(self, x_test, y_test):
        predictions = np.round(self.predict(x_test))
        accuracy = np.mean(predictions == y_test)
        return accuracy * 100
