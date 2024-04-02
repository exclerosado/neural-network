from nn import Neural_Network
import numpy as np

# Dados de teste baseados no problema da porta lógica XOR
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([[0], [1], [1], [0]])

# Rede com duas entradas, 8 neurônios na camada oculta e uma saída
nn = Neural_Network(2, 8, 1, activation='sigmoid')

# Treinamento em 5000 épocas com taxa de aprendizado (alpha) de 5%
traning = nn.train(X, Y, 5000, 0.05)

print(nn.predict(np.array([[0, 0]])))
print(nn.predict(np.array([[0, 1]])))
print(nn.predict(np.array([[1, 0]])))
print(nn.predict(np.array([[1, 1]])))
