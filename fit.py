import numpy as np


# Converte os dados para uma escala entre 0 e 1
def normalize(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))


# Reverte a normalização dos dados
def revert(x, datas):
    return x * (np.max(datas) - np.min(datas)) + np.min(datas)
