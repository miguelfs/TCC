import numpy as np


def if2carrier(F):
    fs = 2
    return np.exp(2j * np.pi / fs * np.cumsum(np.array(F).T, axis=0)).T
