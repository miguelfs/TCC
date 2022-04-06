import numpy as np


def modrecon(M: np.array, C: np.array) -> np.array:
    if np.shape(M) != np.shape(C):
        raise IndexError('M and C shapes must be equal')
    return M * C
