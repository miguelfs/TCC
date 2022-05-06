import numpy as np


def index(l, i: int):
    if type(l) is np.ndarray and l.shape == (1, 1):
        return l[0, 0]
    # if (l is np.ndarray and l.shape == (1,)) or (l is list and len(l) == 1):
    #     return l[0]
    if type(l) is np.ndarray and len(l.shape) == 2 and l.shape[0] == 1:
        return l[0, i]
    if type(l) is np.ndarray and len(l) == 1:
        return l[0]
    if type(l) is list or (type(l) is np.ndarray and len(l.shape) == 1):
        return l[i]
    if type(l) is not np.array and type(l) is not list:
        return l
    raise ValueError('invalid type or invalid size')