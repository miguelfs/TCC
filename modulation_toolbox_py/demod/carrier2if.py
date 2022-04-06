import numpy as np
from scipy.signal import lfilter


def carrier2if(C):
    if np.size(C) == np.shape(C)[1] and np.shape(C)[0] > 1:
        C = C.T
    fs = 2
    # The first-difference operation approximates a temporal derivative of the carrier phases.
    return fs / (2 * np.pi) * lfilter([1, -1], 1, np.unwrap(np.angle(C).T))
