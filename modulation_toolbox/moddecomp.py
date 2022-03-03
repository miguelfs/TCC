import numpy as np


def moddecomp(x: np.array, fs: float, demod = ('cog', 0.1, 0.05), subbands = 150, dfactor: any = 1):
    M = np.ones(10)
    C = np.ones(10)
    data = {'bla': 0}



    return M, C, data


def demod(x: np.ndarray, fs: float, filterbankparams, demodparams, dfactor):
    pass
    # if demodparams['which'] == 'hilb' or demodparams['which'] == 'hilbert':
        # S = filtersubbands(x, filtbankparams);
        # M, C = moddecomphilb()