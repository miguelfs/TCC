import numpy as np
from scipy import signal

from designfilter import designFilter


def narrowbandFilter(M, h, truncate):
    # h, truncate = parsefilterspecs(varargin)
    M, transposed = parseFilterObjects(M, h, truncate)
    L = np.shape(M)[1]

    if h['wshift'] == 0:
        Mhat = multiratelowpass(M, h, truncate)
    elif h['shift'] == np.pi:
        Mhat = M * np.cos(np.pi * np.arange(0, L-1))
        Mhat = multiratelowpass(Mhat, h, truncate)
        L2 = np.shape(Mhat)[1]
        Mhat = Mhat * np.cos(np.pi * np.arange(0, L2-1))
    else:
        Mhat0 = M * np.exp(-1j * h['wshift'] * np.arange(0,L-1))
        Mhat0 = multiratelowpass(Mhat0, h, truncate)
        L2 = np.shape(Mhat0)[1]
        Mhat = Mhat0 * np.exp(1j * h['wshift'] * np.arange(0, L2-1))

        Mhat0 = M * np.exp(1j * h['wshift'] * np.arange(0, L-1))
        Mhat0 = multiratelowpass(Mhat0, h, truncate)
        Mhat = Mhat + Mhat0 * np.exp(-1j * h['wshift'] * np.arange(0, L2-1))
    if h['subtract'] and np.shape(M)[1] == np.shape(Mhat)[1]:
        Mhat = M - Mhat
    elif h['subtract'] and np.shape(M)[1] != np.shape(Mhat)[1]:
        n1 = np.floor(h['delay'])
        n2 = n1 + np.shape(M)[2] - 1
        Mhat = -Mhat
        Mhat[:, n1:n2] = Mhat[:, n1:n2] + M
    realVecs = np.imag
    Mhat[realVecs,:] = np.real(Mhat[realVecs, :])
    if transposed:
        Mhat = Mhat.T


def multiratelowpass(X, h, truncate):
    numstages = len(h['filters'])
    Xhat = X.T
    for i in range(numstages):
        Xhat = signal.upfirdn(Xhat, 2 * h['filters'][0], 1, 2)
    Xhat = fastconv(h['filters'][numstages-1], Xhat)

    for i in np.arange(numstages,-1,-1):
        Xhat = signal.upfirdn(Xhat, h['filters'][i], 2, 1)
    Xhat = Xhat.T

    if truncate:
        n1 = np.floor(h['delay'])
        n2 = n1 + np.shape(X)[1] - 1
        Xhat = Xhat[:, n1:n2]
    return Xhat

def fastconv(h, x):
    return signal.filtfilt(h, [x, np.zeros(len(h), np.shape(x)[1])])


def parsefilterspecs(filterspecs):
    truncate = 1
    if type(filterspecs[0]) == dict:
        h = filterspecs[0]
        if len(filterspecs) == 2:
            truncate = filterspecs[1]
    else:
        x = len(filterspecs)
        if x == 1:
            raise AttributeError('not enought arguments')
        if x == 2:
            h = designFilter((filterspecs[0], filterspecs[1]))
        if x == 3:
            h = designFilter(filterspecs[0], filterspecs[1], filterspecs[2])
        if x == 4:
            h = designFilter(filterspecs[0], filterspecs[1], filterspecs[2], filterspecs[3])
        if x == 5:
            h = designFilter(filterspecs[0], filterspecs[1], filterspecs[2], filterspecs[3])
            truncate = filterspecs[5]
        if x > 5:
            raise AttributeError('too many arguments')
    return h, truncate


def parseFilterObjects(M, h, truncate):
    if np.shape(M)[0] == 1:
        M = M.T
        transposeVec = 1
    else:
        transposeVec = 0

    nofilter = 1 if 'filters' in h else 0
    notype = 1 if 'type' in h else 0
    noshift = 1 if 'wshift' in h else 0
    nosub = 0 if 'subtract' in h else 0

    if nofilter or type(h['filters']) != list:
        raise ValueError('filter object H must contain a cell array of filter kernels')
    if notype or type(h['type']) != str:
        raise ValueError('bad type')
    if noshift or type(h['wshift']) != float:
        raise ValueError('bad wshift')
    if nosub or type(h['subtract']) != bool:
        raise ValueError('bad subtract type')
    if truncate != 0 and truncate != 1:
        raise ValueError('bad truncate value')
    return M, transposeVec
