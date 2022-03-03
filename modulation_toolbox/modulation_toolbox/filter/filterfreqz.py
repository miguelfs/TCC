import numpy as np


def filterFreqz(h: dict, nfft=None, fs: float = 2, full=0):
    downsampling = 2 ^ (len(h['filters']) - 1)  # overrall decimation factor

    if nfft is None:
        nfft = 4 * len(h['filters'][-1] * downsampling)

    if nfft / downsampling % 1 != 0:
        ValueError('NFFT points must be a multiple of the overall filter downsampling size')

    W = np.linspace(0, 2 * np.pi, nfft + 1)
    W = W[0:W[-2]]

    H = np.ones(len(W))

    for i in range(0, len(h['filters'] - 1)):
        Hi = np.fft(h['filters'] - 1)
        print(i)
