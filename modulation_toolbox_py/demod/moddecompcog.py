import numpy as np
import torch

from modulation_toolbox_py.demod.if2carrier import if2carrier
from modulation_toolbox_py.index import index
import torchaudio.functional as torch_f


def moddecompcog(S: np.ndarray, carrWin: list, carrWinHop=None, centers: [int] = None, bandwidths: [int] = None):
    """"demodulates using center-of-gravity"""
    carrWin = np.array([carrWin])
    if bandwidths is None:
        bandwidths = [2]
    if centers is None:
        centers = [0]
    if carrWinHop is None:
        carrWinHop = np.ceil(len(carrWin) / 2)
    sdim = numbands, ncols = S.shape
    w1, w2, carrWin, carrWinLen, carrWinHop = parseinputs(sdim, carrWin, carrWinHop, centers, bandwidths)
    numFrames = int(np.ceil(ncols / carrWinHop))
    Fmeasured = np.zeros((numbands, numFrames))
    F = np.zeros((numbands, ncols))
    DFTsize = int(2 ** np.ceil(np.log2(carrWinLen)))
    for k in np.arange(0, numbands):
        subband = S[k, :].reshape(1, -1)
        if carrWinLen == ncols:
            # computes IF estimate using one COG computation
            P = np.abs(np.fft.fft(carrWin * subband, DFTsize)) ** 2
            Fmeasured[k, :] = spectralCOG(P, index(w1, k), index(w2, k))
            F[k, :] = Fmeasured[k, 0]
            continue
        leftEdge = 1 - int(np.ceil(carrWinLen / 2))
        n1 = 0
        n2 = leftEdge + carrWinLen - 1
        for i in np.arange(0, numFrames):
            if n1 == 0:
                zeropad = np.zeros((1, n2))
                frame = np.hstack((zeropad, carrWin[-n2:] * subband[n1:n2]))
            elif n2 == ncols:
                zeropad = np.zeros((1, carrWinLen - n2 + n1 - 1))
                frame = np.hstack((carrWin[:n2 - n1 + 1] * subband[n1:n2], zeropad))
            else:
                frame = carrWin * subband[n1:n2]
            P = np.abs(np.fft.fft(frame, DFTsize)) ** 2
            Fmeasured[k, i] = spectralCOG(P, index(w1, k), index(w2, k))
            leftEdge = leftEdge + carrWinHop
            n1 = max(leftEdge, 0)
            n2 = min(leftEdge + carrWinLen - 1, ncols)
        L = min(4, np.floor((numFrames - 1) / 2))
        Ftemp = factorinterp(Fmeasured[k, :], carrWinHop, L, .25)
        F[k, :] = Ftemp[0:ncols]
    C = if2carrier(F)
    M = S * np.conj(C)
    return M, C, F, Fmeasured


def factorinterp(x, R, L, cutoff):
    factors = factor(R)
    y = x
    for i in np.arange(0, len(factors)):
        y = torch_f.resample(torch.tensor(y, dtype=torch.float), orig_freq=1, new_freq=factors[i],
                             lowpass_filter_width=len(x) // 4).detach().numpy()
    return y


def primes(n):  # simple sieve of multiples // https://stackoverflow.com/a/19498432
    odds = range(3, n + 1, 2)
    sieve = set(sum([list(range(q * q, n + 1, q + q)) for q in odds], []))
    return [2] + [p for p in odds if p not in sieve]


def factor(n):  # https://stackoverflow.com/a/16996439
    primfac = []
    d = 2
    while d * d <= n:
        while (n % d) == 0:
            primfac.append(d)  # supposing you want multiple factors repeated
            n //= d
        d += 1
    if n > 1:
        primfac.append(n)
    return primfac


def spectralCOG(P, w1: float, w2: float):
    """computes the spectral center of gravity between frequency bounds w1 and w2"""
    P = np.array(P)
    if w1 > w2:
        raise ValueError('left frequency bound must be greater or equal than the right one')
    N = P.shape[1] if type(P) is np.ndarray and len(P.shape) == 2 else len(P)
    k1, k2 = (np.round(np.array([w1, w2]) * N / 2) % N) + 1
    k1, k2 = int(k1), int(k2)
    if k1 == k2 and w2 - w1 == 2:
        k1 = k1 + 1
    if k1 <= k2:  # no circularity
        kRange = np.arange(k1 - 1, k2)
        omega = 2 / N * np.arange(0, N)
    else:  # account for circularity
        kRange = np.hstack((np.arange(k1 - 1, N), np.arange(0, k2 - 1)))
        omega = 2 / N * np.hstack((np.arange(0, 1 + N // 2), np.arange((N // 2) + 1 - N, 0)))
    P = np.array(P)
    P = np.column_stack(P).reshape(1,-1)
    if len(P.shape) == 2:
        w0 = np.sum(omega[kRange] * P[:,kRange]) / np.sum(P[:,kRange]) if np.sum(np.real(P)) > 0 else np.mean([w1, w2])
    else:
        w0 = np.sum(omega[kRange] * P[kRange]) / np.sum(P[kRange]) if np.sum(np.real(P)) > 0 else np.mean([w1, w2])
    return w0


def parseinputs(sdim, carrWin: list, carrWinHop: int, centers: list, bandwidths: list):
    carrWinLen = len(carrWin)
    numBands, ncols = sdim
    if np.size(carrWin) == 1:
        carrWinLen = carrWin
        carrWin = np.hamming(len(carrWin))
    elif np.size(carrWin) > 1:
        carrWinLen = len(carrWin)
    else:
        ValueError('a carrier-detection window length must be specified')
    if carrWinLen >= ncols:
        print('warning: the carrier window length is greater than the subband length')
        carrWinLen = ncols
        carrWin = np.hamming(carrWinLen)
        carrWinHop = ncols
    elif np.ceil(ncols / carrWinHop) < 8:
        print('warning: Not enough IF data points for interpolation')
        carrWinLen = ncols
        carrWin = np.hamming(carrWinLen)
        carrWinHop = ncols
    if carrWin.shape[0] == 1:
        carrWin = carrWin.T
    if len(centers) != 1 and len(centers) != numBands:
        raise ValueError('number of sub-band centers and bandwidths must be one or equal')
    if len(bandwidths) != 1 and len(bandwidths) != numBands:
        raise ValueError('number of bandwidths centers must be one or equal')
    if min(centers) < -1 or max(centers) > 1:
        raise ValueError('sub-band center frequencies must be between -1 and 1 inclusive')
    if min(bandwidths) < 0 or max(bandwidths) > 2:
        raise ValueError('sub-band bandwidths must be between 0 and 2 inclusive')
    w1 = np.array(centers) - np.array(bandwidths) / 2
    w2 = np.array(centers) + np.array(bandwidths) / 2
    return w1, w2, carrWin, carrWinLen, carrWinHop
