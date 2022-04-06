import numpy as np
from scipy import signal


def filtersubbands(x: np.array, filtbankparams: dict) -> np.array:
    parseInputs(x, filtbankparams)
    if np.shape(x)[0] == 1:
        x = x.T
    if filtbankparams['stft']:
        win = filtbankparams['afilters{1}']
        hop = filtbankparams['dfactor(1)']
        nfft = filtbankparams['numhalfbands']
        fshift = filtbankparams['fshift']
        S = STFT(x, win, hop, nfft, fshift)
        nmid = (len(win) - 1) / 2
        W = windowphaseterm(nmid, nfft)
        S = np.diag(np.conj(W)) @ S
        S = S[:filtbankparams['numsubbands'], :] if not any(np.imag(x)) else S
    else:
        maxLen = maxSubbandLen(len(x), filtbankparams)
        S = np.zeros((filtbankparams[('numbands' if np.isrealobj(x) else 'numhalfbands')], maxLen), dtype=np.complex_)
        for k in range(filtbankparams['numbands']):
            center = index(filtbankparams['centers'], k)
            h = index(filtbankparams['afilters'], k)
            dfactor = index(filtbankparams['dfactor'], k)
            subband = bandpassFilter(x, center, h, dfactor, filtbankparams['fshift'])
            S[k, :] = padToMatch(subband, maxLen).ravel()
            if center != 0 and center != 1:
                break
            n0 = (len(h) - 1) / 2
            W = np.exp(1j * np.pi * center * n0)
            S[k, :] = np.conj(W) * S[k, :]
            if any(np.imag(x)):
                subband = bandpassFilter(x, -center, h, dfactor, filtbankparams['fshift'])
                ind = filtbankparams['numhalfbands'] - k + (filtbankparams['centers'][0] == 0)
                S[ind, :] = W * padToMatch(subband, maxLen)
    if not any(np.imag(x)):  # TODO: np.isrealobj
        hilbertGain = 1 + np.array([f != 0 for f in filtbankparams['centers']] and [f != 1 for f in filtbankparams['centers']])
        S = np.diag(hilbertGain) @ S
    if not filtbankparams['keeptransients']:
        S = trimfiltertransients(S, filtbankparams, len(x))
    return S


def trimfiltertransients(S, filtbankparams, origlen):
    S = np.array(S)
    newlen = int(np.ceil(origlen / filtbankparams['dfactor']))
    S2 = np.zeros((np.shape(S)[0], newlen))
    for k in range(np.shape(S)[0]):
        n1 = int(np.ceil((index(np.array(filtbankparams['afilterorders']), k) / 2) / filtbankparams['dfactor']))
        n2 = int(n1 + newlen - 1)
        S2[k, :] = S[k, n1 - 1:n2]
    return S2


def padToMatch(x, L):
    return np.vstack((x, np.zeros(L - len(x)).reshape(-1, 1)))


def bandpassFilter(x, center, h, dfactor, fshift):
    xmod = vmult(x, np.exp(-1j * np.pi * center * np.arange(0, x.shape[0]))) if fshift else x
    h = vmult(h, np.exp(1j * np.pi * center * np.arange(0, len(h)))) if not fshift else h
    # h = vmult(h, np.exp(1j * np.pi * center * np.arange(0, len(h)))) if not fshift else h.reshape(1, -1)
    subband = signal.upfirdn(h, xmod.T, 1, dfactor).T if dfactor >= (len(h) - 1) / 128 else downsample(fastconv(h.reshape(1, -1), xmod), dfactor)
    return subband


def fastconv(h, x):
    return signal.oaconvolve(x, np.array(h).T)


def downsample(x: np.array, dfactor, phase: int = 0):
    if phase > dfactor:
        raise ValueError('phase must be lower than dfactor')
    return x[phase::dfactor]


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

def vmult(x1, x2):  # multiplies two vectors element-wise, regardless of orientation. Output shape matches x1.
    return x1 * x2.reshape(x1.shape)


def windowphaseterm(nmid, nfft: int):
    W = np.zeros(nfft, 1)
    if nfft % 2 == 0:
        W[:nfft // 2] = np.exp(2j * np.pi * np.arange(0, nfft // 2) / nfft * nmid)
        W[nfft // 2] = 1
        W[:nfft // 2:-1] = np.conj(W[1:nfft // 2])
    else:
        W[:nfft // 2] = np.exp(2j * np.pi * np.arange(0, (nfft + 1) // 2 / nfft * nmid))
        W[:(nfft + 1) // 2:-1] = np.conj(W[1:(nfft + 1) // 2])
    return W


def maxSubbandLen(L, fbparams) -> int:
    return int(max(np.ceil((L + np.array(fbparams['afilterorders'])) / fbparams['dfactor']))
               )


def convbuffer(x, winlen, startindex, hop):
    x = np.array(x).reshape(-1, 1)
    numframes = np.ceil((len(x) - startindex) / hop)
    if startindex <= 0:
        leadin = np.zeros((-startindex, 1))
        n1 = 0
    else:
        leadin = np.zeros((0, 1))
        n1 = startindex
    tail = np.zeros((max(0, int(numframes - 1) * hop + winlen - len(x) - len(leadin)), 1))
    if hop < winlen:
        opt = 'nodelay'
    elif hop > winlen:
        opt = 0
    else:
        opt = []
    v = np.vstack((leadin, x[n1:], tail))
    return buffer(v, winlen, winlen - hop, opt)


def buffer(data, duration, dataOverlap=0, opt=None):
    opt = None if opt == [] else opt
    if len(np.shape(data)) > 1 and np.shape(data)[1] == 1:
        data = data.reshape(-1)
    # opt =np.zeros(dataOverlap, 1) if opt is None else opt
    data = np.append(np.zeros(dataOverlap), data) if opt != 'nodelay' else data
    numberOfSegments = int(np.ceil((len(data) - dataOverlap) / (duration - dataOverlap)))
    tempBuf = [data[i:i + duration] for i in range(0, len(data), (duration - int(dataOverlap)))]
    tempBuf[numberOfSegments - 1] = np.pad(tempBuf[numberOfSegments - 1],
                                           (0, duration - np.shape(tempBuf[numberOfSegments - 1])[0]), 'constant')
    tempBuf2 = np.vstack(tempBuf[:numberOfSegments])
    tempBuf2 = tempBuf2.T
    if opt == 'nodelay':
        return tempBuf2
    return tempBuf2
    # elif opt is None:
    #     first_column = tempBuf2[:,1]
    #     for i in range(len(first_column - 1)):
    #         tempBuf2 = [tempBuf2[:,1], tempBuf]
    #     return tempBuf2


def STFT(x, win, hop, nfft: int, fshift):
    winLen: int = len(win)
    winpoly = convbuffer(win[::1], nfft, 0, nfft)
    S = convbuffer(x, nfft, -winLen + 1, hop)
    S = np.diag(winpoly[:, 0]) @ S
    for k in range(2, 1 + int(np.ceil(winLen / nfft))):
        Stemp = convbuffer(x, nfft, -winLen + 1 + (k - 1) * nfft, hop)
        Stemp = winpoly[:, k] * Stemp
        Ltemp = len(Stemp[0, :])
        S[:, 0:Ltemp] = S[:, 0:Ltemp] + Stemp
    S = colcircshift(S, np.arange(-winLen + 1, len(x) + 1, hop) % nfft) if fshift else np.roll(S, - winLen + 1)
    S = np.fft.fft(S, nfft, 0)
    return S


def colcircshift(X, shifts):
    X = np.array(X)
    for i in range(len(X[0, :])):
        X[:, i] = np.roll(X[:, i], shifts[i])
    return X


def parseInputs(x, fbparams):
    if len(x) <= 0:
        raise ValueError('vector input is required for x')
    if type(fbparams['fshift']) != bool and fbparams['fshift'] not in {0, 1}:
        raise TypeError('fshift must be boolean')
    if fbparams['stft']:
        winlen = len(fbparams['afilters'][0])
        if winlen < 2:
            raise ValueError('winlen must be greather than 1')
        if fbparams['dfactor'] is list and len(fbparams['dfactor'] > 1):
            raise ValueError('dfactor is scalar')
        if fbparams['dfactor'] < 1 or fbparams['dfactor'] is not int:
            raise ValueError('dfactor must be positive integer greater than zero')
        if fbparams['numhalfbands'] < 1 or winlen / fbparams['numhalfbands'] % 2 != 1:
            raise ValueError('The analysis window length divided by numhalfbands must be an odd integer.')
    else:
        if not np.all((np.array(fbparams['centers'][1:]) - np.array(fbparams['centers'][:-1])) > 0):
            raise ValueError('subband center frequencies must be strictly increasing')
        elif min(fbparams['centers']) < 0 or max(fbparams['centers']) > 1:
            raise ValueError('center freqs must be in range [0 1]')
        if min(fbparams['bandwidths']) <= 0 or max(fbparams['bandwidths']) >= 2:
            raise ValueError('bandwidths must be in range [0 2], non inclusive')
        elif len(fbparams['bandwidths']) > 1 and len(fbparams['bandwidths']) != fbparams['numbands']:
            raise ValueError('The number of subband bandwidths must be one or equal to the number of subband centers.')
        if type(fbparams['dfactor']) in {np.ndarray, list} and len(fbparams['dfactor']) != fbparams['numbands']:
            raise ValueError('the number of downsampling factor must be one or equal to numbands')
        if type(fbparams['dfactor']) in {np.ndarray, list} and [all(n > 0 for n in lst) for lst in fbparams['dfactor']]:
            raise ValueError('dfactors must be all positive')
