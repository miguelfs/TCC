import numpy as np
import scipy.special
from numpy.linalg import pinv
from scipy.fft import fft


def designfilterbankstft(numhalfbands: int, sharpness: int = 1, dfactor: int = 1, keeptransients: bool = True):
    fshift = 1 if dfactor > 1 else 0
    winlen = numhalfbands * sharpness
    aOrder = winlen - 1
    errorCheck(numhalfbands, sharpness, dfactor, fshift)
    # design analysis window
    n = np.arange((-winlen + 1) / 2, (winlen + 1) / 2, 1)
    analysisWin = generate_analysis_win(n, sharpness, winlen)
    synthWin = STFTwindow(analysisWin, dfactor, numhalfbands)
    sOrder = len(synthWin) - 1
    bandwidths = windowBandwidth(analysisWin, 1 / np.sqrt(2))
    numbands = int(np.floor(numhalfbands / 2) + 1)
    centers = np.arange(0, ((2 * numbands) - 1) / numhalfbands, 2 / numhalfbands)
    filtbankparams = dict()
    filtbankparams['numbands'] = numbands
    filtbankparams['numhalfbands'] = numhalfbands
    filtbankparams['dfactor'] = dfactor
    filtbankparams['centers'] = centers
    filtbankparams['bandwidths'] = bandwidths
    filtbankparams['afilters'] = [analysisWin]
    filtbankparams['afilterorders'] = aOrder
    filtbankparams['sfilters'] = [synthWin]
    filtbankparams['sfilterorders'] = sOrder
    filtbankparams['fshift'] = fshift
    filtbankparams['stft'] = 1
    filtbankparams['keeptransients'] = keeptransients
    return filtbankparams


def generate_analysis_win(n, sharpness, winlen):
    analysisWin = np.hamming(winlen)
    analysisWin = analysisWin * scipy.special.diric((2 * np.pi / winlen * n), sharpness)
    analysisWin = analysisWin / sum(analysisWin)
    return analysisWin


def errorCheck(numhalfbands, sharpness, dfactor, fshift):
    if sharpness % 2 == 0 or sharpness <= 0:
        raise ValueError('sharpness must be odd integer greater than 0')
    if numhalfbands <= 0:
        raise ValueError('numhalfbands must be greater than 0')
    if dfactor <= 0:
        raise ValueError('dfactor must be greater than 0')


def STFTwindow(window, hopDistance, DFTsize):
    transpose = True if (len(np.shape(window)) == 2 and np.shape(window)[0] == 1) or np.shape(window) == 1 else False
    window = window.reshape(-1, 1)
    L1 = len(window)
    L2 = DFTsize

    M = L1 / L2  # downsampling rate
    M2 = (M - 1) / 2
    R = hopDistance

    if M % 1 != 0 or M % 2 == 0:
        ValueError('window length must be divisible by DFT size with odd value')

    F = np.zeros((int(M * min(R, L2)), L2))  # down-sampled subsets of analysis window
    D = np.zeros((int(M * min(R, L2)), 1))
    for p in range(int(M)):
        for n in range(min(int(R), int(L2))):
            temp = upsample(downsample(window[p * L2: (p + 1) * L2], R, n), R, n)
            if len(temp) < L2:
                temp = np.vstack((temp.reshape(-1, 1), np.zeros((L2 - len(temp), 1))))
            elif len(temp) > L2:
                temp = temp[0:L2]
            F[int(p * min(R, L2) + n), :] = temp.T
            D[int(p * min(R, L2) + n), 0] = p == M2
    compWin = pinv(F) @ D
    compWin = compWin[::-1]
    if transpose:
        compWin = compWin.T
    return compWin


def windowBandwidth(window, magnitude):
    nfft = 4 * len(window)
    nyq = int(np.ceil((nfft + 1) / 2))
    H = np.abs(fft(window, nfft))
    H = H[:nyq] / max(H[:nyq])
    k = np.where(H < magnitude)[0][0]
    bandwidth = 2 * ((k + 1) / nfft * 2)
    return bandwidth


def upsampled_frame(sample, dfactor):
    frame = np.zeros(dfactor).tolist()
    frame[0] = sample
    # frame[0] = sample[0]
    return frame


def upsample(x: np.array, dfactor, phase: int = 0):
    if phase >= dfactor:
        raise ValueError('phase must be lower than dfactor')
    # if len(np.shape(x)) == 2:
    #     x = x.reshape(-1)
    if type(x) is np.ndarray and len(x.shape) == 2 and x.shape[0] == 1:
        lists = [upsampled_frame(i, dfactor) for i in x[0, :]]
    elif type(x) is np.ndarray and len(x.shape) == 2 and x.shape[1] == 1:
        lists = [upsampled_frame(i, dfactor) for i in x[:, 0]]
    else:
        lists = [upsampled_frame(i, dfactor) for i in x]
    y = sum(lists, [])
    return np.roll(y, phase)


def downsample(x, dfactor, phase: int = 0):
    if phase > dfactor:
        raise ValueError('phase must be lower than dfactor')
    if type(x) is np.ndarray and len(x.shape) == 2 and x.shape[0] == 1:
        return x[:, phase::dfactor]
    if type(x) is np.ndarray and len(x.shape) == 2 and x.shape[1] == 1:
        return x[phase::dfactor, :]
    return x[phase::dfactor]
