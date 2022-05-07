import librosa
import scipy
import torch
from scipy import signal, interpolate
from modulation_toolbox_py.filterbank.filtersubbands import buffer
import numpy as np
import torchaudio.functional as torch_f


def detectPitch(x: np.ndarray, fs: float, voicingSens=.5, medFiltLen=3, freqCutoff=2000, display=False,
                interpMethod='linear'):
    """Detects the fundamental frequency of a harmonic signal with nonharmonic segments interpolated by straight
    lines """
    checkInputs(x, fs, medFiltLen, voicingSens, freqCutoff, display)
    x = x.T if x.shape[0] == 1 else x
    isRowVector = x.shape[0] == 1
    origLen = len(x)
    origFs = fs
    dfactor = max(1, int(np.floor(fs / 2 / freqCutoff)))
    x = signal.decimate(x.T, dfactor).T
    fs = fs / dfactor
    F0min = 50
    F0max = 600
    winDur = .05
    winHop = int(np.ceil(.025 * fs))
    winLen = int(np.ceil(winDur * fs))
    ARorder = 12
    modelOrder = 4
    numFrames = int(np.ceil(len(x) / winHop))
    xcorrlen = 2 * winLen - 1
    win = np.hamming(winLen)
    win = win / scipy.linalg.norm(winLen)
    B = buffer2(x, winLen, winHop, -winHop, numFrames)
    R = np.real(np.fft.ifft(np.abs(np.fft.fft(np.diag(win) @ B, xcorrlen)) ** 2))
    R = R[0:winLen, :]
    frameRMS = np.sqrt(sum(np.abs(B) ** 2))
    voicing = signal.medfilt(detectVoicing(np.log(frameRMS), voicingSens).astype(int), 3)
    if not any(voicing):
        voicing = np.ones(np.size(voicing))
        print('No voicing detected, so this signal will be treated as ''voiced''. Consider revising parameter values.')
    F0m = np.zeros((numFrames, 1))
    for i in range(numFrames):
        if not voicing[i]:
            continue
        if i > 0:
            bw = whiten(B[:, i].reshape(-1, 1), ARorder, B[winHop - ARorder: winHop, i])
        else:
            bw = whiten(B[:, i].reshape(-1, 1), ARorder)
        R2 = np.correlate(bw.ravel(), bw.ravel(), mode='full')[winLen - 1:].reshape(-1, 1)
        numPeaks = useHalfSampleDelay = 1
        peakInd1, _ = findPeaks(R[:, i].reshape(-1, 1), int(np.round(fs / F0max)), numPeaks, useHalfSampleDelay)
        peakInd2, _ = findPeaks(R2, np.round(fs / F0max), numPeaks, useHalfSampleDelay)
        tempF01 = fs / (peakInd1 - 1)
        tempF02 = fs / (peakInd2 - 1)
        tempF01viable = peakInd1 != -1 and F0max >= tempF01 >= F0min
        tempF02viable = peakInd2 != -1 and F0max >= tempF02 >= F0min

        lastVoicedF0 = np.where(F0m[:i - 1] > 0)[-1]
        diff1 = np.abs(F0m[lastVoicedF0] - tempF01) if lastVoicedF0 is None else np.abs(tempF01 - (F0max + F0min) / 2)
        diff2 = np.abs(F0m[lastVoicedF0] - tempF02) if lastVoicedF0 is None else np.abs(tempF02 - (F0max + F0min) / 2)
        if not tempF01viable and not tempF02viable:
            voicing[i] = 0
            continue
        elif diff1 <= diff2 or not tempF02viable:
            tempF0 = tempF01
        elif diff1 > diff2 or not tempF01viable:
            tempF0 = tempF02
        else:
            voicing[i] = 0
            continue
        freqRes = int(round(fs / winLen))
        numRecursions = int(np.floor(-np.log(1 / freqRes) / np.log(5)))
        for k in range(numRecursions):
            lowerBound = int(round((tempF0 - freqRes) / 2))
            upperBound = int(round((tempF0 + freqRes) / 2))
            F0step = int(round(freqRes / 5))
            tempF0 = lsharm_freqtrack(x=B[:, i], freqs=np.arange(lowerBound, upperBound + F0step, F0step), fs=fs,
                                      weights=np.ones((modelOrder, 1)))
            freqRes = F0step
        F0m[i] = tempF0
    F0m[F0m < F0min] = 0
    F0m[F0m > F0max] = 0
    F0m = signal.medfilt(F0m.ravel().T, medFiltLen).reshape((-1, 1))
    F0 = F0m
    voicing = F0 != 0
    if all(F0 == 0):
        F0 = np.zeros((1, origLen)) if isRowVector else np.zeros((origLen, 1))
        return F0, F0m, voicing
    voicing2 = voicing
    if voicing[0] == 0:
        F0[0] = F0[np.where(voicing2 == 1)[0][0]]
        voicing2[0] = 1
    if voicing[-1] == 0:
        F0[-1] = F0[np.where(voicing2 == 1)[0][-1]]
        voicing2[-1] = 1
    vSamples = np.argwhere(voicing2 == 1)[:, 0]
    if interpMethod == "linear":

        fn = interpolate.interp1d(vSamples, F0[vSamples].ravel())
        F0 = fn(np.arange(0, len(F0m)))
    else:
        F0 = interpolate.pchip_interpolate(vSamples, F0[vSamples], np.arange(0, len(F0m)))
    L = min(4, int(np.floor((numFrames - 1) / 2)))
    F0 = factorinterp(F0, dfactor * winHop, L, .25)
    F0 = F0[0:origLen]

    if display:  # TODO
        print('TODO')
    F0, F0m = F0 / origFs * 2, F0m / origFs * 2
    if isRowVector:
        F0 = F0.T
        F0m = F0m.T
        voicing = voicing.T
    return F0, F0m, voicing


def findPeaks(x, minDistance: int, numPeaks, halfSampleShift):
    x = np.array(x)
    minDistance = int(minDistance)
    if len(np.shape(x)) == 2 and x.shape[1] == 1:
        transposed = True
        x = x.T
        x = x.ravel()
    else:
        transposed = False
    if halfSampleShift:
        p1, v1 = findPeaks(x, minDistance, numPeaks, 0)
        p2, v2 = findPeaks(nonIntGroupDelay(x, .5), minDistance, numPeaks, 0)
        localPeakPos = np.vstack((p1, p2 - .5))
        localPeakVal = np.vstack((v1, v2))
    else:
        x[:minDistance] = x[minDistance - 1]
        localPeaks = (x[1:-1] > x[:-2]) & (x[1:-1] > x[2:])
        localPeaks = localPeaks.reshape(-1, 1) if type(localPeaks) is not list and x.shape[0] > 1 and len(
            x.shape) == 1 else localPeaks
        localPeaks = np.vstack(([False], localPeaks, [False]))
        localPeakPos = np.where(localPeaks)[0]
        localPeakVal = x[localPeakPos]
    # sortedPeaks = [] if not len(localPeakVal) else np.sort(localPeakVal)[::-1]
    idx = [] if not len(localPeakVal) else np.argsort(localPeakVal).astype('int')[::-1]
    if numPeaks <= len(idx):
        peakPosOut = localPeakPos[idx[:numPeaks]]
        peakValOut = localPeakVal[idx[:numPeaks]]
    else:
        peakPosOut = np.concatenate(
            (np.zeros((0, 1)) if not len(idx) else localPeakPos[idx], -np.ones((numPeaks - len(idx), 1))))
        peakValOut = np.concatenate(
            (np.zeros((0, 1)) if not len(idx) else localPeakVal[idx], np.empty((numPeaks - len(idx), 1), dtype=object)))
    if transposed:
        peakPosOut = peakPosOut.T
        peakValOut = peakValOut.T
    return peakPosOut.ravel()[0], peakValOut.ravel()[0]


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


def factorinterp(x, R, L, cutoff):
    factors = factor(R)
    y = x
    for i in np.arange(0, len(factors)):
        y = torch_f.resample(torch.tensor(y, dtype=torch.float), orig_freq=1, new_freq=factors[i],
                             lowpass_filter_width=len(x) // 4).detach().numpy()
    return y


# https://github.com/staticfloat/libsquiggly/blob/79c63c119a60e2e9c558aefcda6b1c1ac413a47a/libsquiggly/instfreq/lsharm.py
def lsharm_freqtrack(x, freqs=None, weights=None, fs=2.0, skip=1):
    win_len = len(x)
    if weights is None:
        weights = [1, .5, .5]
    if freqs is None:
        freqs = np.linspace(.5 * fs / len(weights), fs / len(weights), win_len)

    # Build Goertzel filterbanks, excluding any filters that exceed nyquist
    fundamental_filters = {}
    for f0 in freqs:
        filterbank = []
        max_order = min(len(weights), int(fs / (2 * f0)))
        for k in range(max_order):
            b = weights[k] * np.array([1, -np.exp(-2j * np.pi * (k + 1) * f0 / fs)])
            a = weights[k] * \
                np.array([1, -2 * np.cos(2 * np.pi * (k + 1) * f0 / fs), 1])
            filterbank += [(b, a)]
        fundamental_filters[f0] = filterbank

    # Zero-pad x so that we can actually perform the Goertzel-filtering at
    # every point
    datalen = len(x)
    x = np.hstack((np.zeros(win_len // 2), x, np.zeros(win_len // 2)))
    lsharm_estimate = np.zeros(datalen // skip)

    for idx in range(datalen // skip):
        # Grab the window of data centered about x[idx] in the original
        # non-padded signal
        windowed_x = x[idx * skip:idx * skip + win_len]

        # Calculate total power of each fundamental frequency, as well as error
        P = np.zeros(len(freqs))

        # For each fundamental frequency, apply Goertzel filters to data:
        for f_idx in range(len(freqs)):
            f0 = freqs[f_idx]
            filterbank = fundamental_filters[f0]
            C = np.zeros(len(filterbank), dtype=complex)

            for k in range(len(filterbank)):
                # Filter with Goertzel filter
                temp = scipy.signal.lfilter(filterbank[k][0], filterbank[k][1], windowed_x)

                # Phase-correct last element of Goertzel filter and square it
                # away in C
                C[k] = np.exp(-2j * np.pi * (k + 1) * f0 / fs *
                              (win_len - 1)) * temp[-1] / np.sqrt(win_len)

            # Store total power of this fundamental frequency
            P[f_idx] = np.sqrt(np.real(np.vdot(C, C)))

        # Save highest frequency estimate into lsharm_estimage
        lsharm_estimate[idx] = freqs[np.argmax(P)]

    # Return the goods, after upsampling them
    # return np.resample(lsharm_estimate, datalen)
    #  Select the f0 which describes the data in terms of capturing the most signal energy...
    maxP, index = np.max(P), np.argmax(P)
    f0 = freqs[index]
    return f0


def nonIntGroupDelay(x: np.array, delay):
    x = x.reshape(-1, 1) if type(x) is not list and x.shape[0] > 1 and len(x.shape) == 1 else np.array(x, ndmin=2)
    if len(x.shape) == 2 and x.shape[0] == 1:
        len_x = x.shape[1]
    elif len(x.shape) == 2 and x.shape[1] == 1:
        len_x = x.shape[0]
    else:
        raise IndexError('should be 2 dimensional')
    DCIndex = int(np.floor(len_x / 2))
    linearPhaseTerm = np.exp(-2j * np.pi * delay / len_x * (np.arange(len_x) - DCIndex).T)
    linearPhaseTerm = np.roll(linearPhaseTerm, -DCIndex).reshape(-1, 1)
    X_delayed = np.fft.fft(x, axis=-1 if x.shape[0] == 1 else 0) * linearPhaseTerm
    xs = np.real(np.fft.ifft(X_delayed, axis=0))
    return xs


def bimodalGaussianMixture(x, numIter=25, numTrials=10) -> (dict, dict):
    theta1 = dict()
    theta2 = dict()
    N = len(x)
    prevMaxL = -np.inf
    for k in range(numTrials):
        mu1 = x[int(np.round((N - 1) * np.random.rand()))]
        sigma1 = np.var(x, ddof=1)
        mu2 = x[int(np.round((N - 1) * np.random.rand()))]
        sigma2 = np.var(x, ddof=1)
        p = np.zeros((numIter + 1, 1))
        p[0] = .5
        for i in range(numIter):
            L1 = p[i] * GaussianLikelihood(x, mu1, sigma1)
            L1orL2 = L1 + (1 - p[i]) * GaussianLikelihood(x, mu2, sigma2)
            LR = L1 / L1orL2
            mu1 = sum(LR * x) / sum(LR)
            mu2 = sum((1 - LR) * x) / sum(1 - LR)
            temp1 = np.sum(LR * (x - mu1) ** 2) / sum(LR)
            temp2 = np.sum((1 - LR) * (x - mu2) ** 2) / sum((1 - LR))

            if temp1 == 0 or temp2 == 0:
                p = np.arange(1, i)
                break
            else:
                sigma1 = temp1
                sigma2 = temp2
            p[i + 1] = sum(LR) / N
        L1 = p[-1] * GaussianLikelihood(x, mu1, sigma1)
        L2 = (1 - p[-1]) * GaussianLikelihood(x, mu2, sigma2)
        L = sum(np.log(L1 + L2))
        if L > prevMaxL:
            prevMaxL = L
            theta1['mu'] = mu1
            theta1['sigma'] = sigma1
            theta1['p'] = p[-1][0]

            theta2['mu'] = mu2
            theta2['sigma'] = sigma2
            theta2['p'] = 1 - p[-1][0]
    return theta1, theta2


def GaussianLikelihood(x, mu, variance):
    x = np.array(x)
    return 1 / np.sqrt(2 * np.pi * variance) * np.exp(-(x - mu) ** 2 / 2 / variance)


def percentile(x, p):
    x = np.sort(x)
    return x[int(np.round(p * (len(x) - 1)))]


def detectVoicing(logRMS, voicingSensitivity=.5):
    logRMS = np.array(logRMS)
    logRMS[np.isinf(logRMS)] = min(logRMS[np.invert(np.isinf(logRMS))])
    numTrials = 10
    numIter = int(np.round(len(logRMS) / 4))
    d1, d2 = bimodalGaussianMixture(logRMS, numIter, numTrials)
    q = [d2['sigma'] - d1['sigma'],
         -2 * (d2['sigma'] * d1['mu'] - d1['sigma'] * d2['mu']),
         d2['sigma'] * d1['mu'] ** 2
         - d1['sigma'] * d2['mu'] ** 2
         - 2 * d1['sigma'] * d2['sigma'] * np.log(d1['p'] / d2['p'] * np.sqrt(d2['sigma'] / d1['sigma']))]
    if np.abs(q[0]) < np.finfo(float).eps < np.abs(d1['mu'] - d2['mu']):
        t = (d2['mu'] ** 2 - d1['mu'] ** 2 + 2 * d2['sigma'] * np.log(d1['p'] / d2['p'])) / 2 / (d2['mu'] - d1['mu'])
        voiced = logRMS > t
        return voiced
    delta = q[1] ** 2 - 4 * q[0] * q[2]
    if delta < 0:
        raise ValueError('real delta was expected')
    t1 = (-q[1] + np.sqrt(delta)) / (2 * q[0])
    t2 = (-q[1] - np.sqrt(delta)) / (2 * q[0])
    t1Between = (d1['mu'] < t1 < d2['mu']) or (d2['mu'] < t1 < d1['mu'])
    t2Between = (d1["mu"] < t2 < d2['mu']) or (d2['mu'] < t2 < d1['mu'])
    if t1Between and not t2Between:
        thresh = t1
    elif t2Between and not t1Between:
        thresh = t2
    else:
        voiced = logRMS > percentile(logRMS, 1 - voicingSensitivity)
        return voiced
    muUnvoiced, muVoiced = (d1['mu'], d2['mu']) if d1['mu'] < d2['mu'] else (d2['mu'], d1['mu'])
    thresh = thresh - 2 * (voicingSensitivity - 0.5) * (
        thresh - muUnvoiced) if voicingSensitivity > 0.5 else thresh + 2 * (0.5 - voicingSensitivity) * (
        muVoiced - thresh)
    voiced = logRMS > thresh
    return voiced


# https://stackoverflow.com/questions/53081956/basic-linear-prediction-example
def custom_lpc(y, m):
    "Return m linear predictive coefficients for sequence y using Levinson-Durbin prediction algorithm"
    # step 1: compute autoregression coefficients R_0, ..., R_m
    R = [y.dot(y)]
    if R[0] == 0:
        return [1] + [0] * (m - 2) + [-1]
    else:
        for i in range(1, m + 1):
            r = y[i:].dot(y[:-i])
            R.append(r)
        R = np.array(R)
        # step 2:
        A = np.array([1, -R[1] / R[0]])
        E = R[0] + R[1] * A[1]
        for k in range(1, m):
            if (E == 0):
                E = 10e-17
            alpha = - A[:k + 1].dot(R[k + 1:0:-1]) / E
            A = np.hstack([A, 0])
            A = A + alpha * A[::-1]
            E *= (1 - alpha ** 2)
        return A


def whiten(x, ARorder: int, xPrev=None):
    a = custom_lpc(y=x.ravel(), m=ARorder)
    hankel_arg = x.ravel()[ARorder - 1::-1] if xPrev is None else xPrev
    zi = a[:0:-1] @ scipy.linalg.hankel(hankel_arg)
    if len(x.shape) != 2:
        raise ValueError('shape must be matrix')
    y, zf = signal.lfilter(b=a, a=1, x=x, zi=zi.reshape((1, -1)))
    return y


def buffer2(x, winlen: int, hop: int, startindex: int, numframes: int):
    x = x.reshape(1, -1) if type(x) is np.ndarray and len(x.shape) == 1 else x
    if startindex <= -1:
        x = x.T if not x.shape[1] == 1 else x
        prepend = np.zeros((-startindex, 1))
        x = np.vstack((prepend, x))
    else:
        x = x[1 + startindex:]
    data_len = winlen + (numframes - 1) * hop
    if data_len - len(x) >= 0:
        x = np.vstack((x, np.zeros((data_len - len(x), 1))))
    return buffer(x, winlen, winlen - hop, 'nodelay') if hop <= winlen else buffer(x, winlen, winlen - hop, 0)


def checkInputs(_, fs, medFiltLen, voicingSens, freqCutoff, display):
    if fs <= 0:
        raise ValueError('sample rate must be positive')
    if medFiltLen <= 0:
        raise ValueError('medfiltLen must be positive')
    if voicingSens > 1 or voicingSens < 0:
        raise ValueError('voicing sensibility must be between 0 and 1 inclusive')
    if freqCutoff < 0 or freqCutoff > fs / 2:
        raise ValueError('freq cutoff must be under fs/2')
