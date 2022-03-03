import numpy as np
import scipy.signal as signal


def filterbanksynth(S: np.array, filterbankparams):
    parseInputs(S, filterbankparams)
    if filterbankparams['stft']:  # Synthesis based on the inverse short-time Fourier transform
        return synthesis_by_stft(S, filterbankparams)
    return synthesis_by_filterbank(S, filterbankparams)  # General multirate filterbank synthesis


def synthesis_by_filterbank(S, filterbankparams):
    dfactor = filterbankparams['dfactor']
    afilterorders = filterbankparams['afilterorders']
    keeptransients = filterbankparams['keeptransients']
    L = len(S[0, :])  # final post-processing len
    finalLen = min(L * dfactor - afilterorders) if keeptransients else min(L * dfactor)
    y = np.zeros((1, finalLen))

    for k in range(len(filterbankparams['numbands'])):
        center = index(filterbankparams['centers'], k)  # center frequency
        g = index(filterbankparams['sfilters'], k)  # FIR synthesis kernel
        dfactor = index(filterbankparams['dfactor'], k)  # downsample factor
        order1 = index(filterbankparams['afilterorders'], k)  # decimating filter order, analysis
        order2 = index(filterbankparams['sfilterorders'], k)  # interpolating filter order, synthesis

        if center != 1:  # complex phase term for undoing the causal phase characteristic of the subband filter
            n0 = (len(g) - 1) / 2
            W = np.exp(1j * np.pi * center * n0)
            S[k, :] = np.conj(W) * S[k, :]
        yk = bandpassExpansion(S[k, :], center, g, dfactor, filterbankparams['fshift'])

        if np.shape(S)[0] == filterbankparams['numbands']:  # S came from real-valued signal
            yk = np.real(yk)
        elif np.shape(S)[0] == filterbankparams[
            'numhalfbands'] and center != 0 and center != 1:  # S came from complex-valued signal
            if filterbankparams['centers'][0] == 0:
                ind = filterbankparams['numhalfbands'] - k + 2
            else:
                ind = filterbankparams['numhalfbands'] - k + 1
            yk = yk + bandpassExpansion(W * S[ind, :], -center, g, dfactor, filterbankparams['fshift'])
        grpDelay = order1 + order2 if filterbankparams['keeptransients'] else order2  # compensate for group delay
        yk = yk[1 + grpDelay // 2: 0 - grpDelay // 2]
        y = y + matchLen(yk, finalLen)


def matchLen(x, L): # TODO: test this
    return [x, np.zeros((1, L - len(x)))] if len(x) < L else x[0:L]


def bandpassExpansion(subband, center, g, dfactor, fshift):
    if fshift is False:  # convert to a modulated bandpass filter
        g = vmult(g, np.exp(1j * np.pi * center * np.arange(max(np.shape(g)))))
    if dfactor >= (np.size(g) - 1) / 128:  # interpolate
        yk = signal.upfirdn(x=subband, h=g, up=dfactor, down=1)  # efficient for large upsampling
    else:
        pass
        yk = fastconv(g, upsample(subband.T, dfactor)).T;
    if fshift:  # shift the subbband in frequency back to its original position
        yk = vmult(yk, np.exp(1j * np.pi * center * np.arange(0, len(yk))))
    return yk


def upsampled_frame(sample, dfactor):
    frame = np.zeros(dfactor, dtype=np.complex_).tolist()
    frame[0] = sample
    return frame


def upsample(x: np.array, dfactor):
    lists = [upsampled_frame(i, dfactor) for i in x]
    return sum(lists, [])


def fastconv(h, x):
    z = np.zeros((np.size(h) - 1, np.shape(x)[1]))
    return signal.filtfilt(b=h, x=np.vstack([x, z]))


def vmult(x1, x2):  # multiplies two vectors element-wise, regardless of orientation. Output shape matches x1.
    return x1 * x2.reshape(x1.shape)


def index(l: list, i: int):
    return l[i] if type(l) is list else l


def synthesis_by_stft(S, filterbankparams) -> np.array:
    win = filterbankparams['sfilters'][0]
    hop = filterbankparams['dfactor'][0]
    nfft = filterbankparams['numhalfbands']
    fshift = filterbankparams['fshift']
    freqdownsample = len(filterbankparams['afilters'][0]) / filterbankparams['numhalfbands']
    if np.shape(S)[0] == filterbankparams['numbands']:  # S is associated with subband signals from analytic signal.
        hilbertGain = 1 + int(filterbankparams['centers'] != 0 & (filterbankparams['centers'] != 1))
        # val = np.diag(1 / sparse.csr_matrix(hilbertGain).toarray())
        S = (1 / hilbertGain) * S

        if filterbankparams['numhalfbands'] % 2 == 1:  # odd number of subbands, no nyquist band
            S = np.vstack((S, np.conj(np.flipud(S)[:-1])))
        else:
            S = np.vstack((S, np.conj(S[1:-1])))
    nmid = len(filterbankparams['afilters'][0]) - 1 / 2  # scalar
    W = windowphaseterm(nmid, nfft)
    S = np.diag(W) * S
    if filterbankparams['keeptransients']:  # delay the subband array
        n1 = np.ceil((1 + filterbankparams['afilterorders'][0] / 2) / filterbankparams['dfactor'])
        S = np.hstack([np.zeros((len(S), n1 - 1)), S])  # TODO: should be dtype = np.complex_ ?
    y, grpDelay = ISTFT(S, win, hop, fshift, freqdownsample)
    y = y[np.floor(grpDelay):]
    if filterbankparams['keeptransients']:
        y = y[:-int(np.ceil(grpDelay))]  # accomodate group delay
    else:
        synthDelay = (len(win) - 1) / 2  # remove the synthesis stage transients
        y = y[:-int(np.ceil(synthDelay))]
    if np.linalg.norm(np.im(y)) / np.linalg.norm(np.real(y)) < 1e-3:  # removes small imaginary artifacts
        y = np.real(y)
    return y


def ISTFT(S, win, hop, fshift, freqdownsample):
    win = np.array(win)
    winLen = max(np.shape(win))
    if win.ndim == 1:
        win = win.reshape(1, winLen)  # column vector
    else:
        win = win.reshape(min(np.shape(win)), winLen)
    S = np.array(S)
    if S.ndim == 1:
        S = S.reshape(1, len(S))
    numFrames = len(S[0, :])
    nfft = len(S[:, 0])
    analysisWinLen: int
    if freqdownsample == 1:
        analysisWinLen = winLen
    else:
        analysisWinLen = freqdownsample * nfft

    y = np.zeros((1, (numFrames - 1) * hop + winLen), dtype=np.complex_)
    S = np.fft.ifft(S)

    winPosition = -winLen + 1  # window starts with its left edge at winPosition

    for i in range(numFrames):
        ind = np.arange(winPosition, winPosition + winLen) % nfft
        frame = win * S[ind, i]  # apply synthesis window
        n1 = i * hop
        n2 = n1 + winLen
        y[:, n1:n2] = y[:, n1:n2] + frame
        if fshift:
            winPosition = winPosition + hop
    grpDelay = (winLen - 1 + analysisWinLen - 1) / 2
    return y, grpDelay


def windowphaseterm(nmid, nfft: int):
    W = np.zeros([nfft, 1], dtype=np.complex_)
    if nfft % 2 == 0:  # even-length nfft.
        W[0:nfft // 2] = np.exp(2j * np.pi * np.array(range(nfft // 2)) / nfft * nmid).reshape(nfft // 2, 1)
        W[nfft // 2] = 1
        W[nfft // 2 + 1:] = np.conj(W[1:nfft // 2][::-1])
    else:
        W[0:(nfft // 2) + 1] = np.exp(2j * np.pi * np.array(range((nfft // 2) + 1)) / nfft * nmid).reshape(
            (nfft // 2) + 1, 1)
        W[(nfft // 2) + 1:] = np.conj(W[1:(nfft // 2) + 1][::-1])
    return W


def parseInputs(S: np.array, fbparams):
    winlen = len(fbparams['afilters'][0])
    # if fbparams['stft']:
    if winlen < 2:
        ValueError('window length must be greater than 1')
    if type(fbparams['fshift']) is not bool:
        TypeError('fshift must be boolean')
    if type(fbparams['dfactor']) is not int or fbparams['dfactor'] <= 0:
        TypeError('dfactor must be int greater than zero')
    if winlen > fbparams['numhalfbands'] and winlen / fbparams['numhalfbands'] % 2 != 1:
        ValueError('The analysis window length divided by numhalfbands must be an odd integer.')
    if not all(fbparams['centers'][i] <= fbparams['centers'][i + 1] for i in range(len(fbparams['centers']) - 1)):
        raise ValueError('subband center frequencies must be strictly increasing')
    if min(fbparams['centers']) < 0 or max(fbparams['centers']) > 1:
        raise ValueError('subband center frequencies must be between 0 and 1, inclusive')
    if min(fbparams['bandwidths']) <= 0 or max(fbparams['bandwidths']) >= 2:
        raise ValueError('subband bandwidths must be between 0 and 2, non-inclusive')
    if len(fbparams['bandwidths']) > 1 and len(fbparams['bandwidths']) != len(fbparams['centers']):
        raise ValueError('The number of subband bandwidths must be one or equal to the number of subband centers.')
    t_b_zip = zip(fbparams['transbands'], fbparams['bandwidths'])
    if sum([1 if t <= 0 else 0 for t in fbparams['transbands']]) or sum([1 if t > b else 0 for t, b in t_b_zip]):
        raise ValueError('Subband transition bandwidths must be nonzero positive and less than the -6dB bandwidths.')
    if np.shape(S)[0] != fbparams['numbands'] and np.shape(S)[0] != fbparams['numhalfbands']:
        raise IndexError('rows count in S matrix must equal the number of subbands specified by FILTBANKPARAMS.')
