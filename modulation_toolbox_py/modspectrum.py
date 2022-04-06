import numpy as np
from scipy import signal, interpolate
from modulation_toolbox_py.moddecomp import moddecomp
import matplotlib.pyplot as plt


def modspectrum(x: list, fs: float, decompparams, opinputs, verbose: str = '') -> (np.array, np.array, np.array, any):
    M, C, data = moddecomp(x, fs, decompparams[0], decompparams[1], 'maximal')
    modnfft, modtaper, modtapername, demean, normalize = parseSpectralInputs(opinputs, np.shape(M)[1])
    if normalize:
        M = 1 / np.sqrt(sum(np.abs(M) ** 2, 2)) * M
    if demean:
        M = signal.detrend(M.T, type='constant')
    P = np.fft.fft2(M * modtaper, [modnfft, modnfft])

    if data['demodparams'] == 'harm' or data['demodparams'] == 'harmcog':
        afreqs = list((range(0, np.shape(M)[0])))
        axisLabel = 'Harmonic number'
    else:
        afreqs = np.round(data['filtbankparams']['centers'] * fs / 2)
        axisLabel = 'Acoustic Frequency (Hz)'
    mfreqs = np.arange(0, modnfft) / modnfft * data['modfs']
    if verbose == 'verbose':
        printSpectralParams(modtapername, modnfft, data['modfs'], np.shape(M)[1])
    temp = 20 * np.log10(np.fft.fftshift(np.abs(P), 2))
    if 1 in [min(np.diff(afreqs)) == x for x in np.diff(afreqs)]:
        stepsize = min(50, min(np.diff(afreqs) / 2))
        afreqs2 = np.arange(afreqs[0], afreqs[-1] - stepsize, stepsize)
        # TODO: interp1d below is not safe
        temp = interpolate.interp1d(afreqs, temp, kind='nearest')
        plt.plot(mfreqs - data['modfs'] / 2, afreqs2, temp)
    else:
        plt.plot(mfreqs - data['modfs'] / 2, afreqs, temp)
        plt.xlabel = 'Modulation frequency (Hz)'
        plt.ylabel = axisLabel
        plt.title('Modulation Spectrogram')
        plt.xlim(1.25 * -data['modbandwidth'] / 2, 1.25 * data['modbandwidth'] / 2)


def isnumeric(x: any) -> bool:
    return isinstance(x, (int, float, complex)) and not isinstance(x, bool)


def parseSpectralInputs(inputs, L: int):
    numInputs = len(inputs)
    modnfft: int = L
    modtaper = signal.windows.boxcar(L)
    modtapername = 'rect'
    demean = 0
    normalize = 0
    k = 1
    while k <= numInputs:
        if inputs[k] is None:
            k = k + 1
        if inputs[k] == 'cell':
            extras = inputs[k]
            inputs[k] = []
            inputs = [np.reshape(inputs, 1), np.reshape(extras, 1)]
            numInputs = len(inputs)
        if type(inputs[k]) != int or int(inputs[k]) < 1:
            raise ValueError('must be integer greater than zero')
        k = k + 1
        if inputs[k] == 'hamming':
            modtaper = signal.windows.hamming(L)
            modtapername = 'hamming'
        elif inputs[k] in {'rect', 'rectangle', 'rectangular'}:
            modtaper = signal.windows.boxcar(L)
            modtapername = 'rect'
        elif inputs[k] in {'bartlett', 'bart'}:
            modtaper = signal.windows.bartlett(L)
            modtapername = 'bart'
        elif inputs[k] in {'hann', 'hanning', 'vonhann'}:
            modtaper = signal.windows.hann(L)
            modtapername = 'hann'
        elif inputs[k] in {'demean', 'de-mean'}:
            demean = 1
        elif inputs[k] in {'normalize', 'norm'}:
            normalize = 1
        else:
            raise Exception('unrecognized input ', inputs[k])
        k = k + 1

    modtaper = modtaper / np.linalg.norm(modtaper)
    return modnfft, modtaper, modtapername, demean, normalize


def printSpectralParams(modulationTaperName, nfft, modulationFs, signalLength):
    print('****************************************')
    print(' Modulator signal length (seconds):  ', signalLength / modulationFs)
    print('Modulator sampling rate in hz: ', modulationFs)
    print('Modulator signal length in samples: ', signalLength)
    print('DFT size in samples:', nfft)
    print('Analysis taper shape:', modulationTaperName)
