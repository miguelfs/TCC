import numpy as np

from modulation_toolbox_py.demod.moddecompcog import moddecompcog
from modulation_toolbox_py.demod.moddecomphilb import moddecomphilb
from modulation_toolbox_py.filterbank.cutoffs2fbdesign import cutoffs2fbdesign
from modulation_toolbox_py.filterbank.designfilterbank import designfilterbank
from modulation_toolbox_py.filterbank.designfilterbank_stft import designfilterbankstft
from modulation_toolbox_py.filterbank.filtersubbands import filtersubbands


def moddecomp(x: np.array, fs: float, demodparams=('cog', 0.1, 0.05), subbands=150, dfactor: any = 1,
              verbose: any = ''):
    filtbankparams, demodparams, dfactor, modfs, verbose = parseinputs(x, fs, demodparams, subbands, dfactor, verbose)
    M, C, F0, modbandwidth = demod(x, fs, filtbankparams, demodparams, dfactor)


def demod(x: np.ndarray, fs: float, filterbankparams, demodparams, dfactor):
    if type(demodparams) != list or type(demodparams) != tuple:
        raise TypeError('demodparams must be a cell array')
    if demodparams[0] == 'hilb' or demodparams[0] == 'hilbert':
        S = filtersubbands(x, filterbankparams)
        M, C, _ = moddecomphilb(S)
        modbandwidth = min(fs/filterbankparams['dfactor'], max(filterbankparams['bandwidths']/2*fs))
    elif demodparams[0] == 'cog':
        S = filtersubbands(x, filterbankparams)
        M, C, _ = moddecompcog(S, carrWin= demodparams[1], carrWinHop=demodparams[2], centers=demodparams[3], bandwidths=demodparams[5])
        modbandwidth = min(fs/filterbankparams['dfactor'], max(filterbankparams['bandwidths']/2*fs))
    elif demodparams[0] == 'harm' or demodparams[1] == 'harmonic':
        F0 = detectpitch()
    return None, None, None, None
    pass
    # if demodparams['which'] == 'hilb' or demodparams['which'] == 'hilbert':
    # S = filtersubbands(x, filtbankparams);
    # M, C = moddecomphilb()


def parseinputs(x, fs, demod, subbands, dfactor, verbose=''):
    demod = ['cog'] if demod is None else None
    if len(np.shape(x)) > 1:
        raise IndexError('time series x must be 1D')
    if fs is None or fs <= 0:
        raise ValueError('fs must be a positive number')
    verbose = True if verbose == 'verbose' else False
    if type(demod[0]) != str:
        raise ValueError('filter type must be str')
    demodparams = demod[0]
    if subbands is not None:
        freqdiv = subbands
    elif demod[0] == 'harm' or demod[0] == 'harmcog':
        # the default bandwidth will depend on the detected pitch
        freqdiv = []
    else:
        freqdiv = 150 if fs > 150 else fs / 4

    # check subband cutoff frequencies
    if len(freqdiv) > 1 and ((min(freqdiv) < 0) or (max(freqdiv) > fs / 2)):
        raise ValueError('subband cutoffs must be in range [0, fs/2] inclusive')
    elif len(freqdiv) == 1 and ((min(freqdiv) <= 0) or (max(freqdiv) >= fs / 2)):
        raise ValueError('subband cutoffs must be in range [0, fs/2] non-inclusive')
    # check decimation factor
    if type(dfactor) == int and dfactor <= 0:
        raise ValueError('dfactor must be positive')
    if type(dfactor) == str and dfactor != 'maximal':
        raise ValueError('invalid dfactor string value')
    filtbankparams, freqdiv, dfactor = parsefilterbank(fs, demodparams, freqdiv, dfactor)
    modfs = fs / dfactor
    fulldemodparams = parsedemod(fs, demodparams, filtbankparams, freqdiv, dfactor, modfs)
    return filtbankparams, fulldemodparams, dfactor, modfs, verbose


def parsedemod(fs, demodparams, filtbankparams, freqdiv, dfactor, modfs):
    if demodparams[0] == 'cog' or demodparams[0] == 'harmcog':
        cogwinlen = .1 if (len(demodparams) < 2 or len(demodparams[1]) == 0) else demodparams[1]
        cogwinhop = cogwinlen / 2 if (len(demodparams) < 3 or len(demodparams[2]) == 0) else demodparams[2]
    if demodparams[0] == 'cog':
        cogcenters = 0 if filtbankparams['stft'] else filtbankparams['centers']
        cogbandwidths = dfactor * filtbankparams['bandwidths']
        if any(i > 2 for i in cogbandwidths):
            cogbandwidths = min(cogbandwidths, 2 + np.zeros(len(cogbandwidths)))
    if demodparams[0] == 'harm' or demodparams[0] == 'harmcog':
        index = 2 * (demodparams[0] == 'harm') + 4 * (demodparams[0] == 'harmcog')
        numcarriers = [] if len(demodparams) < index else demodparams[index]
        voicingsens = .5 if len(demodparams) < index + 1 or type(demodparams[index + 1]) is not list else \
            demodparams[index + 1]
        F0smoothness: int = 2 if len(demodparams) < index + 2 or type(demodparams[index + 2] is not list) else \
            demodparams[index + 2]
        if F0smoothness < 1 or F0smoothness % 1 != 0:
            raise ValueError('FOsmoothness must be a positive integer')
        elif F0smoothness <= 3:
            medfiltlen = 1 + 2 * (F0smoothness - 1)
            F0freqrange = min(2000, fs / 2)
        else:
            medfiltlen = 5
            F0freqrange = min(2000 * (F0smoothness - 2), fs / 2)
        modbandwidth = freqdiv / fs * 2
    if demodparams[0] == 'hilb' or demodparams[0] == 'hilbert':
        fulldemodparams = ['hilb']
    elif demodparams[0] == 'cog':
        fulldemodparams = ['cog', np.ceil(modfs * cogwinlen), np.ceil(modfs * cogwinhop), cogcenters, cogbandwidths]
    elif demodparams[0] == 'harm' or demodparams[0] == 'harmonic':
        fulldemodparams = ['harm', voicingsens, medfiltlen, F0freqrange, numcarriers, modbandwidth]
    elif demodparams[0] == 'harmcog':
        fulldemodparams = ['harmcog', voicingsens, medfiltlen, F0freqrange, np.ceil(fs * cogwinlen),
                           np.ceil(fs * cogwinhop), numcarriers, modbandwidth]
    else:
        raise ValueError('unrecognized demodulation method')
    return fulldemodparams


def parsefilterbank(fs, demodparams, freqdiv, dfactor):
    # filtbankparams = None
    if demodparams[0] == 'harm' or demodparams[0] == 'harmcog':
        # harmonic-based demodulation
        if freqdiv is list and len(freqdiv) > 1:
            raise ValueError('freqdiv must be scalar > fs/2')
        filtbankparams = dict()
        if dfactor == 'maximal':
            freqdiv = 75 if freqdiv is None else np.floor(fs / 2 / freqdiv)
    elif len(freqdiv) == 1:
        # uniformly spaced filterbank subband demodulation
        numhalfbands = np.floor(fs / freqdiv)
        sharpness = 15
        dfactor = np.floor(fs / 2 / freqdiv) if dfactor == 'maximal' else dfactor
        filtbankparams = designfilterbankstft(numhalfbands, sharpness, dfactor, keeptransients=False)
    elif len(freqdiv) > 1:
        # non-uniformly spaced filterbank subband demodulation
        subbandcenters, subbandwidths = cutoffs2fbdesign(np.array(freqdiv) / fs * 2)
        dfactor = max(1, np.floor(1 / max(subbandwidths))) if dfactor == 'maximal' else dfactor
        filtbankparams = designfilterbank(subbandcenters, subbandwidths, [], dfactor, keeptransients=False)
    else:  # TODO: remove this
        raise ValueError('non mapped filtbankparams')
    if filtbankparams is list and len(filtbankparams) > 0 and fs / dfactor < max(fs / 2 * filtbankparams['bandwidths']):
        raise ValueError('FS/DFACTOR should be greater than min( SUBBANDS ) to avoid aliasing')
    return filtbankparams, freqdiv, dfactor
