import numpy as np

from modulation_toolbox_py.filterbank.designfilterbank import designfilterbank
from modulation_toolbox_py.filterbank.filtersubbands import filtersubbands


def modsynth(M: np.array, C, data, verbose) -> np.array:
    if type(C) == str:
         C = getVocoderCarriers(C, data, np.shape(M))
    if 'filtbankparams' in data:
        S = modrecon(M, C)

       

def getVocoderCarriers(method, data, modsize: tuple) -> np.array:
    numcarriers = modsize[0]
    if method == 'noise':
        if 'filtbankparams' in data:
            C = filtersubbands(np.random.rand(data['origlen'], 1), data['filtbankparams'])
        else: # noise-excited filterbank consisting of subbands
            medFO = np.median(data['FO'] / data['fs'] * 2)
            filtbank = designfilterbank(np.arange(1,numcarriers + 1) * medFO, list(medFO), [], 1)
            C = filtersubbands(np.random.rand(data['origlen'], 1), filtbank)
            difflen = np.shape(C)[1] - data['origlen']
            C = C[:, int(np.ceil(difflen/2)) : - int(np.floor(difflen/2))]
            C = 2 / medFO / numcarriers * C
    elif method == 'sine':
        if 'filtbankparams' in data:
            C = np.ones(modsize)
        else:
            medF0 = np.median(data['F0'] / data['fs'] * 2)
            C = np.exp(1j * np.pi * medF0 * np.reshape(np.arange(1, numcarriers + 1), (-1,1)) * np.arange(0, data['origlen']))
    else:
        raise Exception('unrecognized vocoder method: ', method)

