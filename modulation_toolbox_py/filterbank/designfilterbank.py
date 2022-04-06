import numpy as np
from scipy import signal


def designfilterbank(centers: list, bandwidths: list, transbands=None, dfactor: int = 1,
                     keeptransients: bool = 1):
    if transbands is None:
        transbands = []
    fshift = 1 if dfactor > 1 else 0
    if not transbands:  # if transband list is empty
        transbands = [b / 10 for b in bandwidths]

    filterbankparams = parseInputs(centers, bandwidths, transbands, dfactor, fshift, keeptransients)
    for i in range(len(filterbankparams['bandwidths'])):
        B = filterbankparams['bandwidths'][i]
        wc = B / 2
        trans = index(transbands, i)
        order = nearestEven(np.ceil(6.6 / trans))
        h = signal.firwin(numtaps=int(order), cutoff=wc)
        filterbankparams['afilters'].append(h)
        filterbankparams['afilterorders'].append(order)
        _dfactor = index(filterbankparams['dfactor'], i)
        filterbankparams['sfilters'].append(_dfactor * h if _dfactor > 1 else 1)
        filterbankparams['sfilterorders'].append(order if _dfactor > 1 else 0)
    return filterbankparams


def index(l: list, i: int):
    return l[i] if type(l) is list else l


def parseInputs(centers, bandwidths, transbands, dfactor, fshift, keeptransients) -> dict:
    if not all(centers[i] <= centers[i + 1] for i in range(len(centers) - 1)):
        raise ValueError('subband center frequencies must be strictly increasing')
    if min(centers) < 0 or max(centers) > 1:
        raise ValueError('subband center frequencies must be between 0 and 1, inclusive')
    if min(bandwidths) <= 0 or max(bandwidths) >= 2:
        raise ValueError('subband bandwidths must be between 0 and 2, non-inclusive')
    if len(bandwidths) > 1 and len(bandwidths) != len(centers):
        raise ValueError('The number of subband bandwidths must be one or equal to the number of subband centers.')
    t_b_zip = zip(transbands, bandwidths)
    if sum([1 if t <= 0 else 0 for t in transbands]) or sum([1 if t > b else 0 for t, b in t_b_zip]):
        raise ValueError('Subband transition bandwidths must be nonzero positive and less than the -6dB bandwidths.')
    if dfactor <= 0:
        raise ValueError('dfactor must be greater than zero')
    numbands = len(centers)
    numhalfbands: int = 0
    if centers[0] == 0 or centers[-1] == 1:
        numhalfbands = 2 * numbands - 1
    if centers[0] == 0 and centers[-1] == 1:
        numhalfbands = 2 * numbands - 2
    if not (centers[0] == 0 or centers[-1] == 1):
        numhalfbands = 2 * numbands
    filterbankparams = dict()
    filterbankparams['numbands'] = numbands
    filterbankparams['numhalfbands'] = numhalfbands
    filterbankparams['dfactor'] = dfactor
    filterbankparams['centers'] = centers
    filterbankparams['bandwidths'] = bandwidths
    filterbankparams['afilters'] = []
    filterbankparams['afilterorders'] = []
    filterbankparams['sfilters'] = []
    filterbankparams['sfilterorders'] = []
    filterbankparams['fshift'] = fshift
    filterbankparams['stft'] = 0
    filterbankparams['keeptransients'] = keeptransients
    return filterbankparams


def nearestEven(x: float) -> float:
    if x % 2 == 0:
        return x
    if x % 2 < 1:
        return np.floor(x)
    if x % 2 > 1:
        return np.ceil(x)
    if x % 2 == 1:
        return x + 1
