import numpy as np

def parseInputs(filterband: tuple, filtertype: str, transband: float, dev: tuple):
    for freq in filterband:
        if freq < 0 or freq > 1:
            raise ValueError('filterband frequencies must be between 0 and 1, inclusive.')
    if filterband[1] <= filterband[0]:
        raise ValueError('the lower band must be lower than the upper band')
    if transband <= 0:
        raise ValueError('transband must be a non-negative scalar higher than 0')
    h = dict()
    if filtertype == 'pass' and filterband[0] == 0:  # TODO - I think I can enhance this
        h['type'] = 'lowpass'
        if filterband[1] < 0.5:
            h['wshift'] = 0
            h['subtract'] = False
            fpass = filterband[1]
        else:
            h['wshift'] = np.pi
            h['subtract'] = True
            fpass = 1 - filterband[1]
    elif filtertype == 'stop' and filterband[0] == 0:
        h['type'] = 'highpass'
        if filterband[1] < 0.5:
            h['wshift'] = 0
            h['subtract'] = True
            fpass = filterband[1]
        else:
            h['wshift'] = np.pi
            fpass = 1 - filterband[1]
    elif filtertype == 'pass' and filterband[0] != 0:  # TODO - I think I can enhance this
        h['type'] = 'bandpass'
        if filterband[1] < 0.5:
            h['wshift'] = 2 * np.pi / 2 * (filterband[0] + filterband[1]) / 2
            h['subtract'] = False
            fpass = (filterband[1] - filterband[0]) / 2
        else:
            h['wshift'] = np.pi
            h['subtract'] = True
            fpass = 1 - filterband[1]
    elif filtertype == 'stop' and filterband[0] != 0:
        h['type'] = 'bandstop'
        h['wshift'] = 2 * np.pi / 2 * (filterband[0]) / 2
        fpass = (filterband[1] - filterband[0]) / 2
    else:
        AttributeError('Filter tye must be either "pass" or "stop"')
    h['filterband'] = filterband
    h['transband'] = transband

    if (h['type'] == 'lowpass' and filterband[1] + transband > 1) or \
        (h['type'] == 'highpass' and filterband[1] - transband < 0) or \
        (h['type'] == 'bandpass' and (filterband[0] - transband < 0 or filterband[1] + transband > 1)) or \
        (h['type'] == 'bandstop' and 2 * np.pi > filterband[1] - filterband[0]):
        ValueError('The specified transition band is too large given the bandpass cutoff frequencies.')

    fstop = fpass + transband
    if (dev[0] + dev[1] <= 0):
        ValueError('the DEV vector must contain two positive elements')
    h['dev'] = dev
    return (h, fpass, fstop)

def getTransband(transband, filterband):
    if transband is None:
        if filterband[0] == 0:
            return min(filterband[0], 1 - filterband[1]) / 5 # TODO - is the same transband?
        else:
            return min(filterband[1] - filterband[0], min(filterband[0], 1 - filterband[1])) / 5;


def designFilter(filterband: tuple, filtertype: str, transband: float = None, dev: tuple = (.001, .01) ):
    """
Designs a multirate narrowband FIR filter with linear phase for use with
NARROWBANDFILTER.
    Parameters
    ----------
    filterband: A two-element vector defining a frequency band, normalized such that Nyquist = 1, formatted as:
[0 FC]  - lowpass or highpass
[F1 F2] - bandpass or bandstop
    filtertype: A string indicating the type of filter to implement:
'pass' for lowpass and bandpass, 'stop' for highpass and bandstop.
    transband: A scalar defining the transition bandwidth between passband and stopband,
in normalized frequency units.
The default is 1/5 the passband bandwidth.
    dev: A two-element vector defining passband and stopband ripple,
in linear units. The default is [.001 and .01], or approximately 0.01 and -40 dB.
    """
    fs  = 2
    transband = getTransband(transband, filterband)
    h, fpass, fstop = parseInputs(filterband, filtertype, transband, dev)
    delta0 = min(dev)

    idx = 1
    while idx == 1:
        if fs/2 <= 4 * fstop: # leads t unstable half-band filter designs
            do_hb = 0
        else:
            nhb = len(halfbandfir('minorder', fstop / (fs / 2), delta0)) - 1
            f = [fpass fstop]
            a = [1 0]
            nfb = remord(f, a, dev, fs)
            do_hb = nhb < nfb/4

