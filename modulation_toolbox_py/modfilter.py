import numpy as np
from modulation_toolbox_py.moddecomp import moddecomp
from modulation_toolbox_py.filter.designfilter import designFilter
from modulation_toolbox_py.parsedecompinputs import parsedecompinputs
from modulation_toolbox_py.filter.narrowbandfilter import narrowbandFilter


def modfilter(x: np.array, fs: float, filterband: tuple, filtertype: str, decompparams, verbose: str, opinputs, demod = ('cog', 0.1, 0.05), subbands: any = 150):
    # decompparams, verbose, opinputs = parsedecompinputs(varargin)
    M, C, data = moddecomp( x, fs, decompparams[0], decompparams[1], 'maximal')
    chcekmodfilterparams(filterband, fs)

    h = designFilter(tuple(ti / data['modfs'] * 2 for ti in filterband), filtertype)
    Mfilt = narrowbandFilter(M, h, 1)


def chcekmodfilterparams(filterband, fs):
    if len(filterband) != 2 or filterband[0] > fs/2 or filterband[0] > fs/2 or filterband[0] < 0 or filterband[1] < 0:
        raise ValueError('the frequency band must contain two non-negative elements each less than or equal to fs/2')
    if filterband(1) <= filterband(0):
        raise ValueError('frequency band values must be strictly increasing')



def indexcellarray(c1, indx):
    c2 = []
    for i in range(indx):
        c2[i] = c1[indx(i)]
    return c2