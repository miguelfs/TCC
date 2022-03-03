import numpy as np
from scipy import signal


def remezOrder(fp, Dev):  # TODO : where this eq came from??
    A = abs(np.log10(np.abs(Dev)))
    N = np.fix(1 - 11.01217 * (0.5 - fp) + (-0.005309 * A ** 3 + 0.06848 * A ** 2 + 1.0702 * A - 0.4278) / (0.5 - fp))
    return N


def adjustOrder(N):
    while (N + 2) != 4 * np.fix((N + 2) / 4):
        N = N + 1
    return N


def estimateOrder(fp, Dev) -> int:
    N = remezOrder(fp, Dev)
    N = adjustOrder(N)
    b = halfbandfir(N=N, fp=fp)
    h1 = min(abs(sum(b)), abs(1 - abs(sum(b))))
    if h1 > Dev:
        N += 4
    return int(N)


def validateParseInputValues(N: int, fp: float, Dev: float):
    if Dev is not None:
        if Dev <= 0 or Dev >= 0.5:
            raise ValueError('Dev must be 0 < Dev(linear) < 0.5')
    if fp <= 0 or fp >= 0.5:
        raise ValueError('passband edge frequency must satisfy 0 < Fp < 0.5')
    if N is not None:
        if (N + 2) != 4 * np.fix((N + 2) / 4):
            raise ValueError('N order must be element of {2,6,10,14,18,...,n,n+4,...}')


def remezDesign(N: int, fp, Dev, minOrder):
    # N_new = 23
    # b = signal.remez(numtaps=N+1, bands=bands, desired=desired, fs=2)
    N = int(N+1)
    b = signal.remez(N, bands=[0, fp, 1.0 - fp, 1], desired=[1, 0], fs=2, grid_density=61)
    b[1::2] = [0 for _ in b[1::2]]
    b[N // 2] = 1 / 2

    if minOrder:
        h1 = min(abs(sum(b)), abs(1 - abs(sum(b))))
        # if h1 > Dev:
        #     raise ValueError(f'Dev_designed={h1} > Dev_specified={Dev}, increase order or Dev')
    return b


# ex: halfbandfir(22, 0.1))
# ex: halfbandfir('minorder',.4,0.01)
def validateParseInputArguments(N, fp, Dev, minOrder):
    if N is not None and fp is not None and Dev is None and minOrder is False:
        return
    if N is None and fp is not None and Dev is not None and minOrder is True:
        return
    raise ValueError('illegal argument combination')
    # if minOrder is True and Dev is not None and fp is not None:
    #     return
    # if N is not None and minOrder is True:
    #     raise ValueError('illegal argument combination 3')
    # if N is not None and fp is not None and Dev is not None and minOrder is True:
    #     raise ValueError('illegal argument combination 1')
    # if minOrder is True and fp is not None and Dev is not None and N is not None:
    #     raise ValueError('illegal argument combination 2')
    # if (Dev is not None) and minOrder is False:
    #     raise ValueError('Peak ripple, Dev, can be specified for minimum order design, only.')


def halfbandfir(fp: float, Dev: float = None, N: int = None, minOrder: bool = False) -> np.array:
    validateParseInputArguments(N=N, fp=fp, Dev=Dev, minOrder=minOrder)
    validateParseInputValues(N=N, fp=fp, Dev=Dev)
    N = estimateOrder(fp, Dev) if N is None else N
    return remezDesign(N, fp, Dev, minOrder)
