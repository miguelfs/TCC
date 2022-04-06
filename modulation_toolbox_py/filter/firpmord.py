import numpy as np


def db_to_dev(rp, rs) -> tuple:  # rb = passband ripple in db, rs = stopband ripple in db
    dev = ((10 ** (rp / 20) - 1) / (10 ** (rp / 20) + 1), 10 ** (-rs / 20))
    return dev


def firpmord(fcuts, mags, devs, fsamp):
    fcuts = np.array([f / fsamp for f in fcuts])
    if np.any(fcuts > 1 / 2):
        raise ValueError("Invalid range.")
    if np.any(fcuts < 0):
        raise ValueError("The frequency must be positive.")
    mags = np.array(mags)
    devs = np.array(devs)

    # turn vectors into column vectors
    fcuts = fcuts.reshape(-1, 1)
    mags = mags.reshape(-1, 1)
    devs = devs.reshape(-1, 1)

    mf = fcuts.shape[0]
    mm = mags.shape[0]
    nbands = mm

    if len(mags) != len(devs):
        raise ValueError("Mismatched vector length.")

    if mf != 2 * (nbands - 1):
        raise ValueError("Invalid length.")

    zz = (mags == 0).astype(float)
    devs = devs / (zz + mags)

    f1 = fcuts[:mf - 1:2]
    f2 = fcuts[1:mf:2]

    n = np.argmin(f2 - f1)
    if nbands == 2:
        L = remlpord(f1[n], f2[n], devs[0], devs[1])
    else:
        L = 0
        for i in range(1, nbands - 1):
            L1 = remlpord(f1[i - 1], f2[i - 1], devs[i], devs[i - 1])
            L2 = remlpord(f1[i], f2[i], devs[i], devs[i + 1])
            L = max(L, max(L1, L2))

    N = int(np.ceil(L) - 1)

    # ff = np.array([[0], [2 * fcuts], [1]])
    ff = np.concatenate((np.array([[0]]), 2 * fcuts, np.array([[1]])), axis=0)
    # aa = np.repeat(mags, 2, axis=0)
    aa = mags
    wts = np.ones_like(devs) * np.max(devs) / devs
    if aa[-1] != 0 and N % 2 != 0:
        N += 1

    return N, (ff * fsamp / 2).reshape(-1), aa.reshape(-1), wts.reshape(-1)
    # return N, ff.reshape(-1), aa.reshape(-1), wts.reshape(-1)


def remlpord(freq1, freq2, delta1, delta2):
    AA = np.array([[-4.278e-01, -4.761e-01, 0], [-5.941e-01, 7.114e-02, 0], [-2.660e-03, 5.309e-03, 0]])
    d1 = np.log10(delta1.item())
    d2 = np.log10(delta2.item())
    D = np.array([[1, d1, d1 * d1]]) @ AA @ np.array([[1], [d2], [d2 * d2]])
    bb = np.array([[11.01217], [0.51244]])
    fK = np.array([[1.0, d1 - d2]]) @ bb
    df = abs(freq2 - freq1)
    L = D / df - fK * df + 1
    return L
