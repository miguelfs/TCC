import numpy as np


def cutoffs2fbdesign(cutoffs: list) -> (list, list):
    # converts frequency cutoff values to subband center frequencies and bandwidths
    includeBaseband = cutoffs[0] == 0
    includeNyquist = cutoffs[-1] == 1
    if includeBaseband:
        cutoffs = cutoffs[1:]
    if includeNyquist:
        cutoffs = cutoffs[0:-1]
    cutoffs = np.array(cutoffs)
    if not np.all((cutoffs[1:] - cutoffs[:-1]) > 0):
        raise ValueError('sub-band cutoff frequencies must be strictly increasing')
    elif min(cutoffs) <= 0 or max(cutoffs) >= 1:
        raise ValueError('sub-band cutoff frequencies must be in the range [0 1] non-inclusive')
    bandwidths = flat([[2 * cutoffs[0]], cutoffs[1:] - cutoffs[:-1], [2 * (1 - cutoffs[-1])]])
    centers = flat([[0], (cutoffs[:-1] + cutoffs[1:]) / 2, [1]])
    if not includeBaseband:
        bandwidths = bandwidths[1:]
        centers = centers[1:]
    if not includeNyquist:
        bandwidths = bandwidths[:-1]
        centers = centers[:-1]
    return centers, bandwidths


def flat(l: list):
    return [item for sublist in l for item in sublist]
