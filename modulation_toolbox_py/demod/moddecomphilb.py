import numpy as np

from modulation_toolbox_py.demod.carrier2if import carrier2if


def moddecomphilb(S) -> (np.ndarray, np.ndarray, np.ndarray):
    """uses hilbert envelope do incoherently demodulate into non-negative modulators and complex carriers"""
    S = S.reshape(1, -1)
    M = np.abs(S)
    C = np.exp(1j * np.angle(S))
    F = carrier2if(C)
    return M, C, F
