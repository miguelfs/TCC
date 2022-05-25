import unittest

from modulation_toolbox_py.filterbank.designfilterbank_stft import designfilterbankstft
from modulation_toolbox_py.filterbank.filterbankfreqz import filterbankfreqz


class TestFilterbankfreqz(unittest.TestCase):
    def test_filterbankfreqz(self):
        numHalfBands = 64
        sharpness = 9
        decFactor = 64 // 4
        fs = 16000
        fb1 = designfilterbankstft(numHalfBands, sharpness, decFactor)
        filterbankfreqz(fb1, nfft=None, fs=16000)