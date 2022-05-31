import unittest

from modulation_toolbox_py.filterbank.designfilterbank import designfilterbank
from modulation_toolbox_py.filterbank.designfilterbank_stft import designfilterbankstft
from modulation_toolbox_py.filterbank.filterbankfreqz import filterbankfreqz


class TestFilterbankfreqz(unittest.TestCase):
    def test_filterbankfreqz(self):
        numHalfBands = 64
        sharpness = 9
        decFactor = 64 // 4
        fb = designfilterbankstft(numHalfBands, sharpness, decFactor)
        filterbankfreqz(fb, nfft=None, fs=16000)

    def test_filterbankfreqz_again(self):
        centers = [0.009375, 0.01875, 0.0375, 0.0750, 0.1500, 0.3000, 0.6000, ]
        bandwidths = [0.00625, 0.0125, 0.0250, 0.0500, 0.1000, 0.2000, 0.4000]
        fb = designfilterbank(centers, bandwidths)
        filterbankfreqz(fb, nfft=None, fs=16000)
