import unittest
from designfilterbank_stft import windowBandwidth, designfilterbankstft, downsample
import numpy as np


class TestDesignFilterBankSTFT(unittest.TestCase):
    def test_designfilterbank(self):
        filtbankparams = designfilterbankstft(numhalfbands=9, sharpness=1, dfactor=1)
        self.assertEqual(filtbankparams['numbands'], 5)
        self.assertEqual(filtbankparams['numhalfbands'], 9)
        self.assertEqual(filtbankparams['dfactor'], 1)
        np.testing.assert_array_almost_equal(
            filtbankparams['centers'],
            [0, 0.2222, 0.4444, 0.6667, 0.8889], decimal=3)
        self.assertEqual(filtbankparams['bandwidths'], 4 / 9)
        np.testing.assert_array_almost_equal(filtbankparams['afilters'][0], [
            0.0182,
            0.0488,
            0.1227,
            0.1967,
            0.2273,
            0.1967,
            0.1227,
            0.0488,
            0.0182,
        ], decimal=3)
        self.assertEqual(filtbankparams['afilterorders'], 8)
        np.testing.assert_array_almost_equal(filtbankparams['sfilters'][0], np.reshape([
            0.1105,
            0.2966,
            0.7459,
            1.1951,
            1.3812,
            1.1951,
            0.7459,
            0.2966,
            0.1105,
        ], (-1, 1)), decimal=3)
        self.assertEqual(filtbankparams['sfilterorders'], 8)
        self.assertEqual(filtbankparams['fshift'], False)
        self.assertEqual(filtbankparams['stft'], True)
        self.assertEqual(filtbankparams['keeptransients'], True)

    def test_another_designfilterbank(self):
        filterbankparams = designfilterbankstft(10, 5, 3, True)
        self.assertEqual(filterbankparams['numbands'], 6)
        self.assertEqual(filterbankparams['numhalfbands'], 10)
        self.assertEqual(filterbankparams['dfactor'], 3)
        np.testing.assert_array_almost_equal(filterbankparams['centers'],
                                             [0, 0.2000, 0.4000, 0.6000, 0.8000, 1])
        self.assertEqual(filterbankparams['bandwidths'], 0.2000)
        np.testing.assert_array_almost_equal(filterbankparams['afilters'][0],
                                             [
                                                 0.0016,
                                                 0.0015,
                                                 0.0014,
                                                 0.0011,
                                                 0.0005,
                                                 - 0.0006,
                                                 - 0.0021,
                                                 - 0.0040,
                                                 - 0.0062,
                                                 - 0.0084,
                                                 - 0.0102,
                                                 - 0.0111,
                                                 - 0.0105,
                                                 - 0.0080,
                                                 - 0.0033,
                                                 0.0039,
                                                 0.0134,
                                                 0.0250,
                                                 0.0380,
                                                 0.0518,
                                                 0.0654,
                                                 0.0778,
                                                 0.0881,
                                                 0.0955,
                                                 0.0994,
                                                 0.0994,
                                                 0.0955,
                                                 0.0881,
                                                 0.0778,
                                                 0.0654,
                                                 0.0518,
                                                 0.0380,
                                                 0.0250,
                                                 0.0134,
                                                 0.0039,
                                                 - 0.0033,
                                                 - 0.0080,
                                                 - 0.0105,
                                                 - 0.0111,
                                                 - 0.0102,
                                                 - 0.0084,
                                                 - 0.0062,
                                                 - 0.0040,
                                                 - 0.0021,
                                                 - 0.0006,
                                                 0.0005,
                                                 0.0011,
                                                 0.0014,
                                                 0.0015,
                                                 0.0016,
                                             ], decimal=3)
        self.assertEqual(filterbankparams['afilterorders'], 49)
        np.testing.assert_array_almost_equal(filterbankparams['sfilters'][0], [
            -0.7481,
            - 0.7353,
            0.9549,
            5.7474,
            9.7941,
            9.7941,
            5.7474,
            0.9549,
            - 0.7353,
            - 0.7481,
        ], decimal=3)
        self.assertEqual(filterbankparams['sfilterorders'], 9)
        self.assertEqual(filterbankparams['fshift'], 1)
        self.assertEqual(filterbankparams['stft'], 1)
        self.assertEqual(filterbankparams['keeptransients'], 1)

    def test_window_bandwidth(self):
        result = windowBandwidth([1, 1, 1, 1, 1, 1, 1, 1], .5)
        self.assertEqual(result, .5)
        result = windowBandwidth([.1, .2, 1, 1, 1, 1, .21, .1], .4)
        self.assertEqual(result, .75)

    def test_downsampling(self):
        x = [1, 2, 3, 4, 5, 6, 7, 8]
        self.assertEqual(x, downsample(x, 1, 0))
        self.assertEqual([1, 3, 5, 7], downsample(x, 2, 0))
        self.assertEqual([2, 4, 6, 8], downsample(x, 2, 1))