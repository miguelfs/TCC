import unittest
from designfilterbank_stft import windowBandwidth, designfilterbankstft, downsample, upsample, STFTwindow, \
    generate_analysis_win
import numpy as np


class TestDesignFilterBankSTFT(unittest.TestCase):
    def test_designfilterbankstft(self):
        numHalfBands = 64
        sharpness = 9
        decFactor = 64 // 4
        fb1 = designfilterbankstft(numHalfBands, sharpness, decFactor)
        self.assertEqual(fb1['numbands'], 33)
        self.assertEqual(fb1['numhalfbands'], 64)
        self.assertEqual(fb1['dfactor'], 16)
        self.assertEqual(fb1['bandwidths'], 0.03125)
        self.assertEqual(fb1['afilterorders'], 575)
        self.assertEqual(fb1['sfilterorders'], 63)
        self.assertEqual(fb1['fshift'], 1)
        self.assertEqual(fb1['stft'], 1)
        self.assertEqual(fb1['keeptransients'], 1)
        # arrays
        self.assertEqual(fb1['afilters'][0].size, 576)
        self.assertEqual(fb1['sfilters'][0].size, 64)
        a = np.array(fb1['afilters'][0])

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
        my_values = list(filterbankparams['afilters'][0])
        self.assertEqual(filterbankparams['afilterorders'], 49)
        self.assertEqual(filterbankparams['sfilterorders'], 9)
        self.assertEqual(filterbankparams['fshift'], 1)
        self.assertEqual(filterbankparams['stft'], 1)
        self.assertEqual(filterbankparams['keeptransients'], 1)

    def test_window_bandwidth(self):
        result = windowBandwidth([1, 1, 1, 1, 1, 1, 1, 1], .5)
        self.assertEqual(result, .5)
        result = windowBandwidth([.1, .2, 1, 1, 1, 1, .21, .1], .4)
        self.assertEqual(result, .75)

    def test_win(self):
        compWin = STFTwindow(np.array([.1, .2, .25, .5, .5, .5, .5, .25, .2, .1]), 2, 2)
        np.testing.assert_array_almost_equal(compWin.T, [[0.816326530612245, 0.816326530612245]])
        compWin = STFTwindow(np.array([.1, .2, .25, .5, .5, 1, 1, .5, .5, .25, .2, .1]), 2, 4)
        np.testing.assert_array_almost_equal(compWin.T, [
            [-0.429653364404311, 1.150597145353918, 1.150597145353918, -0.429653364404311]])
        compWin = STFTwindow(np.array([.1, .2, .25, .5, .5, 1, 1, .5, .5, .25, .2, .1]), 1, 4)
        np.testing.assert_array_almost_equal(compWin.T, [[-.6, .8, .8, -.6]])
        compWin = STFTwindow(
            np.array(
                [.1, .1, .1, .1, .1, .1, .1, .1, .25, .25, .25, .25, .25, .25, .25, .25, .5, .5, .5, .5, .5, .5, .1, .1,
                 .1, .1, .1, .1, .1, .1]), 8, 6)
        np.testing.assert_array_almost_equal(compWin.T, [
            [1.459854014598540, 1.459854014598540, 0.632911392405063, 0.632911392405063, 0.729927007299270,
             0.729927007299270]])

    def test_generate_analysis_win(self):
        analysis_win = generate_analysis_win(8,2,8)
        np.testing.assert_array_almost_equal(analysis_win, [
            0.020725388601036,
            0.065594479571239,
            0.166414411818628,
            0.247265720009097,
            0.247265720009097,
            0.166414411818628,
            0.065594479571239,
            0.020725388601036,
        ])

    def test_downsampling(self):
        x = [1, 2, 3, 4, 5, 6, 7, 8]
        self.assertEqual(x, downsample(x, 1, 0))
        self.assertEqual([1, 3, 5, 7], downsample(x, 2, 0))
        self.assertEqual([2, 4, 6, 8], downsample(x, 2, 1))
        x = [.001, .002, .01, .02, .1, .2, 1, 2]
        value = downsample(x, 4, 0)
        np.testing.assert_array_equal([.001, .1], value)
        x = np.array([[.001, .002, .01, .02, .1, .2, 1, 2]])
        value = downsample(x, 4, 0)
        np.testing.assert_array_equal([[.001, .1]], value)
        x = np.array([[.001, .002, .01, .02, .1, .2, 1, 2]]).T
        value = downsample(x, 4, 0)
        np.testing.assert_array_equal([[.001], [.1]], value)

    def test_upsampling(self):
        x = np.array([[.1, .2, 1, 2]])
        value = upsample(x, 4, 0)
        np.testing.assert_array_equal(value, [.1, 0, 0, 0, .2, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0])
