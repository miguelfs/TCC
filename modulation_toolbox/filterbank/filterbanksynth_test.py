import unittest
import numpy as np
import pandas as pd
from modulation_toolbox.filterbank.filterbanksynth import windowphaseterm, upsample, ISTFT, matchLen, filterbanksynth
from modulation_toolbox.filterbank.designfilterbank import designfilterbank


class TestFilterBankSynth(unittest.TestCase):

    def testUpsample(self):
        x = [0, 0, 0, 1, 0, 0, 2, 0, 0, 3, 0, 0]
        self.assertEqual(x, upsample([0, 1, 2, 3], 3))

    def testMatchLen(self):
        # np.testing.assert_array_equal([10, 10, 0, 0, 0], matchLen([10, 10], 5))
        # np.testing.assert_array_equal(np.array([10, 10, 0, 0, 0]), matchLen([10, 10], 5))
        np.testing.assert_array_equal(np.array([[10, 10, 0, 0, 0]]), matchLen([10, 10], 5))

    def testPhaseWindow(self):
        np.testing.assert_array_almost_equal(
            windowphaseterm(4, 4),
            [[1 + 0j], [1 + 0j], [1 + 0j], [1 + 0j]],
            decimal=15)
        self.assertEqual(windowphaseterm(8, 1), 1)
        np.testing.assert_array_almost_equal(windowphaseterm(4, 2), [[1], [1]])
        np.testing.assert_array_almost_equal(
            windowphaseterm(4, 6),
            [[1 + 0j], [-0.5 - 0.866025403784439j], [-.5 + 0.866025403784439j], [1 + 0j], [-.5 - 0.866025403784439j],
             [-.5 + 0.866025403784439j]],
            decimal=15)
        np.testing.assert_array_almost_equal(
            windowphaseterm(3, 5),
            [[1 + 0j],
             [-0.809016994374948 - 0.587785252292473j],
             [0.309016994374948 + 0.951056516295154j],
             [0.309016994374948 - 0.951056516295154j],
             [-0.809016994374948 + 0.587785252292473j]],
            decimal=10)

    def test_istft(self):
        y, delay = ISTFT([0, 1, 0, -1, 0, 1, 0, -1], win=[1, 1, 1, 1], hop=2, fshift=False, freqdownsample=1)
        array_a = [0.00j, 0.00j, 0.00j, 0.00j, 0.500j, 0.500j, 0.500j, 0.500j, 0.00j, 0.00j, 0.00j, 0.00j,
                   0.00 - 0.500j,
                   0.00 - 0.500j, - 0.500j, - 0.500j, 0.00j, 0.00j]
        np.testing.assert_array_almost_equal(
            y.reshape(18), array_a, decimal=10
        )
        self.assertEqual(delay, 3)

        y, delay = ISTFT([0, 1, 0, -1, 0, 1, 0, -1], win=[1, 1, 1, 1], hop=2, fshift=True, freqdownsample=8)
        np.testing.assert_array_almost_equal(
            y.reshape(18), array_a, decimal=10
        )
        self.assertEqual(delay, 5)

        y, delay = ISTFT([1, 1j, -1, -1j, 1, 1j, -1, -1j], win=[0.5, 1, 1, 0.5], hop=2, fshift=False, freqdownsample=2)
        array_b = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 1, 1, 0.5, 0, 0]
        np.testing.assert_array_almost_equal(y.reshape(18), array_b, decimal=10)
        self.assertEqual(delay, 2)

        y, delay = ISTFT([1, 1j, -1, -1j, 1, 1j, -1, -1j], win=[0.5, 1, 1, 0.5], hop=2, fshift=False, freqdownsample=8)
        np.testing.assert_array_almost_equal(y.reshape(18), array_b, decimal=10)
        self.assertEqual(delay, 5)

    def test_filterbanksynth_runs(self):
        fb = designfilterbank([.1, .2, .3, .4, .5, .6], [.01, .01, .01, .02, .02, .02])

    def test_filterbanksynth(self):
        Scog = openS('fb1_mock/Scog.csv')
        Shilb = openS('fb1_mock/Shilb.csv')
        yhilb = np.genfromtxt('fb1_mock/yhilb.csv', delimiter=',')
        ycog = np.genfromtxt('fb1_mock/ycog.csv', delimiter=',')

        fb1 = get_fb1()

        y_out_cog = filterbanksynth(Scog, fb1)
        np.testing.assert_array_almost_equal(ycog, y_out_cog)


def openS(path):
    S = pd.read_csv(path, sep=",", header=None)
    return S.applymap(lambda s: complex(s.replace('i', 'j'))).values


def get_params():
    return designfilterbank([.1, .2, .3, .4, .5, .6], [.01, .01, .01, .02, .02, .02])


def get_fb1():
    fb1 = dict()
    fb1['numbands'] = 33
    fb1['numhalfbands'] = 64
    fb1['dfactor'] = [16]
    fb1['centers'] = np.genfromtxt('fb1_mock/centers.csv', delimiter=',')
    fb1['bandwidths'] = 0.03125
    fb1['afilters'] = [np.genfromtxt('fb1_mock/afilters.csv', delimiter=',')]
    fb1['afilterorders'] = 575
    fb1['sfilters'] = [np.genfromtxt('fb1_mock/sfilters.csv', delimiter=',')]
    fb1['sfilterorders'] = 63
    fb1['fshift'] = 1
    fb1['stft'] = 1
    fb1['keeptransients'] = 1
    return fb1
