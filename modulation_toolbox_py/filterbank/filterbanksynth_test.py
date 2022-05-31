import unittest
import numpy as np
import pandas as pd
from modulation_toolbox_py.filterbank.designfilterbank_stft import designfilterbankstft
from modulation_toolbox_py.filterbank.filterbanksynth import windowphaseterm, upsample, ISTFT, matchLen, \
    filterbanksynth, bandpassExpansion
from modulation_toolbox_py.filterbank.designfilterbank import designfilterbank
from modulation_toolbox_py.filterbank.filtersubbands import filtersubbands


class TestFilterBankSynth(unittest.TestCase):

    def test_impulse_response(self):
        nfft = 576
        numHalfBands = 64
        sharpness = 9
        decFactor = 64 // 4
        filterbankparams = designfilterbankstft(numHalfBands, sharpness, decFactor)
        leftPad = int(np.floor((nfft - 1) / 2))
        rightPad = int(np.ceil((nfft - 1) / 2))
        input_pulse = np.vstack((np.zeros((leftPad, 1)), np.array([[1]]), np.zeros((rightPad, 1))))
        impSubbands = filtersubbands(input_pulse, filterbankparams)
        result = filterbanksynth(impSubbands, filterbankparams)
        np.testing.assert_almost_equal(np.abs(np.max(result)), 1, decimal=7)

    def test_bandpassExpansion(self):
        ones = np.ones((7, 14))
        val = bandpassExpansion(ones, 0.005, 1, 1, 0)
        np.testing.assert_array_equal(ones, val)

    def testUpsample(self):
        x = [0, 0, 0, 1, 0, 0, 2, 0, 0, 3, 0, 0]
        self.assertEqual(x, upsample([0, 1, 2, 3], 3))

    def testMatchLen(self):
        # np.testing.assert_array_equal([10, 10, 0, 0, 0], matchLen([10, 10], 5))
        # np.testing.assert_array_equal(np.array([10, 10, 0, 0, 0]), matchLen([10, 10], 5))
        np.testing.assert_array_equal(np.array([[10, 10, 0, 0, 0]]), matchLen([10, 10], 5))

    def testPhaseWindow(self):
        np.testing.assert_array_almost_equal(
            windowphaseterm(4, 4).reshape(-1, 1),
            [[1 + 0j], [1 + 0j], [1 + 0j], [1 + 0j]],
            decimal=15)
        self.assertEqual(windowphaseterm(8, 1).reshape(-1, 1), 1)
        np.testing.assert_array_almost_equal(windowphaseterm(4, 2).reshape(-1, 1), [[1], [1]])
        np.testing.assert_array_almost_equal(
            windowphaseterm(4, 6).reshape(-1, 1),
            [[1 + 0j], [-0.5 - 0.866025403784439j], [-.5 + 0.866025403784439j], [1 + 0j], [-.5 - 0.866025403784439j],
             [-.5 + 0.866025403784439j]],
            decimal=15)
        np.testing.assert_array_almost_equal(
            windowphaseterm(3, 5).reshape(-1, 1),
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

    def test_istft_more(self):
        S = np.array([
            [.01 + .01j, 0, 0, .03 + .03j, 0, 0, 0, 0],
            [1 + 1j, 0, 0, 3 + 3j, 0, 0, 0, 0],
            [10 + 10j, 0, 0, 30 + 30j, 0, 0, 0, 0],
            [-100 - 100j, 0, 0, -300 - 300j, 0, 0, 0, 0],
        ], dtype=np.complex_)
        win = np.array([[.1, .2, .2, .1, ]]).T
        hop = 1
        freqdownsample = 2
        y, delay = ISTFT(S, win, hop, 1, freqdownsample)
        np.testing.assert_array_almost_equal(
            y,
            [[
                -2.774750000000000 + 2.275250000000000j,
                5.450500000000001 + 5.450500000000001j,
                4.550500000000000 - 5.549500000000000j,
                - 8.899000000000001 - 8.899000000000001j,
                - 16.648500000000002 + 13.651499999999999j,
                16.351499999999998 + 16.351499999999998j,
                6.825749999999999 - 8.324250000000001j,
                0.000000000000000 - 0.000000000000000j,
                0.000000000000000 - 0.000000000000000j,
                0.000000000000000 - 0.000000000000000j,
                0.000000000000000 - 0.000000000000000j,
            ]]
        )

    def test_filterbanksynth_runs(self):
        fb = designfilterbank([.1, .2, .3, .4, .5, .6], [.01, .01, .01, .02, .02, .02])

    def test_filterbanksynth(self):
        Scog = openS('fb1_mock/Scog.csv')
        ycog_real = np.genfromtxt('fb1_mock/y_cog_real.csv', delimiter=',').reshape(1, -1)
        ycog_imag = np.genfromtxt('fb1_mock/y_cog_imag.csv', delimiter=',').reshape(1, -1)

        fb1 = get_fb1()
        a = np.reshape(np.linspace(-1, 1, fb1['numbands'] ** 2), (fb1['numbands'], fb1['numbands'])).T
        b = np.hstack((a, a, a, a, a, a, a, a))
        y_out = filterbanksynth(b, fb1)
        print(y_out)
        # np.equal(np.shape(y_out_cog), np.shape(ycog_real))
        # np.testing.assert_array_almost_equal(ycog_real, np.real(y_out_cog), decimal=1)
        # np.testing.assert_array_almost_equal(ycog_imag, np.imag(y_out_cog), decimal=1)


def openS(path):
    S = pd.read_csv(path, sep=",", header=None)
    return S.applymap(lambda s: complex(s.replace('i', 'j'))).values


def get_params():
    return designfilterbank([.1, .2, .3, .4, .5, .6], [.01, .01, .01, .02, .02, .02])


# def design(numhalfbands=64, sharpness=9, decFactor=64 / 4):
#     return designfilterbank(numHalfbands, sharpness, decFactor);


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
