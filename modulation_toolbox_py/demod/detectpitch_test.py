import unittest

import numpy as np

from modulation_toolbox_py.demod.detectpitch import checkInputs, buffer2, whiten, nonIntGroupDelay, findPeaks, \
    GaussianLikelihood, percentile, bimodalGaussianMixture, detectVoicing, detectPitch, lsharm_freqtrack, factorinterp


class TestDetectpitch(unittest.TestCase):

    def test_detectpitch(self):
        [F0, F0m, voicing] = detectPitch(
            x=np.random.rand(200, 1),
            fs=800,
            voicingSens=.25,
            medFiltLen=5,
            freqCutoff=400,
            display=False
        )

    def test_factorinterp(self):
        np.testing.assert_array_equal(
            factorinterp([0, 0, 0, 0], 1, 1, 1),
            [0, 0, 0, 0]
        )

    def test_lsharm(self):
        f0 = lsharm_freqtrack(
            x=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, -1, -1, -1, -1],
            fs=4,
            freqs=[2],
            weights=np.ones((1, 1))
        )
        self.assertEqual(f0, 2)
        f0 = lsharm_freqtrack(
            x=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, -1, -1, -1, -1],
            fs=4,
            freqs=[1],
            weights=np.ones((1, 1))
        )
        self.assertEqual(f0, 1)

    # TODO: testing the commented value fails the test
    def test_detectvoicing(self):
        np.testing.assert_array_equal(detectVoicing([.1, .2, .1, .2], .5), [False, True, False, True])
        np.testing.assert_array_equal(detectVoicing([.1, .2, .1, 1], .5), [False, False, False, True])
        np.testing.assert_array_equal(detectVoicing([.1, .5, .1, 1], .4), [False, False, False, True])
        np.testing.assert_array_equal(detectVoicing([.1, .5, .1, 1], .7), [False, True, False, True])
        np.testing.assert_array_equal(detectVoicing([.1, .2, .3, .4], .9), [False, False, True, True])

    def test_bimodalGaussianMixture(self):
        t1, t2 = bimodalGaussianMixture([1, 0, 0, 0], 1, 1)
        self.assertAlmostEqual(t1['mu'], .25)
        self.assertAlmostEqual(t1['sigma'], 0.1875)
        self.assertAlmostEqual(t1['p'], 0.5)
        self.assertAlmostEqual(t2['mu'], 0.25)
        self.assertAlmostEqual(t2['sigma'], 0.1875)
        self.assertAlmostEqual(t2['p'], 0.5)

    def test_percentile(self):
        self.assertEqual(percentile([0, 1, 2, 3, 4, 5, 6], 0 / 6), 0)
        self.assertEqual(percentile([0, 1, 2, 3, 4, 5, 6], 1 / 6), 1)
        self.assertEqual(percentile([0, 1, 2, 3, 4, 5, 6], 2 / 6), 2)
        self.assertEqual(percentile([0, 1, 2, 3, 4, 5, 6], 3 / 6), 3)
        self.assertEqual(percentile([0, 1, 2, 3, 4, 5, 6], 4 / 6), 4)
        self.assertEqual(percentile([0, 1, 2, 3, 4, 5, 6], 5 / 6), 5)

    def test_gaussian_likelihood(self):
        self.assertAlmostEqual(GaussianLikelihood(1, 0, 1), 0.241970724519143)
        self.assertAlmostEqual(GaussianLikelihood(2, 1, 1), 0.241970724519143)
        np.testing.assert_array_almost_equal(GaussianLikelihood([1, 1, 0, 0], .5, 1),
                                             [.3520653267, .3520653267, .3520653267, .3520653267])
        np.testing.assert_array_almost_equal(GaussianLikelihood([1, 10, 100], .1, 10),
                                             [.1211, 0.0009, 0], decimal=4)
        np.testing.assert_array_almost_equal(GaussianLikelihood([.2, .3, .4, .5], .2, .2),
                                             [.8921, .8700, .8072, .7123], decimal=4)
        np.testing.assert_array_almost_equal(GaussianLikelihood([.001, .005, .007, .009], 2, .2),
                                             [0.8080, 0.8112, 0.8127, 0.8143], decimal=4)
        # np.testing.assert_array_almost_equal(GaussianLikelihood([1, 1, 1, 1, 0, 0, 0, 0], 1, 0.2857142857142857),
        #                                      [.3732, .3732, .3732, .3732, .0648, .0648, .0648, .0648])

    def test_find_peaks(self):
        peakPosOut, peakValOut = findPeaks([0, 10, 0, 11, 0], 1, 1, 0)
        np.testing.assert_equal(peakPosOut, 3)
        np.testing.assert_equal(peakValOut, 11)

        peakPosOut, peakValOut = findPeaks([-0.0549, 0.3423, 1.6969, 1.6968, 0.3423, -0.0549, 0.0314,
                                            ], 1, 1, 0)
        np.testing.assert_array_equal(peakPosOut, 2)
        np.testing.assert_array_equal(peakValOut, 1.6969)

        peakPosOut, peakValOut = findPeaks(
            np.array([[-0.0549, 0.3423, 1.6969, 0.3423, -0.0549, 0.0314, 1.6968, 0.0314, 0]]).T, 1, 1, 1)
        np.testing.assert_array_equal(peakPosOut, 2)
        np.testing.assert_array_equal(peakValOut, 1.6969)

        peakPosOut, peakValOut = findPeaks([0, 10, 0, 10, 0], 1, 1, 0)
        np.testing.assert_equal(peakPosOut, 3)
        np.testing.assert_equal(peakValOut, 10)

        peakPosOut, peakValOut = findPeaks([10, 20, 30, 40, 50, 60], 1, 1, 0)
        np.testing.assert_equal(peakPosOut, -1)
        np.testing.assert_equal(peakValOut, None)

    def test_whiten(self):
        np.testing.assert_array_almost_equal(
            whiten(np.array([[.1, .1, .1, .1]]), ARorder=1),
            np.array([[.025, .025, .025, .025]])
        )
        np.testing.assert_array_almost_equal(
            whiten(np.array([[1.0, 1.0, 1.0, 1.0]]), ARorder=4),
            np.array([[.4, .4, .4, .4]]))
        np.testing.assert_array_almost_equal(
            whiten(np.array([[0, 1.0, 0, -1]]), ARorder=4),
            np.array([[1 / 3, 1.0, 1 / 3, -1 / 3]]))
        np.testing.assert_array_equal(
            np.shape(whiten(np.ones((40, 1)), ARorder=12)),
            (40, 1))
        np.testing.assert_array_equal(
            np.shape(whiten(np.ones((40, 1)), ARorder=12)),
            (40, 1))
        np.testing.assert_array_equal(
            np.shape(whiten(np.ones((40,)), ARorder=12)),
            (40, 1))

    def test_buffer2(self):
        np.testing.assert_array_equal(
            buffer2([.1, .1, .1, .1, .5, -.5, 0, 0], 2, 1, 0, 1),
            np.array([[.1, .1, .1, .5, -.5, 0],
                      [.1, .1, .5, -.5, 0, 0]])
        )
        np.testing.assert_array_equal(
            buffer2([.1, .1, .1, .1, .2, .2, .5, .5], 1, 2, 0, 1),
            np.array([[.1, .1, .2, .5]]))
        np.testing.assert_equal(
            np.shape(buffer2(np.ones((400, 1)), winlen=40, hop=20, startindex=-20, numframes=20)),
            (40, 20))
        np.testing.assert_equal(
            np.shape(buffer2(np.ones((50, 1)), winlen=5, hop=10, startindex=-5, numframes=5)),
            (5, 6))
        np.testing.assert_equal(
            np.shape(buffer2(np.ones((200, 1)), winlen=40, hop=20, startindex=-20, numframes=10)),
            (40, 10))

    def test_nonIntGroupDelay(self):
        np.testing.assert_array_equal(
            nonIntGroupDelay(np.array([1, 1, 1, 1]).T, 0),
            [[1],
             [1],
             [1],
             [1]]
        )
        np.testing.assert_array_equal(
            nonIntGroupDelay([1, 1, 1, 1], 0),
            [[4, 0, 0, 0],
             [0, 0, 0, 0],
             [0, 0, 0, 0],
             [0, 0, 0, 0]]
        )
        np.testing.assert_array_equal(
            nonIntGroupDelay(np.array([[1, 1, 1, 1]]), 0),
            [[4, 0, 0, 0],
             [0, 0, 0, 0],
             [0, 0, 0, 0],
             [0, 0, 0, 0]]
        )
        np.testing.assert_array_almost_equal(
            nonIntGroupDelay(np.array([[1, 0, 0, 0]]), 2),
            [[0, 0, 0, 0],
             [0, 0, 0, 0],
             [1, 1, 1, 1],
             [0, 0, 0, 0]]
        )
        np.testing.assert_array_almost_equal(
            nonIntGroupDelay(np.array([[-1, 1, 1, -1]]), 2),
            [[0, 0, 0, 0],
             [0, 0, 0, 0],
             [0, -2, 0, -2],
             [0, 0, 0, 0]])
        np.testing.assert_array_almost_equal(
            nonIntGroupDelay(np.array([[0, 1, 2, 1, 0, 0, 0]]).T, .5),
            np.array([-0.0549, 0.3423, 1.6969, 1.6969, 0.3423, - 0.0549, 0.0314]).reshape(-1, 1), decimal=4,
        )

    def test_nonintgroupdelay_vertical(self):
        np.testing.assert_array_almost_equal(
            nonIntGroupDelay(np.array([[1, 0, 0], [10, 0, 0], [100, 0, 0]]), 0),
            np.array([[1, 0, 0], [10, 0, 0], [100, 0, 0]])
        )
        np.testing.assert_array_almost_equal(
            nonIntGroupDelay(np.array([[1, 0, 0], [10, 0, 0], [100, 0, 0]]), 1),
            np.array([[100, 0, 0], [1, 0, 0], [10, 0, 0]])
        )
        np.testing.assert_array_almost_equal(
            nonIntGroupDelay(np.array([[1, 2, 3], [10, 20, 30], [100, 200, 300]]), .5),
            np.array([[64, 128, 192], [-26, -52, -78], [73, 146, 219]])
        )

    def test_checkinputs(self):
        checkInputs([], fs=48000, medFiltLen=10, voicingSens=.5, freqCutoff=24000, display=False)
