import unittest

import numpy as np
from modulation_toolbox_py.demod.moddecompcog import parseinputs, spectralCOG, factorinterp, moddecompcog


class TestModdecompcog(unittest.TestCase):
    def test_moddecompcog(self):
        M, C, F, Fmeasured = moddecompcog(np.ones((4, 4)), [1, 1], 1, [1], [1])
        np.testing.assert_almost_equal(
            M,
            np.array([[-1, 1, -1, 1],
                      [-1, 1, -1, 1],
                      [-1, 1, -1, 1],
                      [-1, 1, -1, 1]])
        )
        np.testing.assert_almost_equal(
            C,
            np.array([[-1, 1, -1, 1],
                      [-1, 1, -1, 1],
                      [-1, 1, -1, 1],
                      [-1, 1, -1, 1]])
        )
        np.testing.assert_almost_equal(
            F,
            np.array([[1, 1, 1, 1],
                      [1, 1, 1, 1],
                      [1, 1, 1, 1],
                      [1, 1, 1, 1]])
        )
        np.testing.assert_almost_equal(
            Fmeasured,
            np.array([[1],
                      [1],
                      [1],
                      [1]])
        )

    def test_moddecompcog_2(self):
        M, C, F, Fmeasured = moddecompcog(np.identity(4), [2, 4], 1, [1], [1])
        np.testing.assert_almost_equal(
            M,
            np.array([[-1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, -1, 0],
                      [0, 0, 0, 1]])
        )
        np.testing.assert_almost_equal(
            C,
            np.array([[-1, 1, -1, 1],
                      [-1, 1, -1, 1],
                      [-1, 1, -1, 1],
                      [-1, 1, -1, 1]])
        )
        np.testing.assert_almost_equal(
            F,
            np.array([[1, 1, 1, 1],
                      [1, 1, 1, 1],
                      [1, 1, 1, 1],
                      [1, 1, 1, 1]])
        )
        np.testing.assert_almost_equal(
            Fmeasured,
            np.array([[1],
                      [1],
                      [1],
                      [1]])
        )

    def test_moddecompcog_3(self):
        M, C, F, Fmeasured = moddecompcog(np.array([[0, .1, .2, .3], [1j, 0, 0, 0], [0, 0, 1, 0],
                                                    [0, 1, -1j, .3]]),
                                          [2, 4], 1, [1], [1])
        np.testing.assert_almost_equal(
            M,
            np.array([[0, 1, -.2, .3],
                      [-1j, 0, 0, 0],
                      [0, 0, -1, 0],
                      [0, -0.4606 - 0.8876j, 0.0674 - 0.9977j, -0.1727 + 0.2453j]]), decimal=4
        )
        np.testing.assert_almost_equal(
            F,
            np.array([[1, 1, 1, 1],
                      [1, 1, 1, 1],
                      [1, 1, 1, 1],
                      [1.3262, 1.3262, 1.3262, 1.3262]]), decimal=4,
        )
        np.testing.assert_almost_equal(
            Fmeasured,
            np.array([[1],
                      [1],
                      [1],
                      [1.3262]]), decimal=4
        )

    def test_parseinputs(self):
        self.assertEqual(
            parseinputs(np.array([0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 0, 1, 1, 1, 1]), [1, 1], 1, [.2, 1], [.2, 1]),
            np.array([[.1], [.5]])
        )

    def test_factorinterp(self):
        np.testing.assert_almost_equal(
            factorinterp([1, 1, 0, 0], 2, 1, .5),
            np.array([1, 1, 1, .5, 0, 0, 0, 0]),
            decimal=2
        )

    def test_spectralcog(self):
        np.testing.assert_almost_equal(spectralCOG([[.2, .4, .2, .1], [.1, .1, 0, 0]], .5, 1.5), 1),
        np.testing.assert_almost_equal(spectralCOG([[2.89, 0.9522, 0., 0.9522]], .5, 1.5), 1),
        np.testing.assert_almost_equal(spectralCOG([2.89, 0.952, 10, 5.95], .5, 1.5), 1.147, decimal=3),
        self.assertAlmostEqual(spectralCOG([0, 0, 0, 0], .1, .2), .15)
        self.assertAlmostEqual(spectralCOG([0, 1, 0, -1], .2, .8), .5)
        self.assertAlmostEqual(spectralCOG([0, 1, 0, -1], 0, 2), 1)
        np.testing.assert_almost_equal(
            spectralCOG(np.array([0, .3, - 1j, .6, -1, .2j, 4, .2], dtype=np.complex_), 0, 3), .5421 - .4208j,
            decimal=4)
