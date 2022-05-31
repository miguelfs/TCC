import unittest
import numpy as np
from modulation_toolbox_py.filterbank.cutoffs2fbdesign import cutoffs2fbdesign


class TestCutoffs2fbDesignTest(unittest.TestCase):
    def test_properly(self):
        centers, bandwidths = cutoffs2fbdesign([.1, .2, .3, .4])
        np.testing.assert_array_almost_equal(centers, [.15, .25, .35])
        np.testing.assert_array_almost_equal(bandwidths, [.1, .1, .1])

        centers, bandwidths = cutoffs2fbdesign([.25, .75])
        np.testing.assert_array_equal(centers, [.5])
        np.testing.assert_array_equal(bandwidths, [.5])

        centers, bandwidths = cutoffs2fbdesign([0, .25, .75, 1])
        np.testing.assert_array_equal(centers, [0, .5, 1])
        np.testing.assert_array_equal(bandwidths, [.5, .5, .5])

        centers, bandwidths = cutoffs2fbdesign([0, .5, 1])
        np.testing.assert_array_equal(centers, [0, 1])
        np.testing.assert_array_equal(bandwidths, [1, 1])

    def test_like_tutorial3(self):
        fs = 16000
        cutoffs = (2 / fs) * np.array([50, 100, 200, 400, 800, 1600, 3200, 6400])
        centers, bandwidths = cutoffs2fbdesign(cutoffs)
        np.testing.assert_array_almost_equal(centers, [0.009375, 0.01875, 0.0375, 0.0750, 0.1500, 0.3000, 0.6000, ],
                                             decimal=16)
        np.testing.assert_array_almost_equal(bandwidths, [0.00625, 0.0125, 0.0250, 0.0500, 0.1000, 0.2000, 0.4000, ],
                                             decimal=16)
        print(centers)
        print(bandwidths)

    def test_errors(self):
        self.assertRaises(ValueError, cutoffs2fbdesign, [1, 2, 3])
        self.assertRaises(ValueError, cutoffs2fbdesign, [-.5, 0, .5])
        self.assertRaises(ValueError, cutoffs2fbdesign, [.1, .3, .2])
