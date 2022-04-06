import unittest
import numpy as np
from modulation_toolbox_py.filterbank.cutoffs2fbdesign import cutoffs2fbdesign


class TestCutoffs2fbdesign_test(unittest.TestCase):
    def test_properly(self):
        centers, bandwidths = cutoffs2fbdesign([.1, .2, .3, .4])
        np.testing.assert_array_equal(centers, [.15, .25, .35])
        np.testing.assert_array_equal(bandwidths, [.1, .1, .1])

        centers, bandwidths = cutoffs2fbdesign([.25, .75])
        np.testing.assert_array_equal(centers, [.5])
        np.testing.assert_array_equal(bandwidths, [.5])

        centers, bandwidths = cutoffs2fbdesign([0, .25, .75, 1])
        np.testing.assert_array_equal(centers, [0, .5, 1])
        np.testing.assert_array_equal(bandwidths, [.5, .5, .5])

        centers, bandwidths = cutoffs2fbdesign([0, .5, 1])
        np.testing.assert_array_equal(centers, [0, 1])
        np.testing.assert_array_equal(bandwidths, [1, 1])

    def test_errors(self):
        self.assertRaises(ValueError, cutoffs2fbdesign, [1, 2, 3])
        self.assertRaises(ValueError, cutoffs2fbdesign, [-.5, 0, .5])
        self.assertRaises(ValueError, cutoffs2fbdesign, [.1, .3, .2])
