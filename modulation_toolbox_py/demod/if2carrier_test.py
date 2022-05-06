import unittest
import numpy as np
from modulation_toolbox_py.demod.if2carrier import if2carrier


class TestCarrier2if(unittest.TestCase):
    def test_carrier2if(self):
        np.testing.assert_almost_equal(if2carrier([1, 1, 1, 1]), [-1, 1, -1, 1])
        np.testing.assert_almost_equal(if2carrier([[1, 1, 1, 1],
                                                   [0, 0, 0, 0],
                                                   [1, 1, 1, 1],
                                                   [0, 0, 0, 0], ]),
                                       [
                                           [-1, 1, -1, 1],
                                           [1, 1, 1, 1],
                                           [-1, 1, -1, 1],
                                           [1, 1, 1, 1],
                                       ])
        np.testing.assert_almost_equal(if2carrier([1, 0, 0, 0]), [-1, -1, -1, -1])
        np.allclose(if2carrier([1, 0, 1j, 0]), [-1, -1, -.0432, -.0432])
        np.allclose(if2carrier([1, 1, 1j, 0]), [-1, -1, -.0432, -.0432])
