import unittest

import numpy as np

from modulation_toolbox_py.demod.moddecompcog import factor, factorinterp
from modulation_toolbox_py.demod.moddecomphilb import moddecomphilb


class TestModdecomphilb(unittest.TestCase):
    def test_moddecomphilb(self):
        np.testing.assert_array_equal(moddecomphilb([0, 0, 0, 0]), [0, 0, 0, 0])
        np.testing.assert_array_equal(moddecomphilb([0, 1j, 0, -.1, 0, .2, 0, -.2j, ]),
                                      [0, 1.0000, 0, 0.1000, 0, 0.2000, 0, 0.2000])

    def test_factorinterp(self):
        # np.testing.assert_array_equal(
        #     factorinterp([0, 1, 2, 3], 1, 1, 1), [0, 1, 2, 3])
        np.testing.assert_array_equal(
            factorinterp([0, 1, 2, 3], 4, 1, 1), [0, .6366, 1, 1.9099, 2, 3.101, 3, 4.4563])

    def test_factor(self):
        self.assertEqual(factor(10), [2, 5])
        self.assertEqual(factor(20), [2, 2, 5])
