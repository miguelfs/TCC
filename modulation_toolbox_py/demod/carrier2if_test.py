import unittest
import numpy as np
from modulation_toolbox_py.demod.carrier2if import carrier2if


class TestCarrier2if(unittest.TestCase):
    def test_carrier2if(self):
        np.testing.assert_array_equal(carrier2if([1, 1, 1, 1]), [0, 0, 0, 0])
        np.testing.assert_array_equal(carrier2if([1, 1j, 0, -1j]), [0, .5, -.5, -.5])
        np.testing.assert_array_equal(carrier2if([0, .5 + .5j, 1 - .2j, 1 - 1j]),
                                      [0, 0.2500, -0.3128, -0.1872])
