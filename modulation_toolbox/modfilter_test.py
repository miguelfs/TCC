import unittest
from modulation_toolbox.modfilter import modfilter
import numpy as np


class TestModfilter(unittest.TestCase):

    def test_result(self):
        x = np.linspace(0, np.pi, 1024)
        yHilb = modfilter(x, 4800, [0, 2], 'pass', 'hilb')
        self.assertEqual(sum(yHilb), 870.3233)

