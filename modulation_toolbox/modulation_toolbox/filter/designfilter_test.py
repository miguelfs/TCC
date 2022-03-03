from designfilter import designFilter
import unittest
import numpy as np


class TestDesignFilter(unittest.TestCase):

    def test_low_pass(self):
        result = designFilter(filterband=(0, .25), filtertype='pass', transband=.2, dev=(.05, .05))
        self.assertEqual(result['type'], 'lowpass')
        self.assertEqual(result['wshift'], 0)
        self.assertEqual(result['filterband'], (0, .25))
        self.assertEqual(result['subtract'], 0)
        self.assertEqual(result['transband'], .2)
        self.assertEqual(result['dev'], (.05, .05))
        np.testing.assert_array_almost_equal(result['filters'][0],
                                             [-0.0216, -0.0755, -0.0076, 0.1190, 0.2792, 0.3488, 0.2792, 0.1190,
                                              -0.0076, -0.0755, -0.0216], decimal=4)
        self.assertEqual(result['delay'], 5)

