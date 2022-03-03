import unittest
from modulation_toolbox.modulation_toolbox.filter.firpmord import firpmord, db_to_dev
import numpy as np
import scipy.signal as signal


class TestFirpmord(unittest.TestCase):

    def test_individual_results(self):
        dev = db_to_dev(3, 40)
        n, fo, a0, w = firpmord([500, 600], [1, 0], dev, 2000)
        self.assertEqual(n, 22)
        # np.testing.assert_array_equal(fo, [0, 0.5, 0.6, 1])
        np.testing.assert_array_equal(fo, [0, 500, 600, 1000])
        np.testing.assert_array_equal(a0, [1, 0])
        np.testing.assert_array_equal(w, [1, dev[0] * 100])

    def test_array_results(self):
        dev = db_to_dev(3, 40)
        result = firpmord([500, 600], [1, 0], dev, 2000)
        self.assertEqual(result[0], 22)
        np.testing.assert_array_equal(result[1], [0, 500, 600, 1000])
        np.testing.assert_array_equal(result[2], [1, 0])
        np.testing.assert_array_equal(result[3], [1, 100 * dev[0]])

    def test_remez(self):
        dev = db_to_dev(3, 40)
        result = firpmord([500, 600], [1, 0], dev, 2000)
        filter_coeficients = signal.remez(
            numtaps=result[0] + 1,
            bands=result[1],
            desired=result[2],
            weight=result[3],
            fs=2000,
        )

        proof = [-0.0127136125752405, 0.0104266485260949, 0.0598406969000730, 0.0516256781047958, -0.0232653343352842,
                 -0.0306216140996613, 0.0511429522028190, 0.0340312653600146, -0.0976494774316472, -0.0353725834937516,
                 0.315392807998724, 0.535938080310936, 0.315392807998724, -0.0353725834937516, -0.0976494774316472,
                 0.0340312653600146, 0.0511429522028190, -0.0306216140996613, -0.0232653343352842, 0.0516256781047958,
                 0.0598406969000730, 0.0104266485260949, -0.0127136125752405]
        self.assertEqual(len(filter_coeficients), len(proof))
        # import matplotlib.pyplot as plt
        # plt.plot(filter)
        np.testing.assert_array_almost_equal(filter_coeficients, proof, decimal=10)
