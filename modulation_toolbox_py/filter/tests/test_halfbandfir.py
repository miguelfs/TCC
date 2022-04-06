from modulation_toolbox_py.filter.halfbandfir import halfbandfir
import unittest
import matlab.engine
import numpy as np

eng = matlab.engine.start_matlab()


class TestHalfbandFIR(unittest.TestCase):
    def test_illegal_argument_combinations(self):
        self.assertRaises(ValueError, halfbandfir, N=10, fp=.1, Dev=.1)
        self.assertRaises(ValueError, halfbandfir, N=10, fp=.1, minOrder=True)
        self.assertRaises(ValueError, halfbandfir, N=10, fp=.1, Dev=.1, minOrder=True)
        self.assertRaises(ValueError, halfbandfir, N=10, fp=.6, Dev=.1, minOrder=True)

    def test_filter_coefficient_values_min_order_true(self):
        np.testing.assert_array_almost_equal(
            [-0.0072, 0, 0.0136, 0, -0.0263, 0, 0.0486, 0, -0.0965, 0, 0.3150, 0.5000, 0.3150, 0, -0.0965, 0, 0.0486, 0,
             -0.0263, 0, 0.0136, 0, -0.0072, ],
            halfbandfir(minOrder=True, fp=.4, Dev=.01), decimal=4)

        np.testing.assert_array_almost_equal(
            [0.2929, 0.5000, 0.2929],
            halfbandfir(minOrder=True, fp=.25, Dev=.25), decimal=4)

        np.testing.assert_array_almost_equal(
            [0.2563, 0.5000, 0.2563],
            halfbandfir(minOrder=True, fp=.1, Dev=.1), decimal=4)

    def test_filter_coefficients_values_min_order_false(self):
        np.testing.assert_array_almost_equal(
            [0.2563, 0.5000, 0.2563],
            halfbandfir(N=2, fp=.1), decimal=4)

        np.testing.assert_array_almost_equal(
            [0.0066, 0, -0.0510, 0, 0.2944, 0.5000, 0.2944, 0, -0.0510, 0, 0.0066],
            halfbandfir(N=10, fp=.1), decimal=4)

        np.testing.assert_array_almost_equal(
            [0.0059, 0, -0.0488, 0, 0.2930, 0.5000, 0.2930, 0, -0.0488, 0, 0.0059],
            halfbandfir(N=10, fp=.01), decimal=4)

        # np.testing.assert_array_almost_equal(
        #     [0.0059, 0, - 0.0488, 0, 0.2930, 0.5000, 0.2930, 0, - 0.0488, 0, 0.0059],
        #     halfbandfir(N=10, fp=.1), decimal=4)

    def test_len_min_order_true(self):
        self.assertEqual(23, len(halfbandfir(minOrder=True, fp=.4, Dev=.01)))
        self.assertEqual(3, len(halfbandfir(minOrder=True, fp=.25, Dev=.25)))
        self.assertEqual(3, len(halfbandfir(minOrder=True, fp=.1, Dev=.1)))
        self.assertEqual(95, len(halfbandfir(minOrder=True, fp=.45, Dev=0.0001)))

    def test_len_min_order_false(self):
        self.assertEqual(11, len(halfbandfir(N=10, fp=.4)))
        self.assertEqual(19, len(halfbandfir(N=18, fp=.25)))
        self.assertEqual(23, len(halfbandfir(N=22, fp=.1)))
        self.assertEqual(3, len(halfbandfir(N=2, fp=.1)))

