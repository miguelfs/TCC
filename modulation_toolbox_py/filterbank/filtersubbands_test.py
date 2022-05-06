import unittest
import numpy as np

from modulation_toolbox_py.filterbank.designfilterbank import designfilterbank
from modulation_toolbox_py.filterbank.filtersubbands import buffer, convbuffer, colcircshift, STFT, \
    trimfiltertransients, filtersubbands, fastconv, bandpassFilter
from modulation_toolbox_py.index import index


class TestFiltersubbands(unittest.TestCase):

    def test_filtersubbands(self):
        filtbankparams = designfilterbank([.2, .4], [.05, .08], [.01, .01], 4, True)
        x = np.sin(2 * np.pi * np.arange(0, 1.1, .1)).reshape(1, -1)
        S = filtersubbands(x, filtbankparams)
        self.assertEqual(np.shape(S), (2, 168))
        import matplotlib.pyplot as plt
        plt.plot(np.abs(S))
        plt.show()
        plt.plot(np.abs(S[0, :]))
        plt.show()
        pass

    def test_STFT(self):
        np.testing.assert_array_equal(
            STFT([0, 0, 0, 0], [1, 1], 1, 2, fshift=False),
            [[0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0]]
        )
        np.testing.assert_array_equal(
            STFT([1, 1j, 0, -.2 - .2j], [4], 1, 2, fshift=True),
            [[4, 4j, 0, -.8 - .8j],
             [4, -4j, 0, .8 + .8j]]
        )

    def test_colcircshift(self): np.testing.assert_array_equal(
        colcircshift([[1, 2, 3, 4], [10, 20, 30, 40]], [1, 0, 0, 0]),
        [[10, 2, 3, 4], [1, 20, 30, 40]])

    def test_bandpassFilter(self):
        np.testing.assert_array_almost_equal(
            bandpassFilter(np.array([1, 2, 1]), .2, np.array([1, 1]), 1, False),
            [1, 2.809 + .5878j, 2.618 + 1.1756j, .809 + .5878j],
            decimal=4)
        np.testing.assert_array_almost_equal(
            bandpassFilter(np.array([[1, 2, 1], [1, 1, 1]]), .2, np.array([1, 1]), 1, False),
            [[1, 2, 1],
             [1.809 + .5878j, 2.618 + 1.1756j, 1.809 + .5878j],
             [.809 + .5878j, .809 + .5878j, .809 + .5878j],
             ], decimal=4, )
        np.testing.assert_array_almost_equal(
            bandpassFilter(np.array([[1, 2, 1], [1, 1, 1]]), .4, np.array([1, 1]), 2, False),
            [[1, 2, 1],
             [.309 + .9511j, .309 + .9511j, .309 + .9511j],
             ], decimal=4, )

    # np.testing.assert_array_almost_equal(
    #     bandpassFilter(np.array([[0, 1, 0, 1]]), 1, np.array([1, 1]), 2, True),
    #     [[0, -1, -1]], decimal=4,
    # )

    def test_fastconv(self):
        x = np.array(
            [[1, 2, 1],
             [-1, 0, 0]]
        )
        h = np.array([[1, 1]])
        np.testing.assert_array_almost_equal(fastconv(h, x), np.array([[1, 2, 1], [0, 2, 1], [-1, 0, 0]]))
        x = np.array(
            [[0, 1, 0],
             [-1, 0, 0],
             [2, 0, 0]]
        )
        h = np.array([[1, 1]])
        np.testing.assert_array_almost_equal(fastconv(h, x),
                                             np.array([
                                                 [0, 1, 0],
                                                 [-1, 1, 0],
                                                 [1, 0, 0],
                                                 [2, 0, 0]
                                             ]))
    def test_index(self):
        self.assertEqual(index(42, 0), 42)
        self.assertEqual(index(42, 1), 42)
        self.assertEqual(index(42.0, 0), 42.0)
        self.assertEqual(index(42.0, 1), 42.0)
        self.assertEqual(index([42], 0), 42)
        self.assertEqual(index(np.array([42]), 0), 42)
        self.assertEqual(index(np.array([42]), 1), 42)
        self.assertEqual(index(np.array([[42]]), 0), 42)
        self.assertEqual(index(np.array([[42]]), 1), 42)
        self.assertEqual(index([1, 42], 1), 42)
        self.assertEqual(index(np.array([1, 42]), 1), 42)
        self.assertEqual(index(np.array([[1, 42]]), 0), 1)
        self.assertEqual(index(np.array([[1, 42]]), 1), 42)


    def test_trimfiltertransients(self):
        filtbankparams = {'dfactor': 2, 'afilterorders': [[1]]}
        np.testing.assert_array_equal(
            trimfiltertransients(
                [[1, 1, 1, 1],
                 [10, 10, 10, 10],
                 [0, 0, 0, 0],
                 [2, 2, 2, 2]], filtbankparams, 4),
            [[1, 1],
             [10, 10],
             [0, 0],
             [2, 2]]
        )

    def test_convbuffer(self):
        np.testing.assert_array_equal(
            convbuffer([1, -1, 1, -1], 2, 0, 1),
            [[1, -1, 1, -1],
             [-1, 1, -1, 0]]
        )
        np.testing.assert_array_equal(
            convbuffer([0, .5, -1, 1, 0, -1, .2, 0], 3, 2, 2),
            [[-1, 0, .2],
             [1, -1, 0],
             [0, .2, 0]]
        )
        np.testing.assert_array_equal(
            convbuffer([0, .5, -1, 1j, 0, -1, .2, 0], 3, 2, 2),
            [[-1, 0, .2],
             [1j, -1, 0],
             [0, .2, 0]]
        )
        np.testing.assert_array_equal(
            convbuffer([1, 1j, -1, -1j, .5, .5 + .5j, 0, 0], 4, 1, 2),
            [[1j, -1j, .5 + .5j, 0],
             [-1, .5, 0, 0],
             [-1j, .5 + .5j, 0, 0],
             [.5, 0, 0, 0]]
        )

    def test_buffer_new(self):
        np.testing.assert_array_equal(
            buffer([1, 2, 3, 4, 5, 6, 7, 8], 1, -1, 0),
            [[1, 3, 5, 7]]
        )
        np.testing.assert_array_equal(
            buffer([1, 2, 3, 4, 5, 6, 7, 8], 1, -2, 0),
            [[1, 4, 7]]
        )
        np.testing.assert_array_equal(
            buffer([1, 2, 3, 4, 5, 6, 7, 8], 2, -2, 0),
            [[1, 5], [2, 6]]
        )

    def test_buffer(self):
        v = np.array([[1.], [-1.], [1.], [-1.], [0.]])
        v2 = np.array([1, -1, 1, -1, 0]).T
        np.testing.assert_array_equal(
            buffer(v, 2, 1, 'nodelay'),
            [[1, -1, 1, -1],
             [-1, 1, -1, 0]])
        np.testing.assert_array_equal(
            buffer(v2, 2, 1, 'nodelay'),
            [[1, -1, 1, -1],
             [-1, 1, -1, 0]])
        np.testing.assert_array_equal(
            buffer([1, 2, 3, 4, 5, 6, 7, 8], 4, 0),
            np.array([
                [1, 5],
                [2, 6],
                [3, 7],
                [4, 8]]))
        np.testing.assert_array_equal(
            buffer([1, 2, 3, 4, 5, 6, 7, 8], 4, 1),
            np.array([[0, 3, 6],
                      [1, 4, 7],
                      [2, 5, 8],
                      [3, 6, 0]]))
        np.testing.assert_array_equal(
            buffer([1, 2, 3, 4, 5, 6, 7, 8], 4, 2),
            np.array(
                [[0, 1, 3, 5],
                 [0, 2, 4, 6],
                 [1, 3, 5, 7],
                 [2, 4, 6, 8]]))
        np.testing.assert_array_equal(
            buffer([1, 2, 3, 4, 5, 6, 7, 8, 9, 0], 4, 2, 'nodelay'),
            np.array([
                [1, 3, 5, 7],
                [2, 4, 6, 8],
                [3, 5, 7, 9],
                [4, 6, 8, 0]]))
        np.testing.assert_array_equal(
            buffer([1, 2, 3, 4, 5, 6, 7, 8], 4, 2, 'nodelay'),
            np.array([
                [1, 3, 5],
                [2, 4, 6],
                [3, 5, 7],
                [4, 6, 8]]))
        np.testing.assert_array_equal(
            buffer([1, 2, 3, 4, 5, 6, 7, 8], 4, 1, 'nodelay'),
            np.array([
                [1, 4, 7],
                [2, 5, 8],
                [3, 6, 0],
                [4, 7, 0]]))
        np.testing.assert_array_equal(
            buffer([1, 2, 3, 4, 5, 6, 7, 8], 3, 2, 'nodelay'),
            np.array(
                [[1, 2, 3, 4, 5, 6],
                 [2, 3, 4, 5, 6, 7],
                 [3, 4, 5, 6, 7, 8]]
            ))
