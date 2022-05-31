import unittest

from modulation_toolbox_py.filterbank.designfilterbank import designfilterbank


class TestDesignFilterBank(unittest.TestCase):

    def test_errors(self):
        self.assertRaises(ValueError, designfilterbank, centers=[.2, .1, .3], bandwidths=[.01, .01, .01])
        self.assertRaises(ValueError, designfilterbank, centers=[.1, .2, 1.1], bandwidths=[.01, .01, .01])
        self.assertRaises(ValueError, designfilterbank, centers=[.1, .2, 1.1], bandwidths=[.01, .01, .01])
        self.assertRaises(ValueError, designfilterbank, centers=[.1, .2, .3], bandwidths=[.01, .01, 2.1])
        self.assertRaises(ValueError, designfilterbank, centers=[.1, .2, .3], bandwidths=[.01, .01, 2.1])
        self.assertRaises(ValueError, designfilterbank, centers=[.1, .2, .3], bandwidths=[.01, .01, .1], dfactor=0)
        self.assertRaises(ValueError, designfilterbank, centers=[.1, .2, .3], bandwidths=[.01, .01, .1], dfactor=0)

    def test_runs_successfully(self):
        self.assertTrue(ValueError, designfilterbank(centers=[.1, .2, .3], bandwidths=[.01, .01, .01]))

    def test_tutorial3_example(self):
        centers = [0.009375, 0.01875, 0.0375, 0.0750, 0.1500, 0.3000, 0.6000, ]
        bandwidths = [0.00625, 0.0125, 0.0250, 0.0500, 0.1000, 0.2000, 0.4000]
        designfilterbank(centers, bandwidths)

    def test_result(self):
        result = designfilterbank([.1, .2, .3, .4, .5, .6], [.01, .01, .01, .02, .02, .02])
        self.assertEqual(result['numbands'], 6)
        self.assertEqual(result['numhalfbands'], 12)
        self.assertEqual(result['dfactor'], 1)
        self.assertEqual(result['centers'], [.1, .2, .3, .4, .5, .6])
        self.assertEqual(result['bandwidths'], [.01, .01, .01, .02, .02, .02])
        self.assertEqual(len(result['afilters']), 6)

        self.assertEqual(len(result['afilters'][0]), 6600)
        self.assertEqual(len(result['afilters'][1]), 6600)
        self.assertEqual(len(result['afilters'][2]), 6600)

        self.assertEqual(len(result['afilters'][3]), 3300)
        self.assertEqual(len(result['afilters'][4]), 3300)
        self.assertEqual(len(result['afilters'][5]), 3300)

        self.assertEqual(len(result['afilterorders']), 6)
        self.assertEqual(result['afilterorders'], [6600, 6600, 6600, 3300, 3300, 3300])
        self.assertEqual(len(result['sfilters']), 6)
        self.assertEqual(result['sfilters'], [1, 1, 1, 1, 1, 1])
        self.assertEqual(result['sfilterorders'], [0, 0, 0, 0, 0, 0])
        self.assertEqual(result['fshift'], 0)
        self.assertEqual(result['stft'], 0)
        self.assertEqual(result['keeptransients'], 1)