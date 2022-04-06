import unittest

import numpy as np

from modulation_toolbox_py.filterbank.designfilterbank import designfilterbank
from modulation_toolbox_py.moddecomp import parsefilterbank, parsedemod


class TestModdecomp(unittest.TestCase):
    def test_parseinputs_cog_properly(self):
        filtbankparams, freqdiv, dfactor = parsefilterbank(100, ['cog', 0.1, 0.05], freqdiv=[.5, .6], dfactor=1)
        self.assertEqual(filtbankparams['numbands'], 1)
        self.assertEqual(filtbankparams['numhalfbands'], 2)
        self.assertEqual(filtbankparams['dfactor'], 1)
        self.assertEqual(filtbankparams['centers'], [.011])
        self.assertEqual(filtbankparams['bandwidths'], [.002])
        self.assertEqual(filtbankparams['afilterorders'][0], [33000])
        self.assertEqual(filtbankparams['sfilterorders'][0], 0)
        self.assertEqual(filtbankparams['fshift'], 0)
        self.assertEqual(filtbankparams['stft'], 0)
        self.assertEqual(filtbankparams['keeptransients'], 0)
        self.assertEqual(freqdiv, [.5, .6])
        self.assertEqual(dfactor, 1)

    def test_parseinputs_harm_properly(self):
        filtbankparams, freqdiv, dfactor = parsefilterbank(1000, ['harm'], .1, 2)
        self.assertEqual(filtbankparams, {})
        self.assertEqual(freqdiv, .1)
        self.assertEqual(dfactor, 2)

    def test_parseinputs_raise_error(self):
        self.assertRaises(ValueError, parsefilterbank, 100, ['harm'], [.5, .6], 1)

    def test_parsedemod_properly(self):
        filterbank = designfilterbank([.25, .75], [.1, .1], [.01, .01], 1, 0)
        np.testing.assert_array_equal(
            np.array(parsedemod(1000, ['cog'], filterbank, 1, 1, 1), dtype=object),
            np.array(['cog', [1], [1], [.25, .75], [.1, .1]], dtype=object),
        )
        np.testing.assert_array_equal(
            parsedemod(1000, ['hilb'], filterbank, 1, 1, 1),
            ['hilb'],
        )
        np.testing.assert_array_equal(
            np.array(parsedemod(2000, ['harmcog'], filterbank, 500, 4, 2), dtype=object),
            np.array(['harmcog', .5, 3, 1000, 200, 100, [], .5], dtype=object),
        )
        np.testing.assert_array_equal(
            np.array(parsedemod(200, ['harm'], filterbank, 50, 2, 10), dtype=object),
            np.array(['harm', .5, 3, 100, [], .5], dtype=object),
        )
