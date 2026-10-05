import tempfile
from pathlib import Path
import unittest
import numpy as np
from scripts.curve90.labels import LabelStore


class LabelStoreTests(unittest.TestCase):
    def test_roundtrip_and_independent_read_ownership(self):
        with tempfile.TemporaryDirectory() as temp:
            labels = LabelStore(Path(temp)/'labels.dat', (3, 4, 8, 7))
            values = np.arange(4*8*7, dtype='float64').reshape(4, 8, 7)/11
            values[0, 0, 0] = np.nan
            labels[1] = values
            copy = labels[1]
            np.testing.assert_array_equal(copy, values.astype('float32'))
            labels[1] = np.zeros_like(values)
            np.testing.assert_array_equal(copy, values.astype('float32'))
            np.testing.assert_array_equal(labels[1], 0)
            np.testing.assert_array_equal(labels[0], 0)
            self.assertEqual(labels.shape, (3, 4, 8, 7))


if __name__ == '__main__':
    unittest.main()
