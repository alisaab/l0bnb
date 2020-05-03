import unittest
import math
import numpy as np
from l0bnb import fit_path
from l0bnb import gen_synthetic


class TestPreprocessing(unittest.TestCase):
    def test_preprocessing(self):
        """ Test for normalization and intercept."""
        X, y, b = gen_synthetic(n=1000, p=50, supp_size=5, snr=10e10)
        intercept = 100.5
        y += intercept
        l0bnb_sols = fit_path(X,
                              y,
                              lambda_2=0,
                              max_nonzeros=5,
                              gap_tol=0.01,
                              normalize=True,
                              intercept=True)
        self.assertTrue(np.isclose(l0bnb_sols[-1]["B"], b, rtol=1e-4).all())
        self.assertTrue(
            math.isclose(l0bnb_sols[-1]["B0"], intercept, rel_tol=1e-4))


if __name__ == '__main__':
    unittest.main()
