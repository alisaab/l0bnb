import unittest
import math
import numpy as np
from l0bnb import fit_path
from l0bnb import gen_synthetic


class TestFitPath(unittest.TestCase):
    def test_objective(self, run_gurobi=False):
        """ Compares the objective of L0BnB with Gurobi.

        Test for different choices of n and lambda_2.
        """
        cases = [(25, 100), (50, 100), (100, 0.1), (100, 0)]  # (n,lambda_2).
        for case in cases:
            print("Case: ", case)
            n = case[0]
            lambda_2 = case[1]
            X, y, b = gen_synthetic(n=n, p=50, supp_size=10, snr=5)
            lambda_0s = []
            l0bnb_objectives = []
            l0bnb_sols = fit_path(X,
                                  y,
                                  lambda_2=lambda_2,
                                  max_nonzeros=10,
                                  gap_tol=0.00001,
                                  normalize=False,
                                  intercept=False)
            for sol in l0bnb_sols:
                lambda_0 = sol["lambda_0"]
                lambda_0s.append(lambda_0)
                b_estimated = sol["B"]
                intercept = sol["B0"]
                y_estimated = np.dot(X, b_estimated) + intercept
                r = y - y_estimated
                obj = 0.5 * np.dot(r, r) + lambda_0 * np.count_nonzero(
                    b_estimated) + lambda_2 * np.dot(b_estimated, b_estimated)
                l0bnb_objectives.append(obj)

            if run_gurobi:
                gurobi_objectives = []
                warm_start = list(
                    np.nonzero(l0bnb_sols[0]["B"])
                    [0])  # Used to define the initial bigM in Gurobi.
                gurobi_sols = fit_path(X,
                                       y,
                                       lambda_2=lambda_2,
                                       max_nonzeros=10,
                                       solver="gurobi",
                                       lambda_0_grid=lambda_0s,
                                       lambda_0_grid_warm_start=warm_start,
                                       normalize=False,
                                       intercept=False)
                for sol in gurobi_sols:
                    lambda_0 = sol["lambda_0"]
                    b_estimated = sol["B"]
                    intercept = sol["B0"]
                    y_estimated = np.dot(X, b_estimated) + intercept
                    r = y - y_estimated
                    obj = 0.5 * np.dot(r, r) + lambda_0 * np.count_nonzero(
                        b_estimated) + lambda_2 * np.dot(
                            b_estimated, b_estimated)
                    gurobi_objectives.append(obj)

            else:
                gurobi_stored_objectives = {
                    (25, 100): [
                        140.7496992712402, 134.73678148063837,
                        132.7966398218242, 131.92442262876952,
                        131.45536177721596, 130.21455471115567,
                        129.69084568186238, 127.15922098560188,
                        126.09395441700497, 125.5881926722805
                    ],
                    (50, 100): [
                        252.01738710057455, 249.60196757751095,
                        237.91468825572306, 237.1552209007437,
                        235.3958121283535, 234.38717770829828,
                        229.4140027046285, 223.64955049476524,
                        222.8225000852473, 222.00086312088638
                    ],
                    (100, 0.1): [
                        582.8799315049895, 511.31843817806424,
                        485.195434198173, 456.735738689179, 292.79760067332904
                    ],
                    (100, 0): [
                        582.8775383112129, 511.1380945270483,
                        484.75318292430666, 456.3438073701726,
                        291.9964981896575
                    ],
                }
                gurobi_objectives = gurobi_stored_objectives[case]
            for i in range(len(l0bnb_objectives)):
                objectives_close = math.isclose(l0bnb_objectives[i],
                                                gurobi_objectives[i],
                                                rel_tol=1e-4)
                if not objectives_close:
                    print("Objectives are different!")
                    print("Case: ", case)
                    print("l0bnb: ", l0bnb_objectives)
                    print("gurobi: ", gurobi_objectives)
                self.assertTrue(objectives_close)


if __name__ == '__main__':
    unittest.main()
