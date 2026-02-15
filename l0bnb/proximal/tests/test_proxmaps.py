import numpy as np

from l0bnb.proximal import (lagrange_prox,
                            bound_prox,
                            dual_cost_bound,
                            dual_cost_lagrange)

def test_prox():

    v = np.random.standard_normal(100)

    lips, lam_2, M, C = 1.5, 0.02, 2, 5

    for M in [M, np.inf]:
        for lam_2 in [0, lam_2]:
            beta_star, z_star, delta_star = bound_prox(v, lips, lam_2, M, C)

            beta_star_L, z_star_L = lagrange_prox(v, lips, lam_2, M, delta_star)

            np.testing.assert_allclose(beta_star_L, beta_star)
            np.testing.assert_allclose(z_star_L, z_star)

def test_dual_costs():

    lam_2, M, C, lam_0 = 0.02, 2, 5, 4
    
    v = np.random.standard_normal(100)
    for M in [M, np.inf]:
        for lam_2 in [0, lam_2]:
            dual_cost_bound(v, lam_2, M, C)
            dual_cost_lagrange(v, lam_2, M, lam_0)
    
    
