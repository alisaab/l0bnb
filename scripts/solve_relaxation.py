import numpy as np
import sys
from scipy import optimize as sci_opt
# import random

from scripts.generate_data import GenData as gen_data
from l0bnb.relaxation import relaxation_solve
from l0bnb.tree import BNBTree

n = 1000
p = 100000
rho = 0.5
supp_size = 5
snr = 10.0
m = 1.2
l0 = 50.0
l2 = 10.0
using_upper_bound = True
inttol = 1e-4
gaptol = 2e-2
reltol = 1e-3
branching = 'maxfrac'  # 'strong1
l1solver = 'l1cd'
mu = 0.95
bnb_algorithm = 'BFS'
corr = 'CLarge'  # 'I'  #
# generate data
print("Generating data!")
x, y, features, covariance = gen_data(corr, rho, n, p, supp_size, snr)  # CLarge
xi_xi = np.sum(x * x, axis=0)
if not using_upper_bound:
    upper_bound = sys.maxsize
    upper_bound_solution = None
else:
    support = np.nonzero(features)[0]
    x_support = x[:, support]
    x_ridge = np.sqrt(2 * l2) * np.identity(len(support))
    x_upper = np.concatenate((x_support, x_ridge), axis=0)
    y_upper = np.concatenate((y, np.zeros(len(support))), axis=0)
    res = sci_opt.lsq_linear(x_upper, y_upper, (-m, m))  # account for intercept later
    upper_bound = res.cost + l0 * len(support)
    upper_bound_solution = features
    upper_bound_solution[support] = res.x

# [0, 20000, 40000, 60000, 80000]
# [0, 20000, 40000, 60000, 80000]
# [38782]
# []


zlb = np.zeros(x.shape[1])
zlb[[0, 20000, 40000, 60000, 80000]] = 1
zub = np.ones(x.shape[1])
zub_c = np.ones(x.shape[1])
zub_c[[38782]] = 0
#
t = BNBTree(x, y)
parent_initial_guess, parent_r, initial_guess, r = \
    t.solve(l0, l2, m, upperbound=upper_bound, uppersol=upper_bound_solution,
            gaptol=gaptol, branching=branching)


# cost, beta, z, r, support = \
#     relaxation_solve(x, y, l0, l2, m, xi_xi, zlb, zub, None, None, reltol=1e-7)

cost_2, beta_2, z_2, r_2, support_2 = \
    relaxation_solve(x, y, l0, l2, m, xi_xi, zlb, zub, parent_initial_guess,
                     parent_r, reltol=1e-4)

# cost_2, beta_2, z_2, r_2, support_2 = \
#     relaxation_solve(x, y, l0, l2, m, xi_xi, zlb, zub, parent_initial_guess,
#                      parent_r, reltol=1e-7)

# beta_3, r_3, z_3, cost_3 = \
#     relaxation_solve(x, y, l0, l2, m, xi_xi, zlb, zub_c, None, None,
#                      reltol=1e-4)

cost_4, beta_4, z_4, r_4, support_4 = \
    relaxation_solve(x, y, l0, l2, m, xi_xi, zlb, zub_c, initial_guess, r,
                     reltol=1e-4)

cost_4, beta_4, z_4, r_4, support_4 = \
    relaxation_solve(x, y, l0, l2, m, xi_xi, zlb, zub_c, initial_guess, r,
                     reltol=1e-6)
# output_beta, output_z, ObjVal = l0gurobi(x, y, l0, l2, m, zlb, zub, relaxed=True)