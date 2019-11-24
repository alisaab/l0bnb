import numpy as np
import sys
import matplotlib.pyplot as plt
from time import time
from scipy import optimize as sci_opt
# import random

from scripts.generate_data import GenData as gen_data
from l0bnb.tree import BNBTree

# snr = [100, 70, 50, 30, 20, 15, 10]
snr = [5]
n = 1000
p = 100000
rho = 0.8
supp_size = 5
m = 1.2
l0 = 100.0
l2 = 20.0
using_upper_bound = True
inttol = 1e-4
gaptol = 1e-2
reltol = 1e-5
branching = 'maxfrac'  # 'strong'  #
l1solver = 'l1cd'
mu = 1
bnb_algorithm = 'DFS'
corr = 'I'  # 'CLarge'  #
number_of_dfs_levels = 2
# generate data
print("Generating data!")

times = []
levels = []
num_of_nodes = []
optimal_support = []


for i in snr:
    x, y, features, covariance = gen_data(corr, rho, n, p, supp_size, i)
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

    t = BNBTree(x, y, reltol=reltol, bnb_algorithm=bnb_algorithm)
    st = time()
    sol = t.solve(l0, l2, m, upperbound=upper_bound,
                  uppersol=upper_bound_solution, gaptol=gaptol,
                  branching=branching, mu=mu,
                  number_of_dfs_levels=number_of_dfs_levels)
    times.append(time() - st)
    levels.append(max(sol[2]))
    num_of_nodes.append(t.number_of_nodes)
    optimal_support.append(t.get_lower_optimal_node().support)

plt.figure()
plt.plot(snr, times)
plt.figure()
plt.plot(snr, levels)
plt.figure()
plt.plot(snr, num_of_nodes)
plt.figure()
plt.plot(snr, [len(i) for i in optimal_support])
plt.show()
