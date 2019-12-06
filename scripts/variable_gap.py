import numpy as np
import sys
from time import time
from scipy import optimize as sci_opt
# import random

from scripts.generate_data import GenData as gen_data
from l0bnb.tree import BNBTree
from l0bnb.viz import graph_plot, show_plots

snr = 10

n = 1000
p = 100000
rho = 0.1
supp_size = 10
m = 1.2
l0 = 100.0
l2 = 20.0
using_upper_bound = True
inttol = 1e-4
gaptols = [1, 0.5, 0.1, 0.05, 1e-2, 0.005, 0.001, 0.0005, 0.0001]
reltol = 1e-5
branching = 'maxfrac'  # 'strong'  #
l1solver = 'l1cd'
mu = 1
corr = 'CLarge'  # 'I'  #

times = []
levels = []
num_of_nodes = []
optimal_support = []


for gaptol in gaptols:
    print(f"Solving for gap = {gaptol}")
    x, y, features, covariance = gen_data(corr, rho, n, p, supp_size, snr)
    print("Generated Data")
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

    t = BNBTree(x, y, reltol=reltol)
    st = time()
    sol = t.solve(l0, l2, m, upperbound=upper_bound,
                  uppersol=upper_bound_solution, gaptol=gaptol,
                  branching=branching, mu=mu, number_of_dfs_levels=2)
    times.append(time() - st)
    levels.append(max(sol[2]))
    num_of_nodes.append(t.number_of_nodes)
    optimal_support.append(list(np.where(abs(sol[0]) > inttol)[0]))  # .append(t.get_lower_optimal_node().support)

graph_plot(gaptols, times, 'gap', 'time (s)', 'time', True)
graph_plot(gaptols, levels, 'gap', 'levels', '# of levels', True)
graph_plot(gaptols, num_of_nodes, 'gap', 'nodes', '# of nodes', True)
graph_plot(gaptols, [len(i) for i in optimal_support], 'gap', 'support',
           'support size', True)

show_plots()
