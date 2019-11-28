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
rhos = [0, 0.1, 0.2, 0.3, 0.5, 0.6, 0.7]
supp_size = 10
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
corr = 'CLarge'  # 'I'  #

times = []
levels = []
num_of_nodes = []
optimal_support = []


for rho in rhos:
    print(f"Solving for rho = {rho}")
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

graph_plot(rhos, times, 'p', 'time', 'time (s)', True)
graph_plot(rhos, levels, 'p', 'levels', '# of levels', True)
graph_plot(rhos, num_of_nodes, 'p', 'nodes', '# of nodes', True)
graph_plot(rhos, [len(i) for i in optimal_support], 'p', 'support',
           'support size', True)


graph_plot(rhos, times, 'p', 'time (s)', 'solving times', False)
graph_plot(rhos, levels, 'p', 'levels', '# of levels', False)
graph_plot(rhos, num_of_nodes, 'p', 'nodes', '# of nodes', False)
graph_plot(rhos, [len(i) for i in optimal_support], 'p', 'features',
           'support size', False)

show_plots()
