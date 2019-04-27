import numpy as np
import sys
import networkx as nx
import matplotlib.pyplot as plt
from time import time
from scipy import optimize as sci_opt
# import random

from l0bnb.bnb import bnb
from profiler import profile
from generate_data import GenData as gen_data

n = 1000
p = 100000
rho = 0
supp_size = 5
snr = 10
m = 1.2
l0 = 50
l2 = 50
using_upper_bound = True
inttol = 1e-4
gaptol = 1e-5
reltol = 1e-4
branching = 'maxfrac'
l1solver = 'l1cd'
mu = 0.9

# generate data
x, y, features, covariance = gen_data('I', rho, n, p, supp_size, snr)

# from l0bnb.node import Node
# zlb = np.zeros(p)
# zlb[3] = 1
# zub = np.ones(x.shape[1])
# try_node_r = Node(0, None, zlb, zub)
# try_node_r.compute_lower_bound(x, y, l0, l2, m, reltol, 'l1cd', None)
# print(try_node_r.lower_bound, '\n', try_node_r.lower_bound_solution, '\n', try_node_r.lower_bound_z)
# print('---------------------------')
# try_node_g = Node(0, None, zlb, zub)
# try_node_g.compute_lower_bound(x, y, l0, l2, m, reltol, 'gurobi', None)
# print(try_node_g.lower_bound, '\n', try_node_g.lower_bound_solution, '\n', try_node_g.lower_bound_z)
# from l0bnb.gurobi_solve import l0gurobi
# from l0bnb.relaxation import relaxation_solve
# a,b,c = l0gurobi(x, y, l0, l2, m, zlb, zub, False)
# print('----------')
# a1, _, b1, c1 = relaxation_solve(x, y, l0, l2, m, zlb, zub, None, None)
# print(a,b,c)

# for i in range(x.shape[1]):
#     x[:, i] /= np.linalg.norm(x[:, i])

# Find best upper bound
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

# Solve the best subset problem
# st = time()
# solB, objB, edges, lower_bound = bnb(x, y, l0, l2, m, upperbound=upper_bound, uppersol=upper_bound_solution,
#                                      inttol=inttol, gaptol=gaptol, reltol=reltol, branching=branching, mu=mu,
#                                      l1solver=l1solver)
# tot_time = time() - st
# G = nx.Graph()
# G.add_edges_from(edges)
# pos = nx.drawing.nx_agraph.graphviz_layout(G, prog='dot')
# nx.draw(G, pos)
# to_write = ('n = ' + str(n) + '\np = ' + str(p) + '\nsnr = ' + str(snr) + '\nsupport size = ' + str(supp_size)
#             + '\nlambda = ' + str(l0) + '\nm = ' + str(m) + '\ntotal time = ' + str(round(tot_time, 3)) +
#             '\nupper bound = ' + str(using_upper_bound) + '\nbranching = ' + branching + '\nl1solver = ' + l1solver
#             + '\ngap = ' + str(gaptol) + '\nnumber of nodes = ' + str(len(edges)) + '\nlevels = ' + str(
#             max(lower_bound)))
# plt.text(20, 200, to_write)
# plt.savefig('11.eps', format='eps', dpi=1000)

# upper_bound = 0.5*np.dot(y-np.matmul(X, features), y-np.matmul(X, features)) + 10*sum(features != 0)
# print(upper_bound)
print(profile(bnb, x, y, l0, l2, m, upperbound=upper_bound, uppersol=upper_bound_solution,
              inttol=inttol, gaptol=gaptol, reltol=reltol, branching=branching, mu=mu,
              l1solver=l1solver))
# beta, z, cost = relaxation_solve(X, y, 120, 100/120, lb, ub, kes)
# st = time()
# solG, zG, objG = best_subset_selection(X, y, 120, 10, {i: 0 for i in range(p)}, {i: 1 for i in range(p)}, relaxed=False)
# print(time() - st)
# st = time()
