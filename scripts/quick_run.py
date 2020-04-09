from time import time
# import random

from profiler import profile
from scripts.generate_data import GenData as gen_data
from l0bnb.tree import BNBTree

n = 100
p = 10000
rho = 0.5
supp_size = 15
snr = 10.0
m = 1.2
l0 = 100.0
l2 = 20.0
using_upper_bound = True
inttol = 1e-4
gaptol = 1e-2
reltol = 1e-5
branching = 'maxfrac'  # 'strong'  #
l1solver = 'mosek'
mu = 1
bnb_algorithm = 'BFS'
corr = 'I'  # 'CLarge'  #
# generate data
print("Generating data!")
x, y, features, covariance = gen_data(corr, rho, n, p, supp_size, snr)  # CLarge

t = BNBTree(x, y, reltol=reltol)
st = time()
sol = t.solve(l0, l2, m, warm_start=features, gaptol=gaptol,
              branching=branching,  mu=mu, verbose=True, l1solver=l1solver)
