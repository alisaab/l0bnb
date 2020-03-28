import sys
from time import time

from scipy import optimize as sci_opt
import cvxpy as cp
import numpy as np

from scripts.generate_data import GenData as gen_data
from l0bnb.tree import BNBTree

P_VALUES = [1e2, 1e3]  # , 1e4, 1e5, 1e6]
N = int(1e3)
SNR = 10.0
SUPPORT_SIZE = 10
RHO = 0
L0 = 100
L2 = 5
INT_TOL = 1e-4
GAP_TOL = 1e-2
REL_TOL = 1e-5
CORR_MATRIX = 'I'

M = 1.2


def third_party_optimizer(y, x, l0, l2, m, optimizer, warm_start, gaptol,
                          inttol):
    st = time()
    p = x.shape[1]
    B = cp.Variable(p)
    z = cp.Variable(p, boolean=True)
    s = cp.Variable(p)

    cons_box = [z>=0, z<=1, s>=0]
    cons_bigm = [B >= -m*z, B <= m*z]
    cons_conic = [z[i] >= cp.quad_over_lin(B[i], s[i]) for i in range(p)]
    constraints = cons_box + cons_bigm + cons_conic

    # Build the objective
    obj = cp.Minimize(0.5*cp.sum_squares(x*B - y) + l0*cp.sum(z) + l2*cp.sum(s))

    # Define the problem
    prob = cp.Problem(obj, constraints)
    B.value = warm_start
    prob.solve(warm_start=True)
    if optimizer == 'mosek':
        result = prob.solve(solver=cp.MOSEK, verbose=False,
                            mosek_params={'MSK_DPAR_MIO_TOL_REL_GAP': gaptol,
                                          'MSK_DPAR_MIO_TOL_FEAS': inttol
                                          }
                            )
    if optimizer == 'gurobi':
        result = prob.solve(solver=cp.GUROBI, verbose=False,
                            gurobi_params={'MIPGap': gaptol,
                                           'IntFeasTol': inttol})
    sol_time = time() - st
    return result, B.value, sol_time


def l0bnb_solve(y, x, l0, l2, m, reltol, gaptol, inttol, upper_bound,
                upper_bound_solution):
    t = BNBTree(x, y, reltol=reltol, inttol=inttol)
    st = time()
    uppersol, obj_value, lower_bound, best_gap = \
        t.solve(l0, l2, m, gaptol=gaptol, upperbound=upper_bound,
                uppersol=upper_bound_solution, verbose=False)
    tot_time = time() - st
    return obj_value, uppersol, best_gap, tot_time


def get_upper_bound(y, x, beta, l0, l2, m, ):
    support = np.nonzero(beta)[0]
    x_support = x[:, support]
    x_ridge = np.sqrt(2 * l2) * np.identity(len(support))
    x_upper = np.concatenate((x_support, x_ridge), axis=0)
    y_upper = np.concatenate((y, np.zeros(len(support))), axis=0)
    res = sci_opt.lsq_linear(x_upper, y_upper,
                             (-m, m))  # account for intercept later
    upper_bound = res.cost + l0 * len(support)
    upper_bound_solution = beta
    upper_bound_solution[support] = res.x
    return upper_bound, upper_bound_solution


def _print(obj_val, sol_time, best_gap, sol, features, solver):
    print(f'solver: {solver}, sol_time:{sol_time:.3} seconds, '
          f'Obj: {obj_val:.3}, gap <= {best_gap:.3}, recovery error: '
          f'{max(abs(sol - features))/max(abs(features)):.3}')


def _package(obj_val, sol_time, best_gap, sol, features, p, solver, results):
    if solver not in results:
        results[solver] = {}
    results[solver][p] = {'obj': obj_val, 'sol_time': sol_time,
                          'gap <=': best_gap, 'rec_error':
                              max(abs(sol - features))/max(abs(features))}


if __name__ == '__main__':
    results = {}
    for p in P_VALUES:
        p = int(p)
        print(f"Solving for p = {p}")
        x, y, features, covariance = \
            gen_data(CORR_MATRIX, RHO, N, p, SUPPORT_SIZE, SNR)
        upper_bound, upper_bound_solution = sys.maxsize, None  # get_upper_bound(y, x, features, L0, L2, M)

        obj_value, sol, best_gap, sol_time = \
            l0bnb_solve(y, x, L0, L2, M, REL_TOL, GAP_TOL, INT_TOL, upper_bound,
                        upper_bound_solution)
        _print(obj_value, sol_time, best_gap, sol, features, 'l0bnb')
        _package(obj_value, sol_time, best_gap, sol, features, p, 'l0bnb',
                 results)
        obj_value, sol, sol_time = third_party_optimizer(y, x, L0, L2, M,
                                                         'mosek',
                                                         upper_bound_solution,
                                                         GAP_TOL, INT_TOL)
        _print(obj_value, sol_time, GAP_TOL, sol, features, 'mosek')
        _package(obj_value, sol_time, GAP_TOL, sol, features, p, 'mosek',
                 results)
        obj_value, sol, sol_time = third_party_optimizer(y, x, L0, L2, M,
                                                         'gurobi',
                                                         upper_bound_solution,
                                                         GAP_TOL, INT_TOL)
        _print(obj_value, sol_time, GAP_TOL, sol, features, 'gurobi')
        _package(obj_value, sol_time, GAP_TOL, sol, features, p, 'gurobi',
                 results)
        print('--------------------------------------------------------------')
    print(results)

