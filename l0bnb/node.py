import copy
import numpy as np
from scipy import optimize as sci_opt

from .relaxation import relaxation_solve, coordinate_descent
from .gurobi_solve import l0gurobi


class Node:
    def __init__(self, node_num, parent, new_zlb, new_zub):
        self.node_num = node_num
        self.parent = parent
        self.zlb = new_zlb
        self.zub = new_zub

        self.level = parent.level + 1 if parent else 0
        self.initial_guess = copy.deepcopy(self.parent.lower_bound_solution) if parent else None
        self.r = copy.deepcopy(self.parent.r) if parent else None

        self.upper_bound = None
        self.upper_bound_solution = None
        self.lower_bound_solution = None
        self.lower_bound_z = None
        self.lower_bound = None

    def compute_lower_bound(self, x, y, m, lambda_value, reltol, solver, initial_guess):
        if solver == 'l1cd':
            self.lower_bound_solution, self.r, self.lower_bound_z, self.lower_bound = \
                relaxation_solve(x, y, m, lambda_value/m, self.zlb, self.zub, initial_guess, self.r, reltol)
        elif solver == 'gurobi':
            self.lower_bound_solution, self.lower_bound_z, self.lower_bound = \
                l0gurobi(x, y, m, lambda_value, self.zlb, self.zub, relaxed=True)
        return self.lower_bound

    def compute_upper_bound(self, x, y, m, lambda_value, int_tol):
        support = list(np.where(abs(self.lower_bound_solution) > int_tol)[0])
        x_support = x[:, support]
        res = sci_opt.lsq_linear(x_support, y, (-m, m))  # account for intercept later
        self.upper_bound = res.cost + lambda_value*len(support)
        self.upper_bound_solution = np.zeros(x.shape[1])
        self.upper_bound_solution[support] = res.x
        return self.upper_bound

    def strong_branch_solve(self, x, m, lambda_value, support):
        _, cost, _ = coordinate_descent(x, self.initial_guess, self.parent.lower_bound, m, lambda_value, self.zlb,
                                        self.zub, support, self.r, 0.9)
        return cost
