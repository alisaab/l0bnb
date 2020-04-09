from copy import deepcopy

import numpy as np
from scipy import optimize as sci_opt

from .relaxation import relaxation_solve, coordinate_descent
from ._third_party import l0gurobi, l0mosek


class Node:
    def __init__(self, parent, zlb, zub, **kwargs):
        """
        Initialize a Node

        Parameters
        ----------
        parent: Node or None
            the parent Node
        zlb: np.array
            p x 1 array representing the lower bound of the integer variables z
        zub: np.array
            p x 1 array representing the upper bound of the integer variables z

        Other Parameters
        ----------------
        x: np.array
            The data matrix (n x p). If not specified the data will be inherited
            from the parent node
        y: np.array
            The data vector (n x 1). If not specified the data will be inherited
            from the parent node
        xi_xi: np.array
            The norm of each column in x (p x 1). If not specified the data will
            be inherited from the parent node
        l0: float
            The zeroth norm coefficient. If not specified the data will
            be inherited from the parent node
        l2: float
            The second norm coefficient. If not specified the data will
            be inherited from the parent node
        m: float
            The bound for the features (\beta). If not specified the data will
            be inherited from the parent node
        """
        # self.parent = parent
        self.parent_cost = parent.lower_bound_value if parent else None
        self.x = kwargs.get('x', parent.x if parent else None)
        self.y = kwargs.get('y', parent.y if parent else None)
        self.xi_xi = kwargs.get('xi_xi', parent.xi_xi if parent else None)
        self.l0 = kwargs.get('l0', parent.l0 if parent else None)
        self.l2 = kwargs.get('l2', parent.l2 if parent else None)
        self.m = kwargs.get('m', parent.m if parent else None)
        self.initial_guess = deepcopy(parent.lower_bound_solution) \
            if parent else None
        self.r = deepcopy(parent.r) if parent else None

        self.p = self.x.shape[1]
        self.level = parent.level + 1 if parent else 0

        self.zlb = zlb
        self.zub = zub

        self.upper_bound_value = None
        self.upper_bound_solution = None
        self.lower_bound_value = None
        self.lower_bound_solution = None
        self.lower_bound_z = None
        self.dual_value = None
        self.support = None

    def lower_solve(self, solver, reltol, inttol=None):
        if solver == 'l1cd':
            self.lower_bound_value, self.dual_value, self.lower_bound_solution,\
                self.lower_bound_z, self.r, self.support = \
                relaxation_solve(self.x, self.y, self.l0, self.l2, self.m,
                                 self.xi_xi, self.zlb, self.zub,
                                 self.initial_guess, self.r, reltol)
            self.support = list(self.support)
        elif solver == 'gurobi':
            if not inttol:
                raise ValueError('inttol should be specified to use gurobi')
            self.lower_bound_solution, self.lower_bound_z, \
                self.lower_bound_value = \
                l0gurobi(self.x, self.y, self.l0, self.l2, self.m, self.zlb,
                         self.zub, relaxed=True)
            self.support = \
                list(np.where(abs(self.lower_bound_solution) > inttol)[0])
        elif solver == 'mosek':
            self.lower_bound_solution, self.lower_bound_z, \
                self.lower_bound_value, self.dual_value = \
                l0mosek(self.x, self.y, self.l0, self.l2, self.m, self.zlb,
                        self.zub)
            self.support = \
                list(np.where(abs(self.lower_bound_solution) > inttol)[0])
        return self.lower_bound_value, self.dual_value

    def upper_solve(self):
        x_support = self.x[:, self.support]
        x_ridge = np.sqrt(2 * self.l2) * np.identity(len(self.support))
        x_upper = np.concatenate((x_support, x_ridge), axis=0)
        y_upper = np.concatenate((self.y, np.zeros(len(self.support))), axis=0)
        # account for intercept later
        res = sci_opt.lsq_linear(x_upper, y_upper, (-self.m, self.m))
        self.upper_bound_value = res.cost + self.l0 * len(self.support)
        self.upper_bound_solution = np.zeros(self.p)
        self.upper_bound_solution[self.support] = res.x
        return self.upper_bound_value

    def strong_branch_solve(self, x, l0, l2, m, xi_xi, support):
        golden_ratio = np.sqrt(l0 / l2) if l2 != 0 else np.Inf
        threshold = 2 * np.sqrt(l0 * l2) if golden_ratio <= m \
            else l0 / m + l2 * m
        _, cost, _ = \
            coordinate_descent(x, self.initial_guess,
                               self.parent_cost,
                               l0, l2, golden_ratio, threshold, m, xi_xi,
                               self.zlb, self.zub, support, self.r, 0.9)
        return cost

    def __str__(self):
        return f'level: {self.level}, lower cost: {self.lower_bound_value}, ' \
            f'upper cost: {self.upper_bound_value}'

    def __repr__(self):
        return self.__str__()

    def __lt__(self, other):
        if self.level == other.level:
            if self.lower_bound_value is None and other.lower_bound_value:
                return True
            if other.lower_bound_value is None and self.lower_bound_value :
                return False
            elif not self.lower_bound_value and not other.lower_bound_value:
                return self.parent_cost > \
                       other.parent_cost
            return self.lower_bound_value > other.lower_bound_value
        return self.level < other.level
