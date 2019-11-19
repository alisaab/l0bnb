import queue
import sys

import numpy as np

from .node import Node
from .utilities import branch, is_integral


class BNBTree:
    def __init__(self, x, y, bnb_algorithm='BFS', inttol=1e-4, reltol=1e-4):
        """
        Creates a branch & bound Tree to solve the integer programming problem.

        :param x: np.array
            n x p array
        :param y: np.array
            n x 1 array
        :param bnb_algorithm: str
            'BFS' or 'DFS'
        :param inttol: float
            the tolerance for considering a number an integer
        :param reltol: float
            The relative tolerance of the change in the objective value after
            which the relaxation solve would be terminated
        """

        self.x = x
        self.y = y
        self.bnb_algorithm = bnb_algorithm
        self.inttol = inttol
        self.reltol = reltol
        self.xi_xi = np.sum(x * x, axis=0)

        # The number of features
        self.p = x.shape[1]
        self.n = x.shape[0]

        if bnb_algorithm == 'BFS':
            self.node_queue = queue.Queue()
        # elif bnb_algorithm == 'DFS':
        #     self.node_queue = queue.LifoQueue()
        else:
            raise ValueError(bnb_algorithm + ' is not supported')

        self.levels = {}
        self.leaves = []

        self.root = None

    def solve(self, l0, l2, m, gaptol=1e-2, upperbound=sys.maxsize,
              uppersol=None, branching='maxfrac', l1solver='l1cd', mu=0.95):
        """
        Solve the nonlinear optimization problem using a branch and bound
        algorithm

        Parameters
        ----------
        l0: float
            The zeroth norm coefficient
        l2: float
            The second norm coefficient
        m: float
            features bound
        gaptol: float
            the relative gap between the upper and lower bound after which the
            algorithm will be terminated
        upperbound: float
            the upper bound of the objective value
        uppersol: np.array
            (p x 1) array representing the solution of the upper bound
        branching: str
            'maxfrac' or 'strong'
        l1solver: str
            'l1cd' or 'gurobi'
        mu: float
            Used with strong branching

        Returns
        -------
        dict
            The solution
        """
        # upper and lower bounds
        zlb = np.zeros(self.p)
        zub = np.ones(self.p)

        # root node
        self.root = Node(None, zlb, zub, x=self.x, y=self.y, l0=l0, l2=l2, m=m,
                         xi_xi=self.xi_xi)
        self.node_queue.put(self.root)

        # lower and upper bounds initialization
        lower_bound = {}
        level_count = {0: 1}
        best_gap = upperbound

        min_open_level = 0

        while self.node_queue.qsize() > 0:
            # get node
            current_node = self.node_queue.get()
            # prune?
            if current_node.parent and upperbound <= \
                    current_node.parent.lower_bound_value:
                level_count[current_node.level] -= 1
                self.leaves.append(current_node)
                continue

            # calculate lower bound and update
            current_lower_bound = current_node.\
                lower_solve(l1solver, self.reltol, self.inttol)
            lower_bound[current_node.level] = \
                min(current_lower_bound,
                    lower_bound.get(current_node.level, sys.maxsize))
            level_count[current_node.level] -= 1

            # update gap?
            if level_count[min_open_level] == 0:
                del level_count[min_open_level]
                min_value = lower_bound[min_open_level]
                best_gap = (upperbound - min_value)/min_value
                print(min_open_level, min_value, upperbound, best_gap)
                # arrived at a solution?
                if best_gap <= gaptol:
                    self.leaves += [current_node] + list(self.node_queue.queue)
                    return uppersol, upperbound, lower_bound, best_gap
                min_open_level += 1

            # integral solution?
            if is_integral(current_node.lower_bound_z, self.inttol):
                current_upper_bound = current_lower_bound
                if current_upper_bound < upperbound:
                    upperbound = current_upper_bound
                    uppersol = current_node.lower_bound_solution
                    self.leaves.append(current_node)
            # branch?
            elif current_lower_bound < upperbound:
                current_upper_bound = current_node.upper_solve()
                if current_upper_bound < upperbound:
                    upperbound = current_upper_bound
                    uppersol = current_node.upper_bound_solution
                left_node, right_node = branch(current_node, self.x, l0, l2, m,
                                               self.xi_xi, self.inttol,
                                               branching, mu)
                level_count[current_node.level + 1] = \
                    level_count.get(current_node.level + 1, 0) + 2
                self.node_queue.put(right_node)
                self.node_queue.put(left_node)

            # prune?
            else:
                self.leaves.append(current_node)

        return uppersol, upperbound, lower_bound, best_gap

    def get_lower_optimal_node(self):
        self.leaves = sorted(self.leaves)
        if self.leaves[-1].lower_bound_value:
            return self.leaves[-1]
        else:
            return self.leaves[-1].parent

    @staticmethod
    def support_list(current_node):
        list_ = []
        while current_node:
            list_.append(current_node.support)
            current_node = current_node.parent
        return list_

    def optimal_support_list(self):
        list_ = []
        current_node = self.get_lower_optimal_node()
        while current_node:
            list_.append(current_node.support)
            current_node = current_node.parent
        return list_
