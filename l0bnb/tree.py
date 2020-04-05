import queue
import sys

import numpy as np

from .node import Node
from .utilities import branch, is_integral


class BNBTree:
    def __init__(self, x, y, inttol=1e-4, reltol=1e-4):
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
        self.inttol = inttol
        self.reltol = reltol
        self.xi_xi = np.sum(x * x, axis=0)

        # The number of features
        self.p = x.shape[1]
        self.n = x.shape[0]

        self.node_bfs_queue = queue.Queue()
        self.node_dfs_queue = queue.LifoQueue()

        self.levels = {}
        # self.leaves = []
        self.number_of_nodes = 0

        self.root = None

    def solve(self, l0, l2, m, gaptol=1e-2, upperbound=sys.maxsize,
              uppersol=None, branching='maxfrac', l1solver='l1cd', mu=0.95,
              number_of_dfs_levels=0, verbose=True):
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
        number_of_dfs_levels: int
            number of levels to solve as dfs

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
        self.node_bfs_queue.put(self.root)

        # lower and upper bounds initialization
        lower_bound = {}
        dual_bound = {}
        self.levels = {0: 1}
        best_gap = upperbound

        min_open_level = 0
        if verbose:
            print(f'solving using {number_of_dfs_levels} dfs levels')

        while self.node_bfs_queue.qsize() > 0 or self.node_dfs_queue.qsize() > 0:
            # get node
            if self.node_dfs_queue.qsize() > 0:
                current_node = self.node_dfs_queue.get()
            else:
                current_node = self.node_bfs_queue.get()
            # print(current_node.level, upperbound, self.levels)
            # prune?
            if current_node.parent_cost and upperbound <= \
                    current_node.parent_cost:
                self.levels[current_node.level] -= 1
                # self.leaves.append(current_node)
                continue

            # calculate lower bound and update
            self.number_of_nodes += 1
            current_lower_bound, current_dual_cost = current_node.\
                lower_solve(l1solver, self.reltol, self.inttol)
            lower_bound[current_node.level] = \
                min(current_lower_bound,
                    lower_bound.get(current_node.level, sys.maxsize))
            dual_bound[current_node.level] = \
                min(current_dual_cost,
                    dual_bound.get(current_node.level, sys.maxsize))
            self.levels[current_node.level] -= 1
            # update gap?
            if self.levels[min_open_level] == 0:
                del self.levels[min_open_level]
                min_value = max([j for i, j in dual_bound.items()
                                 if i <= min_open_level])
                best_gap = (upperbound - min_value)/abs(min_value)
                if verbose:
                    print(min_open_level, (min_value, lower_bound[min_open_level]),
                          upperbound, best_gap)
                # arrived at a solution?
                if best_gap <= gaptol:
                    # self.leaves += [current_node] + \
                    #                list(self.node_bfs_queue.queue) + \
                    #                list(self.node_dfs_queue.queue)
                    return uppersol, upperbound, lower_bound, best_gap
                min_open_level += 1

            # integral solution?
            if is_integral(current_node.lower_bound_z, self.inttol):
                current_upper_bound = current_lower_bound
                if current_upper_bound < upperbound:
                    upperbound = current_upper_bound
                    uppersol = current_node.lower_bound_solution
                    # self.leaves.append(current_node)
                    if verbose:
                        print('itegral:', current_node)
            # branch?
            elif current_dual_cost < upperbound:
                current_upper_bound = current_node.upper_solve()
                if current_upper_bound < upperbound:
                    upperbound = current_upper_bound
                    uppersol = current_node.upper_bound_solution
                left_node, right_node = branch(current_node, self.x, l0, l2, m,
                                               self.xi_xi, self.inttol,
                                               branching, mu)
                self.levels[current_node.level + 1] = \
                    self.levels.get(current_node.level + 1, 0) + 2
                if current_node.level < min_open_level + number_of_dfs_levels:
                    self.node_dfs_queue.put(right_node)
                    self.node_dfs_queue.put(left_node)
                else:
                    self.node_bfs_queue.put(right_node)
                    self.node_bfs_queue.put(left_node)

            # prune?
            else:
                pass
                # self.leaves.append(current_node)

        min_value = max([j for i, j in dual_bound.items()
                         if i <= min_open_level])
        best_gap = (upperbound - min_value)/abs(min_value)
        return uppersol, upperbound, lower_bound, best_gap

    # def get_lower_optimal_node(self):
    #     self.leaves = sorted(self.leaves)
    #     if self.leaves[-1].lower_bound_value:
    #         return self.leaves[-1]
    #     else:
    #         return self.leaves[-1].parent
    #
    # @staticmethod
    # def support_list(current_node):
    #     list_ = []
    #     while current_node:
    #         list_.append(current_node.support)
    #         current_node = current_node.parent
    #     return list_
    #
    # def optimal_support_list(self):
    #     list_ = []
    #     current_node = self.get_lower_optimal_node()
    #     while current_node:
    #         list_.append(current_node.support)
    #         current_node = current_node.parent
    #     return list_
