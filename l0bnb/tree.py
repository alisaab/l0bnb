import time
import queue
import sys

import numpy as np
from scipy import optimize as sci_opt

from .node import Node
from .utilities import branch, is_integral


class BNBTree:
    def __init__(self, x, y, inttol=1e-4, reltol=1e-4):
        """
        Initiate a BnB Tree to solve the least squares regression problem with
        l0l2 regularization

        Parameters
        ----------
        x: np.array
            n x p numpy array
        y: np.array
            1 dimensional numpy array of size n
        inttol: float
            The integral tolerance of a variable.
        reltol: float
            primal-dual relative tolerance
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

    def solve(self, l0, l2, m, gaptol=1e-2, warm_start=None, mu=0.95,
              branching='maxfrac', l1solver='l1cd', number_of_dfs_levels=0,
              verbose=False):
        """
        Solve the least squares problem with l0l2 regularization

        Parameters
        ----------
        l0: float
            The zeroth norm coefficient
        l2: float
            The second norm coefficient
        m: float
            features bound (big M)
        gaptol: float
            the relative gap between the upper and lower bound after which the
            algorithm will be terminated
        warm_start: np.array
            (p x 1) array representing a warm start
        branching: str
            'maxfrac' or 'strong'
        l1solver: str
            'l1cd', 'gurobi' or 'mosek'
        mu: float
            Used with strong branching
        number_of_dfs_levels: int
            number of levels to solve as dfs
        verbose: int
            print progress
        Returns
        -------
        tuple
            uppersol, upperbound, lower_bound, best_gap, sol_time
        """
        st = time.time()
        if warm_start is None:
            upperbound = sys.maxsize
            uppersol = None
        else:
            if verbose:
                print("using a warm start")
            support = np.nonzero(warm_start)[0]
            x_support = self.x[:, support]
            x_ridge = np.sqrt(2 * l2) * np.identity(len(support))
            x_upper = np.concatenate((x_support, x_ridge), axis=0)
            y_upper = np.concatenate((self.y, np.zeros(len(support))), axis=0)
            res = sci_opt.lsq_linear(x_upper, y_upper, (-m, m))
            upperbound = res.cost + l0 * len(support)
            uppersol = warm_start
            uppersol[support] = res.x
        if verbose:
            print(f"initializing using a warm start took {time.time() - st}")
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
                best_gap = (upperbound - min_value)/abs(upperbound)
                if verbose:
                    print(f'l: {min_open_level}, (d: {min_value}, '
                          f'p: {lower_bound[min_open_level]}), u: {upperbound},'
                          f' g: {best_gap}, t: {time.time() - st} s')
                # arrived at a solution?
                if best_gap <= gaptol:
                    # self.leaves += [current_node] + \
                    #                list(self.node_bfs_queue.queue) + \
                    #                list(self.node_dfs_queue.queue)
                    return uppersol, upperbound, lower_bound, best_gap, \
                           time.time() - st
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
        best_gap = (upperbound - min_value)/abs(upperbound)
        return uppersol, upperbound, lower_bound, best_gap, time.time() - st

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
