import numpy as np
import sys
import queue

from .node import Node
from .utilities import branch, is_integral


def bnb(x, y, lambda_value, m, inttol=1e-4, gaptol=1e-2, reltol=1e-4, upperbound=sys.maxsize, uppersol=None,
        branching='maxfrac', l1solver='l1cd', mu=0.9):
    # The number of features
    p = x.shape[1]

    # storing nodes for breadth first search
    node_queue = queue.Queue()

    # upper and lower bounds
    zlb = np.zeros(p)
    zub = np.ones(x.shape[1])

    # root node
    node_queue.put(Node(0, None, zlb, zub))

    # lower and upper bounds initialization
    lower_bound = {}
    edges = []
    while node_queue.qsize() > 0:
        # get node
        current_node = node_queue.get()

        # prune?
        if current_node.parent and upperbound <= current_node.parent.lower_bound:
            continue

        # arrived at a solution?
        if current_node.level not in lower_bound and current_node.level > 0:
            print(current_node.level - 1, lower_bound, upperbound)
            if (upperbound - lower_bound[current_node.level - 1])/lower_bound[current_node.level - 1] <= gaptol:
                return uppersol, upperbound, edges, lower_bound

        # calculate lower bound and update
        current_lower_bound = current_node.compute_lower_bound(x, y, m, lambda_value, reltol, l1solver,
                                                               current_node.initial_guess)
        lower_bound[current_node.level] = min(current_lower_bound, lower_bound.get(current_node.level, sys.maxsize))
        # integral solution?
        if is_integral(current_node.lower_bound_z, inttol):
            current_upper_bound = current_lower_bound
            if current_upper_bound < upperbound:
                print('integral', current_node.node_num, current_node.level, current_upper_bound)
                upperbound = current_upper_bound
                uppersol = current_node.lower_bound_solution
        # branch?
        elif current_lower_bound < upperbound:
            current_upper_bound = current_node.compute_upper_bound(x, y, m, lambda_value, inttol)
            if current_upper_bound < upperbound:
                upperbound = current_upper_bound
                uppersol = current_node.upper_bound_solution
            branch(node_queue, current_node, x, m, lambda_value, inttol, branching, mu)
            edges += [(current_node.node_num, current_node.node_num * 2 + 1),
                      (current_node.node_num, current_node.node_num * 2 + 2)]
        # prune?
        else:
            # print('prune', current_node.node_num, current_node.level)
            pass
    return uppersol, upperbound, edges, lower_bound
