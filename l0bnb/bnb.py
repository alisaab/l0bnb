import numpy as np
import sys
import queue

from .node import Node
from .utilities import branch, is_integral


# TODO: Compare the effect of the bigM alone
# TODO: trees warm starts

def bnb(x, y, l0, l2, m, inttol=1e-4, gaptol=1e-2, reltol=1e-4, upperbound=sys.maxsize, uppersol=None,
        branching='maxfrac', l1solver='l1cd', mu=0.9, bnb_algorithm='BFS'):
    # The number of features
    p = x.shape[1]
    xi_xi = np.sum(x*x, axis=0)

    # storing nodes for breadth first search
    if bnb_algorithm == 'BFS':
        node_queue = queue.Queue()
    elif bnb_algorithm == 'DFS':
        node_queue = queue.LifoQueue()
    else:
        raise ValueError(bnb_algorithm + ' is not supported')

    # upper and lower bounds
    zlb = np.zeros(p)
    zub = np.ones(x.shape[1])

    # root node
    node_queue.put(Node(0, None, zlb, zub))

    # lower and upper bounds initialization
    lower_bound = {}
    level_count = {0: 1}
    best_gap = upperbound

    min_open_level = 0
    edges = []
    while node_queue.qsize() > 0:
        # get node
        current_node = node_queue.get()
        # print(current_node.level, current_node.node_num)
        # prune?
        if current_node.parent and upperbound <= current_node.parent.lower_bound:
            level_count[current_node.level] -= 1
            continue

        # calculate lower bound and update
        current_lower_bound = current_node.compute_lower_bound(x, y, l0, l2, m, xi_xi, reltol, l1solver,
                                                               current_node.initial_guess)
        lower_bound[current_node.level] = min(current_lower_bound, lower_bound.get(current_node.level, sys.maxsize))
        level_count[current_node.level] -= 1

        # update gap?
        if level_count[min_open_level] == 0:
            del level_count[min_open_level]
            best_gap = (upperbound - lower_bound[min_open_level])/lower_bound[min_open_level]
            print(min_open_level, lower_bound[min_open_level], upperbound, best_gap)
            # arrived at a solution?
            if best_gap <= gaptol:
                return uppersol, upperbound, edges, lower_bound, best_gap
            min_open_level += 1

        # integral solution?
        if is_integral(current_node.lower_bound_z, inttol):
            current_upper_bound = current_lower_bound
            if current_upper_bound < upperbound:
                print('integral', current_node.node_num, current_node.level, current_upper_bound)
                upperbound = current_upper_bound
                uppersol = current_node.lower_bound_solution
        # branch?
        elif current_lower_bound < upperbound:
            current_upper_bound = current_node.compute_upper_bound(x, y, l0, l2, m, inttol)
            if current_upper_bound < upperbound:
                upperbound = current_upper_bound
                uppersol = current_node.upper_bound_solution
            left_node, right_node = branch(current_node, x, l0, l2, m, xi_xi, inttol, branching, mu)
            level_count[current_node.level + 1] = level_count.get(current_node.level + 1, 0) + 2
            node_queue.put(right_node)
            node_queue.put(left_node)
            edges += [(current_node.node_num, current_node.node_num * 2 + 1),
                      (current_node.node_num, current_node.node_num * 2 + 2)]

        # prune?
        else:
            # print('prune', current_node.node_num, current_node.level)
            pass
    return uppersol, upperbound, edges, lower_bound, best_gap
