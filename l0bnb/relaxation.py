import copy
import warnings

from numba import njit, NumbaDeprecationWarning, NumbaPendingDeprecationWarning
from numba.typed import List
import numpy as np

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)


@njit(cache=True)
def _calculate_cost(beta, r, l0, l2, golden_ratio, m, zlb, zub):
    s = beta ** 2 * (np.abs(beta) > golden_ratio) + \
        np.abs(beta) * golden_ratio * (np.abs(beta) <= golden_ratio)
    s = s * (golden_ratio <= m) + np.abs(beta) * m * (golden_ratio > m)
    s = s * (zlb < 1) + beta ** 2 * (zlb == 1)
    z = np.abs(beta) / m
    z[z > 0] = np.maximum(z[z > 0], beta[z > 0] ** 2 / s[z > 0])
    z = np.minimum(np.maximum(zlb, z), zub)
    return np.dot(r, r) / 2 + l0 * np.sum(z[z > 0]) + l2 * np.sum(s[s > 0]), z


@njit(cache=True)
def _calculate_dual_cost(y, beta, r, rx, l0, l2, golden_ratio, m, zlb, zub,
                         support):
    a = 2 * m * l2
    lambda_ = a if golden_ratio <= m else (l0 / m + l2 * m)
    gamma = np.zeros(len(beta))

    for i in support:
        if zub[i] == 0:
            continue
        if abs(rx[i]) <= a:
            gamma[i] = 0
        else:
            gamma[i] = (np.abs(rx[i]) - a) * np.sign(- rx[i])

    c = - rx - gamma
    pen = (c * c / (4 * l2) - l0) * zub
    pen1 = pen * zlb
    if golden_ratio <= m:
        pen2 = np.maximum(0, pen * (1 - zlb))
        pen = pen1 + pen2
    else:
        pen = pen1
        gamma = np.zeros(len(support))
        counter = 0
        for i in support:
            if (1 - zlb[i])*zub[i]:
                gamma[counter] = np.maximum(0, (np.abs(rx[i]) - lambda_))
            counter += 1
    return -np.dot(r, r) / 2 + np.dot(r, y) - np.sum(pen) -\
           m * np.sum(np.abs(gamma))


@njit(cache=True)
def _coordinate_descent_loop(x, beta, index_map, l2, golden_ratio, threshold, m,
                             xi_xi, zlb, zub, support, r):
    zub_active_is_zero = np.where(zub == 0)[0]
    zlb_active_normal = np.where(np.logical_and(zlb == 0, zub > 0))[0]
    zlb_active_is_one = np.where(zlb > 0)[0]
    set_add = support.add
    # set_discard = support.discard
    dot_product = np.dot
    # to_remove = List()

    for i in zub_active_is_zero:
        r = r + beta[i] * x[:, i]
        # set_discard(index_map[i])
        # to_remove.append(i)
        beta[i] = 0

    for i in zlb_active_normal:
        x_i = x[:, i]
        r = r + beta[i] * x_i
        ri_xi = dot_product(r, x_i)
        abs_ri_xi = np.abs(ri_xi)
        if abs_ri_xi <= threshold:
            # set_discard(index_map[i])
            # to_remove.append(i)
            beta[i] = 0
        else:
            if index_map[i] not in support:
                set_add(index_map[i])
            criteria = (abs_ri_xi - threshold) / xi_xi[i]
            criteria = criteria if golden_ratio > m \
                else criteria if criteria < golden_ratio else \
                abs_ri_xi / (xi_xi[i] + 2 * l2)
            beta[i] = (criteria if criteria < m else m) * np.sign(ri_xi)
            r = r - beta[i] * x_i

    for i in zlb_active_is_one:
        x_i = x[:, i]
        r = r + beta[i] * x_i
        ri_xi = dot_product(r, x_i)
        criteria = abs(ri_xi) / (2 * l2 + xi_xi[i])
        if index_map[i] not in support:
            set_add(index_map[i])
        beta[i] = (criteria if criteria < m else m) * np.sign(ri_xi)
        r = r - beta[i] * x_i

    return beta, r  # , to_remove


def coordinate_descent(x, beta, cost, l0, l2, golden_ratio, threshold, m, xi_xi,
                       zlb, zub, support, r, reltol):
    active_set = sorted(list(support))
    zlb_active = zlb[active_set]
    zub_active = zub[active_set]
    beta_active = beta[active_set]
    xi_xi_active = xi_xi[active_set]
    x_active = x[:, active_set]
    numba_set = List()
    [numba_set.append(x) for x in active_set]

    tol = 1
    while tol > reltol:
        old_cost = cost

        beta_active, r = \
            _coordinate_descent_loop(x_active, beta_active, numba_set, l2,
                                     golden_ratio, threshold, m, xi_xi_active,
                                     zlb_active, zub_active, support, r)
        beta[active_set] = beta_active
        cost, _ = _calculate_cost(beta[active_set], r, l0, l2, golden_ratio, m,
                                  zlb[active_set], zub[active_set])
        tol = abs(1 - old_cost / cost)
        # active_set = np.delete(active_set, to_remove)
        # zlb_active = np.delete(zlb_active, to_remove)
        # zub_active = np.delete(zub_active, to_remove)
        # beta_active = np.delete(beta_active, to_remove)
        # xi_xi_active = np.delete(xi_xi_active, to_remove)
        # x_active = np.delete(x_active, to_remove, 1)
        # print('new', cost, tol, len(support))
    return beta, cost, r


def initial_active_set(y, x, beta, l2, golden_ratio, threshold, m, xi_xi, zlb,
                       zub, support, r):
    num_of_similar_supports = 0
    correlations = np.matmul(y, x) / xi_xi
    partition = np.argpartition(-correlations, int(0.2 * len(beta)))
    active_set = list(partition[0: int(0.2 * len(beta))])
    zlb_active = zlb[active_set]
    zub_active = zub[active_set]
    beta_active = beta[active_set]
    xi_xi_active = xi_xi[active_set]
    x_active = x[:, active_set]
    while num_of_similar_supports < 3:
        old_support = copy.deepcopy(support)
        typed_a = List()
        [typed_a.append(x) for x in active_set]
        beta_active, r = \
            _coordinate_descent_loop(x_active, beta_active, typed_a, l2,
                                     golden_ratio, threshold, m, xi_xi_active,
                                     zlb_active, zub_active, support, r)
        if old_support == support:
            num_of_similar_supports += 1
        else:
            num_of_similar_supports = 0
    beta[active_set] = beta_active
    return support, r


@njit(cache=True, parallel=True)
def _above_threshold_indices(zub, r, x, threshold):
    rx = r @ x
    above_threshold = np.where(zub * np.abs(r @ x) - threshold > 0)[0]
    return above_threshold, rx


def relaxation_solve(x, y, l0, l2, m, xi_xi, zlb, zub, beta_init, r,
                     reltol=1e-4):
    p = x.shape[1]
    golden_ratio = np.sqrt(l0 / l2) if l2 != 0 else np.Inf
    threshold = 2 * np.sqrt(l0 * l2) if golden_ratio <= m else l0 / m + l2 * m
    if beta_init is None:
        beta = np.zeros(p)
        r = y - np.matmul(x, beta)
        support, r = initial_active_set(y, x, beta, l2, golden_ratio, threshold,
                                        m, xi_xi, zlb, zub, {0}, r)
    else:
        beta = beta_init
        support = set([i.item() for i in abs(beta).nonzero()[0]])
    cost, _ = _calculate_cost(beta, r, l0, l2, golden_ratio, m, zlb, zub)
    while True:
        beta, cost, r = \
            coordinate_descent(x, beta, cost, l0, l2, golden_ratio, threshold,
                               m, xi_xi, zlb, zub, support, r, reltol)
        above_threshold, rx = _above_threshold_indices(zub, r, x, threshold)
        outliers = [i for i in above_threshold if i not in support]
        if not outliers:
            dual_cost = _calculate_dual_cost(y, beta, r, rx, l0, l2,
                                             golden_ratio, m, zlb, zub, support)
            if (cost - dual_cost)/abs(cost) < reltol:
                break
            else:
                if reltol < 1e-8:
                    break
                reltol /= 10
        support = support | set([i.item() for i in outliers])
    support = set([i.item() for i in abs(beta).nonzero()[0]])
    cost, z = _calculate_cost(beta, r, l0, l2, golden_ratio, m, zlb, zub)
    z = np.minimum(np.maximum(zlb, z), zub)
    return cost, dual_cost, beta, z, r, support
